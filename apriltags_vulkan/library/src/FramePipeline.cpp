#include "vkapriltag/FramePipeline.h"

#include <utility>

namespace apriltag_vulkan {

FramePipeline::FramePipeline(GpuDetector &detector, QuadDecode &quad_decode,
                             TagDecoder &tag_decoder)
    : detector_(detector), quad_decode_(quad_decode), tag_decoder_(tag_decoder) {
  worker_ = std::thread([this] { WorkerLoop(); });
}

FramePipeline::~FramePipeline() {
  // Drain first: joining while Detect() is running would destroy the members it uses.
  if (in_flight_) {
    try {
      HarvestInFlight();
    } catch (...) {
      // A GPU error on the last frame must not escape a destructor.
    }
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  worker_.join();
}

void FramePipeline::WorkerLoop() {
  for (;;) {
    const uint8_t *frame = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return running_ || stop_; });
      if (stop_ && !running_) return;
      frame = job_frame_;
    }

    std::exception_ptr error;
    try {
      detector_.Detect(frame);
    } catch (...) {
      error = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      job_error_ = error;
      running_ = false;
      finished_ = true;
    }
    cv_.notify_all();
  }
}

void FramePipeline::HarvestInFlight() {
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return finished_; });
    finished_ = false;
    error = job_error_;
    job_error_ = nullptr;
  }
  in_flight_ = false;

  if (error) {
    // The detector's results for this frame are undefined; drop it (have_done_ is unchanged).
    std::rethrow_exception(error);
  }

  // The worker is idle, so no lock is needed.
  std::swap(done_extents_, detector_.last_selected_extents);
  // last_line_fit_points is a non-owning span overwritten by the next Detect(), so it is copied.
  done_points_.assign(detector_.last_line_fit_points.begin(),
                      detector_.last_line_fit_points.end());
  done_profile_ = detector_.last_profile();
  done_frame_ = flight_frame_;
  done_width_ = flight_width_;
  done_height_ = flight_height_;
  done_reversed_border_ = flight_reversed_border_;
  have_done_ = true;
}

zarray_t *FramePipeline::DecodeHarvested() {
  if (!have_done_) return nullptr;
  quads_ = quad_decode_.Decode(done_points_);
  return tag_decoder_.Decode(quads_, done_frame_, done_width_, done_height_,
                             done_reversed_border_);
}

zarray_t *FramePipeline::Push(const uint8_t *gray_frame, uint32_t width, uint32_t height,
                              bool reversed_border) {
  if (in_flight_) HarvestInFlight();

  flight_frame_ = gray_frame;
  flight_width_ = width;
  flight_height_ = height;
  flight_reversed_border_ = reversed_border;
  {
    std::lock_guard<std::mutex> lock(mu_);
    job_frame_ = gray_frame;
    running_ = true;
  }
  cv_.notify_one();
  in_flight_ = true;

  // Runs on the caller's thread while the worker drives the GPU pass for the pushed frame.
  return DecodeHarvested();
}

zarray_t *FramePipeline::Flush() {
  if (!in_flight_ && !have_done_) return nullptr;
  if (in_flight_) HarvestInFlight();
  zarray_t *result = DecodeHarvested();
  have_done_ = false;
  return result;
}

}  // namespace apriltag_vulkan
