#include "vkapriltag/FramePipeline.h"

#include <utility>

#include "vkapriltag/common/Profiler.h"

namespace apriltag_vulkan {

FramePipeline::FramePipeline(GpuDetector &detector, QuadDecode &quad_decode,
                             TagDecoder &tag_decoder)
    : quad_decode_(quad_decode), tag_decoder_(tag_decoder) {
  slots_.push_back(std::make_unique<Slot>());
  slots_[0]->detector = &detector;
  Start();
}

FramePipeline::FramePipeline(GpuDetector &detector, GpuDetector &second_detector,
                             QuadDecode &quad_decode, TagDecoder &tag_decoder)
    : quad_decode_(quad_decode), tag_decoder_(tag_decoder) {
  for (GpuDetector *d : {&detector, &second_detector}) {
    slots_.push_back(std::make_unique<Slot>());
    slots_.back()->detector = d;
  }
  Start();
}

void FramePipeline::Start() {
  for (auto &slot : slots_) {
    Slot *s = slot.get();
    s->worker = std::thread([this, s] { WorkerLoop(*s); });
  }
}

FramePipeline::~FramePipeline() {
  // Drain first: joining while Detect() is running would destroy the members it uses.
  for (auto &slot : slots_) {
    if (!slot->in_flight) continue;
    try {
      HarvestInFlight(*slot);
    } catch (...) {
      // A GPU error on the last frames must not escape a destructor.
    }
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &slot : slots_) slot->worker.join();
}

void FramePipeline::WorkerLoop(Slot &slot) {
  prof::SetThreadName("gpu-driver");
  for (;;) {
    const uint8_t *frame = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return slot.running || stop_; });
      if (stop_ && !slot.running) return;
      frame = slot.frame;
    }

    std::exception_ptr error;
    try {
      VKA_PROF_SCOPE("pipeline.gpu_pass");
      slot.detector->Detect(frame);
    } catch (...) {
      error = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      slot.error = error;
      slot.running = false;
      slot.finished = true;
    }
    cv_.notify_all();
  }
}

void FramePipeline::HarvestInFlight(Slot &slot) {
  std::exception_ptr error;
  {
    // Time the CPU tail waited for the GPU pass.
    VKA_PROF_SCOPE("pipeline.wait_gpu");
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return slot.finished; });
    slot.finished = false;
    error = slot.error;
    slot.error = nullptr;
  }
  slot.in_flight = false;

  if (error) {
    // The detector's results for this frame are undefined; drop it (have_done_ is unchanged).
    std::rethrow_exception(error);
  }

  // The slot's worker is idle, so no lock is needed.
  GpuDetector &detector = *slot.detector;
  std::swap(done_extents_, detector.last_selected_extents);
  // last_line_fit_points is a non-owning span overwritten by the next Detect(), so it is copied.
  done_points_.assign(detector.last_line_fit_points.begin(),
                      detector.last_line_fit_points.end());
  done_profile_ = detector.last_profile();
  done_frame_ = slot.frame;
  done_width_ = slot.width;
  done_height_ = slot.height;
  done_reversed_border_ = slot.reversed_border;
  have_done_ = true;
}

zarray_t *FramePipeline::DecodeHarvested() {
  if (!have_done_) return nullptr;
  VKA_PROF_SCOPE("pipeline.cpu_tail");
  quads_ = quad_decode_.Decode(done_points_);
  return tag_decoder_.Decode(quads_, done_frame_, done_width_, done_height_,
                             done_reversed_border_);
}

zarray_t *FramePipeline::Push(const uint8_t *gray_frame, uint32_t width, uint32_t height,
                              bool reversed_border) {
  Slot &slot = *slots_[next_];
  const bool harvested = slot.in_flight;
  if (harvested) HarvestInFlight(slot);

  slot.frame = gray_frame;
  slot.width = width;
  slot.height = height;
  slot.reversed_border = reversed_border;
  {
    std::lock_guard<std::mutex> lock(mu_);
    slot.running = true;
  }
  cv_.notify_all();
  slot.in_flight = true;
  next_ = (next_ + 1) % slots_.size();

  // Runs on the caller's thread while the workers drive the GPU passes in flight.
  return harvested ? DecodeHarvested() : nullptr;
}

zarray_t *FramePipeline::Flush() {
  // Slots are used in turn, so the first one in flight from next_ holds the oldest frame.
  for (size_t k = 0; k < slots_.size(); ++k) {
    Slot &slot = *slots_[(next_ + k) % slots_.size()];
    if (!slot.in_flight) continue;
    HarvestInFlight(slot);
    return DecodeHarvested();
  }
  have_done_ = false;
  return nullptr;
}

}  // namespace apriltag_vulkan
