#include "vkapriltag/PipelinedDetector.h"

#include <cstring>
#include <utility>

namespace apriltag_vulkan {

PipelinedDetector::PipelinedDetector(vk::Context &ctx, const DetectorConfig &config,
                                     apriltag_detector_t *td)
    : config_(config),
      gpu_detector_(ctx, config_),
      quad_decode_(config_),
      tag_decoder_(td) {
  const size_t gray_bytes = static_cast<size_t>(config_.width) * config_.height;
  gray_buffers_[0].resize(gray_bytes);
  gray_buffers_[1].resize(gray_bytes);
}

PipelinedDetector::~PipelinedDetector() {
  if (has_pending_ && tail_thread_.joinable()) {
    tail_thread_.join();
  }
}

PipelinedDetector::Result PipelinedDetector::JoinPending() {
  if (!has_pending_) return Result{};
  tail_thread_.join();
  has_pending_ = false;
  Result out = std::move(pending_result_);
  out.valid = true;
  return out;
}

PipelinedDetector::Result PipelinedDetector::Detect(const uint8_t *gray_frame) {
  // Joining here (before touching this frame's buffer) is what guarantees
  // gray_buffers_[cur_buffer_] is free: the tail that last read it was
  // launched two calls ago, and is joined at the START of every call, so by
  // construction it was already joined one call before this one.
  Result out = JoinPending();

  const size_t gray_bytes = static_cast<size_t>(config_.width) * config_.height;
  std::memcpy(gray_buffers_[cur_buffer_].data(), gray_frame, gray_bytes);

  gpu_detector_.Detect(gray_buffers_[cur_buffer_].data());

  pending_result_ = Result{};
  pending_result_.profile = gpu_detector_.last_profile();
  std::vector<MinMaxExtentsGpu> extents = gpu_detector_.last_selected_extents;
  std::vector<RawLineFitPoint> points = gpu_detector_.last_line_fit_points;

  has_pending_ = true;
  const int buf_idx = cur_buffer_;
  tail_thread_ = std::thread([this, extents = std::move(extents), points = std::move(points),
                             buf_idx]() {
    pending_result_.quads = quad_decode_.Decode(extents, points);
    pending_result_.detections =
        tag_decoder_.Decode(pending_result_.quads, gray_buffers_[buf_idx].data(), config_.width,
                            config_.height, config_.reversed_border);
  });

  cur_buffer_ ^= 1;
  return out;
}

PipelinedDetector::Result PipelinedDetector::Flush() { return JoinPending(); }

}  // namespace apriltag_vulkan
