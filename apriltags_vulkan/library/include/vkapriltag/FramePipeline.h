#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

namespace apriltag_vulkan {

// Overlaps frame N's GPU pass with frame N-1's CPU tail. Detect() stays serialised (one frame in
// flight) and copies its results to host memory, so the CPU tail reads only host data. Improves
// throughput; each frame's detections are returned one Push() late.
class FramePipeline {
 public:
  // None of the three are owned; all must outlive the FramePipeline and not be used directly.
  FramePipeline(GpuDetector &detector, QuadDecode &quad_decode, TagDecoder &tag_decoder);
  ~FramePipeline();

  FramePipeline(const FramePipeline &) = delete;
  FramePipeline &operator=(const FramePipeline &) = delete;

  // Starts the GPU pass for `gray_frame` and decodes the previous frame meanwhile. Returns the
  // previous frame's detections (nullptr on the first call), owned by the TagDecoder and valid
  // until the next Push()/Flush(). `gray_frame` must stay valid until the next Push()/Flush()
  // returns, and the previous call's frame must still be valid. Rethrows any exception from the
  // GPU pass.
  zarray_t *Push(const uint8_t *gray_frame, uint32_t width, uint32_t height,
                 bool reversed_border);

  // Finishes the frame in flight and returns its detections (nullptr if none).
  zarray_t *Flush();

  // Profile of the most recently completed frame (the one last returned).
  const GpuDetector::DetectProfile &last_profile() const { return done_profile_; }

  // Candidate quads for that same completed frame.
  const std::vector<DetectedQuad> &last_quads() const { return quads_; }

  // Selected blob extents for that same frame.
  const std::vector<MinMaxExtentsGpu> &last_selected_extents() const { return done_extents_; }

 private:
  void WorkerLoop();
  // Blocks until the in-flight GPU pass finishes and swaps its results into done_*.
  void HarvestInFlight();
  zarray_t *DecodeHarvested();

  GpuDetector &detector_;
  QuadDecode &quad_decode_;
  TagDecoder &tag_decoder_;

  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  // Guarded by mu_: the worker owns the job between `running_` and `finished_` going true.
  bool running_ = false;
  bool finished_ = false;
  const uint8_t *job_frame_ = nullptr;
  std::exception_ptr job_error_;

  bool in_flight_ = false;
  bool have_done_ = false;

  // The frame on the GPU, and the frame whose results are staged in done_extents_/done_points_.
  const uint8_t *flight_frame_ = nullptr;
  uint32_t flight_width_ = 0, flight_height_ = 0;
  bool flight_reversed_border_ = false;

  const uint8_t *done_frame_ = nullptr;
  uint32_t done_width_ = 0, done_height_ = 0;
  bool done_reversed_border_ = false;

  // Extents are swapped with the detector's vector; points are copied (see HarvestInFlight).
  std::vector<MinMaxExtentsGpu> done_extents_;
  std::vector<RawLineFitPoint> done_points_;
  GpuDetector::DetectProfile done_profile_{};
  std::vector<DetectedQuad> quads_;
};

}  // namespace apriltag_vulkan
