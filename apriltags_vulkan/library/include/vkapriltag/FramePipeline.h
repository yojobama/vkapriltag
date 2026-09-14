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

// Overlaps frame N's GPU pass with frame N-1's CPU tail.
//
// Serially, a frame costs GpuDetector::Detect + QuadDecode + TagDecoder, and
// the GPU is idle for the whole CPU tail. Measured on an RX 9060 XT that tail
// is 0.36 ms against a 0.69 ms GPU pass; on an Orange Pi 5 Plus (Mali-G610)
// it is 1.15 ms against 3.9 ms. Running them concurrently makes the frame cost
// max(GPU, CPU) instead of the sum - about 1.3x on the AMD part and 1.25x on
// the Pi.
//
// This needs no device-side double buffering. Detect() ends by copying its two
// result payloads out of the readback buffer into last_selected_extents /
// last_line_fit_points, so once it returns the CPU tail reads only host
// memory; the sole hazard is Detect() overwriting those two vectors on the
// next frame, and Push() swaps them into its own pair first. Detect() itself
// stays strictly serialized - one frame in flight at a time - so every device
// buffer, descriptor set and command buffer inside GpuDetector keeps its
// existing single-threaded access pattern.
//
// This is a THROUGHPUT optimization, not a latency one. Any individual frame's
// detections still take as long to produce, and now come back one Push() late.
// A control loop that must act on the newest frame immediately should keep
// calling GpuDetector/QuadDecode/TagDecoder directly.
class FramePipeline {
 public:
  // None of the three are owned; all must outlive the FramePipeline. They must
  // not be used directly while it exists - it drives them from two threads.
  FramePipeline(GpuDetector &detector, QuadDecode &quad_decode, TagDecoder &tag_decoder);
  ~FramePipeline();

  FramePipeline(const FramePipeline &) = delete;
  FramePipeline &operator=(const FramePipeline &) = delete;

  // Starts the GPU pass for `gray_frame` and, while it runs, decodes the
  // PREVIOUS frame on the calling thread. Returns that previous frame's
  // detections - so the first call returns nullptr, and every later call
  // returns the frame before the one just pushed. The returned zarray_t* is
  // owned by the TagDecoder and is valid until the next Push()/Flush().
  //
  // `gray_frame` is read asynchronously by the GPU pass and must stay valid
  // until the NEXT Push() (or Flush()) returns, so callers need two frame
  // buffers - handing the same camera buffer back twice in a row will corrupt
  // the in-flight frame. The frame passed to the previous call must ALSO still
  // be valid, since this call decodes it.
  //
  // Rethrows, on the calling thread, any exception the GPU pass threw.
  zarray_t *Push(const uint8_t *gray_frame, uint32_t width, uint32_t height,
                 bool reversed_border);

  // Finishes the frame still in flight and returns its detections, or nullptr
  // if there is none. Leaves the pipeline empty, so the next Push() again
  // returns nullptr.
  zarray_t *Flush();

  // Profile of the most recently COMPLETED frame - i.e. the one whose
  // detections the last Push()/Flush() returned, not the one in flight.
  const GpuDetector::DetectProfile &last_profile() const { return done_profile_; }

  // Candidate quads for that same completed frame.
  const std::vector<DetectedQuad> &last_quads() const { return quads_; }

  // Selected blob extents for that same completed frame. QuadDecode no longer
  // takes these, but GpuDetector still populates them and the detector's own
  // copy already belongs to the frame in flight by the time a caller could
  // look, so the completed frame's are kept here.
  const std::vector<MinMaxExtentsGpu> &last_selected_extents() const { return done_extents_; }

 private:
  void WorkerLoop();
  // Blocks until the in-flight GPU pass finishes, then swaps its results into
  // done_*. Rethrows anything it threw.
  void HarvestInFlight();
  zarray_t *DecodeHarvested();

  GpuDetector &detector_;
  QuadDecode &quad_decode_;
  TagDecoder &tag_decoder_;

  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  // Guarded by mu_: the worker owns the job between `running_` going true and
  // `finished_` going true.
  bool running_ = false;
  bool finished_ = false;
  const uint8_t *job_frame_ = nullptr;
  std::exception_ptr job_error_;

  bool in_flight_ = false;
  bool have_done_ = false;

  // The frame currently on the GPU, and the one whose results are staged in
  // done_extents_/done_points_.
  const uint8_t *flight_frame_ = nullptr;
  uint32_t flight_width_ = 0, flight_height_ = 0;
  bool flight_reversed_border_ = false;

  const uint8_t *done_frame_ = nullptr;
  uint32_t done_width_ = 0, done_height_ = 0;
  bool done_reversed_border_ = false;

  // Extents are swapped (never moved) with the detector's vector, so the two
  // buffers ping-pong and keep their capacity - Detect() would otherwise
  // reallocate every frame, which this library goes out of its way to avoid.
  // Points must be copied instead; see HarvestInFlight.
  std::vector<MinMaxExtentsGpu> done_extents_;
  std::vector<RawLineFitPoint> done_points_;
  GpuDetector::DetectProfile done_profile_{};
  std::vector<DetectedQuad> quads_;
};

}  // namespace apriltag_vulkan
