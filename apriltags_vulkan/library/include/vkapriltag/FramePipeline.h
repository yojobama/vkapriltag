#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

namespace apriltag_vulkan {

// Overlaps GPU passes with the CPU tail, one Push() late with one detector. A second detector takes
// alternate frames, so the next GPU pass does not wait for the host; results are two Push() late.
class FramePipeline {
 public:
  // Nothing is owned; everything must outlive the FramePipeline and not be used directly.
  FramePipeline(GpuDetector &detector, QuadDecode &quad_decode, TagDecoder &tag_decoder);
  // `second_detector` must have the same DetectorConfig as `detector` and its own vk::Lane (lanes
  // are not thread-safe; both may use the same queue). It doubles the device memory.
  FramePipeline(GpuDetector &detector, GpuDetector &second_detector, QuadDecode &quad_decode,
                TagDecoder &tag_decoder);
  ~FramePipeline();

  FramePipeline(const FramePipeline &) = delete;
  FramePipeline &operator=(const FramePipeline &) = delete;

  // Frames in flight before a Push() returns detections: 1, or 2 with a second detector.
  size_t depth() const { return slots_.size(); }

  // Starts the GPU pass for `gray_frame` and decodes the frame pushed depth() calls earlier
  // meanwhile. Returns that frame's detections (nullptr until depth() frames have been pushed),
  // owned by the TagDecoder and valid until the next Push()/Flush(). `gray_frame` must stay valid
  // until depth() further Push()/Flush() calls have returned. Rethrows any exception from the GPU
  // pass.
  zarray_t *Push(const uint8_t *gray_frame, uint32_t width, uint32_t height,
                 bool reversed_border);

  // Finishes the oldest frame in flight and returns its detections; nullptr when none is left.
  // Call it until it returns nullptr to drain every frame.
  zarray_t *Flush();

  // Profile of the most recently completed frame (the one last returned).
  const GpuDetector::DetectProfile &last_profile() const { return done_profile_; }

  // Candidate quads for that same completed frame.
  const std::vector<DetectedQuad> &last_quads() const { return quads_; }

  // Selected blob extents for that same frame.
  const std::vector<MinMaxExtentsGpu> &last_selected_extents() const { return done_extents_; }

 private:
  // One detector, the thread that drives it, and the frame it is working on.
  struct Slot {
    GpuDetector *detector = nullptr;
    std::thread worker;
    // Guarded by mu_: the worker owns the job between `running` and `finished` going true.
    bool running = false;
    bool finished = false;
    std::exception_ptr error;

    bool in_flight = false;
    const uint8_t *frame = nullptr;
    uint32_t width = 0, height = 0;
    bool reversed_border = false;
  };

  void Start();
  void WorkerLoop(Slot &slot);
  // Blocks until the slot's GPU pass finishes and swaps its results into done_*.
  void HarvestInFlight(Slot &slot);
  zarray_t *DecodeHarvested();

  QuadDecode &quad_decode_;
  TagDecoder &tag_decoder_;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;

  std::vector<std::unique_ptr<Slot>> slots_;
  // The slot the next Push() uses; when every slot is in flight it holds the oldest frame.
  size_t next_ = 0;

  bool have_done_ = false;
  // The frame whose results are staged in done_extents_/done_points_.
  const uint8_t *done_frame_ = nullptr;
  uint32_t done_width_ = 0, done_height_ = 0;
  bool done_reversed_border_ = false;

  // Extents are swapped with the detector's vector; points are copied (see HarvestInFlight).
  std::vector<MinMaxExtentsGpu> done_extents_;
  std::vector<RawLineFitPoint> done_points_;
  std::vector<GpuQuadFit> done_quad_fits_;
  GpuDetector::DetectProfile done_profile_{};
  std::vector<DetectedQuad> quads_;
};

}  // namespace apriltag_vulkan
