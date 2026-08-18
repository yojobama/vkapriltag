#pragma once

#include <cstdint>
#include <thread>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Runs frame N+1's GPU stage (GpuDetector::Detect) concurrently with frame
// N's CPU tail (QuadDecode + TagDecoder), instead of the strictly serial
// Detect() -> Decode() -> Decode() sequence every caller of the individual
// classes uses today.
//
// GpuDetector::Detect() is itself fully synchronous - four SubmitAndWait()
// calls, host copies complete before it returns - so no device buffer needs
// double-buffering; only two host-side things do, and this class exists
// because both are easy to get wrong by hand:
//
//  - GpuDetector::last_selected_extents / last_line_fit_points are
//    overwritten by the next Detect() call, so they must be copied out
//    before the CPU tail can safely run in the background while the caller
//    moves on to the next frame's Detect().
//  - The raw grayscale frame TagDecoder samples from must not be mutated
//    (e.g. by the caller capturing the next frame into the same buffer)
//    while the tail thread is still reading it. This class keeps two
//    internal copies and alternates between them.
//
// Pipelining raises THROUGHPUT, not per-frame latency: a call's Result is
// always the PREVIOUS frame's completed detections, so this trades one
// frame of latency for the CPU tail running fully hidden behind the next
// frame's GPU work.
//
// Not thread-safe: Detect()/Flush() must be called from a single thread
// (the internal tail thread is managed entirely inside this class).
class PipelinedDetector {
 public:
  struct Result {
    // False only for the very first Detect() call (no previous frame yet).
    bool valid = false;

    // Owned by the internal TagDecoder; valid until the NEXT Detect() or
    // Flush() call returns a new Result, exactly like TagDecoder::Decode()'s
    // own contract.
    zarray_t *detections = nullptr;
    std::vector<DetectedQuad> quads;
    GpuDetector::DetectProfile profile;
  };

  // `td` must already have the desired tag family(-ies) added; matches
  // TagDecoder's own contract (not owned by this class).
  PipelinedDetector(vk::Context &ctx, const DetectorConfig &config, apriltag_detector_t *td);
  ~PipelinedDetector();

  PipelinedDetector(const PipelinedDetector &) = delete;
  PipelinedDetector &operator=(const PipelinedDetector &) = delete;

  // Copies gray_frame (config.width * config.height bytes) internally, so
  // the caller may overwrite/reuse it immediately after this call returns.
  // Runs the GPU stage for THIS frame synchronously, then hands the CPU
  // tail for THIS frame to a background thread and returns the PREVIOUS
  // frame's now-complete Result.
  Result Detect(const uint8_t *gray_frame);

  // Waits for the last submitted frame's tail to finish and returns its
  // Result. Call once after the last Detect() in a run - otherwise the
  // final frame's detections are silently dropped.
  Result Flush();

 private:
  Result JoinPending();

  DetectorConfig config_;
  GpuDetector gpu_detector_;
  QuadDecode quad_decode_;
  TagDecoder tag_decoder_;  // used exclusively by the tail thread

  std::vector<uint8_t> gray_buffers_[2];
  int cur_buffer_ = 0;

  std::thread tail_thread_;
  Result pending_result_;
  bool has_pending_ = false;
};

}  // namespace apriltag_vulkan
