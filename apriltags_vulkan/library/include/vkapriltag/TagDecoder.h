#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/GpuDetector.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Wires the CPU-computed quad corner candidates (DetectedQuad, produced by
// QuadDecode from the GPU pipeline's output) through the fetched `apriltag`
// C library's per-family bit-sampling/hamming decode (quad_decode_index) and
// cross-family duplicate reconciliation (reconcile_detections) - both exposed
// as non-static entry points via cmake/patches/apriltag-expose-decode-steps.patch -
// producing
// final decoded tags: id, hamming distance, decision margin, center, and a
// homography-refined set of corners (not just the raw geometric quad
// corners QuadDecode computed). This exactly mirrors what
// GpuDetector::DecodeTags()/QuadDecodeTask() do in the original CUDA
// implementation (apriltag_detect.cu), minus RefineEdges (camera-distortion
// based edge refinement - out of scope, see README.md).
//
// Scope: this stops at apriltag_detection_t (2D detection). Pose estimation
// (apriltag_pose.h, which additionally requires a calibrated camera
// matrix/tag size) is intentionally not wired up.
//
// quad_decode_index is run in parallel, one task per candidate quad, over a
// WorkerPool exactly as QuadDecode already parallelizes its own combinatorial
// fit - upstream itself calls quad_decode_index concurrently from its own
// workpool (apriltag.c's quad_decode_task), taking td->mutex around its one
// shared write (appending to the detections array), so this is the intended
// usage, not a race we're introducing. Each quad gets its own scratch
// zarray_t (per_quad_) instead of sharing detections_ directly, so the merge
// back into detections_ happens in quad order rather than completion order -
// the output (including which of two near-duplicate detections
// reconcile_detections keeps) is bit-identical to the fully serial version
// regardless of thread count or scheduling.
class TagDecoder {
 public:
  // `td` must already have the desired tag family(-ies) added via
  // apriltag_detector_add_family(); TagDecoder does not own `td`.
  // `cpu_threads` is the total degree of parallelism (see WorkerPool); 0
  // selects hardware_concurrency, overridable via APRILTAG_CPU_THREADS (see
  // ResolveThreadCount) - the same resolution QuadDecode uses, so the env var
  // affects both CPU-tail phases identically.
  explicit TagDecoder(apriltag_detector_t *td, uint32_t cpu_threads = 0);
  ~TagDecoder();

  TagDecoder(const TagDecoder &) = delete;
  TagDecoder &operator=(const TagDecoder &) = delete;

  // Decodes `quads` (as produced by QuadDecode::Decode) against the
  // full-resolution grayscale frame they were computed from
  // (width*height bytes, tightly packed). `reversed_border` must match the
  // border polarity of the tag family being decoded (this port, like the
  // CUDA original, only supports detecting a single border polarity per
  // run).
  //
  // Returns a zarray_t* of apriltag_detection_t* owned by this TagDecoder
  // (valid until the next Decode() call or destruction) - use zarray_size()/
  // zarray_get(), or print_detections(), to inspect it.
  zarray_t *Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                   uint32_t width, uint32_t height, bool reversed_border);

 private:
  apriltag_detector_t *td_;  // not owned
  std::unique_ptr<WorkerPool> pool_;
  zarray_t *poly0_;
  zarray_t *poly1_;
  zarray_t *detections_;
  // One scratch zarray per candidate quad this frame - see the class
  // comment. Grown, never shrunk; each entry is zarray_truncate'd to empty
  // at the start of the quad that reuses it rather than destroyed and
  // recreated.
  std::vector<zarray_t *> per_quad_;
};

}  // namespace apriltag_vulkan
