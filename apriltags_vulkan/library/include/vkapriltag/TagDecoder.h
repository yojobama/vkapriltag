#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vkapriltag/RefineEdges.h"
#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/GpuDetector.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Decodes CPU-computed quad candidates (DetectedQuad from QuadDecode) into tags via the `apriltag`
// library's quad_decode_index and reconcile_detections (exposed by
// cmake/patches/apriltag-expose-decode-steps.patch): id, hamming distance, decision margin,
// centre and homography-refined corners. If `td->refine_edges` is set, each quad is refined
// (see RefineEdgesMethod) before decoding. Stops at 2D detection; pose is not computed here.
// Quads are decoded in parallel over a WorkerPool and merged in quad order, so the output does
// not depend on thread count.
class TagDecoder {
 public:
  // `td` must already have its tag families added; it is not owned.
  // `decimation` must match DetectorConfig::decimation; it is stored in td->quad_decimate for
  // refine_edges (used only if `td->refine_edges` is set).
  // `cpu_threads` is the total parallelism (0 = hardware_concurrency, or APRILTAG_CPU_THREADS).
  // `refine_method` may be overridden by APRILTAG_VK_REFINE; used only if `td->refine_edges` is set.
  explicit TagDecoder(apriltag_detector_t *td, uint32_t decimation = 1, uint32_t cpu_threads = 0,
                      RefineEdgesMethod refine_method = RefineEdgesMethod::kExact);
  ~TagDecoder();

  TagDecoder(const TagDecoder &) = delete;
  TagDecoder &operator=(const TagDecoder &) = delete;

  // Decodes `quads` against the full-resolution grayscale frame (width*height bytes, tightly
  // packed). `reversed_border` must match the border polarity of the family (one polarity per run).
  // Returns detections owned by this TagDecoder, valid until the next Decode() or destruction.
  zarray_t *Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                   uint32_t width, uint32_t height, bool reversed_border);

 private:
  apriltag_detector_t *td_;  // not owned
  std::unique_ptr<WorkerPool> pool_;
  RefineEdgesMethod refine_method_;
  zarray_t *poly0_;
  zarray_t *poly1_;
  zarray_t *detections_;
  // One scratch zarray per candidate quad; grown, never shrunk, and truncated when reused.
  std::vector<zarray_t *> per_quad_;
  // Per-thread scratch for the unrefined decode that gates kUltraFast refinement.
  std::vector<zarray_t *> probe_;
};

}  // namespace apriltag_vulkan
