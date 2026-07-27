#pragma once

#include <cstdint>
#include <vector>

#include "gpu/GpuDetector.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Wires the CPU-computed quad corner candidates (DetectedQuad, produced by
// QuadDecode from the GPU pipeline's output) through the fetched `apriltag`
// C library's per-family bit-sampling/hamming decode (quad_decode_index) and
// cross-family duplicate reconciliation (reconcile_detections), producing
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
class TagDecoder {
 public:
  // `td` must already have the desired tag family(-ies) added via
  // apriltag_detector_add_family(); TagDecoder does not own `td`.
  explicit TagDecoder(apriltag_detector_t *td);
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
  zarray_t *poly0_;
  zarray_t *poly1_;
  zarray_t *detections_;
};

}  // namespace apriltag_vulkan
