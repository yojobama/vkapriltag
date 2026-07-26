#pragma once

#include <vector>

#include "gpu/GpuDetector.h"
#include "gpu/Types.h"

namespace apriltag_vulkan {

// CPU tail of the detector pipeline. GpuDetector::Detect() only computes,
// per selected blob, its perimeter point count/bounding box (extents) and
// the RAW (non-cumulative) per-point line-fit moments; everything from here
// on - cumulative range sums, peak finding, and the small (<= C(10,4) = 210
// combination) per-blob combinatorial quad search - is scalar/branchy work
// on a tiny amount of data (a few thousand candidate blobs/points per
// frame at most), and is both much simpler and much easier to verify as
// plain C++ than as GPU compute shaders. See line_fit_filter.cu's
// DoFitLines / DoFitQuads / apriltag_detect.cu's UpdateFitQuads for the
// original CUDA algorithms this ports.
class QuadDecode {
 public:
  explicit QuadDecode(const DetectorConfig &config) : config_(config) {}

  // Runs the full CPU tail (peak finding, combinatorial quad fit, corner
  // intersection, geometric sanity checks, decimation-scale correction) and
  // returns the resulting quads' corners in full-resolution pixel
  // coordinates (matching the original un-decimated input image passed to
  // GpuDetector::Detect()).
  std::vector<DetectedQuad> Decode(const std::vector<MinMaxExtentsGpu> &selected_extents,
                                   const std::vector<RawLineFitPoint> &line_fit_points) const;

 private:
  DetectorConfig config_;
};

}  // namespace apriltag_vulkan
