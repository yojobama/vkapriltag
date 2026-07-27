#pragma once

#include <memory>
#include <vector>

#include "common/WorkerPool.h"
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
//
// PERFORMANCE: this becomes the single most expensive stage once the GPU
// pipeline is properly sized - ~10 ms single-threaded at 1080p with 440
// blobs, against a few milliseconds for all the GPU work on a discrete card.
// The per-blob fits are completely independent, so they run across a
// persistent worker pool. Results are collected in blob order, so the output
// is identical to the serial version regardless of thread scheduling.
class QuadDecode {
 public:
  // Honours APRILTAG_CPU_THREADS when config.cpu_threads is left at 0.
  explicit QuadDecode(const DetectorConfig &config);

  // Runs the full CPU tail (peak finding, combinatorial quad fit, corner
  // intersection, geometric sanity checks, decimation-scale correction) and
  // returns the resulting quads' corners in full-resolution pixel
  // coordinates (matching the original un-decimated input image passed to
  // GpuDetector::Detect()).
  std::vector<DetectedQuad> Decode(const std::vector<MinMaxExtentsGpu> &selected_extents,
                                   const std::vector<RawLineFitPoint> &line_fit_points) const;

  unsigned threads() const { return pool_->threads(); }

 private:
  DetectorConfig config_;
  // unique_ptr so a const Decode() can still hand work to the (stateful) pool.
  std::unique_ptr<WorkerPool> pool_;
};

}  // namespace apriltag_vulkan
