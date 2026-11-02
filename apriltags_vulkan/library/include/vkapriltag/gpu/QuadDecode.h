#pragma once

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/Types.h"

namespace apriltag_vulkan {

// Reusable per-blob-fit scratch for the peaks-based fit path. One instance per
// WorkerPool slot, indexed by the `slot` ParallelFor passes to each task.
struct QuadFitScratch {
  std::vector<LineFitMoments> cs;
  std::vector<double> error;
  std::vector<double> filtered;
  std::vector<std::pair<double, uint32_t>> peaks;
};

// CPU tail of the detector pipeline: cumulative sums, peak finding and the
// per-blob quad fit over the GPU's raw line-fit points. Blobs are fitted in
// parallel on a worker pool; results are returned in blob order.
class QuadDecode {
 public:
  // Honours APRILTAG_CPU_THREADS when config.cpu_threads is left at 0.
  explicit QuadDecode(const DetectorConfig &config);

  // Fits quads to the line-fit points and returns their corners in
  // full-resolution (un-decimated) pixel coordinates.
  std::vector<DetectedQuad> Decode(std::span<const RawLineFitPoint> line_fit_points) const;

  unsigned threads() const { return pool_->threads(); }

  // DP corner-seeding outcomes from the last Decode() (both 0 unless quad_fit_method == kDp).
  struct DpStats {
    uint32_t attempts = 0;
    uint32_t fallbacks = 0;
  };
  DpStats last_dp_stats() const { return last_dp_stats_; }

 private:
  DetectorConfig config_;
  std::unique_ptr<WorkerPool> pool_;
  // One entry per pool slot.
  mutable std::vector<QuadFitScratch> scratch_;
  mutable DpStats last_dp_stats_;
};

}  // namespace apriltag_vulkan
