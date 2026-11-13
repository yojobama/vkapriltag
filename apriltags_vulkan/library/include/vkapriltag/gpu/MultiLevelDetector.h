#pragma once

#include <memory>
#include <vector>

#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

namespace apriltag_vulkan {

// One level of a multi-level detection: its decimation, whether it averages blocks, and the band of
// tag sides (full-resolution pixels; 0 leaves an end open) it proposes quads for.
struct LevelSpec {
  uint32_t decimation = 1;
  bool box = false;
  uint32_t min_tag_pixels = 0;
  uint32_t max_tag_pixels = 0;
};

// Runs a GpuDetector and QuadDecode per level, coarsest to finest, and merges their quads: a tag
// found at a coarse level is not proposed again by a finer one, which only looks for smaller tags.
class MultiLevelDetector {
 public:
  MultiLevelDetector(vk::Context &ctx, const DetectorConfig &base, std::vector<LevelSpec> levels);

  // Quads from every level, in full-resolution pixel coordinates, without duplicates.
  std::vector<DetectedQuad> Detect(const uint8_t *gray);

  // Time inside the GPU stages of the last Detect(), summed over the levels.
  double last_gpu_ms() const { return last_gpu_ms_; }

 private:
  struct Level {
    LevelSpec spec;
    std::unique_ptr<GpuDetector> detector;
    std::unique_ptr<QuadDecode> quad_decode;
  };
  std::shared_ptr<WorkerPool> pool_;  // the levels run one after another
  std::vector<Level> levels_;
  double last_gpu_ms_ = 0.0;
};

}  // namespace apriltag_vulkan
