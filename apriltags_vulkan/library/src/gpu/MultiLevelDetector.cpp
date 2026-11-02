#include "vkapriltag/gpu/MultiLevelDetector.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace apriltag_vulkan {
namespace {

struct QuadShape {
  double cx, cy, side;
};

QuadShape ShapeOf(const DetectedQuad &q) {
  QuadShape s{0, 0, 0};
  for (int k = 0; k < 4; ++k) {
    s.cx += 0.25 * q.p[k][0];
    s.cy += 0.25 * q.p[k][1];
    s.side += 0.25 * std::hypot(q.p[(k + 1) & 3][0] - q.p[k][0], q.p[(k + 1) & 3][1] - q.p[k][1]);
  }
  return s;
}

// Whether two quads outline the same tag: centres within a third of a side, sides within a quarter.
bool SameTag(const QuadShape &a, const QuadShape &b) {
  const double side = std::max(a.side, b.side);
  return std::hypot(a.cx - b.cx, a.cy - b.cy) < side / 3.0 && std::fabs(a.side - b.side) < side / 4.0;
}

}  // namespace

MultiLevelDetector::MultiLevelDetector(vk::Context &ctx, const DetectorConfig &base,
                                       std::vector<LevelSpec> levels) {
  // Coarsest first, so a finer level's duplicate of a quad is the one dropped.
  std::stable_sort(levels.begin(), levels.end(),
                   [](const LevelSpec &a, const LevelSpec &b) { return a.decimation > b.decimation; });
  for (const LevelSpec &spec : levels) {
    DetectorConfig config = base;
    config.decimation = spec.decimation;
    config.box_decimate = spec.box;
    config.min_tag_pixels = spec.min_tag_pixels;
    config.max_tag_pixels = spec.max_tag_pixels;
    Level level;
    level.spec = spec;
    level.detector = std::make_unique<GpuDetector>(ctx, config);
    level.quad_decode = std::make_unique<QuadDecode>(config);
    levels_.push_back(std::move(level));
  }
}

std::vector<DetectedQuad> MultiLevelDetector::Detect(const uint8_t *gray) {
  std::vector<DetectedQuad> merged;
  std::vector<QuadShape> shapes;
  last_gpu_ms_ = 0.0;
  for (Level &level : levels_) {
    level.detector->Detect(gray);
    last_gpu_ms_ += level.detector->last_profile().gpu_ms;
    for (const DetectedQuad &q : level.quad_decode->Decode(level.detector->last_line_fit_points)) {
      const QuadShape shape = ShapeOf(q);
      const bool seen = std::any_of(shapes.begin(), shapes.end(),
                                    [&](const QuadShape &s) { return SameTag(s, shape); });
      if (seen) continue;
      shapes.push_back(shape);
      merged.push_back(q);
    }
  }
  return merged;
}

}  // namespace apriltag_vulkan
