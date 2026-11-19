#pragma once

#include <vector>

#include "vkapriltag/Family.h"

namespace apriltag_vulkan {

// One decoded tag. H maps tag coordinates ([-1, 1] across the border square) to pixels, row-major;
// p are the corners in the order the decode reports them.
struct Detection {
  const Family *family = nullptr;
  int id = 0;
  int hamming = 0;  // wrong bits corrected
  float decision_margin = 0.0f;
  double H[9] = {};
  double c[2] = {};
  double p[4][2] = {};
};

using DetectionList = std::vector<Detection>;

}  // namespace apriltag_vulkan
