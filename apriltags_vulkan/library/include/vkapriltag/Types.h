#pragma once

#include <cstdint>

namespace apriltag_vulkan {

// A non-owning view of an 8-bit grey image: row y starts at data + y * stride.
struct Image {
  const uint8_t *data = nullptr;
  int32_t width = 0;
  int32_t height = 0;
  int32_t stride = 0;
};

// A tag candidate: corners in full-resolution pixels, counter-clockwise on screen.
struct Quad {
  float p[4][2] = {};
  bool reversed_border = false;
};

}  // namespace apriltag_vulkan
