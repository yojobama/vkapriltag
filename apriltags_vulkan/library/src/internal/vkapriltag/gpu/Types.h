#pragma once

#include <cstdint>

// C++ mirrors of the GLSL structs in shaders/common.glsl. Field order and
// widths must match exactly (std430 layout, all members 4 bytes).
// QBPoint has no C++ mirror: it is a single packed uint32 on the GPU.

namespace apriltag_vulkan {

// Mirrors common.glsl's MinMaxExtentsGpu (without starting_offset and rep0/rep1).
struct MinMaxExtentsGpu {
  int32_t min_x = 0;
  int32_t min_y = 0;
  int32_t max_x = 0;
  int32_t max_y = 0;
  uint32_t count = 0;
  // Must stay adjacent to count: the pair is accumulated as one 64-bit word on the GPU.
  int32_t pxgx_plus_pygy_sum = 0;
  int32_t gx_sum = 0;
  int32_t gy_sum = 0;

  double cx() const { return (min_x + max_x) * 0.5 + 0.05118; }
  double cy() const { return (min_y + max_y) * 0.5 + -0.028581; }
  double dot() const {
    double sum2 = static_cast<double>(pxgx_plus_pygy_sum) * 2.0 -
                  static_cast<double>(min_x + max_x) * gx_sum -
                  static_cast<double>(min_y + max_y) * gy_sum;
    return sum2 * 0.5 - 0.05118 * gx_sum + 0.028581 * gy_sum;
  }
};
static_assert(sizeof(MinMaxExtentsGpu) == 32, "MinMaxExtentsGpu must match std430 layout");

// Mirrors common.glsl's IPoint (x/y packed into one word). Used only for buffer sizing.
struct IPoint {
  uint32_t blob_index = 0;
  uint32_t xy = 0;
  uint32_t theta_key = 0;
};
static_assert(sizeof(IPoint) == 12, "IPoint must match std430 layout");

// Mirrors common.glsl's RawLineFitPoint: four values packed into two words.
struct RawLineFitPoint {
  uint32_t xy2 = 0;     // x2 in bits [0:13], y2 in bits [14:27]
  uint32_t w_blob = 0;  // W in bits [0:9], blob_index in bits [10:31]

  int32_t x2() const { return static_cast<int32_t>(xy2 & 0x3FFFu); }
  int32_t y2() const { return static_cast<int32_t>((xy2 >> 14) & 0x3FFFu); }
  int32_t W() const { return static_cast<int32_t>(w_blob & 0x3FFu); }
  uint32_t blob_index() const { return w_blob >> 10; }
};
static_assert(sizeof(RawLineFitPoint) == 8, "RawLineFitPoint must match std430 layout");

}  // namespace apriltag_vulkan
