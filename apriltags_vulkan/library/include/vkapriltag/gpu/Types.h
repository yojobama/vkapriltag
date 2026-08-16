#pragma once

#include <cstdint>

// C++ mirrors of the GLSL structs in shaders/common.glsl. Field order and
// widths must match exactly (all members are 4-byte, so std430 layout has
// no implicit padding beyond what's declared here).

namespace apriltag_vulkan {

struct QBPoint {
  uint32_t rep0 = 0;
  uint32_t rep1 = 0;
  uint32_t x = 0;
  uint32_t y = 0;
  int32_t gx = 0;
  int32_t gy = 0;
  uint32_t valid = 0;
  uint32_t pad0 = 0;
};
static_assert(sizeof(QBPoint) == 32, "QBPoint must match std430 layout");

struct MinMaxExtentsGpu {
  int32_t min_x = 0;
  int32_t min_y = 0;
  int32_t max_x = 0;
  int32_t max_y = 0;
  uint32_t starting_offset = 0;
  uint32_t count = 0;
  int32_t gx_sum = 0;
  int32_t gy_sum = 0;
  int32_t pxgx_plus_pygy_sum = 0;
  uint32_t rep0 = 0;
  uint32_t rep1 = 0;
  uint32_t pad0 = 0;

  double cx() const { return (min_x + max_x) * 0.5 + 0.05118; }
  double cy() const { return (min_y + max_y) * 0.5 + -0.028581; }
  double dot() const {
    double sum2 = static_cast<double>(pxgx_plus_pygy_sum) * 2.0 -
                  static_cast<double>(min_x + max_x) * gx_sum -
                  static_cast<double>(min_y + max_y) * gy_sum;
    return sum2 * 0.5 - 0.05118 * gx_sum + 0.028581 * gy_sum;
  }
};
static_assert(sizeof(MinMaxExtentsGpu) == 48, "MinMaxExtentsGpu must match std430 layout");

struct IPoint {
  uint32_t blob_index = 0;
  uint32_t x = 0;
  uint32_t y = 0;
  int32_t gx = 0;
  int32_t gy = 0;
  uint32_t theta_key = 0;
  uint32_t pad0 = 0;
  uint32_t pad1 = 0;
};
static_assert(sizeof(IPoint) == 32, "IPoint must match std430 layout");

struct RawLineFitPoint {
  int32_t x2 = 0;
  int32_t y2 = 0;
  int32_t W = 0;
  uint32_t blob_index = 0;

  int32_t Mx() const { return W * x2; }
  int32_t My() const { return W * y2; }
  int64_t Mxx() const { return static_cast<int64_t>(W) * x2 * x2; }
  int64_t Mxy() const { return static_cast<int64_t>(W) * x2 * y2; }
  int64_t Myy() const { return static_cast<int64_t>(W) * y2 * y2; }
};
static_assert(sizeof(RawLineFitPoint) == 16, "RawLineFitPoint must match std430 layout");

// CPU-side cumulative line fit moments for a range of points (mirrors
// frc971::apriltag::LineFitMoments). Built by prefix-summing RawLineFitPoint
// values with true 64 bit / double precision arithmetic (no GPU width
// restrictions on the CPU).
struct LineFitMoments {
  int32_t Mx = 0;
  int32_t My = 0;
  int32_t W = 0;
  int64_t Mxx = 0;
  int64_t Myy = 0;
  int64_t Mxy = 0;
  int32_t N = 0;
};

// Final fitted quad corners in un-decimated pixel coordinates (mirrors
// frc971::apriltag::QuadCorners).
struct QuadCorners {
  double corners[4][2] = {};
};

}  // namespace apriltag_vulkan
