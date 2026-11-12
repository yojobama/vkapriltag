#pragma once

// Internal CPU geometry helpers shared by the refine, decode, polish and tracking code (not installed).
// Every helper keeps the exact expression order of the copy it replaced, so results are bit-identical.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace apriltag_vulkan::geom {

// Four-tap bilinear blend of the pixels at (xi, yi) with fractions (fx, fy); the order is upstream's
// verbatim, and reassociating it changes the rounding. `Img` has buf, stride, width and height.
template <typename S, typename Img>
inline S BilinearAt(const Img *im, int xi, int yi, S fx, S fy) {
  const uint8_t *row0 = im->buf + yi * im->stride + xi;
  const uint8_t *row1 = row0 + im->stride;
  return (1 - fx) * (1 - fy) * row0[0] + fx * (1 - fy) * row0[1] + (1 - fx) * fy * row1[0] +
         fx * fy * row1[1];
}

// Whether the 2x2 pixel block at (xi, yi) lies inside the image.
template <typename Img>
inline bool BlockInside(const Img *im, int xi, int yi) {
  return xi >= 0 && xi + 1 < im->width && yi >= 0 && yi + 1 < im->height;
}

// Bilinear sample at pixel-edge coordinates (pixel i covers [i, i+1)), computed in S; false outside.
template <typename S, typename Img>
inline bool SampleEdge(const Img *im, double x, double y, double *out) {
  const S fx = static_cast<S>(x) - S(0.5), fy = static_cast<S>(y) - S(0.5);
  const S xt = std::floor(fx), yt = std::floor(fy);
  const int xi = static_cast<int>(xt), yi = static_cast<int>(yt);
  if (!BlockInside(im, xi, yi)) return false;
  *out = BilinearAt<S>(im, xi, yi, fx - xt, fy - yt);
  return true;
}

// Median (the element at n / 2 once sorted) of v[0..n), which is reordered.
template <typename T>
inline T MedianInPlace(T *v, size_t n) {
  std::nth_element(v, v + n / 2, v + n);
  return v[n / 2];
}

}  // namespace apriltag_vulkan::geom
