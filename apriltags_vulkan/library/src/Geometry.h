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

// Corner of two lines, each [Ex Ey nx ny] (a point on it and a normal): the line `li` is met by `lj`.
// *det is the determinant, near zero for parallel lines; the corner is meaningless then.
inline void LineCorner(const double li[4], const double lj[4], double *x, double *y, double *det) {
  const double A00 = li[3], A01 = -lj[3];
  const double A10 = -li[2], A11 = lj[2];
  const double B0 = -li[0] + lj[0];
  const double B1 = -li[1] + lj[1];
  *det = A00 * A11 - A10 * A01;
  const double W00 = A11 / *det, W01 = -A01 / *det;
  const double L0 = W00 * B0 + W01 * B1;
  *x = li[0] + L0 * A00;
  *y = li[1] + L0 * A10;
}

// Intersection of the line through (ax, ay) with direction (adx, ady) and the line through (bx, by) with
// direction (bdx, bdy); false when they are parallel (|cross| < 1e-9).
inline bool IntersectDirected(double ax, double ay, double adx, double ady, double bx, double by,
                              double bdx, double bdy, double *x, double *y) {
  const double cross = adx * bdy - ady * bdx;
  if (std::fabs(cross) < 1e-9) return false;
  const double t = ((bx - ax) * bdy - (by - ay) * bdx) / cross;
  *x = ax + t * adx;
  *y = ay + t * ady;
  return true;
}

// Projective map with row-major matrix h[0..7] and h22 as the last entry, applied to (x, y); *w is the
// denominator.
inline void ProjectHomography(const double *h, double h22, double x, double y, double *u, double *v, double *w) {
  *w = h[6] * x + h[7] * y + h22;
  *u = (h[0] * x + h[1] * y + h[2]) / *w;
  *v = (h[3] * x + h[4] * y + h[5]) / *w;
}

// The same for a full row-major 3x3 matrix.
inline void ProjectHomography(const double h[9], double x, double y, double *u, double *v) {
  double w;
  ProjectHomography(h, h[8], x, y, u, v, &w);
}

}  // namespace apriltag_vulkan::geom
