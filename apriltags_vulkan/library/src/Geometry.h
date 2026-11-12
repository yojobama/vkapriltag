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

// Determinant of a row-major 3x3 matrix, by cofactors of the first row.
inline double Det3(const double a[9]) {
  const double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
  return a[0] * c00 + a[1] * c01 + a[2] * c02;
}

// The inverse of a row-major 3x3 matrix whose determinant is 1 / inv_det: the adjugate scaled by inv_det.
inline void Adjugate3(const double a[9], double inv_det, double out[9]) {
  const double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
  out[0] = c00 * inv_det;
  out[1] = (a[2] * a[7] - a[1] * a[8]) * inv_det;
  out[2] = (a[1] * a[5] - a[2] * a[4]) * inv_det;
  out[3] = c01 * inv_det;
  out[4] = (a[0] * a[8] - a[2] * a[6]) * inv_det;
  out[5] = (a[2] * a[3] - a[0] * a[5]) * inv_det;
  out[6] = c02 * inv_det;
  out[7] = (a[1] * a[6] - a[0] * a[7]) * inv_det;
  out[8] = (a[0] * a[4] - a[1] * a[3]) * inv_det;
}

// Solves A x = b for the leading n x n block of a (rows kStride apart) by Gaussian elimination with
// partial pivoting, destroying a and b; false if a pivot is below 1e-18.
template <int kStride>
bool SolveGauss(int n, double a[][kStride], double b[], double x[]) {
  for (int c = 0; c < n; ++c) {
    int piv = c;
    for (int r = c + 1; r < n; ++r) {
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    }
    if (std::fabs(a[piv][c]) < 1e-18) return false;
    if (piv != c) {
      for (int k = 0; k < n; ++k) std::swap(a[c][k], a[piv][k]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < n; ++r) {
      const double m = a[r][c] / a[c][c];
      if (m == 0.0) continue;
      for (int k = c; k < n; ++k) a[r][k] -= m * a[c][k];
      b[r] -= m * b[c];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < n; ++k) s -= a[r][k] * x[k];
    x[r] = s / a[r][r];
  }
  return true;
}

// Andrew's monotone chain over points p[0..n) already sorted by (x, y) and unique, n >= 3: writes the hull,
// counter-clockwise, to h (room for 2 n points) and returns its size. cross(o, a, b) is positive for a left turn.
template <typename P, typename Cross>
size_t MonotoneChain(const P *p, size_t n, P *h, Cross cross) {
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
    h[k++] = p[i];
  }
  for (size_t i = n - 1, lower = k + 1; i-- > 0;) {
    while (k >= lower && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
    h[k++] = p[i];
  }
  return k - 1;
}

}  // namespace apriltag_vulkan::geom
