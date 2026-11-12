#include "BorderSample.h"

#include "Geometry.h"

#include <algorithm>
#include <cmath>

extern "C" {
#include "common/homography.h"
}

namespace apriltag_vulkan {

bool ReadBorder(const matd_t *H, const image_u8_t *im, int width_cells, BorderLevels *out) {
  const int w = width_cells;
  if (w < 4 || w > 16) return false;
  const double cells = w;
  // The cell (c, r) of the W x W square has its centre at (c + 0.5, r + 0.5) cells from the corner.
  auto at = [&](double cu, double cv, double *value) {
    double x, y;
    homography_project(H, -1.0 + 2.0 * cu / cells, -1.0 + 2.0 * cv / cells, &x, &y);
    return geom::SampleEdge<double>(im, x, y, value);
  };
  double border[64], ring[80];
  int nb = 0, nr = 0;
  double side_sum[4] = {0, 0, 0, 0};
  int side_n[4] = {0, 0, 0, 0};
  for (int r = 0; r < w; ++r) {
    for (int c = 0; c < w; c += (r == 0 || r == w - 1) ? 1 : w - 1) {
      double v;
      if (!at(c + 0.5, r + 0.5, &v)) return false;
      border[nb++] = v;
      // a non-corner cell belongs to one side
      if (r == 0 && c > 0 && c < w - 1) side_sum[0] += v, ++side_n[0];
      if (r == w - 1 && c > 0 && c < w - 1) side_sum[2] += v, ++side_n[2];
      if (c == 0 && r > 0 && r < w - 1) side_sum[3] += v, ++side_n[3];
      if (c == w - 1 && r > 0 && r < w - 1) side_sum[1] += v, ++side_n[1];
    }
  }
  for (int i = 0; i < w + 2; ++i) {
    double v;
    if (!at(i - 0.5, -0.5, &v)) return false;
    ring[nr++] = v;
    if (!at(i - 0.5, w + 0.5, &v)) return false;
    ring[nr++] = v;
  }
  for (int i = 1; i < w + 1; ++i) {
    double v;
    if (!at(-0.5, i - 0.5, &v)) return false;
    ring[nr++] = v;
    if (!at(w + 0.5, i - 0.5, &v)) return false;
    ring[nr++] = v;
  }
  double sorted[64];
  std::copy(border, border + nb, sorted);
  out->black = geom::MedianInPlace(sorted, nb);
  out->white = geom::MedianInPlace(ring, nr);
  out->cells = nb;
  if (out->white <= out->black) return false;
  const double mid = 0.5 * (out->black + out->white);
  out->dark = 0;
  for (int i = 0; i < nb; ++i) out->dark += border[i] < mid;
  for (int k = 0; k < 4; ++k) {
    out->side_level[k] = side_n[k] > 0 ? (side_sum[k] / side_n[k] - out->black) / (out->white - out->black) : 0;
  }
  return true;
}

}  // namespace apriltag_vulkan
