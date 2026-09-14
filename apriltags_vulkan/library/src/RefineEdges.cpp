#include "vkapriltag/RefineEdges.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

extern "C" {
// Exposed non-static by cmake/patches/apriltag-expose-decode-steps.patch, same
// as quad_decode_index - see TagDecoder.h.
void refine_edges(apriltag_detector_t *td, image_u8_t *im_orig, struct quad *quad);
}

namespace apriltag_vulkan {
namespace {

// The four-tap bilinear tap order is upstream's, verbatim. Reassociating it
// would change the low bits of g1/g2 and cost kExact its bit-identity.
template <typename S>
inline S BilinearAt(const image_u8_t *im, int xi, int yi, S fx, S fy) {
  const uint8_t *row0 = im->buf + yi * im->stride + xi;
  const uint8_t *row1 = row0 + im->stride;
  return (1 - fx) * (1 - fy) * row0[0] + fx * (1 - fy) * row0[1] + (1 - fx) * fy * row1[0] +
         fx * fy * row1[1];
}

// `Sample` is the precision of the innermost search loop only; everything
// outside it is double regardless - see RefineEdgesMethod::kFast.
template <typename Sample>
void RefineEdgesT(int quad_decimate, const image_u8_t *im, struct quad *quad) {
  double lines[4][4];  // for each line, [Ex Ey nx ny]

  for (int edge = 0; edge < 4; edge++) {
    const int a = edge, b = (edge + 1) & 3;

    double nx = quad->p[b][1] - quad->p[a][1];
    double ny = -quad->p[b][0] + quad->p[a][0];
    const double mag = std::sqrt(nx * nx + ny * ny);
    nx /= mag;
    ny /= mag;

    if (quad->reversed_border) {
      nx = -nx;
      ny = -ny;
    }

    const int nsamples = std::max(16, static_cast<int>(mag / 8));

    double Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

    const int range = quad_decimate + 1;
    const int steps_per_unit = 4;
    const int max_steps = 2 * steps_per_unit * range + 1;
    const Sample step_length = Sample(1) / steps_per_unit;
    const Sample delta = Sample(0.5);
    const Sample grange = 1;
    const Sample snx = static_cast<Sample>(nx);
    const Sample sny = static_cast<Sample>(ny);

    for (int s = 0; s < nsamples; s++) {
      const double alpha = (1.0 + s) / (nsamples + 1);
      const double x0 = alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0];
      const double y0 = alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1];

      const Sample sx0 = static_cast<Sample>(x0);
      const Sample sy0 = static_cast<Sample>(y0);

      Sample Mn = 0;
      Sample Mcount = 0;

      for (int step = 0; step < max_steps; ++step) {
        const Sample n = -range + step_length * step;

        const Sample x1 = sx0 + (n + grange) * snx - delta;
        const Sample y1 = sy0 + (n + grange) * sny - delta;
        const Sample x1t = std::trunc(x1);
        const Sample y1t = std::trunc(y1);
        const Sample a1 = x1 - x1t;
        const Sample b1 = y1 - y1t;
        const int x1i = static_cast<int>(x1t);
        const int y1i = static_cast<int>(y1t);
        if (x1i < 0 || x1i + 1 >= im->width || y1i < 0 || y1i + 1 >= im->height) continue;

        const Sample x2 = sx0 + (n - grange) * snx - delta;
        const Sample y2 = sy0 + (n - grange) * sny - delta;
        const Sample x2t = std::trunc(x2);
        const Sample y2t = std::trunc(y2);
        const Sample a2 = x2 - x2t;
        const Sample b2 = y2 - y2t;
        const int x2i = static_cast<int>(x2t);
        const int y2i = static_cast<int>(y2t);
        if (x2i < 0 || x2i + 1 >= im->width || y2i < 0 || y2i + 1 >= im->height) continue;

        const Sample g1 = BilinearAt<Sample>(im, x1i, y1i, a1, b1);
        const Sample g2 = BilinearAt<Sample>(im, x2i, y2i, a2, b2);
        if (g1 < g2) continue;  // gradient points the wrong way

        const Sample weight = (g2 - g1) * (g2 - g1);
        Mn += weight * n;
        Mcount += weight;
      }

      if (Mcount == 0) continue;

      const double n0 = static_cast<double>(Mn) / static_cast<double>(Mcount);
      const double bestx = x0 + n0 * nx;
      const double besty = y0 + n0 * ny;

      Mx += bestx;
      My += besty;
      Mxx += bestx * bestx;
      Mxy += bestx * besty;
      Myy += besty * besty;
      N++;
    }

    const double Ex = Mx / N, Ey = My / N;
    const double Cxx = Mxx / N - Ex * Ex;
    const double Cxy = Mxy / N - Ex * Ey;
    const double Cyy = Myy / N - Ey * Ey;

    // atan2f/cosf/sinf, not the double forms: upstream uses the float ones
    // here even though everything around them is double.
    const double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
    lines[edge][0] = Ex;
    lines[edge][1] = Ey;
    lines[edge][2] = cosf(normal_theta);
    lines[edge][3] = sinf(normal_theta);
  }

  for (int i = 0; i < 4; i++) {
    const double A00 = lines[i][3], A01 = -lines[(i + 1) & 3][3];
    const double A10 = -lines[i][2], A11 = lines[(i + 1) & 3][2];
    const double B0 = -lines[i][0] + lines[(i + 1) & 3][0];
    const double B1 = -lines[i][1] + lines[(i + 1) & 3][1];

    const double det = A00 * A11 - A10 * A01;

    if (std::fabs(det) > 0.001) {
      const double W00 = A11 / det, W01 = -A01 / det;
      const double L0 = W00 * B0 + W01 * B1;

      quad->p[(i + 1) & 3][0] = lines[i][0] + L0 * A00;
      quad->p[(i + 1) & 3][1] = lines[i][1] + L0 * A10;
    }
    // else: degenerate intersection, keep the corner we had (upstream does
    // the same).
  }
}

}  // namespace

RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured) {
  if (const char *m = std::getenv("APRILTAG_VK_REFINE")) {
    const std::string method = m;
    if (method == "upstream") return RefineEdgesMethod::kUpstream;
    if (method == "exact") return RefineEdgesMethod::kExact;
    if (method == "fast") return RefineEdgesMethod::kFast;
  }
  return configured;
}

void RefineEdges(RefineEdgesMethod method, apriltag_detector_t *td, image_u8_t *im,
                 struct quad *quad) {
  switch (method) {
    case RefineEdgesMethod::kExact:
      RefineEdgesT<double>(td->quad_decimate, im, quad);
      return;
    case RefineEdgesMethod::kFast:
      RefineEdgesT<float>(td->quad_decimate, im, quad);
      return;
    case RefineEdgesMethod::kUpstream:
      refine_edges(td, im, quad);
      return;
  }
}

}  // namespace apriltag_vulkan
