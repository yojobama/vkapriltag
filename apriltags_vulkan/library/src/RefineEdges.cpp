#include "vkapriltag/RefineEdges.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

extern "C" {
// Exposed non-static by cmake/patches/apriltag-expose-decode-steps.patch.
void refine_edges(apriltag_detector_t *td, image_u8_t *im_orig, struct quad *quad);
}

namespace apriltag_vulkan {
namespace {

// The four-tap bilinear order is upstream's verbatim; reassociating it breaks kExact's bit-identity.
template <typename S>
inline S BilinearAt(const image_u8_t *im, int xi, int yi, S fx, S fy) {
  const uint8_t *row0 = im->buf + yi * im->stride + xi;
  const uint8_t *row1 = row0 + im->stride;
  return (1 - fx) * (1 - fy) * row0[0] + fx * (1 - fy) * row0[1] + (1 - fx) * fy * row1[0] +
         fx * fy * row1[1];
}

// Moves each corner to the intersection of its two adjacent fitted lines.
void IntersectLines(const double lines[4][4], struct quad *quad) {
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
    // else: degenerate intersection; keep the existing corner (as upstream).
  }
}

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

      // Bilinear tap at n + grange for virtual step v; false when it falls outside the image.
      auto tap = [&](int v, Sample *g) {
        const Sample n = -range + step_length * v;
        const Sample x1 = sx0 + (n + grange) * snx - delta;
        const Sample y1 = sy0 + (n + grange) * sny - delta;
        const Sample x1t = std::trunc(x1);
        const Sample y1t = std::trunc(y1);
        const int x1i = static_cast<int>(x1t);
        const int y1i = static_cast<int>(y1t);
        if (x1i < 0 || x1i + 1 >= im->width || y1i < 0 || y1i + 1 >= im->height) return false;
        *g = BilinearAt<Sample>(im, x1i, y1i, x1 - x1t, y1 - y1t);
        return true;
      };

      // The tap at n - grange is the n + grange tap of kLag = 2 * grange * steps_per_unit steps
      // earlier, kept in a ring.
      constexpr int kLag = 8;
      Sample ring_g[kLag];
      bool ring_in[kLag];
      for (int v = -kLag; v < 0; ++v) ring_in[v + kLag] = tap(v, &ring_g[v + kLag]);

      for (int step = 0; step < max_steps; ++step) {
        Sample g1 = 0;
        const bool in1 = tap(step, &g1);
        const int slot = step & (kLag - 1);
        const bool in2 = ring_in[slot];
        const Sample g2 = ring_g[slot];
        ring_in[slot] = in1;
        ring_g[slot] = g1;
        if (!in1 || !in2) continue;
        if (g1 < g2) continue;  // gradient points the wrong way

        const Sample n = -range + step_length * step;
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

    // Float atan2f/cosf/sinf, as upstream.
    const double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
    lines[edge][0] = Ex;
    lines[edge][1] = Ey;
    lines[edge][2] = cosf(normal_theta);
    lines[edge][3] = sinf(normal_theta);
  }

  IntersectLines(lines, quad);
}

// Single-precision refinement. Each sample's search profile is evaluated once per position: the
// gradient's far tap at one step is the near tap eight steps earlier. The weighting pass is
// branch-free with independent accumulators, and the line fit uses moments centred on the edge
// midpoint so float suffices.
constexpr int kStepsPerUnit = 4;
constexpr int kTapLag = 2 * kStepsPerUnit;  // steps between a tap and the same point as the far tap
constexpr int kMaxProfile = 128;
constexpr int kLanes = 4;

void RefineEdgesFast(int quad_decimate, const image_u8_t *im, struct quad *quad) {
  const int range = quad_decimate + 1;
  const int max_steps = 2 * kStepsPerUnit * range + 1;
  const int profile_len = max_steps + kTapLag;
  if (profile_len > kMaxProfile) {
    RefineEdgesT<double>(quad_decimate, im, quad);
    return;
  }

  const float step_length = 1.0f / kStepsPerUnit;
  const float delta = 0.5f;
  const int max_x = im->width - 1;  // bilinear taps need x + 1 < width
  const int max_y = im->height - 1;

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
    const float snx = static_cast<float>(nx);
    const float sny = static_cast<float>(ny);

    // Moments of the refined points relative to the edge midpoint.
    const double xm = 0.5 * (quad->p[a][0] + quad->p[b][0]);
    const double ym = 0.5 * (quad->p[a][1] + quad->p[b][1]);
    float Mx = 0, My = 0, Mxx = 0, Mxy = 0, Myy = 0, N = 0;

    for (int s = 0; s < nsamples; s++) {
      const double alpha = (1.0 + s) / (nsamples + 1);
      const double x0 = alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0];
      const double y0 = alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1];
      const float sx0 = static_cast<float>(x0);
      const float sy0 = static_cast<float>(y0);

      // profile[m] samples the image at offset -range - 1 + m / kStepsPerUnit along the normal.
      float profile[kMaxProfile];
      float valid[kMaxProfile];
      for (int m = 0; m < profile_len; ++m) {
        const float u = -range - 1 + step_length * m;
        const float x = sx0 + u * snx - delta;
        const float y = sy0 + u * sny - delta;
        const float xt = std::trunc(x);
        const float yt = std::trunc(y);
        const float fx = x - xt;
        const float fy = y - yt;
        const int xi = static_cast<int>(xt);
        const int yi = static_cast<int>(yt);
        const bool in = xi >= 0 && xi < max_x && yi >= 0 && yi < max_y;
        const int xc = std::min(std::max(xi, 0), max_x - 1);
        const int yc = std::min(std::max(yi, 0), max_y - 1);
        profile[m] = BilinearAt<float>(im, xc, yc, fx, fy);
        valid[m] = in ? 1.0f : 0.0f;
      }

      float Mn[kLanes] = {}, Mc[kLanes] = {};
      for (int step = 0; step < max_steps; ++step) {
        const float g2 = profile[step];
        const float g1 = profile[step + kTapLag];
        const float d = g2 - g1;
        const float ok = (valid[step] * valid[step + kTapLag]) * (g1 >= g2 ? 1.0f : 0.0f);
        const float weight = ok * d * d;
        const float n = -range + step_length * step;
        Mn[step % kLanes] += weight * n;
        Mc[step % kLanes] += weight;
      }
      const float Mnsum = (Mn[0] + Mn[1]) + (Mn[2] + Mn[3]);
      const float Mcount = (Mc[0] + Mc[1]) + (Mc[2] + Mc[3]);
      if (Mcount == 0) continue;

      const float n0 = Mnsum / Mcount;
      const float bx = static_cast<float>(x0 - xm) + n0 * snx;
      const float by = static_cast<float>(y0 - ym) + n0 * sny;
      Mx += bx;
      My += by;
      Mxx += bx * bx;
      Mxy += bx * by;
      Myy += by * by;
      N++;
    }

    const float mx = Mx / N, my = My / N;
    const float Cxx = Mxx / N - mx * mx;
    const float Cxy = Mxy / N - mx * my;
    const float Cyy = Myy / N - my * my;

    const double normal_theta = .5 * atan2f(-2 * Cxy, (Cyy - Cxx));
    lines[edge][0] = xm + mx;
    lines[edge][1] = ym + my;
    lines[edge][2] = cosf(static_cast<float>(normal_theta));
    lines[edge][3] = sinf(static_cast<float>(normal_theta));
  }

  IntersectLines(lines, quad);
}

// Dense-profile refinement. Along each side a profile of bilinear taps across the edge is taken
// every ~1.5 px; the edge sits at the centre of mass of the positive gradient lobe around the
// profile's strongest step (unsquared weights are exact for an area-sampled edge at any blur
// width), and a strength-weighted total-least-squares line is fitted through the points.
constexpr float kProfileStep = 0.25f;   // pixels between taps
constexpr double kProfileSpacing = 1.5;  // pixels between profiles along a side
constexpr int kMaxProfilePoints = 96;
constexpr float kLobeCut = 0.05f;        // the lobe ends where the gradient falls below this fraction
constexpr float kMinStep = 3.0f;         // weakest accepted step, in grey levels over two taps

void RefineEdgesProfile(int quad_decimate, const image_u8_t *im, struct quad *quad) {
  const float range = static_cast<float>(quad_decimate + 1);
  const int taps = 2 * static_cast<int>(range / kProfileStep) + 3;
  constexpr int kMaxTaps = 96;
  if (taps > kMaxTaps) {
    RefineEdgesT<double>(quad_decimate, im, quad);
    return;
  }
  const int max_x = im->width - 1;
  const int max_y = im->height - 1;
  double lines[4][4];

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
    const int np = std::min(kMaxProfilePoints, std::max(8, static_cast<int>(mag / kProfileSpacing)));

    const double xm = 0.5 * (quad->p[a][0] + quad->p[b][0]);
    const double ym = 0.5 * (quad->p[a][1] + quad->p[b][1]);
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    int used = 0;

    for (int s = 0; s < np; s++) {
      const double alpha = (1.0 + s) / (np + 1);
      const double x0 = alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0];
      const double y0 = alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1];
      float prof[kMaxTaps];
      bool all_in = true;
      for (int m = 0; m < taps; ++m) {
        const float u = -range - kProfileStep + kProfileStep * m;
        const float x = static_cast<float>(x0) + u * static_cast<float>(nx) - 0.5f;
        const float y = static_cast<float>(y0) + u * static_cast<float>(ny) - 0.5f;
        const float xt = std::floor(x), yt = std::floor(y);
        const int xi = static_cast<int>(xt), yi = static_cast<int>(yt);
        if (xi < 0 || xi >= max_x || yi < 0 || yi >= max_y) {
          all_in = false;
          break;
        }
        prof[m] = BilinearAt<float>(im, xi, yi, x - xt, y - yt);
      }
      if (!all_in) continue;

      // Gradient along the normal (outside brighter than inside for a normal border).
      float g[kMaxTaps];
      float best = 0;
      int bj = -1;
      for (int j = 1; j < taps - 1; ++j) {
        g[j] = prof[j + 1] - prof[j - 1];
        if (g[j] > best) {
          best = g[j];
          bj = j;
        }
      }
      if (bj < 0 || best < kMinStep) continue;
      int jl = bj, jr = bj;
      while (jl > 1 && g[jl - 1] > kLobeCut * best) jl--;
      while (jr < taps - 2 && g[jr + 1] > kLobeCut * best) jr++;
      double gs = 0, gj = 0;
      for (int j = jl; j <= jr; ++j) {
        gs += g[j];
        gj += static_cast<double>(g[j]) * j;
      }
      const double n0 = -range - kProfileStep + kProfileStep * (gj / gs);
      const double bx = x0 + n0 * nx - xm, by = y0 + n0 * ny - ym;
      const double w = best;
      sw += w;
      sx += w * bx;
      sy += w * by;
      sxx += w * bx * bx;
      sxy += w * bx * by;
      syy += w * by * by;
      used++;
    }
    if (used < 3) {
      // Too few usable profiles: keep this side where it was.
      lines[edge][0] = xm;
      lines[edge][1] = ym;
      lines[edge][2] = nx;
      lines[edge][3] = ny;
      continue;
    }
    const double mx = sx / sw, my = sy / sw;
    const double Cxx = sxx / sw - mx * mx;
    const double Cxy = sxy / sw - mx * my;
    const double Cyy = syy / sw - my * my;
    const double normal_theta = 0.5 * std::atan2(-2 * Cxy, Cyy - Cxx);
    lines[edge][0] = xm + mx;
    lines[edge][1] = ym + my;
    lines[edge][2] = std::cos(normal_theta);
    lines[edge][3] = std::sin(normal_theta);
  }
  IntersectLines(lines, quad);
}

}  // namespace

RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured) {
  if (const char *m = std::getenv("APRILTAG_VK_REFINE")) {
    const std::string method = m;
    if (method == "upstream") return RefineEdgesMethod::kUpstream;
    if (method == "exact") return RefineEdgesMethod::kExact;
    if (method == "fast") return RefineEdgesMethod::kFast;
    if (method == "ultrafast") return RefineEdgesMethod::kUltraFast;
    if (method == "profile") return RefineEdgesMethod::kProfile;
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
    case RefineEdgesMethod::kUltraFast:
      RefineEdgesFast(td->quad_decimate, im, quad);
      return;
    case RefineEdgesMethod::kProfile:
      RefineEdgesProfile(td->quad_decimate, im, quad);
      return;
    case RefineEdgesMethod::kUpstream:
      refine_edges(td, im, quad);
      return;
  }
}

}  // namespace apriltag_vulkan
