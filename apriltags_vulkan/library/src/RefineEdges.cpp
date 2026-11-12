#include "vkapriltag/RefineEdges.h"

#include "TestHooks.h"

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

// Refinements of the profile fit, from APRILTAG_VK_PROFILE_FEATURES (a bit mask, read once; all five by default).
enum ProfileFeature : unsigned {
  kDropOutliers = 1u, kCoarseToFine = 2u, kHalfwayCrossing = 4u, kClipBias = 8u, kWidenBlur = 16u
};
unsigned ProfileFeatures() {
  static const unsigned features = [] {
    const char *v = std::getenv("APRILTAG_VK_PROFILE_FEATURES");
    return v != nullptr ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : 31u;
  }();
  return features;
}

constexpr double kTruncatedFraction = 0.25;  // profiles touching a window end that mark a blurred quad
constexpr double kMinBlurRamp = 2.0;  // narrowest 20% to 80% rise, in pixels, of a blurred edge
constexpr double kLevelTaps = 20;      // taps across the window of the level pass (finest step 0.25 px)
constexpr double kMinLevelContrast = 50.0;  // smallest white minus black, in grey levels, the level pass accepts
constexpr double kLevelSpacing = 1.5;  // pixels between its profiles along a side
constexpr int kMaxProfileTaps = 96;
constexpr double kCrossingCells = 0.6;  // ramps wider than this many cells use the halfway crossing

// Bilinear sample of the image at pixel-edge coordinates (x, y); false outside it.
bool SampleAt(const image_u8_t *im, double x, double y, double *out) {
  const float fx = static_cast<float>(x) - 0.5f, fy = static_cast<float>(y) - 0.5f;
  const float xt = std::floor(fx), yt = std::floor(fy);
  const int xi = static_cast<int>(xt), yi = static_cast<int>(yt);
  if (xi < 0 || xi + 1 >= im->width || yi < 0 || yi + 1 >= im->height) return false;
  *out = BilinearAt<float>(im, xi, yi, fx - xt, fy - yt);
  return true;
}

// Weighted total-least-squares line through the points (relative to the side's midpoint): its centre
// and normal angle. With kDropOutliers, points far from the line are dropped and the line refitted.
void FitSide(double *bxs, double *bys, double *ws, int used, double range, double *mx, double *my,
             double *normal_theta) {
  auto fit = [&](int n) {
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < n; ++i) {
      sw += ws[i];
      sx += ws[i] * bxs[i];
      sy += ws[i] * bys[i];
      sxx += ws[i] * bxs[i] * bxs[i];
      sxy += ws[i] * bxs[i] * bys[i];
      syy += ws[i] * bys[i] * bys[i];
    }
    *mx = sx / sw;
    *my = sy / sw;
    const double Cxx = sxx / sw - *mx * *mx;
    const double Cxy = sxy / sw - *mx * *my;
    const double Cyy = syy / sw - *my * *my;
    *normal_theta = 0.5 * std::atan2(-2 * Cxy, Cyy - Cxx);
  };
  fit(used);
  if ((ProfileFeatures() & kDropOutliers) != 0) {
    // Drop points further than about a pixel from the line and refit, if most of them agree.
    const double tol = 1.0 + 0.1 * range;
    const double lx = -std::sin(*normal_theta), ly = std::cos(*normal_theta);  // the line's direction
    int kept = 0;
    for (int i = 0; i < used; ++i) {
      if (std::fabs((bxs[i] - *mx) * -ly + (bys[i] - *my) * lx) > tol) continue;
      bxs[kept] = bxs[i];
      bys[kept] = bys[i];
      ws[kept] = ws[i];
      ++kept;
    }
    if (kept < used && kept >= 3 && 2 * kept >= used) fit(kept);
  }
}

// One pass: profiles within `range` pixels either side of each current side, the sides refitted and
// intersected into new corners. Reports the mean width of the gradient lobes in *edge_width.
// With `cell` > 0 and kHalfwayCrossing, an edge whose ramp is wider than 0.6 of a cell is placed where
// the image crosses halfway between the white and black a half cell either side of it, which stays
// exact for any symmetric blur under about two cells (the gradient's centre of mass does not).
// Returns false, leaving the quad as it was, if a side's fit is degenerate.
bool ProfilePass(float range, double cell, const image_u8_t *im, struct quad *quad,
                 double *edge_width, double *truncated = nullptr) {
  const int taps = 2 * static_cast<int>(range / kProfileStep) + 3;
  constexpr int kMaxTaps = 96;
  if (taps > kMaxTaps) return false;
  const bool halfway = (ProfileFeatures() & kHalfwayCrossing) != 0;
  double width_sum = 0;
  int width_n = 0, touching = 0;
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

    const float fnx = static_cast<float>(nx), fny = static_cast<float>(ny);
    const double xm = 0.5 * (quad->p[a][0] + quad->p[b][0]);
    const double ym = 0.5 * (quad->p[a][1] + quad->p[b][1]);
    // The refined points of this side, relative to its midpoint, and their weights.
    double bxs[kMaxProfilePoints], bys[kMaxProfilePoints], ws[kMaxProfilePoints];
    int used = 0;

    for (int s = 0; s < np; s++) {
      const double alpha = (1.0 + s) / (np + 1);
      const double x0 = alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0];
      const double y0 = alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1];
      // The tap positions first (a loop the compiler can vectorise), then the loads. The positions
      // move along a line, so the first and last taps bound every tap's pixel.
      int xis[kMaxTaps], yis[kMaxTaps];
      float fxs[kMaxTaps], fys[kMaxTaps];
      const float fx0 = static_cast<float>(x0), fy0 = static_cast<float>(y0);
      for (int m = 0; m < taps; ++m) {
        const float u = -range - kProfileStep + kProfileStep * m;
        const float x = fx0 + u * fnx - 0.5f;
        const float y = fy0 + u * fny - 0.5f;
        const float xt = std::floor(x), yt = std::floor(y);
        xis[m] = static_cast<int>(xt);
        yis[m] = static_cast<int>(yt);
        fxs[m] = x - xt;
        fys[m] = y - yt;
      }
      if (xis[0] < 0 || xis[0] >= max_x || yis[0] < 0 || yis[0] >= max_y || xis[taps - 1] < 0 ||
          xis[taps - 1] >= max_x || yis[taps - 1] < 0 || yis[taps - 1] >= max_y) {
        continue;
      }
      float prof[kMaxTaps];
      for (int m = 0; m < taps; ++m) prof[m] = BilinearAt<float>(im, xis[m], yis[m], fxs[m], fys[m]);

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
      if (jl == 1 || jr == taps - 2) ++touching;
      width_sum += (jr - jl + 1) * kProfileStep;
      ++width_n;
      double cj = gj / gs;  // the edge's position, in taps
      if (cell > 0 && halfway &&
          (jr - jl + 1) * kProfileStep > kCrossingCells * cell) {
        const double half = 0.5 * cell;
        for (int iter = 0; iter < 3; ++iter) {
          const double pos = -range - kProfileStep + kProfileStep * cj;
          double white, black;
          if (!SampleAt(im, x0 + (pos - half) * nx, y0 + (pos - half) * ny, &white) ||
              !SampleAt(im, x0 + (pos + half) * nx, y0 + (pos + half) * ny, &black)) {
            break;
          }
          const double mid = 0.5 * (white + black);  // along +n the image goes from black to white
          // the profile's rising crossing of `mid` nearest the current estimate
          double best_c = -1, best_d = 1e9;
          for (int j = std::max(0, jl - 2); j + 1 <= std::min(taps - 1, jr + 2); ++j) {
            if (!(prof[j] <= mid && prof[j + 1] > mid)) continue;
            const double c = j + (mid - prof[j]) / (prof[j + 1] - prof[j]);
            if (std::fabs(c - cj) < best_d) {
              best_d = std::fabs(c - cj);
              best_c = c;
            }
          }
          if (best_c < 0) break;
          const bool settled = std::fabs(best_c - cj) < 1e-3;
          cj = best_c;
          if (settled) break;
        }
      }
      const double n0 = -range - kProfileStep + kProfileStep * cj;
      bxs[used] = x0 + n0 * nx - xm;
      bys[used] = y0 + n0 * ny - ym;
      ws[used] = best;
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
    double mx = 0, my = 0, normal_theta = 0;
    FitSide(bxs, bys, ws, used, range, &mx, &my, &normal_theta);
    lines[edge][0] = xm + mx;
    lines[edge][1] = ym + my;
    lines[edge][2] = std::cos(normal_theta);
    lines[edge][3] = std::sin(normal_theta);
  }
  IntersectLines(lines, quad);
  *edge_width = width_n > 0 ? width_sum / width_n : 0.0;
  if (truncated != nullptr) *truncated = width_n > 0 ? static_cast<double>(touching) / width_n : 0.0;
  return true;
}

// For quads whose edges are motion-blurred. Each profile spans the outer quiet zone to the middle of
// the border; the edge is placed where it crosses halfway between the side's white and black levels
// (the white from the outer end, the black from the inner end), which is exact for a symmetric blur.
// *ramp receives the median 20% to 80% rise. Returns false, leaving the quad as it was, if no side
// could be refitted.
bool LevelPass(double cell, const image_u8_t *im, struct quad *quad, double *ramp = nullptr) {
  const double r_out = std::min(0.5 * cell, 20.0), r_in = std::min(0.5 * cell, 20.0);
  const float step = static_cast<float>(std::max<double>(kProfileStep, std::min(1.0, (r_in + r_out) / kLevelTaps)));
  const int taps = static_cast<int>((r_in + r_out) / step) + 1;
  if (taps < 8 || taps > kMaxProfileTaps) return false;
  const int ends = std::max(2, static_cast<int>(1.0f / step));
  const int max_x = im->width - 1;
  const int max_y = im->height - 1;
  double lines[4][4];
  bool any = false;
  double widths[4 * kMaxProfilePoints];
  int nwidths = 0;

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
    const int np = std::min(kMaxProfilePoints, std::max(8, static_cast<int>(mag / kLevelSpacing)));
    const float fnx = static_cast<float>(nx), fny = static_cast<float>(ny);
    const double xm = 0.5 * (quad->p[a][0] + quad->p[b][0]);
    const double ym = 0.5 * (quad->p[a][1] + quad->p[b][1]);

    // Tap m of the profile through (x0, y0); false outside the image.
    auto at = [&](float fx0, float fy0, int m, float *v) {
      const float u = static_cast<float>(-r_in) + step * m;
      const float x = fx0 + u * fnx - 0.5f;
      const float y = fy0 + u * fny - 0.5f;
      const float xt = std::floor(x), yt = std::floor(y);
      const int xi = static_cast<int>(xt), yi = static_cast<int>(yt);
      if (xi < 0 || xi >= max_x || yi < 0 || yi >= max_y) return false;
      *v = BilinearAt<float>(im, xi, yi, x - xt, y - yt);
      return true;
    };
    double px[kMaxProfilePoints], py[kMaxProfilePoints], outer[kMaxProfilePoints], inner[kMaxProfilePoints];
    int count = 0;
    // Three profiles whose ends differ by too little rule the side out before the rest are sampled.
    bool contrast = false;
    for (int k = 1; k <= 3 && !contrast; ++k) {
      const double alpha = k / 4.0;
      const float fx0 = static_cast<float>(alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0]);
      const float fy0 = static_cast<float>(alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1]);
      float lo = 0, hi = 0;
      contrast = at(fx0, fy0, 0, &lo) && at(fx0, fy0, taps - 1, &hi) && hi - lo >= kMinLevelContrast;
    }
    for (int s = 0; s < np && contrast; s++) {
      const double alpha = (1.0 + s) / (np + 1);
      const double x0 = alpha * quad->p[a][0] + (1 - alpha) * quad->p[b][0];
      const double y0 = alpha * quad->p[a][1] + (1 - alpha) * quad->p[b][1];
      const float fx0 = static_cast<float>(x0), fy0 = static_cast<float>(y0);
      bool inside = true;
      double lo = 0, hi = 0;
      for (int m = 0; m < ends && inside; ++m) {
        float vl = 0, vh = 0;
        inside = at(fx0, fy0, m, &vl) && at(fx0, fy0, taps - 1 - m, &vh);
        lo += vl;
        hi += vh;
      }
      if (!inside) continue;
      inner[count] = lo / ends;
      outer[count] = hi / ends;
      px[count] = x0;
      py[count] = y0;
      ++count;
    }
    double bxs[kMaxProfilePoints], bys[kMaxProfilePoints], ws[kMaxProfilePoints];
    int used = 0;
    if (count >= 3) {
      double tmp[kMaxProfilePoints];
      std::copy(outer, outer + count, tmp);
      std::nth_element(tmp, tmp + count / 2, tmp + count);
      const double white = tmp[count / 2];
      std::copy(inner, inner + count, tmp);
      std::nth_element(tmp, tmp + count / 2, tmp + count);
      const double black = tmp[count / 2];
      if (white - black >= kMinLevelContrast) {
        const double mid = 0.5 * (white + black), margin = 0.25 * (white - black);
        for (int i = 0; i < count; ++i) {
          if (outer[i] < mid + margin || inner[i] > mid - margin) continue;
          float pr[kMaxProfileTaps];
          for (int m = 0; m < taps; ++m) at(static_cast<float>(px[i]), static_cast<float>(py[i]), m, &pr[m]);
          int below = 0, below20 = 0, below80 = 0;
          for (int m = 0; m < taps; ++m) {
            below += pr[m] < mid;
            below20 += pr[m] < mid - 0.3 * (white - black);
            below80 += pr[m] < mid + 0.3 * (white - black);
          }
          widths[nwidths++] = step * (below80 - below20);
          if (below < 1 || below >= taps) continue;
          double frac = 0.5;
          if (pr[below] > pr[below - 1]) {
            frac = std::min(1.0, std::max(0.0, (mid - pr[below - 1]) / (pr[below] - pr[below - 1])));
          }
          const double n0 = -r_in + step * (below - 1 + frac);
          bxs[used] = px[i] + n0 * nx - xm;
          bys[used] = py[i] + n0 * ny - ym;
          ws[used] = 1.0;
          ++used;
        }
      }
    }
    if (used < 3) {
      lines[edge][0] = xm;
      lines[edge][1] = ym;
      lines[edge][2] = nx;
      lines[edge][3] = ny;
      continue;
    }
    any = true;
    double mx = 0, my = 0, normal_theta = 0;
    FitSide(bxs, bys, ws, used, 0.5 * (r_in + r_out), &mx, &my, &normal_theta);
    lines[edge][0] = xm + mx;
    lines[edge][1] = ym + my;
    lines[edge][2] = std::cos(normal_theta);
    lines[edge][3] = std::sin(normal_theta);
  }
  if (!any) return false;
  if (ramp != nullptr && nwidths > 0) {
    std::nth_element(widths, widths + nwidths / 2, widths + nwidths);
    *ramp = widths[nwidths / 2];
  }
  IntersectLines(lines, quad);
  return true;
}

// Heckbert's projective map of the unit square onto quad q, applied to (u, v); q's corners 0..3 are
// the square's (0,0), (1,0), (1,1) and (0,1).
void SquareToQuad(const struct quad *q, double u, double v, double *x, double *y) {
  const double dx1 = q->p[1][0] - q->p[2][0], dx2 = q->p[3][0] - q->p[2][0];
  const double dy1 = q->p[1][1] - q->p[2][1], dy2 = q->p[3][1] - q->p[2][1];
  const double sx = q->p[0][0] - q->p[1][0] + q->p[2][0] - q->p[3][0];
  const double sy = q->p[0][1] - q->p[1][1] + q->p[2][1] - q->p[3][1];
  const double den = dx1 * dy2 - dx2 * dy1;
  const double g = den != 0 ? (sx * dy2 - dx2 * sy) / den : 0;
  const double h = den != 0 ? (dx1 * sy - sx * dy1) / den : 0;
  const double a = q->p[1][0] - q->p[0][0] + g * q->p[1][0], b = q->p[3][0] - q->p[0][0] + h * q->p[3][0];
  const double d = q->p[1][1] - q->p[0][1] + g * q->p[1][1], e = q->p[3][1] - q->p[0][1] + h * q->p[3][1];
  const double w = g * u + h * v + 1;
  *x = (a * u + b * v + q->p[0][0]) / w;
  *y = (d * u + e * v + q->p[0][1]) / w;
}

// Centre of mass of the strongest positive gradient lobe of prof[from..to] (falling: the strongest
// negative one); -1 if there is none.
double LobeCentroid(const double *prof, int from, int to, bool falling) {
  double g[64], best = 0;
  int bj = -1;
  for (int j = from; j <= to; ++j) {
    g[j] = falling ? prof[j - 1] - prof[j + 1] : prof[j + 1] - prof[j - 1];
    if (g[j] > best) {
      best = g[j];
      bj = j;
    }
  }
  if (bj < 0) return -1;
  int jl = bj, jr = bj;
  while (jl > from && g[jl - 1] > 0.05 * best) jl--;
  while (jr < to && g[jr + 1] > 0.05 * best) jr++;
  double sg = 0, sj = 0;
  for (int j = jl; j <= jr; ++j) {
    sg += g[j];
    sj += g[j] * j;
  }
  return sj / sg;
}

// Clipped exposure shrinks every black region by about the same distance delta, so a border that
// should be one cell wide measures cell - 2 delta while the fitted outline is itself 2 delta small.
// Measures the border's width across white payload cells and returns delta in pixels (0 when the
// levels are not clipped, too few cells can be measured, or the sides disagree).
double ClipBias(const struct quad *q, int width_cells, const image_u8_t *im) {
  const double cells = width_cells;
  double black_v[64], white_v[64];
  int nb = 0, nw = 0;
  auto at = [&](double u, double v, double *out) {
    double x, y;
    SquareToQuad(q, u, v, &x, &y);
    return SampleAt(im, x, y, out);
  };
  for (int i = 0; i < width_cells && nb < 62; ++i) {  // border cells along the top and bottom
    double v;
    if (at((i + 0.5) / cells, 0.5 / cells, &v)) black_v[nb++] = v;
    if (at((i + 0.5) / cells, (cells - 0.5) / cells, &v)) black_v[nb++] = v;
  }
  for (int i = 0; i < width_cells && nw < 62; ++i) {  // the quiet ring half a cell outside
    double v;
    if (at((i + 0.5) / cells, -0.5 / cells, &v)) white_v[nw++] = v;
    if (at((i + 0.5) / cells, (cells + 0.5) / cells, &v)) white_v[nw++] = v;
  }
  if (nb < 4 || nw < 4) return 0;
  std::nth_element(black_v, black_v + nb / 2, black_v + nb);
  std::nth_element(white_v, white_v + nw / 2, white_v + nw);
  const double black = black_v[nb / 2], white = white_v[nw / 2];
  if (white < 250 && black > 4) return 0;
  const double mid = 0.5 * (black + white);

  constexpr int kN = 33;
  double sum = 0, side_sum[4] = {0, 0, 0, 0};
  int n = 0, side_n[4] = {0, 0, 0, 0};
  for (int side = 0; side < 4; ++side) {
    for (int c = 1; c < width_cells - 1; ++c) {
      const double a = (c + 0.5) / cells;
      // (u, v) at depth t (in cells) inward from side `side`, at position a along it
      auto uv = [&](double t, double *u, double *v) {
        const double tt = t / cells;
        *u = side == 0 ? a : side == 1 ? 1 - tt : side == 2 ? a : tt;
        *v = side == 0 ? tt : side == 1 ? a : side == 2 ? 1 - tt : a;
      };
      double u, v, sample;
      uv(1.5, &u, &v);
      if (!at(u, v, &sample) || sample < mid) continue;  // a dark payload cell
      double prof[kN], px[kN], py[kN];
      bool ok = true;
      for (int j = 0; j < kN && ok; ++j) {
        uv(-0.5 + 2.0 * j / (kN - 1), &u, &v);
        SquareToQuad(q, u, v, &px[j], &py[j]);
        ok = SampleAt(im, px[j], py[j], &prof[j]);
      }
      if (!ok) continue;
      const double jo = LobeCentroid(prof, 1, kN / 2 - 1, true);
      const double ji = LobeCentroid(prof, kN / 2 + 1, kN - 2, false);
      if (jo < 0 || ji < 0) continue;
      auto along = [&](double j, double *x, double *y) {
        const int j0 = static_cast<int>(j);
        const double f = j - j0;
        *x = px[j0] + (px[j0 + 1] - px[j0]) * f;
        *y = py[j0] + (py[j0 + 1] - py[j0]) * f;
      };
      double ox, oy, ix, iy;
      along(jo, &ox, &oy);
      along(ji, &ix, &iy);
      const double border_px = std::hypot(ix - ox, iy - oy);
      const double cell_px = std::hypot(px[24] - px[8], py[24] - py[8]);  // depth 0 to depth 1
      const double d = (cell_px - border_px) / (2.0 * (1.0 - 1.0 / cells));
      sum += d;
      side_sum[side] += d;
      ++n;
      ++side_n[side];
    }
  }
  if (n < 4) return 0;
  const double delta = sum / n;
  for (int side = 0; side < 4; ++side) {
    if (side_n[side] > 0 && std::fabs(side_sum[side] / side_n[side] - delta) > 0.05 + 0.25 * std::fabs(delta)) {
      return 0;  // the sides disagree: the border is not one cell wide
    }
  }
  return std::fabs(delta) > 0.02 && std::fabs(delta) < 1.5 ? delta : 0;
}

// Moves every side of the quad outward by d pixels and re-intersects them.
void GrowQuad(struct quad *q, double d) {
  double cx = 0, cy = 0, px[4], py[4], dx[4], dy[4];
  for (int k = 0; k < 4; ++k) {
    cx += 0.25 * q->p[k][0];
    cy += 0.25 * q->p[k][1];
  }
  for (int k = 0; k < 4; ++k) {
    const double ax = q->p[k][0], ay = q->p[k][1];
    const double ex = q->p[(k + 1) & 3][0] - ax, ey = q->p[(k + 1) & 3][1] - ay;
    const double len = std::hypot(ex, ey);
    double nx = ey / len, ny = -ex / len;
    if ((ax - cx) * nx + (ay - cy) * ny < 0) {
      nx = -nx;
      ny = -ny;
    }
    px[k] = ax + nx * d;
    py[k] = ay + ny * d;
    dx[k] = ex / len;
    dy[k] = ey / len;
  }
  for (int k = 0; k < 4; ++k) {
    const int p = (k + 3) & 3;
    const double cross = dx[p] * dy[k] - dy[p] * dx[k];
    if (std::fabs(cross) < 1e-9) return;
    const double t = ((px[k] - px[p]) * dy[k] - (py[k] - py[p]) * dx[k]) / cross;
    q->p[k][0] = static_cast<float>(px[p] + t * dx[p]);
    q->p[k][1] = static_cast<float>(py[p] + t * dy[p]);
  }
}

void RefineEdgesProfile(const apriltag_detector_t *td, const image_u8_t *im, struct quad *quad) {
  const int quad_decimate = static_cast<int>(td->quad_decimate);
  float reach = static_cast<float>(quad_decimate + 1);
  double edge_width = 0;
  // The cell size in pixels, from the first family's border width; 0 when there is none.
  double cell = 0;
  if (zarray_size(td->tag_families) > 0) {
    apriltag_family_t *family = nullptr;
    zarray_get(td->tag_families, 0, &family);
    double side = 0;
    for (int k = 0; k < 4; ++k) {
      side += 0.25 * std::hypot(quad->p[(k + 1) & 3][0] - quad->p[k][0], quad->p[(k + 1) & 3][1] - quad->p[k][1]);
    }
    cell = side / family->width_at_border;
  }
  double truncated = 0;
  if (!ProfilePass(reach, cell, im, quad, &edge_width, &truncated)) {
    RefineEdgesT<double>(quad_decimate, im, quad);
    return;
  }
  if (cell <= 0) return;

  // Lobes running into the window ends mean a blurred quad: place its edges by the halfway crossing.
  bool widened = false;
  if ((ProfileFeatures() & kWidenBlur) != 0 && truncated > kTruncatedFraction) {
    struct quad before = *quad;
    double ramp = 0;
    if (LevelPass(cell, im, quad, &ramp) && ramp >= kMinBlurRamp) {
      widened = true;
    } else {
      *quad = before;
    }
  }

  // Narrow the window to the measured edge width (never above half a cell, so the next payload edge
  // stays out) and refit around the refined sides, while that still shrinks it.
  for (int pass = 0; pass < 3 && !widened && (ProfileFeatures() & kCoarseToFine) != 0; ++pass) {
    double next = std::max(1.5, 1.25 * edge_width);
    if (cell >= 3.0) next = std::min(next, 0.45 * cell);
    if (next >= 0.9 * reach) break;
    reach = static_cast<float>(next);
    struct quad before = *quad;
    if (!ProfilePass(reach, cell, im, quad, &edge_width)) {
      *quad = before;
      break;
    }
  }
  if ((ProfileFeatures() & kClipBias) != 0) {
    apriltag_family_t *family = nullptr;
    zarray_get(td->tag_families, 0, &family);
    const double delta = ClipBias(quad, family->width_at_border, im);
    if (delta != 0) GrowQuad(quad, delta);
  }
}

}  // namespace

namespace testing {

void IntersectLines(const double lines[4][4], struct quad *quad) { apriltag_vulkan::IntersectLines(lines, quad); }
void FitSide(double *bxs, double *bys, double *ws, int used, double range, double *mx, double *my,
             double *normal_theta) {
  apriltag_vulkan::FitSide(bxs, bys, ws, used, range, mx, my, normal_theta);
}

}  // namespace testing

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
      RefineEdgesProfile(td, im, quad);
      return;
    case RefineEdgesMethod::kUpstream:
      refine_edges(td, im, quad);
      return;
  }
}

}  // namespace apriltag_vulkan
