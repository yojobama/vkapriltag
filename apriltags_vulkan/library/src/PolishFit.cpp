#include "vkapriltag/PolishFit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace apriltag_vulkan {
namespace {

constexpr int kMaxCells = 16;     // widest supported grid
constexpr int kMaxCols = 24;      // columns (or rows) one pixel's blur window may touch
constexpr int kNL = 8;            // homography parameters
constexpr int kNP = 13;           // homography, two blur widths, three levels
constexpr double kReach = 3.5;    // blur window, in standard deviations
constexpr int kMaxPixels = 6000;  // working set cap

// Standard normal CDF and density, tabulated over [-4, 4].
struct NormalTable {
  static constexpr int kSteps = 128;
  static constexpr int kN = 8 * kSteps + 2;
  double cdf[kN], pdf[kN];
  NormalTable() {
    for (int i = 0; i < kN; ++i) {
      const double z = -4.0 + static_cast<double>(i) / kSteps;
      cdf[i] = 0.5 * std::erfc(-z / std::sqrt(2.0));
      pdf[i] = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
    }
  }
};
const NormalTable kNormal;

inline void NormalAt(double z, double *cdf, double *pdf) {
  if (z <= -4.0) {
    *cdf = 0.0;
    *pdf = 0.0;
    return;
  }
  if (z >= 4.0) {
    *cdf = 1.0;
    *pdf = 0.0;
    return;
  }
  const double t = (z + 4.0) * NormalTable::kSteps;
  const int i = static_cast<int>(t);
  const double f = t - i;
  *cdf = kNormal.cdf[i] + f * (kNormal.cdf[i + 1] - kNormal.cdf[i]);
  *pdf = kNormal.pdf[i] + f * (kNormal.pdf[i + 1] - kNormal.pdf[i]);
}

// A blurred indicator of each cell column (index -1 is everything left of the grid, W everything
// right of it) at one position, with derivatives with respect to the position and the blur width.
struct Axis {
  int lo, hi;
  double a[kMaxCols], da_du[kMaxCols], da_ds[kMaxCols];
};

void ComputeAxis(double u, double s, int width, bool derivatives, Axis *ax) {
  const double reach = kReach * s;
  int lo = std::clamp(static_cast<int>(std::floor(u - reach)), -1, width);
  int hi = std::clamp(static_cast<int>(std::floor(u + reach)), -1, width);
  if (hi - lo > kMaxCols - 3) {
    const int mid = std::clamp(static_cast<int>(std::floor(u)), -1, width);
    lo = std::max(lo, mid - (kMaxCols - 3) / 2);
    hi = std::min(hi, lo + kMaxCols - 3);
  }
  ax->lo = lo;
  ax->hi = hi;
  double cdf[kMaxCols + 1], pdf[kMaxCols + 1], zs[kMaxCols + 1];
  const double inv_s = 1.0 / s;
  for (int k = lo; k <= hi + 1; ++k) {
    const int idx = k - lo;
    if (k < 0) {
      cdf[idx] = 0.0;
      pdf[idx] = 0.0;
      zs[idx] = 0.0;
    } else if (k > width) {
      cdf[idx] = 1.0;
      pdf[idx] = 0.0;
      zs[idx] = 0.0;
    } else {
      zs[idx] = (k - u) * inv_s;
      NormalAt(zs[idx], &cdf[idx], &pdf[idx]);
    }
  }
  for (int i = lo; i <= hi; ++i) {
    const int idx = i - lo;
    ax->a[idx] = cdf[idx + 1] - cdf[idx];
    if (derivatives) {
      ax->da_du[idx] = (pdf[idx] - pdf[idx + 1]) * inv_s;
      ax->da_ds[idx] = (pdf[idx] * zs[idx] - pdf[idx + 1] * zs[idx + 1]) * inv_s;
    }
  }
}

struct Fit {
  int width;
  uint8_t cls[kMaxCells][kMaxCells];  // [row][column]: 0 black, 1 white
  double cx, cy, scale;               // pixel -> normalised coordinates
  std::vector<float> nx, ny, obs;     // working set: normalised position and observed value
};

inline int ClassOf(const Fit &f, int row, int col) {
  return (row < 0 || row >= f.width || col < 0 || col >= f.width) ? 2 : f.cls[row][col];
}

struct Mapping {
  double g[8];  // normalised pixel -> cell: u = (g0 x + g1 y + g2) / w, v = (g3 x + g4 y + g5) / w, w = g6 x + g7 y + 1
};

inline void Project(const Mapping &m, double x, double y, double *u, double *v, double *w) {
  *w = m.g[6] * x + m.g[7] * y + 1.0;
  *u = (m.g[0] * x + m.g[1] * y + m.g[2]) / *w;
  *v = (m.g[3] * x + m.g[4] * y + m.g[5]) / *w;
}

bool Invert3(const double a[9], double out[9]) {
  const double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
  const double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
  if (std::fabs(det) < 1e-300) return false;
  const double id = 1.0 / det;
  out[0] = c00 * id;
  out[1] = (a[2] * a[7] - a[1] * a[8]) * id;
  out[2] = (a[1] * a[5] - a[2] * a[4]) * id;
  out[3] = c01 * id;
  out[4] = (a[0] * a[8] - a[2] * a[6]) * id;
  out[5] = (a[2] * a[3] - a[0] * a[5]) * id;
  out[6] = c02 * id;
  out[7] = (a[1] * a[6] - a[0] * a[7]) * id;
  out[8] = (a[0] * a[4] - a[1] * a[3]) * id;
  return true;
}

// The cell-to-normalised-pixel matrix of a mapping (the inverse of g, with g22 = 1).
bool CellToNormalised(const Mapping &m, double out[9]) {
  const double a[9] = {m.g[0], m.g[1], m.g[2], m.g[3], m.g[4], m.g[5], m.g[6], m.g[7], 1.0};
  return Invert3(a, out);
}

void CellCorners(const Mapping &m, int width, double xy[4][2]) {
  double inv[9];
  CellToNormalised(m, inv);
  const double cu[4] = {0, static_cast<double>(width), static_cast<double>(width), 0};
  const double cv[4] = {0, 0, static_cast<double>(width), static_cast<double>(width)};
  for (int k = 0; k < 4; ++k) {
    const double w = inv[6] * cu[k] + inv[7] * cv[k] + inv[8];
    xy[k][0] = (inv[0] * cu[k] + inv[1] * cv[k] + inv[2]) / w;
    xy[k][1] = (inv[3] * cu[k] + inv[4] * cv[k] + inv[5]) / w;
  }
}

struct Params {
  Mapping map;
  double su, sv;  // blur standard deviation along the cell axes, in pixels (pixel footprint included)
  double lev[3];  // black, white, outside
};

struct Accum {
  double cost = 0;
  double jtj[kNP][kNP];
  double jtr[kNP];
  int n = 0;
};

// Evaluates the model over the working set. With `normal` set, accumulates the normal equations;
// `kcut` is the Huber threshold.
void Evaluate(const Fit &f, const Params &p, double kcut, bool normal, Accum *acc) {
  acc->cost = 0;
  acc->n = 0;
  if (normal) {
    for (auto &r : acc->jtj) std::fill(std::begin(r), std::end(r), 0.0);
    std::fill(std::begin(acc->jtr), std::end(acc->jtr), 0.0);
  }
  const int W = f.width;
  const size_t count = f.obs.size();
  Axis ax_u, ax_v;
  for (size_t idx = 0; idx < count; ++idx) {
    const double x = f.nx[idx], y = f.ny[idx];
    double u, v, w;
    Project(p.map, x, y, &u, &v, &w);
    const double iw = 1.0 / w;
    const double gux = (p.map.g[0] - u * p.map.g[6]) * iw, guy = (p.map.g[1] - u * p.map.g[7]) * iw;
    const double gvx = (p.map.g[3] - v * p.map.g[6]) * iw, gvy = (p.map.g[4] - v * p.map.g[7]) * iw;
    const double gu = std::sqrt(gux * gux + guy * guy) / f.scale;  // cells per pixel
    const double gv = std::sqrt(gvx * gvx + gvy * gvy) / f.scale;
    const double s_u = p.su * gu, s_v = p.sv * gv;
    ComputeAxis(u, s_u, W, normal, &ax_u);
    ComputeAxis(v, s_v, W, normal, &ax_v);

    double S[3] = {0, 0, 0}, Su[3] = {0, 0, 0}, Sv[3] = {0, 0, 0}, Ssu[3] = {0, 0, 0}, Ssv[3] = {0, 0, 0};
    for (int j = ax_v.lo; j <= ax_v.hi; ++j) {
      const double cv = ax_v.a[j - ax_v.lo];
      const double dcv = normal ? ax_v.da_du[j - ax_v.lo] : 0.0;
      const double dcs = normal ? ax_v.da_ds[j - ax_v.lo] : 0.0;
      for (int i = ax_u.lo; i <= ax_u.hi; ++i) {
        const int c = ClassOf(f, j, i);
        const double au = ax_u.a[i - ax_u.lo];
        S[c] += au * cv;
        if (normal) {
          Su[c] += ax_u.da_du[i - ax_u.lo] * cv;
          Sv[c] += au * dcv;
          Ssu[c] += ax_u.da_ds[i - ax_u.lo] * cv;
          Ssv[c] += au * dcs;
        }
      }
    }
    double pred = p.lev[0] * S[0] + p.lev[1] * S[1] + p.lev[2] * S[2];
    const double ob = f.obs[idx];
    // Saturated pixels the model also puts at or beyond the range carry no information.
    if ((ob >= 255.0 && pred >= 255.0) || (ob <= 0.0 && pred <= 0.0)) continue;
    const double r = ob - pred;
    const double ar = std::fabs(r);
    const double wt = ar <= kcut ? 1.0 : kcut / ar;
    acc->cost += ar <= kcut ? r * r : kcut * (2.0 * ar - kcut);
    ++acc->n;
    if (!normal) continue;

    double jr[kNP];
    const double pu = p.lev[0] * Su[0] + p.lev[1] * Su[1] + p.lev[2] * Su[2];
    const double pv = p.lev[0] * Sv[0] + p.lev[1] * Sv[1] + p.lev[2] * Sv[2];
    const double dg_u[8] = {x * iw, y * iw, iw, 0, 0, 0, -u * x * iw, -u * y * iw};
    const double dg_v[8] = {0, 0, 0, x * iw, y * iw, iw, -v * x * iw, -v * y * iw};
    for (int k = 0; k < 8; ++k) jr[k] = pu * dg_u[k] + pv * dg_v[k];
    jr[8] = (p.lev[0] * Ssu[0] + p.lev[1] * Ssu[1] + p.lev[2] * Ssu[2]) * gu;
    jr[9] = (p.lev[0] * Ssv[0] + p.lev[1] * Ssv[1] + p.lev[2] * Ssv[2]) * gv;
    jr[10] = S[0];
    jr[11] = S[1];
    jr[12] = S[2];
    for (int a = 0; a < kNP; ++a) {
      const double ja = wt * jr[a];
      acc->jtr[a] += ja * r;
      for (int b = a; b < kNP; ++b) acc->jtj[a][b] += ja * jr[b];
    }
  }
  if (normal) {
    for (int a = 0; a < kNP; ++a) {
      for (int b = 0; b < a; ++b) acc->jtj[a][b] = acc->jtj[b][a];
    }
  }
}

// Solves (A) x = b by Gaussian elimination with partial pivoting; false if singular.
bool Solve(double a[kNP][kNP], double b[kNP], double x[kNP]) {
  for (int c = 0; c < kNP; ++c) {
    int piv = c;
    for (int r = c + 1; r < kNP; ++r) {
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    }
    if (std::fabs(a[piv][c]) < 1e-18) return false;
    if (piv != c) {
      for (int k = 0; k < kNP; ++k) std::swap(a[c][k], a[piv][k]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < kNP; ++r) {
      const double m = a[r][c] / a[c][c];
      if (m == 0.0) continue;
      for (int k = c; k < kNP; ++k) a[r][k] -= m * a[c][k];
      b[r] -= m * b[c];
    }
  }
  for (int r = kNP - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < kNP; ++k) s -= a[r][k] * x[k];
    x[r] = s / a[r][r];
  }
  return true;
}

// Builds the working set: the pixels whose blur window touches cells of more than one class.
void BuildSet(Fit *f, const Mapping &map, double half_window_px, int image_w, int image_h, const image_u8_t *im) {
  const int W = f->width;
  double inv[9];
  CellToNormalised(map, inv);
  double minx = 1e30, miny = 1e30, maxx = -1e30, maxy = -1e30;
  const double ext = 0.8;
  const double cu[4] = {-ext, W + ext, W + ext, -ext}, cv[4] = {-ext, -ext, W + ext, W + ext};
  for (int k = 0; k < 4; ++k) {
    const double w = inv[6] * cu[k] + inv[7] * cv[k] + inv[8];
    const double x = (inv[0] * cu[k] + inv[1] * cv[k] + inv[2]) / w * f->scale + f->cx;
    const double y = (inv[3] * cu[k] + inv[4] * cv[k] + inv[5]) / w * f->scale + f->cy;
    minx = std::min(minx, x);
    maxx = std::max(maxx, x);
    miny = std::min(miny, y);
    maxy = std::max(maxy, y);
  }
  const int x0 = std::max(0, static_cast<int>(std::floor(minx))), x1 = std::min(image_w - 1, static_cast<int>(std::ceil(maxx)));
  const int y0 = std::max(0, static_cast<int>(std::floor(miny))), y1 = std::min(image_h - 1, static_cast<int>(std::ceil(maxy)));
  f->nx.clear();
  f->ny.clear();
  f->obs.clear();
  for (int py = y0; py <= y1; ++py) {
    for (int px = x0; px <= x1; ++px) {
      const double x = (px + 0.5 - f->cx) / f->scale, y = (py + 0.5 - f->cy) / f->scale;
      double u, v, w;
      Project(map, x, y, &u, &v, &w);
      if (u < -ext || u > W + ext || v < -ext || v > W + ext) continue;
      const double iw = 1.0 / w;
      const double gux = (map.g[0] - u * map.g[6]) * iw, guy = (map.g[1] - u * map.g[7]) * iw;
      const double gvx = (map.g[3] - v * map.g[6]) * iw, gvy = (map.g[4] - v * map.g[7]) * iw;
      const double ru = half_window_px * std::sqrt(gux * gux + guy * guy) / f->scale;
      const double rv = half_window_px * std::sqrt(gvx * gvx + gvy * gvy) / f->scale;
      const int i0 = std::clamp(static_cast<int>(std::floor(u - ru)), -1, W), i1 = std::clamp(static_cast<int>(std::floor(u + ru)), -1, W);
      const int j0 = std::clamp(static_cast<int>(std::floor(v - rv)), -1, W), j1 = std::clamp(static_cast<int>(std::floor(v + rv)), -1, W);
      const int first = ClassOf(*f, j0, i0);
      bool varies = false;
      for (int j = j0; j <= j1 && !varies; ++j) {
        for (int i = i0; i <= i1; ++i) {
          if (ClassOf(*f, j, i) != first) {
            varies = true;
            break;
          }
        }
      }
      if (!varies) continue;
      f->nx.push_back(static_cast<float>(x));
      f->ny.push_back(static_cast<float>(y));
      f->obs.push_back(static_cast<float>(im->buf[py * im->stride + px]));
    }
  }
  const size_t n = f->obs.size();
  if (n > static_cast<size_t>(kMaxPixels)) {
    const size_t stride = (n + kMaxPixels - 1) / kMaxPixels;
    size_t k = 0;
    for (size_t i = 0; i < n; i += stride, ++k) {
      f->nx[k] = f->nx[i];
      f->ny[k] = f->ny[i];
      f->obs[k] = f->obs[i];
    }
    f->nx.resize(k);
    f->ny.resize(k);
    f->obs.resize(k);
  }
}

// Linear least squares for the three levels at fixed mapping and blur.
bool FitLevels(const Fit &f, Params *p) {
  Params q = *p;
  Axis ax_u, ax_v;
  double ata[3][3] = {}, atb[3] = {};
  const int W = f.width;
  for (size_t idx = 0; idx < f.obs.size(); ++idx) {
    double u, v, w;
    Project(q.map, f.nx[idx], f.ny[idx], &u, &v, &w);
    const double iw = 1.0 / w;
    const double gux = (q.map.g[0] - u * q.map.g[6]) * iw, guy = (q.map.g[1] - u * q.map.g[7]) * iw;
    const double gvx = (q.map.g[3] - v * q.map.g[6]) * iw, gvy = (q.map.g[4] - v * q.map.g[7]) * iw;
    ComputeAxis(u, q.su * std::sqrt(gux * gux + guy * guy) / f.scale, W, false, &ax_u);
    ComputeAxis(v, q.sv * std::sqrt(gvx * gvx + gvy * gvy) / f.scale, W, false, &ax_v);
    double S[3] = {0, 0, 0};
    for (int j = ax_v.lo; j <= ax_v.hi; ++j) {
      for (int i = ax_u.lo; i <= ax_u.hi; ++i) S[ClassOf(f, j, i)] += ax_u.a[i - ax_u.lo] * ax_v.a[j - ax_v.lo];
    }
    for (int a = 0; a < 3; ++a) {
      atb[a] += S[a] * f.obs[idx];
      for (int b = 0; b < 3; ++b) ata[a][b] += S[a] * S[b];
    }
  }
  for (int a = 0; a < 3; ++a) ata[a][a] += 1e-6 * (ata[0][0] + ata[1][1] + ata[2][2]) + 1e-9;
  double m[9] = {ata[0][0], ata[0][1], ata[0][2], ata[1][0], ata[1][1], ata[1][2], ata[2][0], ata[2][1], ata[2][2]};
  double inv[9];
  if (!Invert3(m, inv)) return false;
  for (int a = 0; a < 3; ++a) p->lev[a] = inv[3 * a] * atb[0] + inv[3 * a + 1] * atb[1] + inv[3 * a + 2] * atb[2];
  return true;
}

double MedianAbsResidual(const Fit &f, const Params &p) {
  // A cheap robust scale: the median absolute residual of the current model.
  thread_local std::vector<float> res;
  res.clear();
  Axis ax_u, ax_v;
  const int W = f.width;
  for (size_t idx = 0; idx < f.obs.size(); ++idx) {
    double u, v, w;
    Project(p.map, f.nx[idx], f.ny[idx], &u, &v, &w);
    const double iw = 1.0 / w;
    const double gux = (p.map.g[0] - u * p.map.g[6]) * iw, guy = (p.map.g[1] - u * p.map.g[7]) * iw;
    const double gvx = (p.map.g[3] - v * p.map.g[6]) * iw, gvy = (p.map.g[4] - v * p.map.g[7]) * iw;
    ComputeAxis(u, p.su * std::sqrt(gux * gux + guy * guy) / f.scale, W, false, &ax_u);
    ComputeAxis(v, p.sv * std::sqrt(gvx * gvx + gvy * gvy) / f.scale, W, false, &ax_v);
    double S[3] = {0, 0, 0};
    for (int j = ax_v.lo; j <= ax_v.hi; ++j) {
      for (int i = ax_u.lo; i <= ax_u.hi; ++i) S[ClassOf(f, j, i)] += ax_u.a[i - ax_u.lo] * ax_v.a[j - ax_v.lo];
    }
    res.push_back(static_cast<float>(std::fabs(f.obs[idx] - (p.lev[0] * S[0] + p.lev[1] * S[1] + p.lev[2] * S[2]))));
  }
  if (res.empty()) return 0;
  std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
  return res[res.size() / 2];
}

}  // namespace

bool PolishDetection(const image_u8_t *im, apriltag_detection_t *det) {
  const apriltag_family_t *fam = det->family;
  if (fam == nullptr || fam->reversed_border) return false;
  const int W = static_cast<int>(fam->width_at_border);
  if (W < 4 || W > kMaxCells || det->id < 0 || det->id >= static_cast<int>(fam->ncodes)) return false;

  static thread_local Fit fit;
  fit.width = W;
  for (int j = 0; j < W; ++j) {
    for (int i = 0; i < W; ++i) fit.cls[j][i] = 0;
  }
  const uint64_t code = fam->codes[det->id];
  for (uint32_t b = 0; b < fam->nbits; ++b) {
    const uint32_t bx = fam->bit_x[b], by = fam->bit_y[b];
    if (bx >= static_cast<uint32_t>(W) || by >= static_cast<uint32_t>(W)) return false;
    if ((code >> (fam->nbits - 1 - b)) & 1u) fit.cls[by][bx] = 1;
  }

  // The detection's tag-to-pixel homography, composed with tag [-1, 1] -> cell coordinates.
  const matd_t *H = det->H;
  if (H == nullptr) return false;
  const double hw = 0.5 * W;
  double m[9];  // cell -> pixel
  for (int r = 0; r < 3; ++r) {
    const double h0 = MATD_EL(H, r, 0), h1 = MATD_EL(H, r, 1), h2 = MATD_EL(H, r, 2);
    m[3 * r + 0] = h0 / hw;
    m[3 * r + 1] = h1 / hw;
    m[3 * r + 2] = h2 - h0 - h1;
  }
  double cxy[2] = {0, 0}, side = 0;
  for (int k = 0; k < 4; ++k) {
    cxy[0] += 0.25 * det->p[k][0];
    cxy[1] += 0.25 * det->p[k][1];
    side += 0.25 * std::hypot(det->p[(k + 1) & 3][0] - det->p[k][0], det->p[(k + 1) & 3][1] - det->p[k][1]);
  }
  const double cell_px = side / W;
  if (cell_px < 1.2 || !std::isfinite(side)) return false;
  fit.cx = cxy[0];
  fit.cy = cxy[1];
  fit.scale = 0.5 * side;

  // pixel -> cell, in normalised pixel coordinates: inverse(m) * [scale 0 cx; 0 scale cy; 0 0 1].
  double minv[9];
  if (!Invert3(m, minv)) return false;
  double gn[9];
  const double tmat[9] = {fit.scale, 0, fit.cx, 0, fit.scale, fit.cy, 0, 0, 1};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) gn[3 * r + c] = minv[3 * r] * tmat[c] + minv[3 * r + 1] * tmat[3 + c] + minv[3 * r + 2] * tmat[6 + c];
  }
  if (std::fabs(gn[8]) < 1e-12) return false;
  Params p;
  for (int k = 0; k < 8; ++k) p.map.g[k] = gn[k] / gn[8];
  const Mapping start = p.map;
  double start_corners[4][2];
  CellCorners(start, W, start_corners);

  // Blur: try a few widths on a narrow working set first.
  const double candidates[] = {0.6, 1.0, 1.6, 2.5, 4.0};
  BuildSet(&fit, p.map, std::min(3.0, 0.5 * cell_px + 0.5), im->width, im->height, im);
  if (fit.obs.size() < 40) return false;
  double best_cost = 1e300, best_s = 1.0;
  Accum acc;
  p.lev[0] = 20;
  p.lev[1] = 235;
  p.lev[2] = 235;
  for (double s : candidates) {
    p.su = p.sv = s;
    if (!FitLevels(fit, &p)) continue;
    Evaluate(fit, p, 1e9, false, &acc);
    if (acc.n > 0 && acc.cost < best_cost) {
      best_cost = acc.cost;
      best_s = s;
    }
  }
  p.su = p.sv = best_s;
  if (!FitLevels(fit, &p)) return false;
  double window = std::min(kReach * best_s, 0.75 * cell_px + 0.5);
  window = std::max(window, 2.0);
  BuildSet(&fit, p.map, window, im->width, im->height, im);
  if (fit.obs.size() < 40) return false;
  FitLevels(fit, &p);

  const double kcut = std::max(6.0, 4.0 * 1.4826 * MedianAbsResidual(fit, p));
  Evaluate(fit, p, kcut, false, &acc);
  const double start_cost = acc.cost;
  double lambda = 1e-3;
  double cost = start_cost;
  double prev_corners[4][2];
  CellCorners(p.map, W, prev_corners);
  for (int iter = 0; iter < 12; ++iter) {
    Evaluate(fit, p, kcut, true, &acc);
    cost = acc.cost;
    double diag[kNP];
    for (int k = 0; k < kNP; ++k) diag[k] = std::max(acc.jtj[k][k], 1e-12);
    bool accepted = false;
    for (int attempt = 0; attempt < 8 && !accepted; ++attempt) {
      double a[kNP][kNP], b[kNP], delta[kNP];
      for (int r = 0; r < kNP; ++r) {
        for (int c = 0; c < kNP; ++c) a[r][c] = acc.jtj[r][c];
        a[r][r] += lambda * diag[r];
        b[r] = acc.jtr[r];
      }
      if (!Solve(a, b, delta)) {
        lambda *= 10;
        continue;
      }
      Params q = p;
      for (int k = 0; k < 8; ++k) q.map.g[k] += delta[k];
      q.su = std::clamp(q.su + delta[8], 0.35, 8.0);
      q.sv = std::clamp(q.sv + delta[9], 0.35, 8.0);
      for (int k = 0; k < 3; ++k) q.lev[k] += delta[10 + k];
      Accum trial;
      Evaluate(fit, q, kcut, false, &trial);
      if (trial.n > 0 && trial.cost < cost) {
        p = q;
        cost = trial.cost;
        lambda = std::max(lambda / 4, 1e-7);
        accepted = true;
      } else {
        lambda *= 5;
      }
    }
    if (!accepted) break;
    double corners[4][2];
    CellCorners(p.map, W, corners);
    double move = 0;
    for (int k = 0; k < 4; ++k) {
      move = std::max(move, std::hypot((corners[k][0] - prev_corners[k][0]) * fit.scale, (corners[k][1] - prev_corners[k][1]) * fit.scale));
      prev_corners[k][0] = corners[k][0];
      prev_corners[k][1] = corners[k][1];
    }
    if (move < 0.002) break;
  }

  if (!(cost < start_cost)) return false;
  double corners[4][2];
  CellCorners(p.map, W, corners);
  double limit = std::max(0.5, std::min(2.0, 0.3 * cell_px));
  for (int k = 0; k < 4; ++k) {
    if (!std::isfinite(corners[k][0]) || !std::isfinite(corners[k][1])) return false;
    if (std::hypot((corners[k][0] - start_corners[k][0]) * fit.scale, (corners[k][1] - start_corners[k][1]) * fit.scale) > limit) return false;
  }

  // New tag -> pixel homography: cell -> normalised pixel -> pixel, composed with tag -> cell.
  double cn[9];
  if (!CellToNormalised(p.map, cn)) return false;
  double mc[9];  // cell -> pixel
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const double row[3] = {r == 0 ? fit.scale : 0.0, r == 1 ? fit.scale : 0.0, r == 0 ? fit.cx : r == 1 ? fit.cy : 1.0};
      mc[3 * r + c] = row[0] * cn[c] + row[1] * cn[3 + c] + row[2] * cn[6 + c];
    }
  }
  matd_t *newH = det->H;
  for (int r = 0; r < 3; ++r) {
    const double c0 = mc[3 * r], c1 = mc[3 * r + 1], c2 = mc[3 * r + 2];
    MATD_EL(newH, r, 0) = c0 * hw;
    MATD_EL(newH, r, 1) = c1 * hw;
    MATD_EL(newH, r, 2) = c2 + c0 * hw + c1 * hw;
  }
  auto project = [&](double x, double y, double *ox, double *oy) {
    const double w = MATD_EL(newH, 2, 0) * x + MATD_EL(newH, 2, 1) * y + MATD_EL(newH, 2, 2);
    *ox = (MATD_EL(newH, 0, 0) * x + MATD_EL(newH, 0, 1) * y + MATD_EL(newH, 0, 2)) / w;
    *oy = (MATD_EL(newH, 1, 0) * x + MATD_EL(newH, 1, 1) * y + MATD_EL(newH, 1, 2)) / w;
  };
  project(0, 0, &det->c[0], &det->c[1]);
  for (int k = 0; k < 4; ++k) project((k == 1 || k == 2) ? 1 : -1, (k < 2) ? 1 : -1, &det->p[k][0], &det->p[k][1]);
  return true;
}

}  // namespace apriltag_vulkan
