#include "vkapriltag/PolishFit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace apriltag_vulkan {
namespace {

constexpr int kMaxCells = 16;   // widest supported grid
constexpr int kMaxCols = 24;    // columns (or rows) one pixel's blur window may touch
constexpr int kNP = 16;         // homography, three blur/motion terms, three levels, two illumination slopes
constexpr double kReach = 3.5;  // blur window, in standard deviations
constexpr int kMaxTaps = 13;    // samples along the motion blur segment

unsigned DevFlags() {
  static const unsigned v = [] {
    const char *e = std::getenv("APRILTAG_VK_POLISH_DEV");
    return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 0u;
  }();
  return v;
}

int MaxPixels() {
  static const int v = [] {
    const char *e = std::getenv("APRILTAG_VK_POLISH_N");
    return e ? std::atoi(e) : 8000;
  }();
  return v;
}

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
  uint8_t cls[kMaxCells][kMaxCells];  // [row][column]: 0 black, 1 white, 3 unknown
  double cx, cy, scale;               // pixel -> normalised coordinates
  std::vector<float> nx, ny, obs;     // working set: normalised position and observed value
  int thin = 1;                       // lattice step the working set was thinned by
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
  bool motion = false;
  // Without motion: blur standard deviations along the cell axes. With it: su = sv is the blur
  // standard deviation and (mx, my) the straight motion blur segment, all in pixels.
  double su = 1, sv = 1, mx = 0, my = 0;
  double lev[3] = {20, 235, 235};  // black, white, outside
  double lx = 0, ly = 0;           // relative illumination slope per normalised pixel
};

// The model at one pixel: the cell classes' blurred coverage and, with `normal`, the derivatives of
// the predicted value with respect to the parameters (columns as in the normal equations).
struct PixelEval {
  double S[4];
  double dbase[kNP];  // derivative of the prediction; the three level columns are set by the caller
};

// Level weights of the four classes: unknown cells read as the mid-level.
inline void ClassLevels(const Params &p, double l[4]) {
  l[0] = p.lev[0];
  l[1] = p.lev[1];
  l[2] = p.lev[2];
  l[3] = 0.5 * (p.lev[0] + p.lev[1]);
}

void ModelPixel(const Fit &f, const Params &p, const double lv[4], double x, double y, bool normal, PixelEval *e) {
  int taps = 1;
  double length = 0;
  if (p.motion) {
    length = std::hypot(p.mx, p.my);
    if (length > 0.05) taps = std::clamp(static_cast<int>(length) + 2, 3, kMaxTaps);
  }
  for (int c = 0; c < 4; ++c) e->S[c] = 0;
  if (normal) std::fill(std::begin(e->dbase), std::end(e->dbase), 0.0);
  const double inv_taps = 1.0 / taps;
  Axis ax_u, ax_v;
  for (int t = 0; t < taps; ++t) {
    const double frac = taps == 1 ? 0.0 : (t + 0.5) / taps - 0.5;
    const double xt = x + frac * p.mx / f.scale, yt = y + frac * p.my / f.scale;
    double u, v, w;
    Project(p.map, xt, yt, &u, &v, &w);
    const double iw = 1.0 / w;
    const double gux = (p.map.g[0] - u * p.map.g[6]) * iw, guy = (p.map.g[1] - u * p.map.g[7]) * iw;
    const double gvx = (p.map.g[3] - v * p.map.g[6]) * iw, gvy = (p.map.g[4] - v * p.map.g[7]) * iw;
    const double gu = std::sqrt(gux * gux + guy * guy) / f.scale;  // cells per pixel
    const double gv = std::sqrt(gvx * gvx + gvy * gvy) / f.scale;
    ComputeAxis(u, p.su * gu, f.width, normal, &ax_u);
    ComputeAxis(v, p.sv * gv, f.width, normal, &ax_v);
    double S[4] = {0, 0, 0, 0}, Su[4] = {0, 0, 0, 0}, Sv[4] = {0, 0, 0, 0}, Ssu[4] = {0, 0, 0, 0}, Ssv[4] = {0, 0, 0, 0};
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
    for (int c = 0; c < 4; ++c) e->S[c] += S[c] * inv_taps;
    if (!normal) continue;
    double pu = 0, pv = 0, psu = 0, psv = 0;
    for (int c = 0; c < 4; ++c) {
      pu += lv[c] * Su[c];
      pv += lv[c] * Sv[c];
      psu += lv[c] * Ssu[c];
      psv += lv[c] * Ssv[c];
    }
    pu *= inv_taps;
    pv *= inv_taps;
    const double dg_u[8] = {xt * iw, yt * iw, iw, 0, 0, 0, -u * xt * iw, -u * yt * iw};
    const double dg_v[8] = {0, 0, 0, xt * iw, yt * iw, iw, -v * xt * iw, -v * yt * iw};
    for (int k = 0; k < 8; ++k) e->dbase[k] += pu * dg_u[k] + pv * dg_v[k];
    if (p.motion) {
      e->dbase[8] += (psu * gu + psv * gv) * inv_taps;
      // a shift of the sample position by frac * (mx, my) / scale
      const double along_x = pu * gux + pv * gvx, along_y = pu * guy + pv * gvy;
      e->dbase[9] += frac * along_x / f.scale;
      e->dbase[13] += frac * along_y / f.scale;
    } else {
      e->dbase[8] += psu * gu * inv_taps;
      e->dbase[9] += psv * gv * inv_taps;
    }
  }
}

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
  double lv[4];
  ClassLevels(p, lv);
  const size_t count = f.obs.size();
  PixelEval e;
  for (size_t idx = 0; idx < count; ++idx) {
    ModelPixel(f, p, lv, f.nx[idx], f.ny[idx], normal, &e);
    const double base = lv[0] * e.S[0] + lv[1] * e.S[1] + lv[2] * e.S[2] + lv[3] * e.S[3];
    const double gain = 1.0 + p.lx * f.nx[idx] + p.ly * f.ny[idx];
    const double pred = gain * base;
    const double ob = f.obs[idx];
    // Saturated pixels the model also puts at or beyond the range carry no information.
    if ((ob >= 255.0 && pred >= 255.0) || (ob <= 0.0 && pred <= 0.0)) continue;
    const double r = ob - pred;
    const double ar = std::fabs(r);
    const double wt = ar <= kcut ? 1.0 : kcut / ar;
    acc->cost += ar <= kcut ? r * r : kcut * (2.0 * ar - kcut);
    ++acc->n;
    if (!normal) continue;
    e.dbase[10] = e.S[0] + 0.5 * e.S[3];
    e.dbase[11] = e.S[1] + 0.5 * e.S[3];
    e.dbase[12] = e.S[2];
    for (int a = 0; a < 14; ++a) e.dbase[a] *= gain;
    e.dbase[14] = base * f.nx[idx];
    e.dbase[15] = base * f.ny[idx];
    for (int a = 0; a < kNP; ++a) {
      const double ja = wt * e.dbase[a];
      acc->jtr[a] += ja * r;
      for (int c = a; c < kNP; ++c) acc->jtj[a][c] += ja * e.dbase[c];
    }
  }
  if (normal) {
    for (int a = 0; a < kNP; ++a) {
      for (int c = 0; c < a; ++c) acc->jtj[a][c] = acc->jtj[c][a];
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

// Builds the working set: the pixels whose blur window touches cells of more than one class, on a
// lattice thinned so that about `limit` of them remain.
void BuildSet(Fit *f, const Mapping &map, double half_window_px, int limit, const image_u8_t *im) {
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
  const int x0 = std::max(0, static_cast<int>(std::floor(minx))), x1 = std::min(im->width - 1, static_cast<int>(std::ceil(maxx)));
  const int y0 = std::max(0, static_cast<int>(std::floor(miny))), y1 = std::min(im->height - 1, static_cast<int>(std::ceil(maxy)));
  f->nx.clear();
  f->ny.clear();
  f->obs.clear();
  // About half the tag's area lies within reach of an edge.
  const double area = static_cast<double>(x1 - x0 + 1) * (y1 - y0 + 1);
  const int k = std::max(1, static_cast<int>(0.45 * area / std::max(limit, 1)));
  const int r = std::max(1, static_cast<int>(std::lround(k * 0.382)));
  f->thin = k;
  for (int py = y0; py <= y1; ++py) {
    const int off = ((-(py * r)) % k + k) % k;
    int px = x0 + (((off - x0) % k) + k) % k;
    for (; px <= x1; px += k) {
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
      bool varies = false, unknown = false;
      for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
          const int c = ClassOf(*f, j, i);
          varies |= c != first;
          unknown |= c == 3;
        }
      }
      if (!varies || unknown) continue;
      f->nx.push_back(static_cast<float>(x));
      f->ny.push_back(static_cast<float>(y));
      f->obs.push_back(static_cast<float>(im->buf[py * im->stride + px]));
    }
  }
}

// Linear least squares for the three levels at fixed mapping and blur.
bool FitLevels(const Fit &f, Params *p, double *sse = nullptr) {
  double ata[3][3] = {}, atb[3] = {}, yy = 0;
  const double unit[4] = {1, 1, 1, 1};
  PixelEval e;
  for (size_t idx = 0; idx < f.obs.size(); ++idx) {
    ModelPixel(f, *p, unit, f.nx[idx], f.ny[idx], false, &e);
    // The unknown class reads as the mid-level of black and white.
    const double gain = 1.0 + p->lx * f.nx[idx] + p->ly * f.ny[idx];
    const double sw[3] = {gain * (e.S[0] + 0.5 * e.S[3]), gain * (e.S[1] + 0.5 * e.S[3]), gain * e.S[2]};
    for (int a = 0; a < 3; ++a) {
      atb[a] += sw[a] * f.obs[idx];
      yy += a == 0 ? static_cast<double>(f.obs[idx]) * f.obs[idx] : 0.0;
      for (int c = 0; c < 3; ++c) ata[a][c] += sw[a] * sw[c];
    }
  }
  for (int a = 0; a < 3; ++a) ata[a][a] += 1e-6 * (ata[0][0] + ata[1][1] + ata[2][2]) + 1e-9;
  double m[9] = {ata[0][0], ata[0][1], ata[0][2], ata[1][0], ata[1][1], ata[1][2], ata[2][0], ata[2][1], ata[2][2]};
  double inv[9];
  if (!Invert3(m, inv)) return false;
  for (int a = 0; a < 3; ++a) p->lev[a] = inv[3 * a] * atb[0] + inv[3 * a + 1] * atb[1] + inv[3 * a + 2] * atb[2];
  if (sse != nullptr) *sse = yy - (p->lev[0] * atb[0] + p->lev[1] * atb[1] + p->lev[2] * atb[2]);
  return true;
}

// A cheap robust scale: the median absolute residual of the current model.
double MedianAbsResidual(const Fit &f, const Params &p) {
  thread_local std::vector<float> res;
  res.clear();
  double lv[4];
  ClassLevels(p, lv);
  PixelEval e;
  for (size_t idx = 0; idx < f.obs.size(); ++idx) {
    ModelPixel(f, p, lv, f.nx[idx], f.ny[idx], false, &e);
    const double gain = 1.0 + p.lx * f.nx[idx] + p.ly * f.ny[idx];
    res.push_back(static_cast<float>(std::fabs(f.obs[idx] - gain * (lv[0] * e.S[0] + lv[1] * e.S[1] + lv[2] * e.S[2] + lv[3] * e.S[3]))));
  }
  if (res.empty()) return 0;
  std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
  return res[res.size() / 2];
}

// Levenberg-Marquardt on the normal equations; returns the final cost and leaves the best
// parameters in *p. Stops when the corners move less than `tol` pixels.
double Optimise(const Fit &f, Params *p, double kcut, int max_iter, double tol, int width, double *initial_cost = nullptr) {
  Accum cur, trial;
  Evaluate(f, *p, kcut, true, &cur);
  if (initial_cost != nullptr) *initial_cost = cur.cost;
  double lambda = 1e-3;
  double prev[4][2];
  CellCorners(p->map, width, prev);
  for (int iter = 0; iter < max_iter; ++iter) {
    double diag[kNP];
    for (int k = 0; k < kNP; ++k) diag[k] = std::max(cur.jtj[k][k], 1e-9);
    bool accepted = false;
    for (int attempt = 0; attempt < 6 && !accepted; ++attempt) {
      double a[kNP][kNP], b[kNP], delta[kNP];
      for (int r = 0; r < kNP; ++r) {
        for (int c = 0; c < kNP; ++c) a[r][c] = cur.jtj[r][c];
        a[r][r] += lambda * diag[r];
        b[r] = cur.jtr[r];
      }
      if (!Solve(a, b, delta)) {
        lambda *= 10;
        continue;
      }
      Params q = *p;
      for (int k = 0; k < 8; ++k) q.map.g[k] += delta[k];
      if (q.motion) {
        q.su = q.sv = std::clamp(q.su + delta[8], 0.35, 6.0);
        q.mx += delta[9];
        q.my += delta[13];
        const double len = std::hypot(q.mx, q.my);
        if (len > 24.0) {
          q.mx *= 24.0 / len;
          q.my *= 24.0 / len;
        }
      } else {
        q.su = std::clamp(q.su + delta[8], 0.35, 8.0);
        q.sv = std::clamp(q.sv + delta[9], 0.35, 8.0);
      }
      for (int k = 0; k < 3; ++k) q.lev[k] += delta[10 + k];
      q.lx += delta[14];
      q.ly += delta[15];
      Evaluate(f, q, kcut, true, &trial);
      if (trial.n > 0 && trial.cost < cur.cost) {
        *p = q;
        std::swap(cur, trial);
        lambda = std::max(lambda / 4, 1e-7);
        accepted = true;
      } else {
        lambda *= 5;
      }
    }
    if (!accepted) break;
    double corners[4][2];
    CellCorners(p->map, width, corners);
    double move = 0;
    for (int k = 0; k < 4; ++k) {
      move = std::max(move, std::hypot((corners[k][0] - prev[k][0]) * f.scale, (corners[k][1] - prev[k][1]) * f.scale));
      prev[k][0] = corners[k][0];
      prev[k][1] = corners[k][1];
    }
    if (move < tol) break;
  }
  return cur.cost;
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

  const int limit = MaxPixels();
  // Blur: try a few widths on a narrow working set first.
  const double candidates[] = {0.7, 1.2, 2.0, 3.5};
  BuildSet(&fit, p.map, std::min(3.0, 0.5 * cell_px + 0.5), limit, im);
  if (fit.obs.size() < 40) return false;
  double best_cost = 1e300, best_s = 1.0;
  Accum acc;
  for (double s : candidates) {
    p.su = p.sv = s;
    double sse = 0;
    if (!FitLevels(fit, &p, &sse)) continue;
    if (sse < best_cost) {
      best_cost = sse;
      best_s = s;
    }
  }
  p.su = p.sv = best_s;
  if (!FitLevels(fit, &p)) return false;
  if (fit.thin > 1 && (DevFlags() & 16) == 0) {
    // A tag too large for the working set keeps its profile fit unless it is blurred or clipped.
    size_t saturated = 0;
    for (float v : fit.obs) saturated += v >= 255.0f;
    if (best_s < 1.6 && saturated * 20 < fit.obs.size()) return false;
  }
  if (det->hamming > 0) {
    // A cell whose centre reads far from the level its code bit implies (covered, or a corrected bit) is left out.
    const double range = std::fabs(p.lev[1] - p.lev[0]);
    double inv_map[9];
    if (range > 8.0 && CellToNormalised(p.map, inv_map)) {
      uint8_t marked[kMaxCells][kMaxCells] = {};
      for (int j = 0; j < W; ++j) {
        for (int i = 0; i < W; ++i) {
          const double cu = i + 0.5, cv = j + 0.5;
          const double w = inv_map[6] * cu + inv_map[7] * cv + inv_map[8];
          const double px = (inv_map[0] * cu + inv_map[1] * cv + inv_map[2]) / w * fit.scale + fit.cx;
          const double py = (inv_map[3] * cu + inv_map[4] * cv + inv_map[5]) / w * fit.scale + fit.cy;
          const int ix = static_cast<int>(px), iy = static_cast<int>(py);
          if (ix < 0 || iy < 0 || ix >= im->width || iy >= im->height) continue;
          const double expected = fit.cls[j][i] ? p.lev[1] : p.lev[0];
          if (std::fabs(im->buf[iy * im->stride + ix] - expected) > 0.3 * range) marked[j][i] = 1;
        }
      }
      for (int j = 0; j < W; ++j) {
        for (int i = 0; i < W; ++i) {
          if (marked[j][i]) fit.cls[j][i] = 3;
        }
      }
    }
  }
  double window = std::min(2.5 * best_s + 0.3, 0.75 * cell_px + 0.5);
  window = std::max(window, 1.8);
  BuildSet(&fit, p.map, window, limit, im);
  if (fit.obs.size() < 40) return false;
  FitLevels(fit, &p);

  const double kcut = std::max(6.0, 4.0 * 1.4826 * MedianAbsResidual(fit, p));
  double start_cost = 0;
  double cost = Optimise(fit, &p, kcut, 8, 0.01, W, &start_cost);

  bool moving_accepted = false;
  if (std::max(p.su, p.sv) > 1.4 && (DevFlags() & 8) == 0) {
    // A wide blur may be motion: refit with a straight blur segment, from the best of eight directions,
    // on a thinner working set (the model costs a sample per tap).
    static thread_local Fit small;
    small.width = W;
    std::copy(&fit.cls[0][0], &fit.cls[0][0] + kMaxCells * kMaxCells, &small.cls[0][0]);
    small.cx = fit.cx;
    small.cy = fit.cy;
    small.scale = fit.scale;
    BuildSet(&small, p.map, window, std::min(limit, 2500), im);
    if (small.obs.size() >= 40) {
      const double sigma0 = 0.7;
      const double smax = std::max(p.su, p.sv);
      const double length0 = std::sqrt(12.0 * std::max(smax * smax - sigma0 * sigma0, 1.0));
      Evaluate(small, p, kcut, false, &acc);
      const double cost_plain = acc.cost;
      Params moving = p;
      moving.motion = true;
      moving.su = moving.sv = sigma0;
      double best_move = 1e300;
      Params best = moving;
      for (int d = 0; d < 8; ++d) {
        const double ang = d * 3.14159265358979323846 / 8.0;
        moving.mx = length0 * std::cos(ang);
        moving.my = length0 * std::sin(ang);
        double sse = 0;
        if (!FitLevels(small, &moving, &sse)) continue;
        if (sse < best_move) {
          best_move = sse;
          best = moving;
        }
      }
      if (best_move < 1e300) {
        const double cost_moving = Optimise(small, &best, kcut, 10, 0.01, W);
        if (cost_moving < 0.97 * cost_plain) {
          p = best;
          moving_accepted = true;
          if ((DevFlags() & 32) == 0) Optimise(fit, &p, kcut, 3, 0.01, W);
        }
      }
    }
  }

  if (!moving_accepted && !(cost < start_cost)) return false;
  double corners[4][2];
  CellCorners(p.map, W, corners);
  const double limit_px = std::max(0.5, std::min(2.0, 0.3 * cell_px));
  for (int k = 0; k < 4; ++k) {
    if (!std::isfinite(corners[k][0]) || !std::isfinite(corners[k][1])) return false;
    if (std::hypot((corners[k][0] - start_corners[k][0]) * fit.scale, (corners[k][1] - start_corners[k][1]) * fit.scale) > limit_px) return false;
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
