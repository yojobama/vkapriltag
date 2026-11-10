#include "vkapriltag/PolishFit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <chrono>
#include <cstdio>
#include <vector>

namespace apriltag_vulkan {
namespace {

struct ProfCounters {
  double t[12] = {};
  long n[12] = {};
  long tags = 0, pixels = 0, evals = 0;
  long hist[10][10] = {};
  ~ProfCounters() {
    if (tags == 0 || std::getenv("APRILTAG_VK_POLISH_PROF") == nullptr) return;
    static const char *names[12] = {"setup", "cand", "scan", "select", "freeze+lev", "opt-model", "opt-accum", "motion", "final", "total", "gather", "dots"};
    for (int i = 0; i < 12; ++i) std::fprintf(stderr, "PROF %-10s %8.1f us/tag\n", names[i], t[i] * 1e6 / tags);
    for (int a = 1; a < 10; ++a) {
      for (int b = 1; b < 10; ++b) {
        if (hist[a][b] > 0) std::fprintf(stderr, "HIST nu=%d nv=%d %.1f%%\n", a, b, 100.0 * hist[a][b] / (pixels > 0 ? pixels : 1));
      }
    }
    std::fprintf(stderr, "PROF box area/tag %.0f\n", static_cast<double>(n[0]) / tags);
    std::fprintf(stderr, "PROF tags %ld px/tag %.0f evals/tag %.2f\n", tags, static_cast<double>(pixels) / tags, static_cast<double>(evals) / tags);
  }
};
thread_local ProfCounters g_prof_local;
inline double NowS() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

constexpr int kMaxCells = 16;        // widest supported grid
constexpr int kPad = kMaxCells + 2;  // class grid with one outside row and column on each side
constexpr int kMaxC = 8;             // columns (or rows) one pixel's frozen window may hold
constexpr int kMaxJ = 11;            // homography plus up to three blur parameters
constexpr int kStrideJ = 12;
constexpr int kMaxT = 16;           // all parameters: geometry, blur, three levels, two slopes


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

struct Knobs {
  double blur_extra, win_mult, min_win, smax_mult, tol, iters, reach;
  Knobs() {
    blur_extra = Get("APRILTAG_VK_POLISH_BLUREXTRA", 2.0);
    win_mult = Get("APRILTAG_VK_POLISH_WINMULT", 0.8);
    min_win = Get("APRILTAG_VK_POLISH_MINWIN", 0.5);
    smax_mult = Get("APRILTAG_VK_POLISH_SMAXM", 1.3);
    tol = Get("APRILTAG_VK_POLISH_TOL", 0.004);
    iters = Get("APRILTAG_VK_POLISH_ITERS", 6);
    reach = Get("APRILTAG_VK_POLISH_REACH", 1.6);
  }
  static double Get(const char *name, double fallback) {
    const char *e = std::getenv(name);
    return e ? std::atof(e) : fallback;
  }
};
const Knobs kKnobs;

// Standard normal CDF, density and CDF integral (z Phi(z) + phi(z)), tabulated over [-4, 4].
struct NormalTable {
  static constexpr int kSteps = 32;
  static constexpr int kN = 8 * kSteps + 2;
  float cdf[kN], pdf[kN], gint[kN];
  NormalTable() {
    for (int i = 0; i < kN; ++i) {
      const double z = -4.0 + static_cast<double>(i) / kSteps;
      const double c = 0.5 * std::erfc(-z / std::sqrt(2.0));
      const double p = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
      cdf[i] = static_cast<float>(c);
      pdf[i] = static_cast<float>(p);
      gint[i] = static_cast<float>(z * c + p);
    }
  }
};
const NormalTable kNormal;

inline void Lut(float z, float *cdf, float *pdf) {
  float t = (z + 4.0f) * NormalTable::kSteps;
  t = std::min(std::max(t, 0.0f), 8.0f * NormalTable::kSteps - 0.001f);
  const int i = static_cast<int>(t);
  const float f = t - static_cast<float>(i);
  *cdf = kNormal.cdf[i] + f * (kNormal.cdf[i + 1] - kNormal.cdf[i]);
  *pdf = kNormal.pdf[i] + f * (kNormal.pdf[i + 1] - kNormal.pdf[i]);
}

inline void LutG(float z, float *cdf, float *gint) {
  if (z >= 4.0f) {
    *cdf = 1.0f;
    *gint = z;
    return;
  }
  float t = (z + 4.0f) * NormalTable::kSteps;
  t = std::max(t, 0.0f);
  const int i = static_cast<int>(t);
  const float f = t - static_cast<float>(i);
  *cdf = kNormal.cdf[i] + f * (kNormal.cdf[i + 1] - kNormal.cdf[i]);
  *gint = kNormal.gint[i] + f * (kNormal.gint[i + 1] - kNormal.gint[i]);
}

// Blurred cell-column coverage along one axis at position u: a[t] for column lo + t, and its
// derivatives with respect to u, the blur width s and (box model) the box length l.
struct Axis {
  float a[kMaxC], du[kMaxC], ds[kMaxC], dl[kMaxC];
};

template <bool Box, bool Deriv>
inline void AxisEval(float u, float s, float l, int lo, int n, int width, Axis *ax) {
  if (n == 1) {
    ax->a[0] = 1.0f;
    if (Deriv) ax->du[0] = ax->ds[0] = ax->dl[0] = 0.0f;
    return;
  }
  float c[kMaxC + 1], dt[kMaxC + 1], dsb[kMaxC + 1], dlb[kMaxC + 1];
  const float inv_s = 1.0f / s;
  const float inv_l = Box ? 1.0f / l : 0.0f;
  c[0] = 0.0f;
  dt[0] = dsb[0] = dlb[0] = 0.0f;
  c[n] = 1.0f;
  dt[n] = dsb[n] = dlb[n] = 0.0f;
  for (int t = 1; t < n; ++t) {
    const int k = lo + t;
    if (k < 0) {
      c[t] = 0.0f;
      dt[t] = dsb[t] = dlb[t] = 0.0f;
    } else if (k > width) {
      c[t] = 1.0f;
      dt[t] = dsb[t] = dlb[t] = 0.0f;
    } else if (Box) {
      const float d = static_cast<float>(k) - u;
      const float zp = (d + 0.5f * l) * inv_s, zm = (d - 0.5f * l) * inv_s;
      float cp, gp, cm, gm;
      LutG(zp, &cp, &gp);
      LutG(zm, &cm, &gm);
      const float cc = s * inv_l * (gp - gm);
      c[t] = cc;
      if (Deriv) {
        dt[t] = (cp - cm) * inv_l;
        dsb[t] = cc * inv_s - (zp * cp - zm * cm) * inv_l;
        dlb[t] = (0.5f * (cp + cm) - cc) * inv_l;
      }
    } else {
      const float z = (static_cast<float>(k) - u) * inv_s;
      float cc, pp;
      Lut(z, &cc, &pp);
      c[t] = cc;
      if (Deriv) {
        dt[t] = pp * inv_s;
        dsb[t] = -pp * z * inv_s;
      }
    }
  }
  for (int t = 0; t < n; ++t) {
    ax->a[t] = c[t + 1] - c[t];
    if (Deriv) {
      ax->du[t] = dt[t] - dt[t + 1];
      ax->ds[t] = dsb[t + 1] - dsb[t];
      if (Box) ax->dl[t] = dlb[t + 1] - dlb[t];
    }
  }
}

inline int FloorI(float x) { return static_cast<int>(std::max(x, -60.0f) + 64.0f) - 64; }

struct Fit {
  int width = 0;
  uint8_t cls[kPad][kPad];  // [row + 1][column + 1]: 0 black, 1 white, 2 outside, 3 unknown
  double cx = 0, cy = 0, scale = 1;
  int n = 0;
  std::vector<float> x, y, obs, u0, v0, gu, gv;
  std::vector<int8_t> i0, j0;
  std::vector<uint8_t> nu, nv;
  std::vector<float> A, B, C;  // black (plus half the unknown), white (plus half the unknown), outside coverage
  std::vector<float> J;        // kStrideJ columns per pixel: derivative of the base prediction
  void Clear() {
    n = 0;
    x.clear();
    y.clear();
    obs.clear();
    u0.clear();
    v0.clear();
    gu.clear();
    gv.clear();
    i0.clear();
    j0.clear();
    nu.clear();
    nv.clear();
  }
};

inline int ClassOf(const Fit &f, int row, int col) { return f.cls[row + 1][col + 1]; }

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
  bool box = false;  // motion model: s[0] blur, (s[1], s[2]) blur segment in pixels
  int taps = 5;
  // Gaussian model: s[0], s[1] are the blur standard deviations along the cell axes. Box model:
  // s[0] is the blur standard deviation and s[1], s[2] the motion blur extent along each axis.
  double s[3] = {1, 1, 0};
  double lev[3] = {20, 235, 235};  // black, white, outside
  double lx = 0, ly = 0;           // relative illumination slope per normalised pixel
  double smax = 8;                 // upper bound of the Gaussian blur widths
};

// Local scale of the mapping at a normalised pixel: cells per pixel along u and v.
inline void LocalScale(const Mapping &m, double x, double y, double scale, double *u, double *v, double *gu, double *gv) {
  double w;
  Project(m, x, y, u, v, &w);
  const double iw = 1.0 / w;
  const double gux = (m.g[0] - *u * m.g[6]) * iw, guy = (m.g[1] - *u * m.g[7]) * iw;
  const double gvx = (m.g[3] - *v * m.g[6]) * iw, gvy = (m.g[4] - *v * m.g[7]) * iw;
  *gu = std::sqrt(gux * gux + guy * guy) / scale;
  *gv = std::sqrt(gvx * gvx + gvy * gvy) / scale;
}

// The bounding box of the tag in the image, its largest cell-per-pixel scale and the lattice step that
// keeps the working set under `limit` pixels.
struct BoxInfo {
  int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
  double gmax = 0;
  double area = 0;
  int thin = 1;
};

void ComputeBox(const Fit &f, const Mapping &map, int limit, const image_u8_t *im, BoxInfo *b) {
  const int W = f.width;
  double inv[9];
  CellToNormalised(map, inv);
  double minx = 1e30, miny = 1e30, maxx = -1e30, maxy = -1e30;
  const double ext = 0.8;
  const double cu[4] = {-ext, W + ext, W + ext, -ext}, cv[4] = {-ext, -ext, W + ext, W + ext};
  double gmax = 0;
  for (int k = 0; k < 4; ++k) {
    const double w = inv[6] * cu[k] + inv[7] * cv[k] + inv[8];
    const double nx = (inv[0] * cu[k] + inv[1] * cv[k] + inv[2]) / w;
    const double ny = (inv[3] * cu[k] + inv[4] * cv[k] + inv[5]) / w;
    const double x = nx * f.scale + f.cx;
    const double y = ny * f.scale + f.cy;
    minx = std::min(minx, x);
    maxx = std::max(maxx, x);
    miny = std::min(miny, y);
    maxy = std::max(maxy, y);
    double u, v, gu, gv;
    LocalScale(map, nx, ny, f.scale, &u, &v, &gu, &gv);
    gmax = std::max(gmax, std::max(gu, gv));
  }
  b->x0 = std::max(0, static_cast<int>(std::floor(minx)));
  b->x1 = std::min(im->width - 1, static_cast<int>(std::ceil(maxx)));
  b->y0 = std::max(0, static_cast<int>(std::floor(miny)));
  b->y1 = std::min(im->height - 1, static_cast<int>(std::ceil(maxy)));
  b->gmax = gmax;
  b->area = static_cast<double>(std::max(0, b->x1 - b->x0 + 1)) * std::max(0, b->y1 - b->y0 + 1);
  b->thin = std::max(1, static_cast<int>(0.45 * b->area / std::max(limit, 1)));
}

// A sparse sample of pixels beside the class-changing cell edges, for the blur scan.
void BuildScan(Fit *scan, const Fit &f, const Mapping &map, double cell_px, const image_u8_t *im) {
  const int W = f.width;
  scan->Clear();
  double inv[9];
  CellToNormalised(map, inv);
  static const double kOffset[4] = {-4.2, 1.4, -1.4, 4.2};
  int count = 0;
  const auto add = [&](double u, double v) {
    const double w = inv[6] * u + inv[7] * v + inv[8];
    const double X = (inv[0] * u + inv[1] * v + inv[2]) / w * f.scale + f.cx;
    const double Y = (inv[3] * u + inv[4] * v + inv[5]) / w * f.scale + f.cy;
    if (!(X >= 0 && Y >= 0 && X < im->width && Y < im->height)) return;
    const int px = static_cast<int>(X), py = static_cast<int>(Y);
    const double xn = (px + 0.5 - f.cx) / f.scale, yn = (py + 0.5 - f.cy) / f.scale;
    double u2, v2, gu, gv;
    LocalScale(map, xn, yn, f.scale, &u2, &v2, &gu, &gv);
    scan->x.push_back(static_cast<float>(xn));
    scan->y.push_back(static_cast<float>(yn));
    scan->obs.push_back(static_cast<float>(im->buf[static_cast<size_t>(py) * im->stride + px]));
    scan->u0.push_back(static_cast<float>(u2));
    scan->v0.push_back(static_cast<float>(v2));
    scan->gu.push_back(static_cast<float>(gu));
    scan->gv.push_back(static_cast<float>(gv));
  };
  const double inv_cell = 1.0 / cell_px;
  for (int axis = 0; axis < 2; ++axis) {
    for (int line = 0; line <= W; ++line) {
      for (int pos = 0; pos < W; ++pos) {
        // Class on the low and high side of the unit edge at `line` beside position pos.
        const int lo = axis == 0 ? f.cls[pos + 1][line] : f.cls[line][pos + 1];
        const int hi = axis == 0 ? f.cls[pos + 1][line + 1] : f.cls[line + 1][pos + 1];
        if (lo == hi || lo == 3 || hi == 3) continue;
        for (int along = 0; along < 2; ++along) {
          const double a = pos + (along == 0 ? 0.3 : 0.7);
          const double d = kOffset[count++ & 3] * inv_cell;
          if (axis == 0) add(line + d, a);
          else add(a, line + d);
        }
      }
    }
  }
  scan->n = static_cast<int>(scan->x.size());
  scan->width = W;
  std::copy(&f.cls[0][0], &f.cls[0][0] + kPad * kPad, &scan->cls[0][0]);
  scan->cx = f.cx;
  scan->cy = f.cy;
  scan->scale = f.scale;
}

// The working set directly: pixels whose window (half_window_px) touches more than one known cell class and no
// unknown cell, on the lattice of `box.thin`, with the cell window of each pixel frozen for reach_px.
void BuildFit(Fit *f, const Mapping &map, double half_window_px, double reach_px, const BoxInfo &box, const image_u8_t *im) {
  const int W = f->width;
  f->n = 0;
  if (box.x1 < box.x0 || box.y1 < box.y0) return;
  const float g0 = static_cast<float>(map.g[0]), g1 = static_cast<float>(map.g[1]), g2 = static_cast<float>(map.g[2]);
  const float g3 = static_cast<float>(map.g[3]), g4 = static_cast<float>(map.g[4]), g5 = static_cast<float>(map.g[5]);
  const float g6 = static_cast<float>(map.g[6]), g7 = static_cast<float>(map.g[7]);
  const float inv_scale = static_cast<float>(1.0 / f->scale);
  const float hw = static_cast<float>(half_window_px);
  const float reach = static_cast<float>(reach_px);
  const float rmax = hw * static_cast<float>(box.gmax) * 1.02f;
  const float ext = 0.8f, lim = static_cast<float>(W) + ext;
  const float cxf = static_cast<float>(f->cx), cyf = static_cast<float>(f->cy);
  const int k = box.thin;
  g_prof_local.n[0] += static_cast<long>(box.area);
  const int lr = std::max(1, static_cast<int>(std::lround(k * 0.382)));
  const int x0 = box.x0, x1 = box.x1;
  const int width = (x1 - x0) / k + 2;
  thread_local std::vector<float> row_u, row_v, row_gu;
  row_u.resize(width);
  row_v.resize(width);
  row_gu.resize(width);
  const size_t cap = static_cast<size_t>(box.y1 - box.y0 + 1) * static_cast<size_t>(width);
  const auto grow = [cap](auto &v) {
    if (v.size() < cap) v.resize(cap);
  };
  grow(f->x);
  grow(f->y);
  grow(f->obs);
  grow(f->u0);
  grow(f->v0);
  grow(f->gu);
  grow(f->gv);
  grow(f->i0);
  grow(f->j0);
  grow(f->nu);
  grow(f->nv);
  grow(f->A);
  grow(f->B);
  grow(f->C);
  size_t out = 0;
  for (int py = box.y0; py <= box.y1; ++py) {
    const uint8_t *row = im->buf + static_cast<size_t>(py) * im->stride;
    const float y = (static_cast<float>(py) + 0.5f - cyf) * inv_scale;
    const float a0 = g1 * y + g2, b0 = g4 * y + g5, c0 = g7 * y + 1.0f;
    const int off = ((-(py * lr)) % k + k) % k;
    const int xfirst = x0 + (((off - x0) % k) + k) % k;
    if (xfirst > x1) continue;
    const int count = (x1 - xfirst) / k + 1;
    const float xs = (static_cast<float>(xfirst) + 0.5f - cxf) * inv_scale;
    const float xstep = static_cast<float>(k) * inv_scale;
    for (int i = 0; i < count; ++i) {
      const float x = xs + static_cast<float>(i) * xstep;
      const float iw = 1.0f / (g6 * x + c0);
      const float u = (g0 * x + a0) * iw, v = (g3 * x + b0) * iw;
      row_u[i] = u;
      row_v[i] = v;
      row_gu[i] = iw;
    }
    for (int i = 0; i < count; ++i) {
      const float u = row_u[i], v = row_v[i];
      if (u < -ext || u > lim || v < -ext || v > lim) continue;
      if (FloorI(u - rmax) == FloorI(u + rmax) && FloorI(v - rmax) == FloorI(v + rmax)) continue;
      const float iw = row_gu[i];
      const float gux = (g0 - u * g6) * iw, guy = (g1 - u * g7) * iw;
      const float gvx = (g3 - v * g6) * iw, gvy = (g4 - v * g7) * iw;
      const float gu = std::sqrt(gux * gux + guy * guy) * inv_scale;
      const float gv = std::sqrt(gvx * gvx + gvy * gvy) * inv_scale;
      const float ru = hw * gu, rv = hw * gv;
      const int i0 = std::clamp(FloorI(u - ru), -1, W), i1 = std::clamp(FloorI(u + ru), -1, W);
      const int j0 = std::clamp(FloorI(v - rv), -1, W), j1 = std::clamp(FloorI(v + rv), -1, W);
      if (i0 == i1 && j0 == j1) continue;
      const int first = ClassOf(*f, j0, i0);
      bool varies = false, unknown = false;
      for (int j = j0; j <= j1; ++j) {
        for (int ii = i0; ii <= i1; ++ii) {
          const int cl = ClassOf(*f, j, ii);
          varies |= cl != first;
          unknown |= cl == 3;
        }
      }
      if (!varies || unknown) continue;
      f->x[out] = xs + static_cast<float>(i) * xstep;
      f->y[out] = y;
      f->obs[out] = static_cast<float>(row[xfirst + i * k]);
      f->u0[out] = u;
      f->v0[out] = v;
      f->gu[out] = gu;
      f->gv[out] = gv;
      for (int axis = 0; axis < 2; ++axis) {
        const float pos = axis == 0 ? u : v;
        const float rr = reach * (axis == 0 ? gu : gv);
        int lo = std::clamp(FloorI(pos - rr), -1, W);
        int hi = std::clamp(FloorI(pos + rr), -1, W);
        if (hi - lo + 1 > kMaxC) {
          const int mid = std::clamp(FloorI(pos), -1, W);
          lo = std::max(lo, mid - (kMaxC - 1) / 2);
          hi = std::min(hi, lo + kMaxC - 1);
        }
        if (axis == 0) {
          f->i0[out] = static_cast<int8_t>(lo);
          f->nu[out] = static_cast<uint8_t>(hi - lo + 1);
        } else {
          f->j0[out] = static_cast<int8_t>(lo);
          f->nv[out] = static_cast<uint8_t>(hi - lo + 1);
        }
      }
      ++out;
    }
  }
  f->n = static_cast<int>(out);
}

// Fixes each pixel's column and row window for blur widths up to sigma_px.
void Freeze(Fit *f, double sigma_px, double extra_px = 0) {
  const int W = f->width;
  const float reach = static_cast<float>(kKnobs.reach * sigma_px + extra_px);
  f->i0.resize(f->n);
  f->j0.resize(f->n);
  f->nu.resize(f->n);
  f->nv.resize(f->n);
  for (int idx = 0; idx < f->n; ++idx) {
    for (int axis = 0; axis < 2; ++axis) {
      const float pos = axis == 0 ? f->u0[idx] : f->v0[idx];
      const float rr = reach * (axis == 0 ? f->gu[idx] : f->gv[idx]);
      int lo = std::clamp(FloorI(pos - rr), -1, W);
      int hi = std::clamp(FloorI(pos + rr), -1, W);
      if (hi - lo + 1 > kMaxC) {
        const int mid = std::clamp(FloorI(pos), -1, W);
        lo = std::max(lo, mid - (kMaxC - 1) / 2);
        hi = std::min(hi, lo + kMaxC - 1);
      }
      if (axis == 0) {
        f->i0[idx] = static_cast<int8_t>(lo);
        f->nu[idx] = static_cast<uint8_t>(hi - lo + 1);
      } else {
        f->j0[idx] = static_cast<int8_t>(lo);
        f->nv[idx] = static_cast<uint8_t>(hi - lo + 1);
      }
    }
  }
  f->A.resize(f->n);
  f->B.resize(f->n);
  f->C.resize(f->n);
}

// The model at every pixel of the working set: class coverages and, with Deriv, the derivatives of the
// base prediction (levels p.lev, unknown cells at the mid-level) with respect to the parameters.
template <bool Box, bool Deriv>
void ModelPass(Fit *f, const Params &p) {
  const float g0 = static_cast<float>(p.map.g[0]), g1 = static_cast<float>(p.map.g[1]), g2 = static_cast<float>(p.map.g[2]);
  const float g3 = static_cast<float>(p.map.g[3]), g4 = static_cast<float>(p.map.g[4]), g5 = static_cast<float>(p.map.g[5]);
  const float g6 = static_cast<float>(p.map.g[6]), g7 = static_cast<float>(p.map.g[7]);
  const float lv[4] = {static_cast<float>(p.lev[0]), static_cast<float>(p.lev[1]), static_cast<float>(p.lev[2]),
                       static_cast<float>(0.5 * (p.lev[0] + p.lev[1]))};
  const float s0 = static_cast<float>(p.s[0]), s1 = static_cast<float>(p.s[1]), s2 = static_cast<float>(p.s[2]);
  const int W = f->width;
  const float inv_scale = static_cast<float>(1.0 / f->scale);
  if (Deriv && f->J.size() < static_cast<size_t>(f->n) * kStrideJ) f->J.resize(static_cast<size_t>(f->n) * kStrideJ);
  Axis au, av;
  if (Deriv && !Box && f->n > 300) {
    for (int idx = 0; idx < f->n; ++idx) ++g_prof_local.hist[std::min(9, static_cast<int>(f->nu[idx]))][std::min(9, static_cast<int>(f->nv[idx]))];
  }
  for (int idx = 0; idx < f->n; ++idx) {
    const float x = f->x[idx], y = f->y[idx];
    const float iw = 1.0f / (g6 * x + g7 * y + 1.0f);
    const float u = (g0 * x + g1 * y + g2) * iw, v = (g3 * x + g4 * y + g5) * iw;
    const float gu = f->gu[idx], gv = f->gv[idx];
    const int i0 = f->i0[idx], j0 = f->j0[idx], nu = f->nu[idx], nv = f->nv[idx];
    if (!Box && nu * nv == 2) {
      // A single cell edge: one boundary along one axis.
      const bool ua = nu == 2;
      const float pos = ua ? u : v;
      const float sp = ua ? s0 : s1;
      const int lo = ua ? i0 : j0;
      const int cl_lo = f->cls[j0 + 1][i0 + 1];
      const int cl_hi = ua ? f->cls[j0 + 1][i0 + 2] : f->cls[j0 + 2][i0 + 1];
      const float inv_s = 1.0f / (sp * (ua ? gu : gv));
      const float z = (static_cast<float>(lo + 1) - pos) * inv_s;
      float cc, pp;
      Lut(z, &cc, &pp);
      float S[4] = {0, 0, 0, 0};
      S[cl_lo] += cc;
      S[cl_hi] += 1.0f - cc;
      f->A[idx] = S[0] + 0.5f * S[3];
      f->B[idx] = S[1] + 0.5f * S[3];
      f->C[idx] = S[2];
      if (Deriv) {
        const float d = (lv[cl_hi] - lv[cl_lo]) * pp * inv_s;
        const float pu = ua ? d : 0.0f, pv = ua ? 0.0f : d;
        const float pb = d * z;
        float *J = &f->J[static_cast<size_t>(idx) * kStrideJ];
        const float xi = x * iw, yi = y * iw;
        const float q = pu * u + pv * v;
        J[0] = pu * xi;
        J[1] = pu * yi;
        J[2] = pu * iw;
        J[3] = pv * xi;
        J[4] = pv * yi;
        J[5] = pv * iw;
        J[6] = -q * xi;
        J[7] = -q * yi;
        J[8] = ua ? pb * gu : 0.0f;
        J[9] = ua ? 0.0f : pb * gv;
      }
      continue;
    }
    if (!Box && nu == 2 && nv == 2) {
      // One boundary on each axis: four cells.
      const int c00 = f->cls[j0 + 1][i0 + 1], c10 = f->cls[j0 + 1][i0 + 2], c01 = f->cls[j0 + 2][i0 + 1], c11 = f->cls[j0 + 2][i0 + 2];
      const float inv_su = 1.0f / (s0 * gu), inv_sv = 1.0f / (s1 * gv);
      const float zu = (static_cast<float>(i0 + 1) - u) * inv_su, zv = (static_cast<float>(j0 + 1) - v) * inv_sv;
      float a0, pdu, b0, pdv;
      Lut(zu, &a0, &pdu);
      Lut(zv, &b0, &pdv);
      const float a1 = 1.0f - a0, b1 = 1.0f - b0;
      float S[4] = {0, 0, 0, 0};
      S[c00] += a0 * b0;
      S[c10] += a1 * b0;
      S[c01] += a0 * b1;
      S[c11] += a1 * b1;
      f->A[idx] = S[0] + 0.5f * S[3];
      f->B[idx] = S[1] + 0.5f * S[3];
      f->C[idx] = S[2];
      if (Deriv) {
        const float l00 = lv[c00], l10 = lv[c10], l01 = lv[c01], l11 = lv[c11];
        const float pu = pdu * inv_su * (b0 * (l10 - l00) + b1 * (l11 - l01));
        const float pv = pdv * inv_sv * (a0 * (l01 - l00) + a1 * (l11 - l10));
        float *J = &f->J[static_cast<size_t>(idx) * kStrideJ];
        const float xi = x * iw, yi = y * iw;
        const float q = pu * u + pv * v;
        J[0] = pu * xi;
        J[1] = pu * yi;
        J[2] = pu * iw;
        J[3] = pv * xi;
        J[4] = pv * yi;
        J[5] = pv * iw;
        J[6] = -q * xi;
        J[7] = -q * yi;
        J[8] = zu * pu * gu;
        J[9] = zv * pv * gv;
      }
      continue;
    }
    if (Box) {
      // Motion blur: a straight segment of taps, each a Gaussian that also covers the tap spacing.
      const float len = std::sqrt(s1 * s1 + s2 * s2);
      const int taps = p.taps;
      const float inv_t = 1.0f / static_cast<float>(taps);
      const float st = std::sqrt(s0 * s0 + len * len * inv_t * inv_t / 12.0f);
      const float dst = s0 / st;
      float S[4] = {0, 0, 0, 0};
      float Jt[kMaxJ] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      for (int t = 0; t < taps; ++t) {
        const float frac = (static_cast<float>(t) + 0.5f) * inv_t - 0.5f;
        const float xt = x + frac * s1 * inv_scale, yt = y + frac * s2 * inv_scale;
        const float iwt = 1.0f / (g6 * xt + g7 * yt + 1.0f);
        const float ut = (g0 * xt + g1 * yt + g2) * iwt, vt = (g3 * xt + g4 * yt + g5) * iwt;
        AxisEval<false, Deriv>(ut, st * gu, 0.0f, i0, nu, W, &au);
        AxisEval<false, Deriv>(vt, st * gv, 0.0f, j0, nv, W, &av);
        float pu = 0, pv = 0, pbu = 0, pbv = 0;
        for (int j = 0; j < nv; ++j) {
          const float cv = av.a[j];
          const uint8_t *crow = &f->cls[j0 + j + 1][i0 + 1];
          for (int i = 0; i < nu; ++i) {
            const int c = crow[i];
            S[c] += au.a[i] * cv * inv_t;
            if (Deriv) {
              const float l = lv[c] * inv_t;
              pu += l * au.du[i] * cv;
              pv += l * au.a[i] * av.du[j];
              pbu += l * au.ds[i] * cv;
              pbv += l * au.a[i] * av.ds[j];
            }
          }
        }
        if (Deriv) {
          const float xi = xt * iwt, yi = yt * iwt;
          const float q = pu * ut + pv * vt;
          Jt[0] += pu * xi;
          Jt[1] += pu * yi;
          Jt[2] += pu * iwt;
          Jt[3] += pv * xi;
          Jt[4] += pv * yi;
          Jt[5] += pv * iwt;
          Jt[6] -= q * xi;
          Jt[7] -= q * yi;
          Jt[8] += (pbu * gu + pbv * gv) * dst;
          const float gux = (g0 - ut * g6) * iwt, guy = (g1 - ut * g7) * iwt;
          const float gvx = (g3 - vt * g6) * iwt, gvy = (g4 - vt * g7) * iwt;
          Jt[9] += frac * inv_scale * (pu * gux + pv * gvx);
          Jt[10] += frac * inv_scale * (pu * guy + pv * gvy);
        }
      }
      f->A[idx] = S[0] + 0.5f * S[3];
      f->B[idx] = S[1] + 0.5f * S[3];
      f->C[idx] = S[2];
      if (Deriv) {
        float *J = &f->J[static_cast<size_t>(idx) * kStrideJ];
        for (int k = 0; k < kMaxJ; ++k) J[k] = Jt[k];
      }
      continue;
    }
    AxisEval<false, Deriv>(u, s0 * gu, 0.0f, i0, nu, W, &au);
    AxisEval<false, Deriv>(v, s1 * gv, 0.0f, j0, nv, W, &av);
    float S[4] = {0, 0, 0, 0};
    float pu = 0, pv = 0, pbu = 0, pbv = 0;
    for (int j = 0; j < nv; ++j) {
      const float cv = av.a[j];
      const uint8_t *crow = &f->cls[j0 + j + 1][i0 + 1];
      for (int i = 0; i < nu; ++i) {
        const int c = crow[i];
        const float wgt = au.a[i] * cv;
        S[c] += wgt;
        if (Deriv) {
          const float l = lv[c];
          pu += l * au.du[i] * cv;
          pv += l * au.a[i] * av.du[j];
          pbu += l * au.ds[i] * cv;
          pbv += l * au.a[i] * av.ds[j];
        }
      }
    }
    f->A[idx] = S[0] + 0.5f * S[3];
    f->B[idx] = S[1] + 0.5f * S[3];
    f->C[idx] = S[2];
    if (Deriv) {
      float *J = &f->J[static_cast<size_t>(idx) * kStrideJ];
      const float xi = x * iw, yi = y * iw;
      const float q = pu * u + pv * v;
      J[0] = pu * xi;
      J[1] = pu * yi;
      J[2] = pu * iw;
      J[3] = pv * xi;
      J[4] = pv * yi;
      J[5] = pv * iw;
      J[6] = -q * xi;
      J[7] = -q * yi;
      J[8] = pbu * gu;
      J[9] = pbv * gv;
    }
  }
}

inline void RunModel(Fit *f, const Params &p, bool deriv) {
  if (p.box) {
    if (deriv) ModelPass<true, true>(f, p);
    else ModelPass<true, false>(f, p);
  } else {
    if (deriv) ModelPass<false, true>(f, p);
    else ModelPass<false, false>(f, p);
  }
}

// Solves (A) x = b by Gaussian elimination with partial pivoting; false if singular.
bool Solve(int N, double a[kMaxT][kMaxT], double b[kMaxT], double x[kMaxT]) {
  for (int c = 0; c < N; ++c) {
    int piv = c;
    for (int r = c + 1; r < N; ++r) {
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    }
    if (std::fabs(a[piv][c]) < 1e-18) return false;
    if (piv != c) {
      for (int k = 0; k < N; ++k) std::swap(a[c][k], a[piv][k]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < N; ++r) {
      const double m = a[r][c] / a[c][c];
      if (m == 0.0) continue;
      for (int k = c; k < N; ++k) a[r][k] -= m * a[c][k];
      b[r] -= m * b[c];
    }
  }
  for (int r = N - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < N; ++k) s -= a[r][k] * x[k];
    x[r] = s / a[r][r];
  }
  return true;
}

// Linear least squares for the three levels at fixed mapping and blur, from the model pass. Returns the SSE.
bool FitLevels(const Fit &f, Params *p, double *sse = nullptr) {
  double ata[3][3] = {}, atb[3] = {}, yy = 0;
  for (int idx = 0; idx < f.n; ++idx) {
    const float gain = 1.0f + static_cast<float>(p->lx) * f.x[idx] + static_cast<float>(p->ly) * f.y[idx];
    const float sw0 = gain * f.A[idx], sw1 = gain * f.B[idx], sw2 = gain * f.C[idx];
    const float o = f.obs[idx];
    atb[0] += sw0 * o;
    atb[1] += sw1 * o;
    atb[2] += sw2 * o;
    yy += static_cast<double>(o) * o;
    ata[0][0] += sw0 * sw0;
    ata[0][1] += sw0 * sw1;
    ata[0][2] += sw0 * sw2;
    ata[1][1] += sw1 * sw1;
    ata[1][2] += sw1 * sw2;
    ata[2][2] += sw2 * sw2;
  }
  ata[1][0] = ata[0][1];
  ata[2][0] = ata[0][2];
  ata[2][1] = ata[1][2];
  for (int a = 0; a < 3; ++a) ata[a][a] += 1e-6 * (ata[0][0] + ata[1][1] + ata[2][2]) + 1e-9;
  double m[9] = {ata[0][0], ata[0][1], ata[0][2], ata[1][0], ata[1][1], ata[1][2], ata[2][0], ata[2][1], ata[2][2]};
  double inv[9];
  if (!Invert3(m, inv)) return false;
  for (int a = 0; a < 3; ++a) p->lev[a] = inv[3 * a] * atb[0] + inv[3 * a + 1] * atb[1] + inv[3 * a + 2] * atb[2];
  if (sse != nullptr) *sse = yy - (p->lev[0] * atb[0] + p->lev[1] * atb[1] + p->lev[2] * atb[2]);
  return true;
}

// The median absolute residual of the current model over every third pixel (a cheap robust scale).
double MedianAbsResidual(const Fit &f, const Params &p) {
  thread_local std::vector<float> res;
  res.clear();
  for (int idx = 0; idx < f.n; idx += 3) {
    const double gain = 1.0 + p.lx * f.x[idx] + p.ly * f.y[idx];
    res.push_back(static_cast<float>(std::fabs(f.obs[idx] - gain * (p.lev[0] * f.A[idx] + p.lev[1] * f.B[idx] + p.lev[2] * f.C[idx]))));
  }
  if (res.empty()) return 0;
  std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
  return res[res.size() / 2];
}

struct Accum {
  double cost = 0;
  int n = 0;
  double jtj[kMaxT][kMaxT];
  double jtr[kMaxT];
};

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
using V4 = __m128;
inline V4 VZero() { return _mm_setzero_ps(); }
inline V4 VLoad(const float *p) { return _mm_loadu_ps(p); }
inline V4 VMulAdd(V4 acc, V4 a, V4 b) { return _mm_add_ps(acc, _mm_mul_ps(a, b)); }
inline double VSum(V4 v) {
  alignas(16) float t[4];
  _mm_store_ps(t, v);
  return (static_cast<double>(t[0]) + t[1]) + (static_cast<double>(t[2]) + t[3]);
}
#elif defined(__ARM_NEON)
using V4 = float32x4_t;
inline V4 VZero() { return vdupq_n_f32(0.0f); }
inline V4 VLoad(const float *p) { return vld1q_f32(p); }
inline V4 VMulAdd(V4 acc, V4 a, V4 b) { return vmlaq_f32(acc, a, b); }
inline double VSum(V4 v) { return static_cast<double>(vgetq_lane_f32(v, 0)) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3); }
#else
struct V4 {
  float v[4];
};
inline V4 VZero() { return V4{{0, 0, 0, 0}}; }
inline V4 VLoad(const float *p) { return V4{{p[0], p[1], p[2], p[3]}}; }
inline V4 VMulAdd(V4 acc, V4 a, V4 b) {
  for (int l = 0; l < 4; ++l) acc.v[l] += a.v[l] * b.v[l];
  return acc;
}
inline double VSum(V4 v) { return (static_cast<double>(v.v[0]) + v.v[1]) + (static_cast<double>(v.v[2]) + v.v[3]); }
#endif

// Dot products of column a with four columns b0..b3 over n8 (a multiple of 8) elements.
inline void Dot4(const float *a, const float *b0, const float *b1, const float *b2, const float *b3, int n8, double *out) {
  V4 s0 = VZero(), s1 = VZero(), s2 = VZero(), s3 = VZero();
  V4 t0 = VZero(), t1 = VZero(), t2 = VZero(), t3 = VZero();
  for (int i = 0; i < n8; i += 8) {
    const V4 a0 = VLoad(a + i), a1 = VLoad(a + i + 4);
    s0 = VMulAdd(s0, a0, VLoad(b0 + i));
    t0 = VMulAdd(t0, a1, VLoad(b0 + i + 4));
    s1 = VMulAdd(s1, a0, VLoad(b1 + i));
    t1 = VMulAdd(t1, a1, VLoad(b1 + i + 4));
    s2 = VMulAdd(s2, a0, VLoad(b2 + i));
    t2 = VMulAdd(t2, a1, VLoad(b2 + i + 4));
    s3 = VMulAdd(s3, a0, VLoad(b3 + i));
    t3 = VMulAdd(t3, a1, VLoad(b3 + i + 4));
  }
  out[0] = VSum(s0) + VSum(t0);
  out[1] = VSum(s1) + VSum(t1);
  out[2] = VSum(s2) + VSum(t2);
  out[3] = VSum(s3) + VSum(t3);
}

// As Evaluate, but the weighted columns are gathered first and the normal equations formed as dot products.
void Evaluate(Fit *f, Params *p, double kcut, int mode, Accum *acc, bool run_model = true) {
  const bool normal = mode >= 1;
  const double tp0 = NowS();
  if (run_model) {
    RunModel(f, *p, normal);
  }
  const double tp1 = NowS();
  g_prof_local.t[5] += tp1 - tp0;
  ++g_prof_local.evals;
  const int n = f->n;
  const float kc = static_cast<float>(kcut);
  const int ng = p->box ? 11 : 10;
  const int nt = ng + 5;
  const int stride = (n + 7) & ~7;
  thread_local std::vector<float> buf;
  if (normal && buf.size() < static_cast<size_t>(nt + 1) * stride) buf.resize(static_cast<size_t>(nt + 1) * stride);
  float *base_col = buf.data();
  const float L0 = static_cast<float>(p->lev[0]), L1 = static_cast<float>(p->lev[1]), L2 = static_cast<float>(p->lev[2]);
  const float lx = static_cast<float>(p->lx), ly = static_cast<float>(p->ly);
  double cost = 0;
  int count = 0;
  for (int idx = 0; idx < n; ++idx) {
    const float a = f->A[idx], bb = f->B[idx], c = f->C[idx];
    const float base = L0 * a + L1 * bb + L2 * c;
    const float gain = 1.0f + lx * f->x[idx] + ly * f->y[idx];
    const float pred = gain * base;
    const float ob = f->obs[idx];
    if ((ob >= 255.0f && pred >= 255.0f) || (ob <= 0.0f && pred <= 0.0f)) continue;
    const float r = ob - pred;
    const float ar = std::fabs(r);
    const float sw = ar <= kc ? 1.0f : std::sqrt(kc / ar);
    cost += ar <= kc ? r * r : kc * (2.0f * ar - kc);
    if (normal) {
      const float *J = &f->J[static_cast<size_t>(idx) * kStrideJ];
      const float g = gain * sw;
      float *dst = base_col + count;
      for (int k = 0; k < ng; ++k) dst[static_cast<size_t>(k) * stride] = g * J[k];
      dst[static_cast<size_t>(ng) * stride] = g * a;
      dst[static_cast<size_t>(ng + 1) * stride] = g * bb;
      dst[static_cast<size_t>(ng + 2) * stride] = g * c;
      const float sb = sw * base;
      dst[static_cast<size_t>(ng + 3) * stride] = sb * f->x[idx];
      dst[static_cast<size_t>(ng + 4) * stride] = sb * f->y[idx];
      dst[static_cast<size_t>(nt) * stride] = sw * r;
    }
    ++count;
  }
  acc->cost = cost;
  acc->n = count;
  const double tp2 = NowS();
  g_prof_local.t[10] += tp2 - tp1;
  if (normal) {
    const int n8 = (count + 7) & ~7;
    for (int k = 0; k <= nt; ++k) {
      float *col = base_col + static_cast<size_t>(k) * stride;
      for (int i = count; i < n8; ++i) col[i] = 0.0f;
    }
    // Columns i..nt (the residual is column nt) in groups of four; the last group repeats a column.
    for (int i = 0; i < nt; ++i) {
      const float *ci = base_col + static_cast<size_t>(i) * stride;
      for (int k = i; k <= nt; k += 4) {
        const float *b[4];
        for (int l = 0; l < 4; ++l) b[l] = base_col + static_cast<size_t>(std::min(k + l, nt)) * stride;
        double v[4];
        Dot4(ci, b[0], b[1], b[2], b[3], n8, v);
        for (int l = 0; l < 4 && k + l <= nt; ++l) {
          if (k + l == nt) {
            acc->jtr[i] = v[l];
          } else {
            acc->jtj[i][k + l] = v[l];
            acc->jtj[k + l][i] = v[l];
          }
        }
      }
    }
  }
  g_prof_local.t[11] += NowS() - tp2;
  g_prof_local.t[6] += NowS() - tp1;
}

// Applies a solved step to the parameters.
void ApplyStep(const double *delta, Params *q) {
  for (int k = 0; k < 8; ++k) q->map.g[k] += delta[k];
  int ng = 10;
  if (q->box) {
    q->s[0] = std::clamp(q->s[0] + delta[8], 0.3, 3.0);
    q->s[1] += delta[9];
    q->s[2] += delta[10];
    const double len = std::hypot(q->s[1], q->s[2]);
    if (len > 24.0) {
      q->s[1] *= 24.0 / len;
      q->s[2] *= 24.0 / len;
    }
    ng = 11;
  } else {
    q->s[0] = std::clamp(q->s[0] + delta[8], 0.35, q->smax);
    q->s[1] = std::clamp(q->s[1] + delta[9], 0.35, q->smax);
  }
  for (int k = 0; k < 3; ++k) q->lev[k] += delta[ng + k];
  q->lx += delta[ng + 3];
  q->ly += delta[ng + 4];
}

bool SolveStep(const Accum &cur, bool box, double lambda, double *delta) {
  const int nt = (box ? 11 : 10) + 5;
  double a[kMaxT][kMaxT], b[kMaxT];
  for (int r = 0; r < nt; ++r) {
    const double diag = std::max(cur.jtj[r][r], 1e-9);
    for (int c = 0; c < nt; ++c) a[r][c] = cur.jtj[r][c];
    a[r][r] += lambda * diag;
    b[r] = cur.jtr[r];
  }
  return Solve(nt, a, b, delta);
}

// Levenberg-Marquardt over the geometry and blur; returns the final cost and leaves the best
// parameters in *p. Stops when the corners move less than `tol` pixels.
double Optimise(Fit *f, Params *p, double kcut, int max_iter, double tol, int width, double *initial_cost = nullptr, bool model_done = false) {
  Accum cur, trial;
  Evaluate(f, p, kcut, 2, &cur, !model_done);
  if (initial_cost != nullptr) *initial_cost = cur.cost;
  double lambda = 1e-3;
  double prev[4][2];
  CellCorners(p->map, width, prev);
  for (int iter = 0; iter < max_iter; ++iter) {
    bool accepted = false;
    for (int attempt = 0; attempt < 6 && !accepted; ++attempt) {
      double delta[kMaxT];
      if (!SolveStep(cur, p->box, lambda, delta)) {
        lambda *= 10;
        continue;
      }
      Params q = *p;
      ApplyStep(delta, &q);
      double corners[4][2];
      CellCorners(q.map, width, corners);
      double move = 0;
      for (int k = 0; k < 4; ++k) move = std::max(move, std::hypot((corners[k][0] - prev[k][0]) * f->scale, (corners[k][1] - prev[k][1]) * f->scale));
      if (move < tol) {
        *p = q;
        return cur.cost;
      }
      Evaluate(f, &q, kcut, 2, &trial);
      if (trial.n > 0 && trial.cost < cur.cost) {
        *p = q;
        std::swap(cur, trial);
        lambda = std::max(lambda / 4, 1e-7);
        accepted = true;
        for (int k = 0; k < 4; ++k) {
          prev[k][0] = corners[k][0];
          prev[k][1] = corners[k][1];
        }
      } else {
        lambda *= 5;
      }
    }
    if (!accepted) break;
  }
  return cur.cost;
}


}  // namespace

bool PolishDetection(const image_u8_t *im, apriltag_detection_t *det) {
  const double tt0 = NowS();
  struct Done {
    double t0;
    ~Done() {
      g_prof_local.t[9] += NowS() - t0;
      ++g_prof_local.tags;
    }
  } done{tt0};
  const apriltag_family_t *fam = det->family;
  if (fam == nullptr || fam->reversed_border) return false;
  const int W = static_cast<int>(fam->width_at_border);
  if (W < 4 || W > kMaxCells || det->id < 0 || det->id >= static_cast<int>(fam->ncodes)) return false;

  static thread_local Fit fit;
  static thread_local Fit scan;
  fit.width = W;
  for (int j = 0; j < kPad; ++j) {
    for (int i = 0; i < kPad; ++i) fit.cls[j][i] = 2;
  }
  for (int j = 0; j < W; ++j) {
    for (int i = 0; i < W; ++i) fit.cls[j + 1][i + 1] = 0;
  }
  const uint64_t code = fam->codes[det->id];
  for (uint32_t b = 0; b < fam->nbits; ++b) {
    const uint32_t bx = fam->bit_x[b], by = fam->bit_y[b];
    if (bx >= static_cast<uint32_t>(W) || by >= static_cast<uint32_t>(W)) return false;
    if ((code >> (fam->nbits - 1 - b)) & 1u) fit.cls[by + 1][bx + 1] = 1;
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
  double tq = NowS();
  g_prof_local.t[0] += tq - tt0;
  BoxInfo box;
  ComputeBox(fit, p.map, limit, im, &box);
  const int thin = box.thin;
  BuildScan(&scan, fit, p.map, cell_px, im);
  g_prof_local.t[1] += NowS() - tq;
  tq = NowS();
  if (scan.n < 40) return false;
  const double candidates[] = {0.7, 1.2, 2.0, 3.5};
  double best_cost = 1e300, best_s = 1.0;
  double best_lev[3] = {p.lev[0], p.lev[1], p.lev[2]};
  for (int ci = 0; ci < 4; ++ci) {
    if (ci == 3 && best_s < 2.0) break;
    const double s = candidates[ci];
    p.s[0] = p.s[1] = s;
    Freeze(&scan, s);
    ModelPass<false, false>(&scan, p);
    double sse = 0;
    if (!FitLevels(scan, &p, &sse)) continue;
    if (sse < best_cost) {
      best_cost = sse;
      best_s = s;
      std::copy(p.lev, p.lev + 3, best_lev);
    }
  }
  p.s[0] = p.s[1] = best_s;
  std::copy(best_lev, best_lev + 3, p.lev);
  g_prof_local.t[2] += NowS() - tq;
  if (thin > 1 && (DevFlags() & 16) == 0) {
    // A tag too large for the working set keeps its profile fit unless it is blurred or clipped.
    size_t saturated = 0;
    for (float v : scan.obs) saturated += v >= 255.0f;
    if (best_s < 1.6 && saturated * 20 < scan.obs.size()) return false;
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
          const double expected = fit.cls[j + 1][i + 1] ? p.lev[1] : p.lev[0];
          if (std::fabs(im->buf[iy * im->stride + ix] - expected) > 0.3 * range) marked[j][i] = 1;
        }
      }
      for (int j = 0; j < W; ++j) {
        for (int i = 0; i < W; ++i) {
          if (marked[j][i]) fit.cls[j + 1][i + 1] = 3;
        }
      }
    }
  }
  const double window_mult = kKnobs.win_mult;
  double window = std::min(window_mult * best_s + 0.3 + kKnobs.blur_extra * std::max(0.0, best_s - 1.2), 0.75 * cell_px + 0.5);
  window = std::max(window, kKnobs.min_win);
  tq = NowS();
  const double reach_s = std::min(kKnobs.smax_mult * best_s, 8.0);
  BuildFit(&fit, p.map, window, kKnobs.reach * reach_s, box, im);
  g_prof_local.t[3] += NowS() - tq;
  tq = NowS();
  if (fit.n < 40) return false;
  g_prof_local.pixels += fit.n;
  p.smax = reach_s;
  ModelPass<false, true>(&fit, p);
  FitLevels(fit, &p);

  const double kcut = std::max(6.0, 4.0 * 1.4826 * MedianAbsResidual(fit, p));
  g_prof_local.t[4] += NowS() - tq;
  tq = NowS();
  double start_cost = 0;
  const double tol = kKnobs.tol;
  const int iters = static_cast<int>(kKnobs.iters);
  double cost = Optimise(&fit, &p, kcut, iters, tol, W, &start_cost, true);

  const double topt = NowS() - tq;
  (void)topt;
  tq = NowS();
  bool moving_accepted = false;
  if (std::max(p.s[0], p.s[1]) > 1.4 && (DevFlags() & 8) == 0) {
    // A wide blur may be motion: refit with a Gaussian blur plus a straight blur segment.
    const double sigma0 = 0.7;
    const double lu = std::sqrt(12.0 * std::max(p.s[0] * p.s[0] - sigma0 * sigma0, 0.5));
    const double lv = std::sqrt(12.0 * std::max(p.s[1] * p.s[1] - sigma0 * sigma0, 0.5));
    const double *g = p.map.g;
    double nux = g[0] - g[2] * g[6], nuy = g[1] - g[2] * g[7], nvx = g[3] - g[5] * g[6], nvy = g[4] - g[5] * g[7];
    const double nul = std::hypot(nux, nuy), nvl = std::hypot(nvx, nvy);
    nux /= nul;
    nuy /= nul;
    nvx /= nvl;
    nvy /= nvl;
    const double det2 = nux * nvy - nuy * nvx;
    if (std::fabs(det2) > 0.2) {
      Params moving = p;
      moving.box = true;
      moving.s[0] = sigma0;
      double best_sse = 1e300;
      Params best_start = moving;
      for (int sign = 0; sign < 2; ++sign) {
        const double a2 = lu, b2 = sign == 0 ? lv : -lv;
        moving.s[1] = (a2 * nvy - b2 * nuy) / det2;
        moving.s[2] = (nux * b2 - nvx * a2) / det2;
        const double len = std::hypot(moving.s[1], moving.s[2]);
        moving.taps = std::clamp(static_cast<int>(std::ceil(len / 1.6)), 3, 9);
        const double st = std::sqrt(sigma0 * sigma0 + len * len / 12.0);
        Freeze(&fit, st, 0.5 * len);
        ModelPass<true, false>(&fit, moving);
        double sse = 0;
        if (FitLevels(fit, &moving, &sse) && sse < best_sse) {
          best_sse = sse;
          best_start = moving;
        }
      }
      if (best_sse < 1e300) {
        Params moving2 = best_start;
        const double len = std::hypot(moving2.s[1], moving2.s[2]);
        Freeze(&fit, std::sqrt(sigma0 * sigma0 + len * len / 12.0) * 1.15, 0.6 * len);
        const double cost_moving = Optimise(&fit, &moving2, kcut, 5, 2.5 * tol, W);
        Params plain = p;
        Accum tmp;
        Evaluate(&fit, &plain, kcut, 0, &tmp);
        if (cost_moving < 0.97 * tmp.cost) {
          p = moving2;
          moving_accepted = true;
        }
      }
    }
  }

  g_prof_local.t[7] += NowS() - tq;
  if (!moving_accepted && cost > start_cost) return false;
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
