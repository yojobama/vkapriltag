#include "SmallQuad.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace apriltag_vulkan {
namespace {

struct Pt {
  double x, y;
};

double Cross(const Pt &o, const Pt &a, const Pt &b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Andrew's monotone chain; the points are sorted in place.
std::vector<Pt> ConvexHull(std::vector<Pt> &p) {
  std::sort(p.begin(), p.end(),
            [](const Pt &a, const Pt &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
  p.erase(std::unique(p.begin(), p.end(),
                      [](const Pt &a, const Pt &b) { return a.x == b.x && a.y == b.y; }),
          p.end());
  const size_t n = p.size();
  if (n < 3) return p;
  std::vector<Pt> h(2 * n);
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    while (k >= 2 && Cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
    h[k++] = p[i];
  }
  for (size_t i = n - 1, t = k + 1; i > 0; --i) {
    while (k >= t && Cross(h[k - 2], h[k - 1], p[i - 1]) <= 0) --k;
    h[k++] = p[i - 1];
  }
  h.resize(k - 1);
  return h;
}

int OtsuThreshold(const int hist[256], int total, double *gap) {
  double sum = 0;
  for (int i = 0; i < 256; ++i) sum += static_cast<double>(i) * hist[i];
  double sum_b = 0, best = -1;
  int w_b = 0, best_t = 0;
  *gap = 0;
  for (int t = 0; t < 256; ++t) {
    w_b += hist[t];
    if (w_b == 0) continue;
    const int w_f = total - w_b;
    if (w_f == 0) break;
    sum_b += static_cast<double>(t) * hist[t];
    const double m_b = sum_b / w_b, m_f = (sum - sum_b) / w_f;
    const double between = static_cast<double>(w_b) * w_f * (m_b - m_f) * (m_b - m_f);
    if (between > best) {
      best = between;
      best_t = t;
      *gap = m_f - m_b;
    }
  }
  return best_t;
}

// Largest fraction of a region's edge pixels that may be dark for a retry to be worth flooding.
constexpr double kRetryRingMax = 0.05;

struct Roi {
  int x0, y0, w, h;
};

// kFit: a quad was written; kEdge: every dark component touched the region's edge; kNone: no tag-like blob.
enum class Fit { kFit, kEdge, kNone };

Fit FitAtThreshold(const uint8_t *grey, int width, const Roi &roi, int thr, double max_ring_dark, double out[4][2]) {
  const int rx0 = roi.x0, ry0 = roi.y0, rw = roi.w, rh = roi.h;
  // Dark pixels (<= thr) in 8-connected components; keep the largest that stays clear of the edge.
  const int pw = rw + 2;
  static thread_local std::vector<uint8_t> blocked;
  static thread_local std::vector<int> stack, cur, best_pixels;
  blocked.assign(static_cast<size_t>(pw) * (rh + 2), 1);
  for (int y = 0; y < rh; ++y) {
    const uint8_t *row = grey + static_cast<size_t>(ry0 + y) * width + rx0;
    uint8_t *out_row = blocked.data() + static_cast<size_t>(y + 1) * pw + 1;
    for (int x = 0; x < rw; ++x) out_row[x] = row[x] > thr;
  }
  if (max_ring_dark < 1.0) {
    int ring = 0;
    for (int x = 0; x < rw; ++x) ring += !blocked[pw + 1 + x] + !blocked[rh * pw + 1 + x];
    for (int y = 0; y < rh; ++y) ring += !blocked[(y + 1) * pw + 1] + !blocked[(y + 1) * pw + rw];
    if (ring > max_ring_dark * (2 * (rw + rh))) return Fit::kEdge;
  }
  const int step[8] = {-1, 1, -pw, pw, -pw - 1, -pw + 1, pw - 1, pw + 1};
  auto flood = [&](int start, bool keep) {
    cur.clear();
    stack.assign(1, start);
    blocked[start] = 1;
    while (!stack.empty()) {
      const int p = stack.back();
      stack.pop_back();
      if (keep) cur.push_back(p);
      for (int k = 0; k < 8; ++k) {
        const int q = p + step[k];
        if (blocked[q]) continue;
        blocked[q] = 1;
        stack.push_back(q);
      }
    }
  };
  bool any_touch = false;
  best_pixels.clear();
  auto edge_pixel = [&](int idx) {
    if (blocked[idx]) return;
    any_touch = true;
    flood(idx, false);
  };
  for (int x = 0; x < rw; ++x) {
    edge_pixel(pw + 1 + x);
    edge_pixel(rh * pw + 1 + x);
  }
  for (int y = 0; y < rh; ++y) {
    edge_pixel((y + 1) * pw + 1);
    edge_pixel((y + 1) * pw + rw);
  }
  for (int y = 1; y <= rh; ++y) {
    for (int x = 1; x <= rw; ++x) {
      const int idx = y * pw + x;
      if (blocked[idx]) continue;
      flood(idx, true);
      if (cur.size() > best_pixels.size()) best_pixels.swap(cur);
    }
  }
  if (best_pixels.size() < 12) return any_touch ? Fit::kEdge : Fit::kNone;

  std::vector<Pt> pts;
  pts.reserve(best_pixels.size() * 4);
  for (int p : best_pixels) {
    const double px = rx0 + p % pw - 1, py = ry0 + p / pw - 1;
    pts.push_back({px - 0.5, py - 0.5});
    pts.push_back({px + 0.5, py - 0.5});
    pts.push_back({px + 0.5, py + 0.5});
    pts.push_back({px - 0.5, py + 0.5});
  }
  const std::vector<Pt> hull = ConvexHull(pts);
  if (hull.size() < 3) return Fit::kNone;

  double best_area = 1e30, best_c[4][2] = {}, best_w = 0, best_h = 0;
  for (size_t i = 0; i < hull.size(); ++i) {
    const Pt &a = hull[i], &b = hull[(i + 1) % hull.size()];
    const double len = std::hypot(b.x - a.x, b.y - a.y);
    if (len < 1e-9) continue;
    const double ux = (b.x - a.x) / len, uy = (b.y - a.y) / len, vx = -uy, vy = ux;
    double umin = 1e30, umax = -1e30, vmin = 1e30, vmax = -1e30;
    for (const Pt &p : hull) {
      const double u = p.x * ux + p.y * uy, v = p.x * vx + p.y * vy;
      umin = std::min(umin, u);
      umax = std::max(umax, u);
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
    const double area = (umax - umin) * (vmax - vmin);
    if (area < best_area) {
      best_area = area;
      best_w = umax - umin;
      best_h = vmax - vmin;
      const double us[4] = {umin, umax, umax, umin}, vs[4] = {vmin, vmin, vmax, vmax};
      for (int k = 0; k < 4; ++k) {
        best_c[k][0] = us[k] * ux + vs[k] * vx;
        best_c[k][1] = us[k] * uy + vs[k] * vy;
      }
    }
  }
  if (best_area > 1e29) return Fit::kNone;
  const double lo = std::min(best_w, best_h), hi = std::max(best_w, best_h);
  if (lo < 5.0 || hi > 1.4 * lo) return Fit::kNone;
  const double fill = static_cast<double>(best_pixels.size()) / best_area;
  if (fill < 0.4 || fill > 0.92) return Fit::kNone;
  for (int k = 0; k < 4; ++k) {
    out[k][0] = best_c[k][0];
    out[k][1] = best_c[k][1];
  }
  return Fit::kFit;
}

Fit FitInRegion(const uint8_t *grey, int width, int height, double cx, double cy, double half,
                double first_limit, double low_limit, double out[4][2]) {
  const int rx0 = std::max(0, static_cast<int>(std::floor(cx - half)));
  const int ry0 = std::max(0, static_cast<int>(std::floor(cy - half)));
  const int rx1 = std::min(width - 1, static_cast<int>(std::ceil(cx + half)));
  const int ry1 = std::min(height - 1, static_cast<int>(std::ceil(cy + half)));
  const int rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1;
  if (rw < 8 || rh < 8) return Fit::kNone;

  int hist[256] = {};
  for (int y = ry0; y <= ry1; ++y) {
    const uint8_t *row = grey + static_cast<size_t>(y) * width;
    for (int x = rx0; x <= rx1; ++x) ++hist[row[x]];
  }
  double gap;
  const int thr = OtsuThreshold(hist, rw * rh, &gap);
  if (gap < 20.0) return Fit::kNone;
  int below = 0, p5 = 0;
  for (; p5 < 255 && below < rw * rh / 20; ++p5) below += hist[p5];
  const Fit first = FitAtThreshold(grey, width, {rx0, ry0, rw, rh}, thr, first_limit, out);
  const int low = (p5 + thr) / 2;
  if (first != Fit::kEdge || low >= thr) return first;
  return FitAtThreshold(grey, width, {rx0, ry0, rw, rh}, low, low_limit, out);
}

}  // namespace

bool FitSmallQuad(const uint8_t *grey, int width, int height, double x0, double y0, double x1,
                  double y1, double out[4][2]) {
  const double cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1);
  const double half = std::max(0.75 * std::max(x1 - x0, y1 - y0), 10.0) + 4.0;
  const Fit small = FitInRegion(grey, width, height, cx, cy, std::min(28.0, half), 1.0, kRetryRingMax, out);
  if (small != Fit::kEdge) return small == Fit::kFit;
  return FitInRegion(grey, width, height, cx, cy, std::min(44.0, 1.8 * half), kRetryRingMax, kRetryRingMax, out) == Fit::kFit;
}

}  // namespace apriltag_vulkan
