#include "vkapriltag/gpu/QuadDecode.h"

#include "vkapriltag/common/Profiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>

namespace apriltag_vulkan {
namespace {

// Gaussian filter taps applied to the windowed error.
constexpr std::array<double, 7> kFilterCoefficients = {
    0.01110899634659290314, 0.13533528149127960205, 0.60653066635131835938,
    1.00000000000000000000, 0.60653066635131835938, 0.13533528149127960205,
    0.01110899634659290314,
};

constexpr int kNMaxima = 10;

// Hypotenuse of two floats, evaluated in double and rounded once to float.
inline float Hypot(float x, float y) {
  const double dx = x, dy = y;
  return static_cast<float>(std::sqrt(dx * dx + dy * dy));
}

// LineFitMoments addition and subtraction.
LineFitMoments Add(const LineFitMoments &a, const LineFitMoments &b) {
  LineFitMoments r;
  r.Mx = a.Mx + b.Mx;
  r.My = a.My + b.My;
  r.W = a.W + b.W;
  r.Mxx = a.Mxx + b.Mxx;
  r.Myy = a.Myy + b.Myy;
  r.Mxy = a.Mxy + b.Mxy;
  return r;
}

LineFitMoments Sub(const LineFitMoments &a, const LineFitMoments &b) {
  LineFitMoments r;
  r.Mx = a.Mx - b.Mx;
  r.My = a.My - b.My;
  r.W = a.W - b.W;
  r.Mxx = a.Mxx - b.Mxx;
  r.Myy = a.Myy - b.Myy;
  r.Mxy = a.Mxy - b.Mxy;
  return r;
}

// Cumulative moments over the circular index range [index0, index1] inclusive, from prefix sums
// `cs` (cs[k] = sum of points [0, k]) of length total_points. index0 >= index1 takes the
// wrap-around branch.
inline LineFitMoments ReadMomentsWindow(const std::vector<LineFitMoments> &cs, size_t total_points,
                                size_t index0, size_t index1) {
  LineFitMoments result;
  if (index0 < index1) {
    result = cs[index1];
    if (index0 > 0) {
      result = Sub(result, cs[index0 - 1]);
    }
    result.N = static_cast<int32_t>(index1 - index0 + 1);
  } else {
    LineFitMoments lfsz = cs[total_points - 1];
    result = lfsz;
    if (index0 > 0) {
      result = Sub(lfsz, cs[index0 - 1]);
    }
    result = Add(result, cs[index1]);
    result.N = static_cast<int32_t>(total_points - index0 + index1 + 1);
  }
  return result;
}

// Sum of squared error of a line fit from its moments.
double FitLineError(int N, int64_t Mx, int64_t My, int64_t Mxx, int64_t Myy, int64_t Mxy,
                    int64_t W) {
  int64_t Cxx = Mxx * W - Mx * Mx;
  int64_t Cxy = Mxy * W - Mx * My;
  int64_t Cyy = Myy * W - My * My;
  const float hypot_cached = Hypot(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
  const float eight_w_squared = static_cast<float>(static_cast<int64_t>(W) * static_cast<int64_t>(W) * 8.0);
  const float eig_small = (static_cast<float>(Cxx + Cyy) - hypot_cached) / eight_w_squared;
  return N * eig_small;
}

// Point-on-line (lineparam01), unit normal (lineparam23), sum and mean squared error from a
// moments window.
void FitLine(const LineFitMoments &moments, double *lineparam01, double *lineparam23, double *err,
            double *mse) {
  int64_t Mx = moments.Mx, My = moments.My, W = moments.W;
  int64_t Cxx = moments.Mxx * W - Mx * Mx;
  int64_t Cxy = moments.Mxy * W - Mx * My;
  int64_t Cyy = moments.Myy * W - My * My;

  const float hypot_cached = Hypot(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
  const float eight_w_squared = static_cast<float>(static_cast<int64_t>(W) * static_cast<int64_t>(W) * 8.0);
  const float eig_small = (static_cast<float>(Cxx + Cyy) - hypot_cached) / eight_w_squared;

  if (lineparam01) {
    lineparam01[0] = static_cast<float>(Mx) / static_cast<float>(W * 2);
    lineparam01[1] = static_cast<float>(My) / static_cast<float>(W * 2);
  }
  if (lineparam23) {
    const float nx1 = static_cast<float>(Cxx - Cyy) - hypot_cached;
    const float ny1 = static_cast<float>(Cxy) * 2;
    const float M1 = nx1 * nx1 + ny1 * ny1;
    const float nx2 = static_cast<float>(Cxy) * 2;
    const float ny2 = static_cast<float>(Cyy - Cxx) - hypot_cached;
    const float M2 = nx2 * nx2 + ny2 * ny2;

    float nx, ny;
    if (M1 > M2) {
      nx = nx1;
      ny = ny1;
    } else {
      nx = nx2;
      ny = ny2;
    }
    float length = Hypot(nx, ny);
    lineparam23[0] = nx / length;
    lineparam23[1] = ny / length;
  }

  *err = moments.N * eig_small;
  *mse = eig_small;
}

// Per-blob quad fit result (combinatorial search or DP seeding).
struct FitQuadResult {
  bool valid = false;
  // Set whenever DP was tried (quad_fit_method == kDp and >= 4 points); dp_used only if it
  // produced the accepted result. For QuadDecode::last_dp_stats().
  bool dp_attempted = false;
  bool dp_used = false;
  uint32_t indices[4] = {};
  LineFitMoments moments[4];
};

// Index of the largest value(i) over [begin, end), the lowest such index on ties, folded into
// (best, best_idx) with ties again going to the lower index. Four lanes run independently.
template <typename T, typename F>
inline void ArgMax(size_t begin, size_t end, F value, T &best, size_t &best_idx) {
  T lane_best[4] = {T(-1), T(-1), T(-1), T(-1)};
  size_t lane_idx[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  size_t i = begin;
  for (; i + 4 <= end; i += 4) {
    for (int k = 0; k < 4; ++k) {
      const T v = value(i + k);
      if (v > lane_best[k]) {
        lane_best[k] = v;
        lane_idx[k] = i + k;
      }
    }
  }
  for (; i < end; ++i) {
    const T v = value(i);
    if (v > lane_best[0]) {
      lane_best[0] = v;
      lane_idx[0] = i;
    }
  }
  for (int k = 0; k < 4; ++k) {
    if (lane_best[k] > best || (lane_best[k] == best && lane_idx[k] < best_idx)) {
      best = lane_best[k];
      best_idx = lane_idx[k];
    }
  }
}

// Seeds 4 corner indices: the two mutually-farthest points (2-pass approximation), then the
// point of maximum perpendicular deviation on each arc. Returns false if the blob does not look
// like a clean quad (too few points, degenerate diameter, empty arc, or no real deviation).
bool FindDpCornerIndices(std::span<const RawLineFitPoint> points, size_t begin, size_t n,
                         uint32_t out_indices[4]) {
  if (n < 8) return false;

  const RawLineFitPoint *pts = points.data() + begin;
  // Squared distance between two points, exact in int64.
  auto sq_dist = [pts](size_t a, size_t b) {
    const int64_t dx = pts[a].x2() - pts[b].x2(), dy = pts[a].y2() - pts[b].y2();
    return dx * dx + dy * dy;
  };

  int64_t best_d = -1;
  size_t a_idx = 0;
  ArgMax(1, n, [&](size_t i) { return sq_dist(0, i); }, best_d, a_idx);
  best_d = -1;
  size_t b_idx = 0;
  ArgMax(0, n, [&](size_t i) { return sq_dist(a_idx, i); }, best_d, b_idx);
  if (a_idx == b_idx) return false;
  const double diameter_sq = static_cast<double>(sq_dist(a_idx, b_idx));
  if (diameter_sq < 1e-6) return false;

  const size_t lo = std::min(a_idx, b_idx), hi = std::max(a_idx, b_idx);

  // Cross product of (b - a) and (p - a), exact in int64, squared in double.
  const int64_t ax = pts[a_idx].x2(), ay = pts[a_idx].y2();
  const int64_t abx = pts[b_idx].x2() - ax, aby = pts[b_idx].y2() - ay;
  auto perp = [&](size_t i) {
    const double cross = static_cast<double>(abx * (pts[i].y2() - ay) - aby * (pts[i].x2() - ax));
    return cross * cross;
  };

  size_t c1_idx = n, c2_idx = n;  // n is not a valid index; doubles as "not found"
  double best_perp1 = -1.0, best_perp2 = -1.0;
  ArgMax(lo + 1, hi, perp, best_perp1, c1_idx);
  ArgMax(0, lo, perp, best_perp2, c2_idx);
  ArgMax(hi + 1, n, perp, best_perp2, c2_idx);
  if (c1_idx == n || c2_idx == n) return false;

  // Reject degenerate shapes: both corners must deviate from the diameter line (threshold is
  // scaled by diameter_sq^2).
  const double min_perp = 0.0001 * diameter_sq * diameter_sq;
  if (best_perp1 < min_perp || best_perp2 < min_perp) return false;

  // Circular order: lo -> c1 -> hi -> c2. c1 lies between lo and hi, c2 outside [lo, hi].
  out_indices[0] = static_cast<uint32_t>(lo);
  out_indices[1] = static_cast<uint32_t>(c1_idx);
  out_indices[2] = static_cast<uint32_t>(hi);
  out_indices[3] = static_cast<uint32_t>(c2_idx);
  return true;
}

// Tries DP corner seeding for one blob using the prefix sums `cs`. Returns an invalid result if
// seeding fails or a segment exceeds max_line_fit_mse (the caller then falls back to peaks).
FitQuadResult TryDpQuad(const DetectorConfig &config, std::span<const RawLineFitPoint> points,
                        size_t begin, size_t total_points,
                        const std::vector<LineFitMoments> &cs) {
  FitQuadResult result;
  uint32_t idx[4];
  if (!FindDpCornerIndices(points, begin, total_points, idx)) return result;

  LineFitMoments seg_moments[4];
  for (int k = 0; k < 4; ++k) {
    seg_moments[k] = ReadMomentsWindow(cs, total_points, idx[k], idx[(k + 1) % 4]);
    double err, mse;
    FitLine(seg_moments[k], nullptr, nullptr, &err, &mse);
    if (mse > config.max_line_fit_mse) return result;
  }

  result.valid = true;
  for (int k = 0; k < 4; ++k) {
    result.indices[k] = idx[k];
    result.moments[k] = seg_moments[k];
  }
  return result;
}

// Windowed error, 7-tap Gaussian filter and peak detection, then a combinatorial search over the
// top 10 peaks, for one blob. Its points are `points[begin, end)`, in perimeter (theta) order.
FitQuadResult FitQuadForBlob(const DetectorConfig &config,
                            std::span<const RawLineFitPoint> points, size_t begin,
                            size_t end, QuadFitScratch &scratch) {
  FitQuadResult result;
  const size_t total_points = end - begin;
  if (total_points < 4) return result;

  // scratch is reused across blobs on this slot; the buffers only grow.
  std::vector<LineFitMoments> &cs = scratch.cs;
  std::vector<double> &error = scratch.error;
  std::vector<double> &filtered = scratch.filtered;
  std::vector<std::pair<double, uint32_t>> &peaks = scratch.peaks;
  cs.resize(total_points);
  error.resize(total_points);
  filtered.resize(total_points);
  peaks.clear();

  {
    // Running sums stay in registers; each field of cs[k] is stored separately.
    int32_t mx = 0, my = 0, w_sum = 0;
    int64_t mxx = 0, mxy = 0, myy = 0;
    for (size_t k = 0; k < total_points; ++k) {
      const RawLineFitPoint &p = points[begin + k];
      // Unpack once: the accessors are bit extractions and cs may alias points as far as the compiler knows.
      const int32_t x2 = p.x2(), y2 = p.y2(), w = p.W();
      const int64_t wx = static_cast<int64_t>(w) * x2;
      const int64_t wy = static_cast<int64_t>(w) * y2;
      mx += static_cast<int32_t>(wx);
      my += static_cast<int32_t>(wy);
      w_sum += w;
      mxx += wx * x2;
      mxy += wx * y2;
      myy += wy * y2;
      LineFitMoments &c = cs[k];
      c.Mx = mx;
      c.My = my;
      c.W = w_sum;
      c.Mxx = mxx;
      c.Myy = myy;
      c.Mxy = mxy;
      c.N = 0;
    }
  }

  if (config.quad_fit_method == DetectorConfig::QuadFitMethod::kDp) {
    result.dp_attempted = true;  // sticks even if we fall through to peaks below
    FitQuadResult dp_result = TryDpQuad(config, points, begin, total_points, cs);
    if (dp_result.valid) {
      dp_result.dp_attempted = true;
      dp_result.dp_used = true;
      return dp_result;
    }
    // Fall back to the peaks-based search.
  }

  const int ksz = std::min<int>(20, static_cast<int>(total_points / 12));

  // Windowed error per point. Circular indices use conditional subtraction instead of `%`.
  const size_t n = total_points;
  const size_t k_off = static_cast<size_t>(ksz);
  for (size_t i = 0; i < n; ++i) {
    const size_t i0 = (i >= k_off) ? (i - k_off) : (i + n - k_off);
    size_t i1 = i + k_off;
    if (i1 >= n) i1 -= n;
    LineFitMoments m = ReadMomentsWindow(cs, n, i0, i1);
    error[i] = FitLineError(m.N, m.Mx, m.My, m.Mxx, m.Myy, m.Mxy, m.W);
  }

  // 7-tap Gaussian filter, applied circularly.
  constexpr size_t kHalf = kFilterCoefficients.size() / 2;  // 3
  for (size_t i = 0; i < n; ++i) {
    double accumulated = 0.0;
    size_t idx = (i >= kHalf) ? (i - kHalf) : (i + n - kHalf);
    for (size_t j = 0; j < kFilterCoefficients.size(); ++j) {
      accumulated += error[idx] * kFilterCoefficients[j];
      if (++idx == n) idx = 0;
    }
    filtered[i] = accumulated;
  }

  // Peaks are strict local maxima of the filtered error, stored negated so an ascending sort
  // puts the strongest first.
  {
    double before = filtered[n - 1];
    double cur = filtered[0];
    for (size_t i = 0; i < n; ++i) {
      const double after = filtered[(i + 1 == n) ? 0 : (i + 1)];
      if (cur > before && cur > after) {
        peaks.emplace_back(-cur, static_cast<uint32_t>(i));
      }
      before = cur;
      cur = after;
    }
  }

  const size_t K = std::min<size_t>(peaks.size(), kNMaxima);
  if (K < 4) return result;

  std::stable_sort(peaks.begin(), peaks.end(),
                   [](const auto &a, const auto &b) { return a.first < b.first; });
  std::array<uint32_t, kNMaxima> point_indices{};
  for (size_t k = 0; k < K; ++k) point_indices[k] = peaks[k].second;
  std::sort(point_indices.begin(), point_indices.begin() + K);

  // Line fit of the segment between every ordered pair of peaks; (b, a) with b > a closes the quad.
  double seg_err[kNMaxima][kNMaxima];
  double seg_mse[kNMaxima][kNMaxima];
  double seg_params[kNMaxima][kNMaxima][2];
  for (size_t a = 0; a < K; ++a) {
    for (size_t b = 0; b < K; ++b) {
      if (a == b) continue;
      FitLine(ReadMomentsWindow(cs, total_points, point_indices[a], point_indices[b]), nullptr,
              seg_params[a][b], &seg_err[a][b], &seg_mse[a][b]);
    }
  }
  const double max_mse = config.max_line_fit_mse;

  // Brute force all C(10, 4) = 210 combinations, keeping the minimum total error.
  double best_error = std::numeric_limits<double>::max();
  uint32_t best[4] = {0, 0, 0, 0};

  for (uint32_t m0 = 0; m0 < kNMaxima - 3; ++m0) {
    for (uint32_t m1 = m0 + 1; m1 < kNMaxima - 2; ++m1) {
      if (m1 >= K) break;
      if (seg_mse[m0][m1] > max_mse) continue;
      const double errm0m1 = seg_err[m0][m1];
      const double *paramsm0m123 = seg_params[m0][m1];

      for (uint32_t m2 = m1 + 1; m2 < kNMaxima - 1; ++m2) {
        if (m2 >= K) break;
        if (seg_mse[m1][m2] > max_mse) continue;
        const double errm1m2 = seg_err[m1][m2];
        const double *paramsm1m223 = seg_params[m1][m2];

        const double dot = paramsm0m123[0] * paramsm1m223[0] + paramsm0m123[1] * paramsm1m223[1];
        if (std::fabs(dot) > config.cos_critical_rad) continue;

        for (uint32_t m3 = m2 + 1; m3 < kNMaxima; ++m3) {
          if (m3 >= K) break;
          if (seg_mse[m2][m3] > max_mse) continue;
          if (seg_mse[m3][m0] > max_mse) continue;

          const double total = errm0m1 + errm1m2 + seg_err[m2][m3] + seg_err[m3][m0];
          if (total < best_error) {
            best_error = total;
            best[0] = m0;
            best[1] = m1;
            best[2] = m2;
            best[3] = m3;
          }
        }
      }
    }
  }

  if (best_error == std::numeric_limits<double>::max()) return result;
  if (!(best_error < config.max_line_fit_mse * total_points)) return result;

  result.valid = true;
  for (int k = 0; k < 4; ++k) result.indices[k] = point_indices[best[k]];
  for (int k = 0; k < 4; ++k) {
    result.moments[k] =
        ReadMomentsWindow(cs, total_points, result.indices[k], result.indices[(k + 1) % 4]);
  }
  return result;
}

// Resolves quad_fit_method: explicit config wins, then APRILTAG_VK_QUADFIT.
DetectorConfig::QuadFitMethod ResolveQuadFitMethod(DetectorConfig::QuadFitMethod configured) {
  if (const char *m = std::getenv("APRILTAG_VK_QUADFIT")) {
    const std::string method = m;
    if (method == "dp") return DetectorConfig::QuadFitMethod::kDp;
    if (method == "peaks") return DetectorConfig::QuadFitMethod::kPeaks;
  }
  return configured;
}

// UpdateFitQuads for one blob: refits the lines from the moments, intersects adjacent lines to
// get corners and applies the geometric checks. Returns false if the quad is rejected.
bool UpdateFitQuad(const DetectorConfig &config, const FitQuadResult &quad, DetectedQuad *out) {
  const double min_tag_width =
      std::max(3.0, config.tag_width / static_cast<double>(config.decimation));
  double lines[4][4];
  for (int i = 0; i < 4; ++i) {
    double err, mse;
    FitLine(quad.moments[i], lines[i], lines[i] + 2, &err, &mse);
  }

  double corners[4][2];
  bool bad_determinant = false;
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3;
    double A00 = lines[i][3], A01 = -lines[j][3];
    double A10 = -lines[i][2], A11 = lines[j][2];
    double B0 = -lines[i][0] + lines[j][0];
    double B1 = -lines[i][1] + lines[j][1];

    double det = A00 * A11 - A10 * A01;
    if (std::fabs(det) < 0.001) {
      bad_determinant = true;
      break;
    }
    double W00 = A11 / det, W01 = -A01 / det;
    double L0 = W00 * B0 + W01 * B1;

    corners[i][0] = lines[i][0] + L0 * A00;
    corners[i][1] = lines[i][1] + L0 * A10;
  }
  if (bad_determinant) return false;

  // Area check: sum of the two triangles (0,1,2) and (2,3,0).
  {
    auto triangle_area = [&](int a, int b, int c) {
      double la = std::hypot(corners[b][0] - corners[a][0], corners[b][1] - corners[a][1]);
      double lb = std::hypot(corners[c][0] - corners[b][0], corners[c][1] - corners[b][1]);
      double lc = std::hypot(corners[a][0] - corners[c][0], corners[a][1] - corners[c][1]);
      double p = (la + lb + lc) / 2;
      double v = p * (p - la) * (p - lb) * (p - lc);
      return v > 0 ? std::sqrt(v) : 0.0;
    };
    double area = triangle_area(0, 1, 2) + triangle_area(2, 3, 0);
    if (area < 0.95 * min_tag_width * min_tag_width) return false;
  }

  // Reject quads that are not convex and consistently wound.
  bool reject = false;
  for (int i = 0; i < 4 && !reject; ++i) {
    int i0 = i, i1 = (i + 1) & 3, i2 = (i + 2) & 3;
    double dx1 = corners[i1][0] - corners[i0][0];
    double dy1 = corners[i1][1] - corners[i0][1];
    double dx2 = corners[i2][0] - corners[i1][0];
    double dy2 = corners[i2][1] - corners[i1][1];
    double cos_dtheta = (dx1 * dx2 + dy1 * dy2) /
                       std::sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
    if (std::fabs(cos_dtheta) > config.cos_critical_rad || dx1 * dy2 < dy1 * dx2) {
      reject = true;
    }
  }
  if (reject) return false;

  // Decimated grid to full-resolution pixels: a plain scale. decimate.comp point-samples the
  // top-left pixel of each block, so no pixel-centre adjustment applies.
  const double decimation = static_cast<double>(config.decimation);
  for (int i = 0; i < 4; ++i) {
    out->p[i][0] = corners[i][0] * decimation;
    out->p[i][1] = corners[i][1] * decimation;
  }
  return true;
}

}  // namespace

QuadDecode::QuadDecode(const DetectorConfig &config)
    : config_(config),
      pool_(std::make_unique<WorkerPool>(ResolveThreadCount(config.cpu_threads), "quad-worker")),
      scratch_(pool_->threads()) {
  config_.quad_fit_method = ResolveQuadFitMethod(config_.quad_fit_method);
  if (const char *v = std::getenv("APRILTAG_VK_DP_PEAKS_CHECK")) config_.dp_peaks_check = v[0] == '1';
}

std::vector<DetectedQuad> QuadDecode::Decode(
    std::span<const RawLineFitPoint> line_fit_points) const {
  return Decode(line_fit_points, {});
}

std::vector<DetectedQuad> QuadDecode::Decode(std::span<const RawLineFitPoint> line_fit_points,
                                             std::span<const GpuQuadFit> gpu_fits) const {
  VKA_PROF_SCOPE("quad.decode_all");
  // fit_quad_dp.comp runs the DP path, so its results stand in only for that method.
  const bool use_gpu =
      !gpu_fits.empty() && config_.quad_fit_method == DetectorConfig::QuadFitMethod::kDp;
  std::vector<DetectedQuad> output;

  // Group the points into contiguous per-blob spans (sorted by blob_index, then theta).
  struct Span {
    size_t begin;
    size_t end;
  };
  std::vector<Span> spans;
  {
    size_t i = 0;
    while (i < line_fit_points.size()) {
      const uint32_t blob_index = line_fit_points[i].blob_index();
      size_t j = i;
      while (j < line_fit_points.size() && line_fit_points[j].blob_index() == blob_index) ++j;
      spans.push_back({i, j});
      i = j;
    }
  }

  // Fit and refit every blob independently; each entry is written by exactly one task.
  struct SpanResult {
    bool dp_attempted = false, dp_used = false, has_quad = false;
    DetectedQuad quad;
  };
  std::vector<SpanResult> per_span(spans.size());
  pool_->ParallelFor(spans.size(), [&](size_t s, unsigned slot) {
    prof::Scope fit_scope("quad.fit_blob");
    fit_scope.set_arg(static_cast<int64_t>(spans[s].end - spans[s].begin));
    SpanResult &r = per_span[s];
    const uint32_t blob = line_fit_points[spans[s].begin].blob_index();
    if (use_gpu && blob < gpu_fits.size() && gpu_fits[blob].status != 2u) {
      r.dp_attempted = true;
      r.dp_used = true;
      if (gpu_fits[blob].status == 1u) {
        std::memcpy(r.quad.p, gpu_fits[blob].p, sizeof(r.quad.p));
        r.has_quad = true;
        if (config_.dp_peaks_check) {
          DetectorConfig peaks_config = config_;
          peaks_config.quad_fit_method = DetectorConfig::QuadFitMethod::kPeaks;
          const FitQuadResult fp = FitQuadForBlob(peaks_config, line_fit_points, spans[s].begin,
                                                  spans[s].end, scratch_[slot]);
          DetectedQuad unused;
          r.has_quad = fp.valid && UpdateFitQuad(config_, fp, &unused);
        }
      }
      return;
    }
    const FitQuadResult fq =
        FitQuadForBlob(config_, line_fit_points, spans[s].begin, spans[s].end, scratch_[slot]);
    r.dp_attempted = fq.dp_attempted;
    r.dp_used = fq.dp_used;
    r.has_quad = fq.valid && UpdateFitQuad(config_, fq, &r.quad);
    if (r.has_quad && fq.dp_used && config_.dp_peaks_check) {
      DetectorConfig peaks_config = config_;
      peaks_config.quad_fit_method = DetectorConfig::QuadFitMethod::kPeaks;
      const FitQuadResult fp = FitQuadForBlob(peaks_config, line_fit_points, spans[s].begin,
                                              spans[s].end, scratch_[slot]);
      DetectedQuad unused;
      r.has_quad = fp.valid && UpdateFitQuad(config_, fp, &unused);
    }
  });

  last_dp_stats_ = DpStats{};
  for (const SpanResult &r : per_span) {
    if (r.dp_attempted) {
      ++last_dp_stats_.attempts;
      if (!r.dp_used) ++last_dp_stats_.fallbacks;
    }
  }

  // Collect survivors in span order so the output is independent of thread scheduling.
  for (const SpanResult &r : per_span) {
    if (r.has_quad) output.push_back(r.quad);
  }

  return output;
}

}  // namespace apriltag_vulkan
