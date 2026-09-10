#include "vkapriltag/gpu/QuadDecode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace apriltag_vulkan {
namespace {

// Mirrors frc971::apriltag::FilterCoefficients() (line_fit_filter.h).
constexpr std::array<double, 7> kFilterCoefficients = {
    0.01110899634659290314, 0.13533528149127960205, 0.60653066635131835938,
    1.00000000000000000000, 0.60653066635131835938, 0.13533528149127960205,
    0.01110899634659290314,
};

constexpr int kNMaxima = 10;

// LineFitMoments + / - (mirrors SumLineFitPoints / the implicit subtraction
// done inline throughout line_fit_filter.cu).
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

// Exact port of frc971::apriltag::ReadMoments (line_fit_filter.cu): reads a
// cumulative range-sum of moments over the circular point index range
// [index0, index1] (inclusive), where `cs` is a prefix-sum array (cs[k] =
// sum of points [0, k] inclusive) of length `total_points`. Faithfully
// replicates the original's `index0 < index1` branch condition (including
// its behavior when index0 == index1, which falls into the "wrap around"
// branch exactly as in the original).
//
// Bug fix: the wrap-around branch used to read cs[index0 - 1]
// unconditionally, unlike the index0 < index1 branch just above it, which
// already guards the equivalent read with `if (index0 > 0)`. index0 == 0
// reaches this branch whenever ksz == 0 (blobs under 12 points, at point
// index 0 - see FitQuadForBlob's `const int ksz = ...`), which read
// cs[SIZE_MAX]: an out-of-bounds read landing on whatever memory happens to
// sit just before the cs vector's buffer. Found while porting this window
// read to GPU (A9) and cross-checking against synthetic inputs small
// enough to hit this path - see the GPU shader's mirroring guard
// (compute_window_error.comp) for the equivalent fix on that side.
LineFitMoments ReadMomentsWindow(const std::vector<LineFitMoments> &cs, size_t total_points,
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

// Exact port of FitLineError (line_fit_filter.cu).
double FitLineError(int N, int64_t Mx, int64_t My, int64_t Mxx, int64_t Myy, int64_t Mxy,
                    int64_t W) {
  int64_t Cxx = Mxx * W - Mx * Mx;
  int64_t Cxy = Mxy * W - Mx * My;
  int64_t Cyy = Myy * W - My * My;
  const float hypot_cached = std::hypotf(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
  const float eight_w_squared = static_cast<float>(static_cast<int64_t>(W) * static_cast<int64_t>(W) * 8.0);
  const float eig_small = (static_cast<float>(Cxx + Cyy) - hypot_cached) / eight_w_squared;
  return N * eig_small;
}

// Exact port of the two-output-variant FitLine (line_fit_filter.cu /
// apriltag_detect.cu's HostFitLine): computes the point-on-line
// (lineparam01) and unit normal (lineparam23), plus sum-of-squared-error and
// mean-squared-error, from a moments window.
void FitLine(const LineFitMoments &moments, double *lineparam01, double *lineparam23, double *err,
            double *mse) {
  int64_t Mx = moments.Mx, My = moments.My, W = moments.W;
  int64_t Cxx = moments.Mxx * W - Mx * Mx;
  int64_t Cxy = moments.Mxy * W - Mx * My;
  int64_t Cyy = moments.Myy * W - My * My;

  const float hypot_cached = std::hypotf(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
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
    float length = std::hypotf(nx, ny);
    lineparam23[0] = nx / length;
    lineparam23[1] = ny / length;
  }

  *err = moments.N * eig_small;
  *mse = eig_small;
}

// Per-blob result of the quad fit, from either the combinatorial search
// (mirrors FitQuad in line_fit_filter.h) or the item-3 DP corner-seeding
// path.
struct FitQuadResult {
  bool valid = false;
  // Set (regardless of `valid`) whenever quad_fit_method == kDp and this
  // blob was eligible (>= 4 points) - i.e. DP was tried at all. dp_used is
  // set only when it actually produced the accepted result; attempted-but-
  // not-used means DP fell back to the combinatorial search. Purely for
  // QuadDecode::last_dp_stats() instrumentation.
  bool dp_attempted = false;
  bool dp_used = false;
  uint32_t indices[4] = {};
  LineFitMoments moments[4];
};

// Coordinates for DP corner seeding: RawLineFitPoint's (x2, y2), the same
// "doubled" decimated-pixel grid the line fit itself uses. Approximate
// (integer boundary-tracing coordinates, not sub-pixel), which is fine here
// since these only SEED which 4 indices to fit lines through - the actual
// corner accuracy comes from the exact same ReadMomentsWindow + FitLine +
// intersection code the peaks path already uses on whichever 4 indices are
// chosen.
struct Point2 {
  double x, y;
};

Point2 PointAt(const std::vector<RawLineFitPoint> &points, size_t begin, size_t idx) {
  const RawLineFitPoint &p = points[begin + idx];
  return {static_cast<double>(p.x2), static_cast<double>(p.y2)};
}

double SqDist(Point2 a, Point2 b) {
  const double dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

// Squared perpendicular distance of p from the (infinite) line through a, b,
// scaled by |ab|^2 (i.e. actual_dist^2 * |ab|^2) - avoids a sqrt/division
// per point; callers only ever compare these against each other or against
// a threshold likewise scaled by diameter_sq.
double ScaledPerpDistSq(Point2 a, Point2 b, Point2 p) {
  const double abx = b.x - a.x, aby = b.y - a.y;
  const double apx = p.x - a.x, apy = p.y - a.y;
  const double cross = abx * apy - aby * apx;
  return cross * cross;
}

// Seeds 4 corner indices geometrically: the two mutually-farthest boundary
// points (a 2-pass O(n) approximation of the polygon's diameter - exact
// all-pairs would be O(n^2)), then the point of maximum perpendicular
// deviation from that diameter on each of the two resulting arcs. Returns
// false (leaving out_indices untouched) whenever the blob doesn't look like
// a clean quad this way: too few points, a degenerate (near-zero) diameter,
// an empty arc, or a "corner" that doesn't meaningfully deviate from the
// diameter line (a triangle or a rounded blob would still produce SOME
// max-distance point on each arc, just an unconvincing one). The caller
// falls back to the peaks-based combinatorial search in every failure case.
bool FindDpCornerIndices(const std::vector<RawLineFitPoint> &points, size_t begin, size_t n,
                         uint32_t out_indices[4]) {
  if (n < 8) return false;

  const Point2 p0 = PointAt(points, begin, 0);
  size_t a_idx = 0;
  double best_d = -1.0;
  for (size_t i = 1; i < n; ++i) {
    const double d = SqDist(p0, PointAt(points, begin, i));
    if (d > best_d) {
      best_d = d;
      a_idx = i;
    }
  }
  const Point2 pa = PointAt(points, begin, a_idx);
  size_t b_idx = 0;
  best_d = -1.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = SqDist(pa, PointAt(points, begin, i));
    if (d > best_d) {
      best_d = d;
      b_idx = i;
    }
  }
  if (a_idx == b_idx) return false;
  const Point2 pb = PointAt(points, begin, b_idx);
  const double diameter_sq = SqDist(pa, pb);
  if (diameter_sq < 1e-6) return false;

  const size_t lo = std::min(a_idx, b_idx), hi = std::max(a_idx, b_idx);

  size_t c1_idx = n, c2_idx = n;  // n is not a valid index; doubles as "not found"
  double best_perp1 = -1.0, best_perp2 = -1.0;
  for (size_t i = lo + 1; i < hi; ++i) {
    const double d = ScaledPerpDistSq(pa, pb, PointAt(points, begin, i));
    if (d > best_perp1) {
      best_perp1 = d;
      c1_idx = i;
    }
  }
  for (size_t i = 0; i < lo; ++i) {
    const double d = ScaledPerpDistSq(pa, pb, PointAt(points, begin, i));
    if (d > best_perp2) {
      best_perp2 = d;
      c2_idx = i;
    }
  }
  for (size_t i = hi + 1; i < n; ++i) {
    const double d = ScaledPerpDistSq(pa, pb, PointAt(points, begin, i));
    if (d > best_perp2) {
      best_perp2 = d;
      c2_idx = i;
    }
  }
  if (c1_idx == n || c2_idx == n) return false;

  // Require both corner candidates to meaningfully deviate from the
  // diameter line. 1% of (diameter * diameter) is a low bar - a real tag
  // corner deviates by roughly half the diameter - it exists only to reject
  // genuinely degenerate shapes (near-triangles, rounded blobs).
  const double min_perp = 0.0001 * diameter_sq * diameter_sq;
  if (best_perp1 < min_perp || best_perp2 < min_perp) return false;

  // Circular order around the perimeter: lo -> c1 -> hi -> c2 -> (back to
  // lo). c1 lies strictly between lo and hi by construction; c2 lies
  // strictly outside [lo, hi], i.e. on the arc that wraps from hi through
  // n-1/0 back to lo. Absolute numeric order doesn't matter beyond that -
  // ReadMomentsWindow's wrap-around branch handles any consecutive pair
  // regardless - only that consecutive entries trace consecutive arcs.
  out_indices[0] = static_cast<uint32_t>(lo);
  out_indices[1] = static_cast<uint32_t>(c1_idx);
  out_indices[2] = static_cast<uint32_t>(hi);
  out_indices[3] = static_cast<uint32_t>(c2_idx);
  return true;
}

// Attempts the DP corner-seeding path for one blob. `cs` is the same
// prefix-sum array FitQuadForBlob already computes for the peaks path.
// Returns an invalid (default) result whenever DP's geometric seeding fails
// outright, or the resulting 4 segments don't pass the same
// max_line_fit_mse gate the combinatorial search applies to every candidate
// segment - both cases mean "fall back to peaks", handled by the caller.
FitQuadResult TryDpQuad(const DetectorConfig &config, const std::vector<RawLineFitPoint> &points,
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

// Ports DoFitLines' per-point windowed-error + 7-tap-gaussian-filter +
// peak-detection, followed by DoFitQuads' top-10-peaks combinatorial search,
// for a single blob. The blob's RawLineFitPoint entries are `points[begin,
// end)`, in perimeter (theta-sorted) order.
//
// Takes a span rather than a copied vector: the caller used to materialize a
// fresh std::vector per blob, which is a heap allocation and a copy of the
// whole run for every one of the several hundred blobs in a frame.
FitQuadResult FitQuadForBlob(const DetectorConfig &config,
                            const std::vector<RawLineFitPoint> &points, size_t begin,
                            size_t end) {
  FitQuadResult result;
  const size_t total_points = end - begin;
  if (total_points < 4) return result;

  // Scratch reused across every blob this thread fits. FitQuadForBlob is
  // called once per blob (several hundred per frame) and used to heap-allocate
  // four vectors each time; the buffers only ever grow.
  thread_local std::vector<LineFitMoments> cs;
  thread_local std::vector<double> error;
  thread_local std::vector<double> filtered;
  thread_local std::vector<std::pair<double, uint32_t>> peaks;
  cs.resize(total_points);
  error.resize(total_points);
  filtered.resize(total_points);
  peaks.clear();

  {
    LineFitMoments running{};
    for (size_t k = 0; k < total_points; ++k) {
      const RawLineFitPoint &p = points[begin + k];
      const int64_t wx = static_cast<int64_t>(p.W) * p.x2;
      const int64_t wy = static_cast<int64_t>(p.W) * p.y2;
      running.Mx += static_cast<int32_t>(wx);
      running.My += static_cast<int32_t>(wy);
      running.W += p.W;
      running.Mxx += wx * p.x2;
      running.Mxy += wx * p.y2;
      running.Myy += wy * p.y2;
      cs[k] = running;
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
    // Falls through to the peaks-based combinatorial search below.
  }

  const int ksz = std::min<int>(20, static_cast<int>(total_points / 12));

  // Windowed error per point (DoFitLines). The circular index arithmetic is
  // done with conditional subtraction rather than `%`: the offsets are always
  // within one period, so a compare-and-subtract is exact, and this loop nest
  // was issuing roughly a dozen integer divisions per boundary point.
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

  // Peak detection: is_peak = my_error > before_error && my_error > after_error;
  // peak.error = -my_error (so ascending sort by error puts the strongest
  // peaks first).
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

  // ComputeM0M1Fit: cache line fits for segment (m0, m1) pairs that can ever
  // appear as the "first segment" of a valid 4-combination (m0 < 7, m1 < 8).
  constexpr size_t kCacheDim = kNMaxima - 3;  // 7
  double error_m0_m1[kCacheDim][kCacheDim];
  double lineparams23_m0_m1[kCacheDim][kCacheDim][2];
  for (auto &row : error_m0_m1) std::fill(std::begin(row), std::end(row),
                                          std::numeric_limits<double>::max());

  for (size_t m0 = 0; m0 < kCacheDim; ++m0) {
    for (size_t m1 = m0 + 1; m1 < kNMaxima - 2; ++m1) {
      if (m1 >= K) continue;  // stays max
      double err01, mse01;
      FitLine(ReadMomentsWindow(cs, total_points, point_indices[m0], point_indices[m1]), nullptr,
             lineparams23_m0_m1[m0][m1 - 1], &err01, &mse01);
      if (mse01 > config.max_line_fit_mse) {
        err01 = std::numeric_limits<double>::max();
      }
      error_m0_m1[m0][m1 - 1] = err01;
    }
  }

  // Brute force all C(10, 4) = 210 combinations of increasing maxima
  // indices, keeping the minimum total error (DoFitQuads).
  double best_error = std::numeric_limits<double>::max();
  uint32_t best[4] = {0, 0, 0, 0};

  for (uint32_t m0 = 0; m0 < kCacheDim; ++m0) {
    for (uint32_t m1 = m0 + 1; m1 < kNMaxima - 2; ++m1) {
      const double errm0m1 = error_m0_m1[m0][m1 - 1];
      if (errm0m1 == std::numeric_limits<double>::max()) continue;
      const double *paramsm0m123 = lineparams23_m0_m1[m0][m1 - 1];

      for (uint32_t m2 = m1 + 1; m2 < kNMaxima - 1; ++m2) {
        if (m2 >= K) break;
        double errm1m2, msem1m2;
        double paramsm1m223[2];
        FitLine(ReadMomentsWindow(cs, total_points, point_indices[m1], point_indices[m2]), nullptr,
               paramsm1m223, &errm1m2, &msem1m2);
        if (msem1m2 > config.max_line_fit_mse) continue;

        const double dot = paramsm0m123[0] * paramsm1m223[0] + paramsm0m123[1] * paramsm1m223[1];
        if (std::fabs(dot) > config.cos_critical_rad) continue;

        for (uint32_t m3 = m2 + 1; m3 < kNMaxima; ++m3) {
          if (m3 >= K) break;

          double errm2m3, msem2m3;
          FitLine(ReadMomentsWindow(cs, total_points, point_indices[m2], point_indices[m3]), nullptr,
                 nullptr, &errm2m3, &msem2m3);
          if (msem2m3 > config.max_line_fit_mse) continue;

          double errm3m0, msem3m0;
          FitLine(ReadMomentsWindow(cs, total_points, point_indices[m3], point_indices[m0]), nullptr,
                 nullptr, &errm3m0, &msem3m0);
          if (msem3m0 > config.max_line_fit_mse) continue;

          const double total = errm0m1 + errm1m2 + errm2m3 + errm3m0;
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

}  // namespace

QuadDecode::QuadDecode(const DetectorConfig &config)
    : config_(config),
      pool_(std::make_unique<WorkerPool>(ResolveThreadCount(config.cpu_threads))) {
  config_.quad_fit_method = ResolveQuadFitMethod(config_.quad_fit_method);
}

std::vector<DetectedQuad> QuadDecode::Decode(const std::vector<MinMaxExtentsGpu> &selected_extents,
                                             const std::vector<RawLineFitPoint> &line_fit_points) const {
  std::vector<DetectedQuad> output;

  // Group line_fit_points into contiguous per-blob spans (the array is
  // sorted by (blob_index, theta) ascending, so each blob's points form one
  // contiguous run). Cheap, and inherently serial.
  struct Span {
    size_t begin;
    size_t end;
  };
  std::vector<Span> spans;
  {
    size_t i = 0;
    while (i < line_fit_points.size()) {
      const uint32_t blob_index = line_fit_points[i].blob_index;
      size_t j = i;
      while (j < line_fit_points.size() && line_fit_points[j].blob_index == blob_index) ++j;
      spans.push_back({i, j});
      i = j;
    }
  }

  // Fit every blob independently - this is where essentially all of the CPU
  // tail's time goes. Each entry is written by exactly one task, so no
  // synchronization is needed beyond the pool's own.
  std::vector<FitQuadResult> per_span(spans.size());
  pool_->ParallelFor(spans.size(), [&](size_t s) {
    per_span[s] = FitQuadForBlob(config_, line_fit_points, spans[s].begin, spans[s].end);
  });

  last_dp_stats_ = DpStats{};
  for (const FitQuadResult &fq : per_span) {
    if (fq.dp_attempted) {
      ++last_dp_stats_.attempts;
      if (!fq.dp_used) ++last_dp_stats_.fallbacks;
    }
  }

  // Collect survivors in span order, NOT completion order, so the result is
  // bit-identical to the serial version regardless of thread scheduling.
  std::vector<FitQuadResult> fit_quads;
  fit_quads.reserve(per_span.size());
  for (const FitQuadResult &fq : per_span) {
    if (fq.valid) fit_quads.push_back(fq);
  }

  // UpdateFitQuads: refit lines from the moments, intersect adjacent lines
  // to get corners, and apply geometric sanity checks.
  const double min_tag_width =
      std::max(3.0, config_.tag_width / static_cast<double>(config_.decimation));
  for (const FitQuadResult &quad : fit_quads) {
    double lines[4][4];
    bool line_ok = true;
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
    if (bad_determinant) continue;
    (void)line_ok;

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
      if (area < 0.95 * min_tag_width * min_tag_width) continue;
    }

    // Reject quads whose cumulative angle change isn't consistent with a
    // convex, consistently-wound quadrilateral.
    bool reject = false;
    for (int i = 0; i < 4 && !reject; ++i) {
      int i0 = i, i1 = (i + 1) & 3, i2 = (i + 2) & 3;
      double dx1 = corners[i1][0] - corners[i0][0];
      double dy1 = corners[i1][1] - corners[i0][1];
      double dx2 = corners[i2][0] - corners[i1][0];
      double dy2 = corners[i2][1] - corners[i1][1];
      double cos_dtheta = (dx1 * dx2 + dy1 * dy2) /
                         std::sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
      if (std::fabs(cos_dtheta) > config_.cos_critical_rad || dx1 * dy2 < dy1 * dx2) {
        reject = true;
      }
    }
    if (reject) continue;

    // AdjustPixelCenters, generalized from the fixed-2x-decimation original:
    // pixel = (c - 0.5) * decimation + 0.5. The two 0.5 offsets have
    // different origins and neither scales with decimation:
    //   - the leading "- 0.5" undoes a constant +1 shift ComputeLineFitPoint
    //     (sort_points_local_body.glsl) applies to every point's x2/y2
    //     before any moment is accumulated - that shift happens entirely
    //     within the DECIMATED grid's own index space, so it is exactly
    //     half a DECIMATED pixel regardless of how coarse that grid is.
    //   - the trailing "+ 0.5" converts from a pixel-INDEX coordinate (0 at
    //     the first pixel) to the caller's pixel-CENTER convention, applied
    //     AFTER scaling to full resolution - so it is exactly half a
    //     FULL-RESOLUTION pixel, independent of decimation.
    // Sanity check: at decimation=1 (no downsampling), the formula must
    // reduce to the identity (corners[] is already in full-resolution
    // units) - (c - 0.5) * 1 + 0.5 = c, confirming both constants are right
    // where they are. Verified empirically against upstream apriltag's own
    // (integer) quad_decimate at 1x/2x/4x - see apriltag_vulkan_validate.
    const double decimation = static_cast<double>(config_.decimation);
    DetectedQuad out;
    for (int i = 0; i < 4; ++i) {
      out.p[i][0] = (corners[i][0] - 0.5) * decimation + 0.5;
      out.p[i][1] = (corners[i][1] - 0.5) * decimation + 0.5;
    }
    output.push_back(out);
  }

  (void)selected_extents;  // Not needed beyond what's already folded into blob_index grouping.
  return output;
}

}  // namespace apriltag_vulkan
