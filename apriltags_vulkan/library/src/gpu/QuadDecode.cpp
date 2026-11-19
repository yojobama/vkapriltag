#include "vkapriltag/gpu/QuadDecode.h"

#include "vkapriltag/common/Options.h"

#include "vkapriltag/common/Profiler.h"

#include "Geometry.h"
#include "vkapriltag/gpu/QuadFitF32.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>

namespace apriltag_vulkan {
namespace {

// Largest bounding box, in full-resolution pixels, of a blob handed on for the small-tag rescue.
constexpr double kMaxRescueBox = 40.0;
constexpr double kRescueMergeGap = 8.0;
// Mean edge gradient a small blob needs to be refitted (noise blobs stay under 40, tag edges are above 70).
constexpr double kMinRescueGradient = 60.0;
// With the rescue on, blobs of fewer points than this and a mean gradient below the second value are dropped.
constexpr size_t kRescueFlatPoints = 64;
constexpr double kRescueFlatGradient = 25.0;
constexpr size_t kMaxRescueBoxes = 64;
// Most full-resolution refits per frame, taken from the candidates scored nearest a small tag.
constexpr size_t kMaxRescueRefits = 4;
constexpr double kRescueIdealSide = 14.0;
constexpr double kRescueGradientWeight = 0.05;

// Per-blob quad fit result (combinatorial search or DP seeding).
struct FitQuadResult {
  bool valid = false;
  // Set whenever DP was tried (quad_fit_method == kDp and >= 4 points); dp_used only if it
  // produced the accepted result. For QuadDecode::last_dp_stats().
  bool dp_attempted = false;
  bool dp_used = false;
  uint32_t indices[4] = {};
  // The sides' origin is the blob's first point.
  int32_t x0 = 0, y0 = 0;
  quadf32::Sides f32_sides;
};

// Index of the largest value(i) over [begin, end), the lowest such index on ties, folded into
// (best, best_idx) with ties again going to the lower index: a vectorisable maximum pass, then a
// search for its first position.
template <typename T, typename F>
inline void ArgMax(size_t begin, size_t end, F value, T &best, size_t &best_idx) {
  if (begin >= end) return;
  T top = T(-1);
  for (size_t i = begin; i < end; ++i) {
    const T v = value(i);
    top = v > top ? v : top;
  }
  if (top < best || (top == best && begin > best_idx)) return;
  size_t i = begin;
  while (value(i) != top) ++i;
  best = top;
  best_idx = i;
}

// A blob's points unpacked into separate arrays.
struct UnpackedPoints {
  const int32_t *x, *y, *w;
};

// Seeds 4 corner indices: the two mutually-farthest points (2-pass approximation), then the
// point of maximum perpendicular deviation on each arc. Returns false if the blob does not look
// like a clean quad (too few points, degenerate diameter, empty arc, or no real deviation).
// Coordinates are below 2^14, so squared distances and cross products fit in int32.
bool FindDpCornerIndices(const UnpackedPoints &pts, size_t n, uint32_t out_indices[4]) {
  if (n < 8) return false;

  auto sq_dist = [&pts](size_t a, size_t b) {
    const int32_t dx = pts.x[a] - pts.x[b], dy = pts.y[a] - pts.y[b];
    return dx * dx + dy * dy;
  };

  int32_t best_d = -1;
  size_t a_idx = 0;
  ArgMax(1, n, [&](size_t i) { return sq_dist(0, i); }, best_d, a_idx);
  best_d = -1;
  size_t b_idx = 0;
  ArgMax(0, n, [&](size_t i) { return sq_dist(a_idx, i); }, best_d, b_idx);
  if (a_idx == b_idx) return false;
  const double diameter_sq = static_cast<double>(sq_dist(a_idx, b_idx));
  if (diameter_sq < 1e-6) return false;

  const size_t lo = std::min(a_idx, b_idx), hi = std::max(a_idx, b_idx);

  const int32_t ax = pts.x[a_idx], ay = pts.y[a_idx];
  const int32_t abx = pts.x[b_idx] - ax, aby = pts.y[b_idx] - ay;
  auto perp = [&](size_t i) {
    const int32_t cross = abx * (pts.y[i] - ay) - aby * (pts.x[i] - ax);
    return cross < 0 ? -cross : cross;
  };

  size_t c1_idx = n, c2_idx = n;  // n is not a valid index; doubles as "not found"
  int32_t best_perp1 = -1, best_perp2 = -1;
  ArgMax(lo + 1, hi, perp, best_perp1, c1_idx);
  ArgMax(0, lo, perp, best_perp2, c2_idx);
  ArgMax(hi + 1, n, perp, best_perp2, c2_idx);
  if (c1_idx == n || c2_idx == n) return false;

  // Reject degenerate shapes: both corners must deviate from the diameter line (threshold is
  // scaled by diameter_sq^2).
  const double min_perp = 0.0001 * diameter_sq * diameter_sq;
  const double p1 = static_cast<double>(best_perp1), p2 = static_cast<double>(best_perp2);
  if (p1 * p1 < min_perp || p2 * p2 < min_perp) return false;

  // Circular order: lo -> c1 -> hi -> c2. c1 lies between lo and hi, c2 outside [lo, hi].
  out_indices[0] = static_cast<uint32_t>(lo);
  out_indices[1] = static_cast<uint32_t>(c1_idx);
  out_indices[2] = static_cast<uint32_t>(hi);
  out_indices[3] = static_cast<uint32_t>(c2_idx);
  return true;
}

// The fit of one blob: DP seeding on the four arcs, falling back to the peaks search.
FitQuadResult FitQuadForBlob(const DetectorConfig &config, std::span<const RawLineFitPoint> points, size_t begin,
                             size_t end, QuadFitScratch &scratch) {
  FitQuadResult result;
  const size_t n = end - begin;
  // A point's W is at most 1023 and sums are 32-bit: far more points than any tag border.
  if (n < 4 || n > INT32_MAX / 1023) return result;
  scratch.px.resize(n + 8);
  scratch.py.resize(n + 8);
  scratch.pw.resize(n + 8);
  quadf32::Unpack(points.data() + begin, n, scratch.px.data(), scratch.py.data(), scratch.pw.data());
  scratch.unpacked_begin = begin;
  scratch.unpacked_n = n;
  const quadf32::Points pts{scratch.px.data(), scratch.py.data(), scratch.pw.data()};
  result.x0 = pts.x[0];
  result.y0 = pts.y[0];
  if (config.quad_fit_method == DetectorConfig::QuadFitMethod::kDp) {
    result.dp_attempted = true;
    uint32_t idx[4];
    const UnpackedPoints unpacked{pts.x, pts.y, pts.w};
    if (FindDpCornerIndices(unpacked, n, idx) && quadf32::FitArcs(pts, n, idx, config.max_line_fit_mse, &result.f32_sides)) {
      result.valid = result.dp_used = true;
      for (int k = 0; k < 4; ++k) result.indices[k] = idx[k];
      return result;
    }
  }
  result.valid = quadf32::FitPeaks(config, pts, n, scratch.f32, result.indices, &result.f32_sides);
  return result;
}

// Smallest bounding-box side, in half-pixel units, of a blob worth a hull fit.
constexpr int32_t kMinHullExtent = 40;

// Quad fit on the convex hull of a blob's boundary points: the four sides are the hull's long edges,
// which skips a corner an occluder has covered, and each side is refitted on the points near its edge.
FitQuadResult FitQuadHull(const DetectorConfig &config, std::span<const RawLineFitPoint> points,
                          size_t begin, size_t end, QuadFitScratch &scratch) {
  FitQuadResult result;
  const size_t n = end - begin;
  if (n < 600 / config.decimation || n > INT32_MAX / 1023) return result;
  auto &pts = scratch.hull_points;
  auto &hull = scratch.hull;
  auto &g = scratch.geom;
  scratch.px.resize(n + 8);
  scratch.py.resize(n + 8);
  scratch.pw.resize(n + 8);
  quadf32::Unpack(points.data() + begin, n, scratch.px.data(), scratch.py.data(), scratch.pw.data());
  scratch.unpacked_begin = begin;
  scratch.unpacked_n = n;
  const quadf32::Points up{scratch.px.data(), scratch.py.data(), scratch.pw.data()};
  int32_t bounds[4];
  quadf32::BoundsAndKeys(up, n, bounds, nullptr);
  {
    const int32_t long_side = std::max(bounds[1] - bounds[0], bounds[3] - bounds[2]),
                  short_side = std::min(bounds[1] - bounds[0], bounds[3] - bounds[2]);
    if (long_side < kMinHullExtent || short_side * 5 < long_side * 3) return result;
  }
  // Only the lowest and highest point of each column can be a hull vertex.
  quadf32::ColumnCandidates(up, n, bounds[0], bounds[1], g, pts);
  if (pts.size() < 8) {
    // Too few candidates to tell: the sorted unique points.
    auto &keys = scratch.hull_keys;
    keys.resize(n);
    for (size_t i = 0; i < n; ++i) keys[i] = static_cast<uint32_t>(up.x[i]) << 14 | static_cast<uint32_t>(up.y[i]);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    pts.clear();
    for (const uint32_t key : keys) pts.emplace_back(static_cast<int32_t>(key >> 14), static_cast<int32_t>(key & 0x3FFF));
    if (pts.size() < 8) return result;
  }

  // Andrew's monotone chain.
  auto cross = [](const std::pair<int32_t, int32_t> &o, const std::pair<int32_t, int32_t> &a,
                  const std::pair<int32_t, int32_t> &b) {
    return static_cast<int64_t>(a.first - o.first) * (b.second - o.second) -
           static_cast<int64_t>(a.second - o.second) * (b.first - o.first);
  };
  if (hull.size() < 2 * pts.size()) hull.resize(2 * pts.size());
  const size_t m = geom::MonotoneChain(pts.data(), pts.size(), hull.data(), cross);
  if (m < 4) return result;

  // Simplify to a few vertices, anchored at the vertex farthest from the first.
  g.hx.resize(m + 1);
  g.hy.resize(m + 1);
  for (size_t i = 0; i < m; ++i) g.hx[i] = hull[i].first, g.hy[i] = hull[i].second;
  g.hx[m] = g.hx[0];
  g.hy[m] = g.hy[0];
  const size_t far = quadf32::FarVertex(g.hx.data(), g.hy.data(), m);
  const double far_d = std::hypot(static_cast<double>(g.hx[far] - g.hx[0]), static_cast<double>(g.hy[far] - g.hy[0]));
  const double min_side_x2 = std::max(3.0, config.tag_width / static_cast<double>(config.decimation));
  if (far_d < min_side_x2) return result;
  g.keep.assign(m, 0);
  g.keep[0] = 1;
  g.keep[far] = 1;
  constexpr double kEps = 2.4;  // 1.2 pixels, in the points' half-pixel units
  quadf32::SimplifyHull(g.hx.data(), g.hy.data(), 0, far, kEps, g.keep.data());
  quadf32::SimplifyHull(g.hx.data(), g.hy.data(), far, m, kEps, g.keep.data());
  std::array<std::pair<double, double>, 16> v;
  size_t k = 0;
  for (size_t i = 0; i < m; ++i) {
    if (!g.keep[i]) continue;
    if (k == 10) return result;
    v[k++] = {static_cast<double>(g.hx[i]), static_cast<double>(g.hy[i])};
  }
  if (k < 4) return result;

  std::array<double, 16> len, ux, uy;
  double perimeter = 0;
  for (size_t i = 0; i < k; ++i) {
    const auto &a = v[i], &b = v[(i + 1) % k];
    len[i] = std::hypot(b.first - a.first, b.second - a.second);
    ux[i] = len[i] > 0 ? (b.first - a.first) / len[i] : 0;
    uy[i] = len[i] > 0 ? (b.second - a.second) / len[i] : 0;
    perimeter += len[i];
  }

  // The four edges, in order, of greatest total length whose neighbours are near-perpendicular and
  // whose intersections lie close to the edges' ends.
  size_t best[4] = {};
  const double best_len = quadf32::ChooseEdges(v.data(), k, len.data(), ux.data(), uy.data(), best);
  if (best_len < 0.6 * perimeter) return result;

  result.x0 = points[begin].x2();
  result.y0 = points[begin].y2();
  constexpr double kBand = 3.0;  // 1.5 pixels
  result.f32_sides.x0 = result.x0;
  result.f32_sides.y0 = result.y0;
  for (int s = 0; s < 4; ++s) {
    const size_t e = best[s];
    const quadf32::HullEdge edge{static_cast<int32_t>(v[e].first), static_cast<int32_t>(v[e].second), ux[e], uy[e], len[e],
                                 std::max(10.0, 0.25 * len[e])};
    quadf32::Centred &side = result.f32_sides.side[s];
    if (!quadf32::FitHullSide(up, n, result.x0, result.y0, edge, kBand, scratch.f32.hull, &side)) return result;
    if (quadf32::FitLine(side).mse > config.max_line_fit_mse) return result;
  }
  result.valid = true;
  return result;
}

// Resolves quad_fit_method: APRILTAG_VK_QUADFIT (dp or peaks) overrides the configured value.
DetectorConfig::QuadFitMethod ResolveQuadFitMethod(const Options &env,
                                                   DetectorConfig::QuadFitMethod configured) {
  switch (env.Choice("APRILTAG_VK_QUADFIT", {"dp", "peaks"})) {
    case 0: return DetectorConfig::QuadFitMethod::kDp;
    case 1: return DetectorConfig::QuadFitMethod::kPeaks;
    default: return configured;
  }
}

// UpdateFitQuads for one blob: refits the lines from the moments, intersects adjacent lines to
// get corners and applies the geometric checks. Returns false if the quad is rejected.
bool UpdateFitQuad(const DetectorConfig &config, const FitQuadResult &quad, DetectedQuad *out) {
  const double min_tag_width =
      std::max(3.0, config.tag_width / static_cast<double>(config.decimation));
  double lines[4][4];
  for (int i = 0; i < 4; ++i) {
    quadf32::LineParams(quad.f32_sides.side[i], quad.x0, quad.y0, lines[i]);
  }

  double corners[4][2];
  bool bad_determinant = false;
  for (int i = 0; i < 4; ++i) {
    double det;
    geom::LineCorner(lines[i], lines[(i + 1) & 3], &corners[i][0], &corners[i][1], &det);
    if (std::fabs(det) < 0.001) {
      bad_determinant = true;
      break;
    }
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

constexpr double kSuspectFraction = 0.02;

// Most hull fits attempted in one frame.
constexpr size_t kMaxHullAttempts = 6;

// Whether every corner of `a` is within `tolerance` pixels of a corner of `b`.
bool SameCorners(const DetectedQuad &a, const DetectedQuad &b, double tolerance) {
  for (int i = 0; i < 4; ++i) {
    double nearest = 1e30;
    for (int j = 0; j < 4; ++j) nearest = std::min(nearest, std::hypot(a.p[i][0] - b.p[j][0], a.p[i][1] - b.p[j][1]));
    if (nearest > tolerance) return false;
  }
  return true;
}

// Whether more than `fraction` of the blob's points are farther than `tolerance` (half-pixel units) from every
// side of the quad (full-resolution pixels); unpacks the blob [begin, end) when the slot does not hold it.
bool HasOutliers(std::span<const RawLineFitPoint> points, size_t begin, size_t end, const DetectedQuad &quad,
                 double decimation, double tolerance, double fraction, QuadFitScratch &scratch) {
  const size_t n = end - begin;
  if (scratch.unpacked_begin != begin || scratch.unpacked_n != n) {
    scratch.px.resize(n + 8);
    scratch.py.resize(n + 8);
    scratch.pw.resize(n + 8);
    quadf32::Unpack(points.data() + begin, n, scratch.px.data(), scratch.py.data(), scratch.pw.data());
    scratch.unpacked_begin = begin;
    scratch.unpacked_n = n;
  }
  double c[4][2];
  for (int i = 0; i < 4; ++i) {
    c[i][0] = quad.p[i][0] * 2.0 / decimation;
    c[i][1] = quad.p[i][1] * 2.0 / decimation;
  }
  return quadf32::HasOutliers({scratch.px.data(), scratch.py.data(), scratch.pw.data()}, n, c, tolerance, fraction);
}

}  // namespace

std::shared_ptr<WorkerPool> MakeWorkerPool(const DetectorConfig &config) {
  return std::make_shared<WorkerPool>(ResolveThreadCount(config.cpu_threads), "cpu-worker",
                                      ResolveWorkerPoolOptions(config.PoolOptions()));
}

QuadDecode::QuadDecode(const DetectorConfig &config, std::shared_ptr<WorkerPool> pool)
    : config_(config),
      pool_(pool ? std::move(pool) : MakeWorkerPool(config)),
      scratch_(pool_->threads()) {
  const Options env = Options::FromEnvironment();
  config_.quad_fit_method = ResolveQuadFitMethod(env, config_.quad_fit_method);
  config_.small_tag_rescue = ResolveSmallTagRescue(env, config_.small_tag_rescue);
  config_.hull_quad = env.Bool("APRILTAG_VK_HULL_QUAD", config_.hull_quad);
}

std::vector<DetectedQuad> QuadDecode::Decode(
    std::span<const RawLineFitPoint> line_fit_points) const {
  VKA_PROF_SCOPE("quad.decode_all");
  for (QuadFitScratch &s : scratch_) s.unpacked_n = SIZE_MAX;
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
    bool dp_attempted = false, dp_used = false, has_quad = false, has_hull = false, hull_candidate = false;
    DetectedQuad quad, hull_quad;
    bool has_box = false;
    DetectedQuad box;
  };
  std::vector<SpanResult> per_span(spans.size());
  pool_->ParallelFor(spans.size(), [&](size_t s, unsigned slot) {
    prof::Scope fit_scope("quad.fit_blob");
    fit_scope.set_arg(static_cast<int64_t>(spans[s].end - spans[s].begin));
    SpanResult &r = per_span[s];
    const bool rescue = config_.small_tag_rescue && config_.decimation > 1;
    const size_t n_points = spans[s].end - spans[s].begin;
    int32_t xlo = INT32_MAX, ylo = INT32_MAX, xhi = 0, yhi = 0;
    double mean_w = 0.0;
    if (rescue) {
      int64_t w_sum = 0;
      for (size_t k = spans[s].begin; k < spans[s].end; ++k) {
        xlo = std::min(xlo, line_fit_points[k].x2());
        xhi = std::max(xhi, line_fit_points[k].x2());
        ylo = std::min(ylo, line_fit_points[k].y2());
        yhi = std::max(yhi, line_fit_points[k].y2());
        w_sum += line_fit_points[k].W();
      }
      mean_w = static_cast<double>(w_sum) / static_cast<double>(n_points);
      if (n_points < kRescueFlatPoints && mean_w < kRescueFlatGradient) return;
    }
    FitQuadResult fq;
    if (!(rescue && n_points < config_.min_cluster_pixels)) {
      fq = FitQuadForBlob(config_, line_fit_points, spans[s].begin, spans[s].end, scratch_[slot]);
    }
    r.dp_attempted = fq.dp_attempted;
    r.dp_used = fq.dp_used;
    r.has_quad = fq.valid && UpdateFitQuad(config_, fq, &r.quad);
    r.quad.gradient = static_cast<float>(mean_w);
    const double d = 0.5 * config_.decimation;
    if (rescue && !r.has_quad && mean_w >= kMinRescueGradient && (xhi - xlo) * d <= kMaxRescueBox &&
        (yhi - ylo) * d <= kMaxRescueBox) {
      r.has_box = true;
      r.box.coarse = true;
      r.box.gradient = static_cast<float>(mean_w);
      const double xs[4] = {xlo * d, xhi * d, xhi * d, xlo * d}, ys[4] = {ylo * d, ylo * d, yhi * d, yhi * d};
      for (int k = 0; k < 4; ++k) r.box.p[k][0] = xs[k], r.box.p[k][1] = ys[k];
    }
    if (config_.hull_quad) {
      const bool large = !r.has_quad || std::hypot(r.quad.p[0][0] - r.quad.p[2][0], r.quad.p[0][1] - r.quad.p[2][1]) >=
                                            kMinHullExtent * config_.decimation * 0.5;
      r.hull_candidate = large && (!r.has_quad || HasOutliers(line_fit_points, spans[s].begin, spans[s].end, r.quad, config_.decimation, 4.0,
                                                  kSuspectFraction, scratch_[slot]));
    }
  });

  // The hull fit runs on the largest candidates only, so a frame full of failing blobs stays cheap.
  if (config_.hull_quad) {
    std::vector<size_t> candidates;
    for (size_t s = 0; s < spans.size(); ++s) {
      if (per_span[s].hull_candidate) candidates.push_back(s);
    }
    if (candidates.size() > kMaxHullAttempts) {
      std::nth_element(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(kMaxHullAttempts),
                       candidates.end(),
                       [&](size_t a, size_t b) {
                         const size_t na = spans[a].end - spans[a].begin, nb = spans[b].end - spans[b].begin;
                         return na != nb ? na > nb : a < b;
                       });
      candidates.resize(kMaxHullAttempts);
    }
    pool_->ParallelFor(candidates.size(), [&](size_t c, unsigned slot) {
      const size_t s = candidates[c];
      SpanResult &r = per_span[s];
      const FitQuadResult hq =
          FitQuadHull(config_, line_fit_points, spans[s].begin, spans[s].end, scratch_[slot]);
      r.has_hull = hq.valid && UpdateFitQuad(config_, hq, &r.hull_quad);
      if (r.has_hull && r.has_quad && SameCorners(r.quad, r.hull_quad, 2.0)) r.has_hull = false;
    });
  }

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
    if (r.has_hull) output.push_back(r.hull_quad);
  }

  // Boxes of fragments of one small tag are merged, so each tag is refitted once.
  std::vector<std::array<double, 5>> boxes;
  for (const SpanResult &r : per_span) {
    if (r.has_box && !r.has_quad) boxes.push_back({r.box.p[0][0], r.box.p[0][1], r.box.p[2][0], r.box.p[2][1], r.box.gradient});
  }
  for (bool merged = !boxes.empty() && boxes.size() <= kMaxRescueBoxes; merged;) {
    merged = false;
    for (size_t a = 0; a < boxes.size() && !merged; ++a) {
      for (size_t b = a + 1; b < boxes.size() && !merged; ++b) {
        const std::array<double, 5> &u = boxes[a], &v = boxes[b];
        const double x0 = std::min(u[0], v[0]), y0 = std::min(u[1], v[1]);
        const double x1 = std::max(u[2], v[2]), y1 = std::max(u[3], v[3]);
        const bool near = v[0] <= u[2] + kRescueMergeGap && u[0] <= v[2] + kRescueMergeGap &&
                          v[1] <= u[3] + kRescueMergeGap && u[1] <= v[3] + kRescueMergeGap;
        if (near && x1 - x0 <= kMaxRescueBox && y1 - y0 <= kMaxRescueBox) {
          boxes[a] = {x0, y0, x1, y1, std::max(u[4], v[4])};
          boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(b));
          merged = true;
        }
      }
    }
  }
  if (boxes.size() > kMaxRescueBoxes) boxes.resize(kMaxRescueBoxes);
  // A box whose centre lies inside a fitted quad is a data cell of a larger tag, not a small tag.
  const size_t n_quads = output.size();
  const bool rescue_enabled = config_.small_tag_rescue && config_.decimation > 1;
  auto inside_quad = [&](double x, double y) {
    for (size_t i = 0; i < n_quads; ++i) {
      const DetectedQuad &q = output[i];
      bool pos = false, neg = false;
      for (int k = 0; k < 4; ++k) {
        const double *a = q.p[k], *b = q.p[(k + 1) & 3];
        const double c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        (c >= 0 ? pos : neg) = true;
      }
      if (!(pos && neg)) return true;
    }
    return false;
  };
  struct Candidate {
    double score;
    size_t index;
    bool is_box;
  };
  std::vector<Candidate> candidates;
  std::vector<DetectedQuad> box_quads;
  auto score = [](double side, double gradient) { return -std::abs(side - kRescueIdealSide) - gradient * kRescueGradientWeight; };
  if (rescue_enabled) {
    for (size_t i = 0; i < n_quads; ++i) {
      const DetectedQuad &q = output[i];
      if (q.coarse || q.gradient < kMinRescueGradient) continue;
      double lo_x = 1e30, hi_x = -1e30, lo_y = 1e30, hi_y = -1e30;
      for (int k = 0; k < 4; ++k) {
        lo_x = std::min(lo_x, q.p[k][0]), hi_x = std::max(hi_x, q.p[k][0]);
        lo_y = std::min(lo_y, q.p[k][1]), hi_y = std::max(hi_y, q.p[k][1]);
      }
      if (hi_x - lo_x <= kMaxRescueBox && hi_y - lo_y <= kMaxRescueBox) {
        candidates.push_back({score(std::max(hi_x - lo_x, hi_y - lo_y), q.gradient), i, false});
      }
    }
  }
  for (const std::array<double, 5> &bx : boxes) {
    if (inside_quad(0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3]))) continue;
    DetectedQuad box;
    box.coarse = true;
    const double xs[4] = {bx[0], bx[2], bx[2], bx[0]}, ys[4] = {bx[1], bx[1], bx[3], bx[3]};
    for (int k = 0; k < 4; ++k) box.p[k][0] = xs[k], box.p[k][1] = ys[k];
    candidates.push_back({score(std::max(bx[2] - bx[0], bx[3] - bx[1]), bx[4]), box_quads.size(), true});
    box_quads.push_back(box);
  }
  // Only the most tag-like candidates are refitted from the full-resolution image.
  if (candidates.size() > kMaxRescueRefits) {
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
      return a.score != b.score ? a.score > b.score : (a.is_box != b.is_box ? !a.is_box : a.index < b.index);
    });
    for (size_t c = kMaxRescueRefits; c < candidates.size(); ++c) {
      if (!candidates[c].is_box) output[candidates[c].index].gradient = 0.0f;
    }
    candidates.resize(kMaxRescueRefits);
  }
  for (const Candidate &c : candidates) {
    if (c.is_box) output.push_back(box_quads[c.index]);
  }

  return output;
}

}  // namespace apriltag_vulkan
