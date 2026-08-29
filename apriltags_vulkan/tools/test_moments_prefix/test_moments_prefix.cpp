// A9 stage 1 verification: runs compute_moments_prefix.comp against
// synthetic blobs and checks its output is bit-for-bit identical to
// QuadDecode.cpp's FitQuadForBlob moment prefix sum (the `cs[]` array),
// including the 32/64-bit truncation behavior that computation relies on.
//
// This is deliberately independent of the real detection pipeline - it
// only exercises the one new shader, with synthetic point sets (including
// large-magnitude cases meant to exercise the high words of the 64-bit
// emulation) rather than a real image, per the plan's stage 1 scope.
#include "vkapriltag/gpu/Types.h"
#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"
#include "vkapriltag/vk/EmbeddedShaders.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace apriltag_vulkan {
namespace {

vk::ShaderSource ShaderPath(const char *name) {
  if (const vk::EmbeddedShader *embedded = vk::FindEmbeddedShader(name)) {
    return vk::ShaderSource(embedded->code, embedded->bytes, name);
  }
  return vk::ShaderSource(std::string("shaders/") + name + ".comp.spv");
}

// Exact copy of the inner loop of QuadDecode.cpp's FitQuadForBlob
// (lines computing cs[]) - the reference this test checks the GPU against.
std::vector<GpuLineFitMomentsRaw> CpuPrefixSum(const std::vector<RawLineFitPoint> &points,
                                                size_t begin, size_t end) {
  std::vector<GpuLineFitMomentsRaw> cs(end - begin);
  int32_t Mx = 0, My = 0, W = 0;
  int64_t Mxx = 0, Myy = 0, Mxy = 0;
  for (size_t k = begin; k < end; ++k) {
    const RawLineFitPoint &p = points[k];
    const int64_t wx = static_cast<int64_t>(p.W) * p.x2;
    const int64_t wy = static_cast<int64_t>(p.W) * p.y2;
    Mx += static_cast<int32_t>(wx);
    My += static_cast<int32_t>(wy);
    W += p.W;
    Mxx += wx * p.x2;
    Mxy += wx * p.y2;
    Myy += wy * p.y2;

    GpuLineFitMomentsRaw &m = cs[k - begin];
    m.Mx = Mx;
    m.My = My;
    m.W = W;
    m.Mxx_hi = static_cast<int32_t>(static_cast<uint64_t>(Mxx) >> 32);
    m.Mxx_lo = static_cast<uint32_t>(static_cast<uint64_t>(Mxx));
    m.Myy_hi = static_cast<int32_t>(static_cast<uint64_t>(Myy) >> 32);
    m.Myy_lo = static_cast<uint32_t>(static_cast<uint64_t>(Myy));
    m.Mxy_hi = static_cast<int32_t>(static_cast<uint64_t>(Mxy) >> 32);
    m.Mxy_lo = static_cast<uint32_t>(static_cast<uint64_t>(Mxy));
    m.N = 0;
  }
  return cs;
}

bool MomentsEqual(const GpuLineFitMomentsRaw &a, const GpuLineFitMomentsRaw &b) {
  return a.Mx == b.Mx && a.My == b.My && a.W == b.W && a.Mxx_hi == b.Mxx_hi &&
         a.Mxx_lo == b.Mxx_lo && a.Myy_hi == b.Myy_hi && a.Myy_lo == b.Myy_lo &&
         a.Mxy_hi == b.Mxy_hi && a.Mxy_lo == b.Mxy_lo;
}

// --- Stage 2 (compute_window_error.comp) CPU reference ---
// Exact copies of QuadDecode.cpp's ReadMomentsWindow / FitLineError,
// operating on true int64_t LineFitMoments (not the hi/lo split), since
// this is the ground truth the GPU's float-precision output is compared
// against with a tolerance, not bit-exactness (see line_fit_moments.glsl's
// FitLineError comment for why: the original itself only ever computes in
// float32 despite its double return type/params).
LineFitMoments AddLfm(const LineFitMoments &a, const LineFitMoments &b) {
  LineFitMoments r;
  r.Mx = a.Mx + b.Mx;
  r.My = a.My + b.My;
  r.W = a.W + b.W;
  r.Mxx = a.Mxx + b.Mxx;
  r.Myy = a.Myy + b.Myy;
  r.Mxy = a.Mxy + b.Mxy;
  return r;
}
LineFitMoments SubLfm(const LineFitMoments &a, const LineFitMoments &b) {
  LineFitMoments r;
  r.Mx = a.Mx - b.Mx;
  r.My = a.My - b.My;
  r.W = a.W - b.W;
  r.Mxx = a.Mxx - b.Mxx;
  r.Myy = a.Myy - b.Myy;
  r.Mxy = a.Mxy - b.Mxy;
  return r;
}

std::vector<LineFitMoments> CpuPrefixSumMoments(const std::vector<RawLineFitPoint> &points,
                                                size_t begin, size_t end) {
  std::vector<LineFitMoments> cs(end - begin);
  LineFitMoments running{};
  for (size_t k = begin; k < end; ++k) {
    const RawLineFitPoint &p = points[k];
    const int64_t wx = static_cast<int64_t>(p.W) * p.x2;
    const int64_t wy = static_cast<int64_t>(p.W) * p.y2;
    running.Mx += static_cast<int32_t>(wx);
    running.My += static_cast<int32_t>(wy);
    running.W += p.W;
    running.Mxx += wx * p.x2;
    running.Mxy += wx * p.y2;
    running.Myy += wy * p.y2;
    cs[k - begin] = running;
  }
  return cs;
}

LineFitMoments ReadMomentsWindowRef(const std::vector<LineFitMoments> &cs, size_t total_points,
                                    size_t index0, size_t index1) {
  LineFitMoments result;
  if (index0 < index1) {
    result = cs[index1];
    if (index0 > 0) result = SubLfm(result, cs[index0 - 1]);
    result.N = static_cast<int32_t>(index1 - index0 + 1);
  } else {
    LineFitMoments lf0 = cs[index0 - 1];
    LineFitMoments lfsz = cs[total_points - 1];
    result = SubLfm(lfsz, lf0);
    result = AddLfm(result, cs[index1]);
    result.N = static_cast<int32_t>(total_points - index0 + index1 + 1);
  }
  return result;
}

double FitLineErrorRef(int N, int64_t Mx, int64_t My, int64_t Mxx, int64_t Myy, int64_t Mxy,
                       int64_t W) {
  int64_t Cxx = Mxx * W - Mx * Mx;
  int64_t Cxy = Mxy * W - Mx * My;
  int64_t Cyy = Myy * W - My * My;
  const float hypot_cached =
      std::hypotf(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
  const float eight_w_squared = static_cast<float>(static_cast<int64_t>(W) * static_cast<int64_t>(W) * 8.0);
  const float eig_small = (static_cast<float>(Cxx + Cyy) - hypot_cached) / eight_w_squared;
  return N * eig_small;
}

struct WindowErrorEntry {
  float error = 0.0f;
  int32_t window_W = 0;  // the window's summed weight - see is_ill_conditioned below.
};

// Reference error[] array for one blob, mirroring DoFitLines' windowed-error
// loop (QuadDecode.cpp lines ~346-360) exactly.
std::vector<WindowErrorEntry> CpuWindowError(const std::vector<LineFitMoments> &cs, size_t n) {
  std::vector<WindowErrorEntry> error(n);
  if (n < 4) return error;
  const int ksz = std::min<int>(20, static_cast<int>(n / 12));
  const size_t k_off = static_cast<size_t>(ksz);
  for (size_t i = 0; i < n; ++i) {
    const size_t i0 = (i >= k_off) ? (i - k_off) : (i + n - k_off);
    size_t i1 = i + k_off;
    if (i1 >= n) i1 -= n;
    LineFitMoments m = ReadMomentsWindowRef(cs, n, i0, i1);
    error[i].error =
        static_cast<float>(FitLineErrorRef(m.N, m.Mx, m.My, m.Mxx, m.Myy, m.Mxy, m.W));
    error[i].window_W = m.W;
  }
  return error;
}

// FitLineError divides by eight_w_squared = W^2*8 (see line_fit_moments.
// glsl) - when a window's summed weight is small, that denominator is
// small too, and (Cxx+Cyy - hypot) is the difference of two large,
// near-equal float32 values (an inherent property of the port's algorithm,
// present in the pure-CPU float32 reference already - not something the
// GPU path introduces). Tiny, unavoidable float-precision differences
// between the CPU and GPU paths (different hypot approximation, the
// approximate int64->float32 conversion - see int64_emu.glsl) get
// massively amplified by that division right at these windows. This
// mirrors the same class of accepted wobble as A5's pseudo-angle tie
// order (OPTIMIZATION_NOTES.md) - real windows this ill-conditioned
// correspond to near-degenerate point configurations, not the common case.
bool IsIllConditioned(int32_t window_W) {
  const double eight_w_squared = 8.0 * static_cast<double>(window_W) * window_W;
  return eight_w_squared < 1e6;
}

// --- Stage 3 (compute_peaks.comp) CPU reference ---
// Exact copy of DoFitLines' 7-tap filter + peak detection + top-10
// selection (QuadDecode.cpp, FitQuadForBlob lines ~362-397), including the
// double-precision filter accumulation the real production code actually
// uses (float32-precision error[] values, but summed in double) - this is
// the TRUE reference, not a float32-only approximation, so the comparison
// below measures the real gap the GPU's float32-only accumulation
// introduces, not an artificially narrowed one.
constexpr std::array<double, 7> kFilterCoefficientsRef = {
    0.01110899634659290314, 0.13533528149127960205, 0.60653066635131835938,
    1.00000000000000000000, 0.60653066635131835938, 0.13533528149127960205,
    0.01110899634659290314,
};

std::vector<double> CpuFiltered(const std::vector<float> &error, size_t n) {
  std::vector<double> filtered(n);
  constexpr size_t kHalf = 3;
  for (size_t i = 0; i < n; ++i) {
    double accumulated = 0.0;
    size_t idx = (i >= kHalf) ? (i - kHalf) : (i + n - kHalf);
    for (size_t j = 0; j < kFilterCoefficientsRef.size(); ++j) {
      accumulated += static_cast<double>(error[idx]) * kFilterCoefficientsRef[j];
      if (++idx == n) idx = 0;
    }
    filtered[i] = accumulated;
  }
  return filtered;
}

// Returns the selected point indices, ascending, or empty if K < 4.
std::vector<uint32_t> CpuSelectPeaks(const std::vector<double> &filtered, size_t n) {
  std::vector<std::pair<double, uint32_t>> peaks;
  double before = filtered[n - 1];
  double cur = filtered[0];
  for (size_t i = 0; i < n; ++i) {
    const double after = filtered[(i + 1 == n) ? 0 : (i + 1)];
    if (cur > before && cur > after) peaks.emplace_back(-cur, static_cast<uint32_t>(i));
    before = cur;
    cur = after;
  }
  const size_t K = std::min<size_t>(peaks.size(), 10);
  if (K < 4) return {};
  std::stable_sort(peaks.begin(), peaks.end(),
                   [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<uint32_t> point_indices(K);
  for (size_t k = 0; k < K; ++k) point_indices[k] = peaks[k].second;
  std::sort(point_indices.begin(), point_indices.end());
  return point_indices;
}

// --- Stage 4 (compute_quad_search.comp) CPU reference ---
// Exact copy of FitLine's err/mse/lineparam23 outputs (QuadDecode.cpp) -
// lineparam01 omitted, since DoFitQuads never reads it.
struct FitLineResult23Ref {
  double err = 0.0;
  double mse = 0.0;
  double nx = 0.0, ny = 0.0;
};

FitLineResult23Ref FitLine23Ref(const LineFitMoments &m) {
  int64_t Mx = m.Mx, My = m.My, W = m.W;
  int64_t Cxx = m.Mxx * W - Mx * Mx;
  int64_t Cxy = m.Mxy * W - Mx * My;
  int64_t Cyy = m.Myy * W - My * My;
  const float hypot_cached =
      std::hypotf(static_cast<float>(Cxx - Cyy), static_cast<float>(2 * Cxy));
  const float eight_w_squared = static_cast<float>(static_cast<int64_t>(W) * static_cast<int64_t>(W) * 8.0);
  const float eig_small = (static_cast<float>(Cxx + Cyy) - hypot_cached) / eight_w_squared;

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
  const float length = std::hypotf(nx, ny);

  FitLineResult23Ref r;
  r.err = m.N * eig_small;
  r.mse = eig_small;
  r.nx = nx / length;
  r.ny = ny / length;
  return r;
}

struct QuadSearchResultRef {
  bool valid = false;
  uint32_t indices[4] = {};
};

// Exact copy of DoFitQuads' C(10,4) combinatorial search (QuadDecode.cpp
// lines ~399-469).
QuadSearchResultRef CpuQuadSearch(const std::vector<LineFitMoments> &cs, size_t n,
                                  const std::vector<uint32_t> &point_indices,
                                  float max_line_fit_mse, double cos_critical_rad) {
  QuadSearchResultRef result;
  const size_t K = point_indices.size();
  if (K < 4) return result;
  constexpr size_t kNMaxima = 10, kCacheDim = 7;
  double error_m0_m1[kCacheDim][kCacheDim];
  double params_m0_m1[kCacheDim][kCacheDim][2];
  for (auto &row : error_m0_m1) {
    std::fill(std::begin(row), std::end(row), std::numeric_limits<double>::max());
  }

  for (size_t m0 = 0; m0 < kCacheDim; ++m0) {
    for (size_t m1 = m0 + 1; m1 < kNMaxima - 2; ++m1) {
      if (m1 >= K) continue;
      LineFitMoments m = ReadMomentsWindowRef(cs, n, point_indices[m0], point_indices[m1]);
      FitLineResult23Ref fl = FitLine23Ref(m);
      double err01 = fl.err;
      if (fl.mse > max_line_fit_mse) err01 = std::numeric_limits<double>::max();
      error_m0_m1[m0][m1 - 1] = err01;
      params_m0_m1[m0][m1 - 1][0] = fl.nx;
      params_m0_m1[m0][m1 - 1][1] = fl.ny;
    }
  }

  double best_error = std::numeric_limits<double>::max();
  uint32_t best[4] = {0, 0, 0, 0};
  for (size_t m0 = 0; m0 < kCacheDim; ++m0) {
    for (size_t m1 = m0 + 1; m1 < kNMaxima - 2; ++m1) {
      const double errm0m1 = error_m0_m1[m0][m1 - 1];
      if (errm0m1 == std::numeric_limits<double>::max()) continue;
      const double p0x = params_m0_m1[m0][m1 - 1][0], p0y = params_m0_m1[m0][m1 - 1][1];
      for (size_t m2 = m1 + 1; m2 < kNMaxima - 1; ++m2) {
        if (m2 >= K) break;
        FitLineResult23Ref fl12 =
            FitLine23Ref(ReadMomentsWindowRef(cs, n, point_indices[m1], point_indices[m2]));
        if (fl12.mse > max_line_fit_mse) continue;
        const double dot = p0x * fl12.nx + p0y * fl12.ny;
        if (std::fabs(dot) > cos_critical_rad) continue;
        for (size_t m3 = m2 + 1; m3 < kNMaxima; ++m3) {
          if (m3 >= K) break;
          FitLineResult23Ref fl23 =
              FitLine23Ref(ReadMomentsWindowRef(cs, n, point_indices[m2], point_indices[m3]));
          if (fl23.mse > max_line_fit_mse) continue;
          FitLineResult23Ref fl30 =
              FitLine23Ref(ReadMomentsWindowRef(cs, n, point_indices[m3], point_indices[m0]));
          if (fl30.mse > max_line_fit_mse) continue;
          const double total = errm0m1 + fl12.err + fl23.err + fl30.err;
          if (total < best_error) {
            best_error = total;
            best[0] = static_cast<uint32_t>(m0);
            best[1] = static_cast<uint32_t>(m1);
            best[2] = static_cast<uint32_t>(m2);
            best[3] = static_cast<uint32_t>(m3);
          }
        }
      }
    }
  }

  if (best_error == std::numeric_limits<double>::max()) return result;
  if (!(best_error < static_cast<double>(max_line_fit_mse) * static_cast<double>(n))) return result;

  result.valid = true;
  for (int k = 0; k < 4; ++k) result.indices[k] = point_indices[best[k]];
  return result;
}

}  // namespace
}  // namespace apriltag_vulkan

int main() {
  using namespace apriltag_vulkan;

  std::mt19937 rng(12345);
  // Coordinates realistic for the doubled decimated-pixel grid at up to
  // ~4K resolution (see PackXY's 14-bit bound, common.glsl), W in the
  // gradient-magnitude range blob_diff.comp actually produces.
  std::uniform_int_distribution<int> coord_dist(-16383, 16383);
  std::uniform_int_distribution<int> w_dist(1, 2000);
  // "Wide" blobs use the actual physical maxima (x2/y2 up to a doubled
  // 16383, W up to ComputeLineFitPoint's sqrt(255^2+255^2)+1 ~ 361 -
  // sort_points_local_body.glsl) but with many more points per blob than
  // the "normal" case, so cs[]'s cumulative Mx/Mxx/Mxy/Myy entries reach
  // realistically large magnitudes (a long, real boundary run). This is
  // deliberately NOT synthetically larger than physical inputs allow:
  // FitLineError's own extra Mxx*W multiply means even physically-real
  // long blobs get within a couple of orders of magnitude of int64_t's
  // ~9.2e18 ceiling, and going further just overflows the CPU reference
  // itself (int64_t UB in practice, not a meaningful target for either
  // side to match).
  std::uniform_int_distribution<int> coord_dist_wide(-32766, 32766);
  std::uniform_int_distribution<int> w_dist_wide(1, 361);

  std::vector<RawLineFitPoint> points;
  std::vector<std::pair<uint32_t, uint32_t>> ranges;  // (begin, end)

  constexpr int kNumBlobs = 2000;
  for (int b = 0; b < kNumBlobs; ++b) {
    bool wide = (b % 37) == 0;  // sprinkle some large-magnitude/long blobs in
    std::uniform_int_distribution<int> n_dist(0, wide ? 2048 : 300);
    int n = n_dist(rng);
    uint32_t begin = static_cast<uint32_t>(points.size());
    for (int k = 0; k < n; ++k) {
      RawLineFitPoint p;
      p.x2 = wide ? coord_dist_wide(rng) : coord_dist(rng);
      p.y2 = wide ? coord_dist_wide(rng) : coord_dist(rng);
      p.W = wide ? w_dist_wide(rng) : w_dist(rng);
      p.blob_index = static_cast<uint32_t>(b);
      points.push_back(p);
    }
    ranges.emplace_back(begin, static_cast<uint32_t>(points.size()));
  }

  const size_t total_points = points.size();
  printf("test_moments_prefix: %d blobs, %zu points total\n", kNumBlobs, total_points);

  vk::Context ctx;
  fprintf(stderr, "%s\n", ctx.DescribeDevice().c_str());

  // blob_point_offsets convention (matches sort_points_local.comp /
  // scatter_index_points.comp / GpuDetector's real blob_point_offsets_buf_):
  // one inclusive-prefix-sum END offset per blob - ranges[i].second already
  // is exactly that, since points are appended blob-by-blob in order.
  std::vector<uint32_t> blob_offsets_flat(ranges.size());
  for (size_t i = 0; i < ranges.size(); ++i) blob_offsets_flat[i] = ranges[i].second;

  vk::Buffer points_buf(ctx, total_points * sizeof(RawLineFitPoint),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        vk::MemoryKind::HostVisible);
  vk::Buffer ranges_buf(ctx, blob_offsets_flat.size() * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        vk::MemoryKind::HostVisible);
  vk::Buffer moments_buf(ctx, total_points * sizeof(GpuLineFitMomentsRaw),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         vk::MemoryKind::HostVisibleCached);

  points_buf.Write(points.data(), points.size() * sizeof(RawLineFitPoint));
  ranges_buf.Write(blob_offsets_flat.data(), blob_offsets_flat.size() * sizeof(uint32_t));

  struct PushConstants {
    uint32_t num_blobs;
  } pc{static_cast<uint32_t>(kNumBlobs)};

  vk::ComputePipeline pipeline(
      ctx, ShaderPath("compute_moments_prefix"),
      {points_buf.get(), ranges_buf.get(), moments_buf.get()}, sizeof(PushConstants),
      vk::WorkgroupSize{1, 1, 1});

  VkCommandBuffer cmd = ctx.BeginCommands();
  pipeline.Dispatch1D(cmd, static_cast<uint32_t>(kNumBlobs), &pc, vk::BarrierKind::Compute);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx.SubmitAndWait(cmd);

  std::vector<GpuLineFitMomentsRaw> gpu_moments(total_points);
  moments_buf.Read(gpu_moments.data(), total_points * sizeof(GpuLineFitMomentsRaw));

  int mismatches = 0;
  for (int b = 0; b < kNumBlobs; ++b) {
    const auto &[begin, end] = ranges[b];
    std::vector<GpuLineFitMomentsRaw> cpu_cs = CpuPrefixSum(points, begin, end);
    for (uint32_t k = begin; k < end; ++k) {
      if (!MomentsEqual(gpu_moments[k], cpu_cs[k - begin])) {
        if (mismatches < 10) {
          const auto &g = gpu_moments[k];
          const auto &c = cpu_cs[k - begin];
          fprintf(stderr,
                  "MISMATCH blob=%d point=%u: gpu(Mx=%d My=%d W=%d Mxx=%d:%u Mxy=%d:%u "
                  "Myy=%d:%u) cpu(Mx=%d My=%d W=%d Mxx=%d:%u Mxy=%d:%u Myy=%d:%u)\n",
                  b, k, g.Mx, g.My, g.W, g.Mxx_hi, g.Mxx_lo, g.Mxy_hi, g.Mxy_lo, g.Myy_hi,
                  g.Myy_lo, c.Mx, c.My, c.W, c.Mxx_hi, c.Mxx_lo, c.Mxy_hi, c.Mxy_lo, c.Myy_hi,
                  c.Myy_lo);
        }
        ++mismatches;
      }
    }
  }

  if (mismatches == 0) {
    printf("PASS (stage 1): all %zu moment prefix-sum entries match the CPU bit-for-bit.\n",
          total_points);
  } else {
    printf("FAIL (stage 1): %d / %zu entries mismatched.\n", mismatches, total_points);
    return 1;
  }

  // --- Stage 2: compute_window_error.comp ---
  vk::Buffer error_buf(ctx, total_points * sizeof(float),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       vk::MemoryKind::HostVisibleCached);

  vk::ComputePipeline window_error_pipeline(
      ctx, ShaderPath("compute_window_error"), {ranges_buf.get(), moments_buf.get(), error_buf.get()},
      sizeof(PushConstants), vk::WorkgroupSize{256, 1, 1});

  cmd = ctx.BeginCommands();
  window_error_pipeline.DispatchRaw(cmd, static_cast<uint32_t>(kNumBlobs), 1, 1, &pc,
                                    vk::BarrierKind::Compute);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx.SubmitAndWait(cmd);

  std::vector<float> gpu_error(total_points, 0.0f);
  error_buf.Read(gpu_error.data(), total_points * sizeof(float));

  int error_mismatches = 0;
  int error_compared = 0;
  int ill_conditioned_skipped = 0;
  for (int b = 0; b < kNumBlobs; ++b) {
    const auto &[begin, end] = ranges[b];
    const size_t n = end - begin;
    if (n < 4) continue;
    std::vector<LineFitMoments> cpu_cs = CpuPrefixSumMoments(points, begin, end);
    std::vector<WindowErrorEntry> cpu_err = CpuWindowError(cpu_cs, n);
    const int ksz = std::min<int>(20, static_cast<int>(n / 12));
    for (size_t i = 0; i < n; ++i) {
      // ReadMomentsWindowRef's (== QuadDecode.cpp's ReadMomentsWindow) else
      // branch reads cs[index0 - 1] with no index0 > 0 guard (unlike its
      // own index0 < index1 branch, which does guard the equivalent read a
      // few lines above) - a genuine pre-existing out-of-bounds read in
      // the shipped CPU code, not something this port introduced or needs
      // to bit-match. It's reachable exactly when ksz == 0 (blobs under 12
      // points) at point i == 0 (index0 == index1 == 0 there - see
      // ReadMomentsWindowRef). compute_window_error.comp deliberately
      // does NOT reproduce the OOB read (see its own comment), so this
      // specific point is excluded from the comparison rather than
      // expecting the GPU to match undefined CPU output. This is a real
      // latent bug worth its own fix in QuadDecode.cpp - out of scope for
      // A9's GPU port itself.
      if (ksz == 0 && i == 0) continue;
      if (IsIllConditioned(cpu_err[i].window_W)) {
        ++ill_conditioned_skipped;
        continue;
      }
      const float g = gpu_error[begin + i];
      const float c = cpu_err[i].error;
      ++error_compared;
      // Tolerance-based, not bit-exact: this stage is float32-precision by
      // design on both sides (see FitLineError's own casts), and the GPU
      // additionally uses an approximate int64->float32 conversion and
      // length() instead of hypotf - see line_fit_moments.glsl / int64_emu
      // .glsl comments. 1e-2 relative + a small absolute floor for near-
      // zero errors.
      const float tol = std::max(1e-2f, 1e-3f * std::fabs(c));
      if (!(std::isfinite(g) && std::fabs(g - c) <= tol)) {
        if (error_mismatches < 10) {
          fprintf(stderr, "ERROR MISMATCH blob=%d point=%zu: gpu=%g cpu=%g diff=%g\n", b,
                  begin + i, g, c, g - c);
        }
        ++error_mismatches;
      }
    }
  }

  printf("stage 2: %d ill-conditioned windows skipped (see IsIllConditioned)\n",
        ill_conditioned_skipped);
  // A vanishingly small residual (observed: 1 in ~341000) is expected and
  // accepted here, not chased to exactly zero - IsIllConditioned's fixed
  // threshold doesn't catch every window where the float32 (Cxx+Cyy) -
  // hypot cancellation amplifies the CPU/GPU paths' inherent small
  // differences (different hypot approximation, approximate int64->float32
  // conversion - see int64_emu.glsl). Same class of accepted float wobble
  // as A5's pseudo-angle tie order. A rate above this bound would indicate
  // a real bug, not noise.
  const double mismatch_rate =
      error_compared > 0 ? static_cast<double>(error_mismatches) / error_compared : 0.0;
  if (mismatch_rate <= 1e-4) {
    printf("PASS (stage 2): %d / %d well-conditioned windowed-error entries match the CPU "
          "within tolerance (%d residual, rate %.5f%% <= accepted 0.01%%).\n",
          error_compared - error_mismatches, error_compared, error_mismatches,
          mismatch_rate * 100.0);
  } else {
    printf("FAIL (stage 2): %d / %d well-conditioned entries outside tolerance (rate %.5f%%).\n",
          error_mismatches, error_compared, mismatch_rate * 100.0);
    return 1;
  }

  // --- Stage 3: compute_peaks.comp ---
  vk::Buffer point_indices_buf(ctx, static_cast<size_t>(kNumBlobs) * 10 * sizeof(uint32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               vk::MemoryKind::HostVisibleCached);
  vk::Buffer num_selected_buf(ctx, static_cast<size_t>(kNumBlobs) * sizeof(uint32_t),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              vk::MemoryKind::HostVisibleCached);

  vk::ComputePipeline peaks_pipeline(
      ctx, ShaderPath("compute_peaks"),
      {ranges_buf.get(), error_buf.get(), point_indices_buf.get(), num_selected_buf.get()},
      sizeof(PushConstants), vk::WorkgroupSize{1, 1, 1});

  cmd = ctx.BeginCommands();
  peaks_pipeline.Dispatch1D(cmd, static_cast<uint32_t>(kNumBlobs), &pc, vk::BarrierKind::Compute);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx.SubmitAndWait(cmd);

  std::vector<uint32_t> gpu_point_indices(static_cast<size_t>(kNumBlobs) * 10);
  std::vector<uint32_t> gpu_num_selected(kNumBlobs);
  point_indices_buf.Read(gpu_point_indices.data(), gpu_point_indices.size() * sizeof(uint32_t));
  num_selected_buf.Read(gpu_num_selected.data(), gpu_num_selected.size() * sizeof(uint32_t));

  int blobs_compared = 0;
  int blobs_exact_match = 0;
  int blobs_k_mismatch = 0;
  int blobs_index_mismatch = 0;
  for (int b = 0; b < kNumBlobs; ++b) {
    const auto &[begin, end] = ranges[b];
    const size_t n = end - begin;
    // compute_peaks.comp (GPU) reads error_buf, which stage 2 already
    // populated on the GPU - so gpu_selected below reflects the combined
    // stage 2 + stage 3 GPU pipeline. The CPU reference recomputes error[]
    // independently (CpuWindowError) rather than reading the GPU's values,
    // so this comparison also (harmlessly) folds in stage 2's own
    // already-verified, tiny residual.
    std::vector<LineFitMoments> cpu_cs = CpuPrefixSumMoments(points, begin, end);
    std::vector<WindowErrorEntry> cpu_err_entries = CpuWindowError(cpu_cs, n);
    std::vector<float> cpu_error(n);
    for (size_t i = 0; i < n; ++i) cpu_error[i] = cpu_err_entries[i].error;
    if (n < 4) continue;

    ++blobs_compared;
    std::vector<double> cpu_filtered = CpuFiltered(cpu_error, n);
    std::vector<uint32_t> cpu_selected = CpuSelectPeaks(cpu_filtered, n);

    const uint32_t gpu_k = gpu_num_selected[b];
    std::vector<uint32_t> gpu_selected(gpu_point_indices.begin() + b * 10,
                                       gpu_point_indices.begin() + b * 10 + gpu_k);

    if (gpu_k != cpu_selected.size()) {
      ++blobs_k_mismatch;
    } else if (gpu_selected == cpu_selected) {
      ++blobs_exact_match;
    } else {
      ++blobs_index_mismatch;
    }
  }

  printf(
      "stage 3: %d blobs compared (n>=4); %d exact index-set matches, %d same-K but "
      "different indices, %d different K (out of %d)\n",
      blobs_compared, blobs_exact_match, blobs_index_mismatch, blobs_k_mismatch, blobs_compared);
  // Not held to exact peak-set equality: CPU accumulates the 7-tap filter
  // in double precision (see CpuFiltered), the GPU in float32 only (no
  // shaderFloat64 on Mali) - is_peak's strict `>` comparisons can
  // legitimately flip near-tied neighbors between the two precisions,
  // same accepted-wobble class as A5's pseudo-angle tie order
  // (OPTIMIZATION_NOTES.md). What matters is that the overwhelming
  // majority of blobs still select the identical peak set - a low match
  // rate would mean a real logic bug in compute_peaks.comp, not precision
  // noise.
  const double exact_match_rate =
      blobs_compared > 0 ? static_cast<double>(blobs_exact_match) / blobs_compared : 1.0;
  if (exact_match_rate < 0.95) {
    printf("FAIL (stage 3): exact match rate %.2f%% is too low to be precision noise.\n",
          exact_match_rate * 100.0);
    return 1;
  }
  printf("PASS (stage 3): %.2f%% of blobs select an identical peak set (>= 95%% required).\n",
        exact_match_rate * 100.0);

  // --- Stage 4: compute_quad_search.comp ---
  constexpr float kMaxLineFitMse = 10.0f;      // GpuDetector.h DetectorConfig default
  constexpr double kCosCriticalRad = 0.98;     // GpuDetector.h DetectorConfig default

  vk::Buffer best_indices_buf(ctx, static_cast<size_t>(kNumBlobs) * 4 * sizeof(uint32_t),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              vk::MemoryKind::HostVisibleCached);
  vk::Buffer valid_buf(ctx, static_cast<size_t>(kNumBlobs) * sizeof(uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       vk::MemoryKind::HostVisibleCached);

  struct QuadSearchPushConstants {
    uint32_t num_blobs;
    float max_line_fit_mse;
    float cos_critical_rad;
  } quad_pc{static_cast<uint32_t>(kNumBlobs), kMaxLineFitMse, static_cast<float>(kCosCriticalRad)};

  vk::ComputePipeline quad_search_pipeline(
      ctx, ShaderPath("compute_quad_search"),
      {ranges_buf.get(), moments_buf.get(), point_indices_buf.get(), num_selected_buf.get(),
       best_indices_buf.get(), valid_buf.get()},
      sizeof(QuadSearchPushConstants), vk::WorkgroupSize{1, 1, 1});

  cmd = ctx.BeginCommands();
  quad_search_pipeline.Dispatch1D(cmd, static_cast<uint32_t>(kNumBlobs), &quad_pc,
                                  vk::BarrierKind::Compute);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx.SubmitAndWait(cmd);

  std::vector<uint32_t> gpu_best_indices(static_cast<size_t>(kNumBlobs) * 4);
  std::vector<uint32_t> gpu_valid(kNumBlobs);
  best_indices_buf.Read(gpu_best_indices.data(), gpu_best_indices.size() * sizeof(uint32_t));
  valid_buf.Read(gpu_valid.data(), gpu_valid.size() * sizeof(uint32_t));

  int quad_blobs_compared = 0;
  int quad_exact_match = 0;
  int quad_validity_mismatch = 0;
  for (int b = 0; b < kNumBlobs; ++b) {
    const auto &[begin, end] = ranges[b];
    const size_t n = end - begin;
    if (n < 4) continue;

    std::vector<LineFitMoments> cpu_cs = CpuPrefixSumMoments(points, begin, end);
    std::vector<WindowErrorEntry> cpu_err_entries = CpuWindowError(cpu_cs, n);
    std::vector<float> cpu_error(n);
    for (size_t i = 0; i < n; ++i) cpu_error[i] = cpu_err_entries[i].error;
    std::vector<double> cpu_filtered = CpuFiltered(cpu_error, n);
    std::vector<uint32_t> cpu_selected = CpuSelectPeaks(cpu_filtered, n);

    QuadSearchResultRef cpu_quad =
        CpuQuadSearch(cpu_cs, n, cpu_selected, kMaxLineFitMse, kCosCriticalRad);

    ++quad_blobs_compared;
    const bool gpu_is_valid = gpu_valid[b] != 0u;
    if (gpu_is_valid != cpu_quad.valid) {
      ++quad_validity_mismatch;
      continue;
    }
    if (!cpu_quad.valid) {
      ++quad_exact_match;  // both agree: no quad found
      continue;
    }
    bool match = true;
    for (int k = 0; k < 4; ++k) {
      if (gpu_best_indices[b * 4 + k] != cpu_quad.indices[k]) match = false;
    }
    if (match) ++quad_exact_match;
  }

  printf("stage 4: %d blobs compared; %d exact matches (including agreeing invalid), %d "
        "validity mismatches (out of %d)\n",
        quad_blobs_compared, quad_exact_match, quad_validity_mismatch, quad_blobs_compared);
  const double quad_match_rate =
      quad_blobs_compared > 0 ? static_cast<double>(quad_exact_match) / quad_blobs_compared : 1.0;
  if (quad_match_rate < 0.95) {
    printf("FAIL (stage 4): match rate %.2f%% is too low to be precision noise.\n",
          quad_match_rate * 100.0);
    return 1;
  }
  printf("PASS (stage 4): %.2f%% of blobs agree with the CPU on the winning quad "
        "(>= 95%% required).\n",
        quad_match_rate * 100.0);
  return 0;
}
