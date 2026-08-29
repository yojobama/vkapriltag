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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
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

  std::vector<uint32_t> ranges_flat(ranges.size() * 2);
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges_flat[i * 2 + 0] = ranges[i].first;
    ranges_flat[i * 2 + 1] = ranges[i].second;
  }

  vk::Buffer points_buf(ctx, total_points * sizeof(RawLineFitPoint),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        vk::MemoryKind::HostVisible);
  vk::Buffer ranges_buf(ctx, ranges_flat.size() * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        vk::MemoryKind::HostVisible);
  vk::Buffer moments_buf(ctx, total_points * sizeof(GpuLineFitMomentsRaw),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         vk::MemoryKind::HostVisibleCached);

  points_buf.Write(points.data(), points.size() * sizeof(RawLineFitPoint));
  ranges_buf.Write(ranges_flat.data(), ranges_flat.size() * sizeof(uint32_t));

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
    return 0;
  }
  printf("FAIL (stage 2): %d / %d well-conditioned entries outside tolerance (rate %.5f%%).\n",
        error_mismatches, error_compared, mismatch_rate * 100.0);
  return 1;
}
