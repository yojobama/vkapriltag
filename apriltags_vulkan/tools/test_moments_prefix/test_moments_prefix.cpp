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

#include <cstdint>
#include <cstdio>
#include <cstring>
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
  // A few large-magnitude blobs stress the high words of the 64-bit
  // emulation (Mxx/Mxy/Myy) directly - realistic inputs alone might not
  // reach that range.
  std::uniform_int_distribution<int> coord_dist_wide(-2000000000, 2000000000);
  std::uniform_int_distribution<int> w_dist_wide(1, 2000000000);

  std::vector<RawLineFitPoint> points;
  std::vector<std::pair<uint32_t, uint32_t>> ranges;  // (begin, end)

  constexpr int kNumBlobs = 2000;
  for (int b = 0; b < kNumBlobs; ++b) {
    std::uniform_int_distribution<int> n_dist(0, 300);
    int n = n_dist(rng);
    uint32_t begin = static_cast<uint32_t>(points.size());
    bool wide = (b % 37) == 0;  // sprinkle some large-magnitude blobs in
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
    printf("PASS: all %zu moment prefix-sum entries match the CPU bit-for-bit.\n", total_points);
    return 0;
  }
  printf("FAIL: %d / %zu entries mismatched.\n", mismatches, total_points);
  return 1;
}
