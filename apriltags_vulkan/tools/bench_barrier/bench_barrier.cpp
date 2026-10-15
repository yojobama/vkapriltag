// Measures the cost of a pipeline barrier: K independent dispatches are replayed with no barrier, an
// execution-only barrier or a memory barrier between them, and frame-time slopes are compared.
// Usage: bench_barrier [--spv <file>] [--rounds N] [--replays N] [--queue-family N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

#ifndef BARRIER_PROBE_SPV
#define BARRIER_PROBE_SPV "barrier_probe.comp.spv"
#endif

namespace {

using namespace apriltag_vulkan;
using Clock = std::chrono::steady_clock;

enum Variant { kNone = 0, kExec = 1, kMem = 2, kVariants = 3 };
const char *kVariantName[kVariants] = {"none", "exec-only", "memory"};

struct Config {
  uint32_t elements;
  int k;
  Variant variant;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  std::vector<double> round_median_us;
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  std::string spv = BARRIER_PROBE_SPV;
  int rounds = 7, replays = 60, family = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--spv" && i + 1 < argc) spv = argv[++i];
    else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (a == "--replays" && i + 1 < argc) replays = std::atoi(argv[++i]);
    else if (a == "--queue-family" && i + 1 < argc) family = std::atoi(argv[++i]);
    else { std::cerr << "unknown argument " << a << "\n"; return 2; }
  }

  vk::ContextOptions options;
  options.queue_family = family;
  vk::Context ctx(options);
  std::unique_ptr<vk::Lane> lane = ctx.CreateLane(0);

  const uint32_t sizes[] = {1, 16384, 262144, 1048576};
  const int ks[] = {16, 144};
  vk::Buffer buf(ctx, 1048576 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::MemoryKind::DeviceLocal);
  vk::WorkgroupSize wg{ctx.caps().wg1d, 1, 1};
  vk::ComputePipeline pipe(ctx, vk::ShaderSource(spv), {buf.get()}, 4, wg);

  std::vector<Config> configs;
  for (uint32_t s : sizes)
    for (int k : ks)
      for (int v = 0; v < kVariants; ++v) configs.push_back({s, k, static_cast<Variant>(v)});

  for (Config &c : configs) {
    c.cmd = lane->BeginPersistentCommands();
    for (int i = 0; i < c.k; ++i) {
      pipe.Dispatch1D(c.cmd, c.elements, &c.elements, vk::BarrierKind::None);
      if (c.variant == kMem) {
        vk::ComputePipeline::Barrier(c.cmd, vk::BarrierKind::Compute);
      } else if (c.variant == kExec) {
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0,
                             nullptr);
      }
    }
    lane->EndCommands(c.cmd);
  }

  for (int r = 0; r < rounds; ++r) {
    for (Config &c : configs) {
      std::vector<double> t;
      for (int i = 0; i < replays + 10; ++i) {
        const auto t0 = Clock::now();
        lane->SubmitPersistentAndWait(c.cmd);
        const auto t1 = Clock::now();
        if (i >= 10) t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      c.round_median_us.push_back(Median(t));
    }
  }

  auto cell = [&](uint32_t s, int k, Variant v) {
    for (const Config &c : configs)
      if (c.elements == s && c.k == k && c.variant == v) return Median(c.round_median_us);
    return 0.0;
  };

  std::cout << "device: " << ctx.caps().name << "  wg1d=" << ctx.caps().wg1d << "  rounds=" << rounds
            << " replays=" << replays << "\n";
  std::cout << "frame time (us, median of round medians) at K=" << ks[0] << " and K=" << ks[1]
            << " dispatches, and the per-dispatch slope\n";
  std::printf("%10s | %-9s | %9s %9s | %12s | %s\n", "elements", "variant", "K=16", "K=144",
              "us/dispatch", "barrier cost vs none (us)");
  for (uint32_t s : sizes) {
    const double slope_none = (cell(s, ks[1], kNone) - cell(s, ks[0], kNone)) / (ks[1] - ks[0]);
    for (int v = 0; v < kVariants; ++v) {
      const double a = cell(s, ks[0], static_cast<Variant>(v));
      const double b = cell(s, ks[1], static_cast<Variant>(v));
      const double slope = (b - a) / (ks[1] - ks[0]);
      std::printf("%10u | %-9s | %9.1f %9.1f | %12.3f | %s%.3f\n", s, kVariantName[v], a, b, slope,
                  v == 0 ? "" : "+", slope - slope_none);
    }
  }
  std::cout << "round-to-round spread of the memory variant at K=144, 262144 elements (us):";
  for (const Config &c : configs)
    if (c.elements == 262144 && c.k == 144 && c.variant == kMem)
      for (double m : c.round_median_us) std::printf(" %.1f", m);
  std::cout << "\n";

  for (Config &c : configs) lane->FreePersistentCommands(c.cmd);
  return 0;
}
