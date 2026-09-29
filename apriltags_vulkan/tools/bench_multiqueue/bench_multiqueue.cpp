// Simulates several cameras sharing one GPU: one Context (one VkDevice), and per stream a Lane
// (queue), GpuDetector, QuadDecode, TagDecoder and FramePipeline, fed by its own thread.
// Usage: bench_multiqueue --pgm <file> [--streams N] [--mode single|per-stream|alternate]
//   [--iterations N] [--decimation N] [--cpu-threads N] [--fps N] [--family N]
// single: every stream shares queue 0. per-stream: one queue per stream on the primary family.
// alternate: queues interleaved across both compute families. --fps paces each stream like a
// camera (0 = free-running).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vkapriltag/FramePipeline.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/common/pgm_io.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

extern "C" {
#include "apriltag.h"
}

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct Stream {
  std::unique_ptr<apriltag_vulkan::vk::Lane> lane;
  std::unique_ptr<apriltag_vulkan::GpuDetector> detector;
  std::unique_ptr<apriltag_vulkan::QuadDecode> quad_decode;
  apriltag_detector_t *td = nullptr;
  std::unique_ptr<apriltag_vulkan::TagDecoder> tag_decoder;
  std::unique_ptr<apriltag_vulkan::FramePipeline> pipeline;
  std::vector<uint8_t> frames[2];
  std::vector<double> latency_ms;
  int late = 0;
};

std::mutex g_expected_mutex;
std::vector<int> g_expected_ids;
bool g_have_expected = false;
std::atomic<int> g_mismatches{0};

std::vector<int> SortedIds(const zarray_t *dets) {
  std::vector<int> ids;
  for (int i = 0; i < zarray_size(dets); ++i) {
    apriltag_detection_t *d = nullptr;
    zarray_get(const_cast<zarray_t *>(dets), i, &d);
    ids.push_back(d->id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

void CheckResult(const zarray_t *dets) {
  const std::vector<int> ids = SortedIds(dets);
  std::lock_guard<std::mutex> lock(g_expected_mutex);
  if (!g_have_expected) {
    g_expected_ids = ids;
    g_have_expected = true;
  } else if (ids != g_expected_ids) {
    g_mismatches.fetch_add(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string pgm_path, mode = "per-stream";
  int streams = 1, iterations = 300, cpu_threads = 0, family = -1;
  double fps = 0.0;
  uint32_t decimation = 2;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (arg == "--pgm") {
      pgm_path = next();
    } else if (arg == "--streams") {
      streams = std::max(1, std::stoi(next()));
    } else if (arg == "--mode") {
      mode = next();
    } else if (arg == "--iterations") {
      iterations = std::max(1, std::stoi(next()));
    } else if (arg == "--decimation") {
      decimation = static_cast<uint32_t>(std::max(1, std::stoi(next())));
    } else if (arg == "--cpu-threads") {
      cpu_threads = std::max(1, std::stoi(next()));
    } else if (arg == "--fps") {
      fps = std::stod(next());
    } else if (arg == "--family") {
      family = std::stoi(next());
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }
  if (pgm_path.empty() || (mode != "single" && mode != "per-stream" && mode != "alternate")) {
    std::cerr << "Usage: bench_multiqueue --pgm <file> [--streams N] "
                 "[--mode single|per-stream|alternate] [--iterations N] [--decimation N] "
                 "[--cpu-threads N] [--fps N] [--family N]\n";
    return 1;
  }
  if (cpu_threads == 0) {
    cpu_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / streams);
  }

  std::vector<uint8_t> gray;
  uint32_t width = 0, height = 0;
  if (!apriltag_vulkan::LoadGrayPgm(pgm_path, &gray, &width, &height)) {
    std::cerr << "Failed to load PGM: " << pgm_path << std::endl;
    return 1;
  }

  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, "tag36h11")) return 1;

  apriltag_vulkan::DetectorConfig config;
  config.width = width;
  config.height = height;
  config.decimation = decimation;
  config.tag_width = static_cast<uint32_t>(tf->width_at_border);
  config.reversed_border = tf->reversed_border;
  config.normal_border = !tf->reversed_border;

  std::vector<Stream> pool(static_cast<size_t>(streams));
  size_t queue_count = 0;
  try {
    apriltag_vulkan::vk::ContextOptions opts;
    opts.queue_family = family;
    opts.queues_per_family = static_cast<uint32_t>(streams);
    opts.use_secondary_compute_family = (mode == "alternate");
    apriltag_vulkan::vk::Context ctx(opts);
    queue_count = ctx.queue_count();

    // Queue slot per stream.
    std::vector<size_t> slots;
    for (size_t q = 0; q < ctx.queue_count(); ++q) {
      if (mode == "alternate" || ctx.queue_slot_family(q) == ctx.queue_family()) slots.push_back(q);
    }

    for (int s = 0; s < streams; ++s) {
      Stream &st = pool[static_cast<size_t>(s)];
      const size_t slot = (mode == "single") ? 0 : slots[static_cast<size_t>(s) % slots.size()];
      st.lane = ctx.CreateLane(slot);
      st.detector = std::make_unique<apriltag_vulkan::GpuDetector>(ctx, *st.lane, config);
      st.quad_decode = std::make_unique<apriltag_vulkan::QuadDecode>(config);
      st.td = apriltag_detector_create();
      apriltag_detector_add_family(st.td, tf);
      st.td->refine_edges = true;
      st.tag_decoder = std::make_unique<apriltag_vulkan::TagDecoder>(
          st.td, decimation, static_cast<uint32_t>(cpu_threads));
      st.pipeline = std::make_unique<apriltag_vulkan::FramePipeline>(*st.detector, *st.quad_decode,
                                                                     *st.tag_decoder);
      st.frames[0] = gray;
      st.frames[1] = gray;
      st.latency_ms.reserve(static_cast<size_t>(iterations));
    }

    for (Stream &st : pool) {  // warm-up, serial
      for (int i = 0; i < 20; ++i) {
        zarray_t *r = st.pipeline->Push(st.frames[i & 1].data(), width, height,
                                        config.reversed_border);
        if (r != nullptr) CheckResult(r);
      }
      if (zarray_t *r = st.pipeline->Flush()) CheckResult(r);
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    Clock::time_point start;
    std::vector<std::thread> threads;
    const double period_ms = fps > 0.0 ? 1000.0 / fps : 0.0;
    for (Stream &st : pool) {
      threads.emplace_back([&, sp = &st] {
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        std::vector<Clock::time_point> pushed(static_cast<size_t>(iterations));
        for (int k = 0; k < iterations; ++k) {
          if (period_ms > 0.0) {
            const auto due = start + std::chrono::duration_cast<Clock::duration>(Ms(period_ms * k));
            std::this_thread::sleep_until(due);
            if (Ms(Clock::now() - due).count() > 0.5 * period_ms) ++sp->late;
          }
          pushed[static_cast<size_t>(k)] = Clock::now();
          zarray_t *r = sp->pipeline->Push(sp->frames[k & 1].data(), width, height,
                                           config.reversed_border);
          if (r != nullptr && k > 0) {
            sp->latency_ms.push_back(Ms(Clock::now() - pushed[static_cast<size_t>(k - 1)]).count());
            CheckResult(r);
          }
        }
        if (zarray_t *r = sp->pipeline->Flush()) {
          sp->latency_ms.push_back(Ms(Clock::now() - pushed.back()).count());
          CheckResult(r);
        }
      });
    }
    while (ready.load() < streams) std::this_thread::yield();
    start = Clock::now();
    go.store(true, std::memory_order_release);
    for (std::thread &t : threads) t.join();
    const double wall_s = std::chrono::duration<double>(Clock::now() - start).count();

    std::vector<double> all;
    int late = 0;
    for (const Stream &st : pool) {
      all.insert(all.end(), st.latency_ms.begin(), st.latency_ms.end());
      late += st.late;
    }
    std::sort(all.begin(), all.end());
    const double total_fps = static_cast<double>(streams) * iterations / wall_s;
    std::cout << "mode=" << mode << " streams=" << streams << " queues=" << queue_count
              << " cpu_threads=" << cpu_threads << " target_fps=" << fps
              << " total_fps=" << total_fps << " per_stream_fps=" << total_fps / streams
              << " latency_med_ms=" << all[all.size() / 2]
              << " latency_p99_ms=" << all[static_cast<size_t>(all.size() * 0.99)]
              << " late=" << late << " mismatches=" << g_mismatches.load() << std::endl;

    pool.clear();  // streams first, then the Context
  } catch (const std::exception &e) {
    std::cerr << "Failed: " << e.what() << std::endl;
    return 1;
  }
  return g_mismatches.load() == 0 ? 0 : 2;
}
