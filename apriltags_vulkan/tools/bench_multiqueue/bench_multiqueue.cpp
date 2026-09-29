// Runs N independent detector streams (one Context + GpuDetector per thread) on the same GPU and
// reports aggregate throughput. Usage: bench_multiqueue --pgm <file> [--streams N]
// [--family auto|0|1|alternate] [--iterations N] [--decimation N] [--cpu-tail]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

struct Stream {
  std::unique_ptr<apriltag_vulkan::vk::Context> ctx;
  std::unique_ptr<apriltag_vulkan::GpuDetector> detector;
  std::unique_ptr<apriltag_vulkan::QuadDecode> quad_decode;
  apriltag_detector_t *td = nullptr;
  std::unique_ptr<apriltag_vulkan::TagDecoder> tag_decoder;
  std::vector<double> frame_ms;
};

}  // namespace

int main(int argc, char **argv) {
  std::string pgm_path, family = "auto";
  int streams = 1, iterations = 300;
  uint32_t decimation = 2;
  bool cpu_tail = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (arg == "--pgm") {
      pgm_path = next();
    } else if (arg == "--streams") {
      streams = std::max(1, std::stoi(next()));
    } else if (arg == "--family") {
      family = next();
    } else if (arg == "--iterations") {
      iterations = std::max(1, std::stoi(next()));
    } else if (arg == "--decimation") {
      decimation = static_cast<uint32_t>(std::max(1, std::stoi(next())));
    } else if (arg == "--cpu-tail") {
      cpu_tail = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }
  if (pgm_path.empty()) {
    std::cerr << "Usage: bench_multiqueue --pgm <file> [--streams N] "
                 "[--family auto|0|1|alternate] [--iterations N] [--decimation N] [--cpu-tail]\n";
    return 1;
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
  try {
    for (int s = 0; s < streams; ++s) {
      apriltag_vulkan::vk::ContextOptions opts;
      opts.use_pipeline_cache = false;  // avoids concurrent writers to one cache file
      if (family == "0") {
        opts.queue_family = 0;
      } else if (family == "1") {
        opts.queue_family = 1;
      } else if (family == "alternate") {
        opts.queue_family = s % 2;
      }
      Stream &st = pool[static_cast<size_t>(s)];
      st.ctx = std::make_unique<apriltag_vulkan::vk::Context>(opts);
      st.detector = std::make_unique<apriltag_vulkan::GpuDetector>(*st.ctx, config);
      if (cpu_tail) {
        st.quad_decode = std::make_unique<apriltag_vulkan::QuadDecode>(config);
        st.td = apriltag_detector_create();
        apriltag_detector_add_family(st.td, tf);
        st.td->refine_edges = true;
        st.tag_decoder = std::make_unique<apriltag_vulkan::TagDecoder>(st.td, decimation, 1);
      }
      st.frame_ms.reserve(static_cast<size_t>(iterations));
    }
  } catch (const std::exception &e) {
    std::cerr << "Setup failed: " << e.what() << std::endl;
    return 1;
  }

  auto run_frame = [&](Stream &st) {
    st.detector->Detect(gray.data());
    if (cpu_tail) {
      const std::vector<apriltag_vulkan::DetectedQuad> quads =
          st.quad_decode->Decode(st.detector->last_line_fit_points);
      st.tag_decoder->Decode(quads, gray.data(), width, height, config.reversed_border);
    }
  };

  for (Stream &st : pool) {  // warm-up, serial
    for (int i = 0; i < 20; ++i) run_frame(st);
  }

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (Stream &st : pool) {
    threads.emplace_back([&, sp = &st] {
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int i = 0; i < iterations; ++i) {
        const auto t0 = Clock::now();
        run_frame(*sp);
        sp->frame_ms.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      }
    });
  }
  while (ready.load() < streams) std::this_thread::yield();
  const auto start = Clock::now();
  go.store(true, std::memory_order_release);
  for (std::thread &t : threads) t.join();
  const double wall_s = std::chrono::duration<double>(Clock::now() - start).count();

  std::vector<double> all;
  for (const Stream &st : pool) all.insert(all.end(), st.frame_ms.begin(), st.frame_ms.end());
  std::sort(all.begin(), all.end());
  const double fps = static_cast<double>(streams) * iterations / wall_s;
  std::cout << "streams=" << streams << " family=" << family << " cpu_tail=" << cpu_tail
            << " fps=" << fps << " median_ms=" << all[all.size() / 2]
            << " p99_ms=" << all[static_cast<size_t>(all.size() * 0.99)] << std::endl;
  return 0;
}
