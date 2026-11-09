// Runs a list of frames as a video sequence through TrackingDetector, with tracking off and on in
// alternating rounds, and reports per-frame times, the path each frame took and the detections.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vkapriltag/TrackingDetector.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/common/pgm_io.h"
#include "vkapriltag/vk/Context.h"

namespace {

using Clock = std::chrono::steady_clock;

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5))];
}

void WriteDets(std::ostream &out, int index, zarray_t *z) {
  std::vector<apriltag_detection_t *> dets;
  for (int i = 0; i < zarray_size(z); ++i) {
    apriltag_detection_t *d;
    zarray_get(z, i, &d);
    dets.push_back(d);
  }
  std::sort(dets.begin(), dets.end(), [](auto *a, auto *b) { return a->id < b->id; });
  out << "F " << index << ' ' << dets.size() << '\n';
  char line[512];
  for (const auto *d : dets) {
    std::snprintf(line, sizeof line, "%d %d %.6g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", d->id,
                  d->hamming, d->decision_margin, d->c[0], d->c[1], d->p[0][0], d->p[0][1], d->p[1][0],
                  d->p[1][1], d->p[2][0], d->p[2][1], d->p[3][0], d->p[3][1]);
    out << line;
  }
}

struct Sample {
  double ms;
  int path;  // 0 full, 1 tracked, 2 search
};

}  // namespace

int main(int argc, char **argv) {
  using namespace apriltag_vulkan;
  std::string list, prefix, family = "tag36h11";
  int rounds = 4;
  double gap_ms = 0.0;
  uint32_t decimation = 2;
  TrackingConfig tc;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--list") list = next();
      else if (a == "--out-prefix") prefix = next();
      else if (a == "--rounds") rounds = std::max(1, std::stoi(next()));
      else if (a == "--gap-ms") gap_ms = std::stod(next());
      else if (a == "--decimation") decimation = static_cast<uint32_t>(std::stoi(next()));
      else if (a == "--keyframe") tc.keyframe_interval = static_cast<uint32_t>(std::stoi(next()));
      else if (a == "--family") family = next();
      else throw std::runtime_error("unknown argument " + a);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\nusage: track_eval --list files.txt [--out-prefix p] [--rounds N] [--gap-ms ms] "
                             "[--decimation N] [--keyframe K] [--family name]\n";
    return 2;
  }
  std::vector<std::vector<uint8_t>> frames;
  uint32_t width = 0, height = 0;
  {
    std::ifstream in(list);
    if (!in) {
      std::cerr << "cannot open " << list << "\n";
      return 1;
    }
    for (std::string line; std::getline(in, line);) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (line.empty()) continue;
      frames.emplace_back();
      if (!LoadGrayPgm(line, &frames.back(), &width, &height)) {
        std::cerr << "cannot load " << line << "\n";
        return 1;
      }
    }
  }
  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, family.c_str())) return 1;
  int rc = 0;
  try {
    vk::Context ctx;
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->refine_edges = true;
    DetectorConfig config;
    config.width = width;
    config.height = height;
    config.decimation = decimation;
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;
    tc.enabled = true;
    TrackingDetector det(ctx, td, config, tc);
    const bool env_on = det.enabled();

    std::vector<Sample> samples[2];  // [tracking off, on]
    bool dumped[2] = {false, false};
    for (int round = -1; round < rounds; ++round) {
      for (int slot = 0; slot < 2; ++slot) {
        const int mode = (round & 1) ? 1 - slot : slot;  // ABBA
        det.SetEnabled(mode == 1 && env_on);
        std::ofstream out;
        const bool dump = round == 0 && !dumped[mode] && !prefix.empty();
        if (dump) out.open(prefix + (mode ? "_track.txt" : "_full.txt"), std::ios::trunc);
        if (round == 0) dumped[mode] = true;
        for (size_t n = 0; n < frames.size(); ++n) {
          if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::microseconds(int(gap_ms * 1000)));
          const auto t0 = Clock::now();
          zarray_t *dets = det.Detect(frames[n].data());
          const auto t1 = Clock::now();
          if (round >= 0) {
            samples[mode].push_back({std::chrono::duration<double, std::milli>(t1 - t0).count(),
                                     static_cast<int>(det.last_path())});
          }
          if (dump) WriteDets(out, int(n), dets);
        }
      }
    }
    for (int mode = 0; mode < 2; ++mode) {
      std::vector<double> all, by_path[3];
      for (const Sample &s : samples[mode]) {
        all.push_back(s.ms);
        by_path[s.path].push_back(s.ms);
      }
      const double total = static_cast<double>(all.size());
      std::printf("SUMMARY mode=%s frames=%zu median_ms=%.4f p95_ms=%.4f mean_ms=%.4f", mode ? "track" : "full",
                  frames.size(), Percentile(all, 0.5), Percentile(all, 0.95),
                  all.empty() ? 0.0 : std::accumulate(all.begin(), all.end(), 0.0) / total);
      const char *names[3] = {"full", "tracked", "search"};
      for (int p = 0; p < 3; ++p) {
        std::printf(" %s=%.1f%%(med %.4f)", names[p], 100.0 * static_cast<double>(by_path[p].size()) / total,
                    Percentile(by_path[p], 0.5));
      }
      std::printf("\n");
    }
    apriltag_detector_destroy(td);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    rc = 1;
  }
  teardown_tag_family(&tf, family.c_str());
  return rc;
}
