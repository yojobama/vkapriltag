// Runs a list of frames as a video sequence through TrackingDetector in several modes (full detector
// on every frame, keyframe tracking, asynchronous tracking, each with or without the wide capture),
// interleaved over rounds, and reports per-frame times, the path each frame took and the detections.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#endif

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

// User plus kernel CPU seconds of this process.
double ProcessCpuSeconds() {
#ifdef _WIN32
  FILETIME create, exit, kernel, user;
  GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user);
  auto sec = [](const FILETIME &f) {
    return static_cast<double>((static_cast<unsigned long long>(f.dwHighDateTime) << 32) | f.dwLowDateTime) * 1e-7;
  };
  return sec(kernel) + sec(user);
#else
  rusage r;
  getrusage(RUSAGE_SELF, &r);
  return static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec) +
         static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
#endif
}

std::string DetsText(int index, zarray_t *z) {
  std::vector<apriltag_detection_t *> dets;
  for (int i = 0; i < zarray_size(z); ++i) {
    apriltag_detection_t *d;
    zarray_get(z, i, &d);
    dets.push_back(d);
  }
  std::sort(dets.begin(), dets.end(), [](auto *a, auto *b) { return a->id < b->id; });
  std::ostringstream out;
  out << "F " << index << ' ' << dets.size() << '\n';
  char line[512];
  for (const auto *d : dets) {
    std::snprintf(line, sizeof line, "%d %d %.6g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", d->id,
                  d->hamming, d->decision_margin, d->c[0], d->c[1], d->p[0][0], d->p[0][1], d->p[1][0],
                  d->p[1][1], d->p[2][0], d->p[2][1], d->p[3][0], d->p[3][1]);
    out << line;
  }
  return out.str();
}

struct Sample {
  double ms;
  int path;  // 0 full, 1 tracked
};

struct ModeStats {
  std::vector<Sample> samples;
  double cpu_s = 0.0;
  double wall_s = 0.0;
  uint64_t frames = 0;
  uint64_t background_runs = 0, tentative_accepted = 0, tentative_rejected = 0, tracks_dropped = 0;
  double copy_ms = 0.0;
};

struct Mode {
  std::string name;
  bool enabled, async, drop_lost;
};

bool ParseMode(const std::string &name, Mode *m) {
  *m = {name, name != "full", name == "async" || name == "asyncsync", name != "asyncsync"};
  return name == "full" || name == "key" || name == "async" || name == "asyncsync";
}

}  // namespace

int main(int argc, char **argv) {
  using namespace apriltag_vulkan;
  std::string list, prefix, family = "tag36h11", mode_list = "full,key,async";
  int rounds = 4, dump_rounds = 1;
  double gap_ms = 0.0;
  bool spin_gap = false;
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
      else if (a == "--dump-rounds") dump_rounds = std::max(1, std::stoi(next()));
      else if (a == "--gap-ms") gap_ms = std::stod(next());
      else if (a == "--spin-gap") spin_gap = true;
      else if (a == "--decimation") decimation = static_cast<uint32_t>(std::stoi(next()));
      else if (a == "--keyframe") tc.keyframe_interval = static_cast<uint32_t>(std::stoi(next()));
      else if (a == "--modes") mode_list = next();
      else if (a == "--async-interval") tc.async_interval = static_cast<uint32_t>(std::max(1, std::stoi(next())));
      else if (a == "--bg-threads") tc.async_threads = static_cast<uint32_t>(std::max(0, std::stoi(next())));
      else if (a == "--no-copy") tc.async_copy = false;
      else if (a == "--family") family = next();
      else throw std::runtime_error("unknown argument " + a);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\nusage: track_eval --list files.txt [--out-prefix p] [--rounds N] [--dump-rounds N] [--gap-ms ms]\n"
                             "  [--decimation N] [--keyframe K] [--modes full,key,async,asyncsync] [--async-interval N]\n"
                             "  [--bg-threads N] [--no-copy] [--family name]\n";
    return 2;
  }
  std::vector<Mode> modes;
  {
    std::istringstream in(mode_list);
    for (std::string tok; std::getline(in, tok, ',');) {
      Mode m;
      if (!ParseMode(tok, &m)) {
        std::cerr << "unknown mode " << tok << "\n";
        return 2;
      }
      modes.push_back(m);
    }
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
      uint32_t w = 0, h = 0;
      if (!LoadGrayPgm(line, &frames.back(), &w, &h)) {
        std::cerr << "cannot load " << line << "\n";
        return 1;
      }
      if (frames.size() == 1) {
        width = w;
        height = h;
      } else if (w != width || h != height) {
        std::cerr << "error: " << line << " is " << w << "x" << h << " but the first frame is " << width << "x"
                  << height << "; all frames must have the same size\n";
        return 1;
      }
    }
  }
  if (frames.empty()) {
    std::cerr << "error: no frames in " << list << "\n";
    return 1;
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
    const TrackingConfig base = det.tracking();

    const size_t nm = modes.size();
    std::vector<ModeStats> stats(nm);
    std::vector<std::vector<std::vector<double>>> frame_ms(nm, std::vector<std::vector<double>>(frames.size()));
    for (int round = -1; round < rounds; ++round) {
      for (size_t slot = 0; slot < nm; ++slot) {
        // Rotates the order each round, and reverses every other one.
        const size_t r = static_cast<size_t>(round + 1);
        const size_t mi = (r & 1) ? (nm - 1 - (slot + r / 2) % nm) : (slot + r / 2) % nm;
        const Mode &m = modes[mi];
        det.tracking() = base;
        det.tracking().enabled = m.enabled;
        det.tracking().async = m.async;
        det.tracking().async_drop_lost = m.drop_lost;
        det.Reset();
        const bool dump = round >= 0 && round < dump_rounds && !prefix.empty();
        std::vector<std::string> text;
        const TrackingDetector::Stats before = det.stats();
        const double cpu0 = ProcessCpuSeconds();
        const auto wall0 = Clock::now();
        for (size_t n = 0; n < frames.size(); ++n) {
          if (gap_ms > 0 && spin_gap) {
            const auto until = Clock::now() + std::chrono::microseconds(int(gap_ms * 1000));
            while (Clock::now() < until) {
            }
          } else if (gap_ms > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(int(gap_ms * 1000)));
          }
          const auto t0 = Clock::now();
          zarray_t *dets = det.Detect(frames[n].data());
          const auto t1 = Clock::now();
          const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          if (round >= 0) {
            stats[mi].samples.push_back({ms, static_cast<int>(det.last_path())});
            frame_ms[mi][n].push_back(ms);
          }
          if (dump) {
            char tail[64];
            std::snprintf(tail, sizeof tail, "# path %d ms %.4f\n", static_cast<int>(det.last_path()), ms);
            text.push_back(DetsText(int(n), dets) + tail);
          }
        }
        const double wall = std::chrono::duration<double>(Clock::now() - wall0).count();
        const double cpu = ProcessCpuSeconds() - cpu0;
        det.WaitBackground();
        if (round >= 0) {
          const TrackingDetector::Stats &after = det.stats();
          stats[mi].cpu_s += cpu;
          stats[mi].wall_s += wall;
          stats[mi].frames += frames.size();
          stats[mi].background_runs += after.background_runs - before.background_runs;
          stats[mi].tentative_accepted += after.tentative_accepted - before.tentative_accepted;
          stats[mi].tentative_rejected += after.tentative_rejected - before.tentative_rejected;
          stats[mi].tracks_dropped += after.tracks_dropped - before.tracks_dropped;
          stats[mi].copy_ms += after.copy_ms - before.copy_ms;
        }
        if (dump) {
          const std::string file = prefix + "_" + m.name + (round == 0 ? "" : "_r" + std::to_string(round)) + ".txt";
          std::ofstream out(file, std::ios::trunc);
          for (const std::string &t : text) out << t;
        }
      }
    }
    for (size_t mi = 0; mi < nm; ++mi) {
      std::vector<double> all, by_path[2];
      for (const Sample &s : stats[mi].samples) {
        all.push_back(s.ms);
        by_path[s.path].push_back(s.ms);
      }
      const double total = static_cast<double>(all.size());
      std::printf("SUMMARY mode=%s frames=%zu median_ms=%.4f p95_ms=%.4f max_ms=%.4f mean_ms=%.4f cpu_ms_per_frame=%.4f",
                  modes[mi].name.c_str(), frames.size(), Percentile(all, 0.5), Percentile(all, 0.95),
                  all.empty() ? 0.0 : *std::max_element(all.begin(), all.end()),
                  all.empty() ? 0.0 : std::accumulate(all.begin(), all.end(), 0.0) / total,
                  stats[mi].frames ? 1000.0 * stats[mi].cpu_s / static_cast<double>(stats[mi].frames) : 0.0);
      const char *names[2] = {"full", "tracked"};
      for (int p = 0; p < 2; ++p) {
        std::printf(" %s=%.1f%%(med %.4f)", names[p], 100.0 * static_cast<double>(by_path[p].size()) / total,
                    Percentile(by_path[p], 0.5));
      }
      if (modes[mi].async) {
        const double runs = static_cast<double>(std::max<uint64_t>(stats[mi].background_runs, 1));
        std::printf(" bg_runs=%.1f/round tent_ok=%.1f tent_rej=%.1f dropped=%.1f copy_us=%.1f",
                    static_cast<double>(stats[mi].background_runs) / rounds,
                    static_cast<double>(stats[mi].tentative_accepted) / rounds,
                    static_cast<double>(stats[mi].tentative_rejected) / rounds,
                    static_cast<double>(stats[mi].tracks_dropped) / rounds, 1000.0 * stats[mi].copy_ms / runs);
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
