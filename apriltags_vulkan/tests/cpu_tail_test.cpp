// Runs the GPU tail and the CPU tail (APRILTAG_VK_CPU_TAIL) over the same frames and compares the
// selected blobs, their sorted points and the final detections.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "eval_common.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

namespace {

using apriltag_vulkan::RawLineFitPoint;

struct Blob {
  std::vector<std::pair<uint32_t, uint32_t>> points;  // (xy2, W) in sorted order
  apriltag_vulkan::MinMaxExtentsGpu extents;
};

void SetEnv(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value != nullptr ? value : "");
#else
  if (value != nullptr) setenv(name, value, 1); else unsetenv(name);
#endif
}

std::vector<Blob> Collect(const apriltag_vulkan::GpuDetector &d) {
  std::map<uint32_t, Blob> by_index;
  for (const RawLineFitPoint &p : d.last_line_fit_points) {
    by_index[p.blob_index()].points.push_back({p.xy2, static_cast<uint32_t>(p.W())});
  }
  std::vector<Blob> blobs;
  for (auto &[index, blob] : by_index) {
    blob.extents = d.last_selected_extents.at(index);
    blobs.push_back(std::move(blob));
  }
  // Canonical order: by the smallest point, then the point count.
  auto key = [](const Blob &b) {
    return std::make_pair(*std::min_element(b.points.begin(), b.points.end()), b.points.size());
  };
  std::sort(blobs.begin(), blobs.end(), [&](const Blob &a, const Blob &b) { return key(a) < key(b); });
  return blobs;
}

bool SameExtents(const apriltag_vulkan::MinMaxExtentsGpu &a, const apriltag_vulkan::MinMaxExtentsGpu &b) {
  return a.min_x == b.min_x && a.min_y == b.min_y && a.max_x == b.max_x && a.max_y == b.max_y && a.count == b.count &&
         a.pxgx_plus_pygy_sum == b.pxgx_plus_pygy_sum && a.gx_sum == b.gx_sum && a.gy_sum == b.gy_sum;
}

// Host-timed GPU phase of frame B, alternating with frame A, with the mode off and with a threshold between
// the two frames' boundary-point counts (every B frame then runs the head, then the rest of the tail).
int Mispredict(const std::string &path_a, const std::string &path_b) {
  apriltag_vulkan::vk::Context ctx;
  const eval::Image a = eval::LoadGrey(path_a), b = eval::LoadGrey(path_b);
  if (a.width != b.width || a.height != b.height) {
    std::cerr << "the frames must have the same size\n";
    return 2;
  }
  apriltag_vulkan::DetectorConfig config;
  config.width = uint32_t(a.width);
  config.height = uint32_t(a.height);
  config.decimation = 2;
  SetEnv("APRILTAG_VK_CPU_TAIL", nullptr);
  apriltag_vulkan::GpuDetector off(ctx, config);
  off.Detect(a.grey.data());
  const uint32_t points_a = off.last_profile().boundary_points;
  off.Detect(b.grey.data());
  const uint32_t points_b = off.last_profile().boundary_points;
  const uint32_t threshold = (points_a + points_b) / 2;
  SetEnv("APRILTAG_VK_CPU_TAIL", "1");
  SetEnv("APRILTAG_VK_CPU_TAIL_MAX_POINTS", std::to_string(threshold).c_str());
  apriltag_vulkan::GpuDetector on(ctx, config);
  SetEnv("APRILTAG_VK_CPU_TAIL", nullptr);
  SetEnv("APRILTAG_VK_CPU_TAIL_MAX_POINTS", nullptr);
  std::vector<double> off_a, off_b, on_a, on_b;
  for (int r = 0; r < 400; ++r) {
    off.Detect(a.grey.data());
    off_a.push_back(off.last_profile().gpu_ms);
    off.Detect(b.grey.data());
    off_b.push_back(off.last_profile().gpu_ms);
    on.Detect(a.grey.data());
    on_a.push_back(on.last_profile().gpu_ms);
    on.Detect(b.grey.data());
    on_b.push_back(on.last_profile().gpu_ms);
  }
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::printf("points A %u, B %u, threshold %u; cpu tail frames %llu, mispredicts %llu of 800 frames\n", points_a,
              points_b, threshold, static_cast<unsigned long long>(on.cpu_tail_frames()),
              static_cast<unsigned long long>(on.cpu_tail_mispredicts()));
  std::printf("gpu phase ms: A off %.4f on %.4f | B off %.4f on %.4f\n", median(off_a), median(on_a), median(off_b),
              median(on_b));
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  std::string list, dir, alt_a, alt_b;
  std::vector<unsigned> decimations = {2, 1};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--list" && i + 1 < argc) list = argv[++i];
    else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
    else if (a == "--mispredict" && i + 2 < argc) {
      alt_a = argv[++i];
      alt_b = argv[++i];
    }
    else if (a == "--decimation" && i + 1 < argc) decimations = {static_cast<unsigned>(std::stoi(argv[++i]))};
    else {
      std::cerr << "usage: cpu_tail_test (--list files.txt | --dir DIR) [--decimation N]\n";
      return 2;
    }
  }
  if (!alt_a.empty()) return Mispredict(alt_a, alt_b);
  if (list.empty() == dir.empty()) {
    std::cerr << "usage: cpu_tail_test (--list files.txt | --dir DIR) [--decimation N]\n";
    return 2;
  }
  const apriltag_vulkan::Family *tf = nullptr;
  if (!setup_tag_family(&tf, "tag36h11")) return 1;

  apriltag_vulkan::vk::Context ctx;
  std::vector<std::string> paths;
  if (!dir.empty()) {
    for (const auto &e : std::filesystem::recursive_directory_iterator(dir)) {
      const std::string ext = e.path().extension().string();
      if (ext == ".png" || ext == ".pgm") paths.push_back(e.path().string());
    }
    std::sort(paths.begin(), paths.end());
  } else {
    paths = eval::ReadList(list);
  }
  size_t threshold_ties = 0, frames = 0, blobs_total = 0, tie_reorders = 0, failures = 0, cpu_frames = 0, det_total = 0;
  double max_corner_diff = 0.0;
  for (unsigned decimation : decimations) {
    std::unique_ptr<apriltag_vulkan::GpuDetector> gpu, cpu;
    std::unique_ptr<apriltag_vulkan::QuadDecode> qd_gpu, qd_cpu;
    std::unique_ptr<apriltag_vulkan::TagDecoder> td_gpu, td_cpu;
    apriltag_vulkan::DetectorConfig config;
    uint32_t cur_w = 0, cur_h = 0;
    for (size_t n = 0; n < paths.size(); ++n) {
      const eval::Image im = eval::LoadGrey(paths[n]);
      if (!gpu || cur_w != uint32_t(im.width) || cur_h != uint32_t(im.height)) {
        cur_w = uint32_t(im.width);
        cur_h = uint32_t(im.height);
        config = apriltag_vulkan::DetectorConfig{};
        config.width = cur_w;
        config.height = cur_h;
        config.decimation = decimation;
        config.tag_width = uint32_t(tf->width_at_border);
        config.reversed_border = tf->reversed_border;
        config.normal_border = !tf->reversed_border;
        SetEnv("APRILTAG_VK_CPU_TAIL", nullptr);
        gpu = std::make_unique<apriltag_vulkan::GpuDetector>(ctx, config);
        SetEnv("APRILTAG_VK_CPU_TAIL", "1");
        SetEnv("APRILTAG_VK_CPU_TAIL_MAX_POINTS", "100000000");
        cpu = std::make_unique<apriltag_vulkan::GpuDetector>(ctx, config);
        SetEnv("APRILTAG_VK_CPU_TAIL", nullptr);
        SetEnv("APRILTAG_VK_CPU_TAIL_MAX_POINTS", nullptr);
        auto pool = apriltag_vulkan::MakeWorkerPool(config);
        qd_gpu = std::make_unique<apriltag_vulkan::QuadDecode>(config, pool);
        qd_cpu = std::make_unique<apriltag_vulkan::QuadDecode>(config, pool);
        apriltag_vulkan::TagDecoderConfig dc;
        dc.families = {tf};
        dc.refine_edges = true;
        td_gpu = std::make_unique<apriltag_vulkan::TagDecoder>(
            dc, decimation, 0u, apriltag_vulkan::RefineEdgesMethod::kProfile, 0.0f, apriltag_vulkan::WorkerPoolOptions(),
            false, config.small_tag_rescue, pool);
        td_cpu = std::make_unique<apriltag_vulkan::TagDecoder>(
            dc, decimation, 0u, apriltag_vulkan::RefineEdgesMethod::kProfile, 0.0f, apriltag_vulkan::WorkerPoolOptions(),
            false, config.small_tag_rescue, pool);
      }
      gpu->Detect(im.grey.data());
      cpu->Detect(im.grey.data());
      ++frames;
      std::vector<Blob> a = Collect(*gpu), b = Collect(*cpu);
      // A blob whose fill ratio sits exactly on a threshold may pass on one side only (division rounding).
      auto smallest = [](const Blob &x) { return *std::min_element(x.points.begin(), x.points.end()); };
      auto on_threshold = [](const Blob &x) {
        const auto &e = x.extents;
        const double fill = double(e.count) / (2.0 * double(e.max_x - e.min_x + e.max_y - e.min_y));
        return std::fabs(fill - 0.5) < 1e-6 || std::fabs(fill - 2.5) < 1e-6;
      };
      for (int side = 0; side < 2; ++side) {
        std::vector<Blob> &mine = side == 0 ? a : b;
        const std::vector<Blob> &other = side == 0 ? b : a;
        for (size_t k = mine.size(); k-- > 0;) {
          const bool found = std::any_of(other.begin(), other.end(), [&](const Blob &y) {
            return smallest(y) == smallest(mine[k]) && y.points.size() == mine[k].points.size();
          });
          if (!found && on_threshold(mine[k])) {
            mine.erase(mine.begin() + static_cast<std::ptrdiff_t>(k));
            ++threshold_ties;
          }
        }
      }
      bool ok = a.size() == b.size();
      bool tie = false;
      if (ok) {
        for (size_t k = 0; k < a.size() && ok; ++k) {
          ok = SameExtents(a[k].extents, b[k].extents) && a[k].points.size() == b[k].points.size();
          if (!ok) {
            const auto &e = a[k].extents;
            const auto &f = b[k].extents;
            std::printf("  blob %zu: gpu n=%zu box %d %d %d %d count %u sum %d gx %d gy %d | cpu n=%zu box %d %d %d %d count %u sum %d gx %d gy %d\n",
                        k, a[k].points.size(), e.min_x, e.min_y, e.max_x, e.max_y, e.count, e.pxgx_plus_pygy_sum, e.gx_sum, e.gy_sum,
                        b[k].points.size(), f.min_x, f.min_y, f.max_x, f.max_y, f.count, f.pxgx_plus_pygy_sum, f.gx_sum, f.gy_sum);
            break;
          }
          if (a[k].points != b[k].points) {
            std::vector<std::pair<uint32_t, uint32_t>> x = a[k].points, y = b[k].points;
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
            ok = x == y;
            tie = tie || ok;
            if (!ok) {
              for (size_t q = 0; q < x.size(); ++q) {
                if (x[q] != y[q]) {
                  std::printf("  blob %zu point %zu: gpu xy2 %08x W %u | cpu xy2 %08x W %u\n", k, q, x[q].first, x[q].second, y[q].first, y[q].second);
                  break;
                }
              }
            }
          }
        }
      }
      blobs_total += a.size();
      if (tie) ++tie_reorders;
      if (!ok) {
        ++failures;
        std::printf("MISMATCH %s decimation %u: gpu %zu blobs, cpu %zu blobs\n", paths[n].c_str(), decimation, a.size(),
                    b.size());
      }
      // Final detections.
      auto quads_a = qd_gpu->Decode(gpu->last_line_fit_points);
      auto quads_b = qd_cpu->Decode(cpu->last_line_fit_points);
      auto dets_a = td_gpu->Decode(quads_a, im.grey.data(), im.width, im.height, config.reversed_border);
      auto dets_b = td_cpu->Decode(quads_b, im.grey.data(), im.width, im.height, config.reversed_border);
      auto order = [](apriltag_vulkan::DetectionList &l) {
        std::sort(l.begin(), l.end(), [](const apriltag_vulkan::Detection &p, const apriltag_vulkan::Detection &q) {
          return std::make_pair(p.id, p.c[0]) < std::make_pair(q.id, q.c[0]);
        });
      };
      order(dets_a);
      order(dets_b);
      det_total += dets_a.size();
      bool dets_ok = dets_a.size() == dets_b.size();
      for (size_t k = 0; dets_ok && k < dets_a.size(); ++k) {
        dets_ok = dets_a[k].id == dets_b[k].id && dets_a[k].hamming == dets_b[k].hamming;
        for (int c = 0; c < 4; ++c) {
          for (int e = 0; e < 2; ++e) {
            max_corner_diff = std::max(max_corner_diff, std::fabs(dets_a[k].p[c][e] - dets_b[k].p[c][e]));
          }
        }
      }
      if (!dets_ok) {
        ++failures;
        std::printf("DETECTIONS DIFFER %s decimation %u: %zu vs %zu\n", paths[n].c_str(), decimation, dets_a.size(),
                    dets_b.size());
      }
      cpu_frames += cpu->cpu_tail_frames() > 0 ? 1 : 0;
    }
    std::printf("decimation %u: cpu tail frames so far %zu of %zu\n", decimation, cpu_frames, frames);
  }
  std::printf("frames %zu, blobs %zu, detections %zu, tie reorders %zu, threshold ties %zu, failures %zu, max corner difference %.3g px\n",
              frames, blobs_total, det_total, tie_reorders, threshold_ties, failures, max_corner_diff);
  if (cpu_frames == 0) {
    std::printf("the CPU tail never ran (device without host-readable boundary buffers): skipped\n");
    return 77;
  }
  return failures == 0 && max_corner_diff < 0.01 ? 0 : 1;
}
