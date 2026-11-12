#include <thread>
#include <chrono>
#include <cstdlib>
// vkapriltag driven like avk_eval: the same images, timing loop and dump format.
// Per frame: Detect() (upload, dispatch, wait, readback) as the GPU part, then QuadDecode + TagDecoder.
// --ref runs the unmodified libapriltag CPU detector instead (apriltag_detector_detect).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "eval_common.h"

#include <memory>
#include <thread>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/common/Options.h"
#include "vkapriltag/common/Profiler.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/MultiLevelDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

extern "C" {
#include "apriltag.h"
}

namespace {

std::vector<eval::Det> ToDets(zarray_t *z) {
  std::vector<eval::Det> dets;
  for (int i = 0; i < zarray_size(z); ++i) {
    apriltag_detection_t *d;
    zarray_get(z, i, &d);
    eval::Det e{d->id, d->hamming, d->decision_margin, {d->c[0], d->c[1]}, {}};
    for (int k = 0; k < 4; ++k) {
      e.p[k][0] = d->p[k][0];
      e.p[k][1] = d->p[k][1];
    }
    dets.push_back(e);
  }
  return dets;
}

}  // namespace

int main(int argc, char **argv) {
  std::string list, out_path;
  int iterations = 1, warmup = 0;
  uint32_t decimation = 1, threads = 0;
  bool ref = false;
  std::string family = "tag36h11";
  std::string levels_arg;  // "4b:28:0,2b:14:40,1:8:20": decimation[b]:min tag px:max tag px per level
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--list") list = next();
    else if (a == "--out") out_path = next();
    else if (a == "--iterations") iterations = std::max(1, std::stoi(next()));
    else if (a == "--warmup") warmup = std::max(0, std::stoi(next()));
    else if (a == "--decimation") decimation = uint32_t(std::max(1, std::stoi(next())));
    else if (a == "--threads") threads = uint32_t(std::max(0, std::stoi(next())));
    else if (a == "--ref") ref = true;
    else if (a == "--family") family = next();
    else if (a == "--levels") levels_arg = next();
    else if (a == "--list-options") {
      std::fputs(apriltag_vulkan::OptionsMarkdown().c_str(), stdout);
      return 0;
    } else if (a == "--list-option-names") {
      for (const auto &d : apriltag_vulkan::OptionTable()) std::printf("%s\n", d.name);
      return 0;
    } else {
      std::cerr << "usage: vk_eval --list files.txt [--out dets.txt] [--iterations N] [--warmup N] "
                   "[--decimation N] [--threads N] [--ref] | --list-options | --list-option-names\n";
      return 2;
    }
  }
  const bool tracing = apriltag_vulkan::prof::EnableFromEnv();
  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, family.c_str())) return 1;
  int rc = 0;
  try {
    const std::vector<std::string> paths = eval::ReadList(list);
    std::ofstream out;
    if (!out_path.empty()) out.open(out_path, std::ios::trunc);
    std::vector<double> all_total;

    if (ref) {
      apriltag_detector_t *td = apriltag_detector_create();
      apriltag_detector_add_family(td, tf);
      td->quad_decimate = float(decimation);
      td->nthreads = int(threads ? threads : std::max(1u, std::thread::hardware_concurrency()));
      std::printf("libapriltag CPU, quad_decimate %u, %d threads\n", decimation, td->nthreads);
      for (size_t n = 0; n < paths.size(); n++) {
        eval::Image im = eval::LoadGrey(paths[n]);
        image_u8_t u8{im.width, im.height, im.width, im.grey.data()};
        eval::Timings t;
        zarray_t *dets = nullptr;
        for (int it = 0; it < warmup + iterations; it++) {
          if (dets) apriltag_detections_destroy(dets);
          if (const char *g = std::getenv("VK_EVAL_GAP_MS")) std::this_thread::sleep_for(std::chrono::microseconds(int(std::atof(g) * 1000)));
          const auto t0 = eval::Clock::now();
          dets = apriltag_detector_detect(td, &u8);
          const auto t1 = eval::Clock::now();
          if (it >= warmup) t.Add(0.0, eval::Ms(t0, t1));
        }
        if (iterations > 1) eval::PrintTimings(paths[n], t, size_t(td->nquads), size_t(zarray_size(dets)));
        all_total.push_back(eval::Median(t.total));
        if (out.is_open()) eval::WriteDets(out, int(n), ToDets(dets));
        apriltag_detections_destroy(dets);
      }
      apriltag_detector_destroy(td);
    } else {
      apriltag_vulkan::vk::Context ctx;
      apriltag_detector_t *td = apriltag_detector_create();
      apriltag_detector_add_family(td, tf);
      // As apriltag_vulkan_validate: upstream's default refine_edges, decoded with the default method.
      td->refine_edges = true;
      std::unique_ptr<apriltag_vulkan::GpuDetector> detector;
      std::unique_ptr<apriltag_vulkan::MultiLevelDetector> multi;
      std::vector<apriltag_vulkan::LevelSpec> level_specs;
      for (size_t pos = 0; pos < levels_arg.size();) {
        size_t end = levels_arg.find(',', pos);
        if (end == std::string::npos) end = levels_arg.size();
        const std::string item = levels_arg.substr(pos, end - pos);
        apriltag_vulkan::LevelSpec spec;
        char *rest = nullptr;
        spec.decimation = uint32_t(std::strtoul(item.c_str(), &rest, 10));
        if (*rest == 'b') {
          spec.box = true;
          ++rest;
        }
        if (*rest == ':') spec.min_tag_pixels = uint32_t(std::strtoul(rest + 1, &rest, 10));
        if (*rest == ':') spec.max_tag_pixels = uint32_t(std::strtoul(rest + 1, &rest, 10));
        level_specs.push_back(spec);
        pos = end + 1;
      }
      if (!level_specs.empty()) {
        decimation = 1;
        for (const auto &l : level_specs) decimation = std::max(decimation, l.decimation);
      }
      std::unique_ptr<apriltag_vulkan::QuadDecode> quad_decode;
      std::unique_ptr<apriltag_vulkan::TagDecoder> tag_decoder;
      apriltag_vulkan::DetectorConfig config;
      std::printf("vkapriltag, decimation %u, cpu threads %u (0 = all)\n", decimation, threads);
      for (size_t n = 0; n < paths.size(); n++) {
        eval::Image im = eval::LoadGrey(paths[n]);
        if ((!detector && !multi) || config.width != uint32_t(im.width) || config.height != uint32_t(im.height)) {
          tag_decoder.reset();
          quad_decode.reset();
          detector.reset();
          multi.reset();
          config = apriltag_vulkan::DetectorConfig{};
          config.width = uint32_t(im.width);
          config.height = uint32_t(im.height);
          config.decimation = decimation;
          config.tag_width = uint32_t(tf->width_at_border);
          config.reversed_border = tf->reversed_border;
          config.normal_border = !tf->reversed_border;
          if (!level_specs.empty()) {
            multi = std::make_unique<apriltag_vulkan::MultiLevelDetector>(ctx, config, level_specs);
          } else {
            detector = std::make_unique<apriltag_vulkan::GpuDetector>(ctx, config);
            quad_decode = std::make_unique<apriltag_vulkan::QuadDecode>(config);
          }
          tag_decoder = std::make_unique<apriltag_vulkan::TagDecoder>(td, decimation, threads);
        }
        eval::Timings t;
        zarray_t *dets = nullptr;
        size_t candidates = 0;
        for (int it = 0; it < warmup + iterations; it++) {
          if (const char *g = std::getenv("VK_EVAL_GAP_MS")) std::this_thread::sleep_for(std::chrono::microseconds(int(std::atof(g) * 1000)));
          const auto t0 = eval::Clock::now();
          std::vector<apriltag_vulkan::DetectedQuad> quads;
          if (multi) {
            quads = multi->Detect(im.grey.data());
          } else {
            detector->Detect(im.grey.data());
          }
          const auto t1 = eval::Clock::now();
          if (!multi) quads = quad_decode->Decode(detector->last_line_fit_points);
          dets = tag_decoder->Decode(quads, im.grey.data(), im.width, im.height,
                                     config.reversed_border);
          const auto t2 = eval::Clock::now();
          candidates = quads.size();
          if (it >= warmup) t.Add(eval::Ms(t0, t1), eval::Ms(t1, t2));
        }
        if (iterations > 1) eval::PrintTimings(paths[n], t, candidates, size_t(zarray_size(dets)));
        all_total.push_back(eval::Median(t.total));
        // Owned by the TagDecoder.
        if (out.is_open()) eval::WriteDets(out, int(n), ToDets(dets));
      }
      tag_decoder.reset();
      apriltag_detector_destroy(td);
    }
    std::printf("images=%zu median_total=%.4f ms\n", paths.size(), eval::Median(all_total));
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    rc = 1;
  }
  if (tracing) apriltag_vulkan::prof::WriteTrace(apriltag_vulkan::prof::EnvTracePath());
  teardown_tag_family(&tf, family.c_str());
  return rc;
}
