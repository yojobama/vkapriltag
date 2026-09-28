#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "v4l2_capture.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

extern "C" {
#include "apriltag.h"
}

namespace {

volatile std::sig_atomic_t g_stop = 0;
void HandleSignal(int) { g_stop = 1; }

// Dumps a grayscale frame as a binary PGM (P5) for visual checking.
void DumpPgm(const std::string &path, const std::vector<uint8_t> &gray, uint32_t width,
            uint32_t height) {
  std::ofstream f(path, std::ios::binary);
  f << "P5\n" << width << " " << height << "\n255\n";
  f.write(reinterpret_cast<const char *>(gray.data()), gray.size());
}

}  // namespace

int main(int argc, char **argv) {
  std::string device = "/dev/video0";
  uint32_t width = 1280;
  uint32_t height = 720;
  std::string family_name = "tag36h11";
  std::string dump_path;  // If non-empty, periodically dump the raw frame here.
  int dump_every_n_frames = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--device") {
      device = next("--device");
    } else if (arg == "--width") {
      width = std::stoul(next("--width"));
    } else if (arg == "--height") {
      height = std::stoul(next("--height"));
    } else if (arg == "--family") {
      family_name = next("--family");
    } else if (arg == "--dump") {
      dump_path = next("--dump");
    } else if (arg == "--dump-every") {
      dump_every_n_frames = std::stoi(next("--dump-every"));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, family_name.c_str())) {
    return 1;
  }

  // Tag decode via the `apriltag` C library (no pose estimation); quad_decimate matches the 2x
  // GPU decimation.
  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, tf);
  td->refine_edges = false;  // Set to true to enable edge refinement.

  try {
    apriltag_vulkan::vk::Context ctx;

    apriltag_vulkan::DetectorConfig config;
    config.width = width;
    config.height = height;
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;

    // decimation must match config.decimation, so this is built after config.
    apriltag_vulkan::TagDecoder tag_decoder(td, config.decimation);

    apriltag_vulkan::GpuDetector detector(ctx, config);
    apriltag_vulkan::QuadDecode quad_decode(config);

    apriltag_vulkan::V4l2Capture capture(device, width, height);
    // The driver negotiated a different resolution than requested; fail rather than run a
    // mismatched pipeline.
    if (capture.width() != width || capture.height() != height) {
      std::cerr << "Requested " << width << "x" << height << " but device negotiated "
               << capture.width() << "x" << capture.height() << "; adjust --width/--height."
               << std::endl;
      apriltag_detector_destroy(td);
      teardown_tag_family(&tf, family_name.c_str());
      return 1;
    }

    std::vector<uint8_t> gray;
    uint64_t frame_index = 0;
    std::cout << "AprilTag Vulkan detector running on " << device << " (" << width << "x"
             << height << "), tag family " << family_name << ". Press Ctrl+C to stop."
             << std::endl;

    while (!g_stop) {
      capture.CaptureGrayFrame(gray);

      detector.Detect(gray.data());
      const auto &profile = detector.last_profile();

      const auto t0 = std::chrono::steady_clock::now();
      std::vector<apriltag_vulkan::DetectedQuad> quads =
          quad_decode.Decode(detector.last_line_fit_points);
      const auto t1 = std::chrono::steady_clock::now();

      zarray_t *detections = tag_decoder.Decode(quads, gray.data(), width, height,
                                              config.reversed_border);
      const auto t2 = std::chrono::steady_clock::now();

      const double quaddecode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

      std::cout << "Frame " << frame_index << ": " << quads.size() << " candidate quad(s), "
           << zarray_size(detections) << " decoded tag(s). "
           << "GPU(total=" << profile.total_ms << " ms, upload=" << profile.upload_ms
           << ", label=" << profile.threshold_label_ms << ", boundary=" << profile.boundary_ms
           << ", sort=" << profile.sort_group_ms << ", linefit=" << profile.linefit_ms
           << ", readback=" << profile.readback_ms << " ms; uf_iters="
           << profile.uf_iterations << (profile.uf_converged ? "" : "!")
           << ", raw_blobs=" << profile.raw_blobs
           << ", submits=" << profile.submits
           << ", blobs=" << profile.selected_blobs << ", points=" << profile.points
           << "), CPU(quad_decode=" << quaddecode_ms << " ms, decode=" << decode_ms << " ms)"
               << std::endl;
      print_detections(detections);

      // Pose estimation is not wired up here; see README.md.

      if (!dump_path.empty() && dump_every_n_frames > 0 &&
          frame_index % dump_every_n_frames == 0) {
        DumpPgm(dump_path, gray, width, height);
      }

      ++frame_index;
    }

    std::cout << "Shutting down." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    apriltag_detector_destroy(td);
    teardown_tag_family(&tf, family_name.c_str());
    return 1;
  }

  apriltag_detector_destroy(td);
  teardown_tag_family(&tf, family_name.c_str());
  return 0;
}
