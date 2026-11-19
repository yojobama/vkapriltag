#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "v4l2_capture.h"
#include "vkapriltag/Detector.h"

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

// A positive integer argument up to 65535; throws std::invalid_argument otherwise.
uint32_t ParsePositive(const char *name, const std::string &text) {
  size_t used = 0;
  unsigned long value = 0;
  try {
    value = std::stoul(text, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used == 0 || used != text.size() || text[0] == '-' || value == 0 || value > 65535) {
    throw std::invalid_argument(std::string(name) + " needs a positive integer, not '" + text +
                                "'");
  }
  return static_cast<uint32_t>(value);
}

}  // namespace

int main(int argc, char **argv) {
  std::string device = "/dev/video0";
  uint32_t width = 1280;
  uint32_t height = 720;
  std::string family_name = "tag36h11";
  std::string dump_path;  // If non-empty, periodically dump the raw frame here.
  int dump_every_n_frames = 0;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next = [&](const char *name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--device") {
        device = next("--device");
      } else if (arg == "--width") {
        width = ParsePositive("--width", next("--width"));
      } else if (arg == "--height") {
        height = ParsePositive("--height", next("--height"));
      } else if (arg == "--family") {
        family_name = next("--family");
      } else if (arg == "--dump") {
        dump_path = next("--dump");
      } else if (arg == "--dump-every") {
        dump_every_n_frames =
            static_cast<int>(ParsePositive("--dump-every", next("--dump-every")));
      } else {
        throw std::invalid_argument("unknown argument: " + arg);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  try {
    // The camera first: the detector is built for the size the driver actually negotiated.
    apriltag_vulkan::V4l2Capture capture(device, width, height);
    if (capture.width() != width || capture.height() != height) {
      std::cerr << "Requested " << width << "x" << height << " but device negotiated "
               << capture.width() << "x" << capture.height() << "; adjust --width/--height."
               << std::endl;
      return 1;
    }

    apriltag_vulkan::DetectorOptions options;
    options.width = width;
    options.height = height;
    options.families = {family_name};
    options.refine_edges = false;  // Set to true to refine each quad's corners before decoding.
    apriltag_vulkan::Detector detector(options);

    std::vector<uint8_t> gray;
    uint64_t frame_index = 0;
    std::cout << "AprilTag Vulkan detector running on " << device << " (" << width << "x"
             << height << "), tag family " << family_name << ". Press Ctrl+C to stop."
             << std::endl;

    while (!g_stop) {
      // A short timeout keeps Ctrl+C responsive when the stream stalls.
      if (!capture.CaptureGrayFrame(gray, 200)) continue;

      const apriltag_vulkan::Result &result = detector.Detect(gray.data());
      const apriltag_vulkan::Timing &timing = result.timing;

      std::cout << "Frame " << frame_index << ": " << timing.candidates << " candidate quad(s), "
                << result.detections.size() << " decoded tag(s). GPU(total=" << timing.gpu_ms
                << " ms, submits=" << timing.submits << "), CPU(quad_decode=" << timing.quad_ms
                << " ms, decode=" << timing.decode_ms << " ms)" << std::endl;
      for (const apriltag_vulkan::Detection &d : result.detections) {
        std::cout << "  id=" << d.id << " hamming=" << d.hamming << " margin=" << d.decision_margin
                  << " centre=(" << d.c[0] << ", " << d.c[1] << ")" << std::endl;
      }

      if (!dump_path.empty() && dump_every_n_frames > 0 &&
          frame_index % dump_every_n_frames == 0) {
        DumpPgm(dump_path, gray, width, height);
      }

      ++frame_index;
    }

    std::cout << "Shutting down." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
