// Records the GPU work of the first frame (dispatches, barriers, fills, copies, submits) and compares it with a
// committed trace for this device; the trace must not change when the frame recording is restructured.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "eval_common.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

namespace {

std::string Slug(const std::string &name) {
  std::string out;
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      out += c;
    } else if (!out.empty() && out.back() != '-') {
      out += '-';
    }
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out;
}

std::vector<std::string> Lines(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

}  // namespace

int main(int argc, char **argv) {
  std::string image, golden_root, profile = "default";
  unsigned decimation = 2;
  bool update = std::getenv("GPU_TRACE_UPDATE") != nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--image") image = next();
    else if (a == "--golden-root") golden_root = next();
    else if (a == "--profile") profile = next();
    else if (a == "--decimation") decimation = static_cast<unsigned>(std::stoi(next()));
    else if (a == "--update") update = true;
    else throw std::runtime_error("unknown argument " + a);
  }
  if (image.empty() || golden_root.empty()) {
    std::cerr << "usage: gpu_trace_test --image PNG --golden-root DIR [--profile NAME] [--decimation N] [--update]\n";
    return 2;
  }

  apriltag_vulkan::vk::Context ctx;
  const std::filesystem::path dir = std::filesystem::path(golden_root) / Slug(ctx.caps().name);
  if (!update && !std::filesystem::is_directory(dir)) {
    std::cout << "no trace goldens for device " << ctx.caps().name << " (" << dir.string() << "): skipped\n";
    return 77;
  }

  const eval::Image frame = eval::LoadGrey(image);
  apriltag_vulkan::DetectorConfig config;
  config.width = static_cast<uint32_t>(frame.width);
  config.height = static_cast<uint32_t>(frame.height);
  config.decimation = decimation;

  apriltag_vulkan::GpuDetector detector(ctx, config);
  apriltag_vulkan::vk::DispatchTrace::Start();
  detector.Detect(frame.grey.data());
  const std::string trace = apriltag_vulkan::vk::DispatchTrace::Stop();

  const std::filesystem::path path = dir / (profile + ".txt");
  if (update) {
    std::filesystem::create_directories(dir);
    std::ofstream(path, std::ios::binary) << trace;
    std::cout << "wrote " << path.string() << " (" << Lines(trace).size() << " lines)\n";
    return 0;
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "missing golden " << path.string() << "\n";
    return 1;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::vector<std::string> want = Lines(buffer.str()), got = Lines(trace);
  const size_t n = std::min(want.size(), got.size());
  for (size_t i = 0; i < n; ++i) {
    if (want[i] != got[i]) {
      std::cerr << "trace differs at line " << i + 1 << ":\n  golden: " << want[i] << "\n  now:    " << got[i] << "\n";
      return 1;
    }
  }
  if (want.size() != got.size()) {
    std::cerr << "trace length differs: golden " << want.size() << " lines, now " << got.size() << "\n";
    return 1;
  }
  std::cout << "trace matches (" << got.size() << " lines)\n";
  return 0;
}
