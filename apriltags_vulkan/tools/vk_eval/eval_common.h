// Shared by vk_eval (vkapriltag) and avk_eval (vulkan-apriltag): image list loading, timing statistics
// and the detection dump format ("F <index> <count>" then one line per detection).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image.h"

namespace eval {

struct Image {
  std::string path;
  int width = 0, height = 0;
  std::vector<uint8_t> grey;
};

inline Image LoadGrey(const std::string &path) {
  Image im;
  im.path = path;
  int comp = 0;
  stbi_uc *data = stbi_load(path.c_str(), &im.width, &im.height, &comp, STBI_grey);
  if (!data) throw std::runtime_error("cannot load " + path);
  // Cropped from the top-left to multiples of 4, which vulkan-apriltag's threshold requires.
  const int w = im.width & ~3, h = im.height & ~3;
  im.grey.resize(size_t(w) * h);
  for (int y = 0; y < h; ++y) std::copy(data + size_t(y) * im.width, data + size_t(y) * im.width + w, im.grey.begin() + size_t(y) * w);
  im.width = w;
  im.height = h;
  stbi_image_free(data);
  return im;
}

// One path per line; a directory argument is not supported, so lists are explicit and ordered.
inline std::vector<std::string> ReadList(const std::string &list) {
  std::ifstream in(list);
  if (!in) throw std::runtime_error("cannot open list " + list);
  std::vector<std::string> paths;
  for (std::string line; std::getline(in, line);) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (!line.empty()) paths.push_back(line);
  }
  return paths;
}

inline double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Det {
  int id, hamming;
  float margin;
  double c[2];
  double p[4][2];
};

inline void WriteDets(std::ostream &out, int index, std::vector<Det> dets) {
  std::sort(dets.begin(), dets.end(), [](const Det &a, const Det &b) {
    if (a.id != b.id) return a.id < b.id;
    return a.c[0] != b.c[0] ? a.c[0] < b.c[0] : a.c[1] < b.c[1];
  });
  out << "F " << index << ' ' << dets.size() << '\n';
  char line[512];
  for (const Det &d : dets) {
    std::snprintf(line, sizeof line, "%d %d %.6g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                  d.id, d.hamming, d.margin, d.c[0], d.c[1], d.p[0][0], d.p[0][1], d.p[1][0],
                  d.p[1][1], d.p[2][0], d.p[2][1], d.p[3][0], d.p[3][1]);
    out << line;
  }
}

using Clock = std::chrono::steady_clock;
inline double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Per-image timings: GPU part (upload, dispatch, wait, readback) and CPU part (decode), and the sum.
struct Timings {
  std::vector<double> gpu, cpu, total;
  void Add(double g, double c) {
    gpu.push_back(g);
    cpu.push_back(c);
    total.push_back(g + c);
  }
};

inline void PrintTimings(const std::string &path, const Timings &t, size_t candidates, size_t dets) {
  std::printf("TIME %s gpu=%.4f cpu=%.4f total=%.4f candidates=%zu detections=%zu\n", path.c_str(),
              Median(t.gpu), Median(t.cpu), Median(t.total), candidates, dets);
}

}  // namespace eval
