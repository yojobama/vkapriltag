// Estimates the CPU cost of the GPU tail's work (group boundary points by blob, extents, select, sort
// by angle, prefix moments) for synthetic frames of N boundary points in K blobs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct Point { uint16_t x, y; uint64_t key; };
struct Blob { uint64_t key; uint32_t count; uint16_t x0, y0, x1, y1; uint32_t offset; uint32_t cursor; bool selected; };
struct Moments { float mx, my, mxx, mxy, myy, w; };

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4500;
  const int k = argc > 2 ? std::atoi(argv[2]) : 12;
  const double select_share = argc > 3 ? std::atof(argv[3]) : 0.33;
  const int reps = 2000;
  std::mt19937 rng(1);
  // Blobs are square outlines of random size; points interleaved as a raster scan would emit them.
  std::vector<Point> pts;
  std::vector<uint8_t> image(1280 * 720, 128);
  for (int b = 0; b < k; ++b) {
    const int per = n / k, side = std::max(4, per / 4);
    const int cx = 100 + rng() % 1000, cy = 100 + rng() % 500;
    const uint64_t key = (uint64_t(rng()) << 32) | rng();
    for (int i = 0; i < per; ++i) {
      const int e = i % 4, t = (i / 4) % side;
      const int x = e == 0 ? t : e == 1 ? side : e == 2 ? side - t : 0, y = e == 0 ? 0 : e == 1 ? t : e == 2 ? side : side - t;
      pts.push_back({uint16_t(cx + x - side / 2), uint16_t(cy + y - side / 2), key});
    }
  }
  std::sort(pts.begin(), pts.end(), [](const Point &a, const Point &b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });

  const uint32_t table = 1024, mask = table - 1;
  std::vector<int32_t> slot(table);
  std::vector<Blob> blobs;
  std::vector<uint32_t> owner(pts.size());
  std::vector<uint64_t> keyed(pts.size());
  std::vector<Moments> out(pts.size());
  std::vector<double> times;
  double sink = 0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    std::fill(slot.begin(), slot.end(), -1);
    blobs.clear();
    // Group and reduce extents.
    for (size_t i = 0; i < pts.size(); ++i) {
      const Point &p = pts[i];
      uint32_t h = uint32_t((p.key * 0x9E3779B97F4A7C15ull) >> 40) & mask;
      while (slot[h] >= 0 && blobs[slot[h]].key != p.key) h = (h + 1) & mask;
      if (slot[h] < 0) {
        slot[h] = int32_t(blobs.size());
        blobs.push_back({p.key, 0, p.x, p.y, p.x, p.y, 0, 0, false});
      }
      Blob &b = blobs[slot[h]];
      b.count++;
      b.x0 = std::min(b.x0, p.x); b.x1 = std::max(b.x1, p.x);
      b.y0 = std::min(b.y0, p.y); b.y1 = std::max(b.y1, p.y);
      owner[i] = uint32_t(slot[h]);
    }
    // Select and assign offsets.
    uint32_t total = 0;
    for (size_t b = 0; b < blobs.size(); ++b) {
      blobs[b].selected = double(b) < select_share * double(blobs.size()) && blobs[b].count >= 24;
      blobs[b].offset = blobs[b].cursor = total;
      if (blobs[b].selected) total += blobs[b].count;
    }
    // Scatter with an angle key about the box centre (slope pseudo-angle, as the sort shader's key).
    for (size_t i = 0; i < pts.size(); ++i) {
      Blob &b = blobs[owner[i]];
      if (!b.selected) continue;
      const float dx = float(pts[i].x) - 0.5f * float(b.x0 + b.x1), dy = float(pts[i].y) - 0.5f * float(b.y0 + b.y1);
      const float ax = std::fabs(dx), ay = std::fabs(dy);
      float a = ay / (ax + ay + 1e-9f);
      if (dx < 0) a = 2.0f - a;
      if (dy < 0) a = 4.0f - a;
      keyed[b.cursor++] = (uint64_t(uint32_t(a * 1.0e6f)) << 32) | uint32_t(i);
    }
    // Sort each blob and build prefix moments weighted by a gradient read from the image.
    for (const Blob &b : blobs) {
      if (!b.selected) continue;
      std::sort(keyed.begin() + b.offset, keyed.begin() + b.offset + b.count);
      Moments m{0, 0, 0, 0, 0, 0};
      for (uint32_t j = b.offset; j < b.offset + b.count; ++j) {
        const Point &p = pts[uint32_t(keyed[j])];
        const uint8_t *px = &image[size_t(p.y) * 1280 + p.x];
        const float w = 1.0f + std::fabs(float(px[1]) - float(px[-1])) + std::fabs(float(px[1280]) - float(px[-1280]));
        const float x = float(p.x), y = float(p.y);
        m.mx += w * x; m.my += w * y; m.mxx += w * x * x; m.mxy += w * x * y; m.myy += w * y * y; m.w += w;
        out[j] = m;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    sink += out[total ? total - 1 : 0].w;
  }
  std::sort(times.begin(), times.end());
  std::printf("n=%zu blobs=%d selected_share=%.2f  median %.1f us  p95 %.1f us  (sink %.0f)\n", pts.size(), k, select_share,
              times[times.size() / 2], times[times.size() * 95 / 100], sink);
  return 0;
}
