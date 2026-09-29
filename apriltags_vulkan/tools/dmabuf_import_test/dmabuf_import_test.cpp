// Tests zero-copy input through dma-buf import (Linux).
//   dmabuf_import_test --pgm <file> [--iterations N]           (dma-heap buffer filled from a PGM)
//   dmabuf_import_test --camera /dev/video0 [--width W --height H] [--frames N]
// Each path runs Detect() on a normal copied frame and on the imported buffer, and requires the
// resulting line-fit points to be identical; timings are host wall time per Detect() including the
// work needed to make the frame available (the luma extraction for the camera's YUYV).
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void SyncDmaBuf(int fd, uint64_t flags) {
  dma_buf_sync sync{};
  sync.flags = flags;
  ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

// Point append order varies between runs, so paths are compared by their fitted quads, which do
// not depend on it.
using Quads = std::vector<apriltag_vulkan::DetectedQuad>;

Quads SortedQuads(const Quads &q) {
  Quads out = q;
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    return a.p[0][0] != b.p[0][0] ? a.p[0][0] < b.p[0][0] : a.p[0][1] < b.p[0][1];
  });
  return out;
}

bool SameQuads(const Quads &x, const Quads &y) {
  if (x.size() != y.size()) return false;
  const Quads a = SortedQuads(x), b = SortedQuads(y);
  for (size_t i = 0; i < a.size(); ++i) {
    for (int k = 0; k < 4; ++k) {
      for (int c = 0; c < 2; ++c) {
        if (std::abs(a[i].p[k][c] - b[i].p[k][c]) > 1e-3) return false;
      }
    }
  }
  return true;
}

apriltag_vulkan::DetectorConfig MakeConfig(uint32_t w, uint32_t h, const apriltag_family_t *tf) {
  apriltag_vulkan::DetectorConfig config;
  config.width = w;
  config.height = h;
  config.decimation = 2;
  config.tag_width = static_cast<uint32_t>(tf->width_at_border);
  config.reversed_border = tf->reversed_border;
  config.normal_border = !tf->reversed_border;
  return config;
}

int AllocHeap(size_t bytes) {
  int heap = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
  if (heap < 0) return -1;
  dma_heap_allocation_data alloc{};
  alloc.len = bytes;
  alloc.fd_flags = O_RDWR | O_CLOEXEC;
  const int r = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &alloc);
  close(heap);
  return r < 0 ? -1 : static_cast<int>(alloc.fd);
}

int RunHeap(const std::string &pgm, int iterations, apriltag_family_t *tf, bool yuyv) {
  std::vector<uint8_t> gray;
  uint32_t w = 0, h = 0;
  if (!apriltag_vulkan::LoadGrayPgm(pgm, &gray, &w, &h)) {
    std::cerr << "Failed to load " << pgm << std::endl;
    return 1;
  }
  const size_t bytes = ((yuyv ? 2 : 1) * gray.size() + 4095) / 4096 * 4096;
  const int fd = AllocHeap(bytes);
  if (fd < 0) {
    std::cerr << "dma-heap allocation failed\n";
    return 1;
  }
  auto *mapped = static_cast<uint8_t *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (mapped == MAP_FAILED) {
    std::cerr << "mmap of the dma-buf failed\n";
    return 1;
  }
  SyncDmaBuf(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
  if (yuyv) {
    for (size_t p = 0; p < gray.size(); ++p) {
      mapped[2 * p] = gray[p];
      mapped[2 * p + 1] = 128;
    }
  } else {
    std::memcpy(mapped, gray.data(), gray.size());
  }
  SyncDmaBuf(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

  apriltag_vulkan::vk::Context ctx;
  const apriltag_vulkan::DetectorConfig config = MakeConfig(w, h, tf);
  apriltag_vulkan::GpuDetector detector(ctx, config);
  const apriltag_vulkan::QuadDecode quad_decode(config);
  if (!ctx.caps().has_external_memory_dma_buf) {
    std::cerr << "device lacks dma-buf import\n";
    return 1;
  }
  detector.ImportDmaBufFrame(mapped, fd, bytes, yuyv);

  std::vector<double> copy_ms, import_ms;
  bool identical = true;
  for (int i = 0; i < iterations + 5; ++i) {
    auto t0 = Clock::now();
    detector.Detect(gray.data());
    const double c = MsSince(t0);
    const Quads ref = quad_decode.Decode(detector.last_line_fit_points);
    t0 = Clock::now();
    detector.Detect(mapped);
    const double m = MsSince(t0);
    identical = identical && SameQuads(ref, quad_decode.Decode(detector.last_line_fit_points));
    if (i >= 5) {
      copy_ms.push_back(c);
      import_ms.push_back(m);
    }
  }
  std::sort(copy_ms.begin(), copy_ms.end());
  std::sort(import_ms.begin(), import_ms.end());
  std::cout << "heap dma-buf: line-fit points identical to the copy path: "
            << (identical ? "YES" : "NO") << "\n  Detect() median: copy=" << copy_ms[copy_ms.size() / 2]
            << " ms, import=" << import_ms[import_ms.size() / 2] << " ms ("
            << detector.last_line_fit_points.size() << " points)\n";
  return identical ? 0 : 2;
}

struct CamBuf {
  uint8_t *start = nullptr;
  size_t length = 0;
  int dmabuf = -1;
};

int RunCamera(const std::string &dev, uint32_t want_w, uint32_t want_h, int frames,
              apriltag_family_t *tf, int sync_mode) {
  const int fd = open(dev.c_str(), O_RDWR);
  if (fd < 0) {
    std::cerr << "cannot open " << dev << std::endl;
    return 1;
  }
  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = want_w;
  fmt.fmt.pix.height = want_h;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0 || fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    std::cerr << "camera does not provide YUYV at the requested size\n";
    return 1;
  }
  const uint32_t w = fmt.fmt.pix.width, h = fmt.fmt.pix.height;
  std::cout << "camera format: YUYV " << w << "x" << h << " stride " << fmt.fmt.pix.bytesperline
            << std::endl;
  if (fmt.fmt.pix.bytesperline != w * 2 || w % 2 != 0 || h % 2 != 0) {
    std::cerr << "row padding or odd size not supported by this test\n";
    return 1;
  }

  v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
    std::cerr << "REQBUFS failed\n";
    return 1;
  }
  std::vector<CamBuf> bufs(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    ioctl(fd, VIDIOC_QUERYBUF, &b);
    bufs[i].length = b.length;
    bufs[i].start = static_cast<uint8_t *>(
        mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, b.m.offset));
    v4l2_exportbuffer ex{};
    ex.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ex.index = i;
    ex.flags = O_RDWR | O_CLOEXEC;
    if (ioctl(fd, VIDIOC_EXPBUF, &ex) < 0) {
      std::cerr << "VIDIOC_EXPBUF failed: " << std::strerror(errno) << std::endl;
      return 1;
    }
    bufs[i].dmabuf = ex.fd;
  }

  apriltag_vulkan::vk::Context ctx;
  const apriltag_vulkan::DetectorConfig config = MakeConfig(w, h, tf);
  apriltag_vulkan::GpuDetector detector(ctx, config);
  const apriltag_vulkan::QuadDecode quad_decode(config);
  if (!ctx.caps().has_external_memory_dma_buf) {
    std::cerr << "device lacks dma-buf import\n";
    return 1;
  }
  try {
    for (CamBuf &b : bufs) detector.ImportDmaBufFrame(b.start, b.dmabuf, b.length, /*yuyv=*/true);
  } catch (const std::exception &e) {
    std::cerr << "dma-buf import of a V4L2 capture buffer failed: " << e.what() << std::endl;
    return 3;
  }
  std::cout << "imported " << bufs.size() << " capture buffers as Vulkan buffers\n";

  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    ioctl(fd, VIDIOC_QBUF, &b);
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd, VIDIOC_STREAMON, &type);

  std::vector<uint8_t> gray(static_cast<size_t>(w) * h);
  std::vector<double> copy_ms, import_ms;
  int identical = 0, differing = 0;
  double points = 0;
  for (int i = 0; i < frames + 5; ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_DQBUF, &b) < 0) {
      std::cerr << "DQBUF failed: " << std::strerror(errno) << std::endl;
      return 1;
    }
    CamBuf &cb = bufs[b.index];
    // The kernel filled the buffer through the CPU cache; flush it for the GPU (which is not
    // cache-coherent with the CPU).
    if (sync_mode == 1) SyncDmaBuf(cb.dmabuf, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
    if (sync_mode == 2) {
      SyncDmaBuf(cb.dmabuf, DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW);
      SyncDmaBuf(cb.dmabuf, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
    }

    auto t0 = Clock::now();  // current path: CPU luma extraction + copy upload
    for (size_t p = 0; p < gray.size(); ++p) gray[p] = cb.start[2 * p];
    detector.Detect(gray.data());
    const double c = MsSince(t0);
    const Quads ref = quad_decode.Decode(detector.last_line_fit_points);
    t0 = Clock::now();  // zero-copy path: the GPU reads the capture buffer in place
    detector.Detect(cb.start);
    const double m = MsSince(t0);
    if (i >= 5) {
      copy_ms.push_back(c);
      import_ms.push_back(m);
      points += static_cast<double>(ref.size());
      (SameQuads(ref, quad_decode.Decode(detector.last_line_fit_points)) ? identical : differing)++;
    }
    ioctl(fd, VIDIOC_QBUF, &b);
  }
  ioctl(fd, VIDIOC_STREAMOFF, &type);
  std::sort(copy_ms.begin(), copy_ms.end());
  std::sort(import_ms.begin(), import_ms.end());
  std::cout << "camera dma-buf import: frames identical to the copy path: " << identical << " / "
            << identical + differing << " (mean " << points / std::max(1, identical + differing)
            << " line-fit points per frame)\n  host time per frame (median): copy path (luma + upload + Detect)="
            << copy_ms[copy_ms.size() / 2] << " ms, import path (Detect)="
            << import_ms[import_ms.size() / 2] << " ms\n";
  return differing == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv) {
  std::string pgm, camera;
  int iterations = 200, frames = 60, sync_mode = 1;
  bool yuyv_heap = false;
  uint32_t width = 640, height = 480;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (arg == "--pgm") pgm = next();
    else if (arg == "--camera") camera = next();
    else if (arg == "--iterations") iterations = std::stoi(next());
    else if (arg == "--yuyv") yuyv_heap = true;
    else if (arg == "--frames") frames = std::stoi(next());
    else if (arg == "--sync") sync_mode = std::stoi(next());
    else if (arg == "--width") width = static_cast<uint32_t>(std::stoi(next()));
    else if (arg == "--height") height = static_cast<uint32_t>(std::stoi(next()));
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }
  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, "tag36h11")) return 1;
  try {
    if (!camera.empty()) return RunCamera(camera, width, height, frames, tf, sync_mode);
    if (!pgm.empty()) return RunHeap(pgm, iterations, tf, yuyv_heap);
  } catch (const std::exception &e) {
    std::cerr << "Failed: " << e.what() << std::endl;
    return 1;
  }
  std::cerr << "Usage: dmabuf_import_test --pgm <file> | --camera /dev/video0 [--width W --height H]\n";
  return 1;
}
