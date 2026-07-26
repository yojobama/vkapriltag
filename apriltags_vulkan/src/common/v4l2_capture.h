#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace apriltag_vulkan {

// Minimal V4L2 mmap-mode capture device, replacing OpenCV's VideoCapture
// (which the original CUDA implementation used). Requests a YUYV (YUV
// 4:2:2) stream at the given resolution - the vast majority of USB/UVC
// cameras support this format natively - and exposes only the luma (Y)
// plane, which is all the AprilTag detector needs.
class V4l2Capture {
 public:
  V4l2Capture(const std::string &device, uint32_t width, uint32_t height);
  ~V4l2Capture();

  V4l2Capture(const V4l2Capture &) = delete;
  V4l2Capture &operator=(const V4l2Capture &) = delete;

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  // Blocks until the next frame is available, extracts its luma (grayscale)
  // plane into `out` (resized to width() * height()), and returns.  Throws
  // std::runtime_error on any V4L2 failure.
  void CaptureGrayFrame(std::vector<uint8_t> &out);

 private:
  struct MappedBuffer {
    void *start = nullptr;
    size_t length = 0;
  };

  int fd_ = -1;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  std::vector<MappedBuffer> buffers_;
};

}  // namespace apriltag_vulkan
