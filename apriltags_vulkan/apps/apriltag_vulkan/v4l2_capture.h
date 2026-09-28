#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace apriltag_vulkan {

// Minimal V4L2 mmap-mode capture device requesting a YUYV stream and exposing only the luma plane.
class V4l2Capture {
 public:
  V4l2Capture(const std::string &device, uint32_t width, uint32_t height);
  ~V4l2Capture();

  V4l2Capture(const V4l2Capture &) = delete;
  V4l2Capture &operator=(const V4l2Capture &) = delete;

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  // Blocks until the next frame, extracts its luma plane into `out` (resized to width() *
  // height()); throws std::runtime_error on V4L2 failure.
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
