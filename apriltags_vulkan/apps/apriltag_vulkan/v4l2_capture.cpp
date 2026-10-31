#include "v4l2_capture.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace apriltag_vulkan {
namespace {

int XIoctl(int fd, unsigned long request, void *arg) {
  int r;
  do {
    r = ioctl(fd, request, arg);
  } while (r == -1 && errno == EINTR);
  return r;
}

void CheckIoctl(int fd, unsigned long request, void *arg, const char *what) {
  if (XIoctl(fd, request, arg) == -1) {
    throw std::runtime_error(std::string(what) + " failed: " + std::strerror(errno));
  }
}

}  // namespace

V4l2Capture::V4l2Capture(const std::string &device, uint32_t width, uint32_t height)
    : width_(width), height_(height) {
  fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open " + device + ": " + std::strerror(errno));
  }
  // Releases the descriptor and mappings if a later step throws.
  struct Cleanup {
    V4l2Capture *self;
    ~Cleanup() {
      if (self != nullptr) self->Release();
    }
  } cleanup{this};

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = width_;
  fmt.fmt.pix.height = height_;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  CheckIoctl(fd_, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT");

  // The driver may adjust width/height/format; reflect that back.
  width_ = fmt.fmt.pix.width;
  height_ = fmt.fmt.pix.height;
  stride_ = fmt.fmt.pix.bytesperline != 0 ? fmt.fmt.pix.bytesperline : 2 * width_;
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    throw std::runtime_error(device + " does not support YUYV capture");
  }
  if (stride_ < 2 * width_) {
    throw std::runtime_error(device + " reports a row shorter than its width");
  }

  constexpr uint32_t kNumBuffers = 4;
  v4l2_requestbuffers req{};
  req.count = kNumBuffers;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  CheckIoctl(fd_, VIDIOC_REQBUFS, &req, "VIDIOC_REQBUFS");

  buffers_.resize(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    CheckIoctl(fd_, VIDIOC_QUERYBUF, &buf, "VIDIOC_QUERYBUF");

    if (buf.length < static_cast<size_t>(stride_) * height_) {
      throw std::runtime_error("V4L2 buffer smaller than one frame");
    }
    void *start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (start == MAP_FAILED) {
      throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }
    buffers_[i].start = start;
    buffers_[i].length = buf.length;

    CheckIoctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF (initial queue)");
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  CheckIoctl(fd_, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
  cleanup.self = nullptr;
}

V4l2Capture::~V4l2Capture() { Release(); }

void V4l2Capture::Release() {
  if (fd_ < 0) return;
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  XIoctl(fd_, VIDIOC_STREAMOFF, &type);
  for (auto &b : buffers_) {
    if (b.start) munmap(b.start, b.length);
  }
  buffers_.clear();
  close(fd_);
  fd_ = -1;
}

bool V4l2Capture::CaptureGrayFrame(std::vector<uint8_t> &out, int timeout_ms) {
  pollfd pfd{fd_, POLLIN, 0};
  const int ready = poll(&pfd, 1, timeout_ms);
  if (ready < 0 && errno != EINTR) {
    throw std::runtime_error(std::string("poll failed: ") + std::strerror(errno));
  }
  if (ready <= 0) return false;

  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (XIoctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
    if (errno == EAGAIN) return false;
    throw std::runtime_error(std::string("VIDIOC_DQBUF failed: ") + std::strerror(errno));
  }

  // A buffer the driver flagged, or one shorter than a frame, is requeued unread.
  const size_t frame_bytes = static_cast<size_t>(stride_) * height_;
  const bool usable = buf.index < buffers_.size() && (buf.flags & V4L2_BUF_FLAG_ERROR) == 0 &&
                      (buf.bytesused == 0 || buf.bytesused >= frame_bytes);
  if (usable) {
    out.resize(static_cast<size_t>(width_) * height_);
    const uint8_t *yuyv = static_cast<const uint8_t *>(buffers_[buf.index].start);
    // YUYV: bytes are [Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...] - luma is every other byte.
    for (uint32_t y = 0; y < height_; ++y) {
      const uint8_t *row = yuyv + static_cast<size_t>(y) * stride_;
      uint8_t *dst = out.data() + static_cast<size_t>(y) * width_;
      for (uint32_t x = 0; x < width_; ++x) dst[x] = row[2 * x];
    }
  }

  CheckIoctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF (requeue)");
  return usable;
}

}  // namespace apriltag_vulkan
