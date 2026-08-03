#include "v4l2_capture.h"

#include <fcntl.h>
#include <linux/videodev2.h>
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

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = width_;
  fmt.fmt.pix.height = height_;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  CheckIoctl(fd_, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT");

  // The driver may adjust width/height/format to the closest supported mode;
  // reflect that back so downstream buffer sizing stays correct.
  width_ = fmt.fmt.pix.width;
  height_ = fmt.fmt.pix.height;
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    close(fd_);
    throw std::runtime_error(device + " does not support YUYV capture");
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

    buffers_[i].length = buf.length;
    buffers_[i].start =
        mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (buffers_[i].start == MAP_FAILED) {
      throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }

    CheckIoctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF (initial queue)");
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  CheckIoctl(fd_, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
}

V4l2Capture::~V4l2Capture() {
  if (fd_ < 0) return;
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  XIoctl(fd_, VIDIOC_STREAMOFF, &type);
  for (auto &b : buffers_) {
    if (b.start) munmap(b.start, b.length);
  }
  close(fd_);
}

void V4l2Capture::CaptureGrayFrame(std::vector<uint8_t> &out) {
  // Frame isn't ready yet on a non-blocking fd until select()/poll() says so;
  // a simple blocking retry loop keeps this file free of extra dependencies.
  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  while (XIoctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
    if (errno != EAGAIN) {
      throw std::runtime_error(std::string("VIDIOC_DQBUF failed: ") + std::strerror(errno));
    }
  }

  out.resize(static_cast<size_t>(width_) * height_);
  const uint8_t *yuyv = static_cast<const uint8_t *>(buffers_[buf.index].start);
  // YUYV: bytes are [Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...] - luma is every other byte.
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = yuyv[i * 2];
  }

  CheckIoctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF (requeue)");
}

}  // namespace apriltag_vulkan
