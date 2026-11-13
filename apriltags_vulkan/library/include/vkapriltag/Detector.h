#pragma once

// The detector's public interface: one object that owns the Vulkan device, the GPU stages and the CPU tail.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vkapriltag/Detection.h"
#include "vkapriltag/Pose.h"
#include "vkapriltag/Version.h"

namespace apriltag_vulkan {

enum class ErrorCode {
  kInvalidArgument,  // an option or an argument is out of range
  kUnsupported,      // the combination of options or the device cannot do it
  kDevice,           // Vulkan or the driver failed
  kRuntime,          // anything else
};

// Every failure the library reports.
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string &message) : std::runtime_error(message), code_(code) {}
  ErrorCode code() const { return code_; }

 private:
  ErrorCode code_;
};

// How frames are processed.
enum class Mode {
  kSync,       // Detect() returns the detections of the frame it was given
  kPipelined,  // Detect() returns the previous frame's detections while the GPU works on this one (Flush() drains)
  kTracking,   // slow scenes skip the GPU: tags found on keyframes are followed on the frames between
};

// One decimation level of a multi-level detector (levels run one after another and their candidates merge).
struct Level {
  uint32_t decimation = 1;
  bool box = false;               // average each block instead of copying its top-left pixel
  uint32_t min_tag_pixels = 0;    // smallest tag side this level looks for (0: no limit)
  uint32_t max_tag_pixels = 0;    // largest tag side this level looks for (0: no limit)
};

// The GPU the detectors run on. Several detectors (cameras) can share one.
struct DeviceOptions {
  int device_index = -1;          // -1 picks the best device (discrete before integrated)
  bool allow_cpu_device = false;  // permit a software rasteriser such as lavapipe
  uint32_t expected_streams = 1;  // detectors that will run at the same time on this device
  bool verbose = false;           // print the device and its limits when it is opened
};

class Device {
 public:
  static std::shared_ptr<Device> Create(const DeviceOptions &options = DeviceOptions());
  ~Device();
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  // The device's name and type, for example "AMD Radeon RX 9060 XT (discrete GPU, Vulkan 1.4.359)".
  std::string Describe() const;

  struct Impl;
  Impl &impl() { return *impl_; }

 private:
  Device();
  std::unique_ptr<Impl> impl_;
};

struct DetectorOptions {
  uint32_t width = 0;   // frame size in pixels; each must divide by `decimation`
  uint32_t height = 0;
  std::vector<std::string> families = {"tag36h11"};  // tag36h11, tag25h9, tag16h5, tagCircle21h7, tagCircle49h12,
                                                      // tagStandard41h12, tagStandard52h13, tagCustom48h12
  uint32_t decimation = 2;  // the GPU works on the frame reduced by this factor; 1 reads every pixel
  Mode mode = Mode::kSync;
  uint32_t pipeline_depth = 1;     // kPipelined: frames in flight, 1 or 2
  bool tracking_async = false;     // kTracking: find new tags on a background thread
  std::vector<Level> levels;       // several decimations at once; empty uses `decimation` alone (kSync only)

  // Accuracy and recall features.
  bool refine_edges = true;        // refine each candidate's corners before decoding
  bool small_tag_rescue = false;   // find tags below about 24 pixels that the decimated image loses
  bool hull_quad = false;          // recover tags whose border is partly covered
  bool polish = false;             // fit the whole tag's appearance for the most precise corners (slower)

  // Pose, from the detections' homographies, when `tag_size` is positive (in the unit you want the pose in).
  CameraIntrinsics intrinsics;
  double tag_size = 0.0;

  uint32_t cpu_threads = 0;        // threads for the CPU stages including the caller's; 0 uses the whole CPU
  uint32_t stream = 0;             // which queue of a shared device this detector uses
};

// Per-frame times in milliseconds. The GPU figure is a wall-clock time from upload to readback.
struct Timing {
  double gpu_ms = 0.0;
  double quad_ms = 0.0;      // quad fit on the CPU (kSync and multi-level)
  double decode_ms = 0.0;    // refine and decode on the CPU (kSync and multi-level)
  double pose_ms = 0.0;
  double total_ms = 0.0;     // the whole Detect() call
  uint32_t candidates = 0;   // quads handed to the decoder
  uint32_t submits = 0;      // GPU queue submissions
};

struct Result {
  bool ready = false;                // false for the first call in kPipelined mode
  std::vector<Detection> detections;
  std::vector<TagPose> poses;        // one per detection when pose is on
  Timing timing;
};

class Detector {
 public:
  // Opens its own device.
  explicit Detector(const DetectorOptions &options);
  // Runs on a device that other detectors share.
  Detector(const DetectorOptions &options, std::shared_ptr<Device> device);
  ~Detector();
  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;

  // Detects the tags of one frame of tightly packed 8-bit greyscale (width * height bytes), or of rows `stride`
  // bytes apart when `stride` is not 0 (they are copied). The result stays valid until the next call.
  const Result &Detect(const uint8_t *grey, size_t stride = 0);

  // kPipelined: the last frame's detections once the GPU has finished it; other modes: the last result.
  const Result &Flush();

  // Reads this frame buffer in place instead of copying it: `bytes` at least width * height, and `ptr` and `bytes`
  // aligned as the device requires. Later calls to Detect() with the same pointer use the import.
  void ImportHostFrame(const uint8_t *ptr, size_t bytes);
  // The same for a dma-buf file descriptor; `key` is the pointer later passed to Detect(). `yuyv` marks packed YUYV
  // frames, of which only the luma is read.
  void ImportDmaBufFrame(const uint8_t *key, int dmabuf_fd, size_t bytes, bool yuyv = false);

  // Forgets the tracks of kTracking mode (for example after a scene cut).
  void Reset();

  // The frame size this detector was made for.
  uint32_t width() const;
  uint32_t height() const;

  // A one-line description of the device and the buffers.
  std::string Describe() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace apriltag_vulkan
