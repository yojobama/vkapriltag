#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan {

// Submits a one-thread dispatch every `interval` while frames are arriving, so a GPU that drops to a
// low-power state after a few milliseconds idle stays awake between camera frames. It stops after
// kActiveWindow without a Touch() and costs nothing while idle.
class KeepAlive {
 public:
  // `lane` is a lane of its own on the detector's queue; `pipeline` binds one scratch buffer.
  KeepAlive(std::unique_ptr<vk::Lane> lane, const vk::ComputePipeline &pipeline,
            std::chrono::milliseconds interval);
  ~KeepAlive();

  KeepAlive(const KeepAlive &) = delete;
  KeepAlive &operator=(const KeepAlive &) = delete;

  // Records that a frame is being submitted.
  void Touch();

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds kActiveWindow{100};

  void Run();

  std::unique_ptr<vk::Lane> lane_;
  const vk::ComputePipeline &pipeline_;
  std::chrono::milliseconds interval_;
  std::atomic<int64_t> last_touch_ns_{0};
  std::atomic<bool> active_{false};
  bool stop_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};

}  // namespace apriltag_vulkan
