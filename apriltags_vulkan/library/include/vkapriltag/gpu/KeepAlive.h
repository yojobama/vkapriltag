#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan {

// Keeps a GPU that drops to a low-power state after a few milliseconds idle awake for the next
// frame, with one-thread dispatches on a lane of its own. Two modes, usable together:
//  - periodic: one dispatch every `interval` while frames are arriving (it stops after
//    kActiveWindow without a Touch());
//  - predictive: learns the frame period from Touch() and NoteFrame() calls and submits a short
//    burst `lead` before the next frame is due, only while the cadence is regular.
// Prewarm() submits a burst at once. Nothing runs while idle.
class KeepAlive {
 public:
  struct Options {
    std::chrono::microseconds interval{0};
    std::chrono::microseconds lead{0};
    unsigned burst = 3;
    uint32_t groups = 1;
    std::chrono::microseconds burst_gap{300};
  };

  // `lane` is a lane of its own on the detector's queue; `pipeline` binds one scratch buffer.
  KeepAlive(std::unique_ptr<vk::Lane> lane, const vk::ComputePipeline &pipeline,
            const Options &options);
  ~KeepAlive();

  KeepAlive(const KeepAlive &) = delete;
  KeepAlive &operator=(const KeepAlive &) = delete;

  // A frame that uses the GPU is starting; `expect_next` arms the predictive burst for the next one.
  void Touch(bool expect_next = true);
  // A frame arrived that does not use the GPU (cadence only).
  void NoteFrame();
  // The frame after the last one noted will use the GPU: arms the predictive burst.
  void ExpectGpuFrame();
  // Submits a burst now.
  void Prewarm();

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds kActiveWindow{100};
  static constexpr size_t kIntervals = 6;

  void Run();
  void RecordArrivalLocked(Clock::time_point now);
  // Expected time of the next frame, or a zero time point when the cadence is not regular.
  Clock::time_point NextFrameLocked() const;
  // Waits until `deadline` (or forever if it is the maximum) or a signal; the lock is held on return.
  void Wait(std::unique_lock<std::mutex> &lock, Clock::time_point deadline);
  void Signal();

  std::unique_ptr<vk::Lane> lane_;
  const vk::ComputePipeline &pipeline_;
  Options options_;
  std::atomic<int64_t> last_touch_ns_{0};
  std::atomic<bool> active_{false};

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool kick_ = false;
  bool armed_ = false;
  Clock::time_point last_arrival_{};
  std::array<int64_t, kIntervals> intervals_ns_{};
  size_t interval_count_ = 0;
  size_t interval_head_ = 0;

#ifdef _WIN32
  void *timer_ = nullptr;
  void *event_ = nullptr;
#endif
  std::thread thread_;
};

}  // namespace apriltag_vulkan
