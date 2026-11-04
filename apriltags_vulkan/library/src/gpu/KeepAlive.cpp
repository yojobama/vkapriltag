#include "vkapriltag/gpu/KeepAlive.h"

namespace apriltag_vulkan {

KeepAlive::KeepAlive(std::unique_ptr<vk::Lane> lane, const vk::ComputePipeline &pipeline,
                     std::chrono::milliseconds interval)
    : lane_(std::move(lane)), pipeline_(pipeline), interval_(interval) {
  thread_ = std::thread([this] { Run(); });
}

KeepAlive::~KeepAlive() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_one();
  thread_.join();
}

void KeepAlive::Touch() {
  last_touch_ns_.store(Clock::now().time_since_epoch().count(), std::memory_order_relaxed);
  if (!active_.load(std::memory_order_acquire)) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_.store(true, std::memory_order_release);
    }
    cv_.notify_one();
  }
}

void KeepAlive::Run() {
  VkCommandBuffer cmd = lane_->BeginPersistentCommands();
  pipeline_.DispatchRaw(cmd, 1, 1, 1, nullptr, vk::BarrierKind::None);
  lane_->EndCommands(cmd);

  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    cv_.wait(lock, [this] { return stop_ || active_.load(std::memory_order_acquire); });
    while (!stop_ && active_.load(std::memory_order_acquire)) {
      cv_.wait_for(lock, interval_, [this] { return stop_; });
      if (stop_) break;
      const Clock::duration since(Clock::now().time_since_epoch().count() -
                                  last_touch_ns_.load(std::memory_order_relaxed));
      if (since > kActiveWindow) {
        active_.store(false, std::memory_order_release);
        break;
      }
      lock.unlock();
      try {
        lane_->SubmitPersistentAndWait(cmd);
      } catch (...) {
        // The device is failing; the next frame reports it.
      }
      lock.lock();
    }
  }
  lock.unlock();
  try {
    lane_->WaitIdle();
  } catch (...) {
  }
  lane_->FreePersistentCommands(cmd);
}

}  // namespace apriltag_vulkan
