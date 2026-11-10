#include "vkapriltag/gpu/KeepAlive.h"

#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace apriltag_vulkan {

namespace {
constexpr auto kForever = std::chrono::steady_clock::time_point::max();
constexpr std::chrono::nanoseconds kMinPeriod = std::chrono::milliseconds(2);
constexpr std::chrono::nanoseconds kMaxPeriod = std::chrono::milliseconds(200);
constexpr std::chrono::nanoseconds kMaxGap = std::chrono::milliseconds(500);
}  // namespace

KeepAlive::KeepAlive(std::unique_ptr<vk::Lane> lane, const vk::ComputePipeline &pipeline,
                     const Options &options)
    : lane_(std::move(lane)), pipeline_(pipeline), options_(options) {
#ifdef _WIN32
  // A high-resolution timer wakes on time whatever the system timer tick is (about 15.6 ms by default).
  timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */,
                                  TIMER_ALL_ACCESS);
  if (timer_ == nullptr) timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
  event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
#endif
  thread_ = std::thread([this] { Run(); });
}

KeepAlive::~KeepAlive() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  Signal();
  thread_.join();
#ifdef _WIN32
  if (timer_ != nullptr) CloseHandle(timer_);
  if (event_ != nullptr) CloseHandle(event_);
#endif
}

void KeepAlive::Signal() {
#ifdef _WIN32
  if (event_ != nullptr) {
    SetEvent(event_);
    return;
  }
#endif
  cv_.notify_one();
}

void KeepAlive::Wait(std::unique_lock<std::mutex> &lock, Clock::time_point deadline) {
  if (stop_) return;
#ifdef _WIN32
  if (event_ != nullptr) {
    HANDLE handles[2] = {event_, timer_};
    DWORD count = 1;
    if (deadline != kForever) {
      const auto due = deadline - Clock::now();
      if (due.count() <= 0) return;
      if (timer_ != nullptr) {
        LARGE_INTEGER relative;
        relative.QuadPart = -std::max<int64_t>(
            1, std::chrono::duration_cast<std::chrono::nanoseconds>(due).count() / 100);
        SetWaitableTimer(timer_, &relative, 0, nullptr, nullptr, FALSE);
        count = 2;
      }
    }
    lock.unlock();
    WaitForMultipleObjects(count, handles, FALSE, (count == 1 && deadline != kForever) ? 1 : INFINITE);
    lock.lock();
    return;
  }
#endif
  if (deadline == kForever) {
    cv_.wait(lock);
  } else {
    cv_.wait_until(lock, deadline);
  }
}

void KeepAlive::RecordArrivalLocked(Clock::time_point now) {
  if (last_arrival_.time_since_epoch().count() != 0) {
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_arrival_);
    if (dt > kMaxGap) {
      interval_count_ = 0;
    } else {
      intervals_ns_[interval_head_] = dt.count();
      interval_head_ = (interval_head_ + 1) % kIntervals;
      interval_count_ = std::min(interval_count_ + 1, kIntervals);
    }
  }
  last_arrival_ = now;
}

KeepAlive::Clock::time_point KeepAlive::NextFrameLocked() const {
  if (interval_count_ < 4) return Clock::time_point{};
  std::array<int64_t, kIntervals> sorted = intervals_ns_;
  std::sort(sorted.begin(), sorted.begin() + interval_count_);
  const int64_t median = sorted[interval_count_ / 2];
  if (median < kMinPeriod.count() || median > kMaxPeriod.count()) return Clock::time_point{};
  for (size_t i = 0; i < interval_count_; ++i) {
    if (std::abs(intervals_ns_[i] - median) * 4 > median) return Clock::time_point{};
  }
  return last_arrival_ + std::chrono::nanoseconds(median);
}

void KeepAlive::Touch(bool expect_next) {
  const Clock::time_point now = Clock::now();
  last_touch_ns_.store(now.time_since_epoch().count(), std::memory_order_relaxed);
  if (options_.lead.count() == 0) {
    if (!active_.load(std::memory_order_acquire)) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.store(true, std::memory_order_release);
      }
      Signal();
    }
    return;
  }
  bool signal = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    RecordArrivalLocked(now);
    if (expect_next && NextFrameLocked().time_since_epoch().count() != 0) {
      armed_ = true;
      signal = true;
    }
    if (!active_.exchange(true, std::memory_order_acq_rel)) signal = options_.interval.count() > 0 || signal;
  }
  if (signal) Signal();
}

void KeepAlive::NoteFrame(bool warm_pool) {
  if (options_.lead.count() == 0) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    RecordArrivalLocked(Clock::now());
    if (!warm_pool || !options_.on_warm) return;
    armed_pool_ = true;
  }
  Signal();
}

void KeepAlive::ExpectGpuFrame() {
  if (options_.lead.count() == 0) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
  }
  Signal();
}

void KeepAlive::Prewarm() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    kick_ = true;
  }
  Signal();
}

void KeepAlive::Run() {
  VkCommandBuffer cmd = lane_->BeginPersistentCommands();
  pipeline_.DispatchRaw(cmd, std::max(1u, options_.groups), 1, 1, nullptr, vk::BarrierKind::None);
  lane_->EndCommands(cmd);

  auto submit = [&] {
    try {
      lane_->SubmitPersistentAndWait(cmd);
    } catch (...) {
      // The device is failing; the next frame reports it.
    }
  };

  const auto interval = std::chrono::duration_cast<Clock::duration>(options_.interval);
  const auto lead = std::chrono::duration_cast<Clock::duration>(options_.lead);
  const auto burst_gap = std::chrono::duration_cast<Clock::duration>(options_.burst_gap);
  Clock::time_point last_periodic = Clock::now();

  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    const Clock::time_point now = Clock::now();
    bool burst = kick_;
    kick_ = false;
    bool periodic = false;
    Clock::time_point deadline = kForever;

    if (interval.count() > 0 && active_.load(std::memory_order_acquire)) {
      const Clock::duration since(now.time_since_epoch().count() -
                                  last_touch_ns_.load(std::memory_order_relaxed));
      if (since > kActiveWindow) {
        active_.store(false, std::memory_order_release);
      } else {
        const Clock::time_point due = last_periodic + interval;
        if (due <= now) {
          periodic = true;
        } else {
          deadline = due;
        }
      }
    }
    if (lead.count() > 0 && armed_) {
      const Clock::time_point next = NextFrameLocked();
      if (next.time_since_epoch().count() == 0 || now >= next || next - last_arrival_ <= lead + std::chrono::microseconds(500)) {
        armed_ = false;
      } else if (next - lead <= now) {
        burst = true;
        armed_ = false;
      } else {
        deadline = std::min(deadline, next - lead);
      }
    }

    bool warm_only = false;
    if (lead.count() > 0 && armed_pool_) {
      const Clock::time_point next = NextFrameLocked();
      if (next.time_since_epoch().count() == 0 || now >= next || next - last_arrival_ <= lead + std::chrono::microseconds(500)) {
        armed_pool_ = false;
      } else if (next - lead <= now) {
        warm_only = !burst;
        burst = true;
        armed_pool_ = false;
      } else {
        deadline = std::min(deadline, next - lead);
      }
    }

    if (burst && options_.on_warm) {
      lock.unlock();
      options_.on_warm();
      lock.lock();
    }
    if (warm_only) continue;
    if (burst) {
      for (unsigned i = 0; i < std::max(1u, options_.burst) && !stop_; ++i) {
        lock.unlock();
        submit();
        lock.lock();
        if (i + 1 < options_.burst) {
          const Clock::time_point until = Clock::now() + burst_gap;
          while (!stop_ && Clock::now() < until) Wait(lock, until);
        }
      }
      last_periodic = Clock::now();
      continue;
    }
    if (periodic) {
      lock.unlock();
      submit();
      lock.lock();
      last_periodic = Clock::now();
      continue;
    }
    Wait(lock, deadline);
    if (interval.count() > 0 && deadline == kForever) last_periodic = Clock::now();
  }
  lock.unlock();
  try {
    lane_->WaitIdle();
  } catch (...) {
  }
  lane_->FreePersistentCommands(cmd);
}

}  // namespace apriltag_vulkan
