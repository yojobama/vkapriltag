#include "vkapriltag/common/WorkerPool.h"

namespace apriltag_vulkan {

WorkerPool::WorkerPool(unsigned threads) {
  unsigned total = threads;
  if (total == 0) total = std::thread::hardware_concurrency();
  if (total == 0) total = 1;

  workers_.reserve(total - 1);
  // Slot 0 is the calling thread; workers get 1, 2, ... (see WorkerPool.h).
  for (unsigned i = 0; i + 1 < total; ++i) {
    const unsigned slot = i + 1;
    workers_.emplace_back([this, slot] { WorkerMain(slot); });
  }
}

WorkerPool::~WorkerPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    ++generation_;  // wake every worker so it can observe stop_
  }
  batch_ready_.notify_all();
  for (auto &t : workers_) {
    if (t.joinable()) t.join();
  }
}

void WorkerPool::DrainBatch(unsigned slot) {
  // fn_ and count_ stay stable until every participant has left this function.
  const std::function<void(size_t, unsigned)> *fn = fn_;
  if (fn == nullptr) return;
  const size_t count = count_;

  for (;;) {
    const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
    if (i >= count) break;
    (*fn)(i, slot);
  }
}

void WorkerPool::WorkerMain(unsigned slot) {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      batch_ready_.wait(lock, [this, &seen] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }

    DrainBatch(slot);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--outstanding_ == 0) batch_done_.notify_one();
    }
  }
}

void WorkerPool::ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn) {
  if (count == 0) return;

  // A single item (or a single-threaded pool) runs inline on the calling thread as slot 0.
  if (workers_.empty() || count == 1) {
    for (size_t i = 0; i < count; ++i) fn(i, 0);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    fn_ = &fn;
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    outstanding_ = workers_.size();
    ++generation_;
  }
  batch_ready_.notify_all();

  // The calling thread also drains the batch, as slot 0.
  DrainBatch(0);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    batch_done_.wait(lock, [this] { return outstanding_ == 0; });
    // Every participant has returned from DrainBatch().
    fn_ = nullptr;
    count_ = 0;
  }
}

}  // namespace apriltag_vulkan
