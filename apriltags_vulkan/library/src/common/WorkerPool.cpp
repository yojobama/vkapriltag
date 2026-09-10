#include "vkapriltag/common/WorkerPool.h"

namespace apriltag_vulkan {

WorkerPool::WorkerPool(unsigned threads) {
  unsigned total = threads;
  if (total == 0) total = std::thread::hardware_concurrency();
  if (total == 0) total = 1;

  workers_.reserve(total - 1);
  // Slot 0 is reserved for ParallelFor's calling thread (see WorkerPool.h);
  // pool workers get 1, 2, ... - captured by value here, not read from any
  // thread_local, so each worker thread knows its own fixed slot with no
  // per-thread storage of any kind.
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
  // fn_ and count_ are guaranteed stable for the whole batch: ParallelFor does
  // not clear them until every participant has left this function.
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

  // Not worth waking anyone for a single item, and this is also the
  // single-threaded configuration's only path. Slot 0: the calling thread.
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

  // The calling thread takes a share of the work rather than blocking idle,
  // as slot 0 (see WorkerPool.h).
  DrainBatch(0);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    batch_done_.wait(lock, [this] { return outstanding_ == 0; });
    // Safe now: every participant has returned from DrainBatch().
    fn_ = nullptr;
    count_ = 0;
  }
}

}  // namespace apriltag_vulkan
