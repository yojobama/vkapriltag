#include "vkapriltag/common/WorkerPool.h"

#include "vkapriltag/common/Options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "vkapriltag/common/Profiler.h"

namespace apriltag_vulkan {

namespace {

// Indices of the cores with the highest cpu_capacity when capacities differ; empty otherwise.
std::vector<int> BigCpus() {
  std::vector<int> big;
#if defined(__linux__)
  std::vector<long> capacity;
  for (int cpu = 0; cpu < 1024; ++cpu) {
    char path[96];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
    std::FILE *f = std::fopen(path, "r");
    if (f == nullptr) break;
    long value = 0;
    const bool ok = std::fscanf(f, "%ld", &value) == 1;
    std::fclose(f);
    if (!ok) return {};
    capacity.push_back(value);
  }
  if (capacity.empty()) return {};
  const long top = *std::max_element(capacity.begin(), capacity.end());
  if (top == *std::min_element(capacity.begin(), capacity.end())) return {};
  for (size_t i = 0; i < capacity.size(); ++i) {
    if (capacity[i] == top) big.push_back(static_cast<int>(i));
  }
#endif
  return big;
}

// Restricts the calling thread to `cpus` (Linux only).
void PinThread(const std::vector<int> &cpus) {
#if defined(__linux__)
  if (cpus.empty()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : cpus) CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpus;
#endif
}

}  // namespace

WorkerPoolOptions ResolveWorkerPoolOptions(const WorkerPoolOptions &configured) {
  WorkerPoolOptions o = configured;
  o.big_cores = Options::FromEnvironment().Bool("APRILTAG_CPU_BIG_CORES", o.big_cores);
  return o;
}

unsigned ResolveThreadCount(uint32_t configured) {
  if (configured > 0) return configured;
  const std::optional<long> parsed = Options::FromEnvironment().Long("APRILTAG_CPU_THREADS");
  if (parsed && *parsed > 0) return static_cast<unsigned>(*parsed < 65536 ? *parsed : 65536);  // WorkerPool clamps further
  return 0;
}

WorkerPool::WorkerPool(unsigned threads, const char *name, const WorkerPoolOptions &options)
    : name_(name), options_(options) {
  const unsigned hardware = std::thread::hardware_concurrency();
  if (options_.big_cores) big_cpus_ = BigCpus();
  unsigned total = threads;
  if (total == 0) total = big_cpus_.empty() ? hardware : static_cast<unsigned>(big_cpus_.size());
  total = std::clamp(total, 1u, std::max(4u * hardware, 16u));

  // Slot 0 is the calling thread; workers get 1, 2, ... (see WorkerPool.h).
  try {
    workers_.reserve(total - 1);
    for (unsigned i = 0; i + 1 < total; ++i) {
      const unsigned slot = i + 1;
      workers_.emplace_back([this, slot] { WorkerMain(slot); });
    }
  } catch (...) {
    StopWorkers();
    throw;
  }
}

WorkerPool::~WorkerPool() {
  StopWorkers();
}

void WorkerPool::StopWorkers() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);  // wake every worker so it can observe stop_
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

  // arg = items this thread claimed; a thread that claims none shows how late it woke.
  prof::Scope drain("pool.drain");
  int64_t claimed = 0;
  for (;;) {
    const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
    if (i >= count) break;
    try {
      (*fn)(i, slot);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
      next_.store(count, std::memory_order_relaxed);  // skips the rest of the batch
    }
    ++claimed;
  }
  drain.set_arg(claimed);
}

void WorkerPool::WorkerMain(unsigned slot) {
  prof::SetThreadName((name_ + " " + std::to_string(slot)).c_str());
  PinThread(big_cpus_);
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      batch_ready_.wait(lock, [this, &seen] {
        return stop_.load(std::memory_order_relaxed) || generation_.load(std::memory_order_relaxed) != seen;
      });
      if (stop_.load(std::memory_order_relaxed)) return;
      seen = generation_.load(std::memory_order_relaxed);
    }

    DrainBatch(slot);

    if (outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(mutex_);  // orders the notification after the caller's wait
      batch_done_.notify_one();
    }
  }
}

void WorkerPool::ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn) {
  if (count == 0) return;
  std::lock_guard<std::mutex> call_lock(call_mutex_);

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
    outstanding_.store(workers_.size(), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }
  batch_ready_.notify_all();

  // The calling thread also drains the batch, as slot 0.
  DrainBatch(0);

  std::exception_ptr error;
  {
    // The tail of the batch: time the caller idles while a worker finishes its last item.
    VKA_PROF_SCOPE("pool.join_wait");
    std::unique_lock<std::mutex> lock(mutex_);
    batch_done_.wait(lock, [this] { return outstanding_.load(std::memory_order_acquire) == 0; });
    // Every participant has returned from DrainBatch().
    fn_ = nullptr;
    count_ = 0;
    error = std::exchange(error_, nullptr);
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace apriltag_vulkan
