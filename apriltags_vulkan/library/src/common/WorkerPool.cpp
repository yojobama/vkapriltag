#include "vkapriltag/common/WorkerPool.h"

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

using Clock = std::chrono::steady_clock;

int64_t NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

inline void CpuRelax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("isb" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

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

std::mutex &RegistryMutex() {
  static std::mutex m;
  return m;
}

std::vector<WorkerPool *> &Registry() {
  static std::vector<WorkerPool *> r;
  return r;
}

}  // namespace

WorkerPoolOptions ResolveWorkerPoolOptions(const WorkerPoolOptions &configured) {
  WorkerPoolOptions o = configured;
  if (const char *v = std::getenv("APRILTAG_CPU_SPIN_US")) o.spin_us = std::atoi(v);
  if (const char *v = std::getenv("APRILTAG_CPU_INLINE")) o.inline_max = std::atoi(v);
  if (const char *v = std::getenv("APRILTAG_CPU_INLINE_IDLE_US")) o.inline_idle_us = std::atoi(v);
  if (const char *v = std::getenv("APRILTAG_CPU_BIG_CORES")) o.big_cores = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_CPU_PIN_CALLER")) o.pin_caller = v[0] != '0';
  return o;
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
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry().push_back(this);
}

WorkerPool::~WorkerPool() {
  {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto &r = Registry();
    r.erase(std::remove(r.begin(), r.end(), this), r.end());
  }
  StopWorkers();
}

void WorkerPool::Warm() {
  if (options_.spin_us <= 0 || workers_.empty()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    warm_epoch_.fetch_add(1, std::memory_order_release);
  }
  batch_ready_.notify_all();
}

void WorkerPool::WarmAll() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  for (WorkerPool *pool : Registry()) pool->Warm();
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

void WorkerPool::SpinForWork(uint64_t seen) {
  const int64_t deadline = NowNs() + static_cast<int64_t>(options_.spin_us) * 1000;
  for (unsigned n = 0;; ++n) {
    if (generation_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire)) return;
    CpuRelax();
    if ((n & 31) == 31 && NowNs() >= deadline) return;
  }
}

void WorkerPool::WorkerMain(unsigned slot) {
  prof::SetThreadName((name_ + " " + std::to_string(slot)).c_str());
  PinThread(big_cpus_);
  uint64_t seen = 0;
  uint64_t seen_warm = 0;
  bool spin = false;
  for (;;) {
    if (spin && options_.spin_us > 0) SpinForWork(seen);
    bool batch = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      batch_ready_.wait(lock, [this, &seen, &seen_warm] {
        return stop_.load(std::memory_order_relaxed) || generation_.load(std::memory_order_relaxed) != seen ||
               warm_epoch_.load(std::memory_order_relaxed) != seen_warm;
      });
      if (stop_.load(std::memory_order_relaxed)) return;
      if (generation_.load(std::memory_order_relaxed) != seen) {
        seen = generation_.load(std::memory_order_relaxed);
        batch = true;
      } else {
        seen_warm = warm_epoch_.load(std::memory_order_relaxed);
      }
    }
    spin = true;
    if (!batch) continue;

    DrainBatch(slot);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--outstanding_ == 0) batch_done_.notify_one();
    }
  }
}

void WorkerPool::PinCaller() {
  const size_t id = std::hash<std::thread::id>()(std::this_thread::get_id()) | 1;
  if (pinned_caller_.load(std::memory_order_relaxed) == id) return;
  PinThread(big_cpus_);
  pinned_caller_.store(id, std::memory_order_relaxed);
}

void WorkerPool::ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn) {
  if (count == 0) return;
  std::lock_guard<std::mutex> call_lock(call_mutex_);

  // A single item (or a single-threaded pool) runs inline on the calling thread as slot 0.
  if (options_.pin_caller && !big_cpus_.empty()) PinCaller();
  if (workers_.empty() || count == 1) {
    for (size_t i = 0; i < count; ++i) fn(i, 0);
    return;
  }

  // A small batch on a pool that has been idle runs here instead of waking the workers.
  if (options_.inline_max > 0 && count <= static_cast<size_t>(options_.inline_max)) {
    const int64_t now = NowNs();
    const bool cold = now - last_end_ns_.load(std::memory_order_relaxed) > static_cast<int64_t>(options_.inline_idle_us) * 1000;
    if (cold) {
      for (size_t i = 0; i < count; ++i) fn(i, 0);
      last_end_ns_.store(NowNs(), std::memory_order_relaxed);
      return;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    fn_ = &fn;
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    outstanding_ = workers_.size();
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
    batch_done_.wait(lock, [this] { return outstanding_ == 0; });
    // Every participant has returned from DrainBatch().
    fn_ = nullptr;
    count_ = 0;
    error = std::exchange(error_, nullptr);
  }
  if (options_.inline_max > 0) last_end_ns_.store(NowNs(), std::memory_order_relaxed);
  if (error) std::rethrow_exception(error);
}

}  // namespace apriltag_vulkan
