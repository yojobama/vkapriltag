#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

namespace apriltag_vulkan {

// Optional behaviours of WorkerPool; all off by default. Negative values mean "unset".
struct WorkerPoolOptions {
  // Microseconds a worker spins for the next batch after a batch or a Warm() before it blocks.
  int spin_us = 0;
  // Spinning workers yield the CPU on each iteration; only workers with slot <= spin_workers spin (0 = all).
  bool spin_yield = false;
  int spin_workers = 0;
  // Batches of at most this many items run on the calling thread when the pool has been idle for
  // longer than inline_idle_us.
  int inline_max = 0;
  int inline_idle_us = 1000;
  // On CPUs whose cores differ in capacity, pins workers to the highest-capacity cores and sizes
  // the default thread count to their number; pin_caller also pins the calling thread.
  bool big_cores = false;
  bool pin_caller = false;
};

// Returns `configured` with the APRILTAG_CPU_SPIN_US, APRILTAG_CPU_INLINE, APRILTAG_CPU_INLINE_IDLE_US,
// APRILTAG_CPU_BIG_CORES and APRILTAG_CPU_PIN_CALLER, APRILTAG_CPU_SPIN_YIELD and APRILTAG_CPU_SPIN_WORKERS environment variables applied.
WorkerPoolOptions ResolveWorkerPoolOptions(const WorkerPoolOptions &configured);

// Persistent worker pool for the CPU tail; work is handed out by an atomic index.
class WorkerPool {
 public:
  // `threads` is the total parallelism including the calling thread (1 runs inline; 0 selects
  // hardware_concurrency), capped at 4 x hardware_concurrency (at least 16).
  // `name` labels the worker threads in profiler traces ("<name> <slot>"); it is copied.
  explicit WorkerPool(unsigned threads = 0, const char *name = "cpu-worker",
                      const WorkerPoolOptions &options = WorkerPoolOptions());
  ~WorkerPool();

  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;

  // Invokes fn(i, slot) once for each i in [0, count) and returns when all have completed; fn must
  // be safe to call concurrently for distinct i. `slot` in [0, threads()) is fixed per thread
  // (0 is the caller), so per-thread scratch can be indexed by it instead of using thread_local.
  // Concurrent calls run one after another. If fn throws, the remaining items are skipped and the
  // first exception is rethrown here once every thread has left the batch.
  void ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn);

  // Makes the workers of this pool spin for options.spin_us so that a batch due soon starts without
  // a wake-up; no effect when spinning is off.
  void Warm();
  // Warm() on every pool that exists.
  static void WarmAll();

  unsigned threads() const { return 1 + static_cast<unsigned>(workers_.size()); }

 private:
  void WorkerMain(unsigned slot);
  void SpinForWork(uint64_t seen);
  bool ShouldSpin(unsigned slot) const;
  void PinCaller();
  void StopWorkers();
  // Claims indices until the batch is exhausted; used by workers and the calling thread.
  void DrainBatch(unsigned slot);

  std::string name_;
  std::vector<std::thread> workers_;

  std::mutex call_mutex_;  // held for a whole ParallelFor
  std::mutex mutex_;
  std::condition_variable batch_ready_;
  std::condition_variable batch_done_;

  const std::function<void(size_t, unsigned)> *fn_ = nullptr;
  size_t count_ = 0;
  std::atomic<size_t> next_{0};
  size_t outstanding_ = 0;  // workers still inside the current batch
  // Read by spinning workers; kept on a cache line of their own, away from the written members.
  alignas(64) std::atomic<uint64_t> generation_{0};
  std::atomic<uint64_t> warm_epoch_{0};
  std::atomic<bool> stop_{false};
  alignas(64) char pad_[1] = {};
  WorkerPoolOptions options_;
  std::atomic<int64_t> last_end_ns_{0};
  std::vector<int> big_cpus_;  // empty unless pinning applies
  std::atomic<size_t> pinned_caller_{0};
  std::exception_ptr error_;  // the batch's first exception
};

// Thread count for the CPU tail: explicit config value, else APRILTAG_CPU_THREADS, else 0
// (WorkerPool default).
inline unsigned ResolveThreadCount(uint32_t configured) {
  if (configured > 0) return configured;
  if (const char *t = std::getenv("APRILTAG_CPU_THREADS")) {
    const long parsed = std::strtol(t, nullptr, 10);
    if (parsed > 0) return static_cast<unsigned>(parsed < 65536 ? parsed : 65536);  // WorkerPool clamps further
  }
  return 0;
}

}  // namespace apriltag_vulkan
