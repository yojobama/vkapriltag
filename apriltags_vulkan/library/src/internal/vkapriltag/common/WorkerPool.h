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

// Optional behaviours of WorkerPool (no effect where core capacities are equal).
struct WorkerPoolOptions {
  // On CPUs whose cores differ in capacity, pins workers to the highest-capacity cores and sizes
  // the default thread count to their number.
  bool big_cores = true;
};

// Returns `configured` with the APRILTAG_CPU_BIG_CORES environment variable applied.
WorkerPoolOptions ResolveWorkerPoolOptions(const WorkerPoolOptions &configured);

// Persistent worker pool for the CPU tail; work is handed out by an atomic index. The stages of one
// pipeline may share a pool (std::shared_ptr) when they run one after another; stages that run
// concurrently on different threads need pools of their own.
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
  // Calls from different threads run one after another (see contended_calls()); a call from inside
  // one of this pool's own tasks throws std::logic_error. If fn throws, the remaining items are
  // skipped and the first exception is rethrown here once every thread has left the batch.
  void ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn);

  unsigned threads() const { return 1 + static_cast<unsigned>(workers_.size()); }

  // Calls that had to wait for another thread inside ParallelFor; non-zero means a pool meant for
  // one thread at a time is shared by two.
  uint64_t contended_calls() const { return contended_.load(std::memory_order_relaxed); }

 private:
  void WorkerMain(unsigned slot);
  void StopWorkers();
  // Claims indices until the batch is exhausted; used by workers and the calling thread.
  void DrainBatch(unsigned slot);

  std::string name_;
  std::vector<std::thread> workers_;

  std::mutex call_mutex_;  // held for a whole ParallelFor
  std::atomic<uint64_t> contended_{0};
  std::mutex mutex_;
  std::condition_variable batch_ready_;
  std::condition_variable batch_done_;

  const std::function<void(size_t, unsigned)> *fn_ = nullptr;
  size_t count_ = 0;
  std::atomic<size_t> next_{0};
  std::atomic<size_t> outstanding_{0};  // workers still inside the current batch
  std::atomic<int> tickets_{0};         // workers that may still join the current batch
  std::atomic<uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
  WorkerPoolOptions options_;
  std::vector<int> big_cpus_;  // empty unless pinning applies
  std::exception_ptr error_;  // the batch's first exception
};

// Thread count for the CPU tail: a non-zero configured value wins (the one setting where the
// configuration beats the environment), else APRILTAG_CPU_THREADS, else 0 (WorkerPool default).
unsigned ResolveThreadCount(uint32_t configured);

}  // namespace apriltag_vulkan
