#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace apriltag_vulkan {

// Persistent worker pool for the CPU tail; work is handed out by an atomic index.
class WorkerPool {
 public:
  // `threads` is the total parallelism including the calling thread (1 runs inline; 0 selects
  // hardware_concurrency).
  explicit WorkerPool(unsigned threads = 0);
  ~WorkerPool();

  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;

  // Invokes fn(i, slot) once for each i in [0, count) and returns when all have completed; fn must
  // be safe to call concurrently for distinct i. `slot` in [0, threads()) is fixed per thread
  // (0 is the caller), so per-thread scratch can be indexed by it instead of using thread_local.
  void ParallelFor(size_t count, const std::function<void(size_t, unsigned)> &fn);

  unsigned threads() const { return 1 + static_cast<unsigned>(workers_.size()); }

 private:
  void WorkerMain(unsigned slot);
  // Claims indices until the batch is exhausted; used by workers and the calling thread.
  void DrainBatch(unsigned slot);

  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable batch_ready_;
  std::condition_variable batch_done_;

  const std::function<void(size_t, unsigned)> *fn_ = nullptr;
  size_t count_ = 0;
  std::atomic<size_t> next_{0};
  size_t outstanding_ = 0;  // workers still inside the current batch
  uint64_t generation_ = 0;
  bool stop_ = false;
};

// Thread count for the CPU tail: explicit config value, else APRILTAG_CPU_THREADS, else 0
// (WorkerPool default).
inline unsigned ResolveThreadCount(uint32_t configured) {
  if (configured > 0) return configured;
  if (const char *t = std::getenv("APRILTAG_CPU_THREADS")) {
    const long parsed = std::strtol(t, nullptr, 10);
    if (parsed > 0) return static_cast<unsigned>(parsed);
  }
  return 0;
}

}  // namespace apriltag_vulkan
