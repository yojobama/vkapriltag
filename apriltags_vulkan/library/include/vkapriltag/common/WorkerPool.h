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

// A minimal persistent worker pool for the detector's CPU tail.
//
// Persistent rather than thread-per-frame on purpose: spawning a dozen
// std::threads costs a few hundred microseconds, which is negligible against
// the ~10 ms the tail used to take but not against the ~1.5 ms it takes once
// parallelized.
//
// Work is handed out by an atomic index rather than statically partitioned,
// because per-blob cost varies by more than an order of magnitude (the
// combinatorial quad search runs only for blobs with >= 4 detected peaks), so
// an even split would leave most threads idle waiting for one straggler.
class WorkerPool {
 public:
  // `threads` is the TOTAL degree of parallelism including the calling thread,
  // so a value of 1 runs everything inline with no synchronization at all.
  // 0 selects std::thread::hardware_concurrency().
  explicit WorkerPool(unsigned threads = 0);
  ~WorkerPool();

  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;

  // Invokes fn(i) exactly once for every i in [0, count), on an unspecified
  // thread, and returns only once all of them have completed. fn must be safe
  // to call concurrently for distinct i.
  void ParallelFor(size_t count, const std::function<void(size_t)> &fn);

  unsigned threads() const { return 1 + static_cast<unsigned>(workers_.size()); }

 private:
  void WorkerMain();
  // Claims indices until the current batch is exhausted. Shared by the
  // workers and the calling thread, so the caller also does a share of the
  // work instead of blocking idle.
  void DrainBatch();

  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable batch_ready_;
  std::condition_variable batch_done_;

  const std::function<void(size_t)> *fn_ = nullptr;
  size_t count_ = 0;
  std::atomic<size_t> next_{0};
  size_t outstanding_ = 0;  // workers still inside the current batch
  uint64_t generation_ = 0;
  bool stop_ = false;
};

// Resolves a WorkerPool's degree of parallelism the same way everywhere it's
// constructed in the CPU tail (QuadDecode, TagDecoder): an explicit config
// value wins, then APRILTAG_CPU_THREADS, then WorkerPool's own
// hardware_concurrency default. Kept in one place so the env var means the
// same thing - and can be set once to affect both phases - regardless of
// which phase is constructing the pool.
inline unsigned ResolveThreadCount(uint32_t configured) {
  if (configured > 0) return configured;
  if (const char *t = std::getenv("APRILTAG_CPU_THREADS")) {
    const long parsed = std::strtol(t, nullptr, 10);
    if (parsed > 0) return static_cast<unsigned>(parsed);
  }
  return 0;
}

}  // namespace apriltag_vulkan
