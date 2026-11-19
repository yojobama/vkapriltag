#pragma once

#include <atomic>
#include <cstdint>
#include <string>

// CPU tracing with GPU timestamp spans on the same timeline; off by default.
// Summary(), WriteTrace() and Reset() must not run while instrumented work is running.
#ifndef VKAPRILTAG_PROFILING
#define VKAPRILTAG_PROFILING 1
#endif

namespace apriltag_vulkan::prof {

using Ns = int64_t;  // nanoseconds on steady_clock's timeline

namespace detail {
extern std::atomic<bool> g_enabled;
void Record(const char *name, Ns t0, Ns t1, int64_t arg);
}  // namespace detail

#if VKAPRILTAG_PROFILING
inline bool Enabled() { return detail::g_enabled.load(std::memory_order_relaxed); }
#else
constexpr bool Enabled() { return false; }
#endif

Ns NowNs();

// Turns recording on or off. Events already recorded are kept.
void Enable(bool on);

// Enables recording when APRILTAG_VK_TRACE is set to a non-empty path and returns whether it did;
// the path is then available from EnvTracePath() for the caller to pass to WriteTrace().
bool EnableFromEnv();
const std::string &EnvTracePath();

// Names the calling thread in the trace. The string is copied.
void SetThreadName(const char *name);

// A named virtual track (for example "GPU") for spans that did not run on a host thread. Returns
// an id for Complete(); registering the same name twice returns the same id.
uint32_t Track(const char *name);

// Records a span with explicit timestamps on a virtual track. `name` must outlive the profiler
// (use a string literal or a static table). `arg` is shown as args.v in the trace.
void Complete(uint32_t track, const char *name, Ns t0, Ns t1, int64_t arg = 0);

// As Complete(), on the calling thread's track; for phases the caller has already timed.
inline void CompleteHere(const char *name, Ns t0, Ns t1, int64_t arg = 0) {
  if (Enabled()) detail::Record(name, t0, t1, arg);
}

// Discards every recorded event and the drop count; thread and track names are kept.
void Reset();

uint64_t EventCount();
// Events discarded because a thread's buffer reached its cap (kMaxEventsPerThread).
uint64_t DroppedEvents();

// One row per span name, sorted by total time: count, total, mean, p50, p99 and max.
std::string Summary();

// Chrome Trace Event JSON ("traceEvents" with complete events). Returns false if the file cannot
// be written.
bool WriteTrace(const std::string &path);

// Records [construction, destruction) under `name` on the calling thread; set_arg() attaches one
// integer.
class Scope {
 public:
  explicit Scope(const char *name) {
    if (Enabled()) {
      name_ = name;
      t0_ = NowNs();
    }
  }
  ~Scope() {
    if (name_ != nullptr) detail::Record(name_, t0_, NowNs(), arg_);
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

  void set_arg(int64_t arg) { arg_ = arg; }

 private:
  const char *name_ = nullptr;  // null while disabled
  Ns t0_ = 0;
  int64_t arg_ = 0;
};

}  // namespace apriltag_vulkan::prof

#define VKA_PROF_CAT2(a, b) a##b
#define VKA_PROF_CAT(a, b) VKA_PROF_CAT2(a, b)
#define VKA_PROF_SCOPE(name) \
  ::apriltag_vulkan::prof::Scope VKA_PROF_CAT(vka_prof_scope_, __LINE__)(name)
