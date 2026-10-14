#include "vkapriltag/common/Profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace apriltag_vulkan::prof {

namespace detail {
std::atomic<bool> g_enabled{false};
}  // namespace detail

namespace {

// Per-thread cap; at 32 bytes an event this is 8 MiB per thread, ample for thousands of frames of
// the instrumented scopes.
constexpr size_t kMaxEventsPerThread = size_t{1} << 18;

struct Event {
  const char *name;
  Ns t0;
  Ns t1;
  int64_t arg;
};

// Written by one thread (or, for virtual tracks, under Registry::mutex), read only when the
// instrumented work is quiescent.
struct Buffer {
  uint32_t tid = 0;
  std::string name;
  std::vector<Event> events;
  uint64_t dropped = 0;

  void Push(const Event &e) {
    if (events.size() >= kMaxEventsPerThread) {
      ++dropped;
      return;
    }
    if (events.capacity() == 0) events.reserve(1 << 12);
    events.push_back(e);
  }
};

// Intentionally leaked: worker threads may record while static destructors run.
struct Registry {
  std::mutex mutex;
  std::vector<std::unique_ptr<Buffer>> buffers;
  std::map<std::string, Buffer *> tracks;
  uint32_t next_tid = 1;
  std::string env_trace_path;
};

Registry &Reg() {
  static Registry *r = new Registry;
  return *r;
}

Buffer *NewBuffer(const std::string &name) {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);
  auto b = std::make_unique<Buffer>();
  b->tid = r.next_tid++;
  b->name = name.empty() ? "thread " + std::to_string(b->tid) : name;
  Buffer *raw = b.get();
  r.buffers.push_back(std::move(b));
  return raw;
}

Buffer *ThreadBuffer() {
  thread_local Buffer *buf = nullptr;
  if (buf == nullptr) buf = NewBuffer("");
  return buf;
}

Ns Epoch() {
  static const Ns epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
  return epoch;
}

std::string JsonEscape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += ' ';
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace

Ns NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

namespace detail {
void Record(const char *name, Ns t0, Ns t1, int64_t arg) {
  ThreadBuffer()->Push({name, t0, t1, arg});
}
}  // namespace detail

void Enable(bool on) {
  Epoch();  // pin the trace origin before the first event
  detail::g_enabled.store(on && VKAPRILTAG_PROFILING, std::memory_order_relaxed);
}

bool EnableFromEnv() {
  const char *path = std::getenv("APRILTAG_VK_TRACE");
  if (path == nullptr || *path == '\0') return false;
  Reg().env_trace_path = path;
  Enable(true);
  return Enabled();
}

const std::string &EnvTracePath() { return Reg().env_trace_path; }

void SetThreadName(const char *name) {
  Buffer *b = ThreadBuffer();
  std::lock_guard<std::mutex> lock(Reg().mutex);
  b->name = name;
}

uint32_t Track(const char *name) {
  Registry &r = Reg();
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.tracks.find(name);
    if (it != r.tracks.end()) return it->second->tid;
  }
  Buffer *b = NewBuffer(name);
  std::lock_guard<std::mutex> lock(r.mutex);
  auto [it, inserted] = r.tracks.emplace(name, b);
  return it->second->tid;
}

void Complete(uint32_t track, const char *name, Ns t0, Ns t1, int64_t arg) {
  if (!Enabled()) return;
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);
  for (auto &b : r.buffers) {
    if (b->tid == track) {
      b->Push({name, t0, t1, arg});
      return;
    }
  }
}

void Reset() {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);
  for (auto &b : r.buffers) {
    b->events.clear();
    b->dropped = 0;
  }
}

uint64_t EventCount() {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);
  uint64_t n = 0;
  for (auto &b : r.buffers) n += b->events.size();
  return n;
}

uint64_t DroppedEvents() {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);
  uint64_t n = 0;
  for (auto &b : r.buffers) n += b->dropped;
  return n;
}

std::string Summary() {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);

  // Spans with the same name on different tracks are reported separately ("name [track]") only
  // for virtual tracks; host threads are merged, since worker threads run the same scopes.
  std::map<std::string, std::vector<double>> durations_us;
  for (auto &b : r.buffers) {
    const bool virtual_track = r.tracks.count(b->name) != 0;
    for (const Event &e : b->events) {
      std::string key = e.name;
      if (virtual_track) key += " [" + b->name + "]";
      durations_us[key].push_back(double(e.t1 - e.t0) / 1.0e3);
    }
  }

  struct Row {
    std::string name;
    size_t n;
    double total, mean, p50, p99, max;
  };
  std::vector<Row> rows;
  for (auto &[name, d] : durations_us) {
    std::sort(d.begin(), d.end());
    double total = 0;
    for (double v : d) total += v;
    auto pct = [&](double p) {
      return d[std::min(d.size() - 1, static_cast<size_t>(p * double(d.size())))];
    };
    rows.push_back({name, d.size(), total, total / double(d.size()), pct(0.5), pct(0.99), d.back()});
  }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.total > b.total; });

  std::ostringstream os;
  char line[256];
  std::snprintf(line, sizeof line, "%-28s %8s %12s %10s %10s %10s %10s\n", "span", "count",
                "total ms", "mean us", "p50 us", "p99 us", "max us");
  os << line;
  for (const Row &row : rows) {
    std::snprintf(line, sizeof line, "%-28s %8zu %12.3f %10.2f %10.2f %10.2f %10.2f\n",
                  row.name.c_str(), row.n, row.total / 1.0e3, row.mean, row.p50, row.p99, row.max);
    os << line;
  }
  uint64_t dropped = 0;
  for (auto &b : r.buffers) dropped += b->dropped;
  if (dropped > 0) os << "(" << dropped << " events dropped: per-thread buffer full)\n";
  return os.str();
}

bool WriteTrace(const std::string &path) {
  Registry &r = Reg();
  std::lock_guard<std::mutex> lock(r.mutex);

  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;

  const Ns origin = Epoch();
  std::fputs("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n", f);
  bool first = true;
  auto sep = [&] {
    if (!first) std::fputs(",\n", f);
    first = false;
  };

  for (auto &b : r.buffers) {
    if (b->events.empty()) continue;
    sep();
    std::fprintf(f,
                 "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":%u,"
                 "\"args\":{\"name\":\"%s\"}}",
                 b->tid, JsonEscape(b->name).c_str());
    // Virtual tracks sort after host threads.
    sep();
    std::fprintf(f,
                 "{\"ph\":\"M\",\"name\":\"thread_sort_index\",\"pid\":1,\"tid\":%u,"
                 "\"args\":{\"sort_index\":%u}}",
                 b->tid, r.tracks.count(b->name) ? 1000 + b->tid : b->tid);
    for (const Event &e : b->events) {
      sep();
      std::fprintf(f,
                   "{\"ph\":\"X\",\"name\":\"%s\",\"pid\":1,\"tid\":%u,\"ts\":%.3f,\"dur\":%.3f,"
                   "\"args\":{\"v\":%lld}}",
                   JsonEscape(e.name).c_str(), b->tid, double(e.t0 - origin) / 1.0e3,
                   double(e.t1 - e.t0) / 1.0e3, static_cast<long long>(e.arg));
    }
  }
  std::fputs("\n]}\n", f);
  return std::fclose(f) == 0;
}

}  // namespace apriltag_vulkan::prof
