// Samples selected Mali counters every PERIOD_US while a command runs; one CSV row per sample.
#include <hwcpipe/counter_database.hpp>
#include <hwcpipe/gpu.hpp>
#include <hwcpipe/sampler.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *kNames[] = {
    "GPU active cycles",
    "Programmable core active cycles",
    "Instruction issue starvation cycles",
    "Arithmetic unit issue cycles",
    "Load/store unit issue cycles",
    "Load/store unit read issues",
    "Load/store unit partial read issues",
    "Load/store unit write issues",
    "Load/store unit atomic issues",
    "Load/store unit read bytes from L2 cache",
    "Load/store unit read bytes from external memory",
    "Load/store unit write bytes to L2 memory system",
    "Output external read bytes",
    "Output external write bytes",
    "L2 cache read lookups",
    "Output external read transactions",
    "MMU level 3 lookup TLB misses",
    "Diverged instructions",
    "Arithmetic instructions",
    "Compute tasks",
    "Full warps",
};

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: hwcnt_timeline PERIOD_US OUT.csv cmd [args...]\n");
    return 1;
  }
  const long period_us = std::atol(argv[1]);
  FILE *out = std::fopen(argv[2], "w");
  auto gpu = hwcpipe::gpu(0);
  auto db = hwcpipe::counter_database{};
  auto config = hwcpipe::sampler_config(gpu);
  std::vector<hwcpipe_counter> chosen;
  std::vector<std::string> chosen_names;
  hwcpipe::counter_metadata meta;
  for (const char *want : kNames) {
    for (hwcpipe_counter c : db.counters_for_gpu(gpu)) {
      (void)db.describe_counter(c, meta);
      if (std::strcmp(meta.name, want) == 0 && !config.add_counter(c)) {
        chosen.push_back(c);
        chosen_names.push_back(want);
      }
    }
  }
  auto sampler = hwcpipe::sampler<>(config);
  if (sampler.start_sampling()) return 3;
  (void)sampler.sample_now();
  std::fprintf(out, "t_ns");
  for (auto &n : chosen_names) std::fprintf(out, ",%s", n.c_str());
  std::fprintf(out, "\n");

  const pid_t pid = fork();
  if (pid == 0) {
    execvp(argv[3], argv + 3);
    _exit(127);
  }
  timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  int status = 0;
  hwcpipe::counter_sample sample;
  while (waitpid(pid, &status, WNOHANG) == 0) {
    next.tv_nsec += period_us * 1000;
    while (next.tv_nsec >= 1000000000) {
      next.tv_nsec -= 1000000000;
      ++next.tv_sec;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
    if (sampler.sample_now()) continue;
    bool first = true;
    for (hwcpipe_counter c : chosen) {
      if (sampler.get_counter_value(c, sample)) continue;
      if (first) {
        std::fprintf(out, "%llu", static_cast<unsigned long long>(sample.timestamp));
        first = false;
      }
      if (sample.type == hwcpipe::counter_sample::type::uint64) {
        std::fprintf(out, ",%llu", static_cast<unsigned long long>(sample.value.uint64));
      } else {
        std::fprintf(out, ",%.6g", sample.value.float64);
      }
    }
    std::fprintf(out, "\n");
  }
  (void)sampler.stop_sampling();
  std::fclose(out);
  return 0;
}
