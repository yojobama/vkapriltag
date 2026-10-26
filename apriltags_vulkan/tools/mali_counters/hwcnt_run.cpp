// Runs a command and prints every Mali hardware counter over its lifetime.
#include <hwcpipe/counter_database.hpp>
#include <hwcpipe/gpu.hpp>
#include <hwcpipe/sampler.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: hwcnt_run cmd [args...]\n");
    return 1;
  }
  auto gpu = hwcpipe::gpu(0);
  if (!gpu) return 2;
  std::fprintf(stderr, "cores=%u bus_width=%u\n", static_cast<unsigned>(gpu.num_shader_cores()),
               static_cast<unsigned>(gpu.bus_width()));
  auto db = hwcpipe::counter_database{};
  auto config = hwcpipe::sampler_config(gpu);
  std::vector<hwcpipe_counter> counters;
  for (hwcpipe_counter c : db.counters_for_gpu(gpu)) {
    if (!config.add_counter(c)) counters.push_back(c);
  }
  auto sampler = hwcpipe::sampler<>(config);
  if (auto ec = sampler.start_sampling()) {
    std::fprintf(stderr, "start_sampling: %s\n", ec.message().c_str());
    return 3;
  }
  (void)sampler.sample_now();
  const pid_t pid = fork();
  if (pid == 0) {
    execvp(argv[1], argv + 1);
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  (void)sampler.sample_now();
  hwcpipe::counter_metadata meta;
  hwcpipe::counter_sample sample;
  for (hwcpipe_counter c : counters) {
    if (sampler.get_counter_value(c, sample)) continue;
    (void)db.describe_counter(c, meta);
    if (sample.type == hwcpipe::counter_sample::type::uint64) {
      std::printf("%s\t%llu\t%s\n", meta.name, static_cast<unsigned long long>(sample.value.uint64), meta.units);
    } else {
      std::printf("%s\t%.6g\t%s\n", meta.name, sample.value.float64, meta.units);
    }
  }
  (void)sampler.stop_sampling();
  return WIFEXITED(status) ? WEXITSTATUS(status) : 4;
}
