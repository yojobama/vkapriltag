#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// vk_eval --ref: runs the unmodified libapriltag CPU detector over the listed images (built only with
// VKAPRILTAG_UPSTREAM_VALIDATION=ON). The dump goes to *out if it is not null; the median total times to *all_total.
void RunReference(const std::vector<std::string> &paths, const std::string &family, uint32_t decimation,
                  uint32_t threads, int warmup, int iterations, std::ofstream *out, std::vector<double> *all_total);
