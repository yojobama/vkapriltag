#pragma once

// Metrics and comparison helpers shared by the PGM-only and OpenCV validate_against_libapriltag variants;
// header-only and OpenCV-free.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/common/Profiler.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {
namespace validate {

// --profile / --trace handling shared by the validate tools. Call Begin() before constructing the
// GpuDetector (it decides at construction whether to create the GPU timestamp pool).
struct ProfileOptions {
  bool summary = false;
  std::string trace_path;

  bool enabled() const { return summary || !trace_path.empty(); }
  void Begin() const {
    if (!enabled()) return;
#if VKAPRILTAG_PROFILING
    prof::Enable(true);
    prof::SetThreadName("main");
#else
    std::cerr << "--profile/--trace ignored: built with VKAPRILTAG_ENABLE_PROFILING=OFF" << std::endl;
#endif
  }
};

// The per-quad CPU-tail breakdown from TagDecoder::Stats (last Decode()).
inline void PrintTagDecodeStats(const apriltag_vulkan::TagDecoder::Stats &s) {
  if (s.quads == 0) return;
  const double refine = s.refine_ms_decoded + s.refine_ms_rejected;
  std::cout << "  tag_decode breakdown (last iteration): quads=" << s.quads
            << ", decoded=" << s.decoded << ", wall=" << s.wall_ms << " ms, cpu=" << s.cpu_ms()
            << " ms (parallelism x" << (s.wall_ms > 0 ? s.cpu_ms() / s.wall_ms : 0.0) << ")\n"
            << "    refine: decoded quads " << s.refine_ms_decoded << " ms, quads that never decode "
            << s.refine_ms_rejected << " ms";
  if (refine > 0) {
    std::cout << " (" << (100.0 * s.refine_ms_rejected / refine) << "% of refine time)";
  }
  std::cout << "\n    probe=" << s.probe_ms << " ms, final decode=" << s.decode_ms
            << " ms, busiest thread=" << s.busy_ms_max << " ms vs mean " << s.busy_ms_mean
            << " ms" << std::endl;
}

// Prints the span summary and writes the trace, as requested. Call once the instrumented work has
// finished.
inline void FinishProfile(const ProfileOptions &opt) {
  if (!opt.enabled() || !prof::Enabled()) return;
  if (opt.summary) {
    std::cout << "Profile summary (all iterations, all threads; spans nest, so totals overlap):\n"
              << prof::Summary();
  }
  if (!opt.trace_path.empty()) {
    if (prof::WriteTrace(opt.trace_path)) {
      std::cout << "Wrote trace (" << prof::EventCount() << " events) to " << opt.trace_path
                << " - open at https://ui.perfetto.dev" << std::endl;
    } else {
      std::cerr << "Failed to write trace to " << opt.trace_path << std::endl;
    }
  }
}

// Prints the per-dispatch GPU span breakdown and the gaps between spans.
// Returns the summed span time, used to report the residual against the host-observed submit+wait cost.
inline double PrintGpuStageBreakdown(const apriltag_vulkan::GpuDetector::DetectProfile &profile) {
  if (!profile.has_gpu_stage_breakdown) {
    std::cout << "  (per-dispatch GPU span breakdown unavailable; "
                 "set APRILTAG_VK_TIMESTAMPS=1 on a device with timestamp queries)"
              << std::endl;
    return 0.0;
  }
  double gpu_span_total = 0.0;
  std::cout << "  GPU stage breakdown (last iteration, APRILTAG_VK_TIMESTAMPS=1):" << std::endl;
  for (size_t s = 0; s < apriltag_vulkan::GpuDetector::kGpuStageNames.size(); ++s) {
    std::cout << "    " << apriltag_vulkan::GpuDetector::kGpuStageNames[s] << "="
              << profile.gpu_stage_ms[s] << " ms" << std::endl;
    gpu_span_total += profile.gpu_stage_ms[s];
  }
  std::cout << "    (sum of spans = " << gpu_span_total << " ms)" << std::endl;

  double intra_submit_gap_total = 0.0;
  double submit_boundary_gap_total = 0.0;
  std::cout << "  GPU inter-span gaps (purely GPU-clock, no CPU/fence time):" << std::endl;
  for (size_t g = 0; g < profile.gpu_gap_ms.size(); ++g) {
    const bool crosses_submit = profile.gap_crosses_submit[g];
    std::cout << "    " << apriltag_vulkan::GpuDetector::kGpuStageNames[g] << "->"
              << apriltag_vulkan::GpuDetector::kGpuStageNames[g + 1] << "="
              << profile.gpu_gap_ms[g] << " ms"
              << (crosses_submit ? "  (submit boundary)" : "") << std::endl;
    (crosses_submit ? submit_boundary_gap_total : intra_submit_gap_total) += profile.gpu_gap_ms[g];
  }
  std::cout << "    (intra-submit gap total = " << intra_submit_gap_total
            << " ms, submit-boundary gap total = " << submit_boundary_gap_total << " ms)"
            << std::endl;
  return gpu_span_total;
}

// Prints the host's own cost of driving the GPU.
inline void PrintHostSubmissionCost(const apriltag_vulkan::GpuDetector::DetectProfile &profile,
                                    double gpu_span_total) {
  std::cout << "  Host-side submission cost (last iteration): begin=" << profile.cpu_begin_ms
            << " ms, submit+wait=" << profile.cpu_submit_wait_ms
            << " ms, counter_reads=" << profile.cpu_counter_read_ms << " ms, over "
            << profile.submits << " submit(s)" << std::endl;
  if (profile.has_gpu_stage_breakdown) {
    std::cout << "    unspanned GPU time (submit+wait minus spans) = "
              << (profile.cpu_submit_wait_ms - gpu_span_total) << " ms" << std::endl;
  }
}

struct DetectionInfo {
  int id = 0;
  double p[4][2] = {};
};

inline std::vector<DetectionInfo> ExtractDetections(const zarray_t *detections) {
  std::vector<DetectionInfo> out;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *det;
    zarray_get(const_cast<zarray_t *>(detections), i, &det);
    DetectionInfo info;
    info.id = det->id;
    for (int c = 0; c < 4; ++c) {
      info.p[c][0] = det->p[c][0];
      info.p[c][1] = det->p[c][1];
    }
    out.push_back(info);
  }
  std::sort(out.begin(), out.end(), [](const DetectionInfo &a, const DetectionInfo &b) {
    return a.id < b.id;
  });
  return out;
}

inline std::vector<int> SortedIds(const std::vector<DetectionInfo> &detections) {
  std::vector<int> ids;
  ids.reserve(detections.size());
  for (const auto &d : detections) ids.push_back(d.id);
  return ids;
}

inline void PrintIds(const char *label, const std::vector<int> &ids) {
  std::cout << label << ": [";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

// Per-tag corner RMS distance, in pixels, for every tag ID present in both `ours` and `ref`.
// Corner index i in `ours` is compared directly with index i in `ref`.
struct CornerComparison {
  int compared_tags = 0;
  double mean_rms = 0.0;  // mean over compared tags of that tag's per-corner RMS
  double max_rms = 0.0;   // worst single tag's per-corner RMS
};

inline CornerComparison CompareCorners(const std::vector<DetectionInfo> &ours,
                                       const std::vector<DetectionInfo> &ref) {
  CornerComparison result;
  double sum_rms = 0.0;
  size_t oi = 0, ri = 0;
  // Both vectors are sorted by id, so this is a linear merge.
  while (oi < ours.size() && ri < ref.size()) {
    if (ours[oi].id < ref[ri].id) {
      ++oi;
    } else if (ref[ri].id < ours[oi].id) {
      ++ri;
    } else {
      double sq_sum = 0.0;
      for (int c = 0; c < 4; ++c) {
        const double dx = ours[oi].p[c][0] - ref[ri].p[c][0];
        const double dy = ours[oi].p[c][1] - ref[ri].p[c][1];
        sq_sum += dx * dx + dy * dy;
      }
      const double rms = std::sqrt(sq_sum / 4.0);
      sum_rms += rms;
      result.max_rms = std::max(result.max_rms, rms);
      ++result.compared_tags;
      ++oi;
      ++ri;
    }
  }
  if (result.compared_tags > 0) result.mean_rms = sum_rms / result.compared_tags;
  return result;
}

// min/median/max of stage timings collected across --iterations repetitions.
struct Stats {
  double best = 0.0, median = 0.0, worst = 0.0;
};

inline Stats ComputeStats(std::vector<double> values) {
  Stats s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.best = values.front();
  s.worst = values.back();
  s.median = values[values.size() / 2];
  return s;
}

// One row of the --csv output.
struct ImageMetrics {
  std::string file;
  uint32_t width = 0, height = 0;
  int iterations = 0;

  Stats gpu_total_ms;
  Stats quad_decode_ms;
  Stats tag_decode_ms;
  Stats pipeline_ms;

  uint32_t candidate_quads = 0;
  uint32_t boundary_points = 0;
  uint32_t raw_blobs = 0;
  uint32_t selected_blobs = 0;
  uint32_t points = 0;
  uint32_t uf_iterations = 0;
  bool uf_converged = true;

  bool ids_match = false;
  CornerComparison corners;

  // DP corner-seeding counters; both 0 when config.quad_fit_method is the default kPeaks.
  // dp_fallbacks/dp_attempts is the fraction of blobs where the combinatorial search ran anyway.
  uint32_t dp_attempts = 0;
  uint32_t dp_fallbacks = 0;
};

// Appends one CSV row, writing the header first if `path` doesn't exist yet. Fields are not escaped.
inline void AppendCsvRow(const std::string &path, const ImageMetrics &m) {
  const bool write_header = !std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!f) {
    std::cerr << "Warning: could not open --csv path '" << path << "' for writing." << std::endl;
    return;
  }
  if (write_header) {
    f << "file,width,height,iterations,"
        "gpu_ms_best,gpu_ms_median,gpu_ms_worst,"
        "quad_decode_ms_best,quad_decode_ms_median,"
        "tag_decode_ms_best,tag_decode_ms_median,"
        "pipeline_ms_best,pipeline_ms_median,"
        "candidate_quads,boundary_points,raw_blobs,selected_blobs,points,"
        "uf_iterations,uf_converged,ids_match,"
        "corner_compared_tags,corner_rms_mean,corner_rms_max,"
        "dp_attempts,dp_fallbacks\n";
  }
  f << m.file << ',' << m.width << ',' << m.height << ',' << m.iterations << ','
    << m.gpu_total_ms.best << ',' << m.gpu_total_ms.median << ',' << m.gpu_total_ms.worst << ','
    << m.quad_decode_ms.best << ',' << m.quad_decode_ms.median << ','
    << m.tag_decode_ms.best << ',' << m.tag_decode_ms.median << ','
    << m.pipeline_ms.best << ',' << m.pipeline_ms.median << ','
    << m.candidate_quads << ',' << m.boundary_points << ',' << m.raw_blobs << ','
    << m.selected_blobs << ',' << m.points << ','
    << m.uf_iterations << ',' << (m.uf_converged ? 1 : 0) << ',' << (m.ids_match ? 1 : 0) << ','
    << m.corners.compared_tags << ',' << m.corners.mean_rms << ',' << m.corners.max_rms << ','
    << m.dp_attempts << ',' << m.dp_fallbacks << '\n';
}

// A zero-padded frame copy whose pointer and allocation are multiples of `alignment`, for importing
// with GpuDetector::ImportHostFrame; alloc_bytes() is the size to pass.
class AlignedFrame {
 public:
  AlignedFrame(const uint8_t *src, size_t bytes, size_t alignment) {
    alloc_bytes_ = (bytes + alignment - 1) / alignment * alignment;
#ifdef _WIN32
    data_ = static_cast<uint8_t *>(_aligned_malloc(alloc_bytes_, alignment));
#else
    data_ = static_cast<uint8_t *>(std::aligned_alloc(alignment, alloc_bytes_));
#endif
    if (data_ == nullptr) throw std::bad_alloc();
    std::memset(data_, 0, alloc_bytes_);
    std::memcpy(data_, src, bytes);
  }
  ~AlignedFrame() {
#ifdef _WIN32
    _aligned_free(data_);
#else
    std::free(data_);
#endif
  }
  AlignedFrame(const AlignedFrame &) = delete;
  AlignedFrame &operator=(const AlignedFrame &) = delete;

  uint8_t *data() const { return data_; }
  size_t alloc_bytes() const { return alloc_bytes_; }

 private:
  uint8_t *data_ = nullptr;
  size_t alloc_bytes_ = 0;
};

// Largest corner distance, in pixels, between each quad in `a` and its nearest quad in `b`
// (order-insensitive: the GPU does not emit quads in a fixed order). Negative if the counts differ.
inline double MaxQuadDeviation(const std::vector<apriltag_vulkan::DetectedQuad> &a,
                               const std::vector<apriltag_vulkan::DetectedQuad> &b) {
  if (a.size() != b.size()) return -1.0;
  double worst = 0.0;
  for (const apriltag_vulkan::DetectedQuad &qa : a) {
    double best = std::numeric_limits<double>::infinity();
    for (const apriltag_vulkan::DetectedQuad &qb : b) {
      double d = 0.0;
      for (int k = 0; k < 4; ++k) {
        d = std::max(d, std::hypot(qa.p[k][0] - qb.p[k][0], qa.p[k][1] - qb.p[k][1]));
      }
      best = std::min(best, d);
    }
    worst = std::max(worst, best);
  }
  return worst;
}

// Appends frame `index`'s quads to `out` sorted by corner values at full precision, so dumps can be
// diffed between runs.
// Writes one line per decoded tag, sorted by id then centre, with every value at full precision.
inline void DumpDetections(std::ostream &out, int index, const zarray_t *detections) {
  std::vector<const apriltag_detection_t *> dets;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *det;
    zarray_get(const_cast<zarray_t *>(detections), i, &det);
    dets.push_back(det);
  }
  std::sort(dets.begin(), dets.end(),
            [](const apriltag_detection_t *a, const apriltag_detection_t *b) {
    if (a->id != b->id) return a->id < b->id;
    return std::lexicographical_compare(a->c, a->c + 2, b->c, b->c + 2);
  });
  out << "F " << index << ' ' << dets.size() << '\n';
  char line[512];
  for (const apriltag_detection_t *d : dets) {
    std::snprintf(line, sizeof line,
                  "%d %d %.9g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", d->id,
                  d->hamming, d->decision_margin, d->c[0], d->c[1], d->p[0][0], d->p[0][1],
                  d->p[1][0], d->p[1][1], d->p[2][0], d->p[2][1], d->p[3][0], d->p[3][1]);
    out << line;
  }
}

inline void DumpCanonicalQuads(std::ostream &out, int index,
                               std::vector<apriltag_vulkan::DetectedQuad> quads) {
  std::sort(quads.begin(), quads.end(),
            [](const apriltag_vulkan::DetectedQuad &a, const apriltag_vulkan::DetectedQuad &b) {
              return std::lexicographical_compare(&a.p[0][0], &a.p[0][0] + 8, &b.p[0][0],
                                                  &b.p[0][0] + 8);
            });
  out << "F " << index << ' ' << quads.size() << '\n';
  char line[320];
  for (const apriltag_vulkan::DetectedQuad &q : quads) {
    std::snprintf(line, sizeof line, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", q.p[0][0],
                  q.p[0][1], q.p[1][0], q.p[1][1], q.p[2][0], q.p[2][1], q.p[3][0], q.p[3][1]);
    out << line;
  }
}

// Frame ingest A/B: detects from `plain` (copied each frame) and `imported` (zero-copy) in ABBA
// order and compares quads with the first copy-path frame; returns the number of differing frames.
inline int RunIngestAb(apriltag_vulkan::GpuDetector &detector,
                       apriltag_vulkan::QuadDecode &quad_decode,
                       apriltag_vulkan::TagDecoder &tag_decoder, const uint8_t *plain,
                       const uint8_t *imported, uint32_t width, uint32_t height,
                       bool reversed_border, int iterations, int rounds = 4) {
  using Clock = std::chrono::steady_clock;
  struct Samples {
    std::vector<double> gpu_total, upload, clear_span, threshold_span, pipeline;
  };
  Samples path[2];  // 0 = copy, 1 = import
  std::vector<apriltag_vulkan::DetectedQuad> reference;
  bool have_reference = false;
  int quad_mismatches = 0;
  // Per path: frames whose quad count differed from the reference, and the worst corner deviation
  // (pixels) of any frame whose count matched.
  int count_differs[2] = {0, 0};
  double worst_deviation[2] = {0.0, 0.0};

  auto run = [&](int which, int n, bool record) {
    const uint8_t *frame = which == 0 ? plain : imported;
    for (int it = 0; it < n; ++it) {
      const auto t0 = Clock::now();
      detector.Detect(frame);
      const auto &profile = detector.last_profile();
      const double gpu_total = profile.total_ms;
      const double upload = profile.upload_ms;
      const double clear_span = profile.gpu_stage_ms[0];
      const double threshold_span = profile.gpu_stage_ms[1];
      const std::vector<apriltag_vulkan::DetectedQuad> quads =
          quad_decode.Decode(detector.last_line_fit_points, detector.last_quad_fits);
      tag_decoder.Decode(quads, frame, width, height, reversed_border);
      const double pipeline = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      if (!have_reference) {
        reference = quads;
        have_reference = true;
      } else {
        const double dev = MaxQuadDeviation(quads, reference);
        if (dev < 0.0) {
          ++quad_mismatches;
          ++count_differs[which];
        } else {
          worst_deviation[which] = std::max(worst_deviation[which], dev);
        }
      }
      if (!record) continue;
      Samples &s = path[which];
      s.gpu_total.push_back(gpu_total);
      s.upload.push_back(upload);
      s.clear_span.push_back(clear_span);
      s.threshold_span.push_back(threshold_span);
      s.pipeline.push_back(pipeline);
    }
  };

  run(0, 50, false);  // warm-up; the first copy-path frame becomes the reference
  run(1, 50, false);
  for (int r = 0; r < rounds; ++r) {
    run(0, iterations, true);
    run(1, iterations, true);
    run(1, iterations, true);
    run(0, iterations, true);
  }

  const bool have_spans = detector.last_profile().has_gpu_stage_breakdown;
  auto row = [&](const char *label, std::vector<double> Samples::*field) {
    const Stats copy = ComputeStats(path[0].*field);
    const Stats imp = ComputeStats(path[1].*field);
    const double delta = copy.median > 0 ? 100.0 * (imp.median - copy.median) / copy.median : 0.0;
    std::printf("  %-22s copy median %8.4f (best %8.4f)   import median %8.4f (best %8.4f)   %+6.1f%%\n",
                label, copy.median, copy.best, imp.median, imp.best, delta);
  };
  std::printf("Ingest A/B, %d rounds x 2 x %d frames per path (ABBA), ms:\n", rounds, iterations);
  row("Detect() total", &Samples::gpu_total);
  row("host upload (memcpy)", &Samples::upload);
  if (have_spans) {
    row("GPU span: clear", &Samples::clear_span);
    row("GPU span: threshold", &Samples::threshold_span);
  } else {
    std::printf("  (GPU span rows skipped: set APRILTAG_VK_TIMESTAMPS=1 or pass --profile)\n");
  }
  row("pipeline_total", &Samples::pipeline);
  std::printf(
      "  quads vs the first copy-path frame (%zu quads): frames with a different count: copy %d, "
      "import %d; worst corner deviation: copy %.3g px, import %.3g px\n",
      reference.size(), count_differs[0], count_differs[1], worst_deviation[0],
      worst_deviation[1]);
  return quad_mismatches;
}

}  // namespace validate
}  // namespace apriltag_vulkan
