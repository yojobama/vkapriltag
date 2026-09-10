#pragma once

// Shared metrics/comparison helpers for the two validate_against_libapriltag
// variants (PGM-only and OpenCV). Kept header-only and dependency-free
// (no OpenCV) so both variants can use it unconditionally.
//
// Phase 0 of the ArUco-derived optimization plan: before touching detector
// code, the harness needs to (a) time the WHOLE pipeline (GPU + quad_decode +
// tag_decode) per iteration, not just Detect(), since items 2/3 are mostly
// CPU-tail changes that a GPU-only timer can't see, and (b) compare per-tag
// CORNER positions against the reference detector, not just decoded ID sets,
// since item 3 (DP corner seeding) can regress corner accuracy while leaving
// the ID set unchanged. Both gaps existed in the original tool.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {
namespace validate {

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

// Per-tag corner RMS distance, in pixels, for every tag ID present in BOTH
// `ours` and `ref`. Corner index correspondence is assumed (both detectors
// derive p[4][2] the same way - homography-refined corners in a fixed
// winding order - from the same input image), so index i in `ours` compares
// directly against index i in `ref` without a nearest-corner search.
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
  // Both vectors are sorted by id (ExtractDetections), so this is a linear
  // merge rather than an O(n*m) search.
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

// min/median/max of a list of stage timings collected across --iterations
// repetitions, so cold-start cost (first-touch page faults, pipeline
// warm-up) doesn't dominate the reported number.
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

// One row of the --csv output: everything needed to diff a before/after run,
// or compare the same run across the host and Orange Pi machines, without
// eyeballing console output.
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

  // Item 3 (DP corner seeding) instrumentation. Both 0 when
  // config.quad_fit_method is the default kPeaks. dp_fallbacks/dp_attempts
  // is the fraction of blobs where DP couldn't produce a valid quad and the
  // combinatorial search ran anyway - a high rate means DP is adding cost
  // without saving the search's.
  uint32_t dp_attempts = 0;
  uint32_t dp_fallbacks = 0;
};

// Appends one CSV row, writing the header first if `path` doesn't exist yet.
// Intentionally simple (no escaping): every field is a filename, a count, or
// a number, none of which can contain a comma in practice here.
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

}  // namespace validate
}  // namespace apriltag_vulkan
