#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vkapriltag/RefineEdges.h"
#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/GpuDetector.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Decodes CPU-computed quad candidates (DetectedQuad from QuadDecode) into tags via the `apriltag`
// library's quad_decode_index and reconcile_detections (exposed by
// cmake/patches/apriltag-expose-decode-steps.patch): id, hamming distance, decision margin,
// centre and homography-refined corners. If `td->refine_edges` is set, each quad is refined
// (see RefineEdgesMethod) before decoding. Stops at 2D detection; pose is not computed here.
// Quads are decoded in parallel over a WorkerPool and merged in quad order, so the output does
// not depend on thread count.
class TagDecoder {
 public:
  // `td` must already have its tag families added; it is not owned.
  // `decimation` must match DetectorConfig::decimation; it is stored in td->quad_decimate for
  // refine_edges (used only if `td->refine_edges` is set).
  // `cpu_threads` is the total parallelism (0 = hardware_concurrency, or APRILTAG_CPU_THREADS).
  // `refine_method` may be overridden by APRILTAG_VK_REFINE; used only if `td->refine_edges` is set.
  // Margin gate in grey levels (0 = off); APRILTAG_VK_PROBE_MARGIN overrides it.
  explicit TagDecoder(apriltag_detector_t *td, uint32_t decimation = 1, uint32_t cpu_threads = 0,
                      RefineEdgesMethod refine_method = RefineEdgesMethod::kExact,
                      float probe_margin = 0.0f);
  ~TagDecoder();

  TagDecoder(const TagDecoder &) = delete;
  TagDecoder &operator=(const TagDecoder &) = delete;

  // Decodes `quads` against the full-resolution grayscale frame (width*height bytes, tightly
  // packed). `reversed_border` must match the border polarity of the family (one polarity per run).
  // Returns detections owned by this TagDecoder, valid until the next Decode() or destruction.
  zarray_t *Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                   uint32_t width, uint32_t height, bool reversed_border);

  // Timings of the last Decode(), filled only while profiling is enabled; CPU times are summed
  // over threads.
  struct Stats {
    uint32_t quads = 0;    // candidates received
    uint32_t decoded = 0;  // candidates that produced a detection (before reconcile_detections)
    double wall_ms = 0.0;  // Decode() start to return, including the merge and reconcile
    double probe_ms = 0.0;  // kUltraFast's unrefined decode gate
    // Refinement time, split by whether the quad went on to decode.
    double refine_ms_decoded = 0.0;
    double refine_ms_rejected = 0.0;
    double decode_ms = 0.0;      // final quad_decode_index
    double busy_ms_max = 0.0;    // busiest thread; with busy_ms_mean, the load imbalance
    double busy_ms_mean = 0.0;   // over the threads that did any work
    double cpu_ms() const {
      return probe_ms + refine_ms_decoded + refine_ms_rejected + decode_ms;
    }
  };
  const Stats &last_stats() const { return last_stats_; }

  // Margin gate (off by default): quads whose unrefined decision margin is below this, or whose
  // border polarity is wrong, skip refinement. APRILTAG_VK_PROBE_MARGIN overrides it.
  void SetProbeMargin(float grey_levels) { probe_margin_ = grey_levels; }
  float probe_margin() const { return probe_margin_; }

  // With APRILTAG_VK_PROBE_STATS=1, histograms each unrefined quad's Hamming distance to the nearest
  // code by outcome (bins 0-14, 15 = farther, 16 = rejected before the lookup).
  struct ProbeHistogram {
    uint64_t decoded[17] = {};
    uint64_t rejected[17] = {};
  };
  const ProbeHistogram &probe_histogram() const { return probe_hist_; }

 private:
  apriltag_detector_t *td_;  // not owned
  std::unique_ptr<WorkerPool> pool_;
  RefineEdgesMethod refine_method_;
  zarray_t *poly0_;
  zarray_t *poly1_;
  zarray_t *detections_;
  // One scratch zarray per candidate quad; grown, never shrunk, and truncated when reused.
  std::vector<zarray_t *> per_quad_;
  // Per-thread scratch for the unrefined decode that gates kUltraFast refinement.
  std::vector<zarray_t *> probe_;

  // One cache line per thread so the per-quad accumulators do not false-share.
  struct alignas(64) SlotStats {
    double probe_ms, refine_ms_decoded, refine_ms_rejected, decode_ms, busy_ms;
    uint32_t quads, decoded;
  };
  std::vector<SlotStats> slot_stats_;
  bool collect_probe_ = false;
  float probe_margin_ = 0.0f;
  std::vector<ProbeHistogram> slot_probe_;  // per thread, merged into probe_hist_ per Decode()
  ProbeHistogram probe_hist_;
  // One row per quad, written to APRILTAG_VK_PROBE_CSV at destruction.
  struct ProbeRow {
    int distance;
    float margin;
    float perimeter;
    bool decoded;
  };
  std::vector<std::vector<ProbeRow>> slot_rows_;
  std::vector<ProbeRow> rows_;
  std::string probe_csv_;
  Stats last_stats_;
};

}  // namespace apriltag_vulkan
