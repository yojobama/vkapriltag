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
  // `probe_margin` is the margin gate in gray levels (see SetProbeMargin; 0 = off); the
  // APRILTAG_VK_PROBE_MARGIN environment variable overrides it.
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

  // Where the last Decode() spent its time. Filled only while prof::Enabled() (the per-quad
  // timing it needs is otherwise skipped); all zero before that. CPU times are summed over
  // threads, so cpu_ms() / wall_ms is the achieved parallelism.
  struct Stats {
    uint32_t quads = 0;    // candidates received
    uint32_t decoded = 0;  // candidates that produced a detection (before reconcile_detections)
    double wall_ms = 0.0;  // Decode() start to return, including the merge and reconcile
    double probe_ms = 0.0;  // kUltraFast's unrefined decode gate
    // Refinement is the dominant cost; split by whether the quad went on to decode, since time
    // spent refining quads that never decode is the part an earlier reject could save.
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

  // Margin gate (off by default). Set through the constructor's `probe_margin`, this setter, or
  // APRILTAG_VK_PROBE_MARGIN=<gray levels> (which overrides the constructor value).
  // When edge refinement is on, each candidate quad is first sampled unrefined, and quads whose
  // decision margin (the contrast between the white and black bit cells, in gray levels) is below
  // this value, or whose border polarity is wrong, are rejected without being refined: junk quads
  // sit at the noise floor (about 3 for sigma 3 noise) while real tags have margins of tens to
  // hundreds. Every quad that passes is refined and decoded as before, so detections are identical
  // to an ungated run unless a real tag's unrefined margin is below the gate. 0 disables.
  void SetProbeMargin(float gray_levels) { probe_margin_ = gray_levels; }
  float probe_margin() const { return probe_margin_; }

  // Experiment: with APRILTAG_VK_PROBE_STATS=1 every candidate quad (when edge refinement is on) is
  // also sampled unrefined, and the Hamming distance from those bits to the nearest code is
  // histogrammed, split by whether the full refine + decode accepted the quad. Bins 0..14 are the
  // distance, 15 means farther, 16 means rejected before the lookup (polarity test or no
  // homography). Cumulative over Decode() calls.
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
  // One row per quad for offline analysis, written to APRILTAG_VK_PROBE_CSV at destruction.
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
