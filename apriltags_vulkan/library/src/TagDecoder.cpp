#include "vkapriltag/TagDecoder.h"

#include <algorithm>

#include "vkapriltag/common/Profiler.h"

extern "C" {
#include "common/g2d.h"
#include "common/matd.h"

// Non-static entry points exposed by cmake/patches/apriltag-expose-decode-steps.patch,
// forward-declared here.
void quad_decode_index(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im,
                       image_u8_t *im_samples, zarray_t *detections);
void reconcile_detections(zarray_t *detections, zarray_t *poly0, zarray_t *poly1);
// Upstream's gradient-based corner refinement, exposed by the same patch.
void refine_edges(apriltag_detector_t *td, image_u8_t *im_orig, struct quad *quad);
}

namespace apriltag_vulkan {
namespace {

double NsToMs(prof::Ns ns) { return static_cast<double>(ns) / 1.0e6; }

int DetectionCompare(const void *_a, const void *_b) {
  apriltag_detection_t *a = *(apriltag_detection_t *const *)_a;
  apriltag_detection_t *b = *(apriltag_detection_t *const *)_b;
  return a->id - b->id;
}

void ClearDetections(zarray_t *detections) {
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    apriltag_detection_destroy(det);
  }
  zarray_truncate(detections, 0);
}

// Empties a per-quad scratch array without destroying its elements; detections_ owns them.
void ResetScratch(zarray_t *scratch) { zarray_truncate(scratch, 0); }

}  // namespace

TagDecoder::TagDecoder(apriltag_detector_t *td, uint32_t decimation, uint32_t cpu_threads,
                       RefineEdgesMethod refine_method)
    : td_(td),
      pool_(std::make_unique<WorkerPool>(ResolveThreadCount(cpu_threads), "tag-worker")),
      refine_method_(ResolveRefineEdgesMethod(refine_method)) {
  // refine_edges() reads td_->quad_decimate for its search radius.
  td_->quad_decimate = static_cast<float>(decimation);
  poly0_ = g2d_polygon_create_zeros(4);
  poly1_ = g2d_polygon_create_zeros(4);
  detections_ = zarray_create(sizeof(apriltag_detection_t *));
  for (unsigned i = 0; i < pool_->threads(); ++i) {
    probe_.push_back(zarray_create(sizeof(apriltag_detection_t *)));
  }
  slot_stats_.assign(pool_->threads(), SlotStats{});
}

TagDecoder::~TagDecoder() {
  // detections_ owns the detection objects; the per_quad_ arrays are freed structurally only.
  ClearDetections(detections_);
  zarray_destroy(detections_);
  for (zarray_t *z : per_quad_) {
    zarray_destroy(z);
  }
  for (zarray_t *z : probe_) {
    zarray_destroy(z);
  }
  zarray_destroy(poly1_);
  zarray_destroy(poly0_);
}

zarray_t *TagDecoder::Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                             uint32_t width, uint32_t height, bool reversed_border) {
  ClearDetections(detections_);

  image_u8_t im{
      .width = static_cast<int32_t>(width),
      .height = static_cast<int32_t>(height),
      .stride = static_cast<int32_t>(width),
      .buf = const_cast<uint8_t *>(gray_frame),
  };

  // One scratch zarray per quad, so tasks share no mutable state beyond td_->mutex.
  while (per_quad_.size() < quads.size()) {
    per_quad_.push_back(zarray_create(sizeof(apriltag_detection_t *)));
  }
  // Only [0, quads.size()): entries beyond hold stale pointers from earlier frames.
  for (size_t i = 0; i < quads.size(); ++i) ResetScratch(per_quad_[i]);

  VKA_PROF_SCOPE("tag.decode_all");
  const bool timing = prof::Enabled();
  const prof::Ns wall0 = timing ? prof::NowNs() : 0;
  if (timing) slot_stats_.assign(pool_->threads(), SlotStats{});

  pool_->ParallelFor(quads.size(), [&](size_t i, unsigned slot) {
    prof::Scope quad_scope("tag.quad");
    SlotStats &st = slot_stats_[slot];
    const prof::Ns t_quad = timing ? prof::NowNs() : 0;
    prof::Ns t_mark = t_quad;
    // Milliseconds since the previous lap; only called when `timing`.
    auto lap = [&] {
      const prof::Ns now = prof::NowNs();
      const double ms = NsToMs(now - t_mark);
      t_mark = now;
      return ms;
    };
    // Every exit goes through here exactly once, so each quad is counted once.
    auto finish = [&](bool decoded, double refine_ms) {
      quad_scope.set_arg(decoded ? 1 : 0);
      if (!timing) return;
      ++st.quads;
      if (decoded) ++st.decoded;
      (decoded ? st.refine_ms_decoded : st.refine_ms_rejected) += refine_ms;
      st.busy_ms += NsToMs(prof::NowNs() - t_quad);
    };

    const DetectedQuad &q = quads[i];
    struct quad quad_original;
    for (int k = 0; k < 4; ++k) {
      quad_original.p[k][0] = static_cast<float>(q.p[k][0]);
      quad_original.p[k][1] = static_cast<float>(q.p[k][1]);
    }
    quad_original.reversed_border = reversed_border;
    quad_original.H = nullptr;
    quad_original.Hinv = nullptr;

    // kUltraFast refines only quads that already decode unrefined; the rest are rejected here.
    if (td_->refine_edges && refine_method_ == RefineEdgesMethod::kUltraFast) {
      bool decodes;
      {
        VKA_PROF_SCOPE("tag.probe");
        zarray_t *probe = probe_[slot];
        zarray_truncate(probe, 0);
        quad_decode_index(td_, &quad_original, &im, /*im_samples=*/nullptr, probe);
        decodes = zarray_size(probe) > 0;
        ClearDetections(probe);
        if (quad_original.H) matd_destroy(quad_original.H);
        if (quad_original.Hinv) matd_destroy(quad_original.Hinv);
        quad_original.H = nullptr;
        quad_original.Hinv = nullptr;
      }
      if (timing) st.probe_ms += lap();
      if (!decodes) {
        finish(false, 0.0);
        return;
      }
    }

    // Refine (if enabled) before decode; both are safe to run concurrently across quads.
    double refine_ms = 0.0;
    if (td_->refine_edges) {
      VKA_PROF_SCOPE("tag.refine");
      RefineEdges(refine_method_, td_, &im, &quad_original);
      if (timing) refine_ms = lap();
    }

    // quad_decode_index appends decodes to per_quad_[i] and allocates quad->H/Hinv.
    {
      VKA_PROF_SCOPE("tag.decode");
      quad_decode_index(td_, &quad_original, &im, /*im_samples=*/nullptr, per_quad_[i]);
    }

    // Free the homographies allocated on the stack-local quad.
    if (quad_original.H) matd_destroy(quad_original.H);
    if (quad_original.Hinv) matd_destroy(quad_original.Hinv);

    if (timing) st.decode_ms += lap();
    finish(zarray_size(per_quad_[i]) > 0, refine_ms);
  });

  // Merge in quad order over [0, quads.size()), as in the reset loop above.
  for (size_t i = 0; i < quads.size(); ++i) {
    zarray_t *z = per_quad_[i];
    for (int j = 0; j < zarray_size(z); ++j) {
      apriltag_detection_t *det;
      zarray_get(z, j, &det);
      zarray_add(detections_, &det);
    }
  }

  reconcile_detections(detections_, poly0_, poly1_);
  zarray_sort(detections_, DetectionCompare);

  if (timing) {
    Stats stats;
    stats.wall_ms = NsToMs(prof::NowNs() - wall0);
    unsigned active = 0;
    double busy_sum = 0.0;
    for (const SlotStats &st : slot_stats_) {
      stats.quads += st.quads;
      stats.decoded += st.decoded;
      stats.probe_ms += st.probe_ms;
      stats.refine_ms_decoded += st.refine_ms_decoded;
      stats.refine_ms_rejected += st.refine_ms_rejected;
      stats.decode_ms += st.decode_ms;
      stats.busy_ms_max = std::max(stats.busy_ms_max, st.busy_ms);
      if (st.busy_ms > 0.0) {
        ++active;
        busy_sum += st.busy_ms;
      }
    }
    stats.busy_ms_mean = active > 0 ? busy_sum / active : 0.0;
    last_stats_ = stats;
  }
  return detections_;
}

}  // namespace apriltag_vulkan
