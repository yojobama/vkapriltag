#include "vkapriltag/TagDecoder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <bit>
#include <string>

#include "BorderSample.h"
#include "SmallQuad.h"
#include "vkapriltag/PolishFit.h"
#include "vkapriltag/common/Profiler.h"

extern "C" {
#include "common/g2d.h"
#include "common/homography.h"
#include "common/matd.h"

// Non-static entry points exposed by cmake/patches/apriltag-expose-decode-steps.patch,
// forward-declared here.
void quad_decode_index(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im,
                       image_u8_t *im_samples, zarray_t *detections);
void reconcile_detections(zarray_t *detections, zarray_t *poly0, zarray_t *poly1);
// Upstream's gradient-based corner refinement, exposed by the same patch.
void refine_edges(apriltag_detector_t *td, image_u8_t *im_orig, struct quad *quad);
// Bit sampling without the code lookup, and the nearest-code distance, exposed by
// cmake/patches/apriltag-expose-code-probe.patch.
float quad_sample_code(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im,
                       uint64_t *rcode);
int apriltag_min_hamming(apriltag_detector_t *td, uint64_t rcode);
}

namespace apriltag_vulkan {
namespace {

// libapriltag's rotate90() on an nbits-bit code word.
uint64_t Rotate90(uint64_t w, int nbits) {
  int p = nbits;
  uint64_t l = 0;
  if (nbits % 4 == 1) {
    p = nbits - 1;
    l = 1;
  }
  w = ((w >> l) << (p / 4 + l)) | (w >> (3 * p / 4 + l) << l) | (w & l);
  return w & ((uint64_t{1} << nbits) - 1);
}

// Whether the set bits of `diff` (bit k from the most significant end is the family's k-th cell)
// form one 8-connected group of cells.
bool ErrorsConnected(const apriltag_family_t *f, uint64_t diff) {
  int cells[8], n = 0;
  for (int k = 0; k < static_cast<int>(f->nbits) && n < 8; ++k) {
    if ((diff >> (f->nbits - 1 - k)) & 1) cells[n++] = k;
  }
  if (n <= 1) return true;
  bool reached[8] = {true};
  for (bool grew = true; grew;) {
    grew = false;
    for (int a = 0; a < n; ++a) {
      if (!reached[a]) continue;
      for (int b = 0; b < n; ++b) {
        if (reached[b]) continue;
        const int dx = static_cast<int>(f->bit_x[cells[a]]) - static_cast<int>(f->bit_x[cells[b]]);
        const int dy = static_cast<int>(f->bit_y[cells[a]]) - static_cast<int>(f->bit_y[cells[b]]);
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1) reached[b] = grew = true;
      }
    }
  }
  for (int a = 0; a < n; ++a) {
    if (!reached[a]) return false;
  }
  return true;
}

void BuildThirdBitIndex(const apriltag_family_t *f, ThirdBitIndex *index) {
  index->family = f;
  int at = 0;
  for (int k = 0; k < 4; ++k) {
    index->shift[k] = at;
    index->bits[k] = (static_cast<int>(f->nbits) - at) / (4 - k);
    at += index->bits[k];
    const uint64_t mask = (uint64_t{1} << index->bits[k]) - 1;
    index->start[k].assign((size_t{1} << index->bits[k]) + 1, 0);
    for (uint32_t i = 0; i < f->ncodes; ++i) ++index->start[k][((f->codes[i] >> index->shift[k]) & mask) + 1];
    for (size_t b = 1; b < index->start[k].size(); ++b) index->start[k][b] += index->start[k][b - 1];
    index->ids[k].assign(f->ncodes, 0);
    std::vector<uint32_t> fill(index->start[k].begin(), index->start[k].end() - 1);
    for (uint32_t i = 0; i < f->ncodes; ++i) index->ids[k][fill[(f->codes[i] >> index->shift[k]) & mask]++] = i;
  }
}

// The one code of the family that `rcode` matches, over the four rotations, with exactly three wrong
// bits that touch (a finger, a cable or glare, not chance); false if there is none or two tie.
bool MatchThreeTouchingBits(const ThirdBitIndex &index, uint64_t rcode, int *id, int *rotation) {
  const apriltag_family_t *f = index.family;
  int found = 0;
  uint64_t r = rcode;
  for (int rot = 0; rot < 4; ++rot, r = Rotate90(r, f->nbits)) {
    uint32_t seen[64];
    int n_seen = 0;
    for (int k = 0; k < 4; ++k) {
      const uint64_t value = (r >> index.shift[k]) & ((uint64_t{1} << index.bits[k]) - 1);
      for (uint32_t p = index.start[k][value]; p < index.start[k][value + 1]; ++p) {
        const uint32_t i = index.ids[k][p];
        if (std::find(seen, seen + n_seen, i) != seen + n_seen) continue;
        if (n_seen < 64) seen[n_seen++] = i;
        const uint64_t diff = r ^ f->codes[i];
        if (std::popcount(diff) > 3) continue;
        if (std::popcount(diff) < 3 || !ErrorsConnected(f, diff)) return false;  // decodable, or not touching
        if (found++ == 0) {
          *id = static_cast<int>(i);
          *rotation = rot;
        }
      }
    }
  }
  return found == 1;
}

// Whether the first family's codes are far enough apart (minimum Hamming distance 9 or more) for a
// third corrected bit to still leave a wide gap to every other code.
bool AllowsThirdBit(const apriltag_detector_t *td) {
  if (zarray_size(td->tag_families) == 0) return false;
  apriltag_family_t *family = nullptr;
  zarray_get(td->tag_families, 0, &family);
  return family->h >= 9;
}

// The most wrong bits worth correcting in a family (at most 2, upstream's default): the largest d
// for which a random bit pattern lands within d bits of one of the family's codes, in any of the four
// rotations, at most 0.5% of the time. tag16h5 gets 0 (3% at one bit), tag36h11 and tag25h9 get 2.
int MaxCorrectedBits(const apriltag_family_t *f) {
  const double patterns = std::ldexp(1.0, static_cast<int>(f->nbits));
  int best = 0;
  double ball = 1.0;  // the number of patterns within d bits of a code
  double choose = 1.0;
  for (int d = 1; d <= 2; ++d) {
    choose = choose * (static_cast<double>(f->nbits) - d + 1) / d;
    ball += choose;
    if (4.0 * f->ncodes * ball / patterns > 0.005) break;
    best = d;
  }
  return best;
}

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

namespace {

// A decoder used from a background thread never pins threads.
WorkerPoolOptions ResolvePoolOptions(const WorkerPoolOptions &configured, bool background_thread) {
  WorkerPoolOptions o = ResolveWorkerPoolOptions(configured);
  if (background_thread) o.big_cores = o.pin_caller = false;
  return o;
}

}  // namespace

TagDecoder::TagDecoder(apriltag_detector_t *td, uint32_t decimation, uint32_t cpu_threads,
                       RefineEdgesMethod refine_method, float probe_margin,
                       const WorkerPoolOptions &pool_options, bool background_thread)
    : td_(td),
      pool_(std::make_unique<WorkerPool>(ResolveThreadCount(cpu_threads), "tag-worker",
                                         ResolvePoolOptions(pool_options, background_thread))),
      refine_method_(ResolveRefineEdgesMethod(refine_method)),
      probe_margin_(probe_margin) {
  // refine_edges() reads td_->quad_decimate for its search radius.
  td_->quad_decimate = static_cast<float>(decimation);
  poly0_ = g2d_polygon_create_zeros(4);
  poly1_ = g2d_polygon_create_zeros(4);
  detections_ = zarray_create(sizeof(apriltag_detection_t *));
  for (unsigned i = 0; i < pool_->threads(); ++i) {
    probe_.push_back(zarray_create(sizeof(apriltag_detection_t *)));
  }
  slot_stats_.assign(pool_->threads(), SlotStats{});
  if (const char *v = std::getenv("APRILTAG_VK_GREY_MARGIN")) grey_margin_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_SMALL_RESCUE")) small_rescue_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_POLISH")) polish_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_SIDE_ONE_OUT")) side_one_out_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_THIRD_BIT")) third_bit_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_FAMILY_LIMITS")) family_limits_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_BORDER_PRECHECK")) border_precheck_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_DARK_REJECT")) dark_reject_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_PROBE_STATS")) collect_probe_ = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_PROBE_MARGIN")) {
    probe_margin_ = static_cast<float>(std::atof(v));
  }
  if (const char *v = std::getenv("APRILTAG_VK_PROBE_CSV")) {
    probe_csv_ = v;
    collect_probe_ = true;
  }
  slot_rows_.assign(pool_->threads(), {});
  slot_probe_.assign(pool_->threads(), ProbeHistogram{});
}

TagDecoder::~TagDecoder() {
  if (!probe_csv_.empty()) {
    if (FILE *f = std::fopen(probe_csv_.c_str(), "a")) {
      for (const ProbeRow &r : rows_) {
        std::fprintf(f, "%d,%.4f,%.2f,%d\n", r.distance, r.margin, r.perimeter, r.decoded ? 1 : 0);
      }
      std::fclose(f);
    }
  }
  if (collect_probe_) {
    uint64_t total = 0;
    for (int b = 0; b < 17; ++b) total += probe_hist_.decoded[b] + probe_hist_.rejected[b];
    if (total > 0) {
      std::fprintf(stderr,
                   "probe histogram (unrefined bits to nearest code; quads accepted / rejected by the "
                   "full refine+decode)\n  distance:");
      for (int b = 0; b < 17; ++b) {
        if (b == 15) {
          std::fprintf(stderr, " >=15");
        } else if (b == 16) {
          std::fprintf(stderr, " n/a");
        } else {
          std::fprintf(stderr, " %d", b);
        }
      }
      std::fprintf(stderr, "\n  accepted:");
      for (int b = 0; b < 17; ++b) {
        std::fprintf(stderr, " %llu", static_cast<unsigned long long>(probe_hist_.decoded[b]));
      }
      std::fprintf(stderr, "\n  rejected:");
      for (int b = 0; b < 17; ++b) {
        std::fprintf(stderr, " %llu", static_cast<unsigned long long>(probe_hist_.rejected[b]));
      }
      std::fprintf(stderr, "\n");
    }
  }
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

  if (third_bit_ && zarray_size(td_->tag_families) > 0) {
    apriltag_family_t *family = nullptr;
    zarray_get(td_->tag_families, 0, &family);
    if (third_index_.family != family) BuildThirdBitIndex(family, &third_index_);
  }
  int min_white_cells_ = 0;
  if (dark_reject_ && zarray_size(td_->tag_families) == 1) {
    apriltag_family_t *family = nullptr;
    zarray_get(td_->tag_families, 0, &family);
    min_white_cells_ = family->nbits;
    for (uint32_t k = 0; k < family->ncodes; ++k) {
      min_white_cells_ = std::min(min_white_cells_, std::popcount(family->codes[k]));
    }
  }

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

    if (q.coarse) {
      double fitted[4][2];
      const double xlo = std::min(q.p[0][0], q.p[2][0]), xhi = std::max(q.p[0][0], q.p[2][0]);
      const double ylo = std::min(q.p[0][1], q.p[2][1]), yhi = std::max(q.p[0][1], q.p[2][1]);
      if (!small_rescue_ || !FitSmallQuad(gray_frame, static_cast<int>(width), static_cast<int>(height), xlo, ylo, xhi, yhi, fitted)) {
        finish(false, 0.0);
        return;
      }
      for (int k = 0; k < 4; ++k) {
        quad_original.p[k][0] = static_cast<float>(fitted[k][0]);
        quad_original.p[k][1] = static_cast<float>(fitted[k][1]);
      }
    }

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

    // Margin gate and probe statistics: sample the unrefined quad.
    int probe_bin = -1;
    float probe_margin = 0.0f;
    // The gate reads the first family only, so it needs exactly one registered.
    const bool gate = probe_margin_ > 0.0f && zarray_size(td_->tag_families) == 1;
    if ((collect_probe_ || gate) && td_->refine_edges &&
        refine_method_ != RefineEdgesMethod::kUltraFast) {
      struct quad unrefined = quad_original;
      unrefined.H = nullptr;
      unrefined.Hinv = nullptr;
      uint64_t rcode = 0;
      const float margin = quad_sample_code(td_, &unrefined, &im, &rcode);
      probe_margin = margin;
      // The nearest-code distance is only needed for the statistics.
      probe_bin = margin < 0.0f ? 16
                  : collect_probe_ ? std::min(15, apriltag_min_hamming(td_, rcode))
                                   : 0;
      if (unrefined.H) matd_destroy(unrefined.H);
      if (unrefined.Hinv) matd_destroy(unrefined.Hinv);
      if (timing) st.probe_ms += lap();
      if (gate && margin < probe_margin_) {
        if (collect_probe_) slot_probe_[slot].rejected[probe_bin]++;
        finish(false, 0.0);
        return;
      }
    }

    // (Skipped with grey_margin_, whose retry needs the quads these checks reject.)
    // A tag's border is black all round: reject a quad where fewer than 22 of 28 border cells (scaled
    // per family) read dark. Quads that cannot be read through the image are kept.
    if (border_precheck_ && !grey_margin_ && zarray_size(td_->tag_families) == 1) {
      apriltag_family_t *family = nullptr;
      zarray_get(td_->tag_families, 0, &family);
      struct quad unrefined = quad_original;
      unrefined.H = nullptr;
      unrefined.Hinv = nullptr;
      uint64_t rcode = 0;
      quad_sample_code(td_, &unrefined, &im, &rcode);
      BorderLevels border;
      const bool read = unrefined.H && ReadBorder(unrefined.H, &im, family->width_at_border, &border);
      if (unrefined.H) matd_destroy(unrefined.H);
      if (unrefined.Hinv) matd_destroy(unrefined.Hinv);
      if (read && border.dark * 28 < 22 * border.cells) {
        finish(false, 0.0);
        return;
      }
    }

    // A quad whose payload shows fewer white cells than the sparsest code has (less the two cells error
    // correction may flip) cannot decode, so skip refining it.
    if (dark_reject_ && !grey_margin_ && !reversed_border && zarray_size(td_->tag_families) == 1) {
      struct quad unrefined = quad_original;
      unrefined.H = nullptr;
      unrefined.Hinv = nullptr;
      uint64_t rcode = 0;
      quad_sample_code(td_, &unrefined, &im, &rcode);
      if (unrefined.H) matd_destroy(unrefined.H);
      if (unrefined.Hinv) matd_destroy(unrefined.Hinv);
      if (std::popcount(rcode) + 2 < min_white_cells_) {
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

    // A grey quiet zone on a bright wall makes the quiet zone's outline the dark blob, so the quad is
    // the (W + 2) cell square: if it did not decode, try its inner W cells, once that decodes unrefined.
    if (grey_margin_ && td_->refine_edges && zarray_size(per_quad_[i]) == 0 && quad_original.H &&
        zarray_size(td_->tag_families) > 0) {
      VKA_PROF_SCOPE("tag.inner");
      apriltag_family_t *family = nullptr;
      zarray_get(td_->tag_families, 0, &family);
      const double r = static_cast<double>(family->width_at_border) / (family->width_at_border + 2);
      struct quad inner;
      const double sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};  // quad_update_homographies' corners
      for (int k = 0; k < 4; ++k) {
        double x, y;
        homography_project(quad_original.H, sx[k] * r, sy[k] * r, &x, &y);
        inner.p[k][0] = static_cast<float>(x);
        inner.p[k][1] = static_cast<float>(y);
      }
      inner.reversed_border = reversed_border;
      inner.H = nullptr;
      inner.Hinv = nullptr;
      zarray_t *probe = probe_[slot];
      zarray_truncate(probe, 0);
      quad_decode_index(td_, &inner, &im, /*im_samples=*/nullptr, probe);
      const bool decodes = zarray_size(probe) > 0;
      ClearDetections(probe);
      if (inner.H) matd_destroy(inner.H);
      if (inner.Hinv) matd_destroy(inner.Hinv);
      inner.H = nullptr;
      inner.Hinv = nullptr;
      if (decodes) {
        RefineEdges(refine_method_, td_, &im, &inner);
        quad_decode_index(td_, &inner, &im, /*im_samples=*/nullptr, per_quad_[i]);
        if (inner.H) matd_destroy(inner.H);
        if (inner.Hinv) matd_destroy(inner.Hinv);
      }
    }

    // A side whose border cells read white was fitted on the quiet zone's outer edge, one cell too far
    // out (the payload still decodes through error correction). Move it in a cell, refit, and keep the
    // result if it decodes the same tag with a darker border.
    if (side_one_out_ && quad_original.H && zarray_size(per_quad_[i]) == 1 &&
        zarray_size(td_->tag_families) > 0) {
      apriltag_family_t *family = nullptr;
      zarray_get(td_->tag_families, 0, &family);
      const int w = family->width_at_border;
      BorderLevels before;
      int bad = 0;
      if (ReadBorder(quad_original.H, &im, w, &before)) {
        for (int k = 0; k < 4; ++k) bad |= (before.side_level[k] > 0.3) << k;
      }
      if (bad != 0 && bad != 15) {
        VKA_PROF_SCOPE("tag.side");
        const double one = 2.0 / w;  // a cell, in the tag's [-1, 1] coordinates
        const double u0 = (bad & 8) ? -1 + one : -1, u1 = (bad & 2) ? 1 - one : 1;
        const double v0 = (bad & 1) ? -1 + one : -1, v1 = (bad & 4) ? 1 - one : 1;
        const double us[4] = {u0, u1, u1, u0}, vs[4] = {v0, v0, v1, v1};
        struct quad moved;
        for (int k = 0; k < 4; ++k) {
          double x, y;
          homography_project(quad_original.H, us[k], vs[k], &x, &y);
          moved.p[k][0] = static_cast<float>(x);
          moved.p[k][1] = static_cast<float>(y);
        }
        moved.reversed_border = reversed_border;
        moved.H = nullptr;
        moved.Hinv = nullptr;
        RefineEdges(refine_method_, td_, &im, &moved);
        zarray_t *probe = probe_[slot];
        zarray_truncate(probe, 0);
        quad_decode_index(td_, &moved, &im, /*im_samples=*/nullptr, probe);
        bool keep = false;
        if (zarray_size(probe) == 1 && moved.H) {
          apriltag_detection_t *old_det = nullptr, *new_det = nullptr;
          zarray_get(per_quad_[i], 0, &old_det);
          zarray_get(probe, 0, &new_det);
          BorderLevels after;
          if (new_det->id == old_det->id && ReadBorder(moved.H, &im, w, &after)) {
            double level_before = 0, level_after = 0;
            for (int k = 0; k < 4; ++k) level_before += before.side_level[k], level_after += after.side_level[k];
            keep = level_after < level_before - 0.2;
          }
          if (keep) {
            apriltag_detection_destroy(old_det);
            zarray_set(per_quad_[i], 0, &new_det, nullptr);
            zarray_truncate(probe, 0);  // the detection now belongs to per_quad_[i]
          }
        }
        if (!keep) ClearDetections(probe);
        if (moved.H) matd_destroy(moved.H);
        if (moved.Hinv) matd_destroy(moved.Hinv);
      }
    }

    // A third corrected bit, accepted only when the three wrong cells touch: damage such as a finger
    // or glare is coherent, a chance match is scattered.
    if (third_bit_ && td_->refine_edges && zarray_size(per_quad_[i]) == 0 && AllowsThirdBit(td_)) {
      VKA_PROF_SCOPE("tag.thirdbit");
      apriltag_family_t *family = nullptr;
      zarray_get(td_->tag_families, 0, &family);
      struct quad sampled = quad_original;
      sampled.H = nullptr;
      sampled.Hinv = nullptr;
      uint64_t rcode = 0;
      const float margin = quad_sample_code(td_, &sampled, &im, &rcode);
      int id = 0, rotation = 0;
      if (margin >= 0 && sampled.H && MatchThreeTouchingBits(third_index_, rcode, &id, &rotation)) {
        apriltag_detection_t *det = static_cast<apriltag_detection_t *>(calloc(1, sizeof(apriltag_detection_t)));
        det->family = family;
        det->id = id;
        det->hamming = 3;
        det->decision_margin = margin;
        const double theta = rotation * 1.5707963267948966;
        matd_t *R = matd_create(3, 3);
        MATD_EL(R, 0, 0) = std::cos(theta);
        MATD_EL(R, 0, 1) = -std::sin(theta);
        MATD_EL(R, 1, 0) = std::sin(theta);
        MATD_EL(R, 1, 1) = std::cos(theta);
        MATD_EL(R, 2, 2) = 1;
        det->H = matd_op("M*M", sampled.H, R);
        matd_destroy(R);
        homography_project(det->H, 0, 0, &det->c[0], &det->c[1]);
        for (int k = 0; k < 4; ++k) {  // counter-clockwise from (-1, -1), as quad_decode_index
          homography_project(det->H, (k == 1 || k == 2) ? 1 : -1, (k < 2) ? 1 : -1, &det->p[k][0], &det->p[k][1]);
        }
        zarray_add(per_quad_[i], &det);
      }
      if (sampled.H) matd_destroy(sampled.H);
      if (sampled.Hinv) matd_destroy(sampled.Hinv);
    }

    // A small quad that did not decode: refit it from the full-resolution image and try again.
    if (small_rescue_ && td_->quad_decimate > 1 && !q.coarse && q.gradient >= 60.0f && td_->refine_edges && zarray_size(per_quad_[i]) == 0) {
      double xlo = 1e30, ylo = 1e30, xhi = -1e30, yhi = -1e30;
      for (int k = 0; k < 4; ++k) {
        xlo = std::min(xlo, q.p[k][0]);
        xhi = std::max(xhi, q.p[k][0]);
        ylo = std::min(ylo, q.p[k][1]);
        yhi = std::max(yhi, q.p[k][1]);
      }
      double fitted[4][2];
      if (xhi - xlo <= 40.0 && yhi - ylo <= 40.0 &&
          FitSmallQuad(gray_frame, static_cast<int>(width), static_cast<int>(height), xlo, ylo, xhi, yhi, fitted)) {
        VKA_PROF_SCOPE("tag.rescue");
        struct quad refit;
        for (int k = 0; k < 4; ++k) {
          refit.p[k][0] = static_cast<float>(fitted[k][0]);
          refit.p[k][1] = static_cast<float>(fitted[k][1]);
        }
        refit.reversed_border = reversed_border;
        refit.H = nullptr;
        refit.Hinv = nullptr;
        RefineEdges(refine_method_, td_, &im, &refit);
        quad_decode_index(td_, &refit, &im, /*im_samples=*/nullptr, per_quad_[i]);
        if (refit.H) matd_destroy(refit.H);
        if (refit.Hinv) matd_destroy(refit.Hinv);
      }
    }

    // Drop corrections beyond what the family can bear before they count as detections.
    if (family_limits_) {
      for (int j = zarray_size(per_quad_[i]) - 1; j >= 0; --j) {
        apriltag_detection_t *det = nullptr;
        zarray_get(per_quad_[i], j, &det);
        // the third-bit match is the deliberate exception to the family limit
        if (det->hamming > MaxCorrectedBits(det->family) && !(third_bit_ && det->hamming == 3)) {
          apriltag_detection_destroy(det);
          zarray_remove_index(per_quad_[i], j, 0);
        }
      }
    }

    if (polish_) {
      VKA_PROF_SCOPE("tag.polish");
      for (int j = 0; j < zarray_size(per_quad_[i]); ++j) {
        apriltag_detection_t *det = nullptr;
        zarray_get(per_quad_[i], j, &det);
        PolishDetection(&im, det);
      }
      if (timing) refine_ms += lap();
    }

    // Free the homographies allocated on the stack-local quad.
    if (quad_original.H) matd_destroy(quad_original.H);
    if (quad_original.Hinv) matd_destroy(quad_original.Hinv);

    if (timing) st.decode_ms += lap();
    if (collect_probe_ && probe_bin >= 0) {
      const bool accepted = zarray_size(per_quad_[i]) > 0;
      (accepted ? slot_probe_[slot].decoded : slot_probe_[slot].rejected)[probe_bin]++;
      if (!probe_csv_.empty()) {
        float perimeter = 0.0f;
        for (int k = 0; k < 4; ++k) {
          const double dx = q.p[k][0] - q.p[(k + 1) % 4][0];
          const double dy = q.p[k][1] - q.p[(k + 1) % 4][1];
          perimeter += static_cast<float>(std::sqrt(dx * dx + dy * dy));
        }
        slot_rows_[slot].push_back({probe_bin, probe_margin, perimeter, accepted});
      }
    }
    finish(zarray_size(per_quad_[i]) > 0, refine_ms);
  });

  for (ProbeHistogram &sp : slot_probe_) {
    for (int b = 0; b < 17; ++b) {
      probe_hist_.decoded[b] += sp.decoded[b];
      probe_hist_.rejected[b] += sp.rejected[b];
    }
    sp = ProbeHistogram{};
  }
  for (std::vector<ProbeRow> &r : slot_rows_) {
    rows_.insert(rows_.end(), r.begin(), r.end());
    r.clear();
  }

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
