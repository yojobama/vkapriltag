#include "vkapriltag/TagDecoder.h"

#include "vkapriltag/common/Options.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include "BorderSample.h"
#include "Geometry.h"
#include "SmallQuad.h"
#include "TestHooks.h"
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

// A struct quad whose homographies (allocated by quad_decode_index and quad_sample_code) are freed with it.
class OwnedQuad {
 public:
  OwnedQuad() { Reset(); }
  // A copy of `from` that owns no homographies.
  explicit OwnedQuad(const struct quad &from) : q(from) { Reset(); }
  OwnedQuad(const OwnedQuad &) = delete;
  OwnedQuad &operator=(const OwnedQuad &) = delete;
  ~OwnedQuad() { Free(); }

  void Free() {
    if (q.H) matd_destroy(q.H);
    if (q.Hinv) matd_destroy(q.Hinv);
    Reset();
  }
  // Hands the caller the homography; the quad no longer owns it.
  matd_t *ReleaseH() {
    matd_t *h = q.H;
    q.H = nullptr;
    return h;
  }

  struct quad q;

 private:
  void Reset() {
    q.H = nullptr;
    q.Hinv = nullptr;
  }
};

// The quad whose corners are the tag-coordinate points (us[k], vs[k]) projected through `h`.
void QuadFromHomography(const matd_t *h, const double us[4], const double vs[4], bool reversed_border,
                        struct quad *out) {
  for (int k = 0; k < 4; ++k) {
    double x, y;
    homography_project(h, us[k], vs[k], &x, &y);
    out->p[k][0] = static_cast<float>(x);
    out->p[k][1] = static_cast<float>(y);
  }
  out->reversed_border = reversed_border;
  out->H = nullptr;
  out->Hinv = nullptr;
}

// The inner `width_cells` cells of a (width_cells + 2)-cell square whose homography is `h`.
void InnerQuad(const matd_t *h, int width_cells, bool reversed_border, struct quad *out) {
  const double r = static_cast<double>(width_cells) / (width_cells + 2);
  const double us[4] = {-r, r, r, -r}, vs[4] = {-r, -r, r, r};  // quad_update_homographies' corners
  QuadFromHomography(h, us, vs, reversed_border, out);
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
  if (background_thread) o.big_cores = false;
  return o;
}

}  // namespace

TagDecoder::TagDecoder(apriltag_detector_t *td, uint32_t decimation, uint32_t cpu_threads,
                       RefineEdgesMethod refine_method, float probe_margin,
                       const WorkerPoolOptions &pool_options, bool background_thread,
                       bool small_tag_rescue)
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
  const Options env = Options::FromEnvironment();
  grey_margin_ = env.Bool("APRILTAG_VK_GREY_MARGIN", grey_margin_);
  small_rescue_ = ResolveSmallTagRescue(env, small_tag_rescue);
  polish_ = env.Bool("APRILTAG_VK_POLISH", polish_);
  side_one_out_ = env.Bool("APRILTAG_VK_SIDE_ONE_OUT", side_one_out_);
  third_bit_ = env.Bool("APRILTAG_VK_THIRD_BIT", third_bit_);
  family_limits_ = env.Bool("APRILTAG_VK_FAMILY_LIMITS", family_limits_);
  border_precheck_ = env.Bool("APRILTAG_VK_BORDER_PRECHECK", border_precheck_);
  dark_reject_ = env.Bool("APRILTAG_VK_DARK_REJECT", dark_reject_);
  early_reject_ = env.Bool("APRILTAG_VK_EARLY_REJECT", early_reject_);
  collect_probe_ = env.Bool("APRILTAG_VK_PROBE_STATS", collect_probe_);
  probe_margin_ = static_cast<float>(env.Float("APRILTAG_VK_PROBE_MARGIN", probe_margin_));
  if (const auto csv = env.String("APRILTAG_VK_PROBE_CSV")) {
    probe_csv_ = *csv;
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

// What every candidate of one Decode() call shares: the frame, the switches and the family, read once.
struct DecodeEnv {
  apriltag_detector_t *td;
  image_u8_t *im;
  bool reversed_border;
  bool timing;
  RefineEdgesMethod refine_method;
  int min_white_cells;
  const ThirdBitIndex *third_index;
  apriltag_family_t *family;  // the first registered family, null if none
  int family_count;
  bool grey_margin, small_rescue, polish, side_one_out, third_bit, family_limits, border_precheck,
      dark_reject, early_reject, collect_probe;
  float probe_margin;
};

namespace {

// Splits a candidate's time into laps; every call is a no-op unless profiling is on.
class StepClock {
 public:
  explicit StepClock(bool on) : on_(on), start_(on ? prof::NowNs() : 0), mark_(start_) {}
  // Adds the milliseconds since the previous lap to *total.
  void Lap(double *total) {
    if (!on_) return;
    const prof::Ns now = prof::NowNs();
    *total += NsToMs(now - mark_);
    mark_ = now;
  }
  double SinceStartMs() const { return on_ ? NsToMs(prof::NowNs() - start_) : 0.0; }

 private:
  bool on_;
  prof::Ns start_, mark_;
};

// A quad sampled as the decoder would, before it is refined: the code word read, the decision margin
// and the homography built for it (null if the quad is degenerate).
struct UnrefinedSample {
  UnrefinedSample(const DecodeEnv &env, const struct quad &from) : quad(from) {
    margin = quad_sample_code(env.td, &quad.q, env.im, &rcode);
  }

  float margin = -1.0f;
  uint64_t rcode = 0;
  OwnedQuad quad;
};

// One quad candidate and the state its steps share.
struct Candidate {
  Candidate(const DecodeEnv &env, const DetectedQuad &input, zarray_t *found, zarray_t *probe)
      : input(input), found(found), probe(probe), clock(env.timing) {
    for (int k = 0; k < 4; ++k) {
      quad.q.p[k][0] = static_cast<float>(input.p[k][0]);
      quad.q.p[k][1] = static_cast<float>(input.p[k][1]);
    }
    quad.q.reversed_border = env.reversed_border;
  }

  const DetectedQuad &input;
  OwnedQuad quad;  // refined in place; owns the homographies quad_decode_index allocates
  zarray_t *found;  // this candidate's detections, owned by detections_ once merged
  zarray_t *probe;  // the thread's scratch for trial decodes
  std::optional<UnrefinedSample> unrefined;  // taken once, shared by the early checks
  int probe_bin = -1;
  float probe_margin = 0.0f;
  StepClock clock;
  double probe_ms = 0.0, refine_ms = 0.0, decode_ms = 0.0;
};

// A coarse box is refitted to a quad from the full-resolution frame; false if it cannot be (or the
// small tag rescue is off), which drops the candidate.
bool NormaliseCandidate(const DecodeEnv &env, Candidate &c) {
  if (!c.input.coarse) return true;
  double fitted[4][2];
  const double xlo = std::min(c.input.p[0][0], c.input.p[2][0]), xhi = std::max(c.input.p[0][0], c.input.p[2][0]);
  const double ylo = std::min(c.input.p[0][1], c.input.p[2][1]), yhi = std::max(c.input.p[0][1], c.input.p[2][1]);
  if (!env.small_rescue || !FitSmallQuad(env.im->buf, env.im->width, env.im->height, xlo, ylo, xhi, yhi, fitted)) {
    return false;
  }
  for (int k = 0; k < 4; ++k) {
    c.quad.q.p[k][0] = static_cast<float>(fitted[k][0]);
    c.quad.q.p[k][1] = static_cast<float>(fitted[k][1]);
  }
  return true;
}

// The unrefined sample of the candidate's quad, taken on first use.
const UnrefinedSample &SharedSample(const DecodeEnv &env, Candidate &c) {
  if (!c.unrefined) c.unrefined.emplace(env, c.quad.q);
  return *c.unrefined;
}

// The margin gate and the probe statistics: a candidate whose unrefined decision margin is below the
// gate is rejected.
bool MarginGateRejects(const DecodeEnv &env, Candidate &c) {
  // The gate reads the first family only, so it needs exactly one registered.
  const bool gate = env.probe_margin > 0.0f && env.family_count == 1;
  if (!((env.collect_probe || gate) && env.td->refine_edges)) return false;
  const UnrefinedSample &sample = SharedSample(env, c);
  c.probe_margin = sample.margin;
  // The nearest-code distance is only needed for the statistics.
  c.probe_bin = sample.margin < 0.0f ? 16
                : env.collect_probe ? std::min(15, apriltag_min_hamming(env.td, sample.rcode))
                                    : 0;
  c.clock.Lap(&c.probe_ms);
  return gate && sample.margin < env.probe_margin;
}

// A border read below 22 of 28 dark cells, or fewer white payload cells than the sparsest code has
// (less two cells of error correction), cannot be a tag. Unreadable quads are kept.
bool SampleCannotDecode(const DecodeEnv &env, const UnrefinedSample &sample) {
  const bool single_family = env.family_count == 1;
  bool bad = false;
  if (env.border_precheck && single_family) {
    BorderLevels border;
    const bool read = sample.quad.q.H && ReadBorder(sample.quad.q.H, env.im, env.family->width_at_border, &border);
    bad = read && border.dark * 28 < 22 * border.cells;
  }
  const bool check_dark = env.dark_reject && !env.reversed_border && single_family;
  if (check_dark && !bad) bad = std::popcount(sample.rcode) + 2 < env.min_white_cells;
  return bad;
}

// The border and dark checks on the unrefined quad. With grey margin on (see early_reject), a quad is
// rejected only if its inner W cells also fail them, as that is what the grey margin retry decodes.
bool PreRefineRejects(const DecodeEnv &env, Candidate &c) {
  const bool single_family = env.family_count == 1;
  const bool check_border = env.border_precheck && single_family;
  const bool check_dark = env.dark_reject && !env.reversed_border && single_family;
  apriltag_family_t *family = single_family ? env.family : nullptr;
  // Under grey margin a coarse box or a quad under 1.5 pixels per cell is not rejected: its unrefined sample is too coarse.
  bool large_enough = true;
  if (env.grey_margin && family) {
    const double min_side = geom::MinSide(c.quad.q.p);
    large_enough = !c.input.coarse && min_side >= 1.5 * family->width_at_border;
    // A quad the small rescue would refit when it fails to decode is kept.
    if (env.small_rescue && env.td->quad_decimate > 1 && !c.input.coarse && c.input.gradient >= 60.0f) {
      const geom::Bounds box = geom::QuadBounds(c.input.p);
      if (box.xhi - box.xlo <= 40.0 && box.yhi - box.ylo <= 40.0) large_enough = false;
    }
  }
  if (!((!env.grey_margin || (env.early_reject && large_enough)) && (check_border || check_dark))) return false;
  const UnrefinedSample &outer = SharedSample(env, c);
  bool reject = SampleCannotDecode(env, outer);
  if (reject && env.grey_margin) {
    reject = false;
    if (outer.quad.q.H) {
      struct quad inner;
      InnerQuad(outer.quad.q.H, family->width_at_border, env.reversed_border, &inner);
      reject = SampleCannotDecode(env, UnrefinedSample(env, inner));
    }
  }
  c.clock.Lap(&c.probe_ms);
  return reject;
}

// Refines the quad's corners against the image, if edge refinement is on.
void RefineCandidate(const DecodeEnv &env, Candidate &c) {
  if (!env.td->refine_edges) return;
  VKA_PROF_SCOPE("tag.refine");
  RefineEdges(env.refine_method, env.td, env.im, &c.quad.q);
  c.clock.Lap(&c.refine_ms);
}

// Decodes the quad into c.found; quad_decode_index allocates the quad's homographies.
void DecodeCandidate(const DecodeEnv &env, Candidate &c) {
  VKA_PROF_SCOPE("tag.decode");
  quad_decode_index(env.td, &c.quad.q, env.im, /*im_samples=*/nullptr, c.found);
}

// A grey quiet zone on a bright wall makes the quiet zone's outline the dark blob, so the quad is
// the (W + 2) cell square: if it did not decode, try its inner W cells, once that decodes unrefined.
void TryInnerSquare(const DecodeEnv &env, Candidate &c) {
  if (!(env.grey_margin && env.td->refine_edges && zarray_size(c.found) == 0 && c.quad.q.H && env.family_count > 0)) {
    return;
  }
  VKA_PROF_SCOPE("tag.inner");
  OwnedQuad inner;
  InnerQuad(c.quad.q.H, env.family->width_at_border, env.reversed_border, &inner.q);
  zarray_truncate(c.probe, 0);
  quad_decode_index(env.td, &inner.q, env.im, /*im_samples=*/nullptr, c.probe);
  const bool decodes = zarray_size(c.probe) > 0;
  ClearDetections(c.probe);
  inner.Free();
  if (!decodes) return;
  RefineEdges(env.refine_method, env.td, env.im, &inner.q);
  quad_decode_index(env.td, &inner.q, env.im, /*im_samples=*/nullptr, c.found);
}

// Moves a side whose border cells read white one cell in and refits; keeps the result if it decodes
// the same tag with a darker border.
void TrySideOneOut(const DecodeEnv &env, Candidate &c) {
  if (!(env.side_one_out && c.quad.q.H && zarray_size(c.found) == 1 && env.family_count > 0)) return;
  const int w = env.family->width_at_border;
  BorderLevels before;
  int bad = 0;
  if (ReadBorder(c.quad.q.H, env.im, w, &before)) {
    for (int k = 0; k < 4; ++k) bad |= (before.side_level[k] > 0.3) << k;
  }
  if (bad == 0 || bad == 15) return;
  VKA_PROF_SCOPE("tag.side");
  const double one = 2.0 / w;  // a cell, in the tag's [-1, 1] coordinates
  const double u0 = (bad & 8) ? -1 + one : -1, u1 = (bad & 2) ? 1 - one : 1;
  const double v0 = (bad & 1) ? -1 + one : -1, v1 = (bad & 4) ? 1 - one : 1;
  const double us[4] = {u0, u1, u1, u0}, vs[4] = {v0, v0, v1, v1};
  OwnedQuad moved;
  QuadFromHomography(c.quad.q.H, us, vs, env.reversed_border, &moved.q);
  RefineEdges(env.refine_method, env.td, env.im, &moved.q);
  zarray_truncate(c.probe, 0);
  quad_decode_index(env.td, &moved.q, env.im, /*im_samples=*/nullptr, c.probe);
  bool keep = false;
  if (zarray_size(c.probe) == 1 && moved.q.H) {
    apriltag_detection_t *old_det = nullptr, *new_det = nullptr;
    zarray_get(c.found, 0, &old_det);
    zarray_get(c.probe, 0, &new_det);
    BorderLevels after;
    if (new_det->id == old_det->id && ReadBorder(moved.q.H, env.im, w, &after)) {
      double level_before = 0, level_after = 0;
      for (int k = 0; k < 4; ++k) level_before += before.side_level[k], level_after += after.side_level[k];
      keep = level_after < level_before - 0.2;
    }
    if (keep) {
      apriltag_detection_destroy(old_det);
      zarray_set(c.found, 0, &new_det, nullptr);
      zarray_truncate(c.probe, 0);  // the detection now belongs to c.found
    }
  }
  if (!keep) ClearDetections(c.probe);
}

// A third corrected bit, accepted only when the three wrong cells touch: damage such as a finger
// or glare is coherent, a chance match is scattered.
void TryThirdBit(const DecodeEnv &env, Candidate &c) {
  if (!(env.third_bit && env.td->refine_edges && zarray_size(c.found) == 0 && AllowsThirdBit(env.td))) return;
  VKA_PROF_SCOPE("tag.thirdbit");
  OwnedQuad sampled(c.quad.q);
  uint64_t rcode = 0;
  const float margin = quad_sample_code(env.td, &sampled.q, env.im, &rcode);
  int id = 0, rotation = 0;
  if (!(margin >= 0 && sampled.q.H && MatchThreeTouchingBits(*env.third_index, rcode, &id, &rotation))) return;
  apriltag_detection_t *det = static_cast<apriltag_detection_t *>(calloc(1, sizeof(apriltag_detection_t)));
  det->family = env.family;
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
  det->H = matd_op("M*M", sampled.q.H, R);
  matd_destroy(R);
  homography_project(det->H, 0, 0, &det->c[0], &det->c[1]);
  for (int k = 0; k < 4; ++k) {  // counter-clockwise from (-1, -1), as quad_decode_index
    homography_project(det->H, (k == 1 || k == 2) ? 1 : -1, (k < 2) ? 1 : -1, &det->p[k][0], &det->p[k][1]);
  }
  zarray_add(c.found, &det);
}

// A small quad that did not decode: refit it from the full-resolution image and try again.
void TrySmallRescue(const DecodeEnv &env, Candidate &c) {
  const DetectedQuad &in = c.input;
  if (!(env.small_rescue && env.td->quad_decimate > 1 && !in.coarse && in.gradient >= 60.0f && env.td->refine_edges &&
        zarray_size(c.found) == 0)) {
    return;
  }
  const geom::Bounds box = geom::QuadBounds(in.p);
  double fitted[4][2];
  if (!(box.xhi - box.xlo <= 40.0 && box.yhi - box.ylo <= 40.0 &&
        FitSmallQuad(env.im->buf, env.im->width, env.im->height, box.xlo, box.ylo, box.xhi, box.yhi, fitted))) {
    return;
  }
  VKA_PROF_SCOPE("tag.rescue");
  OwnedQuad refit;
  for (int k = 0; k < 4; ++k) {
    refit.q.p[k][0] = static_cast<float>(fitted[k][0]);
    refit.q.p[k][1] = static_cast<float>(fitted[k][1]);
  }
  refit.q.reversed_border = env.reversed_border;
  RefineEdges(env.refine_method, env.td, env.im, &refit.q);
  quad_decode_index(env.td, &refit.q, env.im, /*im_samples=*/nullptr, c.found);
}

// How RunSteps ended; only kFinished candidates reach the probe statistics.
enum class Outcome { kDropped, kMarginGate, kPreCheck, kFinished };

Outcome RunSteps(const DecodeEnv &env, Candidate &c) {
  if (!NormaliseCandidate(env, c)) return Outcome::kDropped;

  if (MarginGateRejects(env, c)) return Outcome::kMarginGate;
  if (PreRefineRejects(env, c)) return Outcome::kPreCheck;
  c.unrefined.reset();

  RefineCandidate(env, c);
  DecodeCandidate(env, c);

  TryInnerSquare(env, c);

  TrySideOneOut(env, c);

  TryThirdBit(env, c);

  TrySmallRescue(env, c);

  // Drop corrections beyond what the family can bear before they count as detections.
  if (env.family_limits) {
    for (int j = zarray_size(c.found) - 1; j >= 0; --j) {
      apriltag_detection_t *det = nullptr;
      zarray_get(c.found, j, &det);
      // the third-bit match is the deliberate exception to the family limit
      if (det->hamming > MaxCorrectedBits(det->family) && !(env.third_bit && det->hamming == 3)) {
        apriltag_detection_destroy(det);
        zarray_remove_index(c.found, j, 0);
      }
    }
  }

  if (env.polish) {
    VKA_PROF_SCOPE("tag.polish");
    for (int j = 0; j < zarray_size(c.found); ++j) {
      apriltag_detection_t *det = nullptr;
      zarray_get(c.found, j, &det);
      PolishDetection(env.im, det);
    }
    c.clock.Lap(&c.refine_ms);
  }

  c.clock.Lap(&c.decode_ms);
  return Outcome::kFinished;
}

}  // namespace

void TagDecoder::ProcessCandidate(const DecodeEnv &env, const DetectedQuad &q, size_t i, unsigned slot) {
  prof::Scope quad_scope("tag.quad");
  Candidate c(env, q, per_quad_[i], probe_[slot]);
  const Outcome outcome = RunSteps(env, c);
  const bool decoded = zarray_size(c.found) > 0;
  if (outcome == Outcome::kMarginGate && env.collect_probe) slot_probe_[slot].rejected[c.probe_bin]++;
  if (outcome == Outcome::kFinished && env.collect_probe && c.probe_bin >= 0) {
    (decoded ? slot_probe_[slot].decoded : slot_probe_[slot].rejected)[c.probe_bin]++;
    if (!probe_csv_.empty()) {
      float perimeter = 0.0f;
      for (int k = 0; k < 4; ++k) {
        const double dx = q.p[k][0] - q.p[(k + 1) % 4][0];
        const double dy = q.p[k][1] - q.p[(k + 1) % 4][1];
        perimeter += static_cast<float>(std::sqrt(dx * dx + dy * dy));
      }
      slot_rows_[slot].push_back({c.probe_bin, c.probe_margin, perimeter, decoded});
    }
  }
  quad_scope.set_arg(decoded ? 1 : 0);
  if (!env.timing) return;
  SlotStats &st = slot_stats_[slot];
  ++st.quads;
  if (decoded) ++st.decoded;
  st.probe_ms += c.probe_ms;
  (decoded ? st.refine_ms_decoded : st.refine_ms_rejected) += c.refine_ms;
  st.decode_ms += c.decode_ms;
  st.busy_ms += c.clock.SinceStartMs();
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
  const int family_count = zarray_size(td_->tag_families);
  apriltag_family_t *first_family = nullptr;
  if (family_count > 0) zarray_get(td_->tag_families, 0, &first_family);
  int min_white_cells = 0;
  if (dark_reject_ && family_count == 1) {
    apriltag_family_t *family = first_family;
    min_white_cells = family->nbits;
    for (uint32_t k = 0; k < family->ncodes; ++k) {
      min_white_cells = std::min(min_white_cells, std::popcount(family->codes[k]));
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

  const DecodeEnv env = {
      .td = td_,
      .im = &im,
      .reversed_border = reversed_border,
      .timing = timing,
      .refine_method = refine_method_,
      .min_white_cells = min_white_cells,
      .third_index = &third_index_,
      .family = family_count > 0 ? first_family : nullptr,
      .family_count = family_count,
      .grey_margin = grey_margin_,
      .small_rescue = small_rescue_,
      .polish = polish_,
      .side_one_out = side_one_out_,
      .third_bit = third_bit_,
      .family_limits = family_limits_,
      .border_precheck = border_precheck_,
      .dark_reject = dark_reject_,
      .early_reject = early_reject_,
      .collect_probe = collect_probe_,
      .probe_margin = probe_margin_,
  };
  pool_->ParallelFor(quads.size(), [&](size_t i, unsigned slot) { ProcessCandidate(env, quads[i], i, slot); });

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

namespace testing {

uint64_t Rotate90(uint64_t w, int nbits) { return apriltag_vulkan::Rotate90(w, nbits); }
bool ErrorsConnected(const apriltag_family_t *f, uint64_t diff) { return apriltag_vulkan::ErrorsConnected(f, diff); }
void BuildThirdBitIndex(const apriltag_family_t *f, ThirdBitIndex *index) { apriltag_vulkan::BuildThirdBitIndex(f, index); }
bool MatchThreeTouchingBits(const ThirdBitIndex &index, uint64_t rcode, int *id, int *rotation) {
  return apriltag_vulkan::MatchThreeTouchingBits(index, rcode, id, rotation);
}
int MaxCorrectedBits(const apriltag_family_t *f) { return apriltag_vulkan::MaxCorrectedBits(f); }

}  // namespace testing

}  // namespace apriltag_vulkan
