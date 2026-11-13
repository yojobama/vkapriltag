#include "vkapriltag/TagDecoder.h"

#include "vkapriltag/common/Options.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "BorderSample.h"
#include "DecodeSwitches.h"
#include "Geometry.h"
#include "OwnDecode.h"
#include "SmallQuad.h"
#include "TestHooks.h"
#include "vkapriltag/PolishFit.h"
#include "vkapriltag/common/Profiler.h"

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
bool ErrorsConnected(const Family *f, uint64_t diff) {
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

void BuildThirdBitIndex(const Family *f, ThirdBitIndex *index) {
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
  const Family *f = index.family;
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
bool AllowsThirdBit(const Family *first_family) { return first_family != nullptr && first_family->h >= 9; }

// The most wrong bits worth correcting in a family (at most 2, upstream's default): the largest d
// for which a random bit pattern lands within d bits of one of the family's codes, in any of the four
// rotations, at most 0.5% of the time. tag16h5 gets 0 (3% at one bit), tag36h11 and tag25h9 get 2.
int MaxCorrectedBits(const Family *f) {
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

// A Quad with the homography of its corners; has_h is false if they are degenerate or the
// quad has not been decoded or sampled yet.
struct OwnedQuad {
  OwnedQuad() = default;
  // A copy of `from` without a homography.
  explicit OwnedQuad(const Quad &from) : q(from) {}

  Quad q = {};
  own::Homography h = {};
  bool has_h = false;
};

// The quad whose corners are the tag-coordinate points (us[k], vs[k]) projected through `h`.
void QuadFromHomography(const double *h, const double us[4], const double vs[4], bool reversed_border,
                        Quad *out) {
  for (int k = 0; k < 4; ++k) {
    double x, y;
    own::Project(h, us[k], vs[k], &x, &y);
    out->p[k][0] = static_cast<float>(x);
    out->p[k][1] = static_cast<float>(y);
  }
  out->reversed_border = reversed_border;
}

// The inner `width_cells` cells of a (width_cells + 2)-cell square whose homography is `h`.
void InnerQuad(const double *h, int width_cells, bool reversed_border, Quad *out) {
  const double r = static_cast<double>(width_cells) / (width_cells + 2);
  const double us[4] = {-r, r, r, -r}, vs[4] = {-r, -r, r, r};  // quad_update_homographies' corners
  QuadFromHomography(h, us, vs, reversed_border, out);
}

double NsToMs(prof::Ns ns) { return static_cast<double>(ns) / 1.0e6; }

}  // namespace

namespace {

// A decoder used from a background thread never pins threads.
WorkerPoolOptions ResolvePoolOptions(const WorkerPoolOptions &configured, bool background_thread) {
  WorkerPoolOptions o = ResolveWorkerPoolOptions(configured);
  if (background_thread) o.big_cores = false;
  return o;
}

}  // namespace

TagDecoder::TagDecoder(const TagDecoderConfig &config, uint32_t decimation, uint32_t cpu_threads,
                       RefineEdgesMethod refine_method, float probe_margin,
                       const WorkerPoolOptions &pool_options, bool background_thread,
                       bool small_tag_rescue, std::shared_ptr<WorkerPool> pool)
    : config_(config),
      pool_(pool ? std::move(pool)
                 : std::make_shared<WorkerPool>(ResolveThreadCount(cpu_threads), "tag-worker",
                                                ResolvePoolOptions(pool_options, background_thread))),
      refine_method_(ResolveRefineEdgesMethod(refine_method)),
      decimation_(static_cast<int>(decimation)),
      probe_margin_(probe_margin) {
  own_decoder_ = std::make_unique<own::Decoder>();
  probe_.resize(pool_->threads());
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
}

// What every candidate of one Decode() call shares: the frame, the switches and the family, read once.
struct DecodeEnv : DecodeSwitches {
  bool refine_edges = true;
  const Image *im = nullptr;
  bool reversed_border = false;
  bool timing = false;
  RefineEdgesMethod refine_method = RefineEdgesMethod::kProfile;
  int min_white_cells = 0;
  const ThirdBitIndex *third_index = nullptr;
  const Family *family = nullptr;  // the first registered family, null if none
  int decimation = 1;
  int width_at_border = 0;  // the first family's, 0 if none
  int family_count = 0;
  const own::Decoder *own = nullptr;
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

// Decodes the quad against the families, adding each tag to `found` (only counting if null). Returns
// the number of tags; `raw` receives the first family's raw sample.
int DecodeQuad(const DecodeEnv &env, OwnedQuad &quad, DetectionList *found, own::RawSample *raw = nullptr) {
  own::DecodedTag tags[own::Decoder::kMaxFamilies];
  const int n = env.own->Decode(quad.q.p, quad.q.reversed_border, env.im, &quad.h, raw, tags);
  quad.has_h = n >= 0;
  for (int i = 0; found && i < n; ++i) {
    found->push_back(own::Decoder::ToDetection(tags[i]));
  }
  return std::max(n, 0);
}

// A quad sampled as the decoder would, before it is refined: the code word read, the decision margin
// and the homography built for it (null if the quad is degenerate).
struct UnrefinedSample {
  UnrefinedSample(const DecodeEnv &env, const Quad &from) : quad(from) {
    own::RawSample raw;
    quad.has_h = env.own->SampleCode(quad.q.p, quad.q.reversed_border, env.im, &quad.h, &raw);
    margin = raw.valid ? raw.margin : -1.0f;
    rcode = raw.rcode;
  }

  float margin = -1.0f;
  uint64_t rcode = 0;
  OwnedQuad quad;
};

// One quad candidate and the state its steps share.
struct Candidate {
  Candidate(const DecodeEnv &env, const DetectedQuad &input, DetectionList *found, DetectionList *probe)
      : input(input), found(found), probe(probe), clock(env.timing) {
    for (int k = 0; k < 4; ++k) {
      quad.q.p[k][0] = static_cast<float>(input.p[k][0]);
      quad.q.p[k][1] = static_cast<float>(input.p[k][1]);
    }
    quad.q.reversed_border = env.reversed_border;
  }

  const DetectedQuad &input;
  OwnedQuad quad;  // refined in place; its homography is set by the decode
  DetectionList *found;  // this candidate's detections, copied to the result once merged
  DetectionList *probe;  // the thread's scratch for trial decodes
  std::optional<UnrefinedSample> unrefined;  // taken once, shared by the early checks
  own::RawSample raw;  // the final decode's raw sample, for the third bit
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
  if (!env.small_rescue || !FitSmallQuad(env.im->data, env.im->width, env.im->height, env.im->stride, xlo, ylo, xhi, yhi, fitted)) {
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
  if (!((env.collect_probe || gate) && env.refine_edges)) return false;
  const UnrefinedSample &sample = SharedSample(env, c);
  c.probe_margin = sample.margin;
  // The nearest-code distance is only needed for the statistics.
  c.probe_bin = sample.margin < 0.0f ? 16
                : env.collect_probe ? std::min(15, env.own->MinHamming(sample.rcode))
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
    const bool read = sample.quad.has_h && ReadBorder(sample.quad.h.h, env.im, env.family->width_at_border, &border);
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
  const Family *family = single_family ? env.family : nullptr;
  // Under grey margin a coarse box or a quad under 1.5 pixels per cell is not rejected: its unrefined sample is too coarse.
  bool large_enough = true;
  if (env.grey_margin && family) {
    const double min_side = geom::MinSide(c.quad.q.p);
    large_enough = !c.input.coarse && min_side >= 1.5 * family->width_at_border;
    // A quad the small rescue would refit when it fails to decode is kept.
    if (env.small_rescue && env.decimation > 1 && !c.input.coarse && c.input.gradient >= 60.0f) {
      const geom::Bounds box = geom::QuadBounds(c.input.p);
      if (box.xhi - box.xlo <= 40.0 && box.yhi - box.ylo <= 40.0) large_enough = false;
    }
  }
  if (!((!env.grey_margin || (env.early_reject && large_enough)) && (check_border || check_dark))) return false;
  const UnrefinedSample &outer = SharedSample(env, c);
  bool reject = SampleCannotDecode(env, outer);
  if (reject && env.grey_margin) {
    reject = false;
    if (outer.quad.has_h) {
      Quad inner;
      InnerQuad(outer.quad.h.h, family->width_at_border, env.reversed_border, &inner);
      reject = SampleCannotDecode(env, UnrefinedSample(env, inner));
    }
  }
  c.clock.Lap(&c.probe_ms);
  return reject;
}

// Refines the quad's corners against the image, if edge refinement is on.
void RefineCandidate(const DecodeEnv &env, Candidate &c) {
  if (!env.refine_edges) return;
  VKA_PROF_SCOPE("tag.refine");
  RefineEdges(env.refine_method, env.decimation, env.width_at_border, env.im, &c.quad.q);
  c.clock.Lap(&c.refine_ms);
}

// Decodes the quad into c.found.
void DecodeCandidate(const DecodeEnv &env, Candidate &c) {
  VKA_PROF_SCOPE("tag.decode");
  DecodeQuad(env, c.quad, c.found, &c.raw);
}

// A grey quiet zone on a bright wall makes the quiet zone's outline the dark blob, so the quad is
// the (W + 2) cell square: if it did not decode, try its inner W cells, once that decodes unrefined.
void TryInnerSquare(const DecodeEnv &env, Candidate &c) {
  if (!(env.grey_margin && env.refine_edges && c.found->empty() && c.quad.has_h && env.family_count > 0)) {
    return;
  }
  VKA_PROF_SCOPE("tag.inner");
  OwnedQuad inner;
  InnerQuad(c.quad.h.h, env.family->width_at_border, env.reversed_border, &inner.q);
  if (DecodeQuad(env, inner, nullptr) == 0) return;
  RefineEdges(env.refine_method, env.decimation, env.width_at_border, env.im, &inner.q);
  DecodeQuad(env, inner, c.found);
}

// Moves a side whose border cells read white one cell in and refits; keeps the result if it decodes
// the same tag with a darker border.
void TrySideOneOut(const DecodeEnv &env, Candidate &c) {
  if (!(env.side_one_out && c.quad.has_h && c.found->size() == 1 && env.family_count > 0)) return;
  const int w = env.family->width_at_border;
  BorderLevels before;
  int bad = 0;
  if (ReadBorder(c.quad.h.h, env.im, w, &before)) {
    for (int k = 0; k < 4; ++k) bad |= (before.side_level[k] > 0.3) << k;
  }
  if (bad == 0 || bad == 15) return;
  VKA_PROF_SCOPE("tag.side");
  const double one = 2.0 / w;  // a cell, in the tag's [-1, 1] coordinates
  const double u0 = (bad & 8) ? -1 + one : -1, u1 = (bad & 2) ? 1 - one : 1;
  const double v0 = (bad & 1) ? -1 + one : -1, v1 = (bad & 4) ? 1 - one : 1;
  const double us[4] = {u0, u1, u1, u0}, vs[4] = {v0, v0, v1, v1};
  OwnedQuad moved;
  QuadFromHomography(c.quad.h.h, us, vs, env.reversed_border, &moved.q);
  RefineEdges(env.refine_method, env.decimation, env.width_at_border, env.im, &moved.q);
  c.probe->clear();
  DecodeQuad(env, moved, c.probe);
  if (c.probe->size() == 1 && moved.has_h) {
    bool keep = false;
    const Detection &old_det = (*c.found)[0];
    const Detection &new_det = (*c.probe)[0];
    BorderLevels after;
    if (new_det.id == old_det.id && ReadBorder(moved.h.h, env.im, w, &after)) {
      double level_before = 0, level_after = 0;
      for (int k = 0; k < 4; ++k) level_before += before.side_level[k], level_after += after.side_level[k];
      keep = level_after < level_before - 0.2;
    }
    if (keep) (*c.found)[0] = new_det;
  }
  c.probe->clear();
}

// A third corrected bit, accepted only when the three wrong cells touch: damage such as a finger
// or glare is coherent, a chance match is scattered.
void TryThirdBit(const DecodeEnv &env, Candidate &c) {
  if (!(env.third_bit && env.refine_edges && c.found->empty() && AllowsThirdBit(env.family))) return;
  VKA_PROF_SCOPE("tag.thirdbit");
  const float margin = c.raw.valid ? c.raw.margin : -1.0f;
  const uint64_t rcode = c.raw.rcode;
  const bool has_h = c.quad.has_h;
  const own::Homography h = c.quad.h;
  int id = 0, rotation = 0;
  if (!(margin >= 0 && has_h && MatchThreeTouchingBits(*env.third_index, rcode, &id, &rotation))) return;
  own::DecodedTag tag;
  own::Decoder::FillTag(env.family, id, 3, rotation, margin, h.h, &tag);
  c.found->push_back(own::Decoder::ToDetection(tag));
}

// A small quad that did not decode: refit it from the full-resolution image and try again.
void TrySmallRescue(const DecodeEnv &env, Candidate &c) {
  const DetectedQuad &in = c.input;
  if (!(env.small_rescue && env.decimation > 1 && !in.coarse && in.gradient >= 60.0f && env.refine_edges &&
        c.found->empty())) {
    return;
  }
  const geom::Bounds box = geom::QuadBounds(in.p);
  double fitted[4][2];
  if (!(box.xhi - box.xlo <= 40.0 && box.yhi - box.ylo <= 40.0 &&
        FitSmallQuad(env.im->data, env.im->width, env.im->height, env.im->stride, box.xlo, box.ylo, box.xhi, box.yhi, fitted))) {
    return;
  }
  VKA_PROF_SCOPE("tag.rescue");
  OwnedQuad refit;
  for (int k = 0; k < 4; ++k) {
    refit.q.p[k][0] = static_cast<float>(fitted[k][0]);
    refit.q.p[k][1] = static_cast<float>(fitted[k][1]);
  }
  refit.q.reversed_border = env.reversed_border;
  RefineEdges(env.refine_method, env.decimation, env.width_at_border, env.im, &refit.q);
  DecodeQuad(env, refit, c.found);
}

// Drops corrections beyond what the family can bear before they count as detections.
void ApplyFamilyLimits(const DecodeEnv &env, Candidate &c) {
  if (!env.family_limits) return;
  for (int j = static_cast<int>(c.found->size()) - 1; j >= 0; --j) {
    const Detection &det = (*c.found)[j];
    // the third-bit match is the deliberate exception to the family limit
    if (det.hamming > MaxCorrectedBits(det.family) && !(env.third_bit && det.hamming == 3)) {
      c.found->erase(c.found->begin() + j);
    }
  }
}

// Refits each detection's homography to every cell edge of its code.
void PolishFound(const DecodeEnv &env, Candidate &c) {
  if (!env.polish) return;
  VKA_PROF_SCOPE("tag.polish");
  for (Detection &det : *c.found) PolishDetection(env.im, &det);
  c.clock.Lap(&c.refine_ms);
}

// How RunSteps ended; only kFinished candidates reach the probe statistics.
enum class Outcome { kDropped, kMarginGate, kPreCheck, kFinished };

Outcome RunSteps(const DecodeEnv &env, Candidate &c) {
  if (!NormaliseCandidate(env, c)) return Outcome::kDropped;

  // Checks on the unrefined quad.
  if (MarginGateRejects(env, c)) return Outcome::kMarginGate;
  if (PreRefineRejects(env, c)) return Outcome::kPreCheck;
  c.unrefined.reset();

  RefineCandidate(env, c);
  DecodeCandidate(env, c);

  // Recoveries for a quad that did not decode, or decoded with a side one cell out.
  TryInnerSquare(env, c);
  TrySideOneOut(env, c);
  TryThirdBit(env, c);
  TrySmallRescue(env, c);

  ApplyFamilyLimits(env, c);
  PolishFound(env, c);
  c.clock.Lap(&c.decode_ms);
  return Outcome::kFinished;
}

}  // namespace

void TagDecoder::ProcessCandidate(const DecodeEnv &env, const DetectedQuad &q, size_t i, unsigned slot) {
  prof::Scope quad_scope("tag.quad");
  Candidate c(env, q, &per_quad_[i], &probe_[slot]);
  const Outcome outcome = RunSteps(env, c);
  const bool decoded = !c.found->empty();
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

namespace {

DecodeEnv MakeDecodeEnv(const TagDecoderConfig &config, int decimation, const Image *im, bool reversed_border, bool timing,
                        RefineEdgesMethod refine_method, const ThirdBitIndex *third_index,
                        const DecodeSwitches &switches, const own::Decoder *own_decoder) {
  DecodeEnv env;
  env.own = own_decoder;
  static_cast<DecodeSwitches &>(env) = switches;
  env.refine_edges = config.refine_edges;
  env.decimation = decimation;
  env.im = im;
  env.reversed_border = reversed_border;
  env.timing = timing;
  env.refine_method = refine_method;
  env.third_index = third_index;
  env.family_count = static_cast<int>(config.families.size());
  if (env.family_count > 0) env.family = config.families[0];
  if (env.family) env.width_at_border = env.family->width_at_border;
  if (switches.dark_reject && env.family_count == 1) {
    env.min_white_cells = env.family->nbits;
    for (uint32_t k = 0; k < env.family->ncodes; ++k) {
      env.min_white_cells = std::min(env.min_white_cells, std::popcount(env.family->codes[k]));
    }
  }
  return env;
}

}  // namespace

DecodeEnv TagDecoder::MakeEnv(const Image *im, bool reversed_border, bool timing) {
  if (third_bit_ && !config_.families.empty()) {
    const Family *family = config_.families[0];
    if (third_index_.family != family) BuildThirdBitIndex(family, &third_index_);
  }
  DecodeSwitches switches;
  switches.grey_margin = grey_margin_;
  switches.small_rescue = small_rescue_;
  switches.polish = polish_;
  switches.side_one_out = side_one_out_;
  switches.third_bit = third_bit_;
  switches.family_limits = family_limits_;
  switches.border_precheck = border_precheck_;
  switches.dark_reject = dark_reject_;
  switches.early_reject = early_reject_;
  switches.collect_probe = collect_probe_;
  switches.probe_margin = probe_margin_;
  if (!own_decoder_->Refresh(config_.families, config_.decode_sharpening)) {
    throw std::invalid_argument("TagDecoder: unsupported tag family");
  }
  return MakeDecodeEnv(config_, decimation_, im, reversed_border, timing, refine_method_, &third_index_, switches, own_decoder_.get());
}

void TagDecoder::MergeSlotRecords() {
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
}

TagDecoder::Stats TagDecoder::SumSlotStats() const {
  Stats stats;
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
  return stats;
}

const DetectionList &TagDecoder::Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                                        uint32_t width, uint32_t height, bool reversed_border) {
  const Image im{
      .data = gray_frame,
      .width = static_cast<int32_t>(width),
      .height = static_cast<int32_t>(height),
      .stride = static_cast<int32_t>(width),
  };
  return Decode(quads, im, reversed_border);
}

const DetectionList &TagDecoder::Decode(const std::vector<DetectedQuad> &quads, const Image &image,
                                        bool reversed_border) {
  detections_.clear();
  const Image im = image;

  // One scratch list per quad, so tasks share no mutable state.
  if (per_quad_.size() < quads.size()) per_quad_.resize(quads.size());
  for (size_t i = 0; i < quads.size(); ++i) per_quad_[i].clear();

  VKA_PROF_SCOPE("tag.decode_all");
  const bool timing = prof::Enabled();
  const prof::Ns wall0 = timing ? prof::NowNs() : 0;
  if (timing) slot_stats_.assign(pool_->threads(), SlotStats{});

  const DecodeEnv env = MakeEnv(&im, reversed_border, timing);
  pool_->ParallelFor(quads.size(), [&](size_t i, unsigned slot) { ProcessCandidate(env, quads[i], i, slot); });

  MergeSlotRecords();

  // Merge in quad order over [0, quads.size()), as in the reset loop above.
  for (size_t i = 0; i < quads.size(); ++i) {
    detections_.insert(detections_.end(), per_quad_[i].begin(), per_quad_[i].end());
  }

  own::Reconcile(&detections_);
  own::SortById(&detections_);

  if (timing) {
    last_stats_ = SumSlotStats();
    last_stats_.wall_ms = NsToMs(prof::NowNs() - wall0);
  }
  return detections_;
}

namespace testing {

uint64_t Rotate90(uint64_t w, int nbits) { return apriltag_vulkan::Rotate90(w, nbits); }
bool ErrorsConnected(const Family *f, uint64_t diff) { return apriltag_vulkan::ErrorsConnected(f, diff); }
void BuildThirdBitIndex(const Family *f, ThirdBitIndex *index) { apriltag_vulkan::BuildThirdBitIndex(f, index); }
bool MatchThreeTouchingBits(const ThirdBitIndex &index, uint64_t rcode, int *id, int *rotation) {
  return apriltag_vulkan::MatchThreeTouchingBits(index, rcode, id, rotation);
}
int MaxCorrectedBits(const Family *f) { return apriltag_vulkan::MaxCorrectedBits(f); }

namespace {

// A candidate over the quad with the given corners, with its own scratch arrays.
struct TestRun {
  TestRun(const TagDecoderConfig &config, int decimation, const Image *im, const double corners[4][2],
          bool reversed_border, const DecodeSwitches &switches) {
    if (!own_decoder.Refresh(config.families, config.decode_sharpening)) {
      throw std::invalid_argument("unsupported tag family");
    }
    if (!config.families.empty()) apriltag_vulkan::BuildThirdBitIndex(config.families[0], &third_index);
    for (int k = 0; k < 4; ++k) {
      input.p[k][0] = corners[k][0];
      input.p[k][1] = corners[k][1];
    }
    env = MakeDecodeEnv(config, decimation, im, reversed_border, false, RefineEdgesMethod::kProfile, &third_index, switches,
                        &own_decoder);
    candidate.emplace(env, input, &found, &probe);
  }

  DetectedQuad input;
  DetectionList found;
  DetectionList probe;
  ThirdBitIndex third_index;
  own::Decoder own_decoder;
  DecodeEnv env;
  std::optional<Candidate> candidate;
};

}  // namespace

bool PreRefineRejects(const TagDecoderConfig &config, int decimation, const Image *im, const double corners[4][2],
                      bool reversed_border, const DecodeSwitches &switches) {
  TestRun run(config, decimation, im, corners, reversed_border, switches);
  return apriltag_vulkan::PreRefineRejects(run.env, *run.candidate);
}

StepResult RunCandidateSteps(const TagDecoderConfig &config, int decimation, const Image *im,
                             const double corners[4][2], bool reversed_border, const DecodeSwitches &switches) {
  TestRun run(config, decimation, im, corners, reversed_border, switches);
  StepResult result;
  result.outcome = static_cast<int>(RunSteps(run.env, *run.candidate));
  result.count = static_cast<int>(run.found.size());
  if (result.count > 0) {
    const Detection &det = run.found[0];
    result.id = det.id;
    result.hamming = det.hamming;
    for (int k = 0; k < 4; ++k) std::memcpy(result.corners[k], det.p[k], sizeof(det.p[k]));
  }
  return result;
}

}  // namespace testing

}  // namespace apriltag_vulkan
