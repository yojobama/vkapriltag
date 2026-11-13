// Differential test of the in-house decode (OwnDecode) against the patched libapriltag: for rendered
// tags, perturbed and random quads and the attempts recorded from the corpus, every result must be
// bit-identical; the reconcile and the detection order too.
// usage: own_decode_test <random|corpus|reconcile|bench> [corpus-dir quads-file]
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "OwnDecode.h"
#include "test_check.h"
#include "upstream/UpstreamAdapter.h"
#include "vkapriltag/Family.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

extern "C" {
#include "common/g2d.h"
#include "common/matd.h"
// Exposed by cmake/patches/apriltag-expose-decode-steps.patch and apriltag-expose-code-probe.patch.
void quad_decode_index(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im, image_u8_t *im_samples,
                       zarray_t *detections);
void reconcile_detections(zarray_t *detections, zarray_t *poly0, zarray_t *poly1);
float quad_sample_code(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im, uint64_t *rcode);
}

namespace own = apriltag_vulkan::own;

namespace {

apriltag_vulkan::Image ToImage(const image_u8_t *im) { return {im->buf, im->width, im->height, im->stride}; }

uint64_t Bits(double v) {
  uint64_t b;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}
uint32_t Bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

struct Counters {
  long quads = 0, decoded = 0, multi = 0, degenerate = 0, rejected_polarity = 0, samples_compared = 0;
  long mismatches = 0, corrected = 0;
};

const char *const kFamilies[] = {"tag36h11",         "tag25h9",          "tag16h5",         "tagCircle21h7",
                                 "tagCircle49h12",   "tagStandard41h12", "tagStandard52h13", "tagCustom48h12"};

// The most wrong bits the table is built for: large tables stay at what fits in memory.
int MaxBitsFor(const apriltag_family_t *f, int wanted) {
  if (f->nbits >= 41) return std::min(wanted, 1);
  if (f->nbits >= 25) return std::min(wanted, 2);
  return wanted;
}

// An upstream detector and an own decoder with the same families (each with its own polarity and
// corrected-bit limit).
struct Rig {
  struct Entry {
    const char *name;
    bool reversed;
    int bits;
  };
  explicit Rig(const std::vector<Entry> &entries) : td(apriltag_detector_create()) {
    own_families.reserve(entries.size());
    for (const Entry &e : entries) {
      apriltag_family_t *f = upstream::CreateFamily(e.name);
      f->reversed_border = e.reversed;
      const int bits = MaxBitsFor(f, e.bits);
      apriltag_detector_add_family_bits(td, f, bits);
      families.push_back({f, e.name});
      apriltag_vulkan::Family own_family = *apriltag_vulkan::Family::ByName(e.name);
      own_family.reversed_border = e.reversed;
      own_family.max_hamming = bits;
      own_families.push_back(own_family);
      own_pointers.push_back(&own_families.back());
    }
    td->decode_sharpening = 0.25;
    if (!decoder.Refresh(own_pointers, 0.25)) {
      std::printf("FAIL: own decoder does not support the families\n");
      ++testcheck::Failures();
    }
    // Lets a ctest entry prove that the comparison can fail.
    if (std::getenv("OWN_DECODE_TEST_SABOTAGE")) td->decode_sharpening = 0.3;
  }
  ~Rig() {
    apriltag_detector_destroy(td);
    for (auto &f : families) upstream::DestroyFamily(f.first, f.second);
  }
  Rig(const Rig &) = delete;
  Rig &operator=(const Rig &) = delete;

  // The position in the family list of an own or an upstream family.
  int OwnIndex(const apriltag_vulkan::Family *f) const {
    for (size_t i = 0; i < own_pointers.size(); ++i) {
      if (own_pointers[i] == f) return static_cast<int>(i);
    }
    return -1;
  }
  int UpstreamIndex(const apriltag_family_t *f) const {
    for (size_t i = 0; i < families.size(); ++i) {
      if (families[i].first == f) return static_cast<int>(i);
    }
    return -1;
  }

  apriltag_detector_t *td;
  std::vector<std::pair<apriltag_family_t *, const char *>> families;
  std::vector<apriltag_vulkan::Family> own_families;
  std::vector<const apriltag_vulkan::Family *> own_pointers;
  own::Decoder decoder;
};

bool SameHomography(const double *a, const double *b) {
  for (int i = 0; i < 9; ++i) {
    if (Bits(a[i]) != Bits(b[i])) return false;
  }
  return true;
}

// Compares upstream's and the own decode of one quad (and their raw samples); counts mismatches.
void CompareQuad(Rig &rig, image_u8_t *im, const float p[4][2], bool reversed, Counters *c, const char *what) {
  ++c->quads;
  const apriltag_vulkan::Image oim = ToImage(im);
  struct quad up = {};
  std::memcpy(up.p, p, sizeof(up.p));
  up.reversed_border = reversed;
  zarray_t *dets = zarray_create(sizeof(apriltag_detection_t *));
  quad_decode_index(rig.td, &up, im, nullptr, dets);

  own::Homography h = {};
  own::RawSample first;
  own::DecodedTag tags[own::Decoder::kMaxFamilies];
  const int n = rig.decoder.Decode(p, reversed, &oim, &h, &first, tags);

  bool ok = true;
  const char *why = "";
  const bool up_h = up.H != nullptr;
  if ((n >= 0) != up_h) ok = false, why = "homography validity";
  if (ok && up_h && !SameHomography(h.h, up.H->data)) ok = false, why = "homography bits";
  if (n < 0) ++c->degenerate;
  const int up_n = zarray_size(dets);
  if (ok && std::max(n, 0) != up_n) ok = false, why = "detection count";
  for (int i = 0; ok && i < up_n; ++i) {
    apriltag_detection_t *d;
    zarray_get(dets, i, &d);
    const own::DecodedTag &t = tags[i];
    if (rig.UpstreamIndex(d->family) != rig.OwnIndex(t.family) || d->id != t.id) ok = false, why = "id";
    else if (d->hamming != t.hamming) ok = false, why = "hamming";
    else if (Bits(d->decision_margin) != Bits(t.margin)) ok = false, why = "margin";
    else if (!SameHomography(d->H->data, t.h)) ok = false, why = "detection homography";
    else if (Bits(d->c[0]) != Bits(t.c[0]) || Bits(d->c[1]) != Bits(t.c[1])) ok = false, why = "centre";
    for (int k = 0; ok && k < 4; ++k) {
      if (Bits(d->p[k][0]) != Bits(t.p[k][0]) || Bits(d->p[k][1]) != Bits(t.p[k][1])) ok = false, why = "corner";
    }
    if (!ok) std::printf("  detail: upstream id %d ham %d margin %.9g, own id %d ham %d margin %.9g\n", d->id, d->hamming, d->decision_margin, t.id, t.hamming, t.margin);
  }
  for (int i = 0; i < up_n; ++i) {
    apriltag_detection_t *d;
    zarray_get(dets, i, &d);
    if (d->hamming > 0) ++c->corrected;
  }
  if (up_n > 0) ++c->decoded;
  if (up_n > 1) ++c->multi;

  // The raw sample of the first family of the polarity, as quad_sample_code reads it.
  struct quad probe = {};
  std::memcpy(probe.p, p, sizeof(probe.p));
  probe.reversed_border = reversed;
  uint64_t up_code = 0;
  const float up_margin = quad_sample_code(rig.td, &probe, im, &up_code);
  own::Homography sh = {};
  own::RawSample raw;
  const bool sample_h = rig.decoder.SampleCode(p, reversed, &oim, &sh, &raw);
  ++c->samples_compared;
  const float own_margin = raw.valid ? raw.margin : -1.0f;
  if (sample_h != (probe.H != nullptr)) ok = false, why = "sample homography validity";
  if (ok && sample_h && !SameHomography(sh.h, probe.H->data)) ok = false, why = "sample homography bits";
  if (ok && sample_h && Bits(own_margin) != Bits(up_margin)) {
    ok = false, why = "sample margin";
    std::printf("  detail: upstream margin %.9g, own %.9g\n", up_margin, own_margin);
  }
  if (ok && sample_h && raw.rcode != up_code) {
    ok = false, why = "sample code";
    std::printf("  detail: upstream rcode %llx, own %llx\n", static_cast<unsigned long long>(up_code), static_cast<unsigned long long>(raw.rcode));
  }
  // the first family's sample inside the decode is the same read
  if (ok && n >= 0 && sample_h && !(first.valid == raw.valid && Bits(first.margin) == Bits(raw.margin) && first.rcode == raw.rcode)) ok = false, why = "decode sample";
  if (sample_h && !raw.valid) ++c->rejected_polarity;

  if (!ok) {
    ++c->mismatches;
    if (c->mismatches <= 10) {
      std::printf("MISMATCH (%s: %s) corners %.9g %.9g  %.9g %.9g  %.9g %.9g  %.9g %.9g rev=%d: upstream n=%d own n=%d\n", what,
                  why, p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1], reversed, up_n, n);
    }
  }
  if (up.H) matd_destroy(up.H);
  if (probe.H) matd_destroy(probe.H);
  for (int i = 0; i < zarray_size(dets); ++i) {
    apriltag_detection_t *d;
    zarray_get(dets, i, &d);
    apriltag_detection_destroy(d);
  }
  zarray_destroy(dets);
}

// ---------------------------------------------------------------- scene rendering

constexpr int kSize = 256;

struct Scene {
  std::vector<uint8_t> pix;
  image_u8_t im;
  Scene() : pix(size_t(kSize) * kSize, 128), im{kSize, kSize, kSize, nullptr} { im.buf = pix.data(); }
  Scene(const Scene &) = delete;
};

bool Invert3(const double *h, double *inv) {
  const double a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7], i = h[8];
  const double det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
  if (std::fabs(det) < 1e-18) return false;
  inv[0] = (e * i - f * hh) / det;
  inv[1] = (c * hh - b * i) / det;
  inv[2] = (b * f - c * e) / det;
  inv[3] = (f * g - d * i) / det;
  inv[4] = (a * i - c * g) / det;
  inv[5] = (c * d - a * f) / det;
  inv[6] = (d * hh - e * g) / det;
  inv[7] = (b * g - a * hh) / det;
  inv[8] = (a * e - b * d) / det;
  return true;
}

// Draws the tag whose border square maps through h (tag coordinates [-1, 1] to pixels) onto the scene.
void DrawTag(Scene *scene, apriltag_family_t *f, int id, const double *h, double black, double white, double shade,
             double sigma, double noise, std::mt19937_64 &rng, bool invert, bool keep_background) {
  image_u8_t *tag = apriltag_to_image(f, static_cast<uint32_t>(id));
  const int tw = f->total_width, w = f->width_at_border, start = (tw - w) / 2;
  double inv[9];
  if (!Invert3(h, inv)) {
    image_u8_destroy(tag);
    return;
  }
  std::vector<float> img(size_t(kSize) * kSize);
  std::normal_distribution<double> gauss(0, 1);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      double acc = 0;
      int count = 0;
      for (int sy = 0; sy < 3; ++sy) {
        for (int sx = 0; sx < 3; ++sx) {
          double u, v;
          own::Project(inv, x + (sx + 0.5) / 3, y + (sy + 0.5) / 3, &u, &v);
          const int cu = static_cast<int>(std::floor((u + 1) / 2 * w + start));
          const int cv = static_cast<int>(std::floor((v + 1) / 2 * w + start));
          if (cu < 0 || cv < 0 || cu >= tw || cv >= tw) {
            if (keep_background) continue;
            acc += white;
          } else {
            int val = tag->buf[cv * tag->stride + cu];
            if (invert) val = 255 - val;
            acc += black + (white - black) * val / 255.0;
          }
          ++count;
        }
      }
      img[size_t(y) * kSize + x] = count ? float(acc / count) : float(scene->pix[size_t(y) * kSize + x]);
    }
  }
  image_u8_destroy(tag);
  if (sigma > 0) {
    const int r = int(std::ceil(3 * sigma));
    std::vector<float> k(2 * r + 1), tmp(img.size());
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = float(std::exp(-0.5 * i * i / (sigma * sigma)));
    for (float &t : k) t /= sum;
    for (int pass = 0; pass < 2; ++pass) {
      for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
          float a = 0;
          for (int i = -r; i <= r; ++i) {
            const int xx = pass == 0 ? std::clamp(x + i, 0, kSize - 1) : x, yy = pass == 1 ? std::clamp(y + i, 0, kSize - 1) : y;
            a += k[i + r] * img[size_t(yy) * kSize + xx];
          }
          tmp[size_t(y) * kSize + x] = a;
        }
      }
      img.swap(tmp);
    }
  }
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      double v = img[size_t(y) * kSize + x] * (1.0 + shade * (x / double(kSize) - 0.5));
      if (noise > 0) v += noise * gauss(rng);
      scene->pix[size_t(y) * kSize + x] = uint8_t(std::clamp(int(std::lround(v)), 0, 255));
    }
  }
}

void BorderCorners(const own::Homography &h, float p[4][2]) {
  const double c[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int k = 0; k < 4; ++k) {
    double x, y;
    own::Project(h.h, c[k][0], c[k][1], &x, &y);
    p[k][0] = static_cast<float>(x);
    p[k][1] = static_cast<float>(y);
  }
}

// A random pose: a square of the given side, turned, with its corners moved by up to `jitter` of the side.
void RandomCorners(std::mt19937_64 &rng, double side, double jitter, float p[4][2]) {
  std::uniform_real_distribution<double> u(0, 1);
  const double cx = u(rng) * kSize, cy = u(rng) * kSize, a = u(rng) * 6.283185307179586;
  const double base[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int k = 0; k < 4; ++k) {
    const double x = base[k][0] * side / 2, y = base[k][1] * side / 2;
    p[k][0] = float(cx + x * std::cos(a) - y * std::sin(a) + (u(rng) - 0.5) * 2 * jitter * side);
    p[k][1] = float(cy + x * std::sin(a) + y * std::cos(a) + (u(rng) - 0.5) * 2 * jitter * side);
  }
}

void TestRandom(const char *only_family) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> gauss(0, 1);
  Counters total;
  const struct {
    bool reversed;
    int bits;
  } variants[] = {{false, 2}, {true, 1}, {false, 0}, {true, 3}};
  for (const char *name : kFamilies) {
    if (only_family && std::strcmp(only_family, name)) continue;
    Counters fam;
    for (const auto &v : variants) {
      // a one-family rig, and a two-family rig of the same polarity beside tag16h5
      for (int pair = 0; pair < 2; ++pair) {
        std::vector<Rig::Entry> entries = {{name, v.reversed, v.bits}};
        if (pair == 1) {
          if (!std::strcmp(name, "tag16h5")) continue;
          entries.push_back({"tag16h5", v.reversed, 1});
        }
        Rig rig(entries);
        apriltag_family_t *f = rig.families[0].first;
        for (int s = 0; s < 40; ++s) {
          Scene scene;
          // a tag, possibly partly outside the frame, of the polarity under test
          const double side = std::exp(std::log(14.0) + u(rng) * (std::log(150.0) - std::log(14.0)));
          float corners[4][2];
          RandomCorners(rng, side, s % 3 == 0 ? 0.0 : 0.12, corners);
          own::Homography hq;
          if (!own::ComputeHomography(corners, &hq)) continue;
          const double black = u(rng) * 70, white = 150 + u(rng) * 105;
          const int id = int(rng() % f->ncodes);
          DrawTag(&scene, f, id, hq.h, black, white, (u(rng) - 0.5) * 0.8, (s % 4 == 0) ? 0 : u(rng) * 2.0,
                  (s % 5 == 0) ? 0 : u(rng) * 12, rng, false, false);
          float p[4][2];
          BorderCorners(hq, p);
          CompareQuad(rig, &scene.im, p, v.reversed, &fam, "exact");
          const double sigmas[] = {0.2, 0.6, 1.5, 3.0};
          for (double sg : sigmas) {
            float q[4][2];
            for (int k = 0; k < 4; ++k) {
              q[k][0] = float(p[k][0] + sg * gauss(rng));
              q[k][1] = float(p[k][1] + sg * gauss(rng));
            }
            CompareQuad(rig, &scene.im, q, v.reversed, &fam, "perturbed");
          }
          float rolled[4][2];
          for (int k = 0; k < 4; ++k) std::memcpy(rolled[k], p[(k + 1) % 4], sizeof(rolled[k]));
          CompareQuad(rig, &scene.im, rolled, v.reversed, &fam, "rolled");
          CompareQuad(rig, &scene.im, p, !v.reversed, &fam, "other polarity");
          float junk[4][2];
          for (int k = 0; k < 4; ++k) {
            junk[k][0] = float(u(rng) * (kSize + 40) - 20);
            junk[k][1] = float(u(rng) * (kSize + 40) - 20);
          }
          CompareQuad(rig, &scene.im, junk, v.reversed, &fam, "random");
          // a quad of the right size and place on a scene of noise
          Scene noise;
          for (uint8_t &b : noise.pix) b = uint8_t(rng());
          CompareQuad(rig, &noise.im, p, v.reversed, &fam, "noise");
        }
        // degenerate and extreme quads
        Scene blank;
        const float degenerate[][4][2] = {
            {{50, 50}, {50, 50}, {50, 50}, {50, 50}},
            {{10, 10}, {100, 10}, {100, 10}, {10, 100}},
            {{10, 10}, {50, 50}, {90, 90}, {130, 130}},
            {{10, 10}, {100, 10}, {100, 100}, {10, 100}},
            {{-100000, -100000}, {100000, -100000}, {100000, 100000}, {-100000, 100000}},
            {{1e20f, 1e20f}, {1e20f, 2e20f}, {2e20f, 2e20f}, {2e20f, 1e20f}},
            {{0, 0}, {255, 0}, {255, 255}, {0, 255}},
            {{-3, -3}, {300, -3}, {300, 300}, {-3, 300}},
        };
        for (const auto &d : degenerate) {
          CompareQuad(rig, &blank.im, d, v.reversed, &fam, "degenerate");
          CompareQuad(rig, &blank.im, d, !v.reversed, &fam, "degenerate");
        }
      }
    }
    std::printf("  %-18s quads %6ld  decoded %5ld (corrected %ld, two families %ld)  degenerate %ld  polarity-rejected %ld  mismatches %ld\n",
                name, fam.quads, fam.decoded, fam.corrected, fam.multi, fam.degenerate, fam.rejected_polarity, fam.mismatches);
    total.quads += fam.quads;
    total.decoded += fam.decoded;
    total.mismatches += fam.mismatches;
    total.degenerate += fam.degenerate;
  }
  std::printf("random: %ld quads, %ld decoded, %ld degenerate, %ld mismatches\n", total.quads, total.decoded,
              total.degenerate, total.mismatches);
  CHECK(total.mismatches == 0);
  CHECK(total.decoded > 1000);
}

// ---------------------------------------------------------------- recorded corpus attempts

struct RecordedQuad {
  float p[4][2];
  bool reversed;
};
struct RecordedFrame {
  std::string image;
  std::vector<RecordedQuad> quads;
};
struct RecordedSet {
  std::string family;
  std::vector<RecordedFrame> frames;
};

std::vector<std::string> ReadLines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) lines.push_back(line);
  }
  return lines;
}

bool LoadRecorded(const std::string &corpus, const std::string &quads_file, std::vector<RecordedSet> *sets) {
  std::vector<std::string> lines = ReadLines(quads_file);
  if (lines.empty()) return false;
  std::vector<std::string> images;
  for (const std::string &line : lines) {
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "S") {
      std::string scenario, family;
      is >> scenario >> family;
      sets->push_back({family, {}});
      images = ReadLines(corpus + "/" + scenario + "/list.txt");
      for (std::string &im : images) im = corpus + "/" + scenario + "/" + im;
    } else if (tag == "D") {
      const size_t k = sets->back().frames.size();
      sets->back().frames.push_back({k < images.size() ? images[k] : std::string(), {}});
    } else if (tag == "Q") {
      RecordedQuad q;
      int rev;
      is >> rev;
      q.reversed = rev != 0;
      for (int k = 0; k < 4; ++k) is >> q.p[k][0] >> q.p[k][1];
      sets->back().frames.back().quads.push_back(q);
    }
  }
  return true;
}

void TestCorpus(const std::string &corpus, const std::string &quads_file) {
  std::vector<RecordedSet> sets;
  CHECK(LoadRecorded(corpus, quads_file, &sets));
  Counters total;
  for (const RecordedSet &set : sets) {
    Rig rig({{set.family.c_str(), false, 2}});
    for (const RecordedFrame &frame : set.frames) {
      int w, h, ch;
      uint8_t *pixels = frame.image.empty() ? nullptr : stbi_load(frame.image.c_str(), &w, &h, &ch, 1);
      if (!pixels) {
        std::printf("FAIL: cannot load %s\n", frame.image.c_str());
        ++testcheck::Failures();
        continue;
      }
      image_u8_t im{w, h, w, pixels};
      for (const RecordedQuad &q : frame.quads) CompareQuad(rig, &im, q.p, q.reversed, &total, "corpus");
      stbi_image_free(pixels);
    }
  }
  std::printf("corpus: %ld recorded attempts, %ld decoded, %ld degenerate, %ld mismatches\n", total.quads, total.decoded,
              total.degenerate, total.mismatches);
  CHECK(total.mismatches == 0);
  CHECK(total.quads > 1000);
}

// ---------------------------------------------------------------- reconcile

apriltag_detection_t *MakeDetection(apriltag_family_t *f, int id, int hamming, float margin, const double p[4][2]) {
  apriltag_detection_t *d = static_cast<apriltag_detection_t *>(std::calloc(1, sizeof(apriltag_detection_t)));
  d->family = f;
  d->id = id;
  d->hamming = hamming;
  d->decision_margin = margin;
  d->H = matd_create(3, 3);
  std::memcpy(d->p, p, sizeof(d->p));
  return d;
}

apriltag_vulkan::Detection MakeOwnDetection(const apriltag_vulkan::Family *f, int id, int hamming, float margin,
                                             const double p[4][2]) {
  apriltag_vulkan::Detection d;
  d.family = f;
  d.id = id;
  d.hamming = hamming;
  d.decision_margin = margin;
  std::memcpy(d.p, p, sizeof(d.p));
  return d;
}

void TestReconcile() {
  Rig rig({{"tag36h11", false, 2}, {"tag25h9", false, 2}});
  std::mt19937_64 rng(777);
  std::uniform_real_distribution<double> u(0, 1);
  long sets = 0, removed = 0, mismatches = 0, equal_order = 0;
  zarray_t *poly0 = g2d_polygon_create_zeros(4);
  zarray_t *poly1 = g2d_polygon_create_zeros(4);
  for (int iter = 0; iter < 30000; ++iter) {
    const int n = 2 + int(rng() % 14);
    std::vector<std::array<double, 8>> shapes;
    zarray_t *a = zarray_create(sizeof(apriltag_detection_t *));
    apriltag_vulkan::DetectionList b;
    const double span = 30 + u(rng) * 60;
    for (int i = 0; i < n; ++i) {
      double p[4][2];
      if (!shapes.empty() && u(rng) < 0.2) {
        std::memcpy(p, shapes[rng() % shapes.size()].data(), sizeof(p));  // an identical shape
      } else {
        const double cx = u(rng) * span, cy = u(rng) * span, side = 6 + u(rng) * 40, ang = u(rng) * 6.283185307179586;
        const double base[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (int k = 0; k < 4; ++k) {
          const double x = base[k][0] * side / 2, y = base[k][1] * side / 2;
          p[k][0] = cx + x * std::cos(ang) - y * std::sin(ang) + (u(rng) - 0.5) * side * 0.3;
          p[k][1] = cy + x * std::sin(ang) + y * std::cos(ang) + (u(rng) - 0.5) * side * 0.3;
        }
        if (u(rng) < 0.3) {  // quantise so that ties in the corner comparison occur
          for (auto &pt : p) pt[0] = std::floor(pt[0]), pt[1] = std::floor(pt[1]);
        }
      }
      std::array<double, 8> sh;
      std::memcpy(sh.data(), p, sizeof(p));
      shapes.push_back(sh);
      const size_t which_family = rng() % 2;
      apriltag_family_t *f = rig.families[which_family].first;
      const int id = int(rng() % 3);
      const int hamming = int(rng() % 3);
      const float margin = u(rng) < 0.5 ? float(int(rng() % 4) * 10) : float(u(rng) * 40);
      apriltag_detection_t *d1 = MakeDetection(f, id, hamming, margin, p);
      zarray_add(a, &d1);
      b.push_back(MakeOwnDetection(rig.own_pointers[which_family], id, hamming, margin, p));
    }
    reconcile_detections(a, poly0, poly1);
    own::Reconcile(&b);
    ++sets;
    removed += n - zarray_size(a);
    bool same = zarray_size(a) == static_cast<int>(b.size());
    for (int i = 0; same && i < zarray_size(a); ++i) {
      apriltag_detection_t *x;
      zarray_get(a, i, &x);
      const apriltag_vulkan::Detection &y = b[i];
      same = x->id == y.id && rig.UpstreamIndex(x->family) == rig.OwnIndex(y.family) && x->hamming == y.hamming &&
             Bits(x->decision_margin) == Bits(y.decision_margin) && std::memcmp(x->p, y.p, sizeof(x->p)) == 0;
    }
    if (same) {
      qsort(a->data, a->size, a->el_sz, [](const void *l, const void *r) {
        return (*static_cast<apriltag_detection_t *const *>(l))->id - (*static_cast<apriltag_detection_t *const *>(r))->id;
      });
      own::SortById(&b);
      for (int i = 0; same && i < zarray_size(a); ++i) {
        apriltag_detection_t *x;
        zarray_get(a, i, &x);
        const apriltag_vulkan::Detection &y = b[i];
        same = std::memcmp(x->p, y.p, sizeof(x->p)) == 0 && x->id == y.id;
      }
      equal_order += same;
    }
    if (!same) {
      ++mismatches;
      if (mismatches <= 5) std::printf("MISMATCH reconcile set %d\n", iter);
    }
    for (int i = 0; i < zarray_size(a); ++i) {
      apriltag_detection_t *d;
      zarray_get(a, i, &d);
      apriltag_detection_destroy(d);
    }
    zarray_destroy(a);
  }
  zarray_destroy(poly0);
  zarray_destroy(poly1);
  std::printf("reconcile: %ld sets, %ld detections removed, %ld mismatches\n", sets, removed, mismatches);
  CHECK(mismatches == 0);
  CHECK(removed > 5000);
}

// ---------------------------------------------------------------- speed of the decode attempts

double Seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Bench(const std::string &corpus, const std::string &quads_file) {
  std::vector<RecordedSet> sets;
  if (!LoadRecorded(corpus, quads_file, &sets)) {
    std::printf("no recorded attempts\n");
    return;
  }
  struct Frame {
    int w, h;
    uint8_t *pixels;
    std::vector<RecordedQuad> quads;
    size_t set;
  };
  std::vector<Frame> frames;
  for (size_t s = 0; s < sets.size(); ++s) {
    if (sets[s].family != "tag36h11") continue;
    for (const RecordedFrame &f : sets[s].frames) {
      int w, h, ch;
      uint8_t *px = stbi_load(f.image.c_str(), &w, &h, &ch, 1);
      if (px) frames.push_back({w, h, px, f.quads, s});
    }
  }
  Rig rig({{"tag36h11", false, 2}});
  long attempts = 0;
  for (const Frame &f : frames) attempts += static_cast<long>(f.quads.size());
  std::printf("bench: %ld recorded tag36h11 attempts over %zu frames\n", attempts, frames.size());
  for (int round = 0; round < 5; ++round) {
    long sink = 0;
    double t0 = Seconds();
    for (const Frame &f : frames) {
      image_u8_t im{f.w, f.h, f.w, f.pixels};
      for (const RecordedQuad &q : f.quads) {
        struct quad up = {};
        std::memcpy(up.p, q.p, sizeof(up.p));
        up.reversed_border = q.reversed;
        zarray_t *dets = zarray_create(sizeof(apriltag_detection_t *));
        quad_decode_index(rig.td, &up, &im, nullptr, dets);
        sink += zarray_size(dets);
        for (int i = 0; i < zarray_size(dets); ++i) {
          apriltag_detection_t *d;
          zarray_get(dets, i, &d);
          apriltag_detection_destroy(d);
        }
        zarray_destroy(dets);
        if (up.H) matd_destroy(up.H);
      }
    }
    const double t_up = Seconds() - t0;
    t0 = Seconds();
    for (const Frame &f : frames) {
      const apriltag_vulkan::Image oim{f.pixels, f.w, f.h, f.w};
      for (const RecordedQuad &q : f.quads) {
        own::DecodedTag tags[own::Decoder::kMaxFamilies];
        own::Homography h;
        const int n = rig.decoder.Decode(q.p, q.reversed, &oim, &h, nullptr, tags);
        sink += std::max(n, 0);
      }
    }
    const double t_own = Seconds() - t0;
    t0 = Seconds();
    for (const Frame &f : frames) {
      for (const RecordedQuad &q : f.quads) {
        own::Homography h;
        sink += own::ComputeHomography(q.p, &h);
      }
    }
    const double t_hom = Seconds() - t0;
    t0 = Seconds();
    for (const Frame &f : frames) {
      const apriltag_vulkan::Image oim{f.pixels, f.w, f.h, f.w};
      for (const RecordedQuad &q : f.quads) {
        own::Homography h;
        own::RawSample raw;
        sink += rig.decoder.SampleCode(q.p, q.reversed, &oim, &h, &raw);
      }
    }
    const double t_sample = Seconds() - t0;
    std::printf("    own: homography %.0f ns, + sampling %.0f ns, + lookup and detection %.0f ns per attempt\n",
                1e9 * t_hom / attempts, 1e9 * t_sample / attempts, 1e9 * t_own / attempts);
    std::printf("  round %d: upstream %.0f attempts/s, own %.0f attempts/s (x%.2f)  [%ld]\n", round, attempts / t_up,
                attempts / t_own, t_up / t_own, sink);
  }
  for (Frame &f : frames) stbi_image_free(f.pixels);
}

// ---------------------------------------------------------------- family tables

// Every built-in family equals the table libapriltag builds for the same name.
void TestFamilies() {
  long checked = 0;
  for (const char *name : kFamilies) {
    apriltag_family_t *up = upstream::CreateFamily(name);
    const apriltag_vulkan::Family *own_family = apriltag_vulkan::Family::ByName(name);
    CHECK(up != nullptr && own_family != nullptr);
    if (!up || !own_family) continue;
    bool same = std::strcmp(own_family->name, up->name) == 0 && own_family->ncodes == up->ncodes &&
                own_family->nbits == up->nbits && own_family->width_at_border == up->width_at_border &&
                own_family->total_width == up->total_width && own_family->reversed_border == up->reversed_border &&
                own_family->h == up->h && own_family->max_hamming == 2;
    for (uint32_t i = 0; same && i < up->ncodes; ++i) same = own_family->codes[i] == up->codes[i];
    for (uint32_t i = 0; same && i < up->nbits; ++i) {
      same = own_family->bit_x[i] == up->bit_x[i] && own_family->bit_y[i] == up->bit_y[i];
    }
    if (!same) std::printf("MISMATCH family %s\n", name);
    CHECK(same);
    checked += up->ncodes;
    upstream::DestroyFamily(up, name);
  }
  CHECK(apriltag_vulkan::Family::ByName("tag99h99") == nullptr);
  CHECK(apriltag_vulkan::Family::All().size() == std::size(kFamilies));
  std::printf("families: %zu families, %ld codes compared\n", std::size(kFamilies), checked);
}

}  // namespace

int main(int argc, char **argv) {
  const std::string which = argc > 1 ? argv[1] : "all";
  const std::string corpus = argc > 2 ? argv[2] : "";
  const std::string quads = argc > 3 ? argv[3] : "";
  bool ran = false;
  if (which == "all" || which == "random") TestRandom(argc > 4 ? argv[4] : nullptr), ran = true;
  if (which == "all" || which == "reconcile") TestReconcile(), ran = true;
  if (which == "all" || which == "families") TestFamilies(), ran = true;
  if ((which == "all" || which == "corpus") && !corpus.empty()) TestCorpus(corpus, quads), ran = true;
  if (which == "bench") Bench(corpus, quads), ran = true;
  if (!ran) {
    std::printf("usage: own_decode_test <random|reconcile|families|corpus|bench> [corpus-dir quads-file]\n");
    return 2;
  }
  if (testcheck::Failures() != 0) {
    std::printf("%d check(s) failed\n", testcheck::Failures());
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
