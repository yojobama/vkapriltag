// Differential test of the in-house decode (OwnDecode) against what upstream libapriltag produced for the same
// quads, recorded in tests/own_decode_vectors.bin by tools/upstream/gen_decode_vectors.cpp: every result must be
// bit-identical (ids, Hamming distance, margins, homographies, centres, corners, raw samples, reconcile and order).
// usage: own_decode_test <random|reconcile|families|corpus|bench|all> <vectors.bin> [corpus-dir [family]]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DecodeVectors.h"
#include "OwnDecode.h"
#include "test_check.h"
#include "vkapriltag/Family.h"

namespace own = apriltag_vulkan::own;
namespace dv = decode_vectors;

namespace {

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

bool SameDoubles(const double *a, const double *b, int n) {
  for (int i = 0; i < n; ++i) {
    if (Bits(a[i]) != Bits(b[i])) return false;
  }
  return true;
}

struct Counters {
  long quads = 0, decoded = 0, multi = 0, degenerate = 0, rejected_polarity = 0, mismatches = 0, corrected = 0;
};

// An own decoder with the families of a recorded rig.
struct Rig {
  Rig(const std::vector<dv::RigEntry> &entries, double sharpening) {
    own_families.reserve(entries.size());
    for (const dv::RigEntry &e : entries) {
      const apriltag_vulkan::Family *base = apriltag_vulkan::Family::ByName(e.family);
      if (base == nullptr) {
        std::printf("FAIL: unknown family %s\n", e.family.c_str());
        ++testcheck::Failures();
        continue;
      }
      apriltag_vulkan::Family f = *base;
      f.reversed_border = e.reversed;
      f.max_hamming = e.bits;
      own_families.push_back(f);
      own_pointers.push_back(&own_families.back());
    }
    if (!decoder.Refresh(own_pointers, sharpening)) {
      std::printf("FAIL: own decoder does not support the families\n");
      ++testcheck::Failures();
    }
  }
  Rig(const Rig &) = delete;
  Rig &operator=(const Rig &) = delete;

  int Index(const apriltag_vulkan::Family *f) const {
    for (size_t i = 0; i < own_pointers.size(); ++i) {
      if (own_pointers[i] == f) return static_cast<int>(i);
    }
    return -1;
  }

  std::vector<apriltag_vulkan::Family> own_families;
  std::vector<const apriltag_vulkan::Family *> own_pointers;
  own::Decoder decoder;
};

// Decodes the recorded quad with the own decoder and compares everything with the recording.
void CompareCase(const Rig &rig, const dv::Case &c, const apriltag_vulkan::Image &oim, Counters *counters) {
  ++counters->quads;
  const dv::Expect &e = c.expect;
  own::Homography h = {};
  own::RawSample first;
  own::DecodedTag tags[own::Decoder::kMaxFamilies];
  const int n = rig.decoder.Decode(c.p, c.reversed, &oim, &h, &first, tags);

  bool ok = true;
  const char *why = "";
  if ((n >= 0) != e.h_valid) ok = false, why = "homography validity";
  if (ok && e.h_valid && !SameDoubles(h.h, e.h, 9)) ok = false, why = "homography bits";
  if (n < 0) ++counters->degenerate;
  const int up_n = static_cast<int>(e.dets.size());
  if (ok && std::max(n, 0) != up_n) ok = false, why = "detection count";
  for (int i = 0; ok && i < up_n; ++i) {
    const dv::Det &d = e.dets[i];
    const own::DecodedTag &t = tags[i];
    if (d.family != rig.Index(t.family) || d.id != t.id) ok = false, why = "id";
    else if (d.hamming != t.hamming) ok = false, why = "hamming";
    else if (Bits(d.margin) != Bits(t.margin)) ok = false, why = "margin";
    else if (!SameDoubles(d.h, t.h, 9)) ok = false, why = "detection homography";
    else if (!SameDoubles(d.c, t.c, 2)) ok = false, why = "centre";
    else if (!SameDoubles(&d.p[0][0], &t.p[0][0], 8)) ok = false, why = "corner";
    if (!ok) std::printf("  detail: recorded id %d ham %d margin %.9g, own id %d ham %d margin %.9g\n", d.id, d.hamming, d.margin, t.id, t.hamming, t.margin);
  }
  for (const dv::Det &d : e.dets) counters->corrected += d.hamming > 0;
  if (up_n > 0) ++counters->decoded;
  if (up_n > 1) ++counters->multi;

  // the raw sample of the first family of the polarity
  own::Homography sh = {};
  own::RawSample raw;
  const bool sample_h = rig.decoder.SampleCode(c.p, c.reversed, &oim, &sh, &raw);
  const float own_margin = raw.valid ? raw.margin : -1.0f;
  if (sample_h != e.s_valid) ok = false, why = "sample homography validity";
  if (ok && sample_h && !SameDoubles(sh.h, e.sh, 9)) ok = false, why = "sample homography bits";
  if (ok && sample_h && Bits(own_margin) != Bits(e.s_margin)) {
    ok = false, why = "sample margin";
    std::printf("  detail: recorded margin %.9g, own %.9g\n", e.s_margin, own_margin);
  }
  if (ok && sample_h && raw.rcode != e.s_code) {
    ok = false, why = "sample code";
    std::printf("  detail: recorded rcode %llx, own %llx\n", static_cast<unsigned long long>(e.s_code), static_cast<unsigned long long>(raw.rcode));
  }
  // the first family's sample inside the decode is the same read
  if (ok && n >= 0 && sample_h && !(first.valid == raw.valid && Bits(first.margin) == Bits(raw.margin) && first.rcode == raw.rcode)) ok = false, why = "decode sample";
  if (sample_h && !raw.valid) ++counters->rejected_polarity;

  if (!ok) {
    ++counters->mismatches;
    if (counters->mismatches <= 10) {
      const float(*p)[2] = c.p;
      std::printf("MISMATCH (%s: %s) corners %.9g %.9g  %.9g %.9g  %.9g %.9g  %.9g %.9g rev=%d: recorded n=%d own n=%d\n",
                  dv::kKindNames[c.kind], why, p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1],
                  c.reversed, up_n, n);
    }
  }
}

// Runs every case of a category (and of one family if named); `sharpening` is what the own decoder uses.
void RunCases(const dv::File &f, uint8_t category, const std::string &corpus, const char *only_family, double sharpening) {
  Counters total;
  std::map<std::string, Counters> per_family;
  size_t cached_image = SIZE_MAX;
  std::vector<uint8_t> pixels;
  int w = 0, h = 0;
  std::vector<std::unique_ptr<Rig>> rigs(f.rigs.size());
  for (const dv::Case &c : f.cases) {
    if (c.category != category) continue;
    if (only_family && f.rigs[c.rig][0].family != only_family) continue;
    if (!rigs[c.rig]) rigs[c.rig] = std::make_unique<Rig>(f.rigs[c.rig], sharpening);
    if (cached_image != c.image) {
      pixels = dv::Pixels(f.images[c.image], corpus, &w, &h);
      cached_image = c.image;
      if (pixels.empty()) {
        std::printf("FAIL: cannot load %s\n", f.images[c.image].path.c_str());
        ++testcheck::Failures();
        continue;
      }
    }
    if (pixels.empty()) continue;
    const apriltag_vulkan::Image oim{pixels.data(), w, h, w};
    Counters one;
    CompareCase(*rigs[c.rig], c, oim, &one);
    Counters &fam = per_family[f.rigs[c.rig][0].family];
    for (Counters *t : {&total, &fam}) {
      t->quads += one.quads, t->decoded += one.decoded, t->multi += one.multi, t->degenerate += one.degenerate;
      t->rejected_polarity += one.rejected_polarity, t->mismatches += one.mismatches, t->corrected += one.corrected;
    }
  }
  if (category == dv::kRandom) {
    for (const auto &[name, fam] : per_family) {
      std::printf("  %-18s quads %6ld  decoded %5ld (corrected %ld, two families %ld)  degenerate %ld  polarity-rejected %ld  mismatches %ld\n",
                  name.c_str(), fam.quads, fam.decoded, fam.corrected, fam.multi, fam.degenerate, fam.rejected_polarity, fam.mismatches);
    }
  }
  std::printf("%s: %ld recorded attempts, %ld decoded, %ld degenerate, %ld mismatches\n",
              category == dv::kRandom ? "random" : "corpus", total.quads, total.decoded, total.degenerate, total.mismatches);
  CHECK(total.mismatches == 0);
  CHECK(total.decoded > (only_family ? 30 : category == dv::kRandom ? 400 : 500));
  CHECK(total.quads > (only_family ? 100 : 1000));
}

// ---------------------------------------------------------------- reconcile

apriltag_vulkan::Detection MakeOwnDetection(const apriltag_vulkan::Family *f, const dv::ReconcileDet &d) {
  apriltag_vulkan::Detection out;
  out.family = f;
  out.id = d.id;
  out.hamming = d.hamming;
  out.decision_margin = d.margin;
  for (int k = 0; k < 4; ++k) out.p[k][0] = d.p[k][0], out.p[k][1] = d.p[k][1];
  return out;
}

bool SameAs(const apriltag_vulkan::Detection &y, const dv::ReconcileDet &x, bool full) {
  double p[4][2];
  for (int k = 0; k < 4; ++k) p[k][0] = x.p[k][0], p[k][1] = x.p[k][1];
  if (std::memcmp(p, y.p, sizeof(p)) != 0 || x.id != y.id) return false;
  return !full || (x.hamming == y.hamming && Bits(x.margin) == Bits(y.decision_margin));
}

void TestReconcile(const dv::File &f) {
  Rig rig({{"tag36h11", false, 2}, {"tag25h9", false, 2}}, 0.25);
  long sets = 0, removed = 0, mismatches = 0;
  for (const dv::ReconcileSet &s : f.reconcile) {
    apriltag_vulkan::DetectionList b;
    for (const dv::ReconcileDet &d : s.dets) b.push_back(MakeOwnDetection(rig.own_pointers[d.family], d));
    own::Reconcile(&b);
    ++sets;
    removed += static_cast<long>(s.dets.size() - s.kept.size());
    const char *stage = nullptr;
    size_t at = 0;
    if (b.size() != s.kept.size()) stage = "survivor count";
    for (size_t i = 0; !stage && i < b.size(); ++i) {
      const dv::ReconcileDet &x = s.dets[s.kept[i]];
      if (!(SameAs(b[i], x, true) && rig.Index(b[i].family) == x.family)) stage = "survivors", at = i;
    }
    // The expected id order: a stable sort of the recorded survivors (qsort's order for equal ids is not portable).
    std::vector<uint8_t> expected = s.kept;
    std::stable_sort(expected.begin(), expected.end(), [&](uint8_t l, uint8_t r) { return s.dets[l].id < s.dets[r].id; });
    if (!stage) {
      own::SortById(&b);
      for (size_t i = 0; !stage && i < b.size(); ++i) {
        if (!SameAs(b[i], s.dets[expected[i]], false)) stage = "id order", at = i;
      }
    }
    if (stage && ++mismatches <= 5) {
      std::printf("MISMATCH reconcile set %ld (%s, position %zu): own ids", sets - 1, stage, at);
      for (const apriltag_vulkan::Detection &d : b) std::printf(" %d", d.id);
      std::printf("; expected survivors");
      for (uint8_t k : s.kept) std::printf(" %d", s.dets[k].id);
      std::printf("; recorded sorted");
      for (uint8_t k : s.sorted) std::printf(" %d", s.dets[k].id);
      std::printf("\n");
    }
  }
  std::printf("reconcile: %ld sets, %ld detections removed, %ld mismatches\n", sets, removed, mismatches);
  CHECK(mismatches == 0);
  CHECK(removed > 1000);
}

// ---------------------------------------------------------------- family tables

// Every built-in family equals the table upstream builds for the same name (fields and a hash of the tables).
void TestFamilies(const dv::File &f) {
  long checked = 0;
  CHECK(f.families.size() == apriltag_vulkan::Family::All().size());
  for (const dv::FamilyRecord &r : f.families) {
    const apriltag_vulkan::Family *own_family = apriltag_vulkan::Family::ByName(r.name);
    CHECK(own_family != nullptr);
    if (!own_family) continue;
    const bool same = r.name == own_family->name && own_family->ncodes == r.ncodes && own_family->nbits == r.nbits &&
                      own_family->width_at_border == r.width_at_border && own_family->total_width == r.total_width &&
                      own_family->reversed_border == r.reversed_border && own_family->h == r.h && own_family->max_hamming == 2 &&
                      dv::FamilyHash(own_family->codes, own_family->ncodes, own_family->bit_x, own_family->bit_y, own_family->nbits) == r.hash;
    if (!same) std::printf("MISMATCH family %s\n", r.name.c_str());
    CHECK(same);
    checked += r.ncodes;
  }
  CHECK(apriltag_vulkan::Family::ByName("tag99h99") == nullptr);
  std::printf("families: %zu families, %ld codes compared\n", f.families.size(), checked);
}

// ---------------------------------------------------------------- speed of the decode attempts

double Seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Replays the recorded corpus attempts of tag36h11 through the own decode.
void Bench(const dv::File &f, const std::string &corpus) {
  struct Frame {
    std::vector<uint8_t> pixels;
    int w, h;
    std::vector<const dv::Case *> quads;
  };
  std::vector<Frame> frames;
  std::map<uint32_t, size_t> index;
  for (const dv::Case &c : f.cases) {
    if (c.category != dv::kCorpus || f.rigs[c.rig][0].family != "tag36h11") continue;
    auto it = index.find(c.image);
    if (it == index.end()) {
      Frame fr;
      fr.pixels = dv::Pixels(f.images[c.image], corpus, &fr.w, &fr.h);
      frames.push_back(std::move(fr));
      it = index.emplace(c.image, frames.size() - 1).first;
    }
    frames[it->second].quads.push_back(&c);
  }
  Rig rig({{"tag36h11", false, 2}}, 0.25);
  long attempts = 0;
  for (const Frame &fr : frames) attempts += static_cast<long>(fr.quads.size());
  std::printf("bench: %ld recorded tag36h11 attempts over %zu frames\n", attempts, frames.size());
  for (int round = 0; round < 5; ++round) {
    long sink = 0;
    const double t0 = Seconds();
    for (const Frame &fr : frames) {
      const apriltag_vulkan::Image oim{fr.pixels.data(), fr.w, fr.h, fr.w};
      for (const dv::Case *q : fr.quads) {
        own::DecodedTag tags[own::Decoder::kMaxFamilies];
        own::Homography h;
        sink += std::max(rig.decoder.Decode(q->p, q->reversed, &oim, &h, nullptr, tags), 0);
      }
    }
    const double t = Seconds() - t0;
    std::printf("  round %d: %.0f attempts/s  [%ld]\n", round, attempts / t, sink);
  }
}

}  // namespace

int main(int argc, char **argv) {
  const std::string which = argc > 1 ? argv[1] : "all";
  if (argc < 3) {
    std::printf("usage: own_decode_test <random|reconcile|families|corpus|bench|all> <vectors.bin> [corpus-dir [family]]\n");
    return 2;
  }
  dv::File f;
  try {
    f = dv::Read(argv[2]);
  } catch (const std::exception &e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  const std::string corpus = argc > 3 ? argv[3] : "";
  const char *only_family = argc > 4 && argv[4][0] ? argv[4] : nullptr;
  // Lets a ctest entry prove that the comparison can fail.
  const double sharpening = std::getenv("OWN_DECODE_TEST_SABOTAGE") ? 0.3 : 0.25;
  bool ran = false;
  if (which == "all" || which == "random") RunCases(f, dv::kRandom, corpus, only_family, sharpening), ran = true;
  if (which == "all" || which == "reconcile") TestReconcile(f), ran = true;
  if (which == "all" || which == "families") TestFamilies(f), ran = true;
  if ((which == "all" || which == "corpus") && !corpus.empty()) RunCases(f, dv::kCorpus, corpus, nullptr, sharpening), ran = true;
  if (which == "bench") Bench(f, corpus), ran = true;
  if (!ran) {
    std::printf("usage: own_decode_test <random|reconcile|families|corpus|bench|all> <vectors.bin> [corpus-dir [family]]\n");
    return 2;
  }
  if (testcheck::Failures() != 0) {
    std::printf("%d check(s) failed\n", testcheck::Failures());
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
