#pragma once

// The recorded upstream decode results (tests/own_decode_vectors.bin): the format, its reader and writer, and the
// images the cases refer to. tools/upstream/gen_decode_vectors.cpp writes it; tests/own_decode_test.cpp reads it.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

namespace decode_vectors {

constexpr char kMagic[8] = {'V', 'K', 'D', 'V', 'E', 'C', '0', '1'};

struct RigEntry {
  std::string family;
  bool reversed = false;
  int bits = 2;  // the corrected-bit limit the table was built for
};

enum class ImageKind : uint8_t { Crop = 0, Fill = 1, Noise = 2, File = 3 };

// An image of w x h: a crop pasted on zeros, a constant, bytes of mt19937_64(seed), or a PNG below the corpus directory.
struct Image {
  ImageKind kind = ImageKind::Fill;
  int w = 0, h = 0;
  int x0 = 0, y0 = 0, cw = 0, ch = 0;
  std::vector<uint8_t> crop;
  uint8_t fill = 0;
  uint64_t seed = 0;
  std::string path;
};

struct Det {
  uint8_t family = 0;
  int32_t id = 0, hamming = 0;
  float margin = 0;
  double h[9] = {}, c[2] = {}, p[4][2] = {};
};

// What libapriltag produced for one quad: the decode and the raw sample of its first family.
struct Expect {
  bool h_valid = false;
  double h[9] = {};
  std::vector<Det> dets;
  bool s_valid = false;
  double sh[9] = {};
  float s_margin = 0;
  uint64_t s_code = 0;
};

enum Category : uint8_t { kRandom = 0, kCorpus = 1 };
inline const char *const kKindNames[] = {"exact", "perturbed", "rolled", "other polarity", "random", "noise", "degenerate", "corpus"};

struct Case {
  uint8_t category = kRandom;
  uint8_t kind = 0;
  uint16_t rig = 0;
  uint32_t image = 0;
  float p[4][2] = {};
  bool reversed = false;
  Expect expect;
};

struct ReconcileDet {
  uint8_t family = 0, id = 0, hamming = 0;
  float margin = 0;
  float p[4][2] = {};
};

// A detection set, the indices of the detections left after reconcile_detections (in order) and after the sort by id.
struct ReconcileSet {
  std::vector<ReconcileDet> dets;
  std::vector<uint8_t> kept, sorted;
};

struct FamilyRecord {
  std::string name;
  uint32_t ncodes = 0, nbits = 0, h = 0;
  int32_t width_at_border = 0, total_width = 0;
  bool reversed_border = false;
  uint64_t hash = 0;  // FNV-1a over the codes and the bit cells
};

struct File {
  std::vector<std::vector<RigEntry>> rigs;
  std::vector<Image> images;
  std::vector<Case> cases;
  std::vector<ReconcileSet> reconcile;
  std::vector<FamilyRecord> families;
};

inline uint64_t Fnv(uint64_t h, const void *data, size_t n) {
  const uint8_t *b = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// Hash of a family's tables, for both the recording and the check.
inline uint64_t FamilyHash(const uint64_t *codes, uint32_t ncodes, const uint32_t *bit_x, const uint32_t *bit_y, uint32_t nbits) {
  uint64_t h = 14695981039346656037ull;
  h = Fnv(h, codes, sizeof(uint64_t) * ncodes);
  h = Fnv(h, bit_x, sizeof(uint32_t) * nbits);
  return Fnv(h, bit_y, sizeof(uint32_t) * nbits);
}

class Out {
 public:
  template <class T>
  void Put(T v) {
    const uint8_t *b = reinterpret_cast<const uint8_t *>(&v);
    bytes.insert(bytes.end(), b, b + sizeof(T));
  }
  void PutBytes(const void *p, size_t n) {
    const uint8_t *b = static_cast<const uint8_t *>(p);
    bytes.insert(bytes.end(), b, b + n);
  }
  void PutString(const std::string &s) {
    Put<uint16_t>(static_cast<uint16_t>(s.size()));
    PutBytes(s.data(), s.size());
  }
  std::vector<uint8_t> bytes;
};

class In {
 public:
  In(const uint8_t *begin, const uint8_t *end) : p_(begin), end_(end) {}
  template <class T>
  T Get() {
    T v;
    Need(sizeof(T));
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  void GetBytes(void *dst, size_t n) {
    Need(n);
    std::memcpy(dst, p_, n);
    p_ += n;
  }
  std::string GetString() {
    std::string s(Get<uint16_t>(), '\0');
    GetBytes(s.data(), s.size());
    return s;
  }
  bool AtEnd() const { return p_ == end_; }

 private:
  void Need(size_t n) {
    if (static_cast<size_t>(end_ - p_) < n) throw std::runtime_error("decode vectors: truncated file");
  }
  const uint8_t *p_, *end_;
};

inline void PutDoubles(Out &o, const double *v, int n) { o.PutBytes(v, sizeof(double) * n); }

inline void Write(const File &f, const std::string &path) {
  Out o;
  o.PutBytes(kMagic, 8);
  o.Put<uint32_t>(static_cast<uint32_t>(f.families.size()));
  for (const FamilyRecord &r : f.families) {
    o.PutString(r.name);
    o.Put(r.ncodes), o.Put(r.nbits), o.Put(r.h), o.Put(r.width_at_border), o.Put(r.total_width);
    o.Put<uint8_t>(r.reversed_border), o.Put(r.hash);
  }
  o.Put<uint32_t>(static_cast<uint32_t>(f.rigs.size()));
  for (const auto &rig : f.rigs) {
    o.Put<uint8_t>(static_cast<uint8_t>(rig.size()));
    for (const RigEntry &e : rig) o.PutString(e.family), o.Put<uint8_t>(e.reversed), o.Put<uint8_t>(static_cast<uint8_t>(e.bits));
  }
  o.Put<uint32_t>(static_cast<uint32_t>(f.images.size()));
  for (const Image &im : f.images) {
    o.Put<uint8_t>(static_cast<uint8_t>(im.kind));
    o.Put<int32_t>(im.w), o.Put<int32_t>(im.h);
    switch (im.kind) {
      case ImageKind::Crop:
        o.Put<int32_t>(im.x0), o.Put<int32_t>(im.y0), o.Put<int32_t>(im.cw), o.Put<int32_t>(im.ch);
        o.PutBytes(im.crop.data(), im.crop.size());
        break;
      case ImageKind::Fill: o.Put<uint8_t>(im.fill); break;
      case ImageKind::Noise: o.Put<uint64_t>(im.seed); break;
      case ImageKind::File: o.PutString(im.path); break;
    }
  }
  o.Put<uint32_t>(static_cast<uint32_t>(f.cases.size()));
  for (const Case &c : f.cases) {
    o.Put(c.category), o.Put(c.kind), o.Put(c.rig), o.Put(c.image), o.Put<uint8_t>(c.reversed);
    o.PutBytes(c.p, sizeof(c.p));
    const Expect &e = c.expect;
    o.Put<uint8_t>(e.h_valid);
    if (e.h_valid) PutDoubles(o, e.h, 9);
    o.Put<uint8_t>(static_cast<uint8_t>(e.dets.size()));
    for (const Det &d : e.dets) {
      o.Put(d.family), o.Put(d.id), o.Put(d.hamming), o.Put(d.margin);
      PutDoubles(o, d.h, 9), PutDoubles(o, d.c, 2), PutDoubles(o, &d.p[0][0], 8);
    }
    o.Put<uint8_t>(e.s_valid);
    if (e.s_valid) PutDoubles(o, e.sh, 9), o.Put(e.s_margin), o.Put(e.s_code);
  }
  o.Put<uint32_t>(static_cast<uint32_t>(f.reconcile.size()));
  for (const ReconcileSet &s : f.reconcile) {
    o.Put<uint8_t>(static_cast<uint8_t>(s.dets.size()));
    for (const ReconcileDet &d : s.dets) o.Put(d.family), o.Put(d.id), o.Put(d.hamming), o.Put(d.margin), o.PutBytes(d.p, sizeof(d.p));
    o.Put<uint8_t>(static_cast<uint8_t>(s.kept.size()));
    o.PutBytes(s.kept.data(), s.kept.size());
    o.PutBytes(s.sorted.data(), s.sorted.size());
  }
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(o.bytes.data()), static_cast<std::streamsize>(o.bytes.size()));
  if (!out) throw std::runtime_error("cannot write " + path);
}

inline File Read(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  In r(data.data(), data.data() + data.size());
  char magic[8];
  r.GetBytes(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("not a decode vector file: " + path);
  File f;
  f.families.resize(r.Get<uint32_t>());
  for (FamilyRecord &x : f.families) {
    x.name = r.GetString();
    x.ncodes = r.Get<uint32_t>(), x.nbits = r.Get<uint32_t>(), x.h = r.Get<uint32_t>();
    x.width_at_border = r.Get<int32_t>(), x.total_width = r.Get<int32_t>();
    x.reversed_border = r.Get<uint8_t>() != 0, x.hash = r.Get<uint64_t>();
  }
  f.rigs.resize(r.Get<uint32_t>());
  for (auto &rig : f.rigs) {
    rig.resize(r.Get<uint8_t>());
    for (RigEntry &e : rig) e.family = r.GetString(), e.reversed = r.Get<uint8_t>() != 0, e.bits = r.Get<uint8_t>();
  }
  f.images.resize(r.Get<uint32_t>());
  for (Image &im : f.images) {
    im.kind = static_cast<ImageKind>(r.Get<uint8_t>());
    im.w = r.Get<int32_t>(), im.h = r.Get<int32_t>();
    switch (im.kind) {
      case ImageKind::Crop:
        im.x0 = r.Get<int32_t>(), im.y0 = r.Get<int32_t>(), im.cw = r.Get<int32_t>(), im.ch = r.Get<int32_t>();
        im.crop.resize(size_t(im.cw) * im.ch);
        r.GetBytes(im.crop.data(), im.crop.size());
        break;
      case ImageKind::Fill: im.fill = r.Get<uint8_t>(); break;
      case ImageKind::Noise: im.seed = r.Get<uint64_t>(); break;
      case ImageKind::File: im.path = r.GetString(); break;
    }
  }
  f.cases.resize(r.Get<uint32_t>());
  for (Case &c : f.cases) {
    c.category = r.Get<uint8_t>(), c.kind = r.Get<uint8_t>(), c.rig = r.Get<uint16_t>(), c.image = r.Get<uint32_t>();
    c.reversed = r.Get<uint8_t>() != 0;
    r.GetBytes(c.p, sizeof(c.p));
    Expect &e = c.expect;
    e.h_valid = r.Get<uint8_t>() != 0;
    if (e.h_valid) r.GetBytes(e.h, sizeof(e.h));
    e.dets.resize(r.Get<uint8_t>());
    for (Det &d : e.dets) {
      d.family = r.Get<uint8_t>(), d.id = r.Get<int32_t>(), d.hamming = r.Get<int32_t>(), d.margin = r.Get<float>();
      r.GetBytes(d.h, sizeof(d.h)), r.GetBytes(d.c, sizeof(d.c)), r.GetBytes(d.p, sizeof(d.p));
    }
    e.s_valid = r.Get<uint8_t>() != 0;
    if (e.s_valid) r.GetBytes(e.sh, sizeof(e.sh)), e.s_margin = r.Get<float>(), e.s_code = r.Get<uint64_t>();
  }
  f.reconcile.resize(r.Get<uint32_t>());
  for (ReconcileSet &s : f.reconcile) {
    s.dets.resize(r.Get<uint8_t>());
    for (ReconcileDet &d : s.dets) {
      d.family = r.Get<uint8_t>(), d.id = r.Get<uint8_t>(), d.hamming = r.Get<uint8_t>(), d.margin = r.Get<float>();
      r.GetBytes(d.p, sizeof(d.p));
    }
    s.kept.resize(r.Get<uint8_t>());
    r.GetBytes(s.kept.data(), s.kept.size());
    s.sorted.resize(s.kept.size());
    r.GetBytes(s.sorted.data(), s.sorted.size());
  }
  if (!r.AtEnd()) throw std::runtime_error("decode vectors: trailing bytes");
  return f;
}

// The pixels of an image (w x h, tightly packed); empty if a PNG cannot be loaded.
inline std::vector<uint8_t> Pixels(const Image &im, const std::string &corpus_dir, int *w, int *h) {
  std::vector<uint8_t> px;
  if (im.kind == ImageKind::File) {
    int ch;
    uint8_t *data = stbi_load((corpus_dir + "/" + im.path).c_str(), w, h, &ch, 1);
    if (data) px.assign(data, data + size_t(*w) * *h), stbi_image_free(data);
    return px;
  }
  *w = im.w, *h = im.h;
  px.assign(size_t(im.w) * im.h, 0);
  if (im.kind == ImageKind::Fill) {
    std::fill(px.begin(), px.end(), im.fill);
  } else if (im.kind == ImageKind::Noise) {
    std::mt19937_64 rng(im.seed);
    for (uint8_t &b : px) b = uint8_t(rng());
  } else {
    for (int y = 0; y < im.ch; ++y) std::memcpy(&px[size_t(im.y0 + y) * im.w + im.x0], &im.crop[size_t(y) * im.cw], im.cw);
  }
  return px;
}

}  // namespace decode_vectors
