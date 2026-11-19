// Records what the patched upstream libapriltag decodes for a deterministic set of quads, as the test vectors
// tests/own_decode_vectors.bin that tests/own_decode_test.cpp compares the library's decode against.
// usage: gen_decode_vectors <corpus-dir> <quads-file> <out.bin> [scenes-per-rig [reconcile-sets]]
#include <algorithm>
#include <array>
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

#include "DecodeVectors.h"
#include "OwnDecode.h"
#include "upstream/UpstreamAdapter.h"

extern "C" {
#include "common/g2d.h"
#include "common/matd.h"
// Exposed by tools/upstream/patches/apriltag-expose-decode-steps.patch and apriltag-expose-code-probe.patch.
void quad_decode_index(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im, image_u8_t *im_samples,
                       zarray_t *detections);
void reconcile_detections(zarray_t *detections, zarray_t *poly0, zarray_t *poly1);
float quad_sample_code(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im, uint64_t *rcode);
}

namespace own = apriltag_vulkan::own;
namespace dv = decode_vectors;

namespace {

const char *const kFamilies[] = {"tag36h11",         "tag25h9",          "tag16h5",         "tagCircle21h7",
                                 "tagCircle49h12",   "tagStandard41h12", "tagStandard52h13", "tagCustom48h12"};

// The most wrong bits the table is built for: large tables stay at what fits in memory.
int MaxBitsFor(const apriltag_family_t *f, int wanted) {
  if (f->nbits >= 41) return std::min(wanted, 1);
  if (f->nbits >= 25) return std::min(wanted, 2);
  return wanted;
}

// An upstream detector with the given families, each with its own polarity and corrected-bit limit.
struct Rig {
  explicit Rig(std::vector<dv::RigEntry> *entries) : td(apriltag_detector_create()) {
    for (dv::RigEntry &e : *entries) {
      apriltag_family_t *f = upstream::CreateFamily(e.family.c_str());
      f->reversed_border = e.reversed;
      e.bits = MaxBitsFor(f, e.bits);
      apriltag_detector_add_family_bits(td, f, e.bits);
      families.push_back({f, e.family});
    }
    td->decode_sharpening = 0.25;
  }
  ~Rig() {
    apriltag_detector_destroy(td);
    for (auto &f : families) upstream::DestroyFamily(f.first, f.second.c_str());
  }
  Rig(const Rig &) = delete;
  Rig &operator=(const Rig &) = delete;

  int Index(const apriltag_family_t *f) const {
    for (size_t i = 0; i < families.size(); ++i) {
      if (families[i].first == f) return static_cast<int>(i);
    }
    return -1;
  }

  apriltag_detector_t *td;
  std::vector<std::pair<apriltag_family_t *, std::string>> families;
};

// The decode and the raw sample of one quad, as upstream gives them.
dv::Expect Record(Rig &rig, image_u8_t *im, const float p[4][2], bool reversed) {
  dv::Expect e;
  struct quad up = {};
  std::memcpy(up.p, p, sizeof(up.p));
  up.reversed_border = reversed;
  zarray_t *dets = zarray_create(sizeof(apriltag_detection_t *));
  quad_decode_index(rig.td, &up, im, nullptr, dets);
  e.h_valid = up.H != nullptr;
  if (up.H) std::memcpy(e.h, up.H->data, sizeof(e.h));
  for (int i = 0; i < zarray_size(dets); ++i) {
    apriltag_detection_t *d;
    zarray_get(dets, i, &d);
    dv::Det r;
    r.family = static_cast<uint8_t>(rig.Index(d->family));
    r.id = d->id;
    r.hamming = d->hamming;
    r.margin = d->decision_margin;
    std::memcpy(r.h, d->H->data, sizeof(r.h));
    std::memcpy(r.c, d->c, sizeof(r.c));
    std::memcpy(r.p, d->p, sizeof(r.p));
    e.dets.push_back(r);
    apriltag_detection_destroy(d);
  }
  zarray_destroy(dets);
  if (up.H) matd_destroy(up.H);

  struct quad probe = {};
  std::memcpy(probe.p, p, sizeof(probe.p));
  probe.reversed_border = reversed;
  uint64_t code = 0;
  e.s_margin = quad_sample_code(rig.td, &probe, im, &code);
  e.s_code = code;
  e.s_valid = probe.H != nullptr;
  if (probe.H) std::memcpy(e.sh, probe.H->data, sizeof(e.sh)), matd_destroy(probe.H);
  return e;
}

// ---------------------------------------------------------------- scene rendering

constexpr int kSize = 256;

struct Scene {
  std::vector<uint8_t> pix;
  Scene() : pix(size_t(kSize) * kSize, 128) {}
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
             double sigma, double noise, std::mt19937_64 &rng) {
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
      for (int sy = 0; sy < 3; ++sy) {
        for (int sx = 0; sx < 3; ++sx) {
          double u, v;
          own::Project(inv, x + (sx + 0.5) / 3, y + (sy + 0.5) / 3, &u, &v);
          const int cu = static_cast<int>(std::floor((u + 1) / 2 * w + start));
          const int cv = static_cast<int>(std::floor((v + 1) / 2 * w + start));
          if (cu < 0 || cv < 0 || cu >= tw || cv >= tw) {
            acc += white;
          } else {
            acc += black + (white - black) * tag->buf[cv * tag->stride + cu] / 255.0;
          }
        }
      }
      img[size_t(y) * kSize + x] = float(acc / 9);
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

// ---------------------------------------------------------------- the vector file

struct Builder {
  dv::File file;

  uint32_t AddImage(dv::Image im) {
    file.images.push_back(std::move(im));
    return static_cast<uint32_t>(file.images.size() - 1);
  }
  dv::Image Fill(int w, int h, uint8_t v) {
    dv::Image im;
    im.kind = dv::ImageKind::Fill;
    im.w = w, im.h = h, im.fill = v;
    return im;
  }
};

// Stores the scene as a crop around `box` (x0, y0, x1, y1), dilated; the rest of the frame is zeros.
dv::Image CropImage(const Scene &scene, const float box[4]) {
  const float ex = box[2] - box[0], ey = box[3] - box[1];
  const float pad = 0.12f * std::max(ex, ey) + 5;
  int x0 = std::max(0, int(std::floor(box[0] - pad))), y0 = std::max(0, int(std::floor(box[1] - pad)));
  int x1 = std::min(kSize, int(std::ceil(box[2] + pad))), y1 = std::min(kSize, int(std::ceil(box[3] + pad)));
  if (x1 <= x0 || y1 <= y0) x0 = y0 = 0, x1 = y1 = 1;
  dv::Image im;
  im.kind = dv::ImageKind::Crop;
  im.w = im.h = kSize;
  im.x0 = x0, im.y0 = y0, im.cw = x1 - x0, im.ch = y1 - y0;
  im.crop.resize(size_t(im.cw) * im.ch);
  for (int y = 0; y < im.ch; ++y) std::memcpy(&im.crop[size_t(y) * im.cw], &scene.pix[size_t(y0 + y) * kSize + x0], im.cw);
  return im;
}

// Runs the quad on the stored image with the upstream rig and appends the case.
void AddCase(Builder &b, Rig &rig, uint16_t rig_index, uint32_t image_index, const float p[4][2], bool reversed,
             uint8_t kind, uint8_t category, std::vector<uint8_t> &pixels, int w, int h) {
  dv::Case c;
  c.category = category;
  c.kind = kind;
  c.rig = rig_index;
  c.image = image_index;
  std::memcpy(c.p, p, sizeof(c.p));
  c.reversed = reversed;
  image_u8_t im{w, h, w, pixels.data()};
  c.expect = Record(rig, &im, p, reversed);
  b.file.cases.push_back(std::move(c));
}

void GenerateRandom(Builder &b, int scenes_per_rig) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> gauss(0, 1);
  const struct {
    bool reversed;
    int bits;
  } variants[] = {{false, 2}, {true, 1}, {false, 0}, {true, 3}};
  const uint32_t blank = b.AddImage(b.Fill(kSize, kSize, 128));
  std::vector<uint8_t> blank_px(size_t(kSize) * kSize, 128);
  for (const char *name : kFamilies) {
    for (const auto &v : variants) {
      for (int pair = 0; pair < 2; ++pair) {
        std::vector<dv::RigEntry> entries = {{name, v.reversed, v.bits}};
        if (pair == 1) {
          if (!std::strcmp(name, "tag16h5")) continue;
          entries.push_back({"tag16h5", v.reversed, 1});
        }
        Rig rig(&entries);
        b.file.rigs.push_back(entries);
        const uint16_t rig_index = static_cast<uint16_t>(b.file.rigs.size() - 1);
        apriltag_family_t *f = rig.families[0].first;
        for (int s = 0; s < scenes_per_rig; ++s) {
          Scene scene;
          const double side = std::exp(std::log(14.0) + u(rng) * (std::log(150.0) - std::log(14.0)));
          float corners[4][2];
          RandomCorners(rng, side, s % 3 == 0 ? 0.0 : 0.12, corners);
          own::Homography hq;
          if (!own::ComputeHomography(corners, &hq)) continue;
          const double black = u(rng) * 70, white = 150 + u(rng) * 105;
          const int id = int(rng() % f->ncodes);
          DrawTag(&scene, f, id, hq.h, black, white, (u(rng) - 0.5) * 0.8, (s % 4 == 0) ? 0 : u(rng) * 2.0,
                  (s % 5 == 0) ? 0 : u(rng) * 12, rng);
          float p[4][2];
          BorderCorners(hq, p);
          struct Pending {
            float q[4][2];
            bool reversed;
            uint8_t kind;
          };
          std::vector<Pending> pending;
          auto add = [&](const float q[4][2], bool rev, uint8_t kind) {
            Pending pd;
            std::memcpy(pd.q, q, sizeof(pd.q));
            pd.reversed = rev, pd.kind = kind;
            pending.push_back(pd);
          };
          add(p, v.reversed, 0);
          const double sigmas[] = {0.2, 0.6, 1.5, 3.0};
          for (double sg : sigmas) {
            float q[4][2];
            for (int k = 0; k < 4; ++k) {
              q[k][0] = float(p[k][0] + sg * gauss(rng));
              q[k][1] = float(p[k][1] + sg * gauss(rng));
            }
            add(q, v.reversed, 1);
          }
          float rolled[4][2];
          for (int k = 0; k < 4; ++k) std::memcpy(rolled[k], p[(k + 1) % 4], sizeof(rolled[k]));
          add(rolled, v.reversed, 2);
          add(p, !v.reversed, 3);
          // the crop covers the exact, perturbed and rolled quads
          float box[4] = {1e30f, 1e30f, -1e30f, -1e30f};
          for (const Pending &pd : pending) {
            for (int k = 0; k < 4; ++k) {
              box[0] = std::min(box[0], pd.q[k][0]), box[1] = std::min(box[1], pd.q[k][1]);
              box[2] = std::max(box[2], pd.q[k][0]), box[3] = std::max(box[3], pd.q[k][1]);
            }
          }
          float junk[4][2];
          for (int k = 0; k < 4; ++k) {
            junk[k][0] = float(u(rng) * (kSize + 40) - 20);
            junk[k][1] = float(u(rng) * (kSize + 40) - 20);
          }
          add(junk, v.reversed, 4);
          const uint32_t image = b.AddImage(CropImage(scene, box));
          int w, h;
          std::vector<uint8_t> px = dv::Pixels(b.file.images[image], "", &w, &h);
          for (const Pending &pd : pending) AddCase(b, rig, rig_index, image, pd.q, pd.reversed, pd.kind, dv::kRandom, px, w, h);
          // a quad of the right size and place on a scene of noise
          dv::Image noise;
          noise.kind = dv::ImageKind::Noise;
          noise.w = noise.h = kSize;
          noise.seed = rng();
          const uint32_t noise_index = b.AddImage(noise);
          std::vector<uint8_t> npx = dv::Pixels(b.file.images[noise_index], "", &w, &h);
          AddCase(b, rig, rig_index, noise_index, p, v.reversed, 5, dv::kRandom, npx, w, h);
        }
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
          AddCase(b, rig, rig_index, blank, d, v.reversed, 6, dv::kRandom, blank_px, kSize, kSize);
          AddCase(b, rig, rig_index, blank, d, !v.reversed, 6, dv::kRandom, blank_px, kSize, kSize);
        }
      }
    }
  }
}

// ---------------------------------------------------------------- recorded corpus attempts

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

void GenerateCorpus(Builder &b, const std::string &corpus, const std::string &quads_file) {
  std::vector<std::string> lines = ReadLines(quads_file);
  std::vector<std::string> images;
  std::unique_ptr<Rig> rig;
  uint16_t rig_index = 0;
  uint32_t image_index = 0;
  std::vector<uint8_t> px;
  int w = 0, h = 0;
  size_t frame = 0;
  std::string scenario;
  for (const std::string &line : lines) {
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "S") {
      std::string family;
      is >> scenario >> family;
      std::vector<dv::RigEntry> entries = {{family, false, 2}};
      rig = std::make_unique<Rig>(&entries);
      b.file.rigs.push_back(entries);
      rig_index = static_cast<uint16_t>(b.file.rigs.size() - 1);
      images = ReadLines(corpus + "/" + scenario + "/list.txt");
      frame = 0;
    } else if (tag == "D") {
      dv::Image im;
      im.kind = dv::ImageKind::File;
      im.path = scenario + "/" + images.at(frame++);
      image_index = b.AddImage(im);
      px = dv::Pixels(b.file.images[image_index], corpus, &w, &h);
      if (px.empty()) {
        std::fprintf(stderr, "cannot load %s\n", im.path.c_str());
        std::exit(1);
      }
    } else if (tag == "Q") {
      int rev;
      float p[4][2];
      is >> rev;
      for (int k = 0; k < 4; ++k) is >> p[k][0] >> p[k][1];
      AddCase(b, *rig, rig_index, image_index, p, rev != 0, 7, dv::kCorpus, px, w, h);
    }
  }
}

// ---------------------------------------------------------------- reconcile

void GenerateReconcile(Builder &b, int count) {
  std::vector<dv::RigEntry> entries = {{"tag36h11", false, 2}, {"tag25h9", false, 2}};
  Rig rig(&entries);
  std::mt19937_64 rng(777);
  std::uniform_real_distribution<double> u(0, 1);
  zarray_t *poly0 = g2d_polygon_create_zeros(4);
  zarray_t *poly1 = g2d_polygon_create_zeros(4);
  for (int iter = 0; iter < count; ++iter) {
    const int n = 2 + int(rng() % 14);
    std::vector<std::array<float, 8>> shapes;
    dv::ReconcileSet set;
    const double span = 30 + u(rng) * 60;
    for (int i = 0; i < n; ++i) {
      std::array<float, 8> sh;
      if (!shapes.empty() && u(rng) < 0.2) {
        sh = shapes[rng() % shapes.size()];  // an identical shape
      } else {
        double p[4][2];
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
        for (int k = 0; k < 8; ++k) sh[k] = static_cast<float>(p[k / 2][k % 2]);
      }
      shapes.push_back(sh);
      dv::ReconcileDet d;
      d.family = static_cast<uint8_t>(rng() % 2);
      d.id = static_cast<uint8_t>(rng() % 3);
      d.hamming = static_cast<uint8_t>(rng() % 3);
      d.margin = u(rng) < 0.5 ? float(int(rng() % 4) * 10) : float(u(rng) * 40);
      std::memcpy(d.p, sh.data(), sizeof(d.p));
      set.dets.push_back(d);
    }
    zarray_t *a = zarray_create(sizeof(apriltag_detection_t *));
    std::vector<apriltag_detection_t *> original;
    for (const dv::ReconcileDet &d : set.dets) {
      apriltag_detection_t *det = static_cast<apriltag_detection_t *>(std::calloc(1, sizeof(apriltag_detection_t)));
      det->family = rig.families[d.family].first;
      det->id = d.id;
      det->hamming = d.hamming;
      det->decision_margin = d.margin;
      det->H = matd_create(3, 3);
      for (int k = 0; k < 4; ++k) det->p[k][0] = d.p[k][0], det->p[k][1] = d.p[k][1];
      zarray_add(a, &det);
      original.push_back(det);
    }
    reconcile_detections(a, poly0, poly1);
    auto indices = [&](std::vector<uint8_t> *out) {
      out->clear();
      for (int i = 0; i < zarray_size(a); ++i) {
        apriltag_detection_t *x;
        zarray_get(a, i, &x);
        out->push_back(static_cast<uint8_t>(std::find(original.begin(), original.end(), x) - original.begin()));
      }
    };
    indices(&set.kept);
    qsort(a->data, a->size, a->el_sz, [](const void *l, const void *r) {
      return (*static_cast<apriltag_detection_t *const *>(l))->id - (*static_cast<apriltag_detection_t *const *>(r))->id;
    });
    indices(&set.sorted);
    for (int i = 0; i < zarray_size(a); ++i) {
      apriltag_detection_t *d;
      zarray_get(a, i, &d);
      apriltag_detection_destroy(d);
    }
    zarray_destroy(a);
    b.file.reconcile.push_back(std::move(set));
  }
  zarray_destroy(poly0);
  zarray_destroy(poly1);
}

void RecordFamilies(Builder &b) {
  for (const char *name : kFamilies) {
    apriltag_family_t *up = upstream::CreateFamily(name);
    dv::FamilyRecord r;
    r.name = up->name;
    r.ncodes = up->ncodes, r.nbits = up->nbits, r.h = up->h;
    r.width_at_border = up->width_at_border, r.total_width = up->total_width;
    r.reversed_border = up->reversed_border;
    r.hash = dv::FamilyHash(up->codes, up->ncodes, up->bit_x, up->bit_y, up->nbits);
    b.file.families.push_back(r);
    upstream::DestroyFamily(up, name);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::printf("usage: gen_decode_vectors <corpus-dir> <quads-file> <out.bin> [scenes-per-rig [reconcile-sets]]\n");
    return 2;
  }
  const int scenes = argc > 4 ? std::atoi(argv[4]) : 6;
  const int sets = argc > 5 ? std::atoi(argv[5]) : 3000;
  Builder b;
  RecordFamilies(b);
  GenerateRandom(b, scenes);
  const size_t random_cases = b.file.cases.size();
  GenerateCorpus(b, argv[1], argv[2]);
  GenerateReconcile(b, sets);
  long decoded = 0, degenerate = 0, corrected = 0, multi = 0;
  for (const dv::Case &c : b.file.cases) {
    decoded += !c.expect.dets.empty();
    degenerate += !c.expect.h_valid;
    multi += c.expect.dets.size() > 1;
    for (const dv::Det &d : c.expect.dets) corrected += d.hamming > 0;
  }
  long removed = 0;
  for (const dv::ReconcileSet &s : b.file.reconcile) removed += static_cast<long>(s.dets.size() - s.kept.size());
  dv::Write(b.file, argv[3]);
  size_t crop_bytes = 0;
  for (const dv::Image &im : b.file.images) crop_bytes += im.crop.size();
  std::printf("%zu bytes of image crops\n", crop_bytes);
  std::printf("%zu families, %zu rigs, %zu images, %zu cases (%zu random, %zu corpus), %ld with a detection, %ld corrected, %ld with two families, %ld degenerate;\n",
              b.file.families.size(), b.file.rigs.size(), b.file.images.size(), b.file.cases.size(), random_cases,
              b.file.cases.size() - random_cases, decoded, corrected, multi, degenerate);
  std::printf("%zu reconcile sets, %ld detections removed\n", b.file.reconcile.size(), removed);
  return 0;
}
