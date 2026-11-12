// CPU-only unit tests of the pure helpers and the CPU stages; no Vulkan device is used.
// usage: unit_tests [decode|geometry|edges|quadfit|options]   (no argument runs all)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "TestHooks.h"
#include "test_check.h"
#include "vkapriltag/RefineEdges.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/common/Options.h"
#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/QuadDecode.h"

extern "C" {
#include "apriltag.h"
}

namespace av = apriltag_vulkan;
namespace tst = apriltag_vulkan::testing;

constexpr double kPi = 3.14159265358979323846;

namespace {

// ---------------------------------------------------------------- tag decoding helpers

struct Family {
  apriltag_family_t *f = nullptr;
  const char *name;
  explicit Family(const char *n) : name(n) { setup_tag_family(&f, n); }
  ~Family() { teardown_tag_family(&f, name); }
};

bool Adjacent(const apriltag_family_t *f, int a, int b) {
  return a != b && std::abs(int(f->bit_x[a]) - int(f->bit_x[b])) <= 1 &&
         std::abs(int(f->bit_y[a]) - int(f->bit_y[b])) <= 1;
}

uint64_t Cell(const apriltag_family_t *f, int k) { return uint64_t{1} << (f->nbits - 1 - k); }

uint64_t RotateTimes(uint64_t w, int nbits, int times) {
  for (int i = 0; i < times; ++i) w = tst::Rotate90(w, nbits);
  return w;
}

void TestRotate90() {
  // Each bit moves a quarter turn: by nbits / 4 places, with bit 0 fixed when nbits % 4 == 1.
  const struct {
    int nbits, fixed, shift;
  } cases[] = {{36, 0, 9}, {16, 0, 4}, {25, 1, 6}, {21, 1, 5}};
  for (const auto &c : cases) {
    const int moving = c.nbits - c.fixed;
    for (int k = 0; k < c.nbits; ++k) {
      const int to = k < c.fixed ? k : c.fixed + (k - c.fixed + c.shift) % moving;
      CHECK(tst::Rotate90(uint64_t{1} << k, c.nbits) == uint64_t{1} << to);
    }
    std::mt19937_64 rng(7 + c.nbits);
    for (int i = 0; i < 50; ++i) {
      const uint64_t w = rng() & ((uint64_t{1} << c.nbits) - 1);
      CHECK(RotateTimes(w, c.nbits, 4) == w);
      CHECK(tst::Rotate90(w, c.nbits) < (uint64_t{1} << c.nbits));
    }
  }
  CHECK(tst::Rotate90(0x1234, 16) == 0x2341);
}

void TestErrorsConnected() {
  Family fam("tag36h11");
  const apriltag_family_t *f = fam.f;
  CHECK(tst::ErrorsConnected(f, 0));
  CHECK(tst::ErrorsConnected(f, Cell(f, 10)));
  int adjacent_cell = -1, distant_cell = -1;
  for (int k = 1; k < int(f->nbits); ++k) {
    if (adjacent_cell < 0 && Adjacent(f, 0, k)) adjacent_cell = k;
    if (distant_cell < 0 && std::abs(int(f->bit_x[0]) - int(f->bit_x[k])) >= 3) distant_cell = k;
  }
  CHECK(adjacent_cell > 0 && distant_cell > 0);
  CHECK(tst::ErrorsConnected(f, Cell(f, 0) | Cell(f, adjacent_cell)));
  CHECK(!tst::ErrorsConnected(f, Cell(f, 0) | Cell(f, distant_cell)));
  // A chain a - b - c is connected through b even though a and c are not adjacent.
  int chain_c = -1;
  for (int k = 0; k < int(f->nbits) && chain_c < 0; ++k) {
    if (k != 0 && k != adjacent_cell && Adjacent(f, adjacent_cell, k) && !Adjacent(f, 0, k)) chain_c = k;
  }
  CHECK(chain_c > 0);
  CHECK(tst::ErrorsConnected(f, Cell(f, 0) | Cell(f, adjacent_cell) | Cell(f, chain_c)));
  CHECK(!tst::ErrorsConnected(f, Cell(f, 0) | Cell(f, chain_c) | Cell(f, distant_cell)));
}

void TestMatchThreeTouchingBits() {
  for (const char *name : {"tag36h11", "tag25h9"}) {
    Family fam(name);
    const apriltag_family_t *f = fam.f;
    av::ThirdBitIndex index;
    tst::BuildThirdBitIndex(f, &index);
    // Three cells that touch in a chain, and three that are pairwise apart.
    int a = 12, b = -1, c = -1;
    for (int k = 0; k < int(f->nbits); ++k) {
      if (b < 0 && Adjacent(f, a, k)) b = k;
    }
    for (int k = 0; k < int(f->nbits); ++k) {
      if (k != a && k != b && Adjacent(f, b, k)) c = k;
    }
    CHECK(b >= 0 && c >= 0);
    const uint64_t touching = Cell(f, a) | Cell(f, b) | Cell(f, c);
    uint64_t apart = Cell(f, 0);
    int picked[3] = {0, -1, -1}, n = 1;
    for (int k = 1; k < int(f->nbits) && n < 3; ++k) {
      bool ok = true;
      for (int i = 0; i < n; ++i) {
        ok = ok && (std::abs(int(f->bit_x[k]) - int(f->bit_x[picked[i]])) >= 3 ||
                    std::abs(int(f->bit_y[k]) - int(f->bit_y[picked[i]])) >= 3);
      }
      if (ok) {
        picked[n++] = k;
        apart |= Cell(f, k);
      }
    }
    CHECK(n == 3);
    CHECK(!tst::ErrorsConnected(f, apart));
    for (int id : {0, 5, int(f->ncodes) / 2, int(f->ncodes) - 1}) {
      for (int rot = 0; rot < 4; ++rot) {
        // The word, rotated `rot` times, is the code with the three touching cells wrong.
        const uint64_t word = RotateTimes(f->codes[id] ^ touching, int(f->nbits), (4 - rot) % 4);
        int got_id = -1, got_rot = -1;
        if (!tst::MatchThreeTouchingBits(index, word, &got_id, &got_rot)) std::printf("  %s id %d rot %d a %d b %d c %d\n", name, id, rot, a, b, c);
        CHECK(tst::MatchThreeTouchingBits(index, word, &got_id, &got_rot));
        CHECK(got_id == id);
        CHECK(got_rot == rot);
        CHECK(!tst::MatchThreeTouchingBits(index, RotateTimes(f->codes[id] ^ apart, int(f->nbits), (4 - rot) % 4), &got_id, &got_rot));
        // Two wrong cells decode normally, so the third-bit match declines them.
        CHECK(!tst::MatchThreeTouchingBits(index, RotateTimes(f->codes[id] ^ (Cell(f, a) | Cell(f, b)), int(f->nbits), (4 - rot) % 4), &got_id, &got_rot));
      }
      int got_id, got_rot;
      CHECK(!tst::MatchThreeTouchingBits(index, f->codes[id], &got_id, &got_rot));
    }
  }
}

void TestMaxCorrectedBits() {
  const struct {
    const char *name;
    int bits;
  } cases[] = {{"tag36h11", 2}, {"tag25h9", 2}, {"tag16h5", 0}};
  for (const auto &c : cases) {
    Family fam(c.name);
    CHECK(tst::MaxCorrectedBits(fam.f) == c.bits);
  }
}

void TestDecode() {
  TestRotate90();
  TestErrorsConnected();
  TestMatchThreeTouchingBits();
  TestMaxCorrectedBits();
}

// ---------------------------------------------------------------- line geometry

void TestIntersectLines() {
  // A line is [Ex Ey nx ny]: a point on it and a normal, with the direction (ny, -nx).
  const double square[4][4] = {{10, 10, 0, 1}, {20, 10, -1, 0}, {20, 20, 0, -1}, {10, 20, 1, 0}};
  struct quad q = {};
  av::testing::IntersectLines(square, &q);
  const double want[4][2] = {{10, 10}, {20, 10}, {20, 20}, {10, 20}};
  for (int i = 0; i < 4; ++i) {
    CHECK_NEAR(q.p[i][0], want[i][0], 1e-5);
    CHECK_NEAR(q.p[i][1], want[i][1], 1e-5);
  }

  // Lines through the sides of an irregular quad, each given by an arbitrary point on the side.
  const double corner[4][2] = {{13.5, 7.25}, {61.0, 12.5}, {55.5, 49.75}, {9.25, 41.0}};
  double lines[4][4];
  for (int i = 0; i < 4; ++i) {
    const double dx = corner[(i + 1) & 3][0] - corner[i][0], dy = corner[(i + 1) & 3][1] - corner[i][1];
    const double len = std::hypot(dx, dy), t = 0.3 + 0.1 * i;
    lines[i][0] = corner[i][0] + t * dx;
    lines[i][1] = corner[i][1] + t * dy;
    lines[i][2] = -dy / len;
    lines[i][3] = dx / len;
  }
  struct quad r = {};
  av::testing::IntersectLines(lines, &r);
  for (int i = 0; i < 4; ++i) {
    CHECK_NEAR(r.p[i][0], corner[i][0], 1e-4);
    CHECK_NEAR(r.p[i][1], corner[i][1], 1e-4);
  }

  // Parallel neighbours keep the corner they had.
  double par[4][4];
  std::memcpy(par, square, sizeof par);
  par[1][2] = 0;
  par[1][3] = 1;
  struct quad k = {};
  for (int i = 0; i < 4; ++i) k.p[i][0] = k.p[i][1] = -7.0f;
  av::testing::IntersectLines(par, &k);
  CHECK(k.p[1][0] == -7.0f && k.p[1][1] == -7.0f);
}

void TestFitSide() {
  const double angle = 0.4, dx = std::cos(angle), dy = std::sin(angle);
  double xs[20], ys[20], ws[20];
  double cx = 0, cy = 0, sw = 0;
  for (int i = 0; i < 20; ++i) {
    const double t = -9.5 + i, off = 3.0;  // the line sits 3 units along its normal
    xs[i] = t * dx - off * dy;
    ys[i] = t * dy + off * dx;
    ws[i] = 1.0 + 0.25 * (i % 5);
    cx += ws[i] * xs[i];
    cy += ws[i] * ys[i];
    sw += ws[i];
  }
  double mx, my, theta;
  av::testing::FitSide(xs, ys, ws, 20, 4.0, &mx, &my, &theta);
  CHECK_NEAR(mx, cx / sw, 1e-9);
  CHECK_NEAR(my, cy / sw, 1e-9);
  // The normal is perpendicular to the line's direction, up to sign.
  CHECK_NEAR(std::fabs(std::cos(theta) * dx + std::sin(theta) * dy), 0.0, 1e-9);

  // A point with negligible weight leaves the fit where it was.
  double xo[21], yo[21], wo[21];
  std::copy(xs, xs + 20, xo);
  std::copy(ys, ys + 20, yo);
  std::copy(ws, ws + 20, wo);
  xo[20] = 40.0;
  yo[20] = -30.0;
  wo[20] = 1e-12;
  double mx2, my2, theta2;
  av::testing::FitSide(xo, yo, wo, 21, 4.0, &mx2, &my2, &theta2);
  CHECK_NEAR(mx2, mx, 1e-6);
  CHECK_NEAR(my2, my, 1e-6);
  CHECK_NEAR(std::fabs(std::cos(theta2) * dx + std::sin(theta2) * dy), 0.0, 1e-6);

  // A horizontal line has a vertical normal.
  double hx[5] = {-2, -1, 0, 1, 2}, hy[5] = {1, 1, 1, 1, 1}, hw[5] = {1, 1, 1, 1, 1};
  av::testing::FitSide(hx, hy, hw, 5, 4.0, &mx, &my, &theta);
  CHECK_NEAR(my, 1.0, 1e-12);
  CHECK_NEAR(std::fabs(std::sin(theta)), 1.0, 1e-9);
}

void TestGeometry() {
  TestIntersectLines();
  TestFitSide();
}

// ---------------------------------------------------------------- edge locators on rendered edges

// A black square (corners given in pixel-edge coordinates, pixel i covering [i, i + 1)) on white,
// supersampled 8x8 and blurred with a Gaussian of the given sigma.
std::vector<uint8_t> RenderQuad(int w, int h, const double c[4][2], double sigma, double black = 0, double white = 255) {
  constexpr int kSs = 8;
  std::vector<float> img(size_t(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int inside = 0;
      for (int sy = 0; sy < kSs; ++sy) {
        for (int sx = 0; sx < kSs; ++sx) {
          const double px = x + (sx + 0.5) / kSs, py = y + (sy + 0.5) / kSs;
          bool pos = false, neg = false;
          for (int k = 0; k < 4; ++k) {
            const double *a = c[k], *b = c[(k + 1) & 3];
            ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]) >= 0 ? pos : neg) = true;
          }
          inside += !(pos && neg);
        }
      }
      img[size_t(y) * w + x] = float(white + (black - white) * inside / (kSs * kSs));
    }
  }
  if (sigma > 0) {
    const int r = int(std::ceil(3 * sigma));
    std::vector<float> k(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = float(std::exp(-0.5 * i * i / (sigma * sigma)));
    for (float &v : k) v /= sum;
    std::vector<float> tmp(img.size());
    for (int pass = 0; pass < 2; ++pass) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          float acc = 0;
          for (int i = -r; i <= r; ++i) {
            const int xx = pass == 0 ? std::clamp(x + i, 0, w - 1) : x, yy = pass == 1 ? std::clamp(y + i, 0, h - 1) : y;
            acc += k[i + r] * img[size_t(yy) * w + xx];
          }
          tmp[size_t(y) * w + x] = acc;
        }
      }
      img.swap(tmp);
    }
  }
  std::vector<uint8_t> out(img.size());
  for (size_t i = 0; i < img.size(); ++i) out[i] = uint8_t(std::lround(std::clamp(img[i], 0.0f, 255.0f)));
  return out;
}

struct EdgeCase {
  double side, angle_deg, sigma;
};

// Worst corner error after refining a quad that starts about a pixel off its true corners.
double RefineError(av::RefineEdgesMethod method, const EdgeCase &c, double *signed_dx = nullptr, double *signed_dy = nullptr) {
  constexpr int W = 160, H = 160;
  const double cx = 79.6, cy = 80.3, a = c.angle_deg * kPi / 180.0;
  double truth[4][2];
  const double base[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int k = 0; k < 4; ++k) {
    const double x = base[k][0] * c.side / 2, y = base[k][1] * c.side / 2;
    truth[k][0] = cx + x * std::cos(a) - y * std::sin(a);
    truth[k][1] = cy + x * std::sin(a) + y * std::cos(a);
  }
  std::vector<uint8_t> pix = RenderQuad(W, H, truth, c.sigma);
  image_u8_t im = {W, H, W, pix.data()};

  Family fam("tag36h11");
  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, fam.f);
  td->quad_decimate = 1;
  td->refine_edges = true;
  const double nudge[4][2] = {{0.8, -0.6}, {-0.7, 0.9}, {0.6, 0.7}, {-0.9, -0.5}};
  struct quad q = {};
  for (int k = 0; k < 4; ++k) {
    q.p[k][0] = float(truth[k][0] + nudge[k][0]);
    q.p[k][1] = float(truth[k][1] + nudge[k][1]);
  }
  q.reversed_border = false;
  av::RefineEdges(method, td, &im, &q);
  double worst = 0, sx = 0, sy = 0;
  for (int k = 0; k < 4; ++k) {
    worst = std::max(worst, std::hypot(q.p[k][0] - truth[k][0], q.p[k][1] - truth[k][1]));
    sx += q.p[k][0] - truth[k][0];
    sy += q.p[k][1] - truth[k][1];
  }
  if (signed_dx) *signed_dx = sx / 4;
  if (signed_dy) *signed_dy = sy / 4;
  apriltag_detector_destroy(td);
  return worst;
}

void TestEdges() {
  // Bounds on the worst corner error (px) for each method, about 1.5x what the methods reach today;
  // the starting quad is about 1 px off. kExact drifts with blur, kProfile handles it by its level pass.
  const struct {
    EdgeCase c;
    double exact_bound, profile_bound;
  } cases[] = {{{60, 0, 0}, 0.2, 0.1},     {{60, 0, 1.0}, 0.3, 0.5},  {{60, 0, 2.0}, 0.85, 0.15},
               {{60, 23, 0}, 0.1, 0.1},    {{60, 23, 1.0}, 0.3, 0.5}, {{60, 41, 1.5}, 0.6, 0.1},
               {{90, 12, 1.0}, 0.3, 0.5}};
  for (const auto &t : cases) {
    double dx = 0, dy = 0;
    const double exact = RefineError(av::RefineEdgesMethod::kExact, t.c, &dx, &dy);
    CHECK(exact <= t.exact_bound);
    CHECK(std::hypot(dx, dy) < 0.15);  // no systematic shift: the pixel-edge convention holds
    const double prof = RefineError(av::RefineEdgesMethod::kProfile, t.c, &dx, &dy);
    CHECK(prof <= t.profile_bound);
    CHECK(std::hypot(dx, dy) < 0.15);
    std::printf("  side %.0f angle %2.0f sigma %.1f: exact %.3f px, profile %.3f px\n", t.c.side, t.c.angle_deg, t.c.sigma, exact, prof);
  }
}

// ---------------------------------------------------------------- quad fit on synthetic rings

// The boundary points of the pixels whose centres lie inside a quad (corners in pixel-index
// coordinates), at the sum of the indices of an inside pixel and its outside neighbour, ordered by
// angle about the centre as the GPU sorts them.
void AddRing(std::vector<av::RawLineFitPoint> *out, uint32_t blob, const double c[4][2], int weight = 100) {
  auto inside = [&](int x, int y) {
    bool pos = false, neg = false;
    for (int k = 0; k < 4; ++k) {
      const double *a = c[k], *b = c[(k + 1) & 3];
      ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0 ? pos : neg) = true;
    }
    return !(pos && neg);
  };
  double mx = 0, my = 0;
  for (int k = 0; k < 4; ++k) mx += c[k][0] / 4, my += c[k][1] / 4;
  struct P {
    double theta;
    int x2, y2;
  };
  std::vector<P> pts;
  for (int y = 1; y < 300; ++y) {
    for (int x = 1; x < 300; ++x) {
      if (!inside(x, y)) continue;
      const int d[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (const auto &n : d) {
        if (inside(x + n[0], y + n[1])) continue;
        const int x2 = 2 * x + n[0], y2 = 2 * y + n[1];
        pts.push_back({std::atan2(y2 * 0.5 - my, x2 * 0.5 - mx), x2, y2});
      }
    }
  }
  std::sort(pts.begin(), pts.end(), [](const P &a, const P &b) { return a.theta < b.theta; });
  for (const P &p : pts) {
    av::RawLineFitPoint r;
    r.xy2 = uint32_t(p.x2) | (uint32_t(p.y2) << 14);
    r.w_blob = uint32_t(weight) | (blob << 10);
    out->push_back(r);
  }
}

av::DetectorConfig FitConfig(av::DetectorConfig::QuadFitMethod method) {
  av::DetectorConfig config;
  config.width = 320;
  config.height = 320;
  config.decimation = 1;
  config.cpu_threads = 1;
  config.quad_fit_method = method;
  return config;
}

// Worst corner distance between a fitted quad and the true corners, over the four cyclic orders.
double WorstCorner(const av::DetectedQuad &q, const double corner[4][2]) {
  double best = 1e30;
  for (int rot = 0; rot < 4; ++rot) {
    double worst = 0;
    for (int k = 0; k < 4; ++k) worst = std::max(worst, std::hypot(q.p[(k + rot) & 3][0] - corner[k][0], q.p[(k + rot) & 3][1] - corner[k][1]));
    best = std::min(best, worst);
  }
  return best;
}

void SquareCorners(double cx, double cy, double side, double angle_deg, double corner[4][2]) {
  const double a = angle_deg * kPi / 180.0, base[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int k = 0; k < 4; ++k) {
    const double x = base[k][0] * side / 2, y = base[k][1] * side / 2;
    corner[k][0] = cx + x * std::cos(a) - y * std::sin(a);
    corner[k][1] = cy + x * std::sin(a) + y * std::cos(a);
  }
}

void TestQuadFit() {
  for (auto method : {av::DetectorConfig::QuadFitMethod::kDp, av::DetectorConfig::QuadFitMethod::kPeaks}) {
    const char *mname = method == av::DetectorConfig::QuadFitMethod::kDp ? "dp" : "peaks";
    av::QuadDecode decode(FitConfig(method));
    // The error is the rasterisation of the ring (about 0.1 px at worst), so a few tenths bound it.
    const struct {
      double side, angle_deg, bound;
    } cases[] = {{40, 0, 0.1}, {40, 17, 0.2}, {64, 33, 0.2}, {90, 37, 0.3}, {120, 8, 0.2}};
    for (const auto &c : cases) {
      double corner[4][2];
      SquareCorners(150.5, 140.5, c.side, c.angle_deg, corner);
      std::vector<av::RawLineFitPoint> points;
      AddRing(&points, 3, corner);
      const std::vector<av::DetectedQuad> quads = decode.Decode(points);
      CHECK(quads.size() == 1);
      if (quads.size() != 1) continue;
      const double err = WorstCorner(quads[0], corner);
      std::printf("  %s side %.0f angle %2.0f: worst corner error %.4f px\n", mname, c.side, c.angle_deg, err);
      CHECK(err <= c.bound);
      CHECK(!quads[0].coarse);
    }

    // Two blobs give two quads, in blob order.
    double first[4][2], second[4][2];
    SquareCorners(80.5, 90.5, 50, 10, first);
    SquareCorners(200.5, 180.5, 60, 30, second);
    std::vector<av::RawLineFitPoint> two;
    AddRing(&two, 1, first);
    AddRing(&two, 7, second);
    const std::vector<av::DetectedQuad> pair = decode.Decode(two);
    CHECK(pair.size() == 2);
    if (pair.size() == 2) {
      CHECK(WorstCorner(pair[0], first) <= 0.3);
      CHECK(WorstCorner(pair[1], second) <= 0.3);
    }

    // Points along one straight line have no four sides.
    std::vector<av::RawLineFitPoint> line;
    for (int i = 0; i < 200; ++i) {
      av::RawLineFitPoint r;
      r.xy2 = uint32_t(100 + 2 * i) | (uint32_t(200 + i) << 14);
      r.w_blob = 100u | (5u << 10);
      line.push_back(r);
    }
    CHECK(decode.Decode(line).empty());
    CHECK(decode.Decode({}).empty());
  }
}

// ---------------------------------------------------------------- options registry

void SetEnv(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value != nullptr ? value : "");
#else
  if (value != nullptr) setenv(name, value, 1); else unsetenv(name);
#endif
}

void TestOptions() {
  std::set<std::string> names;
  for (const av::OptionDescriptor &d : av::OptionTable()) {
    CHECK(names.insert(d.name).second);
    CHECK(d.doc != nullptr && d.doc[0] != '\0');
    CHECK(d.default_text != nullptr && d.default_text[0] != '\0');
    CHECK(d.config != nullptr);
    if (d.precedence == av::OptionPrecedence::kEnvWins || d.precedence == av::OptionPrecedence::kConfigWins) {
      CHECK(d.config[0] != '\0');
    }
    if (d.category != av::OptionCategory::kPlatform) CHECK(std::strncmp(d.name, "APRILTAG_", 9) == 0);
  }
  CHECK(av::OptionTable().size() > 80);

  const char *kVar = "APRILTAG_VK_SMALL_RESCUE";
  SetEnv(kVar, nullptr);
  CHECK(!av::Options::FromEnvironment().Has(kVar));
  CHECK(av::ResolveSmallTagRescue(av::Options::FromEnvironment(), true));
  CHECK(!av::ResolveSmallTagRescue(av::Options::FromEnvironment(), false));
  SetEnv(kVar, "0");
  CHECK(!av::ResolveSmallTagRescue(av::Options::FromEnvironment(), true));
  SetEnv(kVar, "1");
  CHECK(av::ResolveSmallTagRescue(av::Options::FromEnvironment(), false));
  SetEnv(kVar, nullptr);

  // The accessors keep the parsing the library had before the registry.
  SetEnv("APRILTAG_VK_MERGE_BAND", "12abc");
  SetEnv("APRILTAG_VK_WG2D", "32x8");
  SetEnv("APRILTAG_VK_DEVICE", "x");
  SetEnv("APRILTAG_VK_QUADFIT", "peaks");
  SetEnv("APRILTAG_VK_SPIN_WAIT", "0");
  SetEnv("APRILTAG_VK_ADAPTIVE_FLOOR", "1.5");
  const av::Options env = av::Options::FromEnvironment();
  CHECK(env.ULong("APRILTAG_VK_MERGE_BAND").value_or(0) == 12);
  CHECK(env.WxH("APRILTAG_VK_WG2D").value() == std::make_pair(32u, 8u));
  CHECK(env.Choice("APRILTAG_VK_QUADFIT", {"dp", "peaks"}) == 1);
  CHECK(!env.Flag("APRILTAG_VK_SPIN_WAIT"));
  CHECK(env.Has("APRILTAG_VK_SPIN_WAIT"));
  CHECK_NEAR(env.Float("APRILTAG_VK_ADAPTIVE_FLOOR", 2.0), 1.5, 1e-12);
  CHECK_NEAR(env.Float("APRILTAG_VK_TRACKING_CONVERGE", 0.05), 0.05, 1e-12);
  bool threw = false;
  try {
    env.StrictInt("APRILTAG_VK_DEVICE");
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    env.Has("APRILTAG_VK_NO_SUCH_SWITCH");
  } catch (const std::logic_error &) {
    threw = true;
  }
  CHECK(threw);
  for (const char *n : {"APRILTAG_VK_MERGE_BAND", "APRILTAG_VK_WG2D", "APRILTAG_VK_DEVICE", "APRILTAG_VK_QUADFIT",
                        "APRILTAG_VK_SPIN_WAIT", "APRILTAG_VK_ADAPTIVE_FLOOR"}) {
    SetEnv(n, nullptr);
  }

  // The one setting where the configuration beats the environment.
  SetEnv("APRILTAG_CPU_THREADS", "3");
  CHECK(av::ResolveThreadCount(5) == 5);
  CHECK(av::ResolveThreadCount(0) == 3);
  SetEnv("APRILTAG_CPU_THREADS", nullptr);
  CHECK(av::ResolveThreadCount(0) == 0);
}

}  // namespace

int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "all";
  const bool all = !std::strcmp(which, "all");
  bool ran = false;
  if (all || !std::strcmp(which, "decode")) TestDecode(), ran = true;
  if (all || !std::strcmp(which, "geometry")) TestGeometry(), ran = true;
  if (all || !std::strcmp(which, "edges")) TestEdges(), ran = true;
  if (all || !std::strcmp(which, "quadfit")) TestQuadFit(), ran = true;
  if (all || !std::strcmp(which, "options")) TestOptions(), ran = true;
  if (!ran) {
    std::printf("unknown test group %s\n", which);
    return 2;
  }
  if (testcheck::Failures() != 0) {
    std::printf("%d check(s) failed\n", testcheck::Failures());
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
