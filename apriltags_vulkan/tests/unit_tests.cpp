// CPU-only unit tests of the pure helpers and the CPU stages; no Vulkan device is used.
// usage: unit_tests [decode|geometry|geom|edges|quadfit|options|steps|pool]   (no argument runs all)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "Geometry.h"
#include "TestHooks.h"
#include "test_check.h"
#include "vkapriltag/RefineEdges.h"
#include "vkapriltag/Family.h"
#include "vkapriltag/common/Options.h"
#include "vkapriltag/common/WorkerPool.h"
#include "vkapriltag/gpu/CpuDecimate.h"
#include "vkapriltag/gpu/PipelinePlan.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/gpu/StageTable.h"
#include "vkapriltag/vk/Context.h"

namespace av = apriltag_vulkan;
namespace tst = apriltag_vulkan::testing;

constexpr double kPi = 3.14159265358979323846;

namespace {

// ---------------------------------------------------------------- tag decoding helpers

struct Family {
  const av::Family *f = nullptr;
  explicit Family(const char *n) : f(av::Family::ByName(n)) { CHECK(f != nullptr); }
};

// The cells of a family's tag image (total_width squared, 0 or 255), as libapriltag's apriltag_to_image draws them.
std::vector<uint8_t> RenderTag(const av::Family *fam, int idx) {
  const int tw = fam->total_width;
  std::vector<uint8_t> im(size_t(tw) * tw, 0);
  const int white_border_width = fam->width_at_border + (fam->reversed_border ? 0 : 2);
  const int white_border_start = (tw - white_border_width) / 2;
  for (int i = 0; i < white_border_width - 1; i += 1) {
    im[size_t(white_border_start) * tw + white_border_start + i] = 255;
    im[size_t(white_border_start + i) * tw + tw - 1 - white_border_start] = 255;
    im[size_t(tw - 1 - white_border_start) * tw + white_border_start + i + 1] = 255;
    im[size_t(white_border_start + 1 + i) * tw + white_border_start] = 255;
  }
  const int border_start = (tw - fam->width_at_border) / 2;
  const uint64_t code = fam->codes[idx];
  for (uint32_t i = 0; i < fam->nbits; i++) {
    if (code & (uint64_t{1} << (fam->nbits - i - 1))) im[size_t(fam->bit_y[i] + border_start) * tw + fam->bit_x[i] + border_start] = 255;
  }
  return im;
}

bool Adjacent(const av::Family *f, int a, int b) {
  return a != b && std::abs(int(f->bit_x[a]) - int(f->bit_x[b])) <= 1 &&
         std::abs(int(f->bit_y[a]) - int(f->bit_y[b])) <= 1;
}

uint64_t Cell(const av::Family *f, int k) { return uint64_t{1} << (f->nbits - 1 - k); }

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
  const av::Family *f = fam.f;
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
    const av::Family *f = fam.f;
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
  av::Quad q = {};
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
  av::Quad r = {};
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
  av::Quad k = {};
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

// ---------------------------------------------------------------- shared geometry helpers (Geometry.h)

namespace geo = apriltag_vulkan::geom;

void TestBilinearAndMedian() {
  // A 4 x 3 image with two bytes of row padding.
  uint8_t buf[6 * 3] = {0, 100, 8, 8, 0, 0, 200, 40, 8, 8, 0, 0, 16, 32, 64, 128, 0, 0};
  av::Image im = {buf, 4, 3, 6};
  CHECK_NEAR(geo::BilinearAt<double>(&im, 0, 0, 0.25, 0.5), 92.5, 0.0);
  CHECK_NEAR(geo::BilinearAt<float>(&im, 0, 0, 0.25f, 0.5f), 92.5, 0.0);
  CHECK_NEAR(geo::BilinearAt<double>(&im, 2, 1, 1.0, 1.0), 128.0, 0.0);

  double v = -1;
  // Pixel centres are at i + 0.5 in pixel-edge coordinates.
  CHECK(geo::SampleEdge<double>(&im, 0.5, 0.5, &v) && v == 0.0);
  CHECK(geo::SampleEdge<float>(&im, 1.5, 0.5, &v) && v == 100.0);
  CHECK(geo::SampleEdge<double>(&im, 0.75, 1.0, &v) && v == 92.5);
  CHECK(!geo::SampleEdge<double>(&im, 0.25, 0.5, &v));   // left of the first pixel centre
  CHECK(!geo::SampleEdge<double>(&im, 0.5, 2.75, &v));   // below the last row's centre
  CHECK(!geo::SampleEdge<float>(&im, 3.5, 0.5, &v));     // right of the last pixel centre
  CHECK(geo::SampleEdge<float>(&im, 3.25, 1.5, &v));

  double five[5] = {5, 1, 9, 3, 7};
  CHECK(geo::MedianInPlace(five, 5) == 5.0);
  float four[4] = {4, 1, 3, 2};
  CHECK(geo::MedianInPlace(four, 4) == 3.0f);
}

void TestLineCorners() {
  // x = 10 (point (10, 5), normal (1, 0)) meets y = 20 (point (3, 20), normal (0, 1)) at (10, 20).
  const double vertical[4] = {10, 5, 1, 0}, horizontal[4] = {3, 20, 0, 1};
  double x = 0, y = 0, det = 0;
  geo::LineCorner(vertical, horizontal, &x, &y, &det);
  CHECK_NEAR(x, 10.0, 1e-12);
  CHECK_NEAR(y, 20.0, 1e-12);
  CHECK_NEAR(std::fabs(det), 1.0, 1e-12);
  const double parallel[4] = {7, 1, 1, 0};
  geo::LineCorner(vertical, parallel, &x, &y, &det);
  CHECK(std::fabs(det) < 1e-12);

  // Point-and-direction form.
  CHECK(geo::IntersectDirected(0, 0, 1, 0, 3, -2, 0, 1, &x, &y));
  CHECK_NEAR(x, 3.0, 1e-12);
  CHECK_NEAR(y, 0.0, 1e-12);
  CHECK(geo::IntersectDirected(1, 1, 1, 1, 4, 0, -1, 1, &x, &y));  // y = x and y = 4 - x
  CHECK_NEAR(x, 2.0, 1e-12);
  CHECK_NEAR(y, 2.0, 1e-12);
  CHECK(!geo::IntersectDirected(0, 0, 1, 0, 0, 5, 2, 0, &x, &y));
}

void TestProjectHomography() {
  const double affine[9] = {2, 0, 1, 0, 2, 3, 0, 0, 1};
  double u = 0, v = 0, w = 0;
  geo::ProjectHomography(affine, 1.0, 1.0, &u, &v);
  CHECK(u == 3.0 && v == 5.0);
  const double persp[9] = {1, 0, 0, 0, 1, 0, 0.5, 0, 1};  // w = 0.5 x + 1
  geo::ProjectHomography(persp, 2.0, 3.0, &u, &v);
  CHECK(u == 1.0 && v == 1.5);
  // The eight-entry form takes the last entry separately and reports the denominator.
  geo::ProjectHomography(persp, 1.0, 2.0, 3.0, &u, &v, &w);
  CHECK(u == 1.0 && v == 1.5 && w == 2.0);
  // The unit square's corners under a scale and shift.
  const double shift[9] = {10, 0, 5, 0, 10, 7, 0, 0, 1};
  const double cu[4] = {0, 1, 1, 0}, cv[4] = {0, 0, 1, 1};
  const double want[4][2] = {{5, 7}, {15, 7}, {15, 17}, {5, 17}};
  for (int k = 0; k < 4; ++k) {
    geo::ProjectHomography(shift, cu[k], cv[k], &u, &v);
    CHECK(u == want[k][0] && v == want[k][1]);
  }
}

void TestSmallSolvers() {
  const double diag[9] = {2, 0, 0, 0, 4, 0, 0, 0, 5};
  CHECK(geo::Det3(diag) == 40.0);
  double inv[9];
  geo::Adjugate3(diag, 1.0 / 40.0, inv);
  const double want_diag[9] = {0.5, 0, 0, 0, 0.25, 0, 0, 0, 0.2};
  for (int i = 0; i < 9; ++i) CHECK_NEAR(inv[i], want_diag[i], 1e-15);

  // det = 1; the inverse is integral.
  const double m[9] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
  CHECK(geo::Det3(m) == 1.0);
  geo::Adjugate3(m, 1.0, inv);
  const double want[9] = {-24, 18, 5, 20, -15, -4, -5, 4, 1};
  for (int i = 0; i < 9; ++i) CHECK(inv[i] == want[i]);

  // A system that needs a row swap: x = (2, 3, -1).
  double a[4][4] = {{2, 1, -1, 0}, {-3, -1, 2, 0}, {-2, 1, 2, 0}, {0, 0, 0, 1}};
  double b[4] = {8, -11, -3, 7}, x[4] = {};
  CHECK(geo::SolveGauss(3, a, b, x));
  CHECK_NEAR(x[0], 2.0, 1e-12);
  CHECK_NEAR(x[1], 3.0, 1e-12);
  CHECK_NEAR(x[2], -1.0, 1e-12);
  // A zero leading entry is pivoted away; a singular matrix is refused.
  double p[4][4] = {{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 1}};
  double pb[4] = {3, 4, 6, 0};
  CHECK(geo::SolveGauss(3, p, pb, x));
  CHECK(x[0] == 4.0 && x[1] == 3.0 && x[2] == 3.0);
  double sing[4][4] = {{1, 2, 0, 0}, {2, 4, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  double sb[4] = {1, 2, 3, 0};
  CHECK(!geo::SolveGauss(3, sing, sb, x));
}

void TestHull() {
  // Sorted unique points of a 4 x 4 grid plus interior points: the hull is the four corners.
  using P = std::pair<int, int>;
  std::vector<P> pts;
  for (int x = 0; x <= 4; ++x)
    for (int y = 0; y <= 4; ++y) pts.push_back({x, y});
  auto cross = [](const P &o, const P &a, const P &b) {
    return static_cast<int64_t>(a.first - o.first) * (b.second - o.second) -
           static_cast<int64_t>(a.second - o.second) * (b.first - o.first);
  };
  std::vector<P> hull(2 * pts.size());
  hull.resize(geo::MonotoneChain(pts.data(), pts.size(), hull.data(), cross));
  const P want[4] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};  // counter-clockwise from the lowest-x, lowest-y point
  CHECK(hull.size() == 4);
  for (size_t i = 0; i < hull.size() && i < 4; ++i) CHECK(hull[i] == want[i]);

  // Doubles, with a collinear point on an edge that is dropped.
  struct Pt {
    double x, y;
  };
  const Pt sq[6] = {{0, 0}, {0, 2}, {1, 1}, {2, 0}, {2, 2}, {1, 0}};
  std::vector<Pt> sorted(sq, sq + 6);
  std::sort(sorted.begin(), sorted.end(), [](const Pt &a, const Pt &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
  auto dcross = [](const Pt &o, const Pt &a, const Pt &b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); };
  std::vector<Pt> dh(2 * sorted.size());
  dh.resize(geo::MonotoneChain(sorted.data(), sorted.size(), dh.data(), dcross));
  CHECK(dh.size() == 4);
  double area2 = 0;  // twice the signed area: positive for counter-clockwise
  for (size_t i = 0; i < dh.size(); ++i) {
    const Pt &a = dh[i], &b = dh[(i + 1) % dh.size()];
    area2 += a.x * b.y - a.y * b.x;
  }
  CHECK_NEAR(area2, 8.0, 1e-12);
}

void TestQuadShape() {
  const double rect[4][2] = {{1, 2}, {7, 2}, {7, 10}, {1, 10}};  // 6 x 8
  CHECK_NEAR(geo::MeanSide(rect), 7.0, 1e-15);
  CHECK_NEAR(geo::MinSide(rect), 6.0, 1e-15);
  double cx, cy;
  geo::QuadCentre(rect, &cx, &cy);
  CHECK(cx == 4.0 && cy == 6.0);
  const geo::Bounds b = geo::QuadBounds(rect);
  CHECK(b.xlo == 1.0 && b.xhi == 7.0 && b.ylo == 2.0 && b.yhi == 10.0);

  // Float corners (upstream's quad) take the float path and a rotated corner order changes nothing.
  const float fq[4][2] = {{3, 4}, {3, 0}, {0, 0}, {0, 4}};
  CHECK_NEAR(geo::MeanSide(fq), 3.5, 1e-6);
  CHECK_NEAR(geo::MinSide(fq), 3.0, 1e-6);
  geo::QuadCentre(fq, &cx, &cy);
  CHECK(cx == 1.5 && cy == 2.0);
  const geo::Bounds fb = geo::QuadBounds(fq);
  CHECK(fb.xlo == 0.0 && fb.xhi == 3.0 && fb.ylo == 0.0 && fb.yhi == 4.0);
}

void TestGeom() {
  TestBilinearAndMedian();
  TestLineCorners();
  TestProjectHomography();
  TestSmallSolvers();
  TestHull();
  TestQuadShape();
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
  av::Image im = {pix.data(), W, H, W};

  Family fam("tag36h11");
  const double nudge[4][2] = {{0.8, -0.6}, {-0.7, 0.9}, {0.6, 0.7}, {-0.9, -0.5}};
  av::Quad q = {};
  for (int k = 0; k < 4; ++k) {
    q.p[k][0] = float(truth[k][0] + nudge[k][0]);
    q.p[k][1] = float(truth[k][1] + nudge[k][1]);
  }
  q.reversed_border = false;
  av::RefineEdges(method, 1, int(fam.f->width_at_border), &im, &q);
  double worst = 0, sx = 0, sy = 0;
  for (int k = 0; k < 4; ++k) {
    worst = std::max(worst, std::hypot(q.p[k][0] - truth[k][0], q.p[k][1] - truth[k][1]));
    sx += q.p[k][0] - truth[k][0];
    sy += q.p[k][1] - truth[k][1];
  }
  if (signed_dx) *signed_dx = sx / 4;
  if (signed_dy) *signed_dy = sy / 4;
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

// ---------------------------------------------------------------- decode steps on one quad

struct Detector {
  av::TagDecoderConfig cfg;
  explicit Detector(const av::Family *f) {
    cfg.families = {f};
    cfg.refine_edges = true;
  }
};

// A tag drawn on a white canvas, kCell pixels per cell. The family image has a one-cell quiet ring
// that is white by default; `ring_grey` replaces it, and `flips` inverts the listed payload cells.
class TagScene {
 public:
  static constexpr int kCell = 8, kSize = 160, kOrigin = 40;

  TagScene(const av::Family *f, int id, int ring_grey = 255, const std::vector<int> &flips = {})
      : f_(f), pix_(size_t(kSize) * kSize, 255) {
    const int tw = f->total_width, start = (tw - int(f->width_at_border)) / 2;
    std::vector<uint8_t> cells = RenderTag(f, id);
    for (int y = 0; y < tw; ++y) {
      for (int x = 0; x < tw; ++x) {
        if (x == 0 || y == 0 || x == tw - 1 || y == tw - 1) cells[size_t(y) * tw + x] = uint8_t(ring_grey);
      }
    }
    for (int k : flips) {
      uint8_t &c = cells[size_t(f->bit_y[k] + start) * tw + f->bit_x[k] + start];
      c = c ? 0 : 255;
    }
    for (int y = 0; y < tw * kCell; ++y) {
      for (int x = 0; x < tw * kCell; ++x) pix_[size_t(kOrigin + y) * kSize + kOrigin + x] = cells[size_t(y / kCell) * tw + x / kCell];
    }
    im_.reset(new av::Image{pix_.data(), kSize, kSize, kSize});
  }

  const av::Image *image() { return im_.get(); }
  void Fill(int x0, int y0, int x1, int y1, uint8_t v) {
    for (int y = y0; y < y1; ++y) std::fill_n(pix_.begin() + size_t(y) * kSize + x0, x1 - x0, v);
  }
  // The corners of the square spanning cells [first, last) on both axes.
  static void Square(int first, int last, double corners[4][2]) {
    const double lo = kOrigin + first * kCell, hi = kOrigin + last * kCell;
    const double c[4][2] = {{lo, lo}, {hi, lo}, {hi, hi}, {lo, hi}};
    std::memcpy(corners, c, sizeof(c));
  }
  // The black border square: all but the quiet ring.
  void Border(double corners[4][2]) const {
    const int start = (f_->total_width - int(f_->width_at_border)) / 2;
    Square(start, f_->total_width - start, corners);
  }

 private:
  const av::Family *f_;
  std::vector<uint8_t> pix_;
  std::unique_ptr<av::Image> im_;
};

// Payload cells that touch cell `a`, nearest first.
std::vector<int> Neighbours(const av::Family *f, int a) {
  std::vector<int> out;
  for (int b = 0; b < int(f->nbits); ++b) {
    if (Adjacent(f, a, b)) out.push_back(b);
  }
  return out;
}

void TestPreRefineChecks() {
  Family fam("tag36h11");
  Detector det(fam.f);
  const av::DecodeSwitches on;
  av::DecodeSwitches grey_off = on;
  grey_off.grey_margin = false;

  TagScene tag(fam.f, 7);
  double border[4][2];
  tag.Border(border);
  CHECK(!tst::PreRefineRejects(det.cfg, 1, tag.image(), border, false, on));
  CHECK(!tst::PreRefineRejects(det.cfg, 1, tag.image(), border, false, grey_off));

  // A solid black square has the border of a tag but fewer white payload cells than any code.
  std::vector<uint8_t> black(size_t(TagScene::kSize) * TagScene::kSize, 255);
  for (int y = 48; y < 112; ++y) std::fill_n(black.begin() + y * TagScene::kSize + 48, 64, uint8_t{0});
  av::Image black_im = {black.data(), TagScene::kSize, TagScene::kSize, TagScene::kSize};
  const double square[4][2] = {{48, 48}, {112, 48}, {112, 112}, {48, 112}};
  CHECK(tst::PreRefineRejects(det.cfg, 1, &black_im, square, false, on));
  CHECK(tst::PreRefineRejects(det.cfg, 1, &black_im, square, false, grey_off));
  av::DecodeSwitches dark_off = on;
  dark_off.dark_reject = false;
  CHECK(!tst::PreRefineRejects(det.cfg, 1, &black_im, square, false, dark_off));

  // Quads over uniform noise: almost all fail the border or the payload check.
  std::mt19937 rng(7);
  std::vector<uint8_t> noise(size_t(TagScene::kSize) * TagScene::kSize);
  for (uint8_t &v : noise) v = uint8_t(rng() & 255);
  av::Image noise_im = {noise.data(), TagScene::kSize, TagScene::kSize, TagScene::kSize};
  int rejected = 0;
  for (int i = 0; i < 40; ++i) {
    const double x = 20 + (i % 8) * 8.0, y = 20 + (i / 8) * 8.0;
    const double quad[4][2] = {{x, y}, {x + 70, y}, {x + 70, y + 70}, {x, y + 70}};
    rejected += tst::PreRefineRejects(det.cfg, 1, &noise_im, quad, false, on);
  }
  std::printf("  noise quads rejected before refining: %d of 40\n", rejected);
  CHECK(rejected >= 20);

  // A grey ring around the tag makes its outline one cell too big; the grey margin retry covers it.
  TagScene grey(fam.f, 7, 128);
  double outline[4][2];
  TagScene::Square(0, fam.f->total_width, outline);
  CHECK(!tst::PreRefineRejects(det.cfg, 1, grey.image(), outline, false, on));
}

void TestRecoverySteps() {
  Family fam("tag36h11");
  Detector det(fam.f);
  const av::DecodeSwitches on;
  const int id = 7;

  // A decodable tag runs to the end and finds itself.
  TagScene tag(fam.f, id);
  double border[4][2];
  tag.Border(border);
  tst::StepResult r = tst::RunCandidateSteps(det.cfg, 1, tag.image(), border, false, on);
  CHECK(r.outcome == 3 && r.count == 1 && r.id == id && r.hamming == 0);

  // Grey quiet ring: the quad is the ring's outline; only the inner retry decodes it.
  TagScene grey(fam.f, id, 128);
  double outline[4][2];
  TagScene::Square(0, fam.f->total_width, outline);
  r = tst::RunCandidateSteps(det.cfg, 1, grey.image(), outline, false, on);
  CHECK(r.count == 1 && r.id == id);
  av::DecodeSwitches grey_off = on;
  grey_off.grey_margin = false;
  r = tst::RunCandidateSteps(det.cfg, 1, grey.image(), outline, false, grey_off);
  CHECK(r.count == 0);

  // Three touching payload cells wrong: only the third-bit match recovers the tag.
  std::vector<int> flips;
  for (int a = 0; a < int(fam.f->nbits) && flips.empty(); ++a) {
    const std::vector<int> n = Neighbours(fam.f, a);
    if (n.size() >= 2) flips = {a, n[0], n[1]};
  }
  TagScene damaged(fam.f, id, 255, flips);
  damaged.Border(border);
  r = tst::RunCandidateSteps(det.cfg, 1, damaged.image(), border, false, on);
  std::printf("  three touching bits wrong: count %d id %d hamming %d\n", r.count, r.id, r.hamming);
  CHECK(r.count == 1 && r.id == id && r.hamming == 3);
  av::DecodeSwitches third_off = on;
  third_off.third_bit = false;
  r = tst::RunCandidateSteps(det.cfg, 1, damaged.image(), border, false, third_off);
  CHECK(r.count == 0);

  // A dark line one cell above the tag puts the top side out; the side-one-out step moves it back.
  TagScene lined(fam.f, id);
  lined.Fill(48, 41, 112, 43, 0);
  const double out[4][2] = {{48, 41}, {112, 41}, {112, 112}, {48, 112}};
  av::DecodeSwitches side_off = on;
  side_off.side_one_out = false;
  r = tst::RunCandidateSteps(det.cfg, 1, lined.image(), out, false, side_off);
  CHECK(r.count == 1 && r.id == id && r.hamming == 2);
  CHECK_NEAR(r.corners[2][1], 41.0, 0.5);
  r = tst::RunCandidateSteps(det.cfg, 1, lined.image(), out, false, on);
  CHECK(r.count == 1 && r.id == id && r.hamming == 0);
  CHECK_NEAR(r.corners[2][1], 48.0, 0.5);

  // Three scattered wrong cells are not recovered.
  TagScene scattered(fam.f, id, 255, {0, 17, 33});
  scattered.Border(border);
  r = tst::RunCandidateSteps(det.cfg, 1, scattered.image(), border, false, on);
  CHECK(r.count == 0);
}

void TestFamilyLimits() {
  Family fam("tag16h5");
  Detector det(fam.f);
  const av::DecodeSwitches on;
  av::DecodeSwitches limits_off = on;
  limits_off.family_limits = false;
  TagScene one_wrong(fam.f, 3, 255, {5});
  double border[4][2];
  one_wrong.Border(border);
  // One corrected bit is more than tag16h5's codes can bear.
  tst::StepResult r = tst::RunCandidateSteps(det.cfg, 1, one_wrong.image(), border, false, limits_off);
  CHECK(r.count == 1 && r.id == 3 && r.hamming == 1);
  r = tst::RunCandidateSteps(det.cfg, 1, one_wrong.image(), border, false, on);
  CHECK(r.count == 0);
  TagScene exact(fam.f, 3);
  exact.Border(border);
  r = tst::RunCandidateSteps(det.cfg, 1, exact.image(), border, false, on);
  CHECK(r.count == 1 && r.id == 3 && r.hamming == 0);
}

// An image with row padding gives the same steps and corners as the packed one.
void TestStridedImage() {
  Family fam("tag36h11");
  Detector det(fam.f);
  const av::DecodeSwitches on;
  TagScene tag(fam.f, 7);
  double border[4][2];
  tag.Border(border);
  const av::Image *packed = tag.image();
  const int stride = packed->width + 40;
  std::vector<uint8_t> padded(size_t(stride) * packed->height, 77);
  for (int y = 0; y < packed->height; ++y) {
    std::memcpy(padded.data() + size_t(y) * stride, packed->data + size_t(y) * packed->stride, size_t(packed->width));
  }
  const av::Image strided = {padded.data(), packed->width, packed->height, stride};
  const tst::StepResult a = tst::RunCandidateSteps(det.cfg, 1, packed, border, false, on);
  const tst::StepResult b = tst::RunCandidateSteps(det.cfg, 1, &strided, border, false, on);
  CHECK(a.count == 1 && a.id == 7);
  CHECK(a.count == b.count && a.id == b.id && a.hamming == b.hamming);
  CHECK(std::memcmp(a.corners, b.corners, sizeof(a.corners)) == 0);
}

void TestSteps() {
  TestPreRefineChecks();
  TestRecoverySteps();
  TestFamilyLimits();
  TestStridedImage();
}

void TestPool() {
  av::WorkerPool pool(4);
  std::atomic<int> sum{0};
  pool.ParallelFor(100, [&](size_t i, unsigned) { sum += static_cast<int>(i); });
  CHECK(sum == 4950);

  bool threw = false;
  try {
    pool.ParallelFor(8, [&](size_t, unsigned) { pool.ParallelFor(2, [](size_t, unsigned) {}); });
  } catch (const std::logic_error &) {
    threw = true;
  }
  CHECK(threw);
  pool.ParallelFor(8, [&](size_t, unsigned) {});

  // Two threads on one pool take turns and both finish.
  std::atomic<int> total{0};
  const auto work = [&] {
    for (int r = 0; r < 200; ++r) pool.ParallelFor(16, [&](size_t, unsigned) { ++total; });
  };
  std::thread a(work), b(work);
  a.join();
  b.join();
  CHECK(total == 2 * 200 * 16);
}

av::vk::DeviceCaps DiscreteCaps() {
  av::vk::DeviceCaps c;
  c.name = "discrete";
  c.type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  c.max_workgroup_invocations = 1024;
  c.max_shared_memory_bytes = 32768;
  c.has_8bit_storage = c.has_int64_atomics = c.has_conditional_rendering = c.has_device_coherent_memory = true;
  c.has_integer_dot_product = c.has_subgroup_ballot = c.has_subgroup_arithmetic = c.has_subgroup_shuffle = true;
  c.has_host_cached = true;
  c.subgroup_size = 64;
  c.wg1d = 256;
  c.wg2d_x = c.wg2d_y = 16;
  c.scan_wg = 1024;
  return c;
}

av::vk::DeviceCaps IntegratedCaps() {
  av::vk::DeviceCaps c = DiscreteCaps();
  c.name = "integrated";
  c.type = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  c.has_conditional_rendering = c.has_device_coherent_memory = false;
  c.unified_memory = true;
  c.subgroup_size = 16;
  c.wg1d = 128;
  c.wg2d_x = c.wg2d_y = 8;
  return c;
}

std::string ReadWholeFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// The resolved plan of each device class and variation, compared with tests/plan/<class>.txt
// (GPU_TRACE_UPDATE=1 rewrites the files).
void TestPlan(const char *golden_dir) {
  struct Case {
    const char *name;
    int decimation;
    void (*tweak)(av::vk::DeviceCaps &);
    std::vector<std::pair<const char *, const char *>> env;
    bool readback_direct = true;
  };
  const std::vector<Case> cases = {
      {"default_d2", 2, nullptr, {}},
      {"default_d1", 1, nullptr, {}},
      {"default_d4", 4, nullptr, {}},
      {"no_subgroup", 2, [](av::vk::DeviceCaps &c) { c.has_subgroup_ballot = c.has_subgroup_arithmetic = c.has_subgroup_shuffle = false; }, {}},
      {"no_int64_atomic", 2, [](av::vk::DeviceCaps &c) { c.has_int64_atomics = false; }, {}},
      {"no_subgroup_no_int64", 2, [](av::vk::DeviceCaps &c) { c.has_subgroup_ballot = c.has_subgroup_arithmetic = c.has_subgroup_shuffle = c.has_int64_atomics = false; }, {}},
      {"no_8bit", 2, [](av::vk::DeviceCaps &c) { c.has_8bit_storage = false; }, {}},
      {"no_conditional_rendering", 2, [](av::vk::DeviceCaps &c) { c.has_conditional_rendering = false; }, {}},
      {"box_decimate_off", 2, nullptr, {{"APRILTAG_VK_BOX_DECIMATE", "0"}}},
      {"cpu_decimate_off", 2, nullptr, {{"APRILTAG_VK_CPU_DECIMATE", "0"}}},
      {"adaptive_floor_off", 2, nullptr, {{"APRILTAG_VK_ADAPTIVE_FLOOR", "0"}}},
      {"fused_threshold_off", 2, nullptr, {{"APRILTAG_VK_FUSED_THRESHOLD", "0"}}},
      {"uf_init_scan_off", 2, nullptr, {{"APRILTAG_VK_UF_INIT_SCAN", "0"}}},
      {"uf_verify", 2, nullptr, {{"APRILTAG_VK_UF_VERIFY", "1"}}},
      {"sparse_off", 2, nullptr, {{"APRILTAG_VK_SPARSE", "0"}}},
      {"merge_x4_on", 2, nullptr, {{"APRILTAG_VK_MERGE_X4", "1"}}},
      {"merge_band_16", 2, nullptr, {{"APRILTAG_VK_MERGE_BAND", "16"}}},
      {"lean_tail_0", 2, nullptr, {{"APRILTAG_VK_LEAN_TAIL", "0"}}},
      {"lean_tail_3", 2, nullptr, {{"APRILTAG_VK_LEAN_TAIL", "3"}}},
      {"fuse_submits_off", 2, nullptr, {{"APRILTAG_VK_FUSE_SUBMITS", "0"}}},
      {"no_host_cached_readback", 2, nullptr, {}, false},
      {"prerecord_off", 2, nullptr, {{"APRILTAG_VK_PRERECORD", "0"}}},
      {"sort_wg_128", 2, nullptr, {{"APRILTAG_VK_SORT_WG", "128"}, {"APRILTAG_VK_SORT_SPLIT", "512"}}},
      {"small_shared_memory", 2, [](av::vk::DeviceCaps &c) { c.max_shared_memory_bytes = 16384; }, {}},
  };
  for (size_t i = 0; i < av::kNumStages; ++i) {
    const av::StageDesc &d = av::StageInfo(static_cast<av::Stage>(i));
    CHECK(static_cast<size_t>(d.id) == i);
    CHECK(!d.bindings.empty());
  }
  const av::vk::DeviceCaps classes[2] = {DiscreteCaps(), IntegratedCaps()};
  const char *const names[2] = {"discrete", "integrated"};
  for (int k = 0; k < 2; ++k) {
    std::string text;
    for (const Case &cs : cases) {
      av::vk::DeviceCaps caps = classes[k];
      if (cs.tweak != nullptr) cs.tweak(caps);
      for (const auto &e : cs.env) SetEnv(e.first, e.second);
      const av::Options env = av::Options::FromEnvironment();
      av::DetectorConfig config;
      config.width = 640;
      config.height = 480;
      config.decimation = static_cast<uint32_t>(cs.decimation);
      av::PipelinePlan plan = av::ResolvePlan(caps, config, env);
      av::FinalisePlan(plan, caps, env, false, av::BufferOutcome{true, cs.readback_direct, cs.readback_direct, true});
      for (const auto &e : cs.env) SetEnv(e.first, nullptr);
      text += std::string("# ") + cs.name + "\n" + plan.Describe();
    }
    const std::string path = std::string(golden_dir) + "/" + names[k] + ".txt";
    if (std::getenv("GPU_TRACE_UPDATE") != nullptr) {
      std::ofstream(path, std::ios::binary) << text;
      continue;
    }
    const std::string want = ReadWholeFile(path);
    if (want != text) {
      std::printf("plan of the %s class differs from %s\n", names[k], path.c_str());
      std::istringstream a(want), b(text);
      std::string la, lb;
      for (int n = 1; std::getline(a, la) && std::getline(b, lb); ++n) {
        if (la != lb) {
          std::printf("  line %d: golden '%s' now '%s'\n", n, la.c_str(), lb.c_str());
          break;
        }
      }
    }
    CHECK(want == text);
  }
}

void TestDecimate() {
  std::mt19937 rng(7);
  const uint32_t sizes[][3] = {{96, 48, 2}, {64, 32, 4}, {60, 30, 3}, {200, 100, 2}, {200, 100, 4}, {100, 50, 5},
                               {46, 22, 2}, {52, 20, 4}, {8, 8, 8}, {640, 480, 2}, {640, 480, 4}, {1280, 720, 2}};
  for (const auto &s : sizes) {
    const uint32_t w = s[0], h = s[1], d = s[2];
    for (int pattern = 0; pattern < 3; ++pattern) {
      std::vector<uint8_t> src(size_t(w) * h);
      for (auto &v : src) v = pattern == 0 ? uint8_t(rng()) : pattern == 1 ? 255 : 0;
      const uint32_t dw = w / d, dh = h / d;
      for (int box = 0; box < 2; ++box) {
        std::vector<uint8_t> want(size_t(dw) * dh);
        for (uint32_t y = 0; y < dh; ++y) {
          for (uint32_t x = 0; x < dw; ++x) {
            uint32_t sum = 0;
            for (uint32_t j = 0; j < d; ++j) {
              for (uint32_t i = 0; i < d; ++i) sum += src[size_t(y * d + j) * w + x * d + i];
            }
            want[size_t(y) * dw + x] = box ? uint8_t((sum + d * d / 2) / (d * d)) : src[size_t(y * d) * w + x * d];
          }
        }
        std::vector<uint8_t> got(want.size(), 0xAA);
        av::CpuDecimate(src.data(), w, h, d, box != 0, false, got.data());
        CHECK(got == want);
        std::vector<uint32_t> wide(want.size(), 0xAAAAAAAAu);
        av::CpuDecimate(src.data(), w, h, d, box != 0, true, wide.data());
        bool same = true;
        for (size_t i = 0; i < want.size(); ++i) same = same && wide[i] == want[i];
        CHECK(same);
      }
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "all";
  const bool all = !std::strcmp(which, "all");
  bool ran = false;
  if (all || !std::strcmp(which, "decode")) TestDecode(), ran = true;
  if (all || !std::strcmp(which, "geometry")) TestGeometry(), ran = true;
  if (all || !std::strcmp(which, "geom")) TestGeom(), ran = true;
  if (all || !std::strcmp(which, "edges")) TestEdges(), ran = true;
  if (all || !std::strcmp(which, "quadfit")) TestQuadFit(), ran = true;
  if (all || !std::strcmp(which, "options")) TestOptions(), ran = true;
  if (all || !std::strcmp(which, "steps")) TestSteps(), ran = true;
  if (all || !std::strcmp(which, "pool")) TestPool(), ran = true;
  if (all || !std::strcmp(which, "decimate")) TestDecimate(), ran = true;
  if (!std::strcmp(which, "plan")) TestPlan(argc > 2 ? argv[2] : "tests/plan"), ran = true;
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
