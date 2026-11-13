#include "OwnDecode.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Derived from AprilRobotics apriltag v3.4.5 (apriltag.c, common/g2d.c, apriltag_math.h, common/homography.h),
// Copyright (C) 2013-2016 The Regents of The University of Michigan, BSD-2-Clause; see NOTICE.
namespace apriltag_vulkan::own {

namespace {

constexpr int kMaxCells = 32;                   // the widest border (cells) a family may have
constexpr int kMaxBorder = 8 * kMaxCells;       // border samples
constexpr int kMaxGrid = 1024;                  // total_width squared
constexpr double kPi = 3.14159265358979323846;  // M_PI
constexpr double kSingularEps = 1e-8;           // matd.h MATD_EPS

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

// The 3x3 symmetric solve of the grey model: Cholesky, inverse of the lower factor, back substitution.
void Chol3(const double *A, double *R) {
  R[0] = std::sqrt(A[0]);
  R[3] = A[1] / R[0];
  R[6] = A[2] / R[0];
  R[4] = std::sqrt(A[4] - R[3] * R[3]);
  R[7] = (A[5] - R[3] * R[6]) / R[4];
  R[8] = std::sqrt(A[8] - R[6] * R[6] - R[7] * R[7]);
  R[1] = 0;
  R[2] = 0;
  R[5] = 0;
}

void LowerInv3(const double *A, double *R) {
  R[0] = 1 / A[0];
  R[3] = -A[3] * R[0] / A[4];
  R[4] = 1 / A[4];
  R[6] = (-A[6] * R[0] - A[7] * R[3]) / A[8];
  R[7] = -A[7] * R[4] / A[8];
  R[8] = 1 / A[8];
}

void SolveWithInverse(const double *M, const double *B, double *R) {
  double tmp[3];
  tmp[0] = M[0] * B[0];
  tmp[1] = M[3] * B[0] + M[4] * B[1];
  tmp[2] = M[6] * B[0] + M[7] * B[1] + M[8] * B[2];
  R[0] = M[0] * tmp[0] + M[3] * tmp[1] + M[6] * tmp[2];
  R[1] = M[4] * tmp[1] + M[7] * tmp[2];
  R[2] = M[8] * tmp[2];
}

// Upper triangle of J'J for the points (x, y), accumulated in order.
struct NormalMatrix {
  double A[9] = {};
  void Add(double x, double y) {
    A[0] += x * x;
    A[1] += x * y;
    A[2] += x;
    A[4] += y * y;
    A[5] += y;
    A[8] += 1;
  }
};

// matd_plu()'s singular flag for a 3x3 matrix.
bool Singular3(const double *a) {
  double lu[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) lu[i][j] = a[i * 3 + j];
  }
  bool singular = false;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      const int kmax = i < j ? i : j;
      double acc = 0;
      for (int k = 0; k < kmax; k++) acc += lu[i][k] * lu[k][j];
      lu[i][j] -= acc;
    }
    int p = j;
    for (int i = j + 1; i < 3; i++) {
      if (std::fabs(lu[i][j]) > std::fabs(lu[p][j])) p = i;
    }
    if (p != j) {
      for (int k = 0; k < 3; k++) std::swap(lu[p][k], lu[j][k]);
    }
    double LUjj = lu[j][j];
    if (std::fabs(LUjj) < kSingularEps) singular = true;
    if (LUjj != 0) {
      LUjj = 1.0 / LUjj;
      for (int i = j + 1; i < 3; i++) lu[i][j] *= LUjj;
    }
  }
  return singular;
}

// Bilinear grey value at the pixel-edge coordinates (px, py); -1 if the 2x2 neighbourhood leaves the image.
inline double ValueForPixel(const Image *im, double px, double py) {
  const double fx = px - 0.5, fy = py - 0.5;
  int x1, x2, y1, y2;
  if (fx > -2147483648.0 && fx < 2147483647.0 && fy > -2147483648.0 && fy < 2147483647.0) {
    const int tx = static_cast<int>(fx), ty = static_cast<int>(fy);
    x1 = tx - (fx < tx);
    x2 = tx + (fx > tx);
    y1 = ty - (fy < ty);
    y2 = ty + (fy > ty);
  } else {
    x1 = std::floor(fx);
    x2 = std::ceil(fx);
    y1 = std::floor(fy);
    y2 = std::ceil(fy);
  }
  const double x = fx - x1;
  const double y = fy - y1;
  if (x1 < 0 || x2 >= im->width || y1 < 0 || y2 >= im->height) return -1;
  return im->data[y1 * im->stride + x1] * (1 - x) * (1 - y) + im->data[y1 * im->stride + x2] * x * (1 - y) +
         im->data[y2 * im->stride + x1] * (1 - x) * y + im->data[y2 * im->stride + x2] * x * y;
}

// The number of rotations tried by the code lookup.
constexpr int kRotations = 4;

}  // namespace

class FamilyTables {
 public:
  explicit FamilyTables(const Family *f)
      : family(f), w(f->width_at_border), tw(f->total_width), nbits(static_cast<int>(f->nbits)),
        max_hamming(f->max_hamming), reversed(f->reversed_border) {
    ok = BuildBorder() && BuildBits() && BuildIndex();
    for (int r = 0; r < kRotations; ++r) {
      const double theta = r * kPi / 2.0;
      rot_cos[r] = std::cos(theta);
      rot_sin[r] = std::sin(theta);
    }
  }

  const Family *family;
  int w, tw, nbits, max_hamming;
  bool reversed;
  bool ok = false;

  // The border samples: tag coordinates, in the order the grey models take them.
  int nborder = 0;
  double bx[kMaxBorder], by[kMaxBorder];
  // The inverse Cholesky factor of each model's J'J when every border sample is read (white, black).
  double full_inverse[2][9];

  struct Bit {
    double x, y;
    int cell;
    int neighbour[4];  // up, left, right, down: the grid cell of a payload bit there, or -1
  };
  std::vector<Bit> bits;

  // The codes bucketed by each of four bit blocks: a code within three bits of a word equals it
  // exactly in at least one block.
  int shift[4] = {}, block_bits[4] = {};
  std::vector<uint32_t> start[4];
  std::vector<uint32_t> ids[4];

  double rot_cos[kRotations], rot_sin[kRotations];

 private:
  bool BuildBorder() {
    if (w < 2 || w > kMaxCells) return false;
    const float wf = static_cast<float>(w);
    const float patterns[8][5] = {
        {-0.5f, 0.5f, 0, 1, 1},
        {0.5f, 0.5f, 0, 1, 0},
        {static_cast<float>(w + 0.5), 0.5f, 0, 1, 1},
        {static_cast<float>(w - 0.5), 0.5f, 0, 1, 0},
        {0.5f, -0.5f, 1, 0, 1},
        {0.5f, 0.5f, 1, 0, 0},
        {0.5f, static_cast<float>(w + 0.5), 1, 0, 1},
        {0.5f, static_cast<float>(w - 0.5), 1, 0, 0},
    };
    NormalMatrix sums[2];
    nborder = 0;
    for (int p = 0; p < 8; ++p) {
      const float *pat = patterns[p];
      for (int i = 0; i < w; ++i) {
        const float tx01 = (pat[0] + static_cast<float>(i) * pat[2]) / wf;
        const float ty01 = (pat[1] + static_cast<float>(i) * pat[3]) / wf;
        const double tagx = 2 * (static_cast<double>(tx01) - 0.5);
        const double tagy = 2 * (static_cast<double>(ty01) - 0.5);
        bx[nborder] = tagx;
        by[nborder] = tagy;
        ++nborder;
        sums[p & 1].Add(tagx, tagy);
      }
    }
    for (int m = 0; m < 2; ++m) {
      double chol[9];
      Chol3(sums[m].A, chol);
      LowerInv3(chol, full_inverse[m]);
    }
    return true;
  }

  bool BuildBits() {
    if (tw < 1 || tw * tw > kMaxGrid || nbits < 1 || nbits > 63) return false;
    const int min_coord = (w - tw) / 2;
    std::vector<int> cell_of(static_cast<size_t>(tw) * tw, -1);
    bits.resize(nbits);
    for (int i = 0; i < nbits; ++i) {
      const int bity = static_cast<int>(family->bit_y[i]), bitx = static_cast<int>(family->bit_x[i]);
      const int gy = bity - min_coord, gx = bitx - min_coord;
      if (gy < 0 || gy >= tw || gx < 0 || gx >= tw) return false;
      Bit &b = bits[i];
      const double tagx01 = (bitx + 0.5) / w;
      const double tagy01 = (bity + 0.5) / w;
      b.x = 2 * (tagx01 - 0.5);
      b.y = 2 * (tagy01 - 0.5);
      b.cell = gy * tw + gx;
      cell_of[b.cell] = i;
    }
    for (Bit &b : bits) {
      const int gy = b.cell / tw, gx = b.cell % tw;
      const int dy[4] = {-1, 0, 0, 1}, dx[4] = {0, -1, 1, 0};
      for (int k = 0; k < 4; ++k) {
        const int ny = gy + dy[k], nx = gx + dx[k];
        const bool inside = ny >= 0 && ny < tw && nx >= 0 && nx < tw;
        b.neighbour[k] = inside && cell_of[ny * tw + nx] >= 0 ? ny * tw + nx : -1;
      }
    }
    return true;
  }

  bool BuildIndex() {
    if (max_hamming < 0 || max_hamming > 3) return false;
    int at = 0;
    for (int k = 0; k < 4; ++k) {
      shift[k] = at;
      block_bits[k] = (nbits - at) / (4 - k);
      at += block_bits[k];
      if (block_bits[k] > 20) return false;
      const uint64_t mask = (uint64_t{1} << block_bits[k]) - 1;
      start[k].assign((size_t{1} << block_bits[k]) + 1, 0);
      for (uint32_t i = 0; i < family->ncodes; ++i) ++start[k][((family->codes[i] >> shift[k]) & mask) + 1];
      for (size_t b = 1; b < start[k].size(); ++b) start[k][b] += start[k][b - 1];
      ids[k].assign(family->ncodes, 0);
      std::vector<uint32_t> fill(start[k].begin(), start[k].end() - 1);
      for (uint32_t i = 0; i < family->ncodes; ++i) ids[k][fill[(family->codes[i] >> shift[k]) & mask]++] = i;
    }
    return true;
  }
};

namespace {

// Reads the border and payload cells of a quad with homography h; the code word goes to *rcode.
// Returns the decision margin, or -1 (with *rcode 0) if the polarity is wrong.
float SampleFamily(const FamilyTables &f, double sharpening, const Image *im, const double *h,
                   uint64_t *rcode_out) {
  *rcode_out = 0;
  const int w = f.w;
  // Grey models of the white and black border cells: intensity = C0 x + C1 y + C2.
  double B[2][3] = {{0, 0, 0}, {0, 0, 0}};
  uint8_t read[kMaxBorder];
  bool all[2] = {true, true};
  // The border samples projected in one pass over the table, which vectorises.
  double border_x[kMaxBorder], border_y[kMaxBorder];
  for (int k = 0; k < f.nborder; ++k) Project(h, f.bx[k], f.by[k], &border_x[k], &border_y[k]);
  int n = 0;
  for (int p = 0; p < 8; ++p) {
    const int m = p & 1;
    for (int i = 0; i < w; ++i, ++n) {
      const double tagx = f.bx[n], tagy = f.by[n];
      const double px = border_x[n], py = border_y[n];
      const int ix = px;
      const int iy = py;
      if (ix < 0 || iy < 0 || ix >= im->width || iy >= im->height) {
        read[n] = 0;
        all[m] = false;
        continue;
      }
      read[n] = 1;
      const double v = im->data[iy * im->stride + ix];
      B[m][0] += tagx * v;
      B[m][1] += tagy * v;
      B[m][2] += v;
    }
  }
  double C[2][3];
  for (int m = 0; m < 2; ++m) {
    if (all[m]) {
      SolveWithInverse(f.full_inverse[m], B[m], C[m]);
      continue;
    }
    NormalMatrix sums;
    for (int p = m; p < 8; p += 2) {
      for (int i = 0; i < w; ++i) {
        const int k = p * w + i;
        if (read[k]) sums.Add(f.bx[k], f.by[k]);
      }
    }
    double chol[9], inverse[9];
    Chol3(sums.A, chol);
    LowerInv3(chol, inverse);
    SolveWithInverse(inverse, B[m], C[m]);
  }
  const double *cw = C[0], *cb = C[1];
  const double white0 = cw[0] * 0 + cw[1] * 0 + cw[2];
  const double black0 = cb[0] * 0 + cb[1] * 0 + cb[2];
  if ((white0 - black0 < 0) != f.reversed) return -1;

  const int tw = f.tw;
  double values[kMaxGrid];
  std::memset(values, 0, sizeof(double) * tw * tw);
  bool finite = true;
  double bit_x[64], bit_y[64];
  for (size_t k = 0; k < f.bits.size(); ++k) Project(h, f.bits[k].x, f.bits[k].y, &bit_x[k], &bit_y[k]);
  for (size_t k = 0; k < f.bits.size(); ++k) {
    const FamilyTables::Bit &bit = f.bits[k];
    const double v = ValueForPixel(im, bit_x[k], bit_y[k]);
    if (v == -1) continue;
    const double thresh = ((cb[0] * bit.x + cb[1] * bit.y + cb[2]) + (cw[0] * bit.x + cw[1] * bit.y + cw[2])) / 2.0;
    values[bit.cell] = v - thresh;
    finite = finite && std::isfinite(thresh);
  }

  double sharp[64];
  const int nbits = f.nbits;
  if (finite) {
    for (int i = 0; i < nbits; ++i) {
      const FamilyTables::Bit &bit = f.bits[i];
      double s = 0;
      if (bit.neighbour[0] >= 0) s += values[bit.neighbour[0]] * -1;
      if (bit.neighbour[1] >= 0) s += values[bit.neighbour[1]] * -1;
      s += values[bit.cell] * 4;
      if (bit.neighbour[2] >= 0) s += values[bit.neighbour[2]] * -1;
      if (bit.neighbour[3] >= 0) s += values[bit.neighbour[3]] * -1;
      sharp[i] = s;
    }
  } else {
    // Non-finite values: the sharpening kernel's zero taps matter, so run it over the whole grid.
    double all_sharp[kMaxGrid];
    for (int y = 0; y < tw; y++) {
      for (int x = 0; x < tw; x++) {
        all_sharp[y * tw + x] = 0;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if ((y + i - 1) < 0 || (y + i - 1) > tw - 1 || (x + j - 1) < 0 || (x + j - 1) > tw - 1) continue;
            static const double kernel[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
            all_sharp[y * tw + x] += values[(y + i - 1) * tw + (x + j - 1)] * kernel[i * 3 + j];
          }
        }
      }
    }
    for (int i = 0; i < nbits; ++i) sharp[i] = all_sharp[f.bits[i].cell];
  }

  float black_score = 0, white_score = 0;
  float black_score_count = 1, white_score_count = 1;
  uint64_t rcode = 0;
  for (int i = 0; i < nbits; ++i) {
    rcode = (rcode << 1);
    const double v = values[f.bits[i].cell] + sharpening * sharp[i];
    if (v > 0) {
      white_score += v;
      white_score_count++;
      rcode |= 1;
    } else {
      black_score -= v;
      black_score_count++;
    }
  }
  *rcode_out = rcode;
  return fmin(white_score / white_score_count, black_score / black_score_count);
}

// The lowest-numbered code within max_hamming bits of the word over the first rotation that has one.
bool Lookup(const FamilyTables &f, uint64_t rcode, int *id, int *hamming, int *rotation) {
  uint64_t r = rcode;
  const uint64_t *codes = f.family->codes;
  for (int rot = 0; rot < kRotations; ++rot, r = Rotate90(r, f.nbits)) {
    uint32_t best = UINT32_MAX;
    int best_distance = 0;
    for (int k = 0; k < 4; ++k) {
      const uint64_t value = (r >> f.shift[k]) & ((uint64_t{1} << f.block_bits[k]) - 1);
      const uint32_t *ids = f.ids[k].data();
      for (uint32_t pos = f.start[k][value]; pos < f.start[k][value + 1]; ++pos) {
        const uint32_t i = ids[pos];
        if (i >= best) break;
        const int d = std::popcount(r ^ codes[i]);
        if (d <= f.max_hamming) {
          best = i;
          best_distance = d;
          break;
        }
      }
    }
    if (best != UINT32_MAX) {
      *id = static_cast<int>(best);
      *hamming = best_distance;
      *rotation = rot;
      return true;
    }
  }
  return false;
}

}  // namespace

bool ComputeHomography(const float p[4][2], Homography *out) {
  double c[4][4];
  for (int i = 0; i < 4; i++) {
    c[i][0] = (i == 0 || i == 3) ? -1 : 1;
    c[i][1] = (i == 0 || i == 1) ? -1 : 1;
    c[i][2] = p[i][0];
    c[i][3] = p[i][1];
  }
  double A[] = {
      c[0][0], c[0][1], 1,       0,       0, 0, -c[0][0] * c[0][2], -c[0][1] * c[0][2], c[0][2],
      0,       0,       0,       c[0][0], c[0][1], 1, -c[0][0] * c[0][3], -c[0][1] * c[0][3], c[0][3],
      c[1][0], c[1][1], 1,       0,       0, 0, -c[1][0] * c[1][2], -c[1][1] * c[1][2], c[1][2],
      0,       0,       0,       c[1][0], c[1][1], 1, -c[1][0] * c[1][3], -c[1][1] * c[1][3], c[1][3],
      c[2][0], c[2][1], 1,       0,       0, 0, -c[2][0] * c[2][2], -c[2][1] * c[2][2], c[2][2],
      0,       0,       0,       c[2][0], c[2][1], 1, -c[2][0] * c[2][3], -c[2][1] * c[2][3], c[2][3],
      c[3][0], c[3][1], 1,       0,       0, 0, -c[3][0] * c[3][2], -c[3][1] * c[3][2], c[3][2],
      0,       0,       0,       c[3][0], c[3][1], 1, -c[3][0] * c[3][3], -c[3][1] * c[3][3], c[3][3],
  };
  const double epsilon = 1e-10;
  for (int col = 0; col < 8; col++) {
    double max_val = 0;
    int max_val_idx = -1;
    for (int row = col; row < 8; row++) {
      const double val = std::fabs(A[row * 9 + col]);
      if (val > max_val) {
        max_val = val;
        max_val_idx = row;
      }
    }
    if (max_val_idx < 0) return false;
    if (max_val < epsilon) return false;
    if (max_val_idx != col) {
      for (int i = col; i < 9; i++) std::swap(A[col * 9 + i], A[max_val_idx * 9 + i]);
    }
    for (int i = col + 1; i < 8; i++) {
      const double f = A[i * 9 + col] / A[col * 9 + col];
      A[i * 9 + col] = 0;
      for (int j = col + 1; j < 9; j++) A[i * 9 + j] -= f * A[col * 9 + j];
    }
  }
  for (int col = 7; col >= 0; col--) {
    double sum = 0;
    for (int i = col + 1; i < 8; i++) sum += A[col * 9 + i] * A[i * 9 + 8];
    A[col * 9 + 8] = (A[col * 9 + 8] - sum) / A[col * 9 + col];
  }
  const double h[9] = {A[8], A[17], A[26], A[35], A[44], A[53], A[62], A[71], 1};
  if (Singular3(h)) return false;
  std::memcpy(out->h, h, sizeof(h));
  return true;
}

Decoder::Decoder() = default;
Decoder::~Decoder() = default;

bool Decoder::Refresh(std::span<const Family *const> families, double sharpening) {
  sharpening_ = sharpening;
  const int count = static_cast<int>(families.size());
  if (count > kMaxFamilies) return false;
  bool same = static_cast<int>(families_.size()) == count;
  for (int i = 0; same && i < count; ++i) {
    same = families_[i]->family == families[i] && families_[i]->max_hamming == families[i]->max_hamming;
  }
  if (!same) {
    families_.clear();
    for (int i = 0; i < count; ++i) families_.push_back(std::make_unique<FamilyTables>(families[i]));
  }
  for (const auto &f : families_) {
    if (!f->ok) return false;
  }
  return true;
}

bool Decoder::SampleCode(const float p[4][2], bool reversed_border, const Image *im, Homography *h,
                         RawSample *raw) const {
  if (!ComputeHomography(p, h)) return false;
  *raw = RawSample{};
  for (const auto &f : families_) {
    if (f->reversed != reversed_border) continue;
    raw->margin = SampleFamily(*f, sharpening_, im, h->h, &raw->rcode);
    raw->valid = true;
    break;
  }
  return true;
}

void Decoder::FillTag(const Family *family, int id, int hamming, int rotation, float margin,
                      const double *h, DecodedTag *out) {
  out->family = family;
  out->id = id;
  out->hamming = hamming;
  out->rotation = rotation;
  out->margin = margin;
  const double theta = rotation * kPi / 2.0;
  const double c = std::cos(theta), s = std::sin(theta);
  const double R[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double acc = 0;
      for (int k = 0; k < 3; ++k) acc += h[i * 3 + k] * R[k * 3 + j];
      out->h[i * 3 + j] = acc;
    }
  }
  Project(out->h, 0, 0, &out->c[0], &out->c[1]);
  for (int i = 0; i < 4; i++) {
    const int tcx = (i == 1 || i == 2) ? 1 : -1;
    const int tcy = (i < 2) ? 1 : -1;
    Project(out->h, tcx, tcy, &out->p[i][0], &out->p[i][1]);
  }
}

int Decoder::Decode(const float p[4][2], bool reversed_border, const Image *im, Homography *h_out,
                    RawSample *first, DecodedTag *out) const {
  Homography local;
  Homography *h = h_out ? h_out : &local;
  if (!ComputeHomography(p, h)) return -1;
  if (first) *first = RawSample{};
  int count = 0;
  for (const auto &f : families_) {
    if (f->reversed != reversed_border) continue;
    uint64_t rcode;
    const float margin = SampleFamily(*f, sharpening_, im, h->h, &rcode);
    if (first && !first->valid) {
      first->margin = margin;
      first->rcode = rcode;
      first->valid = true;
    }
    if (!(margin >= 0)) continue;
    int id, hamming, rotation;
    if (!Lookup(*f, rcode, &id, &hamming, &rotation)) continue;
    FillTag(f->family, id, hamming, rotation, margin, h->h, &out[count++]);
  }
  return count;
}

int Decoder::MinHamming(uint64_t rcode) const {
  if (families_.empty()) return 64;
  const Family *family = families_[0]->family;
  int best = 64;
  for (int r = 0; r < kRotations; ++r) {
    for (uint32_t c = 0; c < family->ncodes; ++c) best = std::min(best, std::popcount(rcode ^ family->codes[c]));
    rcode = Rotate90(rcode, static_cast<int>(family->nbits));
  }
  return best;
}

Detection Decoder::ToDetection(const DecodedTag &tag) {
  Detection det;
  det.family = tag.family;
  det.id = tag.id;
  det.hamming = tag.hamming;
  det.decision_margin = tag.margin;
  std::memcpy(det.H, tag.h, sizeof(tag.h));
  det.c[0] = tag.c[0];
  det.c[1] = tag.c[1];
  for (int i = 0; i < 4; ++i) {
    det.p[i][0] = tag.p[i][0];
    det.p[i][1] = tag.p[i][1];
  }
  return det;
}

namespace {

struct Segment {
  double p[2], u[2], p1[2];
};

void InitSegment(Segment *s, const double p0[2], const double p1[2]) {
  s->p[0] = p0[0];
  s->p[1] = p0[1];
  s->u[0] = p1[0] - p0[0];
  s->u[1] = p1[1] - p0[1];
  const float mag = sqrtf(static_cast<float>(s->u[0] * s->u[0] + s->u[1] * s->u[1]));
  s->u[0] /= mag;
  s->u[1] /= mag;
  s->p1[0] = p1[0];
  s->p1[1] = p1[1];
}

double Coordinate(const Segment &s, const double q[2]) {
  return (q[0] - s.p[0]) * s.u[0] + (q[1] - s.p[1]) * s.u[1];
}

bool SegmentsIntersect(const Segment &a, const Segment &b) {
  const double m00 = a.u[0], m01 = -b.u[0], m10 = a.u[1], m11 = -b.u[1];
  const double det = m00 * m11 - m01 * m10;
  if (std::fabs(det) < 0.00000001) return false;
  const double i00 = m11 / det, i01 = -m01 / det;
  const double b00 = b.p[0] - a.p[0], b10 = b.p[1] - a.p[1];
  const double x00 = i00 * b00 + i01 * b10;
  const double tmp[2] = {a.u[0] * x00 + a.p[0], a.u[1] * x00 + a.p[1]};

  double lo = Coordinate(a, a.p), hi = Coordinate(a, a.p1), c = Coordinate(a, tmp);
  if ((c < lo && c < hi) || (c > lo && c > hi)) return false;
  lo = Coordinate(b, b.p);
  hi = Coordinate(b, b.p1);
  c = Coordinate(b, tmp);
  if ((c < lo && c < hi) || (c > lo && c > hi)) return false;
  return true;
}

bool EdgesIntersect(const double a[4][2], const double b[4][2]) {
  for (int ia = 0; ia < 4; ia++) {
    Segment sa;
    InitSegment(&sa, a[ia], a[(ia + 1) % 4]);
    for (int ib = 0; ib < 4; ib++) {
      Segment sb;
      InitSegment(&sb, b[ib], b[(ib + 1) % 4]);
      if (SegmentsIntersect(sa, sb)) return true;
    }
  }
  return false;
}

bool ContainsPoint(const double poly[4][2], const double q[2]) {
  int last_quadrant = 0;
  int quad_acc = 0;
  for (int i = 0; i <= 4; i++) {
    const double *p = poly[i % 4];
    int quadrant;
    if (p[0] < q[0]) {
      quadrant = (p[1] < q[1]) ? 2 : 1;
    } else {
      quadrant = (p[1] < q[1]) ? 3 : 0;
    }
    if (i > 0) {
      const int dquadrant = quadrant - last_quadrant;
      switch (dquadrant) {
        case -3:
        case 1:
          quad_acc++;
          break;
        case -1:
        case 3:
          quad_acc--;
          break;
        case -2:
        case 2: {
          const double *p0 = poly[i - 1];
          const double nx = p[1] - q[1];
          const double ny = -p[0] + q[0];
          const double dot = nx * (p0[0] - q[0]) + ny * (p0[1] - q[1]);
          if (dot < 0) {
            quad_acc -= 2;
          } else {
            quad_acc += 2;
          }
          break;
        }
        default:
          break;
      }
    }
    last_quadrant = quadrant;
  }
  return (quad_acc >= 2) || (quad_acc <= -2);
}

void InteriorPoint(const double poly[4][2], double *p) {
  p[0] = (poly[0][0] + poly[1][0] + poly[2][0]) / 3;
  p[1] = (poly[0][1] + poly[1][1] + poly[2][1]) / 3;
}

bool Overlaps(const double a[4][2], const double b[4][2]) {
  if (EdgesIntersect(a, b)) return true;
  double p[2];
  InteriorPoint(b, p);
  if (ContainsPoint(a, p)) return true;
  InteriorPoint(a, p);
  return ContainsPoint(b, p);
}

int PreferSmaller(int pref, double q0, double q1) {
  if (pref) return pref;
  if (q0 < q1) return -1;
  if (q1 < q0) return 1;
  return 0;
}

// Removes entry i by moving the last one into its place.
void RemoveShuffle(DetectionList *list, size_t i) {
  if (i + 1 < list->size()) (*list)[i] = list->back();
  list->pop_back();
}

}  // namespace

void Reconcile(DetectionList *detections) {
  DetectionList &d = *detections;
  for (int i0 = 0; i0 < static_cast<int>(d.size()); i0++) {
    for (int i1 = i0 + 1; i1 < static_cast<int>(d.size()); i1++) {
      const Detection &det0 = d[i0], &det1 = d[i1];
      if (det0.id != det1.id || det0.family != det1.family) continue;
      if (!Overlaps(det0.p, det1.p)) continue;

      int pref = 0;
      pref = PreferSmaller(pref, det0.hamming, det1.hamming);
      pref = PreferSmaller(pref, -det0.decision_margin, -det1.decision_margin);
      for (int i = 0; i < 4; i++) {
        pref = PreferSmaller(pref, det0.p[i][0], det1.p[i][0]);
        pref = PreferSmaller(pref, det0.p[i][1], det1.p[i][1]);
      }
      if (pref < 0) {
        RemoveShuffle(&d, i1);
        i1--;
      } else {
        RemoveShuffle(&d, i0);
        i0--;
        break;
      }
    }
  }
}

void SortById(DetectionList *detections) {
  std::stable_sort(detections->begin(), detections->end(),
                   [](const Detection &l, const Detection &r) { return l.id < r.id; });
}

}  // namespace apriltag_vulkan::own
