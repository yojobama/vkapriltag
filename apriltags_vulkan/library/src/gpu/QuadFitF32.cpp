#include "vkapriltag/gpu/QuadFitF32.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#if (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)) && !defined(VKAPRILTAG_NO_SSE2)
#define VKA_HAVE_SSE2 1
#include <emmintrin.h>
#else
#define VKA_HAVE_SSE2 0
#endif

#if defined(_MSC_VER)
#define VKA_RESTRICT __restrict
#else
#define VKA_RESTRICT __restrict__
#endif

namespace apriltag_vulkan::quadf32 {
namespace {

constexpr int kLanes = 8;
constexpr size_t kNMaxima = 10;

// Gaussian filter taps applied to the windowed error.
constexpr std::array<float, 7> kTaps = {
    0.01110899634659290314f, 0.13533528149127960205f, 0.60653066635131835938f, 1.0f,
    0.60653066635131835938f, 0.13533528149127960205f, 0.01110899634659290314f,
};

// The eight lanes summed in a fixed order.
inline float Sum8(const float *a) { return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])); }

#if VKA_HAVE_SSE2
inline __m128 LoadF(const int32_t *p, __m128i origin) { return _mm_cvtepi32_ps(_mm_sub_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)), origin)); }
inline __m128 LoadW(const int32_t *p) { return _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))); }
#endif

// First pass over the segments [s[0], s[1]) and [s[2], s[3]): weight and first moments about (ox, oy).
// With kSimd the full blocks of eight run in SSE2, adding in the same lanes and order as the scalar loop.
template <bool kSimd>
void Pass1(const Points &p, const size_t s[4], int32_t ox, int32_t oy, float &W, float &sx, float &sy) {
  float aw[kLanes] = {}, ax[kLanes] = {}, ay[kLanes] = {};
  for (int g = 0; g < 2; ++g) {
    size_t k = s[2 * g];
    const size_t to = s[2 * g + 1];
#if VKA_HAVE_SSE2
    if constexpr (kSimd) {
      const __m128i vox = _mm_set1_epi32(ox), voy = _mm_set1_epi32(oy);
      __m128 w0 = _mm_loadu_ps(aw), w1 = _mm_loadu_ps(aw + 4), x0 = _mm_loadu_ps(ax), x1 = _mm_loadu_ps(ax + 4),
             y0 = _mm_loadu_ps(ay), y1 = _mm_loadu_ps(ay + 4);
      for (; k + kLanes <= to; k += kLanes) {
        const __m128 fw0 = LoadW(p.w + k), fw1 = LoadW(p.w + k + 4);
        w0 = _mm_add_ps(w0, fw0);
        w1 = _mm_add_ps(w1, fw1);
        x0 = _mm_add_ps(x0, _mm_mul_ps(fw0, LoadF(p.x + k, vox)));
        x1 = _mm_add_ps(x1, _mm_mul_ps(fw1, LoadF(p.x + k + 4, vox)));
        y0 = _mm_add_ps(y0, _mm_mul_ps(fw0, LoadF(p.y + k, voy)));
        y1 = _mm_add_ps(y1, _mm_mul_ps(fw1, LoadF(p.y + k + 4, voy)));
      }
      _mm_storeu_ps(aw, w0), _mm_storeu_ps(aw + 4, w1), _mm_storeu_ps(ax, x0), _mm_storeu_ps(ax + 4, x1);
      _mm_storeu_ps(ay, y0), _mm_storeu_ps(ay + 4, y1);
    }
#endif
    if constexpr (!kSimd || !VKA_HAVE_SSE2) {
      for (; k + kLanes <= to; k += kLanes) {
        for (int l = 0; l < kLanes; ++l) {
          const float w = static_cast<float>(p.w[k + l]);
          aw[l] += w;
          ax[l] += w * static_cast<float>(p.x[k + l] - ox);
          ay[l] += w * static_cast<float>(p.y[k + l] - oy);
        }
      }
    }
    for (int l = 0; k < to; ++k, ++l) {
      const float w = static_cast<float>(p.w[k]);
      aw[l] += w;
      ax[l] += w * static_cast<float>(p.x[k] - ox);
      ay[l] += w * static_cast<float>(p.y[k] - oy);
    }
  }
  W = Sum8(aw);
  sx = Sum8(ax);
  sy = Sum8(ay);
}

// Second pass: second moments and residual first moments about the offset mean (mux, muy).
template <bool kSimd>
void Pass2(const Points &p, const size_t s[4], int32_t ox, int32_t oy, float mux, float muy, float out[5]) {
  float axx[kLanes] = {}, axy[kLanes] = {}, ayy[kLanes] = {}, arx[kLanes] = {}, ary[kLanes] = {};
  for (int g = 0; g < 2; ++g) {
    size_t k = s[2 * g];
    const size_t to = s[2 * g + 1];
#if VKA_HAVE_SSE2
    if constexpr (kSimd) {
      const __m128i vox = _mm_set1_epi32(ox), voy = _mm_set1_epi32(oy);
      const __m128 vmx = _mm_set1_ps(mux), vmy = _mm_set1_ps(muy);
      __m128 xx0 = _mm_loadu_ps(axx), xx1 = _mm_loadu_ps(axx + 4), xy0 = _mm_loadu_ps(axy), xy1 = _mm_loadu_ps(axy + 4),
             yy0 = _mm_loadu_ps(ayy), yy1 = _mm_loadu_ps(ayy + 4), rx0 = _mm_loadu_ps(arx), rx1 = _mm_loadu_ps(arx + 4),
             ry0 = _mm_loadu_ps(ary), ry1 = _mm_loadu_ps(ary + 4);
      for (; k + kLanes <= to; k += kLanes) {
        const __m128 w0 = LoadW(p.w + k), w1 = LoadW(p.w + k + 4);
        const __m128 dx0 = _mm_sub_ps(LoadF(p.x + k, vox), vmx), dx1 = _mm_sub_ps(LoadF(p.x + k + 4, vox), vmx);
        const __m128 dy0 = _mm_sub_ps(LoadF(p.y + k, voy), vmy), dy1 = _mm_sub_ps(LoadF(p.y + k + 4, voy), vmy);
        const __m128 wx0 = _mm_mul_ps(w0, dx0), wx1 = _mm_mul_ps(w1, dx1);
        const __m128 wy0 = _mm_mul_ps(w0, dy0), wy1 = _mm_mul_ps(w1, dy1);
        xx0 = _mm_add_ps(xx0, _mm_mul_ps(wx0, dx0));
        xx1 = _mm_add_ps(xx1, _mm_mul_ps(wx1, dx1));
        xy0 = _mm_add_ps(xy0, _mm_mul_ps(wx0, dy0));
        xy1 = _mm_add_ps(xy1, _mm_mul_ps(wx1, dy1));
        yy0 = _mm_add_ps(yy0, _mm_mul_ps(wy0, dy0));
        yy1 = _mm_add_ps(yy1, _mm_mul_ps(wy1, dy1));
        rx0 = _mm_add_ps(rx0, wx0);
        rx1 = _mm_add_ps(rx1, wx1);
        ry0 = _mm_add_ps(ry0, wy0);
        ry1 = _mm_add_ps(ry1, wy1);
      }
      _mm_storeu_ps(axx, xx0), _mm_storeu_ps(axx + 4, xx1), _mm_storeu_ps(axy, xy0), _mm_storeu_ps(axy + 4, xy1);
      _mm_storeu_ps(ayy, yy0), _mm_storeu_ps(ayy + 4, yy1), _mm_storeu_ps(arx, rx0), _mm_storeu_ps(arx + 4, rx1);
      _mm_storeu_ps(ary, ry0), _mm_storeu_ps(ary + 4, ry1);
    }
#endif
    if constexpr (!kSimd || !VKA_HAVE_SSE2) {
      for (; k + kLanes <= to; k += kLanes) {
        for (int l = 0; l < kLanes; ++l) {
          const float w = static_cast<float>(p.w[k + l]);
          const float dx = static_cast<float>(p.x[k + l] - ox) - mux;
          const float dy = static_cast<float>(p.y[k + l] - oy) - muy;
          const float wx = w * dx, wy = w * dy;
          axx[l] += wx * dx;
          axy[l] += wx * dy;
          ayy[l] += wy * dy;
          arx[l] += wx;
          ary[l] += wy;
        }
      }
    }
    for (int l = 0; k < to; ++k, ++l) {
      const float w = static_cast<float>(p.w[k]);
      const float dx = static_cast<float>(p.x[k] - ox) - mux;
      const float dy = static_cast<float>(p.y[k] - oy) - muy;
      const float wx = w * dx, wy = w * dy;
      axx[l] += wx * dx;
      axy[l] += wx * dy;
      ayy[l] += wy * dy;
      arx[l] += wx;
      ary[l] += wy;
    }
  }
  out[0] = Sum8(axx);
  out[1] = Sum8(axy);
  out[2] = Sum8(ayy);
  out[3] = Sum8(arx);
  out[4] = Sum8(ary);
}

// The line fit of one set, branch free so the batch loop below vectorises.
inline void LineKernel(float W, float N, float sxx, float sxy, float syy, float &err, float &mse, float &nx,
                       float &ny) {
  const float inv = 1.0f / W;
  const float vxx = sxx * inv, vxy = sxy * inv, vyy = syy * inv;
  const float diff = vxx - vyy, sum = vxx + vyy;
  const float hyp = std::sqrt(diff * diff + 4.0f * vxy * vxy);
  const float eig = (sum - hyp) * 0.125f;
  const float nx1 = diff - hyp, ny1 = 2.0f * vxy;
  const float nx2 = 2.0f * vxy, ny2 = (0.0f - diff) - hyp;
  const float m1 = nx1 * nx1 + ny1 * ny1, m2 = nx2 * nx2 + ny2 * ny2;
  const bool first = m1 > m2;
  const float ax = first ? nx1 : nx2, ay = first ? ny1 : ny2;
  const float length = std::sqrt(ax * ax + ay * ay);
  nx = ax / length;
  ny = ay / length;
  err = N * eig;
  mse = W > 0.0f ? eig : std::numeric_limits<float>::infinity();
}

}  // namespace

template <bool kSimd>
Centred SegmentMomentsImpl(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t blob_x0,
                          int32_t blob_y0) {
  Centred c;
  c.N = static_cast<float>((a1 - a0) + (b1 - b0));
  if (c.N == 0) return c;
  const size_t s[4] = {a0, a1, b0, b1};
  const size_t first = a0 < a1 ? a0 : b0;
  const int32_t ox = p.x[first], oy = p.y[first];
  float W, sx, sy;
  Pass1<kSimd>(p, s, ox, oy, W, sx, sy);
  c.W = W;
  if (!(W > 0.0f)) return c;
  const float inv = 1.0f / W;
  const float mux = sx * inv, muy = sy * inv;
  float m[5];
  Pass2<kSimd>(p, s, ox, oy, mux, muy, m);
  const float rx = m[3] * inv, ry = m[4] * inv;
  c.sxx = m[0] - m[3] * rx;
  c.sxy = m[1] - m[3] * ry;
  c.syy = m[2] - m[4] * ry;
  c.mx = static_cast<float>(ox - blob_x0) + (mux + rx);
  c.my = static_cast<float>(oy - blob_y0) + (muy + ry);
  return c;
}

Centred SegmentMoments(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t blob_x0,
                       int32_t blob_y0) {
  return SegmentMomentsImpl<true>(p, a0, a1, b0, b1, blob_x0, blob_y0);
}

Centred SegmentMomentsScalar(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t blob_x0,
                             int32_t blob_y0) {
  return SegmentMomentsImpl<false>(p, a0, a1, b0, b1, blob_x0, blob_y0);
}

Centred RangeMoments(const Points &p, size_t n, size_t i0, size_t i1, int32_t blob_x0, int32_t blob_y0) {
  if (i0 < i1) return SegmentMoments(p, i0, i1 + 1, 0, 0, blob_x0, blob_y0);
  return SegmentMoments(p, i0, n, 0, i1 + 1, blob_x0, blob_y0);
}

Centred SinglePoint(float x, float y, float w) {
  Centred c;
  c.W = w;
  c.N = 1;
  c.mx = x;
  c.my = y;
  return c;
}

Centred Merge(const Centred &a, const Centred &b) {
  Centred c;
  c.N = a.N + b.N;
  c.W = a.W + b.W;
  if (!(c.W > 0.0f)) return c;
  const float inv = 1.0f / c.W;
  const float dx = b.mx - a.mx, dy = b.my - a.my;
  const float f = a.W * b.W * inv;
  c.mx = a.mx + dx * (b.W * inv);
  c.my = a.my + dy * (b.W * inv);
  c.sxx = a.sxx + b.sxx + dx * dx * f;
  c.sxy = a.sxy + b.sxy + dx * dy * f;
  c.syy = a.syy + b.syy + dy * dy * f;
  return c;
}

LineFit FitLine(const Centred &c) {
  LineFit f;
  LineKernel(c.W, c.N, c.sxx, c.sxy, c.syy, f.err, f.mse, f.nx, f.ny);
  return f;
}

void FitLines(const float *VKA_RESTRICT W, const float *VKA_RESTRICT N, const float *VKA_RESTRICT sxx,
              const float *VKA_RESTRICT sxy, const float *VKA_RESTRICT syy, size_t count, float *VKA_RESTRICT err,
              float *VKA_RESTRICT mse, float *VKA_RESTRICT nx, float *VKA_RESTRICT ny) {
  for (size_t i = 0; i < count; ++i) LineKernel(W[i], N[i], sxx[i], sxy[i], syy[i], err[i], mse[i], nx[i], ny[i]);
}

void LineParams(const Centred &c, int32_t blob_x0, int32_t blob_y0, double out[4]) {
  const LineFit f = FitLine(c);
  out[0] = (static_cast<double>(blob_x0) + static_cast<double>(c.mx)) * 0.5;
  out[1] = (static_cast<double>(blob_y0) + static_cast<double>(c.my)) * 0.5;
  out[2] = f.nx;
  out[3] = f.ny;
}

namespace {

constexpr size_t kMaxWindow = 48;

// Exclusive prefix sums, within one chunk of the extended contour, of the per-point terms about the
// chunk's first point; tot holds the sums over the whole chunk.
struct Chunk {
  float P[6][kMaxWindow];
  float tot[6];
  int32_t ox, oy;
};

// Chunk c covers extended elements [c*L, (c+1)*L); element e is point (e - k) of the contour, wrapping.
void BuildChunk(const Points &p, size_t n, size_t k, size_t L, size_t c, Chunk &ch) {
  int32_t lx[kMaxWindow], ly[kMaxWindow], lw[kMaxWindow];
  const size_t src = (c * L + n - k) % n;
  const size_t head = std::min(L, n - src);
  for (size_t j = 0; j < head; ++j) {
    lx[j] = p.x[src + j];
    ly[j] = p.y[src + j];
    lw[j] = p.w[src + j];
  }
  for (size_t j = head; j < L; ++j) {
    lx[j] = p.x[j - head];
    ly[j] = p.y[j - head];
    lw[j] = p.w[j - head];
  }
  ch.ox = lx[0];
  ch.oy = ly[0];
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
  for (size_t j = 0; j < L; ++j) {
    const float w = static_cast<float>(lw[j]);
    const float dx = static_cast<float>(lx[j] - ch.ox), dy = static_cast<float>(ly[j] - ch.oy);
    const float wx = w * dx, wy = w * dy;
    ch.P[0][j] = a0;
    ch.P[1][j] = a1;
    ch.P[2][j] = a2;
    ch.P[3][j] = a3;
    ch.P[4][j] = a4;
    ch.P[5][j] = a5;
    a0 += w;
    a1 += wx;
    a2 += wy;
    a3 += wx * dx;
    a4 += wx * dy;
    a5 += wy * dy;
  }
  ch.tot[0] = a0;
  ch.tot[1] = a1;
  ch.tot[2] = a2;
  ch.tot[3] = a3;
  ch.tot[4] = a4;
  ch.tot[5] = a5;
}

// Errors of the `rows` windows starting in chunk `cur`: the rest of the chunk (its total less its
// prefix) plus the prefix of the next chunk moved to this chunk's origin.
void CombineChunk(const Chunk &cur, const Chunk &next, size_t L, size_t rows, float *VKA_RESTRICT error) {
  const float dx = static_cast<float>(next.ox - cur.ox), dy = static_cast<float>(next.oy - cur.oy);
  const float Lf = static_cast<float>(L);
  const float t0 = cur.tot[0], t1 = cur.tot[1], t2 = cur.tot[2], t3 = cur.tot[3], t4 = cur.tot[4], t5 = cur.tot[5];
  const float *VKA_RESTRICT c0 = cur.P[0], *VKA_RESTRICT c1 = cur.P[1], *VKA_RESTRICT c2 = cur.P[2],
                            *VKA_RESTRICT c3 = cur.P[3], *VKA_RESTRICT c4 = cur.P[4], *VKA_RESTRICT c5 = cur.P[5];
  const float *VKA_RESTRICT n0 = next.P[0], *VKA_RESTRICT n1 = next.P[1], *VKA_RESTRICT n2 = next.P[2],
                            *VKA_RESTRICT n3 = next.P[3], *VKA_RESTRICT n4 = next.P[4], *VKA_RESTRICT n5 = next.P[5];
  for (int r = 0; r < static_cast<int>(rows); ++r) {
    const float W = (t0 - c0[r]) + n0[r];
    const float Mx = (t1 - c1[r]) + (n1[r] + dx * n0[r]);
    const float My = (t2 - c2[r]) + (n2[r] + dy * n0[r]);
    const float Mxx = (t3 - c3[r]) + (n3[r] + 2.0f * dx * n1[r] + dx * dx * n0[r]);
    const float Mxy = (t4 - c4[r]) + (n4[r] + dx * n2[r] + dy * n1[r] + dx * dy * n0[r]);
    const float Myy = (t5 - c5[r]) + (n5[r] + 2.0f * dy * n2[r] + dy * dy * n0[r]);
    const float inv = 1.0f / W;
    const float vxx = (Mxx - Mx * Mx * inv) * inv, vxy = (Mxy - Mx * My * inv) * inv;
    const float vyy = (Myy - My * My * inv) * inv;
    const float diff = vxx - vyy;
    const float hyp = std::sqrt(diff * diff + 4.0f * vxy * vxy);
    error[r] = Lf * 0.125f * ((vxx + vyy) - hyp);
  }
}

}  // namespace

void WindowErrors(const Points &p, size_t n, size_t k, float *error) {
  const size_t L = 2 * k + 1;
  Chunk store[2];
  int cur = 0;
  BuildChunk(p, n, k, L, 0, store[cur]);
  for (size_t c = 0; c * L < n; ++c) {
    BuildChunk(p, n, k, L, c + 1, store[cur ^ 1]);
    CombineChunk(store[cur], store[cur ^ 1], L, std::min(L, n - c * L), error + c * L);
    cur ^= 1;
  }
}

bool FitArcs(const Points &p, size_t n, const uint32_t idx[4], float max_mse, Sides *out) {
  out->x0 = p.x[0];
  out->y0 = p.y[0];
  for (int k = 0; k < 4; ++k) {
    out->side[k] = RangeMoments(p, n, idx[k], idx[(k + 1) % 4], out->x0, out->y0);
    if (FitLine(out->side[k]).mse > max_mse) return false;
  }
  return true;
}

bool FitPeaks(const DetectorConfig &config, const Points &p, size_t n, Scratch &scratch, uint32_t indices[4],
              Sides *out) {
  const int32_t x0 = p.x[0], y0 = p.y[0];
  out->x0 = x0;
  out->y0 = y0;
  const size_t k = std::min<size_t>(20, n / 12);
  scratch.error.resize(n);
  float *error = scratch.error.data();
  if (k >= 1 && n > 2 * k + 1) {
    WindowErrors(p, n, k, error);
  } else {
    const Centred total = SegmentMoments(p, 0, n, 0, 0, x0, y0);
    for (size_t i = 0; i < n; ++i) {
      const Centred one = SinglePoint(static_cast<float>(p.x[i] - x0), static_cast<float>(p.y[i] - y0),
                                      static_cast<float>(p.w[i]));
      error[i] = FitLine(Merge(total, one)).err;
    }
  }

  // 7-tap filter over the errors padded with their wrap-around.
  constexpr size_t kHalf = kTaps.size() / 2;
  std::vector<float> &padded = scratch.padded;
  padded.resize(n + 2 * kHalf);
  for (size_t j = 0; j < kHalf; ++j) {
    padded[j] = error[(j + n - kHalf) % n];
    padded[n + kHalf + j] = error[j % n];
  }
  std::copy(error, error + n, padded.begin() + kHalf);
  std::vector<float> &filtered = scratch.filtered;
  filtered.resize(n + 2);
  {
    const float *VKA_RESTRICT e = padded.data();
    float *VKA_RESTRICT f = filtered.data() + 1;
    for (size_t i = 0; i < n; ++i) {
      f[i] = e[i] * kTaps[0] + e[i + 1] * kTaps[1] + e[i + 2] * kTaps[2] + e[i + 3] * kTaps[3] +
             e[i + 4] * kTaps[4] + e[i + 5] * kTaps[5] + e[i + 6] * kTaps[6];
    }
    filtered[0] = f[n - 1];
    filtered[n + 1] = f[0];
  }

  // Peaks are strict local maxima, stored negated so an ascending sort puts the strongest first.
  scratch.mask.resize(n);
  {
    const float *VKA_RESTRICT f = filtered.data();
    int32_t *VKA_RESTRICT m = scratch.mask.data();
    for (size_t i = 0; i < n; ++i) m[i] = static_cast<int32_t>(f[i + 1] > f[i]) & static_cast<int32_t>(f[i + 1] > f[i + 2]);
  }
  auto &peaks = scratch.peaks;
  peaks.clear();
  for (size_t i = 0; i < n; ++i) {
    if (scratch.mask[i]) peaks.emplace_back(-filtered[i + 1], static_cast<uint32_t>(i));
  }
  const size_t K = std::min<size_t>(peaks.size(), kNMaxima);
  if (K < 4) return false;
  std::partial_sort(peaks.begin(), peaks.begin() + static_cast<std::ptrdiff_t>(K), peaks.end(),
                    [](const auto &a, const auto &b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
  std::array<uint32_t, kNMaxima> pk{};
  for (size_t j = 0; j < K; ++j) pk[j] = peaks[j].second;
  std::sort(pk.begin(), pk.begin() + K);

  // Arcs between consecutive peaks, then every segment by merging arcs: (a, b) with a < b above the
  // diagonal of the K x K tables, the closing segment (b, a) below it.
  Centred arc[kNMaxima], pt[kNMaxima];
  for (size_t j = 0; j < K; ++j) {
    arc[j] = j + 1 < K ? SegmentMoments(p, pk[j], pk[j + 1], 0, 0, x0, y0)
                       : SegmentMoments(p, pk[j], n, 0, pk[0], x0, y0);
    pt[j] = SinglePoint(static_cast<float>(p.x[pk[j]] - x0), static_cast<float>(p.y[pk[j]] - y0),
                        static_cast<float>(p.w[pk[j]]));
  }
  constexpr size_t kCells = kNMaxima * kNMaxima;
  float sW[kCells] = {}, sN[kCells] = {}, sxx[kCells] = {}, sxy[kCells] = {}, syy[kCells] = {};
  float smx[kCells] = {}, smy[kCells] = {};
  float err[kCells], mse[kCells], nx[kCells], ny[kCells];
  const size_t cells = K * K;
  auto put = [&](size_t cell, const Centred &c) {
    sW[cell] = c.W;
    sN[cell] = c.N;
    sxx[cell] = c.sxx;
    sxy[cell] = c.sxy;
    syy[cell] = c.syy;
    smx[cell] = c.mx;
    smy[cell] = c.my;
  };
  auto get = [&](size_t cell) {
    Centred c;
    c.W = sW[cell];
    c.N = sN[cell];
    c.mx = smx[cell];
    c.my = smy[cell];
    c.sxx = sxx[cell];
    c.sxy = sxy[cell];
    c.syy = syy[cell];
    return c;
  };
  for (size_t a = 0; a < K; ++a) {
    Centred run = arc[a];
    for (size_t b = a + 1; b < K; ++b) {
      put(a * K + b, Merge(run, pt[b]));
      run = Merge(run, arc[b]);
    }
  }
  Centred tail[kNMaxima], head[kNMaxima];
  tail[K - 1] = arc[K - 1];
  for (size_t b = K - 1; b-- > 1;) tail[b] = Merge(arc[b], tail[b + 1]);
  head[1] = arc[0];
  for (size_t a = 2; a < K; ++a) head[a] = Merge(head[a - 1], arc[a - 1]);
  for (size_t b = 1; b < K; ++b) {
    for (size_t a = 0; a < b; ++a) {
      const Centred joined = a == 0 ? tail[b] : Merge(tail[b], head[a]);
      put(b * K + a, Merge(joined, pt[a]));
    }
  }
  for (size_t d = 0; d < K; ++d) sW[d * K + d] = 1.0f, sN[d * K + d] = 1.0f;
  FitLines(sW, sN, sxx, sxy, syy, cells, err, mse, nx, ny);

  const float max_mse = config.max_line_fit_mse;
  float best_error = std::numeric_limits<float>::max();
  size_t best[4] = {0, 0, 0, 0};
  for (size_t m0 = 0; m0 + 3 < K; ++m0) {
    for (size_t m1 = m0 + 1; m1 + 2 < K; ++m1) {
      if (mse[m0 * K + m1] > max_mse) continue;
      const float e01 = err[m0 * K + m1];
      const float n01x = nx[m0 * K + m1], n01y = ny[m0 * K + m1];
      for (size_t m2 = m1 + 1; m2 + 1 < K; ++m2) {
        if (mse[m1 * K + m2] > max_mse) continue;
        const float e12 = err[m1 * K + m2];
        const double dot = static_cast<double>(n01x) * nx[m1 * K + m2] + static_cast<double>(n01y) * ny[m1 * K + m2];
        if (std::fabs(dot) > config.cos_critical_rad) continue;
        for (size_t m3 = m2 + 1; m3 < K; ++m3) {
          if (mse[m2 * K + m3] > max_mse || mse[m3 * K + m0] > max_mse) continue;
          const float total = e01 + e12 + err[m2 * K + m3] + err[m3 * K + m0];
          if (total < best_error) {
            best_error = total;
            best[0] = m0, best[1] = m1, best[2] = m2, best[3] = m3;
          }
        }
      }
    }
  }
  if (best_error == std::numeric_limits<float>::max()) return false;
  if (!(static_cast<double>(best_error) < config.max_line_fit_mse * static_cast<double>(n))) return false;

  for (int j = 0; j < 4; ++j) {
    indices[j] = pk[best[j]];
    out->side[j] = get(best[j] * K + best[(j + 1) % 4]);
  }
  return true;
}

}  // namespace apriltag_vulkan::quadf32
