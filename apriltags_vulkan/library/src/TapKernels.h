#pragma once

// Bilinear blend of many taps at once (internal, not installed). The scalar loop is the reference; the
// SSE2 and NEON kernels blend four taps per step in the same operation order.

#include <cstdint>
#include <cstring>

#include "Geometry.h"

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define VKAPRILTAG_TAP_SSE2 1
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VKAPRILTAG_TAP_NEON 1
#endif

namespace apriltag_vulkan::geom {

// out[m] = BilinearAt(im, xis[m], yis[m], fxs[m], fys[m]) for m in [0, n); every tap's 2x2 block must lie
// inside the image.
template <typename Img>
inline void BlendTaps(const Img *im, const int *xis, const int *yis, const float *fxs, const float *fys, int n,
                      float *out) {
  int m = 0;
#if defined(VKAPRILTAG_TAP_SSE2) || defined(VKAPRILTAG_TAP_NEON)
  const uint8_t *data = im->data;
  const ptrdiff_t stride = im->stride;
  [[maybe_unused]] auto pair = [&](int k) {
    const uint8_t *row0 = data + yis[k] * stride + xis[k];
    uint16_t a, b;
    std::memcpy(&a, row0, 2);
    std::memcpy(&b, row0 + stride, 2);
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
  };
#endif
#if defined(VKAPRILTAG_TAP_SSE2)
  const __m128i low = _mm_set1_epi32(0xFF);
  const __m128 one = _mm_set1_ps(1.0f);
  for (; m + 4 <= n; m += 4) {
    const __m128i v = _mm_set_epi32(static_cast<int>(pair(m + 3)), static_cast<int>(pair(m + 2)),
                                    static_cast<int>(pair(m + 1)), static_cast<int>(pair(m)));
    const __m128 p00 = _mm_cvtepi32_ps(_mm_and_si128(v, low));
    const __m128 p01 = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(v, 8), low));
    const __m128 p10 = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(v, 16), low));
    const __m128 p11 = _mm_cvtepi32_ps(_mm_srli_epi32(v, 24));
    const __m128 fx = _mm_loadu_ps(fxs + m), fy = _mm_loadu_ps(fys + m);
    const __m128 gx = _mm_sub_ps(one, fx), gy = _mm_sub_ps(one, fy);
    __m128 acc = _mm_mul_ps(_mm_mul_ps(gx, gy), p00);
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(fx, gy), p01));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(gx, fy), p10));
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(fx, fy), p11));
    _mm_storeu_ps(out + m, acc);
  }
#elif defined(VKAPRILTAG_TAP_NEON)
  const uint32x4_t low = vdupq_n_u32(0xFF);
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; m + 4 <= n; m += 4) {
    int32_t off[4];
    vst1q_s32(off, vmlaq_n_s32(vld1q_s32(xis + m), vld1q_s32(yis + m), im->stride));
    uint32_t lanes[4];
    for (int k = 0; k < 4; ++k) {
      uint16_t a, b;
      std::memcpy(&a, data + off[k], 2);
      std::memcpy(&b, data + off[k] + stride, 2);
      lanes[k] = static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
    }
    const uint32x4_t v = vld1q_u32(lanes);
    const float32x4_t p00 = vcvtq_f32_u32(vandq_u32(v, low));
    const float32x4_t p01 = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(v, 8), low));
    const float32x4_t p10 = vcvtq_f32_u32(vandq_u32(vshrq_n_u32(v, 16), low));
    const float32x4_t p11 = vcvtq_f32_u32(vshrq_n_u32(v, 24));
    const float32x4_t fx = vld1q_f32(fxs + m), fy = vld1q_f32(fys + m);
    const float32x4_t gx = vsubq_f32(one, fx), gy = vsubq_f32(one, fy);
    float32x4_t acc = vmulq_f32(vmulq_f32(gx, gy), p00);
    acc = vfmaq_f32(acc, vmulq_f32(fx, gy), p01);
    acc = vfmaq_f32(acc, vmulq_f32(gx, fy), p10);
    acc = vfmaq_f32(acc, vmulq_f32(fx, fy), p11);
    vst1q_f32(out + m, acc);
  }
#endif
  if (m < n && n >= 4 && m > 0) {
    // The last four taps again: the tap values do not depend on their neighbours.
    BlendTaps(im, xis + n - 4, yis + n - 4, fxs + n - 4, fys + n - 4, 4, out + n - 4);
    return;
  }
  for (; m < n; ++m) out[m] = BilinearAt<float>(im, xis[m], yis[m], fxs[m], fys[m]);
}

// g[j] = prof[j + 1] - prof[j - 1] for j in [1, taps - 2]; *best is the largest g above zero (0 if none) and
// *bj the first j reaching it (-1 if none).
inline void GradientPeak(const float *prof, int taps, float *g, float *best, int *bj) {
  float top = 0;
  int j = 1;
#if defined(VKAPRILTAG_TAP_SSE2)
  __m128 vmax = _mm_setzero_ps();
  for (; j + 4 <= taps - 1; j += 4) {
    const __m128 v = _mm_sub_ps(_mm_loadu_ps(prof + j + 1), _mm_loadu_ps(prof + j - 1));
    _mm_storeu_ps(g + j, v);
    vmax = _mm_max_ps(vmax, v);
  }
  float lanes[4];
  _mm_storeu_ps(lanes, vmax);
  for (int k = 0; k < 4; ++k) top = std::max(top, lanes[k]);
#elif defined(VKAPRILTAG_TAP_NEON)
  float32x4_t vmax = vdupq_n_f32(0.0f);
  for (; j + 4 <= taps - 1; j += 4) {
    const float32x4_t v = vsubq_f32(vld1q_f32(prof + j + 1), vld1q_f32(prof + j - 1));
    vst1q_f32(g + j, v);
    vmax = vmaxq_f32(vmax, v);
  }
  top = vmaxvq_f32(vmax);
#endif
  for (; j < taps - 1; ++j) {
    g[j] = prof[j + 1] - prof[j - 1];
    top = std::max(top, g[j]);
  }
  *best = top;
  *bj = -1;
  if (top > 0) {
    int k = 1;
    while (g[k] != top) ++k;
    *bj = k;
  }
}

}  // namespace apriltag_vulkan::geom
