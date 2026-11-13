#include "vkapriltag/gpu/CpuDecimate.h"

#include <algorithm>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#define VKA_DECIMATE_NEON 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define VKA_DECIMATE_SSE2 1
#endif

namespace apriltag_vulkan {

namespace {

// Scalar box mean of outputs first..count-1 of one row, any decimation.
void BoxRowScalar(const uint8_t *src, uint32_t stride, uint32_t d, uint32_t first, uint32_t count, uint8_t *out) {
  const uint32_t area = d * d;
  const uint32_t half = area / 2u;
  for (uint32_t i = first; i < count; ++i) {
    uint32_t sum = 0;
    for (uint32_t y = 0; y < d; ++y) {
      const uint8_t *p = src + y * stride + i * d;
      for (uint32_t x = 0; x < d; ++x) sum += p[x];
    }
    out[i] = static_cast<uint8_t>((sum + half) / area);
  }
}

uint32_t BoxRow2(const uint8_t *src, uint32_t stride, uint32_t count, uint8_t *out) {
  uint32_t i = 0;
#if VKA_DECIMATE_NEON
  for (; i + 16 <= count; i += 16) {
    const uint8_t *p = src + i * 2;
    uint16x8_t a0 = vpaddlq_u8(vld1q_u8(p));
    uint16x8_t a1 = vpaddlq_u8(vld1q_u8(p + 16));
    a0 = vpadalq_u8(a0, vld1q_u8(p + stride));
    a1 = vpadalq_u8(a1, vld1q_u8(p + stride + 16));
    vst1q_u8(out + i, vcombine_u8(vrshrn_n_u16(a0, 2), vrshrn_n_u16(a1, 2)));
  }
#elif VKA_DECIMATE_SSE2
  const __m128i low = _mm_set1_epi16(0x00FF);
  const __m128i two = _mm_set1_epi16(2);
  auto sums = [&](const uint8_t *p) {
    const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + stride));
    __m128i s = _mm_add_epi16(_mm_add_epi16(_mm_and_si128(r0, low), _mm_srli_epi16(r0, 8)),
                              _mm_add_epi16(_mm_and_si128(r1, low), _mm_srli_epi16(r1, 8)));
    return _mm_srli_epi16(_mm_add_epi16(s, two), 2);
  };
  for (; i + 16 <= count; i += 16) {
    const uint8_t *p = src + i * 2;
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), _mm_packus_epi16(sums(p), sums(p + 16)));
  }
#endif
  return i;
}

uint32_t BoxRow4(const uint8_t *src, uint32_t stride, uint32_t count, uint8_t *out) {
  uint32_t i = 0;
#if VKA_DECIMATE_NEON
  for (; i + 8 <= count; i += 8) {
    const uint8_t *p = src + i * 4;
    uint16x8_t a0 = vpaddlq_u8(vld1q_u8(p));
    uint16x8_t a1 = vpaddlq_u8(vld1q_u8(p + 16));
    for (uint32_t y = 1; y < 4; ++y) {
      a0 = vpadalq_u8(a0, vld1q_u8(p + y * stride));
      a1 = vpadalq_u8(a1, vld1q_u8(p + y * stride + 16));
    }
    const uint16x4_t m0 = vrshrn_n_u32(vpaddlq_u16(a0), 4);
    const uint16x4_t m1 = vrshrn_n_u32(vpaddlq_u16(a1), 4);
    vst1_u8(out + i, vmovn_u16(vcombine_u16(m0, m1)));
  }
#elif VKA_DECIMATE_SSE2
  const __m128i low = _mm_set1_epi16(0x00FF);
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i eight = _mm_set1_epi32(8);
  auto sums = [&](const uint8_t *p) {
    __m128i s = _mm_setzero_si128();
    for (uint32_t y = 0; y < 4; ++y) {
      const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + y * stride));
      s = _mm_add_epi16(s, _mm_add_epi16(_mm_and_si128(r, low), _mm_srli_epi16(r, 8)));
    }
    return _mm_srli_epi32(_mm_add_epi32(_mm_madd_epi16(s, ones), eight), 4);
  };
  for (; i + 8 <= count; i += 8) {
    const uint8_t *p = src + i * 4;
    const __m128i m = _mm_packs_epi32(sums(p), sums(p + 16));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out + i), _mm_packus_epi16(m, m));
  }
#endif
  return i;
}

void BoxRow(const uint8_t *src, uint32_t stride, uint32_t d, uint32_t count, uint8_t *out) {
  uint32_t done = 0;
  if (d == 2u) {
    done = BoxRow2(src, stride, count, out);
  } else if (d == 4u) {
    done = BoxRow4(src, stride, count, out);
  }
  BoxRowScalar(src, stride, d, done, count, out);
}

void CopyRow(const uint8_t *src, uint32_t d, uint32_t count, uint8_t *out) {
  for (uint32_t i = 0; i < count; ++i) out[i] = src[i * d];
}

}  // namespace

void CpuDecimate(const uint8_t *src, uint32_t width, uint32_t height, uint32_t decimation, bool box, bool wide,
                 void *dst) {
  const uint32_t dw = width / decimation;
  const uint32_t dh = height / decimation;
  std::vector<uint8_t> row(wide ? dw : 0);
  for (uint32_t dy = 0; dy < dh; ++dy) {
    const uint8_t *s = src + static_cast<size_t>(dy) * decimation * width;
    uint8_t *out = wide ? row.data() : static_cast<uint8_t *>(dst) + static_cast<size_t>(dy) * dw;
    if (box) {
      BoxRow(s, width, decimation, dw, out);
    } else {
      CopyRow(s, decimation, dw, out);
    }
    if (wide) {
      uint32_t *o = static_cast<uint32_t *>(dst) + static_cast<size_t>(dy) * dw;
      for (uint32_t i = 0; i < dw; ++i) o[i] = row[i];
    }
  }
}

}  // namespace apriltag_vulkan
