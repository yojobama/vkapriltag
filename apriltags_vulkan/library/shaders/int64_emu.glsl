// Signed 64-bit integer emulation for A9 (GPU quad fit), needed because
// Mali-G610 reports shaderInt64=false (see DeviceCaps::has_shader_int64).
// A value is a two's-complement 64-bit integer split as (hi: high 32 bits,
// signed; lo: low 32 bits, unsigned) - reconstructible on the host as
// (int64_t(hi) << 32) | uint32_t(lo).
//
// Only the operations FitQuadForBlob's moment prefix sum
// (QuadDecode.cpp: Mx/My/W/Mxx/Mxy/Myy accumulation) actually needs are
// provided: a 32x32->64 signed multiply, a 64x32->64 truncated multiply-
// widen (for Mxx += wx * x2, where wx is already 64-bit), and 64-bit add.
// All arithmetic is exactly mod 2^64 (truncating), matching what every real
// C++ compiler does for int64_t overflow in practice - this is a faithful
// bit-for-bit port, not an approximation.
struct I64 {
  int hi;
  uint lo;
};

// a * b (both 32-bit signed), exact 64-bit result. imulExtended is core
// GLSL (since 4.00) - no int64 support required, just correctly-rounded
// 32x32->64 signed multiply split into two 32-bit halves.
I64 Mul32x32To64(int a, int b) {
  int hi;
  uint lo;
  imulExtended(a, b, hi, lo);
  return I64(hi, lo);
}

// a * b, where a is 64-bit and b is 32-bit signed, truncated to 64 bits -
// matches C++'s `int64_t * int32_t` (b promoted to int64_t) assigned back
// into an int64_t. Only the low 64 bits of the true (up to 96-bit) product
// are needed, so this reduces to three 32x32 partial products:
//   a * b = a.lo*b.lo + (a.lo*b.hi + a.hi*b.lo) * 2^32   (mod 2^64)
// b's sign-extended high word (b.hi) only ever contributes through that
// cross term, and only its low 32 bits survive the mod-2^64 truncation.
I64 Mul64By32(I64 a, int b) {
  uint b_lo = uint(b);
  uint b_hi = (b < 0) ? 0xFFFFFFFFu : 0u;
  uint p_hi, p_lo;
  umulExtended(a.lo, b_lo, p_hi, p_lo);
  uint cross = a.lo * b_hi + uint(a.hi) * b_lo;
  return I64(int(p_hi + cross), p_lo);
}

// a + b, mod 2^64.
I64 Add64(I64 a, I64 b) {
  uint lo = a.lo + b.lo;
  uint carry = (lo < a.lo) ? 1u : 0u;
  int hi = a.hi + b.hi + int(carry);
  return I64(hi, lo);
}

// Low 32 bits of a 64-bit value, reinterpreted as signed - matches C++'s
// `int32_t(some_int64_t)` narrowing conversion (bit-pattern preserving).
int Trunc32(I64 a) {
  return int(a.lo);
}
