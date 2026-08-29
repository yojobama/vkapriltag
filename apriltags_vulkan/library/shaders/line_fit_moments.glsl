// Shared struct + arithmetic for cs[] (GpuLineFitMomentsRaw, gpu/Types.h) -
// A9's prefix-summed line-fit moments, one entry per boundary point. Every
// A9 shader stage that reads or writes cs[] entries includes this, so the
// layout and Add/Sub logic exist in exactly one place. Requires
// int64_emu.glsl to already be included.
struct GpuLineFitMomentsRaw {
  int Mx;
  int My;
  int W;
  int Mxx_hi;
  uint Mxx_lo;
  int Myy_hi;
  uint Myy_lo;
  int Mxy_hi;
  uint Mxy_lo;
  int N;
};

I64 GetMxx(GpuLineFitMomentsRaw m) { return I64(m.Mxx_hi, m.Mxx_lo); }
I64 GetMyy(GpuLineFitMomentsRaw m) { return I64(m.Myy_hi, m.Myy_lo); }
I64 GetMxy(GpuLineFitMomentsRaw m) { return I64(m.Mxy_hi, m.Mxy_lo); }

// Mirrors LineFitMoments' Add/Sub (QuadDecode.cpp).
GpuLineFitMomentsRaw AddMoments(GpuLineFitMomentsRaw a, GpuLineFitMomentsRaw b) {
  GpuLineFitMomentsRaw r;
  r.Mx = a.Mx + b.Mx;
  r.My = a.My + b.My;
  r.W = a.W + b.W;
  I64 mxx = Add64(GetMxx(a), GetMxx(b));
  I64 myy = Add64(GetMyy(a), GetMyy(b));
  I64 mxy = Add64(GetMxy(a), GetMxy(b));
  r.Mxx_hi = mxx.hi;
  r.Mxx_lo = mxx.lo;
  r.Myy_hi = myy.hi;
  r.Myy_lo = myy.lo;
  r.Mxy_hi = mxy.hi;
  r.Mxy_lo = mxy.lo;
  r.N = 0;
  return r;
}

GpuLineFitMomentsRaw SubMoments(GpuLineFitMomentsRaw a, GpuLineFitMomentsRaw b) {
  GpuLineFitMomentsRaw r;
  r.Mx = a.Mx - b.Mx;
  r.My = a.My - b.My;
  r.W = a.W - b.W;
  I64 mxx = Sub64(GetMxx(a), GetMxx(b));
  I64 myy = Sub64(GetMyy(a), GetMyy(b));
  I64 mxy = Sub64(GetMxy(a), GetMxy(b));
  r.Mxx_hi = mxx.hi;
  r.Mxx_lo = mxx.lo;
  r.Myy_hi = myy.hi;
  r.Myy_lo = myy.lo;
  r.Mxy_hi = mxy.hi;
  r.Mxy_lo = mxy.lo;
  r.N = 0;
  return r;
}

// Port of FitLineError (QuadDecode.cpp) - float-precision throughout, same
// as the original (every intermediate there is `static_cast<float>` despite
// the function's `double` return type). `length(vec2(...))` substitutes for
// hypotf: not the same overflow-avoiding scaling hypotf uses internally,
// acceptable here since these magnitudes never approach float32's range and
// this stage isn't held to bit-exactness (see int64_emu.glsl's ToFloat64
// comment).
float FitLineError(int N, int Mx, int My, int W, I64 Mxx, I64 Myy, I64 Mxy) {
  I64 Cxx = Sub64(Mul64By32(Mxx, W), Mul32x32To64(Mx, Mx));
  I64 Cxy = Sub64(Mul64By32(Mxy, W), Mul32x32To64(Mx, My));
  I64 Cyy = Sub64(Mul64By32(Myy, W), Mul32x32To64(My, My));

  // Cxx and Cyy individually can be enormous while nearly equal (a
  // near-straight run of boundary points), so the difference/sum MUST be
  // formed in exact 64-bit arithmetic before ever rounding to float32 -
  // exactly mirroring the CPU's `static_cast<float>(Cxx - Cyy)`, which
  // subtracts in int64_t first. Converting Cxx and Cyy to float
  // independently and subtracting the floats (an earlier version of this
  // function did that) reintroduces the catastrophic cancellation this
  // whole int64 emulation exists to avoid.
  float diff_f = ToFloat64(Sub64(Cxx, Cyy));
  float sum_f = ToFloat64(Add64(Cxx, Cyy));
  // Doubled in exact 64-bit arithmetic before rounding, matching the CPU's
  // `static_cast<float>(2 * Cxy)` (2*Cxy computed as int64_t, then cast) -
  // same cancellation-avoidance reasoning as diff_f/sum_f above.
  float cxy2_f = ToFloat64(Add64(Cxy, Cxy));

  float hypot_cached = length(vec2(diff_f, cxy2_f));
  float eight_w_squared = float(W) * float(W) * 8.0;
  float eig_small = (sum_f - hypot_cached) / eight_w_squared;
  return float(N) * eig_small;
}
