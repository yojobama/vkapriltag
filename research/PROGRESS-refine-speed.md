# PROGRESS: refine-speed

Branch `research/2026-11-refine-speed` (from 1879a92). Files: `library/src/RefineEdges.cpp`, `library/src/Geometry.h` (`TapAlong`), new `library/src/TapKernels.h`.
Builds: desktop `$A/build-refine-speed` (branch) and `$A/build-refine-speed-base` (1879a92 from `git archive`); Pi `~/w/refine-speed/{base,new}`.
Micro-benchmark (not committed): `$S/rsbench/bench.cpp` + `build.bat`; five synthetic blurred squares (18 to 110 px), `RefineEdges(kProfile, 2, 8)` per quad,
min of 8 passes, prints us/quad and a checksum of the corners (equal checksum = identical output). `$S/rsbench/base.exe` is 1879a92.

## 1. Instrumentation (desktop, decimation 2, vk_eval one iteration; temporary counters, removed)
| scenario | quads | passes/quad | profiles/quad | taps/profile | usable profiles | quads with no side fitted in pass 1 | level pass |
|---|---|---|---|---|---|---|---|
| one-tag | 168 | 1.05 | 298 | 26.9 | 99% | 2% | 1 |
| many-tags | 251 | 1.1 | 367 | 26.6 | 96% | 0 | 1 |
| dense-tags | 1246 | 1.6 | 180 | 24.7 | 99% | 1.4% | 16 |
| busy-background | 2402 | 2.0 | 142 | 23.1 | 85% | 2.5% | 287 |
| clutter-only | 2344 | 2.1 | 108 | 21.4 | 80% | 1% | 380 |
| jpeg | 8879 | 1.27 | 74 | 24.7 | 25% | 37% | 7148 |
| blur | 331 | 1.03 | 306 | 26.9 | 99% | 0 | 294 |
The first pass (27 taps) dominates the tap count. Skipping later passes when pass 1 fitted no side is bit-identical in principle (a narrower window
holds a subset of the same taps) but applies to 0 to 3% of quads (37% on jpeg, whose quads are cheap); small.
Phase split on the desktop micro-benchmark (MSVC, base): tap positions about 28%, gather+blend about 35%, lobe search / halfway / fit the rest.
MSVC `/Qvec-report:2`: the position loop and nothing else in `ProfilePass` vectorises (the blend loop does not: byte gathers).

## 2. SIMD blend (`TapKernels.h::BlendTaps`, SSE2 / NEON / scalar fallback)
Four taps per step: eight 16-bit loads packed into 4 lanes of [r00 r01 r10 r11], widened by shift and mask, blended in upstream's order.
Desktop micro-benchmark: 30.3 -> 27.6 us/quad, checksum identical (bit-identical on x86; NEON uses fused multiply-add as gcc contracts the scalar loop).
`TapAlong` floor via truncating cast and compare (vectorises without SSE4.1): no measurable change on MSVC (kept, output identical).
