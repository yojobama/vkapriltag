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

## 3. Accuracy of the kept step 2 (desktop, decimation 2, `$A/rs-out/{base_d2,new_d2}`)
`ctest` 44/44 pass. All 26 scenario dumps byte-identical to the base build except one frame of `noise` (about 0.003 px, the known GPU noise).

## 4. Experiments with temporary env knobs (registered in Options.cpp as "experiment", to be removed or finalised)
`APRILTAG_VK_RS_SPACING1` (first-pass profile spacing, default 1.5), `RS_MAXN` (cap on profiles per side), `RS_ABORT` (stop after pass 1 when a side has fewer than max(3, f*np) usable profiles).
Commands: `bash $S/rs_acc.sh <vk_eval.exe> <outdir> 2 ENV=...` then `python $S/cmpn/sum.py $A/napa-corpus out.json <dirs>`.
| variant | found | false | err med/p95 | verdict |
|---|---|---|---|---|
| base | 7146 | 1 | 0.017/0.099 | |
| MAXN=40 | 7146 | 1 | 0.023/0.111 | dropped (error up) |
| SPACING1=2.0 | 7145 | 1 | 0.019/0.103 | dropped (error up, one tag lost) |
| ABORT=0 (a side under 3 usable profiles) | 7146 | 1 | 0.017/0.099 | dumps equal to base up to noise; timing pending |
| ABORT=0.25 | 7146 | 1 | 0.017/0.099 | dumps equal to base up to noise; timing pending |

## 5. Pi perf of the first kernel (commit 9b18962, decimation 2, `perf record` whole process, `~/w/refine-speed/{base,new}`)
ProfilePass event share, base -> new (absolute = share x total events): one-tag 43.3% -> 39.6% (231M -> 206M, -11%); dense-tags 66.3% -> 66.3% (751M -> 712M, -5%);
busy-background 57.8% -> 54.0% (-10%); jpeg 37.2% -> 34.9% (-9%); clutter 50.7% -> 48.0%; many-tags 55.8% -> 51.4%. Short of the 25% target, so the next steps:
perf by source line (new): TapKernels lane build/loads about 24% of ProfilePass, TapAlong about 16%, scalar BilinearAt for the 1 to 3 tail taps about 12%, gradient loop about 20%, lobe/halfway about 8%.

## 6. Second round (commits cf9ed0b..ffcfea9)
NEON: tap offsets (y * stride + x) computed four at a time; the tail of a profile re-blends the last four taps instead of scalar BilinearAt; `GradientPeak` (vector subtract and max, then the first index of the maximum: same result as the scalar scan);
`TapAlong` keeps `std::floor` where it is native (aarch64, SSE4.1), the compare trick only for baseline SSE2.
Desktop micro-benchmark: 30.5 -> 26.5 us/quad, checksum identical.
Abort experiment: `APRILTAG_VK_REFINE_ABORT=f` (registered; default 0.25 after the gate below): in the first pass, half way along a side with usable profiles < f * profiles seen, or at the end of a side with fewer than max(3, f * np), the quad is left unrefined and the later passes are skipped.
Corpus, decimation 2 (found/false/err med/p95): base 7146/1/0.017/0.099; f=0.25 (end of side) identical; f=0.25 with the half-way check identical (`h25_1`); f=0.5 found 7147 (low-contrast +1) but low-contrast p95 0.0514 -> 0.0539 (over the 0.002 limit); f=0.75 breaks border-occlusion (error median 0.20 -> 0.31, p95 1.08 -> 11.8): dropped. Decimation 1 with f=0.25: 7129/0/0.017/0.096 vs base 7129/0/0.017/0.097.
Dropped: profile spacing 2.0 on the first pass (err 0.019/0.103, one tag lost), capping profiles per side at 40 (err 0.023/0.111).
Desktop timing of the abort is inside the noise (shared machine; cpu mean 0.124 vs 0.126 vs 0.119 ms for off / 0.25 / 0.4); jpeg -6 to -11%.
