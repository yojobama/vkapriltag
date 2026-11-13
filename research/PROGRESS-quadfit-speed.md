# Progress: quadfit-speed

Branch research/2026-11-quadfit-speed (from 1879a92). Base build: $A/build-quadfit-speed-base (built, untouched source).

## State log
- Read QuadDecode.cpp, OwnDecode.cpp, BorderSample.cpp, TagDecoder steps. Recompute audit (TagDecoder): the unrefined
  sample is shared (SharedSample) between margin gate and pre-refine check; the inner-square and final decodes use
  different quads (after refine), so the homography is not recomputed for the same quad except when refine leaves it unchanged.
  TryThirdBit reuses c.raw (no resample). ReadBorder runs on the unrefined quad (precheck) and in TrySideOneOut twice.
- Plan: (1) DP path on unpacked int32 arrays (exact in int32: coords < 2^14); (2) peaks path; (3) ReadBorder median.
- Exp 1 (a5fc8a8): DP path on unpacked int32 arrays (x, y, w in QuadFitScratch; sq dist and cross product exact in int32).
  Desktop d2 corpus dumps vs base: identical apart from noise.txt (base itself flips between two outputs, new gives the same two).
  Dumps: $A/qf-dumps (base_d2, base2_d2, new_d2). Pi build: ~/w/quadfit-speed/build_{base,new} (-g -fno-omit-frame-pointer).
  Timing scripts: scratchpad qf_dtime.sh (desktop), ~/w/qf_time.sh (Pi). Results pending.
- Uncommitted next: peaks path computes lineparam23 only for ascending segment pairs.

## Measurements (Pi, decimation 2, jpeg scenario, 3 iterations; QFSTAT temporary instrumentation in QuadDecode.cpp)
Baseline perf share (jpeg): ProfilePass 34.6%, FitQuadForBlob 29.4%, SampleFamily 9.8%, memcpy 3.3%, FitLine 3.0%, ReadBorder 2.5%,
MedianInPlace 1.9%, ComputeHomography 1.4%, Decode 1.0%.
QFSTAT jpeg (45508 blobs, 8.84 M points; DP falls back on 6607 = 14.5% of blobs but they average 596 points):
  stage ns (after DP-unpack change only): peaks prefix 21.9 M, windowed error 62.6 M, filter+peaks 51.7 M, segment lines 19.8 M,
  search 6.7 M, UpdateFitQuad 14.9 M, whole FitQuadForBlob 311 M: peaks is about 53% of the fit, the DP try the rest.
Other scenarios (busy-background / clutter-only / dense-tags / noise / one-tag): fallbacks 10.5 / 11.4 / 41% (260 of 4252 not;
  1748 peaks entries counted incl. early fail) / 4 / 17%; FitQuadForBlob 39 / 39 / 11 / 8 / 4.7 M ns.
Exp 2: DP unpack alone: perf cycles in FitQuadForBlob unchanged (0.93e9 base vs 0.92e9), wall `cpu` medians within noise on both machines
  (desktop 2 rounds: jpeg 0.366 -> 0.375 ms; Pi 5 rounds: jpeg 1.04 -> 1.20 but base itself ranged 1.04-1.23 between runs: cpu median is too noisy for this; use QFSTAT/perf).
Exp 3 (a1b79d7): peaks windowed error in two stages (int64 moments to float arrays, then vector sqrt/div), padded 7-tap filter, partial_sort
  instead of stable_sort (same total order: value then index). jpeg: filter+peaks 51.7 -> 26.8 M ns, windowed error 62.6 -> 57.7, whole fit 311 -> 245 M ns (-21%).
  Desktop dumps, default and APRILTAG_VK_QUADFIT=peaks, jpeg/busy-background/noise/dense-tags: identical to base (noise.txt flips as base does).
Exp 4: ArgMax as a vectorisable max pass plus first-index search (replaces the 4-lane tracking version). Result pending.
