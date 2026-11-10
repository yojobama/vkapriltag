# Polish speed, second pass (branch research/2026-11-polish-speed2)

Base: 01b0a6a (polish at 0.43 ms per ordinary tag). Fast path behind `APRILTAG_VK_POLISH_FAST=1`
(plus `APRILTAG_VK_POLISH=1`); goal is to make it the only path once it matches the old accuracy.

## What the code contains (commits b9501e6, 4d84678)

- SIMD-friendly dot-product normal equations over structure-of-arrays columns (`EvaluateFast`, `Dot4`).
- Blur scan on edge-sampled pixels (`BuildScan`) instead of a bounding-box candidate list.
- Fused working-set build (`BuildFit`: one pass over the box, window freeze fused in), first model pass merged
  with the derivative pass, 2x2 closed-form model path, `APRILTAG_VK_POLISH_REACH` knob (freeze reach).
- Phase profiler: `APRILTAG_VK_POLISH_PROF=1` prints us/tag per phase at exit.
- Optional slope handling (`APRILTAG_VK_POLISH_SLOPES=0`, `_SLOPE_T`): illumination slopes refitted linearly
  instead of in the Gauss-Newton system.

## Tools (scratch dir `$S/polish-speed2/`)

- `build.sh`; `run.sh NAME ENV=..` runs the corpus at d1 and d2 and scores it; `one.sh "scenarios" ENV=..`
  quick median/p95; `c.sh DEC base cand..` diff table; `timing.sh "scenarios" "--threads 1" rounds` ABBA
  off/old/new timings (`FASTENV`, `NEWEXE`); `prof.sh [scenario]` phase profile.
- Exes in the build dir: `vk_eval_old.exe` = 01b0a6a PolishFit.cpp (pre-task polish).

## Measurements at the start of this session (fast path as committed, d2, one thread)

Accuracy (median / p95): identical to old polish in every scenario except motion-blur (p95 0.166 -> 0.219,
d1 0.115 -> 0.159) and small gains elsewhere (blur 0.065 -> 0.052). Recall and false detections unchanged.
Slopes refitted linearly (SLOPES=0): uneven-light p95 0.039 -> 0.058 at T=0.9, 0.044 at 0.97, 0.039 at 1.0
but overexposed 0.047 -> 0.066 at 1.0; rejected, saving is only about 7 %.

Frame ms, off / old / fast (2 ABBA rounds): one-tag 0.31 / 0.76 / 0.48; dense-tags 1.76 / 15.9 / 7.7;
motion-blur 0.36 / 3.7 / 1.6. Per tag in dense-tags: old 0.47 ms, fast 0.21 ms.

Phase profile, dense-tags, us per tag (noisy machine, 190 to 240 total): select 48, freeze+lev 21,
model 45, accumulate 51 (gather 28, dots 23), scan 11, cand 6. 729 pixels per tag, 4.3 evaluations.

## Experiments this session

- ITERS=3: accuracy neutral, small saving; ITERS=2: dense p95 0.041 -> 0.046 (rejected); TOL=0.007 neutral.
- BuildFit computing the gradient magnitudes only for pixels that survive the interior test: no measurable
  change in the select phase.
- Found the motion-blur p95 regression of the fast path (0.166 -> 0.219 at d2): the edge-sampled blur scan
  (offsets +-1.2 px) misjudges wide blur; with the old candidate scan (`OLDSCAN` hack) it is 0.166. Scaling the
  scan offsets (OFFS, px multiplier of the old offsets) d2 p95: 1 -> 0.219, 2 -> 0.195, 3 -> 0.166, 3.5 -> 0.144,
  4 -> 0.115, 5 -> 0.102, 6 -> 0.118, 8 -> 0.142; blur p95 0.052 (1), 0.070 (2.5), 0.065 (3.5 to 6), 0.087 (8);
  overexposed 0.047 (1 to 3.5), 0.051 (4 to 6). Kept 3.5 (offsets -4.2, -1.4, 1.4, 4.2 px): every scenario
  within 5 % of old polish at d1 and d2, motion-blur better (d2 0.166 -> 0.144, d1 0.115 -> 0.108).
- Band rasterisation in BuildFit (per row intervals from the monotone u(x), v(x)): slower (select 58 -> 80 us)
  because the bands still cover about 60 % of the box and the cost is the per-candidate exact test; reverted.
- Double precision dot products: no change in motion-blur p95, so float accumulation is not the problem.
- Fast path is now the only path (`APRILTAG_VK_POLISH=1` alone): slow Evaluate, BuildCand, Select, Cand, the
  slope experiment and its knobs removed. Output equals the previous fast+offsets run to float rounding (a few
  dump lines differ in the last digit).
- Struct-of-arrays Jacobian columns weighted in place (no gather): accumulate phase 51 -> 45 us/tag.
- Motion stage: 131 of 142 tags in motion-blur accept the motion model, with 2000 to 8000 pixels each (wide windows),
  so the box model (pixels x taps) dominates (2.4 ms of 3.6 ms per tag). Fewer taps (spacing 2.4 to 3.2 px, max 4 to 5):
  accuracy ok but no clear time gain. Motion fit on a strided subset of at most 800 pixels followed by one
  Gauss-Newton iteration on the full set: motion-blur p95 unchanged (d1 0.108 -> 0.114, d2 0.144 -> 0.144), median
  unchanged (the full-set final iteration is needed: without it the median rose 0.022 -> 0.035). Frame time
  3.4 -> 2.65 ms (noisy machine).
- Lattice thinning by 2 for blurred tags (best blur >= 1.2 px): p95 unchanged in every scenario except blur d2
  0.065 -> 0.068 (+4.6 %); motion-blur 3.4 -> 2.2 ms. Sharp tags (dense-tags, small-tags, range) never use it.
- ITERS=3 with TOL=0.007: overexposed p95 0.048 -> 0.056 (+17 %), rejected; defaults stay 6 / 0.004.

## Final state (HEAD of this branch)

Fast path is the only path; `APRILTAG_VK_POLISH=1` alone selects it. Profiler removed. Accuracy tables:
`$S/polish-speed2/final_table.txt` (off | old | final, median/p95 and found, d1 and d2). No scenario worse than old
polish by more than 5 % of p95 (blur d2 0.065 -> 0.068 is the worst, +4.6 %; overexposed d1 0.047 -> 0.050,
+6 % but 0.003 px); motion-blur better (d1 0.115 -> 0.104, d2 0.166 -> 0.142); recall and false detections identical.

Frame ms (4 ABBA rounds, median, decimation 2) off / old polish / new polish:

| scenario | 1 thread | default threads |
|---|---|---|
| one-tag | 0.31 / 0.67 / 0.43 | 0.29 / 0.67 / 0.40 |
| dense-tags (30 tags) | 1.52 / 14.26 / 6.80 | 0.58 / 2.46 / 1.49 |
| motion-blur | 0.39 / 3.70 / 1.86 | 0.35 / 3.16 / 1.58 |
| blur | 0.40 / 2.15 / 0.82 | 0.35 / 1.36 / 0.60 |
| jpeg | 2.41 / 3.16 / 2.56 | 0.76 / 1.35 / 1.00 |
| range | 0.16 / 0.70 / 0.36 | 0.18 / 0.68 / 0.38 |
| noise | 0.39 / 1.26 / 0.66 | 0.36 / 0.79 / 0.48 |

Polish overhead per ordinary tag (dense-tags, 1 thread): old 0.42 ms, new 0.18 ms (2.4x). Overhead ratios old/new:
one-tag 3.1x, motion-blur 2.2x, blur 4.3x, jpeg 4.9x, range 2.7x, noise 3.2x. Target of 3 to 4x per ordinary
tag (0.1 to 0.15 ms) not reached; motion-blur 1.5 ms overhead per frame (about 1.5 tags per frame there) not under target.

## Left to do

- Per ordinary tag (about 200 us profiled): select 60 (stage 1 box scan 25, per-candidate window test 35),
  model 50, accumulate 45, freeze + first model + levels 24, scan 13. Ideas: clip rows to the quad before the
  interior test; vectorise the candidate test; variable projection (8 parameters) to shrink the normal equations.
- Motion stage: closed-form segment-blurred edge for single-edge pixels (about 43 % of motion pixels), taps for corners.
- Orange Pi (aarch64) timing not measured here (desktop only).
