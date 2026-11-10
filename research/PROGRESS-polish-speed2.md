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
