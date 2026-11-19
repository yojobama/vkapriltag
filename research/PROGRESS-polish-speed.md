# Progress log: speeding up the whole-tag polish (branch research/2026-11-polish-speed)

Build dir `ext/avk-compare/build-polish-speed`; scratch `$S/polish-speed` (run.sh = corpus run + score with NAME and ENV
arguments, one.sh = quick per-scenario scores, prof.sh, timing.sh = interleaved timing, cmp.py, tagdiff.py,
`vk_eval_old.exe` = build of the previous polish, dumps `off_d{1,2}`, `old_d{1,2}`, `new_d{1,2}`).
Corpus runs: `DECS=2 bash run.sh NAME APRILTAG_VK_POLISH=1`; timing: `bash timing.sh` (ABBA-style rounds, median of 3).

## 1. Where the time went (old polish, one thread, per tag)

dense-tags (about 1700 px per tag): stage 1 (build set, four blur candidates) 1.1 ms, stage 2 0.2 ms, LM 1.1 ms. About
100 to 150 ns per pixel per model evaluation, generic loops, 16x16 normal equations in double, four Jacobian passes plus
rejected trial steps, building the working set twice by scanning the whole bounding box. Motion stage 11 to 25 ms per tag.
The polish is already parallel across tags (TagDecoder worker pool ParallelFor).

## 2. What was changed (all in `PolishFit.cpp`, rewritten; same model, same rejection rules)

1. One candidate pass over the bounding box (row-wise float projection, cheap reject of pixels whose window cannot touch
   an edge) producing an array of candidate pixels; the blur scan, the final working set and any window change select
   from it. The lattice thinning of large tags is applied while scanning.
2. Blur scan on about 160 sampled pixels, candidates 0.7, 1.2, 2.0 (3.5 only if 2.0 won), levels kept from the best.
3. Frozen per-pixel cell windows (columns and rows that matter, fixed at the start mapping), float arithmetic, a LUT for
   the normal CDF and density, outer boundaries of the window not evaluated, a closed-form single-edge path (one
   boundary, one axis) for most pixels.
4. Joint Gauss-Newton/LM over homography, blur, levels and slopes (16 columns) with float block accumulation; a
   block-coordinate version (levels solved separately) converged too slowly on clipped tags and was dropped.
5. Narrower working set: only pixels within `0.8 sigma + 0.3` px (minimum 0.5) of a class-changing edge, widened for
   blurred tags by `2 (sigma - 1.2)`. Old: `2.5 sigma + 0.3`, minimum 1.8. Pixel count per ordinary tag 1700 to 730 with
   no accuracy change (the pixels far from an edge carry no geometry information).
6. Motion blur: the 8 start directions and the full-set refit are gone. The direction comes from the per-axis fitted
   widths (projected extent along each cell axis, two sign candidates tested by closed-form SSE), then one LM with a
   Gaussian plus a straight segment of 3 to 9 taps (each tap Gaussian widened by spacing^2/12). A separable "box per
   axis" model was tried first: fine for axis-aligned motion, no gain for diagonal motion (p95 0.33), so dropped.
7. Cached knob reads (std::getenv per tag costs microseconds on Windows).

Failed or not useful: sampling the Hessian from every third pixel (accuracy collapses: dense-tags p95 0.050 at
stride 3), reusing the Jacobian for later iterations (needs 3 full passes anyway, gain small, accuracy worse),
a uniform pixel cap instead of edge-band selection (large tags lose badly, see section 3 of PROGRESS-corners.md).

## 3. Accuracy (median / p95 px, decimation 2; `cmp.py off_d2 old_d2 new_d2`)

| scenario | off | old polish | new polish |
|---|---|---|---|
| blur | 0.045 / 0.162 | 0.015 / 0.065 | 0.017 / 0.065 |
| motion-blur | 0.059 / 0.456 | 0.018 / 0.167 | 0.023 / 0.166 |
| overexposed | 0.069 / 0.294 | 0.020 / 0.053 | 0.018 / 0.047 |
| underexposed | 0.065 / 0.142 | 0.049 / 0.116 | 0.051 / 0.122 |
| noise | 0.034 / 0.079 | 0.026 / 0.063 | 0.026 / 0.063 |
| jpeg | 0.031 / 0.070 | 0.026 / 0.052 | 0.028 / 0.055 |
| range | 0.016 / 0.100 | 0.009 / 0.048 | 0.010 / 0.045 |
| dense-tags | 0.018 / 0.074 | 0.010 / 0.042 | 0.011 / 0.041 |
| small-tags | 0.021 / 0.080 | 0.011 / 0.038 | 0.012 / 0.041 |
| border-occlusion (d1) | 0.448 / 1.624 | 0.100 / 1.142 | 0.114 / 1.261 |

Decimation 1 is the same picture (see `new_d1.score`). Recall and false detections are identical to the unpolished
run in every scenario at both decimations. Worst relative p95 versus old polish: small-tags +8 %, border-occlusion d1
+10 %, underexposed +5 %, jpeg +6 %; no scenario is worse than without polish.

## 4. Pixel-window trade-off (d2 p95; window = mult * sigma + 0.3, min px)

| mult / min | px per dense tag | dense | small | range | underexposed | jpeg |
|---|---|---|---|---|---|---|
| 2.5 / 1.8 (old) | 1700 | 0.042 | 0.038 | 0.048 | 0.116 | 0.052 |
| 1.2 / 0.8 | 961 | 0.041 | 0.041 | 0.047 | 0.116 | 0.055 |
| 0.8 / 0.5 (default) | 729 | 0.041 | 0.041 | 0.045 | 0.122 | 0.055 |
| 0.6 / 0.3 | 611 | 0.041 | 0.043 | 0.046 | 0.122 | 0.057 |
| 0.4 / 0.2 | 494 | 0.045 | 0.046 | 0.051 | 0.123 | 0.066 |

## 5. Time (see timing.txt for the interleaved medians; ms per frame)

Interleaved (3 rounds, median), decimation 2, ms per frame, off / old polish / new polish:

| scenario | 1 thread | default threads |
|---|---|---|
| one-tag | 0.31 / 2.02 / 0.67 | 0.31 / 1.99 / 0.64 |
| dense-tags (about 30 tags) | 1.60 / 81.1 / 14.2 | 0.59 / 12.8 / 2.42 |
| motion-blur | 0.40 / 26.0 / 4.24 | 0.36 / 20.8 / 3.28 |
| blur | 0.71 / 11.9 / 2.25 | 0.37 / 7.94 / 1.38 |
| jpeg | 2.23 / 7.10 / 2.89 | 0.73 / 3.65 / 1.33 |
| range | 0.17 / 3.14 / 0.72 | (not completed) |

Per ordinary tag (dense-tags): old 2.7 ms, new 0.43 ms (6.4x). Polish overhead per frame: one-tag 1.7 -> 0.36 ms (4.7x),
motion-blur 25.6 -> 3.8 ms (6.7x), blur 11.2 -> 1.5 ms (7.5x), jpeg 4.9 -> 0.65 ms (7.5x), range 3.0 -> 0.55 ms (5.5x).
Target of 10x not reached. Remaining per-pixel cost (bench, one thread): model 30 to 40 ns, with derivatives 40 to 50 ns,
full evaluation with the 16x16 normal equations 70 to 100 ns; about 4 evaluations per tag.

## 6. Dev knobs

`APRILTAG_VK_POLISH_WINMULT`, `_MINWIN`, `_BLUREXTRA`, `_SCANN`, `_TOL`, `_ITERS`, `_N` (pixel cap), `_DEV` (8 = no
motion stage, 16 = polish thinned tags too).

## 7. Left to do

Motion stage still about 5 ms per blurred tag (taps times pixels); a thinner set for the motion fit or fewer taps would
help. The accumulate step could skip the slope columns unless the scene has gradients. Default-on decision: see report.
