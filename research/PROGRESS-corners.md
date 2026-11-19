# Progress log: corner accuracy in degraded scenes (branch research/2026-11-corner-accuracy)

Build dir `ext/avk-compare/build-corners`; scratch `$S/corners` (diag.py, run.sh, score dumps).
Helper scripts (in `$S/corners/`): `diag.py ROOT DUMPDIR [scenarios]` breaks the corner error of a dump down
into side position / side tilt components; `run.sh NAME ENV=VAL...` runs the corpus at decimation 1 and 2
(`DECS=2` for one) and writes `NAME_d{1,2}.score`.

## 1. What the error is made of (baseline, decimation 2, profile fit)

For every matched tag the detected corners are compared with gt.json. Per side the two end points' offset along
the side's outward normal gives a mean (side position) and a difference (side tilt, i.e. the line's angle):

| scenario     | median | p95  | side pos rms | side tilt (end difference) rms | reading |
|--------------|--------|------|--------------|-------------------------------|---------|
| motion-blur  | 0.059  | 0.45 | 0.039        | 0.178                          | tilt, 4.5x the position error: more than variance |
| overexposed  | 0.069  | 0.29 | 0.093        | 0.037                          | position: per-tag isotropic offset (sd 0.09, mean -0.03): the clip-bias estimate |
| blur         | 0.045  | 0.16 | 0.021        | 0.065                          | tilt 3.1x pos = pure variance (a straight-line fit gives sqrt(12)=3.5x) |
| underexposed | 0.065  | 0.14 | 0.030        | 0.065                          | variance (low contrast, sigma 3-6) |
| noise        | 0.034  | 0.08 | 0.010        | 0.034                          | variance, ratio 3.4 |
| jpeg         | 0.031  | 0.07 | 0.010        | 0.037                          | variance |
| range        | 0.016  | 0.10 | 0.019        | 0.128                          | tilt 6.7x: small tags (12-54 px), corner contamination + few profiles |
| one-tag      | 0.010  | 0.04 | 0.004        | -                              | reference |

So almost everything is estimator variance in the side angle (the corner error is dominated by the two sides'
tilts, not their positions), except overexposed (a per-tag bias that varies in size) and motion blur (extra
systematic tilt). The signed bias (mean outward offset) is ~0 in all but overexposed: no scenario has a simple
constant bias. A per-side line fit from ~1.5 px spaced profiles cannot do much better in tilt than it does; the
remaining lever is using many more edge pixels: every cell edge of the tag, i.e. a whole-tag photometric fit.

## 2. Whole-tag photometric polish (`src/PolishFit.cpp`, `APRILTAG_VK_POLISH=1`, default off)

Runs after decode on every decoded detection; the known code gives the full cell grid. Model per pixel:
blurred coverage of black / white / outside cells (separable Gaussian CDFs in tag coordinates, widths from the
local homography so the blur is isotropic in the image), levels (black, white, outside) fitted by linear LS,
blur widths per axis fitted, pixel footprint included in sigma, sensor clipping (saturated pixels that the model
also puts beyond the range are dropped), Huber weights, Levenberg-Marquardt on the 8 homography parameters. Cells
whose centre reads far from the level their code bit implies (corrected bits, covered cells; only when hamming > 0)
are marked unknown and the pixels touching them dropped. Result is rejected (detection unchanged) if the cost
does not drop or a corner moves more than max(0.5, min(2, 0.3 cell)) px.

First version (all edge pixels up to a 6000 cap), decimation 2, median / p95:

| scenario | base | polish |
|---|---|---|
| noise | 0.034 / 0.079 | 0.017 / 0.039 |
| blur | 0.045 / 0.162 | 0.012 / 0.050 |
| overexposed | 0.069 / 0.294 | 0.014 / 0.042 |
| underexposed | 0.065 / 0.142 | 0.030 / 0.061 |
| jpeg | 0.031 / 0.070 | 0.021 / 0.047 |
| motion-blur | 0.059 / 0.456 | 0.054 / 0.285 |
| range | 0.016 / 0.100 | 0.009 / 0.050 |
| one-tag | 0.009 / 0.038 | 0.007 / 0.025 |
| obstruction | 0.009 / 0.040 | 0.010 / 0.039 (after unknown-cell masking; 0.028 / 0.086 without) |
| uneven-light | 0.012 / 0.051 | 0.009 / 0.023 |

Recall and false detections unchanged (the polish only moves corners).
Per-tag cost of the first version: 13 ms single-thread for a 170 px tag (one-tag, 6000 px working set): far too
slow; see the optimisation notes below.

(Notes on experiments: illumination slope terms (multiplicative gradient) gave no change on uneven-light and were
removed; the unknown-cell masking is what fixes obstruction.)

## 3. Later versions

- Motion blur: when the fitted blur is wide (> 1.4 px) the model is refitted with a straight blur segment (mx, my)
  plus a small Gaussian, from the best of eight start directions; accepted when its cost is 3 % lower. Alone this
  took motion-blur p95 (d2, all pixels) from 0.285 to 0.063.
- Illumination slope (multiplicative linear gain, 2 parameters): needed once all pixels are used (uneven-light was
  0.065 / 0.185 at N = infinity without it, 0.007 / 0.023 with it); at a pixel cap the noise hid the bias.
- Accuracy is set by the number of edge pixels N in the working set (error ~ 1/sqrt(N)). With N capped by a uniform
  lattice thinning, large tags got worse than the profile fit (cap 3000: large-tags 0.046 / 0.102 vs 0.009 / 0.043).
  Hence: the working set is capped at APRILTAG_VK_POLISH_N (default 8000, about a 135 px tag), and a tag that would
  need thinning keeps its profile fit unless it is blurred (sigma >= 1.6 px) or has > 5 % saturated pixels in the set
  (there the profile fit is worse than any thinned polish).
- Cost reductions: candidate blur scan in one pass each (levels and SSE in closed form), merged LM passes, max 8
  iterations, stop at 0.01 px, narrower window (2.5 sigma). One-tag (170 px), one thread: 13 ms -> 1.3 ms at N = 1500,
  3.8 ms at N = 6000 (minus 0.44 ms without the polish).
- Dev knobs: APRILTAG_VK_POLISH_DEV bit 8 = no motion model, bit 16 = polish thinned tags too; APRILTAG_VK_POLISH_N.

Results with the committed defaults (N 8000), whole corpus, median / p95 px, base -> polish
(`$S/corners/base_d{1,2}.score`, `n8000_d{1,2}.score`): see section 4.

## 4. Final state (commit after "motion stage on a thinner set") and measurements

Motion stage: eight start directions scored by closed-form SSE on a working set thinned to 2500 px, LM there, then 3
LM iterations on the full set; at most 13 taps. motion-blur frame cost 81 -> 25 ms (one thread).

Whole corpus, median / p95 px, base (v1.10.1 profile fit) -> `APRILTAG_VK_POLISH=1`; recall and false detections are
identical in every scenario at both decimations (dumps `$S/corners/fin_d{1,2}`, scores `fin_d{1,2}.score`):

| scenario | d2 base | d2 polish | d1 base | d1 polish |
|---|---|---|---|---|
| motion-blur (max) | 0.059 / 0.456 (1.15) | 0.018 / 0.167 (0.70) | 0.066 / 0.434 (1.66) | 0.020 / 0.116 (0.34) |
| overexposed | 0.069 / 0.294 | 0.020 / 0.053 | 0.067 / 0.294 | 0.020 / 0.053 |
| blur | 0.045 / 0.162 | 0.015 / 0.065 | 0.039 / 0.138 | 0.017 / 0.057 |
| underexposed | 0.065 / 0.142 | 0.049 / 0.116 | 0.060 / 0.136 | 0.043 / 0.119 |
| noise | 0.034 / 0.079 | 0.026 / 0.063 | 0.031 / 0.069 | 0.025 / 0.057 |
| jpeg | 0.031 / 0.070 | 0.026 / 0.052 | 0.030 / 0.067 | 0.026 / 0.056 |
| range | 0.016 / 0.100 | 0.009 / 0.048 | 0.017 / 0.129 | 0.009 / 0.047 |
| border-occlusion | 0.204 / 1.084 | 0.116 / 1.084 | 0.448 / 1.624 | 0.100 / 1.142 |
| dense-tags | 0.018 / 0.074 | 0.010 / 0.042 | 0.018 / 0.075 | 0.010 / 0.042 |
| small-tags | 0.021 / 0.080 | 0.011 / 0.038 | 0.020 / 0.075 | 0.011 / 0.038 |
| one-tag, large-tags, many-tags | unchanged or slightly better (large tags keep the profile fit) | | | |

noise / jpeg / underexposed gain less because the cap (8000 px) and the sensor noise limit them; with the cap
raised to all pixels (APRILTAG_VK_POLISH_N=100000, d2) noise 0.014 / 0.035, jpeg 0.016 / 0.041, underexposed
0.024 / 0.056, motion-blur 0.010 / 0.063 (about 10x the time on large tags).

Cost, one thread, per frame (interleaved ABBA, `$S/corners/timing.txt`; shared GPU, desktop CPU):
| scenario | off | on |
|---|---|---|
| one-tag (170 px tag) | 0.36 ms | 1.96 ms |
| jpeg | 2.2 ms | 7.1 ms |
| motion-blur | 0.4 ms | 25 ms |
| dense-tags (~30 small tags) | 1.5 ms | 75 ms (2.5 ms per tag) |
All threads (default): dense-tags 1.0 -> 17 ms, jpeg 1.1 -> 4.9, one-tag 0.37 -> 2.7, motion-blur 0.4 -> 83 (before
the motion-stage speed-up; about 25 now single-threaded).
The polish runs only on decoded tags (after decode), never on rejected quads.

Left to do / ideas: per-pixel cost (float and wider SIMD for the 16-parameter normal equations, an analytic fast
path for pixels that touch one column and one row only); an adaptive cap by predicted accuracy instead of a tag
size cut-off; border-occlusion p95 1.08 is one tag whose polish is rejected (corner moves too far); the gt-clip
scenario (overexposed) is now unbiased.
