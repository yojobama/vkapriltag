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
