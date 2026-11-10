# Progress: blur-corners (branch research/2026-11-blur-corners, base fb6b284)

Scratch: `S=/c/Users/yojob/AppData/Local/Temp/claude/34b51eecd9123eb4/0a677b25-5396-4a18-8327-0da17e9c41ad/scratchpad`,
outputs under `$S/blur-corners/` (helper scripts there: `run.sh NAME DEC` runs both fast sequences and scores them,
`sides.py` per-side offset/tilt, `an.py` speed table). Build dir `/c/Users/yojob/Projects/ext/avk-compare/build-blur-corners`.
Base dumps: `cbase_d{1,2}` (corpus), `sbase_d{1,2}` (fast sequences), base binary in `basebin/`.

## 1. Diagnosis (verified by instrumentation, not committed)
Counted per first-pass profile whether the lobe touches a window end (`jl == 1 || jr == taps - 2`):

| sequence | dec | first-pass profiles touching a window end | mean lobe width |
|---|---|---|---|
| fast-1tag-1280 | 2 | 54.8% | 4.06 px (window 6.5) |
| fast-1tag-1280 | 1 | 60.4% | 3.45 (window 5.5) |
| fast-2tag-640 | 2 | 66.5% | 4.54 |
| fast-2tag-640 | 1 | 77.2% | 3.58 |
| slow-1tag-1280 | 2 / 1 | 2.4% / 0.1% | 2.38 |
| medium-2tag-640 | 2 / 1 | 0.6% / 0.4% | 2.37 |

So the truncation part of the hypothesis is confirmed (sharp sequences truncate under 3%, fast ones 55-77%), and the
coarse-to-fine stage rarely helps (it only narrows). Two corrections to the hypothesis:
- Widening the window alone does not fix it. First try: doubled windows (reach 3, 6, 12) with the lobe walk extended
  across dips (bridge of 4 px): d2 fast-1tag 0.155 / 0.298 / 0.677 (from 0.360 / 1.463), d1 got worse. The tails of a
  wide lobe have gradient far below the image noise (sigma 1.5 grey; a 15 px ramp has about 6 grey per 0.5 px), so the
  5% cut no longer ends the walk: noise taps above the cut keep extending the lobe to the window end and the centroid
  collapses towards the window centre. Any thresholded lobe is wrong on wide ramps; per-side tilt errors stayed 0.2-1 px.
- The method that works needs no lobe threshold at all (below).

## 2. What was built: `LevelPass` (kWidenBlur, bit 16 of APRILTAG_VK_PROFILE_FEATURES, default mask 31)
After the normal first pass, if more than 25% of the profiles touched a window end, the sides are refitted by the
halfway crossing between plateau levels:
- profile from half a cell outside to half a cell inside each side (the middle of the quiet zone and of the border:
  both are plateaus for any ramp narrower than a cell, so neither neighbouring edge nor background is in the window);
  step 0.25 px, coarser (up to 0.5) so that at most 80 taps are used;
- white = median over the side's profiles of the outer end, black = median of the inner end;
- edge = interpolated crossing of the mid level, counted as the number of taps below it (robust to noise and to the
  plateaus of a staircase); exact for any symmetric blur; profiles whose ends are not clearly white / black are skipped;
- same weighted line fit and outlier drop as the normal pass (shared `FitSide`), two iterations to recentre the window;
- accepted only if the median 20%-80% rise is at least 2 px (sharp frames whose truncation came from noise, e.g.
  underexposed, keep the old result); when taken, coarse-to-fine is skipped.
Failed variants on the way: window outward 0.8 cell (the outer end then sits inside the quiet-zone/background ramp
for blur above about 10 px: bias +0.4 px growing with blur); black from the 10th percentile (biased low by noise:
underexposed p95 0.136 -> 0.159); both fixed in the committed code.

## 3. Results
(see the sections below; filled in as the gates are run)

Update (final): level pass uses ~20 taps (step up to 1 px), one iteration, contrast gate (3 profiles, then median
white-black >= 50) before the full sampling, ramp >= 2 px. Final numbers, tables, timings and the failed variants are in
PERFORMANCE.md section 8bd. Gates passed: corpus d1/d2 only blur and motion-blur change (also with SMALL_RESCUE+HULL_QUAD
and +POLISH); PROFILE_FEATURES=15 identical to base (corpus, fast sequences); track_eval full/key/async non-fast
sequences identical, fast ones improved; async fast-2tag recall 99.0 -> 98.0 (timing, tracked path loses a tag at 30 px/frame).
Left: fast-2tag at 25-31 px/frame (blur ~ cell, needs an overlapping-ramp model); no unit test (repo has no test tree).
Scripts in $S/blur-corners: gate.sh, timing.sh, t1.sh, t2.sh, diffs.py.
