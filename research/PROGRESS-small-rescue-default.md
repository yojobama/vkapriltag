# Progress: making the small-tag rescue cheap enough for default-on (branch research/2026-11-small-rescue-default)

Result: default left OFF. The cost was cut roughly in half but busy-background (+0.05 ms, +11%), clutter-only (+0.03), dense-tags,
small-tags and family16h5 (+0.02 to +0.03) stay above the gate (5% or 0.02 ms).
Tools (scratchpad `small-rescue-default/`): `run.sh NAME DEC`, `var.sh DEC ROUNDS scen,.. name=ENV ..` (interleaved variants, forward then
reverse per round), `vsum.py`, `cmp.py A.score B.score`.

## Measurements
- Box features dumped for 13 range rescues versus 9441 boxes (clutter-only, busy-background, family16h5), decimation 2: quiet-zone ring light
  fraction, dark fraction, Otsu bimodality, gap, Otsu threshold, size. Quiet zone does not separate (range tags have rings from 0.0 to 0.94,
  some on grey backgrounds). Only corpus-specific signals separate (Otsu threshold, bimodality: clutter is crisp binary), judged overfitting.
  Size is weakly informative (positives 8-24 px, median 14; negatives median 9, 72% under 12).
- Cost split on clutter-only d2 (off 0.457): on 0.564; no refits of boxes 0.525; no retry refits of undecoded quads 0.538; neither 0.480.
  So box refits about 0.04, undecoded-quad retries about 0.03, fixed about 0.02.
- Lower `min_cluster_pixels`: 24 -> 12 loses 4 range tags, 16 loses 7, 10 loses 2; 8 needed (375).
- Empty scenario +0.05 ms with the rescue: about 3 flat noise specks per frame (9-21 points, mean gradient 2-3) reach the device-side tail
  (frames with no blob skip it). Raising `min_white_black_diff` to 8 when the rescue is on removes them (empty 0.205 -> 0.136); 12 and 20 do
  the same. Whole corpus d2 unchanged (family16h5 false detections 3 -> 2).
- Note: getenv calls in per-quad/per-span paths cost enough to double the measured rescue cost on clutter-only: keep instrumentation out of timings.

## Changes (all behind APRILTAG_VK_SMALL_RESCUE=1, decimation > 1)
1. GpuDetector: `min_white_black_diff` raised to at least 8.
2. QuadDecode: boxes and undecoded small quads that qualify for the full-resolution refit are scored (-|side - 14| - 0.05 * gradient), and
   only the best 4 per frame are refitted (box dropped, or retry disabled by clearing `gradient`). Deterministic ordering (score, then index).
Corpus d1 and d2, on and off: scores identical to the 01b0a6a base (range 375/375 at d2) apart from family16h5 false 3 -> 2; no cap-related change
even at 2 refits (the corpus never mixes tags and clutter in one frame, so the cap risk is untested: a real tag in a frame with more than 4
better-scored candidates would be lost).

## Timing, decimation 2, median ms, off -> on (ABBA, 4 rounds, 20 iterations, 3 warmup; noise about +-0.03)
Before (3 rounds, only min diff 8): clutter-only +0.090, busy-background +0.109, family16h5 +0.059, decoys +0.034, empty -0.041.
After (cap 4): see table in the final report; failing the gate: busy-background +0.053, clutter-only +0.030, dense-tags +0.031,
small-tags +0.028, family16h5 +0.024, low-contrast +0.021 (several of these are within noise); range +0.019 (excluded).
Cap sweep (clutter-only / busy / family16h5 off 0.487/0.486/0.340): no cap 0.575/0.591/0.408; 8: 0.504/0.526/0.384; 4: 0.487/0.502/0.383; 2: 0.484/0.513/0.366.

## Left
- The fixed cost (about 0.02 ms: more blobs and points through the tail with min_cluster 8, per-span loop) needs device-side work: filter
  low-gradient small blobs before the point readback, or do the 8-24 px blobs in a separate cheap pass.
- A better tag-likeness score needs real-world clutter; the synthetic corpus clutter is too clean to tune on.
- Orange Pi timing not run (desktop target not met).
