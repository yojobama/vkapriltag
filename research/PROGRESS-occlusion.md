# Progress log: occlusion recall (branch research/2026-11-occlusion-recall)

Tools (scratchpad `occlusion/`, not in the repo): ana.py (per-stage miss table from a diagnostic dump), errs.py (per-tag
corner error), plot.py/plot2.py/crop.py (pictures), run.sh, timing.sh. The diagnostic hooks (APRILTAG_VK_DIAG etc.) are in
commit 3373f2f and were removed afterwards.
Commands: `APRILTAG_VK_HULL_QUAD=1 bash run.sh NAME DEC [scenario...]` runs the corpus and scores it;
`bash timing.sh DEC ROUNDS scenario...` is the interleaved ABBA timing (switch off/on).

## Diagnosis (base, border-occlusion, 230 tags)
Stage that kills each tag (every blob and quad dumped; a quad counts for a tag when within a quarter side):

| stage | d2 | d1 |
|---|---|---|
| found | 99 | 54 |
| quad fitted, decode failed | 32 | 14 |
| blob present, no quad (fit failed) | 80 | 145 |
| no blob near the tag | 19 | 17 |

Cause of the fit failures (plots of the blob's points). The generator's bar crosses one corner, in black, white or grey:
- white bar: it cuts the black ring, so the quiet-zone white joins the payload whites; the black blob's boundary then also
  contains the inner (payload) outline, interleaved by angle with the outer outline, so no side is a line.
- black bar: protrudes from the corner into the quiet zone; DP or peaks put the corner on the bar tip (corner error
  20-60 px); the decode may fail or the corner stays wrong.
- d1 versus d2: the bar is twice as wide in pixels and the boundary has twice the points; the DP/peak fit fails more often
  (145 vs 80 blobs). It is not blur: those blobs exist at both.
- Of the 32/14 fitted-but-undecoded quads, about 20 have exact corners but 3-5 wrong payload bits (bar over the payload,
  beyond 36h11's correction); the rest have a corner on the bar tip.
- Obstruction (payload cells only covered, border intact): the failures are pure payload damage, not geometry; no change.
- 19/17 "no blob": a grey bar the colour of the background removes the border's corner (see crop of frame 1). Raising
  fill_max to 6 did not recover them (not the blob filter); not investigated further.
- SIDE_ONE_OUT, profile-fit outlier drop and third-bit act after the quad exists, so they cannot help the fit failures.

## Experiment 1: convex hull quad (HULL_QUAD)
`FitQuadHull` (QuadDecode.cpp). Runs for a blob with no valid quad, or whose quad leaves more than 2% of the boundary points
farther than 2 px from its sides (`HasOutliers`, early exit); only blobs with at least 600/decimation points and a bounding
box side of at least 20 decimated pixels. Monotone-chain hull, Douglas-Peucker (1.2 px), choose the 4 edges (neighbours
near-perpendicular, corner within 0.4 edge length of the edge ends, >= 60% of the perimeter), then per side the line of
greatest support (1-px histogram of offsets over +-10 degrees in 1 degree steps), refitted on the points within 1.5 px, then
the usual UpdateFitQuad. The extra quad (dropped if equal to the normal one within 2 px) goes to the decoder, which verifies.
Switch: DetectorConfig::hull_quad, APRILTAG_VK_HULL_QUAD=1 (default off).

History: first version (side = hull edge, band 2.5 px) gave recall 157/158 but 80 tags with corner error 10-57 px: for a
black bar the hull edge is a chord over the bar tip, the real side is inside the hull. The support histogram fixed that.
Tried and dropped: hull of every 3rd point (recall 158 -> 124), Akl-Toussaint octagon pre-filter (no gain).

Result (whole corpus, both decimations, versus base dumps): only border-occlusion changes; with the switch off the corpus
scores are identical to base.

| border-occlusion | recall | median px | p95 px |
|---|---|---|---|
| d2 base | 43.0% (99) | 0.204 | 1.084 |
| d2 hull | 68.3% (157) | 0.056 | 0.682 |
| d1 base | 23.5% (54) | 0.448 | 1.624 |
| d1 hull | 68.7% (158) | 0.040 | 0.863 |

Remaining misses with the hull (d1): ~54 fitted quads whose payload is too damaged (most with exact corners, 3-5 bit errors),
17 no-blob, 1 fit failure. NAPA gets 56.5% (130/230).

## Cost
CPU time in jpeg (thread-summed, d2, 150 frames): normal fit 165 ms, hull 180 ms (hull work runs on ~9000 failing large blobs).
Interleaved wall medians are in the final report (timing5.txt in the scratchpad).
