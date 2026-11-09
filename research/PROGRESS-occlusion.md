# Progress log: occlusion recall (branch research/2026-11-occlusion-recall)

Tools (scratchpad `occlusion/`, not in repo): ana.py (per-stage miss table from the APRILTAG_VK_DIAG dump),
errs.py (per-tag corner error), plot.py/plot2.py/crop.py (pictures), run.sh, timing.sh.
Commands: `APRILTAG_VK_HULL_QUAD=1 bash run.sh NAME DEC [scenario...]` runs the corpus and scores it;
`bash timing.sh DEC ROUNDS scenario...` is the interleaved ABBA timing (off/on).

## Diagnosis (base, border-occlusion, 230 tags)
Stage that kills each tag (diag dump of every blob and quad; quad matched to gt within a quarter side):

| stage | d2 | d1 |
|---|---|---|
| found | 99 | 54 |
| quad fitted, decode failed | 32 | 14 |
| blob present, no quad (fit failed) | 80 | 145 |
| no blob near the tag | 19 | 17 |

Cause of the fit failures (plots of the blob's points): the generator's bar crosses one corner.
- white bar: it cuts the black ring, so the quiet-zone white joins the payload whites; the black blob's boundary then
  also contains the inner (payload) outline, interleaved by angle with the outer outline, so no side is a line.
- black bar: protrudes from the corner into the quiet zone; DP or peaks put the corner on the bar tip (corner
  error 20-60 px), and the decode may fail or the corner stays wrong.
- At d1 the bar is twice as wide in pixels and the boundary has twice the points, so the DP/peak fit fails more often
  (145 vs 80 blobs); this is the d1 vs d2 difference, not blur.
- Of the 54 quads that were fitted but undecoded at d2/d1, about 20 have exact corners but 3-5 wrong payload bits
  (bar over the payload: beyond 36h11's correction), the rest have a corner on the bar tip.

## Experiment 1: convex hull quad (HULL_QUAD), final form
FitQuadHull (QuadDecode.cpp), run for a blob with no valid quad, or whose quad leaves more than 2% of the boundary
points farther than 2 px from its sides (OutlierFraction): monotone-chain hull, Douglas-Peucker (1.2 px), choose 4 edges
(neighbours near-perpendicular, corner within 0.4 edge length of the edge ends, >= 60% of the perimeter), then for each
side the line of greatest support (1-px histogram of offsets over +-10 degrees, 1 degree steps; a hull edge can be a
chord over a black bar's tip) refitted on the points within 1.5 px, then the usual UpdateFitQuad. The extra quad
(dropped if it equals the normal one within 2 px) goes to the decoder, which verifies it.
Switch: DetectorConfig::hull_quad, APRILTAG_VK_HULL_QUAD=1.

First version (side = hull edge, band 2.5 px) gave recall 157/158 but 80 tags with corner error 10-57 px (the chord
over the bar tip); the support histogram fixed that.

Result (whole corpus, both decimations, versus base dumps): only border-occlusion changes.

| border-occlusion | recall | median px | p95 px |
|---|---|---|---|
| d2 base | 43.0% (99) | 0.204 | 1.084 |
| d2 hull | 68.3% (157) | 0.056 | 0.682 |
| d1 base | 23.5% (54) | 0.448 | 1.624 |
| d1 hull | 68.7% (158) | 0.040 | 0.863 |

All 25 other scenarios identical to base (recall, false detections, error) at both decimations.
