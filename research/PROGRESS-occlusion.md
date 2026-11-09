# Progress log: occlusion recall (branch research/2026-11-occlusion-recall)

Tools (scratchpad `occlusion/`, not in repo): ana.py (per-stage miss table from the APRILTAG_VK_DIAG dump),
plot.py, crop.py, run.sh. Diagnostic hooks marked DIAG-TEMP in QuadDecode.cpp / TagDecoder.cpp.

## Diagnosis (base, border-occlusion, 230 tags)
Stage that kills each tag (APRILTAG_VK_DIAG; quad matched to gt within a quarter side):

| stage | d2 | d1 |
|---|---|---|
| found | 99 | 54 |
| quad fitted, decode failed | 32 | 14 |
| blob present, no quad (fit_failed) | 80 | 145 |
| no blob near the tag | 19 | 17 |

Cause of fit_failed (plots of the blob's points):
- white bar: it cuts the black ring, so the quiet-zone white joins the payload whites; the black blob's boundary then
  also contains the inner (payload) outline, interleaved by angle with the outer outline, so no side is a line.
- black bar: protrudes from the corner into the quiet zone; the corner is on the bar tip.
Both leave the convex hull's long edges equal to the real sides.

## Experiment 1: convex hull quad (HULL_QUAD)
FitQuadHull (QuadDecode.cpp): monotone-chain hull of the blob's points, Douglas-Peucker (1.2 px), choose the 4 edges
(near-perpendicular neighbours, corner within 0.4 edge length of the edge ends, >= 60% of the perimeter), refit each side
on the points within 2.5 px of its edge, then the usual UpdateFitQuad. Only runs for blobs with no quad from the normal fit;
the extra quad goes to the decoder (decode verifies). Switch: DetectorConfig::hull_quad, APRILTAG_VK_HULL_QUAD=1.
Result (whole corpus, both decimations): border-occlusion 99 -> 157 (d2), 54 -> 158 (d1); no other scenario changes.
