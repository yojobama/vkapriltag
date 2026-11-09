# Progress: range recall at decimation 2 (branch research/2026-11-range-recall)

Tools: build dir `ext/avk-compare/build-range`; scripts in the scratchpad `range/` (`misses.py`, `stage.py`).
Per-frame diagnosis used temporary stderr prints in `QuadDecode::Decode` (span centroid, fit/quad flags) and
`UpdateFitQuad` (reject reason); removed before committing.

## Diagnosis (baseline, decimation 2, `range` 351/375)
Misses: all 24 are tags of 12 px (19 of 25 missed) and 18 px (5 of 25); every larger size is 25/25.
Stage where each dies (matching GPU/CPU spans to the ground-truth tag centre):

| side | tags | found | no blob (min_cluster 24 drops it) | quad fitted, decode failed | fit ok but UpdateFitQuad rejects |
|------|------|-------|----|----|----|
| 12 px | 25 | 6 | 18 | 1 | 0 |
| 18 px | 25 | 20 | 0 | 4 | 1 |

Lowering `min_cluster_pixels` (env hack) from 24 to 16/12/8/4 only moves the misses to the next stage:
with 8 every missed tag has a blob, but `UpdateFitQuad` rejects almost all of them (338 "area" rejects:
the 12 px tag is 6 decimated px, 0.75 px per cell, so the boundary points give a quad of area < 15
decimated px^2, the 0.95*(tag_width/decimation)^2 limit) and the others fit a quad too crude to decode.
Conclusion: the decimated blob is too coarse to fit; the fit has to be redone at full resolution.

## Fix: small-tag rescue (`small_tag_rescue`, `APRILTAG_VK_SMALL_RESCUE=1`, default off)
- GpuDetector: with the switch on and decimation > 1, `min_cluster_pixels` is lowered to 8.
- QuadDecode: a blob whose quad fit fails and whose bounding box is <= 40 full-resolution px is emitted as a
  `DetectedQuad` with `coarse = true` (its bounding box).
- TagDecoder: a coarse box is refitted by `FitSmallQuad` (`src/SmallQuad.cpp`): ROI around the box, Otsu
  threshold (retry at a lower threshold for grey backgrounds), largest 8-connected dark component clear of
  the ROI edge, minimum-area rectangle of its convex hull; then the normal refine + decode. A quad that
  fitted but did not decode (<= 40 px) gets the same refit and a retry. ROI is retried 1.8x larger.
- Only active at decimation > 1.
- Iterations: ROI half-size 10 px was too small (rotated tags cut by the ROI: 4 + 1 misses left); half-size
  max(0.75 * box, 10) + 4 with a 1.8x retry fixed them. 4-connectivity vs 8: no difference here.

## Results
`range` decimation 2: 351/375 -> 375/375 (12 px 6->25, 18 px 20->25). Every other scenario identical at
decimation 1 and 2 (recall, false detections, error), including family16h5 (3 false, unchanged), noise, jpeg.
Commands: `APRILTAG_VK_SMALL_RESCUE=1 bash run_corpus.sh vk_eval.exe corpus out --decimation 2` then score_corpus.py.
