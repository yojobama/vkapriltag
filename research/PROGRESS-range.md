# Progress: range recall at decimation 2 (branch research/2026-11-range-recall)

Tools: build dir `ext/avk-compare/build-range`; scripts in the scratchpad `range/` (`misses.py` per-size miss
table, `stage.py` killing stage per miss, `timing.sh` interleaved ABBA timing). Per-frame diagnosis used
temporary stderr prints in `QuadDecode::Decode` (span centroid, fit/quad flags) and `UpdateFitQuad` (reject
reason); not committed.

## Diagnosis (baseline, decimation 2, `range` 351/375)
All 24 misses are 12 px tags (19 of 25 missed) and 18 px tags (5 of 25); every larger size is 25/25.
Killing stage (GPU/CPU spans matched to the ground-truth tag centre):

| side | tags | found | no blob (min_cluster 24 drops it) | quad fitted, decode failed | fit ok but UpdateFitQuad rejects |
|------|------|-------|----|----|----|
| 12 px | 25 | 6 | 18 | 1 | 0 |
| 18 px | 25 | 20 | 0 | 4 | 1 |

Lowering `min_cluster_pixels` (env hack) 24 -> 16/12/8/4 only moves the misses to the next stage: at 8 every
missed tag has a blob but `UpdateFitQuad` rejects almost all (area < 0.95*(tag_width/decimation)^2 = 15
decimated px^2: a 12 px tag is 6 decimated px, 0.75 px per cell) and the rest fit a quad too crude to decode.
Conclusion: the decimated blob is too coarse; the fit has to be redone at full resolution.

## Fix: small-tag rescue (`small_tag_rescue`, `APRILTAG_VK_SMALL_RESCUE=1`, default off, decimation > 1 only)
- GpuDetector: `min_cluster_pixels` lowered to 8 (the pixel-count floor of labelling is what drops 12 px tags;
  lowering only select_blobs' floor left 17 of 25 missed).
- QuadDecode: spans of fewer than 24 points skip the quad fit; a span that has no quad, mean edge gradient
  (W, 1 + |grad|) >= 60 and a bounding box <= 40 px becomes a `coarse` box (boxes of one tag's fragments are
  merged within 8 px, at most 64 per frame). Spans under 64 points with mean gradient < 25 are dropped.
- TagDecoder: a coarse box is refitted by `FitSmallQuad` (`src/SmallQuad.cpp`: ROI around the box, Otsu
  threshold with a lower retry for grey backgrounds, largest 8-connected dark component clear of the ROI edge,
  minimum-area rectangle of its hull; retried in a 1.8x larger ROI), then the normal refine + decode. A quad of
  gradient >= 60 and <= 40 px that did not decode gets the same refit and retry.
- Iterations: ROI half-size 10 px was too small (4 + 1 rotated tags left); `max(0.75*box, 10) + 4` plus the 1.8x
  retry fixed them. 4- vs 8-connectivity: no difference here.

## Cost findings
- First version (all spans fitted, no gradient filter, no merge): jpeg 0.85 -> 2.9 ms, range +0.1, noise +0.12.
- Mean W separates noise from tags cleanly: noise/jpeg spans < 40 (mostly < 20), tag fragments > 70.
- Lowering min_cluster to 8 adds specks-vs-big-blob clusters: jpeg quads 13573 -> 18194 over the set, all low
  gradient (< 18); the flat-span drop removes them. Retrying every undecoded small quad was the other large
  cost (jpeg cpu 0.55 -> 1.5 ms) until gated on gradient >= 60.

## Results
`range` decimation 2: 351/375 -> 375/375 (12 px 6->25, 18 px 20->25), p95 0.100 -> 0.128 px (more tiny tags in
the set). All other scenarios unchanged at decimation 1 and 2 (recall, false detections incl. family16h5 = 3,
error), apart from run-to-run noise (2 dump lines). Switch off: identical to base.
Commands: `APRILTAG_VK_SMALL_RESCUE=1 bash run_corpus.sh vk_eval.exe corpus out --decimation 2`, score_corpus.py.
