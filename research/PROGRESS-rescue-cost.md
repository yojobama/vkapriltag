# Progress log: cost of small-tag rescue and hull quad (branch research/2026-11-rescue-cost)

Tools (scratchpad `rescue-cost/`, not in the repo): `run.sh NAME DEC` (corpus run plus score, env passes through),
`timing.sh DEC ROUNDS scenario...` (interleaved ABBA, `--iterations 20 --warmup 3`, A = both switches off,
B = both on; `EXE=` picks another binary), `summ.py` (medians), `stats.sh`, the `*.py` analysis scripts.
Temporary instrumentation (global counters, stderr logs per box, per hull attempt, per decoded quad) was used for the
diagnosis and removed; the measurements are below.

## Diagnosis (decimation 2, both switches on, one pass over each scenario, 150 frames)

Small rescue (clutter-only): 4167 coarse boxes (28 per frame) go to `FitSmallQuad`; 138 produce a quad, none decodes.
Almost all cost is the refit itself (thread-summed 600 ms of 606 ms for the box path). Per box the ladder is: Otsu
region (<= 57 px), low threshold retry, larger region (<= 89 px), low threshold retry. 3374 of 4167 boxes ran all four
attempts, every one ending in "every dark component touches the region edge"; 12 817 region fits in total.
`range`: 13 boxes, 13 decode (9 on the first attempt, 4 on a retry).
- Box features do not separate tags from clutter: `range` tag fragments have n 8-58 points, mean gradient 71-160, box side
  4-16; clutter boxes n 8-130, gradient 60-277 (median 136), 72% under 12 px side. A gradient upper bound removes about a
  third only. The clutter marks are checker cells (3-13 px) and rectangle corners, i.e. squares like tiny tags.
- Edge-ring dark fraction (share of the region's border pixels at or below the threshold) separates the retries:
  `range` retries that succeed have ring 0.0 in 6 of 11 cases, clutter retries are 0.15-0.5.
- With the low-threshold retry and the larger region removed completely (ring limit -1) the whole corpus at decimation 2
  is identical to the old output (range 375/375): none of the corpus relies on those stages.

Hull quad:
- jpeg d2: 17 464 hull attempts in 150 frames, 4723 pass the size gates, 4529 build a polygon, only 332 reach the line
  support stage (84 histogram passes over the side's points, about 1 ms each, 70% of the hull time); 0 of them decode.
  Of the pre-stage the sort is 100 of 160 ms, the monotone chain 43 ms.
- Candidates that need the hull (hull quad decodes, normal quad does not) in border-occlusion: 3 (d2) / 2 (d1) with a
  quad, the rest (about 150) are blobs with no quad. In jpeg the tags that decode have 0% outliers; the 1589 (d2)
  candidates with a quad are small (40-80 px diagonal) junk quads with 8-48% outliers.
- Cheap gates tested offline against border-occlusion (needed) versus jpeg (waste): bounding-box aspect >= 0.6 keeps all
  needed and removes 28% of jpeg d2 / 27% of d1; the octagon fill (extreme points in 8 directions) does not separate
  (needed 0.5-0.84, jpeg 0.05-0.71); n per bounding-box perimeter does not separate.
- Cap by frame: candidates ranked by point count, at most K hull fits per frame. border-occlusion recall with K=8,4,2
  is unchanged (157/230 at d2, 158/230 at d1, same p95); K=1 loses 11 (d2) and 23 (d1) tags. jpeg d2 has about 31
  candidates per frame, so K=6 removes about 80% of the attempts.

## Changes (all in the on-state of the two existing switches; off output unchanged)
1. `SmallQuad.cpp`: flood fill on a padded blocked-mask with reused thread-local buffers (edge-connected components
   flooded first without storing pixels). Output identical to the old labelling.
2. `SmallQuad.cpp`: the low-threshold retry and both larger-region fits run only when at most 5% of the region's edge
   pixels are dark (`kRetryRingMax`).
3. `QuadDecode.cpp`: the hull fit is run on at most 6 candidates per frame (largest blobs first, ties by blob order), in
   a second parallel pass (`kMaxHullAttempts`).
Whole corpus at d1 and d2 with both switches on: scores identical to the version before these changes (recall,
false detections, median, p95 per scenario).

4. `QuadDecode.cpp` hull support sweep in float, bounding-box aspect gate 0.6 (no tag lost in border-occlusion; jpeg
   candidates past the gates -28%). No measurable timing change beyond noise.

## Timing (interleaved ABBA, 4 rounds, 20 iterations, 3 warmup; median ms per frame, off versus both on)
Before (binary at 956d502 merge, `timing_before*.txt`) -> after (`timing_after2.txt`):

| scenario | dec | off | on before | on after | delta before | delta after |
|---|---|---|---|---|---|---|
| clutter-only | 2 | 0.45 | 1.010 | 0.611 | +123% | +30% |
| busy-background | 2 | 0.50 | 0.972 | 0.672 | +97% | +38% |
| jpeg | 2 | 0.76 | 0.913 | 0.907 | +21% | +13% |
| noise | 2 | 0.34 | 0.386 | 0.350 | +14% | +5% |
| dense-tags | 2 | 0.59 | 0.609 | 0.645 | +3% | +14% (noise, off 0.567) |
| one-tag | 2 | 0.30 | 0.333 | 0.298 | +12% | +12% |
| decoys | 2 | 0.46 | 0.497 | 0.376 | +9% | +1% |
| range | 2 | 0.17 | 0.200 | 0.190 | +15% | +17% (real rescue work) |
| border-occlusion | 2 | 0.32 | 0.516 | 0.420 | +63% | +49% |
| clutter-only | 1 | 0.53 | 0.610 | 0.543 | +15% | +14% |
| jpeg | 1 | 1.31 | 1.545 | 1.432 | +18% | +18% |
| border-occlusion | 1 | 0.39 | 0.653 | 0.628 | +69% | +81% |

The machine was shared with other agents, so run-to-run noise is about +-0.05 ms; deltas under 0.05 ms are not
significant. Single switches (jpeg d2, 3 rounds): small rescue alone -4% (free), hull alone +15%; clutter-only d2: small
+23%, hull +18% (before the later float/aspect change). Hull with the cap but K=0 (candidate machinery only) is about
+0.01 ms on jpeg d2, K=6 +0.05, K unlimited +0.10: the remaining hull cost is the few fits that still run (one task of
about 1 ms is the critical path of the parallel pass).

## Left
- The 5% target is not reached on clutter-only / busy-background (small rescue: one Otsu + flood per box, 28 boxes per
  frame) or jpeg (hull). Ideas not done: rank boxes and cap them per frame; a cheaper support sweep (subsample points,
  coarse angle steps are unsafe on long sides); do the hull only for blobs whose normal quad failed to decode (needs the
  decode result, i.e. a second pass through TagDecoder); skip box refits when the frame already has many failed boxes.
- Fixed cost of about 0.03 ms on quiet scenes at decimation 2 (one-tag, noise): GPU min_cluster_pixels 8 instead of 24
  (more blobs read back) and the per-span bounding box and gradient loop.
