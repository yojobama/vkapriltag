# Progress: early reject of non-tag candidates (branch research/2026-11-early-reject-cpu, base 0fd3384)

The branch name research/2026-11-early-reject was already taken by an older worktree, hence the suffix.

Scratch: `$S/early-reject/` (S = the scratchpad). Build dirs `ext/avk-compare/build-early-reject` (this
branch) and `build-early-reject-base` (0fd3384 exported with `git archive`). Helper scripts in `$S/early-reject`:
`runall.sh <outdir> <tag> <exe> VAR=val...` (corpus at decimation 1 and 2), `timing.py` (ABBA).

## 1. Where the reject time goes (APRILTAG_VK_TRACE, decimation 2, 300 frames, per-scope totals in ms)

| scenario | quads | tag.quad | tag.refine | tag.decode | tag.inner | tag.thirdbit |
|---|---|---|---|---|---|---|
| clutter-only | 6174 | 318 | 252 | 23 | 18 | 21 |
| jpeg | 27148 | 961 | 669 | 95 | 90 | 91 |
| busy-background | 6286 | 353 | 287 | 24 | 17 | 19 |
| decoys | 1278 | 90 | 78 | 5 | 2 | 4 |

Refine is about 80% of the per-quad time on rejects (about 40 us per quad); the grey-margin retry and the
third-bit lookup are about 5-10% each (scopes include tracing overhead). So the saving is in rejecting
before refine, not in the retries.

## 2. Border and dark checks with grey margin on
Design: sample the unrefined outer quad once (homography, code, border); if it fails, sample the inner W
cells the same way and reject only if that fails too. Switch: `APRILTAG_VK_EARLY_REJECT` (default on).
Lifting `!grey_margin_` with this design alone lost 2 tags in `range` at decimation 2 (quads of 7-8 px
side, about 0.9 px per cell: the unrefined sample is too coarse and refinement is what makes them
decode; the border check rejects both outer and inner). Rejecting only quads with side >= 1.5 px per
cell (12 px for tag36h11) lost none. Minimum-side sweep (found tags vs base, decimation 2 | 1):
8 px: -1 | 0; 10, 12, 16 px: 0 | 0. The sweep also shows tag16h5 false detections 3 -> 0/1/1/2.
