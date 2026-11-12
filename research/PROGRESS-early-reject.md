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

## 3. Other ideas
- Third-bit resample: needs the margin and code from `quad_decode_index` (a patch change to libapriltag);
  about 3 us per undecoded quad (traced), now rarer after the early reject. Not done.
- Unrefined Hamming distance gate: `APRILTAG_VK_PROBE_STATS` shows accepted quads at distance 0 to 10 and rejects
  at 5 to 10; no safe cut. Null result.
- Surviving quads: 76% of clutter-only quads still refine (they look like tags on both checks).

## 4. Gate results (commands: runall.sh, cmp.py, idcmp.py, timing.py, score_seq.py in $S/early-reject)
- Corpus found tags (d1 | d2): base 7129 | 7146; on identical. rescue+hull: 7233 | 7228 identical; +polish identical.
  False: d2 3 -> 1 (family16h5), rescue variants 2 -> 0. p95 and median unchanged.
- EARLY_REJECT=0 against base build: ids and structure identical, coordinates within 5e-4 px (same as base vs base).
- Sequences (ten, decimation 2): score_seq output identical.
- ctest: unit tests pass; golden_test.py all five profiles 0 differing frames.
- Time: see PERFORMANCE.md 8be (t_final.txt). dense-tags +6% | +4% cpu is the one scenario slower than noise.
