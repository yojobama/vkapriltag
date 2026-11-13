# Progress: own decode (Phase 4b stage S1, branch research/2026-11-own-decode, base e992ec9)

Goal of S1: an in-house port of libapriltag's decode path, proven bit-identical in-process, behind
`APRILTAG_VK_OWN_DECODE` (default on; 0 = patched upstream). Upstream types stay at the boundaries (S2 swaps them,
S3 removes the fetch and the patches). Desktop only.

## What was ported (library/src/OwnDecode.h/.cpp, namespace apriltag_vulkan::own)
| Upstream | Own | Notes |
| --- | --- | --- |
| `homography_compute2`, `quad_update_homographies` (incl. the patched singularity check) | `ComputeHomography` | same elimination order, `double h[9]` on the stack, no `matd` |
| `homography_project` | `Project` (inline) | identical expression |
| `quad_decode` (grid, grey models, sharpening, margin) | `SampleFamily` | border sample positions, the models' normal matrices and the inverse Cholesky factor are precomputed per family (valid when all 8w border samples are inside the image, else computed as upstream); sharpening only at payload cells; exact fallback to the full kernel when a value is not finite |
| `quick_decode_codeword` (+ the table) | `Lookup` | no hash table: 4-block index (as `ThirdBitIndex`), result = lowest code index within max_hamming over the first rotation that has one; this is exactly what the insertion-order table returns. max_hamming is inferred from the size of the family's own table (`family->impl`), so it needs no new argument |
| `quad_decode_index` | `Decoder::Decode` | one sample per family, no per-attempt allocation; returns the first family's raw sample |
| `quad_sample_code`, `apriltag_min_hamming` | `Decoder::SampleCode`, `MinHamming` | sampling without the lookup |
| detection construction (R multiply, centre, corners) | `FillTag`, `NewDetection` | same operation order; `cos`/`sin` tables per rotation from the same expression |
| `reconcile_detections`, `g2d_polygon_*` | `Reconcile` | including `zarray_remove_index(..., shuffle=1)` order and the `sqrtf` in `g2d_line_init_from_points` |
| `zarray_sort(DetectionCompare)` | `SortById` | `qsort` with the same comparator (unstable, as upstream, so equal ids keep upstream's order) |

TagDecoder: `OwnedQuad` now holds the quad and its `double h[9]` (no matd); `UnrefinedSample` and the decode calls go
through `DecodeQuad`; the third-bit step reads the raw code and margin the final decode returned instead of
resampling (own path only); `ReadBorder` takes `const double *`. `TryInnerSquare`'s trial decode only counts (no
detection is built). Steps keep their structure.

## Differential test (ctest own_decode_random, own_decode_reconcile, own_decode_corpus, own_decode_comparison_can_fail)
`tests/own_decode_test.cpp`; every comparison is bitwise (ids, hamming, margin float bits, homography, centre, corners,
raw sample, homography validity).
- random: 8 families (tag36h11, tag25h9, tag16h5, tagCircle21h7, tagCircle49h12, tagStandard41h12, tagStandard52h13,
  tagCustom48h12) x 2 border polarities x 4 corrected-bit limits (tables capped by memory: 0..3 up to 25 bits, 2 above, 1
  from 41 bits) x one and two families; rendered tags (random pose, perspective, contrast, shading, blur, noise, tags
  partly outside the frame), exact / perturbed (0.2 to 3 px) / rolled / wrong-polarity / random / noise-image /
  degenerate quads: 22560 quads, 6571 with a detection, 360 degenerate homographies, 0 mismatches.
- corpus: `tests/own_decode_quads.txt` holds every quad TagDecoder passed to a decode or sample on tests/corpus (all ten
  scenarios, d1 and d2, default / small-rescue+hull / border-precheck and dark-reject off): 2974 attempts (the refined
  quads, inner squares, side-one-out and rescue refits included), 927 with a detection, 0 mismatches.
  (Recorded with a temporary hook in `DecodeQuad` and `UnrefinedSample`; images are the committed PNGs.)
- reconcile: 30000 random detection sets (2 to 15 detections, overlaps, containment, identical shapes, quantised corners
  for ties; 78654 detections removed), 0 mismatches, also the detection order.
- `own_decode_comparison_can_fail` runs the random test with a changed sharpening on one side and must fail.
- `own_decode_test bench <corpus> <quads>` replays the recorded attempts.

## Whole-pipeline checks
- ctest: 14 of 14 pass on the branch build (goldens unchanged, options_lint with the new registry entry); the golden test
  also passes with each of the four `APRILTAG_VK_FORCE_NO_*` flags, all four together and with `OWN_DECODE=0`.
- In-process cross-check (temporary, `research/own-decode-xcheck.patch`: every candidate is run through the whole step
  chain with and without the own decode and the detections compared bitwise, including the refined corners): napa
  corpus, 26 scenarios, 0 mismatches over 32030 candidates (7341 with detections) at d2, 43679 at d1, and the same under
  small-rescue+hull, polish, grey margin off, third bit off, family limits off, border-precheck/dark-reject/early-reject
  off, side-one-out off (about 330k candidate comparisons in all, family16h5 included).
- Corpus dumps against a build of e992ec9: the GPU front end is not deterministic (the base differs from itself by one
  or two dump lines in a few scenarios, e.g. noise, blur, busy-background, corners moving by about 1e-3 px), so dumps
  cannot be required to be byte-identical. noise d2 over 16 runs each: base gives two distinct dumps (14 / 2), `OWN_DECODE=0`
  the same two (10 / 6), own the same two (13 / 3); every own dump is one of base's. All 26 scenarios differ from base
  only in such lines (variants: default, rescue+hull, polish, grey margin off, third bit off, family limits off).
- track_eval (full and key modes, 10 sequences, 2 runs each of base and own): 0 differing detection lines.


## Timing (desktop, RX 9060 XT, other agents share the GPU, so medians of interleaved runs)
Harness cpu median per frame (ms), `vk_eval --iterations 8 --warmup 5`, ABBA x 4 rounds (16 runs per row; own vs
`APRILTAG_VK_OWN_DECODE=0`), from `$S/own-decode/timing.py`:

| scenario | d2 own | d2 OWN=0 | ratio | d1 own | d1 OWN=0 | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| one-tag | 0.0595 | 0.0610 | x1.024 | 0.0841 | 0.0861 | x1.024 |
| many-tags | 0.1234 | 0.1266 | x1.027 | 0.1237 | 0.1286 | x1.040 |
| dense-tags | 0.2664 | 0.2842 | x1.066 | 0.1971 | 0.2174 | x1.103 |
| clutter-only | 0.1629 | 0.1739 | x1.067 | 0.1391 | 0.1532 | x1.101 |
| jpeg | 0.3458 | 0.3846 | x1.112 | 0.6166 | 0.7035 | x1.141 |
| busy-background | 0.1905 | 0.2020 | x1.060 | 0.1701 | 0.1815 | x1.067 |
| small-tags | 0.0926 | 0.0963 | x1.039 | 0.0802 | 0.0815 | x1.017 |

Never slower; 2 to 14 percent of the CPU time per frame. (The harness total time, which includes the GPU wait, moved
between x0.87 and x1.08 in the same runs: noise from the shared GPU.)

Decode attempts only (`own_decode_test bench`, the 2052 recorded tag36h11 attempts, quad decode and nothing else, 5
rounds): patched libapriltag 480k attempts/s, own 945k attempts/s (x1.97; first round x2.1 to x2.5 while the caches
warm). Per attempt on this machine: homography 245 ns, plus sampling 1000 ns (cumulative), plus code lookup and
detection construction 1050 ns.

Where the time is now (dense-tags d2, trace scopes summed over 160 frames, 4984 candidates; own vs libapriltag):
`tag.refine` 42.4 vs 42.7 us per candidate (83 percent of a candidate's 51 us), `tag.decode` 2.34 vs 4.35 us,
`tag.inner` 1.76 vs 3.22 us, `tag.thirdbit` 0.35 vs 3.54 us. The decode and recovery steps are now about 5 percent of
a candidate; what is left is the edge refinement. Inside the decode, sampling is 72 percent (about 100 projections
with a division each plus the bilinear reads), the homography 23 percent. Trying a batched projection loop (to let the
compiler divide two at a time) changed nothing and was dropped. Shortcuts in the homography solve or the projections
would change the last bits (signs of zeros, operation order), so they were not taken.

## Not verified / risks
- GCC (the Pi): OwnDecode.cpp was only compiled with MSVC. GCC contracts a*b+c into fused multiply-adds by default for
  both libapriltag's C and this file, and which expressions it fuses depends on how the code is laid out, so bit
  equality is not guaranteed there; the differential test (ctest own_decode_*) decides, and it must be run on the Pi
  before relying on it. If it differs, build OwnDecode.cpp and libapriltag with `-ffp-contract=off` (as QuadDecode.cpp
  already is) or keep `APRILTAG_VK_OWN_DECODE=0` on that platform.
- Detection order: `SortById` uses `qsort` like upstream, so two surviving detections with the same id keep whatever
  order the C library's `qsort` gives (not stable). A stable sort would be better for S3 but would differ from the base.
- The max Hamming distance of a family is inferred from the size of libapriltag's own table (`family->impl`); S2/S3
  replace this with the family's own field. Families wider than 32 cells, with more than 63 bits, more than 8 families
  or a table built for more than 3 bits are not supported by the own path: the whole run then uses libapriltag.
- Dump byte equality against the base cannot be demanded because the GPU front end is not deterministic; see above.

## Derived files (for NOTICE in S3)
library/src/OwnDecode.h and OwnDecode.cpp are ports of AprilRobotics apriltag v3.4.5 (apriltag.c quad_decode,
quad_update_homographies, homography_compute2, quad_decode_index, reconcile_detections; common/g2d.c polygon overlap;
apriltag_math.h Cholesky solve; common/homography.h projection). BSD-2-Clause, Copyright The Regents of The University of
Michigan: carried by the header comment; the NOTICE file should list both. tests/own_decode_test.cpp contains no
upstream code (it calls the upstream functions).

## Commands
- Differential test: `ctest -R own_decode`; speed: `own_decode_test bench tests/corpus tests/own_decode_quads.txt`.
- In-process cross-check of the whole step chain: `git apply research/own-decode-xcheck.patch`, build, run vk_eval with
  `OWN_XCHECK=1` and read the `XCHECK candidates ... mismatches` line on stderr; revert the patch afterwards.
- Corpus runs: `$S/own-decode/run.sh own|base <out> <decimation>` (own = build-own-decode, base = a build of e992ec9 in
  build-own-decode-base), `variants.sh`, `xcheck_all.sh`, `timing.py <scenarios> <decimations> <rounds> <iterations>`.
- Tests build: `cmake -S apriltags_vulkan -B build-own-decode-ct -G Ninja -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF`.
