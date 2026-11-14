# Progress: float32 vectorised quad fit

Branch `research/2026-11-quadfit-f32`, base `950c7a2`. Brief: scratchpad QUADFIT_F32_BRIEF.md.

## Log
- Started. Worktree was at 956d502; branched from 950c7a2 explicitly.
- Implemented kernel A (`library/src/gpu/QuadFitF32.cpp`, header `internal/vkapriltag/gpu/QuadFitF32.h`), DP path (FitArcs), peaks path (chunked windowed error, float filter/peaks, arc merges, batch line fit) in one go; switch `APRILTAG_VK_QUADFIT_F32` (default on), DetectorConfig::quad_fit_f32. Hull fit untouched (integer).
- Unit group `quadfit32` (new ctest `unit_quadfit32`): kernel vs double reference at bases 0/2000/16000, merge, windowed error vs direct, batch line fit. Passes.
- Build: `A=/c/Users/yojob/Projects/ext/avk-compare; B=$A/build-qf32` (cmake as in AGENT_BRIEF2). ctest: all passed except options_lint until PERFORMANCE.md table regenerated (done: scratchpad qf32_regen.py). Goldens pass untouched with the new default.
- Accuracy, decimation 2, same binary, `bash $S/qf32_acc.sh <exe> $A/qf32 d2 2 "X=1"` then `python $S/qf32_cmp.py $A/napa-corpus $A/qf32/d2_off $A/qf32/d2_on`:
  off (integer): 7146/7348, 1 false, err 0.0169/0.0993; on (f32): 7146/7348, 1 false, 0.0169/0.0984.
  Corner differences f32 vs integer: n 28588, median 0, p99 0.0015 px, p99.9 0.106, max 0.186, 4185 nonzero.
- Next: dec 1, peaks mode, hull+rescue, family16h5, sequences, timing (QFSTAT), vectorisation report.
- First timing (corpus A/B, 3 rounds interleaved, dec 2, same binary, `bash $S/benchlock.sh bash $S/qf32_ab.sh $B/tools/vk_eval.exe $A/qf32/t1 2 3 "X=1"`): cpu median off 0.093/0.090/0.093, on 0.092/0.096/0.099 ms: no gain, per-blob QFSTAT showed the first version 25-30% SLOWER than the integer fit on MSVC (MSVC vectorises almost none of the loops: /Qvec-report codes 1100 control flow (?:), 1200, 1305, 1303; lane loops are only SLP'd).
- Changes: SSE2 intrinsics for the two kernel-A passes (same lanes and order as the scalar loop; `SegmentMomentsScalar` is the reference, unit test demands memcmp equality); window error rewritten chunk-by-chunk with small local arrays (no extended arrays, window = chunk total - chunk prefix + translated next prefix) which also dropped the suffix scan.
- Accuracy after these changes (same commands as above, plus d1, peaks p2/p1, hull+rescue h2): identical found/false in all five modes (7146, 7129, 7140, 7128, 7228 found); err p95 off->on d2 0.0993->0.0984, d1 0.0965->0.0959, p2 0.0949->0.0949, p1 0.0943->0.0948, h2 0.0980->0.0976; corner differences max 0.186 px, p99 0.0015-0.0021 px. Sequences (track_eval full,key) identical in recall and error.
- QFSTAT (temporary, removed): min of 4 interleaved runs, fit ns over the scenario, dec 2: default dp: jpeg 125.6->110.5 ms (0.88), dense-tags 5.75->5.43 (0.945), busy-background 18.6->16.5 (0.889), clutter-only 15.5->14.8 (0.956), one-tag 1.161->1.033 (0.889), many-tags 3.65->2.96 (0.812). Forced peaks: jpeg 212.5->232.7 (1.095), dense-tags 1.043, busy-background 1.036, clutter-only 1.043, one-tag 0.897, many-tags 0.751.
- ctest 46/46 with the unit group; goldens untouched.

## Vectorisation report (QuadFitF32.cpp, current commit line numbers)
Commands: MSVC `cl /std:c++20 /O2 /EHsc /DNDEBUG /Qvec-report:2 /c /I<src/internal> /I<include> QuadFitF32.cpp`; gcc (MinGW 16.1, x86-64 SSE2) `g++ -std=c++20 -O3 -DNDEBUG [-fno-math-errno] -fopt-info-vec-optimized -fopt-info-vec-missed -c ...`. Raw outputs were in the scratchpad (qf32vec/). Pi gcc 14: not run (board unreachable).
| Loop | MSVC | gcc x86-64 | note |
| --- | --- | --- | --- |
| kernel A lane blocks (Pass1/Pass2, scalar form, l.76/134) | not (1303 too few iterations); SSE2 intrinsics used instead, bit-equal to scalar | vectorised (16 byte, unroll 4) | |
| kernel A tails (l.84/147) | not | vectorised | |
| window chunk terms and prefix (BuildChunk, l.304) | not (1500 aliasing) | not (serial scan) | six independent running sums, scalar |
| window combine (CombineChunk, l.339) | vectorised | not without -fno-math-errno (sqrt errno control flow); vectorised with it | CMake adds -fno-math-errno for this file on GNU/Clang |
| batch line fit (FitLines, l.263) | not (1100 control flow) | same as combine | |
| 7-tap filter (l.412) | vectorised | vectorised (versioned for aliasing) | |
| peak mask (l.425) | vectorised | vectorised | |
| peak compaction, 4-deep search, merges | scalar | scalar | by design |

## Final timing and decision
- Clean build, corpus A/B, 3 interleaved rounds, dec 2 (`bash $S/benchlock.sh bash $S/qf32_ab.sh $B/tools/vk_eval.exe $A/qf32/t2 2 3 "X=1"`; summarise with `python $S/cmpn/sum.py ...`): all-frames total median off 0.283/0.282/0.274, on 0.277/0.284/0.281 ms; cpu median off 0.083/0.093/0.092, on 0.088/0.093/0.092. No difference beyond noise; per-scenario (jpeg, dense-tags, busy-background, clutter-only, one-tag, many-tags) totals also within noise.
- Fit stage only (QFSTAT, above): -5 to -19% default; peaks forced +4 to +10% on busy scenarios.
- Decision: option kept, default on (gate passes, output equivalent); benefit on the desktop is confined to the fit stage. Pi not reachable ('Host does not exist'), nothing run there; gcc/NEON scalar-lane path untested at run time.
- Hull fit (D) not done. Integer fit left in place; deleting it would remove about 370 lines of QuadDecode.cpp (TryDpQuad, SumMomentsWindow, the peaks body of FitQuadForBlob, prefix helpers); FitLine/Int128 stay for the hull fit.
- Docs: DECISIONS.md section 11 row, CHANGELOG Unreleased, ARCHITECTURE quad fit bullet, PERFORMANCE options table. ctest 46/46.

## Follow-up (review of 9e62d2e): peaks path
Per-stage counters (QFSTAT patch: `qf32_stat3.py` applies, `qf32_unstat3.py` removes; `bash $S/benchlock.sh bash $S/qf32_stat3.sh $E 2 "jpeg busy-background" APRILTAG_VK_QUADFIT=peaks`, best of 5 interleaved runs, ns summed over the scenario, forced peaks, dec 2), BEFORE any change (9e62d2e + tidy):
| jpeg | old integer | new f32 |
| window error (old: prefix sums + windows) | 98.8 ms | 59.0 |
| filter, peak scan, sort | 45.5 | 34.0 |
| arcs (kernel A over all points) | - | 34.4 |
| old: 45 segment fits; new: merges + batch line fit | 30.5 | 58.3 |
| search | 38.2 | 19.1 |
| fit_ns total | 217.5 | 226.8 (new also pays the unpack, about 23 ms; old reads the points in place) |
busy-background: old window 19.7, filter 7.8, seg fits 3.3, search 3.9, total 35.8; new window 8.1, filter 5.5, arcs 4.0, merges+fit 6.0, search 1.8, total 30.1.
Losing stages: merges + batch line fit, arcs, and the unpack. Step 1: kernel A tail in SSE2 (masked lanes, arrays padded by 8) and an SSE2 unpack: arcs 34.4 -> 25.5, total jpeg 226.8 -> 206.2 ms, busy-background 29.1.
Step 2: forward segments only in the batch (45 cells), closing segments fitted lazily as the integer fit does, batch line fit in SSE2 (bit-equal to LineKernel, unit test demands ==). jpeg new: window 59.3, filter 33.6, arcs 26.2, merges + batch fit 20.7, search (now with the closing fits) 38.1, total 193.0 ms against 217.5 old; busy-background total 26.6 against 35.8.
fit_ns per scenario, best of 5 interleaved runs, dec 2, old integer -> new f32 (`qf32_stat3.sh`, 6 scenarios):
- forced peaks: jpeg 214.3 -> 192.3 ms (0.90), dense-tags 13.35 -> 11.51 (0.86), busy-background 31.0 -> 26.8 (0.87), clutter-only 26.2 -> 22.8 (0.87), one-tag 2.99 -> 2.24 (0.75), many-tags 8.40 -> 5.64 (0.67).
- default (dp + peaks fallback): jpeg 127.8 -> 92.3 (0.72), dense-tags 6.28 -> 4.35 (0.69), busy-background 17.5 -> 12.3 (0.70), clutter-only 16.0 -> 11.7 (0.73), one-tag 1.17 -> 0.88 (0.76), many-tags 3.29 -> 2.28 (0.69).
Targets met: forced peaks faster than the integer fit in all six scenarios; default jpeg 92.3 ms below the earlier 110.5 ms.

### Re-gate after the peaks work (clean build, same commands as before)
- ctest 46/46, goldens untouched.
- Accuracy, same binary, F32=0 against default: dec 2 7146/7348, 1 false, err 0.0169/0.0993 -> 0.0169/0.0984, corner diff p99 0.0015, max 0.186; dec 1 7129, 0 false, 0.0167/0.0965 -> 0.0167/0.0959, p99 0.0018, max 0.173; peaks dec 2 7140, 0 false, p95 0.0949 -> 0.0949, p99 0.0018, max 0.171; peaks dec 1 7128, 1 false, 0.0943 -> 0.0948; hull+rescue dec 2 7228, 0 false, 0.0980 -> 0.0976. No scenario differs in found count. Default output is the same as at 9e62d2e up to the GPU run-to-run noise (nonzero corner differences 4190 against 4187), so the sequences were not re-run.
- Corpus A/B (3 interleaved rounds under the lock, `qf32_ab.sh`, dec 2): total median off 0.285/0.289/0.285, on 0.285/0.283/0.284 ms; cpu median off 0.091/0.095/0.093, on 0.092/0.089/0.089 ms.
- Pi: `sh $S/pi.sh 'echo ok'` -> "Host does not exist"; nothing ran there.

### Vectorisation table, final code (MSVC /Qvec-report:2; gcc 16.1 x86-64 `-O3 -fno-math-errno -DVKAPRILTAG_NO_SSE2` so the portable loops are the ones reported)
| Loop | MSVC | gcc |
| --- | --- | --- |
| kernel A block loops, portable form | not (1303); SSE2 intrinsics (blocks and masked tail) used on x86-64, bit-equal | vectorised |
| kernel A tails, portable form | not | not (scalar tail, only used without SSE2) |
| unpack | SSE2 intrinsics (unit test against the accessors) | scalar loop, not checked |
| chunk prefix scan (BuildChunk) | not (1500) | not (serial scan, six independent sums) |
| window combine (CombineChunk) | vectorised | vectorised with -fno-math-errno |
| batch line fit | SSE2 intrinsics, bit-equal to LineKernel (tail of up to 3 sets scalar) | scalar tail only without SSE2; the portable loop under gcc was vectorised with -fno-math-errno earlier, not re-checked in this form |
| 7-tap filter, peak mask | vectorised | vectorised |
| merges, search, closing fits | scalar by design | scalar by design |

## Part D: the hull fit in float32
Baseline (integer path, `APRILTAG_VK_HULL_QUAD=1`, dec 2, QFSTAT counters, run with `bash $S/hull_stat.sh $E 2 "<scenarios>" "0 1"` after applying `qf32_stat3.py` and `hull_stat.py`). Counters: S16 HasOutliers total, S17 hull bounding box, keys, sort, hull, simplify, edge choice, S18 selection pass (4 sides), S19 vote, S20 moments + FitLine; S24 FitQuadHull calls past the size test, S25 hull candidates, S26 accepted hull quads, S27 HasOutliers calls. ms over the scenario (first of 5 runs, the min-selection was broken by a missing bc and is fixed afterwards):
| scenario | HasOutliers | hull pre | selection | vote | moments | attempts | candidates | accepted | HasOutliers calls |
| border-occlusion | 0.39 | 8.30 | 2.65 | 6.33 | 2.15 | 191 | 302 | 188 | 152 |
| jpeg | 3.11 | 1.85 | 0.76 | 1.25 | 0.57 | 886 | 17464 | 34 | 1925 |
| busy-background | 4.41 | 1.07 | 0.41 | 1.04 | 0.26 | 327 | 2241 | 39 | 1051 |
| family16h5 | 3.66 | 0.93 | 0.43 | 0.97 | 0.29 | 300 | 1325 | 51 | 1460 |
Other scenarios: HasOutliers 0.5 to 3.5 ms and few or no hull attempts (obstruction 5 attempts, grey-margin 34, clutter-only 274). Hull-heavy: border-occlusion; HasOutliers-heavy: busy-background, family16h5, jpeg, clutter-only.

Implementation: `quadf32::FitHullSide` (strip mask in SSE2, compaction, vote with an int bin array and four sub-histograms, band mask as a weight array fed to kernel A about the edge start via `SegmentMomentsAbout`), `quadf32::HasOutliers` (blocks of 64, SSE2 over four points, all four sides branch-free, leave after a block once the count exceeds `allowed`), `quadf32::BoundsAndKeys` (SSE2 min/max and keys), `HasOutliersF32` in QuadDecode.cpp reusing the blob's unpacked arrays (tagged by first index and length, reset at the start of each Decode). `FitQuadHull` branches on `config.quad_fit_f32` in the bounding box pass and in the per-side loop; sort, hull, simplify and edge choice are shared.
Unit tests (group quadfit32): bounds and keys against a loop; masked kernel SIMD == scalar (memcmp) and against a double sum over the weighted points; `FitHullSide` against a double copy of the integer algorithm on rotated rectangles with a bump and clutter (angles 0, 7, -23, 41), SIMD == portable bit-equal; `HasOutliers` against the double version on 60 random quads x 3 fractions (at most 1 decision may differ; none did), SIMD == portable.
Stage table after (same method, best of 5, ms over the scenario, old integer -> new f32):
| scenario | HasOutliers | hull pre (shared) | selection | vote | moments |
| border-occlusion | 0.40 -> 0.13 | 8.89 -> 8.82 | 2.63 -> 1.24 | 6.43 -> 4.48 | 2.16 -> 1.05 |
| jpeg | 3.13 -> 1.18 | 1.93 -> 1.94 | 0.74 -> 0.34 | 1.23 -> 0.82 | 0.56 -> 0.23 |
| busy-background | 4.36 -> 1.39 | 1.09 -> 1.12 | 0.38 -> 0.19 | 1.06 -> 0.81 | 0.27 -> 0.15 |
| family16h5 | 3.66 -> 1.21 | 0.99 -> 1.01 | 0.42 -> 0.18 | 0.97 -> 0.73 | 0.29 -> 0.14 |
First attempt (plain float loops, no SSE2, no sub-histograms) was not faster: HasOutliers 3.2 -> 4.8 ms on jpeg (all four sides evaluated for every point against the integer code's early exit per point) and selection 0.74 -> 0.53; with SSE2 and the sub-histograms every stage wins. "Hull pre" is dominated by the key sort, hull and simplify, which stay shared.

Portable clamp: the first float HasOutliers used min/max for the clamp, which gcc will not vectorise without fast-math ("unsupported control flow"); both forms now clamp with 0.5 * ((|t| - |t - 1|) + 1), which gcc vectorises and which keeps SIMD and portable bit-equal. The SSE2 path pays two more operations: final HasOutliers numbers below.

Final stage table (best of 5, ms over the scenario, `APRILTAG_VK_HULL_QUAD=1`, dec 2, old integer -> new f32):
| scenario | HasOutliers | hull pre (shared) | selection | vote | moments |
| border-occlusion | 0.39 -> 0.17 | 8.82 -> 8.86 | 2.55 -> 1.23 | 6.40 -> 4.45 | 2.16 -> 1.05 |
| jpeg | 3.14 -> 1.43 | 1.94 -> 1.95 | 0.78 -> 0.32 | 1.24 -> 0.82 | 0.56 -> 0.23 |
| busy-background | 4.37 -> 1.76 | 1.15 -> 1.11 | 0.39 -> 0.20 | 1.05 -> 0.79 | 0.28 -> 0.15 |
| family16h5 | 3.65 -> 1.46 | 0.97 -> 0.95 | 0.41 -> 0.16 | 0.98 -> 0.73 | 0.29 -> 0.14 |

Gate (clean build, same binary, APRILTAG_VK_QUADFIT_F32=0 against default; found, false, err med/p95 off -> on, corner diff max / p99 / nonzero):
- hull, dec 2: 7204/7348, 1, 0.0170/0.0943 -> 0.0170/0.0940, 0.186 / 0.0015 / 4219 of 28820
- hull, dec 1: 7233/7348, 0, 0.0169/0.0929 -> 0.0169/0.0916, 0.173 / 0.0017 / 5183 of 28932
- hull + rescue, dec 2: 7228/7348, 0, 0.0170/0.0980 -> 0.0170/0.0976, 0.186 / 0.0015 / 4219
- hull + rescue, dec 1: 7233/7348, 0, 0.0169/0.0929 -> 0.0169/0.0916, 0.173 / 0.0017 / 5183
- No scenario differs in found count. Hull quads cannot be told apart in the dumps, so the corner differences are over all detections.
- Default (hull off) dec 2 against the dump of 2493da8: identical found/false/error; corner differences max 0.0029 px on 5 corners (GPU noise).
- ctest 46/46 (golden profile rescue-hull enables the hull; untouched).
- Corpus A/B with the hull on (`bash $S/benchlock.sh bash $S/qf32_ab.sh $E $A/qf32/t4 2 3 "APRILTAG_VK_HULL_QUAD=1"`, 3 interleaved rounds): total median off 0.286/0.283/0.284, on 0.277/0.279/0.284 ms; cpu median off 0.092/0.095/0.096, on 0.087/0.087/0.094 ms (about 5 to 8% lower cpu).
- Pi: `sh $S/pi.sh 'echo ok'` -> "Host does not exist"; not run.

Vectorisation of the new loops (MSVC /Qvec-report:2; gcc 16.1 `-O3 -fno-math-errno -DVKAPRILTAG_NO_SSE2`):
| Loop | MSVC | gcc portable |
| --- | --- | --- |
| bounds and keys | SSE2 intrinsics (scalar tail 1200) | vectorised |
| strip mask | SSE2 intrinsics | vectorised |
| compaction of near points | scalar (1203) | scalar by design |
| vote bin indices | vectorised | vectorised |
| vote increments (four sub-histograms) | scalar (1505) | scalar by design |
| band mask and weights | SSE2 intrinsics | vectorised |
| masked kernel A passes | SSE2 intrinsics, bit-equal to portable | vectorised |
| HasOutliers | SSE2 intrinsics (portable loop 1106/1305) | vectorised after the fabs clamp |
`FitLine`, `Int128`, `NeedsWide` and the integer hull code in QuadDecode.cpp now serve only `APRILTAG_VK_QUADFIT_F32=0`.

## Part E: exact, cheaper hull geometry
Step 0, the split of "hull pre" (db38944, F32=1, `APRILTAG_VK_HULL_QUAD=1`, dec 2; counters from `hull_stat2.py` on top of `hull_stat.py`/`qf32_stat3.py`; `bash $S/benchlock.sh bash $S/hull_stat_pre.sh $E 2 "<scenarios>" "1"`; ms over the scenario, best of 5):
| scenario | bbox + keys | sort + unique + copy | monotone chain | far vertex | simplify + vertex list | lengths + edge choice | calls | n per call | unique per call | m per call | k per call |
| border-occlusion | 0.48 | 6.12 | 2.08 | 0.04 | 0.13 | 0.17 | 191 | 1190 | 1190 | 16.7 | 6.3 |
| jpeg | 1.27 | 22.06 | 6.77 | 0.14 | 0.46 | 0.05 | 886 | 860 | 860 | 9.9 | 5.2 |
| busy-background | 0.18 | 2.63 | 0.95 | 0.02 | 0.07 | 0.03 | 327 | 297 | 297 | 3.4 | 2.5 |
| family16h5 | 0.13 | 2.13 | 0.73 | 0.03 | 0.11 | 0.04 | 300 | 296 | 296 | 4.9 | 3.6 |
(The jpeg and busy-background numbers include calls that return early; sizes are sums over calls divided by calls.) No duplicate points occur in practice (unique = n). The sort/unique/copy is about 70% of the pre stage and the monotone chain about 25%; far vertex, simplification and edge choice together are under 5%, so the column hull (steps 1) is the one that pays; steps 2 to 4 are small and done for exactness-checked cleanliness.

Implementation (shared by the float and the integer per-side paths, no switch): the points are unpacked first in both paths; `BoundsAndKeys` with null keys gives the box; `ColumnCandidates` fills the lowest and highest y of each column and lists them in (x, y) order; fewer than 8 candidates falls back to keys + sort + unique (the old path); `geom::MonotoneChain` unchanged; `FarVertex` (first maximum of the integer squared distance, one `std::hypot` of the winner for the `far_d < min_side_x2` test); `SimplifyHull` (first maximum of the integer cross product over separate x and y arrays with vertex 0 repeated at m, then `|cross| / sqrt(dx*dx + dy*dy)` as the same double expression as the old `LineDistance`; the old coincident-ends metric is kept for dx = dy = 0); `ChooseEdges` (lazy memo of `corner_ok` per ordered pair, the nested loops and tie order unchanged); `keep`, `hull`, the vertex arrays, `near` and `hist` live in the scratch. The old `LineDistance` and `SimplifyChain` are gone from QuadDecode.cpp and survive as references in the unit test.
Exactness evidence:
- Unit test (group quadfit32, `TestHullGeomF32`): 600 random shapes (rectangle outlines axis-aligned and rotated, thin rectangles of height 3 to 10, random clouds, discs, outlines with every point twice and with noise; coordinates near 0, 7000 and 16383): the column hull equals keys + sort + unique + `MonotoneChain` vertex for vertex in all 600 (including the vertical sides of the axis-aligned rectangles); `FarVertex` equals the double hypot loop; the `keep` array of `SimplifyHull` equals the old `SimplifyChain` for eps 2.4, 0.5 and 8.0 on all hulls (3424 kept vertices at eps 2.4); `ChooseEdges` equals the old nested-loop form (same four indices, same total) on 453 polygons with a result.
- Corpus dumps, new binary against a build of db38944 (`bash $S/exact_runs.sh <db38944 exe> <new exe> $A/qf32/ex`, scenario dumps compared with `exact_cmp.py`: files differing, lines, largest value difference; the first number of each pair is base against base, the noise level):
  | run | base vs base | new vs base |
  | hull dec 2, F32 default | 3 files, 3 lines, 0.0027 | 2 files, 2 lines, 0.0000 |
  | hull dec 2, F32=0 | 1, 1, 0.0027 | 1, 1, 0.0027 |
  | hull dec 1, F32 default | 8, 11, 0.0022 | 9, 14, 0.0012 |
  | hull dec 1, F32=0 | 8, 8, 0.0022 | 6, 6, 0.0012 |
  | hull + rescue dec 2, F32 default | 3, 3, 0.0000 | 4, 4, 0.0027 |
  | hull + rescue dec 2, F32=0 | 1, 1, 0.0027 | 1, 1, 0.0027 |
  | default (hull off) dec 2 | not repeated | 1, 1, 0.0000 |
  The differences of the new binary are of the size and number of the base-against-base run-to-run noise.
