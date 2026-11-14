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
