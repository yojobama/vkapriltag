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
