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
