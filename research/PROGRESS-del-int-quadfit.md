# Delete the integer quad fit

Branch `research/2026-11-del-int-quadfit`, base `446aa2b` (research/improvements-2026-11).

## Deleted

- `QuadDecode.cpp`: `FitQuadForBlob` integer body (prefix sums, windowed error, Gaussian filter, peak search,
  combinatorial search), `TryDpQuad`, `SumMomentsWindow`, `FitLine` (int64 / 128-bit moments), `Int128` and
  helpers, `Covariances`, `NeedsWide`, `FitLineError`, `Hypot`, `Add`/`Sub`, `ReadMomentsWindow`, the integer hull
  side fit in `FitQuadHull`, the integer `HasOutliers`, `kFilterCoefficients`, `kNMaxima`, the
  `APRILTAG_VK_QUADFIT_F32` env read.
- `FitQuadForBlobF32` is now `FitQuadForBlob`; `HasOutliersF32` is now `HasOutliers`; `FitQuadResult` lost
  `moments` and `f32`.
- `QuadDecode.h`: integer scratch members (`cs`, `error`, `filtered`, `padded`, `win_*`, `peaks`, `hull_near`,
  `hull_hist`).
- `Types.h`: `LineFitMoments` and the unused `RawLineFitPoint::Mx/My/Mxx/Mxy/Myy` accessors.
- `DetectorConfig::quad_fit_f32`, the `APRILTAG_VK_QUADFIT_F32` row in `Options.cpp` and PERFORMANCE.md.
- Kept: `FindDpCornerIndices` (the f32 path seeds with it), `FitQuadHull` geometry, `UpdateFitQuad`.
- Unit tests: none referenced the integer fit directly (the float32 tests already compare against double-precision
  references inside the test), so none changed.
- Docs: CHANGELOG (Removed entry, float32 entry), DECISIONS (three rows), ARCHITECTURE, PERFORMANCE option table.

## Gates (desktop, MSVC, Release, Ninja)

Scratch dir: `<scratchpad>/dik` (base = `git archive 446aa2b` extract). Library trees `lib-base`, `lib-new`
configured with `cmake -S <tree>/apriltags_vulkan -B lib-<t> -G Ninja -DCMAKE_BUILD_TYPE=Release` through `vs.bat`,
built with `vsbuild.bat`, `ctest --test-dir lib-<t>` through `vs.bat`.

1. ctest: before (446aa2b) 47/47 pass; after 47/47 pass (no SWIG Python in this configuration). No golden or
   gpu_trace file is touched by the diff.
2. Corpus: `ext/avk-compare/ours/vk_eval.cpp` is still written against the pre-2.0 API (it includes
   `vkapriltag/TagDecoder.h`) and does not build, so the in-tree `tools/vk_eval` (same CLI) of `lib-base` and
   `lib-new` was used. `run_corpus.sh <lib-t>/tools/vk_eval.exe <napa-corpus> <out> --decimation N` for N = 2 and 1,
   modes default, `APRILTAG_VK_QUADFIT=peaks`, `APRILTAG_VK_HULL_QUAD=1`, `APRILTAG_VK_HULL_QUAD=1
   APRILTAG_VK_SMALL_RESCUE=1`. Dumps compared with `cmp`; scored with `score_corpus.py`.
3. `git grep -n -i "quadfit_f32\|quad_fit_f32"`: only the CHANGELOG entry and research history text remain.


## Results

ctest: 47/47 before and 47/47 after.

Corpus, base (446aa2b) against new, summed over the 26 scenarios (tags 7348 in every run); per-scenario found,
false and winding counts are equal in all eight runs, and the p50/p95/max corner error columns of the score tables
are equal at three decimals:

| mode | dec 2 found / false | dec 1 found / false |
| --- | --- | --- |
| default | 7146 / 1 | 7129 / 0 |
| `QUADFIT=peaks` | 7140 / 0 | 7128 / 1 |
| `HULL_QUAD=1` | 7204 / 1 | 7233 / 0 |
| `HULL_QUAD=1 SMALL_RESCUE=1` | 7228 / 0 | 7233 / 0 |

Byte identity: the dumps are not byte-reproducible from run to run even for the unchanged binary. The first
base run and the first new run differed in 1 to 12 of 26 files per mode, so base was run a second time ("base2",
`corpus2.sh`). base against base2 (same binary) differs in 1 to 11 files per mode, base against new in 1 to 12, base2
against new in 2 to 10. In every comparison the detection ids, counts and order are identical; only the last digits of
corners differ, by at most 0.0046 px (base against base2 at most 0.0030, base against new at most 0.0046). Three
repeats of base on `noise` at dec 2 were identical to each other and to new but not to the first base run, so the
source is intermittent (not caused by this change). The cause was not investigated.

`git grep -n -i "quadfit_f32\|quad_fit_f32"` finds only the CHANGELOG entry and research history text.
