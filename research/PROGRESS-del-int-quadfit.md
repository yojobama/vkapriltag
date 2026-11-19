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

(Results of the corpus comparison are appended below.)
