# Progress: CPU geometry header (branch research/2026-11-geometry, base bc8cb18)

Pure refactor. One internal header, `library/src/Geometry.h` (namespace `apriltag_vulkan::geom`, header-only,
not installed; the library gets `src/` as a private include directory), replaces duplicated CPU helpers. One commit
per consolidation. No GPU file, shader or public header changed.

## Merged (commit order)
1. `BilinearAt`, `BlockInside`, `SampleEdge<S>`, `MedianInPlace`: RefineEdges (`BilinearAt`, `SampleAt`, three
   `nth_element` medians), BorderSample (`Sample`, `Median`), PolishFit (median residual). `SampleEdge<float>` is the
   old `SampleAt` (float position), `SampleEdge<double>` the old BorderSample sampler; the four-tap order is unchanged.
2. `LineCorner`, `IntersectDirected`: RefineEdges `IntersectLines` and `GrowQuad`, QuadDecode `UpdateFitQuad` corner
   loop and the hull `corner_ok`. Callers keep their own determinant tests (`> 0.001` keep-corner, `< 0.001` reject,
   `< 1e-9` parallel), so NaN and boundary behaviour is as before.
3. `ProjectHomography` (eight entries plus last entry with denominator, and a plain 3x3 form): PolishFit `Project`,
   `CellCorners`, `ComputeBox`, `BuildScan`, the marked-cell pass.
4. `Det3`, `Adjugate3`, `SolveGauss<kStride>`: PolishFit `Invert3` and `Solve`, PoseEstimator `Mat3Det`, `Mat3Inverse`.
   The cofactor expressions of the two inverses are bitwise equal (negated minors), so PoseEstimator output is unchanged
   (`apriltag_pose_validate` output identical apart from its timing column). Each caller keeps its own singularity test.
5. `MonotoneChain`: QuadDecode integer hull and SmallQuad double hull (each keeps its own `cross`).
6. `MeanSide`, `MinSide`, `QuadCentre`, `QuadBounds`: TrackingDetector `Side` and centre, MultiLevelDetector `ShapeOf`,
   TagDecoder (min side and two bounding boxes), RefineEdges profile cell size, PolishFit centre and side.
   `MeanSide` accumulates `0.25 * hypot` per side; TrackingDetector's old `sum * 0.25` is bitwise the same (scaling by
   a power of two is exact). The element type is kept (float `struct quad` corners use the float `hypot`, as before).
7. `SideSampler` (in RefineEdges.cpp) and `TapAlong<S, kTruncate>` / `TapPos` (Geometry.h): the per-side set-up
   (normal with reversed border, midpoint, point at alpha, profile count, stored line) and the tap position arithmetic
   shared by `RefineEdgesT` (kExact, truncating), `ProfilePass` and `LevelPass` (flooring).
   Timing lesson: a first version of `TapAlong` returned a `TapPos` struct and made the default path 15 to 40% slower
   in CPU time (one-tag d2 0.061 to 0.085 ms), because MSVC no longer vectorised `ProfilePass`'s tap-position loop. It
   now writes through pointers and `ProfilePass` copies the normal into local floats first (reading `side.fnx` inside
   the loop alone cost 12%); one-tag d2 is back at 0.0606 against 0.0614 ms.

## Left alone, with reasons
- QuadDecode `FitLine` / `FitLineError`: integer moments with 128-bit fallbacks and float eigenvalue arithmetic that
  mimics upstream; not the same computation as the double weighted TLS of `FitSide` or the unweighted moments of
  `RefineEdgesT` (which also uses `atan2f`, `cosf`, `sinf`). Merging would change rounding.
- RefineEdges `SquareToQuad` (Heckbert: builds the homography from four corners and applies it) is not a copy of a
  projection; upstream `homography_project` calls stay (their input is a `matd_t`).
- PolishFit float-precision inline projections inside `BuildFit` / `ModelPass` (SSE, float g0..g7): different precision.
- TrackingDetector `CoarseStart` bounding box over two point sets, QuadDecode diagonal `hypot` at the rescue box:
  one-off, not copies.
- The "clamped variant" of bilinear sampling named in the brief does not exist in this tree (only the copies above).
- `kExact` keeps `std::trunc` in `TapAlong` and its double type; merging it with the flooring float taps would change
  which pixel a negative position uses.

## Gate (this branch against a build of bc8cb18, dumps compared byte for byte)
Builds: base `ext/avk-compare/build-geometry-base` (source copy `src-geometry-base`), branch `build-geometry`.
Scripts in the scratchpad `geometry/` directory (`gate.sh`, `fullgate.sh`, `seq.sh`, `timing.sh`).

- `ctest`: all 8 tests pass at every commit (unit_geom is new: known-value tests for every helper).
- Corpus d2 default, d1 default, d2 small-rescue+hull, d2 `APRILTAG_VK_POLISH=1`, d2 `PROFILE_FEATURES=15`,
  d2 `APRILTAG_VK_REFINE=exact`: byte-identical except frames that also differ between two runs of the base build.
  That noise is a one-frame, ~1e-3 px corner jitter (GPU ordering): noise.txt at d2, and at d1 a different 4 to 6 files
  per run (busy-background, motion-blur, noise, range, blur, dark-background, family16h5, steep-angle, ...). Four repeat
  runs of each executable on the five d1 files that differed give several variants in both executables, and the main
  variant is the same for base and branch.
- Ten sequences via `track_eval`, modes full and key: detections identical (base against base2 also identical).
- `apriltag_pose_validate`: identical output except the last (timing) column.

## Timing (interleaved ABBA, 3 rounds, `vk_eval --iterations 50 --warmup 5`, median per-frame CPU ms, ratio branch / base)
d2: one-tag 1.007, many-tags 1.022, dense-tags 1.009, jpeg 0.987, blur 0.995, motion-blur 1.005.
d1: one-tag 0.997, many-tags 1.000, dense-tags 0.993, jpeg 1.001, blur 1.005, motion-blur 1.007.
(All within about 2%, which is the spread between repeated runs of one executable.)

## Lines
Library `.cpp` files: +168 / -330 (net -162); `Geometry.h` +215 (documented, with every helper in one place);
`unit_tests.cpp` +170 (group `geom`, run by ctest as `unit_geom`).
