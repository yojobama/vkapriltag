# Facade candidate filters (branch research/2026-11-facade-filters, from main ffefa9b)

Task: `DetectorOptions` gains `min_white_black_diff`, `min_cluster_pixels`, `min_tag_pixels`, `probe_margin`.

## Done

- `Detector.h`: four fields under "Candidate filters" (defaults 5, 24, 0, 0.0).
- `Detector.cpp`: validation (`probe_margin` negative or not finite, `min_white_black_diff` above 255: `kInvalidArgument`);
  fields copied into `DetectorConfig` (sync, pipelined depth 1 and 2, tracking, multi-level base config); a level's own non-zero
  `min_tag_pixels` wins, else the detector-wide one; `probe_margin` is the `TagDecoder` constructor argument.
- `DetectorConfig::probe_margin` (new) carries the margin to `TrackingDetector` (keyframe decoder and async background decoder; the coarse
  tracking decoder keeps 0).
- Environment variables still win: `GpuDetector` and `TagDecoder` read them after the configured values, as before.
- Registry rows `APRILTAG_VK_MIN_DIFF`, `MIN_TAG_PX`, `PROBE_MARGIN` name the `DetectorOptions` field; `PERFORMANCE.md` options block regenerated.
- probe_margin semantics: the gate runs in `TagDecoder` (`MarginGateRejects`) only with `refine_edges` and exactly one family; a candidate whose
  unrefined decision margin is below the value is dropped before refinement and never decodes. `Timing::candidates` is unchanged by it.
- Tests: `facade_filters` (`TestFilters`, `TestFilterLevels`) on corpus `grey-margin` frame 0 (ids 551 largest, 134, 340 smallest); errors added
  to `facade_errors`. Thresholds: boundary points are about 4.8 x the side, so `min_tag_pixels = 400` (= `min_cluster_pixels = 800`) keeps only 551.
- SWIG: plain members appear automatically; `swig_test.py` sets `min_white_black_diff` and `probe_margin` (built and run, passes).
- Docs: `Detector.h`, `ARCHITECTURE.md` 9.3, `README.md`, `CHANGELOG.md` 2.0.0-pre2 (no version bump in CMake), `DECISIONS.md` row.
- Workflow: `release.yml` no longer sets the release `name:` (title is the tag); the unused `version` output line was dropped.

## Gates (MSVC via vcvars wrappers; scratch dir in the session scratchpad)

- main build (`base-build`, Release, Ninja): `ctest` 47/47.
- branch build (`my-build`, Release, Ninja, `-DVKAPRILTAG_BUILD_SWIG=ON`, SWIG 4.4.1, Python 3.14): `ctest` 49/49 (the two extra are `facade_filters`
  and `swig_python`; `options_lint` passes after the PERFORMANCE.md regeneration). `git status` shows no golden, gpu_trace or plan changes.
- Corpus, `run_corpus.sh <vk_eval> .../napa-corpus <out> --decimation 2` plus the same with `--facade`, then `score_corpus.py`: main and branch
  score tables are byte-identical in both modes (7348 tags, 7146 found, 1 false).
