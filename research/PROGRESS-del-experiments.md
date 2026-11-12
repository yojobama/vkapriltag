# Progress: delete unused experiments and dead code (branch research/2026-11-del-experiments, base 02f0162)

Pure deletion; default behaviour byte-identical. One commit per item.

## Build and gate setup
- Base source: `ext/avk-compare/src-del-base` (git archive of 02f0162); base builds `build-del-experiments-base` (repo CMake with tests, vk_eval, track_eval).
- New: `build-del-experiments` (same configuration). Wrappers: `$S/del-exp/{cfg,build,ctest}.bat <src apriltags_vulkan> <build>`.

## 1. TrackingDetector (commit 1)
Removed: `roi_search`/`roi_pad_*`/`search_tag_px` (+ `APRILTAG_VK_TRACKING_SEARCH`, `SearchQuad`, the libapriltag `search_td_`),
`capture`/`capture_speed_px` (+ `APRILTAG_VK_TRACKING_CAPTURE`, `_CAPTURE_SPEED`, `CaptureStart`), `max_speed_px` (+ `_MAX_SPEED`),
`Path::kSearch`/`kCapture`, `frames_search`/`frames_capture`, the track acceleration `a` and `has_v` (only CaptureStart used them),
the write-only `alive` vector, the `strict` flag of `Accept`/`Solve` (always true once the rescue pass is gone), track_eval `keycap`/`asynccap`.
Kept: `CoarseStart` (decimated start of tentative tracks in async mode) with its crop and decoder, renamed from capture to coarse
(`coarse_pad_frac`, `coarse_converge_px`, `EnsureCoarse`); `max_first_shift_px` (used by `Solve`); `--spin-gap` of track_eval (harness gap only).
