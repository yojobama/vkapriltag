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

## 2. WorkerPool (commit 2)
Removed: spinning (`spin_us`, `spin_yield`, `spin_workers`, `cpu_spin_us`, `APRILTAG_CPU_SPIN_US/_YIELD/_WORKERS`, `Warm`, `SpinForWork`,
the lock-free spinner path of `WorkerMain`), cold inline batches (`inline_max`, `inline_idle_us`, `cpu_inline_*`, `APRILTAG_CPU_INLINE*`),
`pin_caller`/`cpu_pin_caller`/`APRILTAG_CPU_PIN_CALLER`, the pool registry and `WorkerPool::WarmAll`.
The warm hooks served only spinning (`Warm()` returned at once with spin off): removed `KeepAlive::Options::on_warm`, the `warm_pool`
argument of `NoteFrame` and the keep-alive `armed_pool_`/`warm_only` path. The GPU pre-warm itself (`Touch`, `ExpectGpuFrame`, `Prewarm`,
`NoteFrame()` cadence recording, `TrackingDetector`'s calls) is unchanged. One timing nuance: `NoteFrame(true)` used to wake the keep-alive thread;
`NoteFrame()` now only records the arrival (the thread is woken by `Touch`/`ExpectGpuFrame` as before).
Kept: `cpu_big_cores`/`APRILTAG_CPU_BIG_CORES`, `APRILTAG_CPU_THREADS`. Dropped the cache-line alignment of `generation_` (spinner-only layout).
