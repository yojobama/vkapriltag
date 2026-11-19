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

## 3. Dead code (commit 3)
Removed: `QuadCorners` (Types.h, unused), the duplicate `#include <bit>` in TagDecoder.cpp, the unread `dwdh_pc` in `RecordUfChunk` (cppcheck unreadVariable).
The write-only `alive` vector went with commit 1.
Kept: `GpuDetector::last_selected_extents` and its per-frame copy: `FramePipeline::last_selected_extents()` exposes it publicly and the
validate tool (`validate_against_libapriltag_opencv.cpp`, blob point-count histogram) reads it; the corpus tools do not.
cppcheck (`--enable=warning,style --std=c++20 -I include src`) leftovers left alone: PoseEstimator array bounds (false positives), push-constant
struct members reported as unused (read by shaders), `TagDecoder.cpp` knownCondition `!keep` (false positive: `keep` is set in the branch above),
Context.cpp 32-bit `long` range compare, QuadDecode integer-overflow report, `os.fill` ignored return values, uninitialised-member and const-style notes.

## 4. GpuDetector unfused flow (commit 4)
`DetectUnfused`: removed `const bool fuse = false` and every `if (fuse)` branch, the `finish_frame` lambda (inlined: it ran once), the
`final_*` copies of the counts (they equalled the unfused values), the unused fused-only indirect dispatches and `qbp_pc_count`.
The local `record_uf_chunk` lambda was identical to `RecordUfChunk(cmd, n)` with its default `trailing_compress`, so it calls that.
Push-constant structs keep their `count_from_buffer` field (passed 0), so the shaders are unchanged. New `GpuDetector::CollectGpuTimings()`
holds the timestamp / dispatch-probe post-processing that `FinishFused` and `DetectUnfused` both had (the two copies differed only in line wrapping).
The unfused flow itself (fallback for devices without host-cached memory, `APRILTAG_VK_FUSE_SUBMITS=0`) is unchanged in what it records and submits.

## Gate (desktop, RX 9060 XT; scripts in the session scratchpad `del-exp/`: gate.sh, golden_env.sh, seq.sh, rep.sh)
- ctest 5/5 pass, goldens unchanged.
- Golden compare passes (0 differing frames, all 5 profiles) under: default, FORCE_NO_8BIT, FORCE_NO_SUBGROUP, FORCE_NO_INT64_ATOMIC,
  FORCE_NO_CONDITIONAL_RENDERING, FUSE_SUBMITS=0, and FUSE_SUBMITS=0 with NO_8BIT / NO_CONDITIONAL_RENDERING / NO_SUBGROUP+NO_INT64_ATOMIC.
- Full napa corpus dumps vs a build of 02f0162 (base run twice for the noise level; differing non-comment lines, max numeric difference):
  d2: base vs base2 1 line (noise), base vs new 0 lines. d1: base vs base2 4 lines (max 0.0012), base vs new 6 lines (max 0.0022).
  FUSE_SUBMITS=0 d2: base vs base2 1, base vs new 2 lines (max 0.0027); d1: 3 vs 4 lines (0.0022). FUSE_SUBMITS=0 with UF_VERIFY=0 d2 and with
  TIMESTAMPS=1 d2: 0 lines differ. No ID or structural difference anywhere.
- track_eval, ten sequences, modes full,key,async,asyncsync, 2 rounds: full and key identical to base in every column (recall, false, median/p95/max error,
  path shares); async and asyncsync identical in error and path shares, recall differs only on enter-3tag-640 and medium-2tag-640, where the background
  timing makes recall bimodal in the base too: enter-3tag-640 async over 8 interleaved runs base 84.8 96.2 83.5 84.8 94.9 84.5 85.0 94.9, new 84.0 96.4 84.8 83.8
  95.7 94.7 83.8 89.6; asyncsync base 85.5 98.5 84.8 84.8 94.2 84.5 85.3 98.0, new 84.8 85.3 84.0 84.3 84.8 98.5 85.5 84.5 (%).
