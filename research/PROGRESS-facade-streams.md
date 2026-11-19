# Facade streams: detectors on a shared Device must not share a lane

Branch research/2026-11-facade-streams (base e9e6195). Build dir ext/avk-compare/build-streams (MSVC Release).

## Step 1: threaded test, unfixed code
- Added group `threads` (`facade_threads`) to tests/facade_test.cpp: shared device (expected_streams 2), detectors with stream 0, 0, 1, 3,
  one thread each, 300 Detect() calls over three many-tags frames, plus an extra detector made and destroyed 10 times meanwhile.
- Unfixed tree, plain run, 3 times: passes (about 0.8 s). The race is not visible in the output on this machine. (Correction: this desktop has two compute-capable families, so the earlier "one compute queue family" was wrong. Context.cpp creates one queue per family for expected_streams 2, so queue_count is 2 with streams interleaved: streams 0 and 0 share slot 0, streams 1 and 3 share slot 1. The single-queue wrap case, as on the Mali, is not exercised here.)
- Unfixed tree with VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation: test still passes, but the layer reports
  UNASSIGNED-Threading-MultipleThreads-Write ("VkCommandPool simultaneously used in thread A and thread B") from vkAllocateCommandBuffers
  and vkBeginCommandBuffer, and a vkQueueSubmit pending-state error. Count: THREADING ERROR lines in val_unfixed.txt below.
- Unfixed: 20 THREADING ERROR lines (32 lines mentioning a VUID/spec state) in one run. Fixed tree under the same layer: 0 errors.

## Step 2: fix (Detector.cpp, TrackingDetector, MultiLevelDetector)
- Detector::Impl: when the device is passed in (or stream > 0), `lane = ctx.CreateLane(ctx.StreamSlot(options.stream))`; no fallback to the default lane.
  A detector that opens its own device with stream 0 still uses the default lane (GPU traces unchanged).
- Audit of what submits through the default lane, per mode, before the fix:
  - sync / pipelined depth 1: GpuDetector(ctx, *lane_ptr, config), lane_ptr was the default lane for stream 0 or an out-of-range stream.
  - pipelined depth 2: second GpuDetector on CreateSiblingLane(*lane_ptr) (own lane, same queue mutex): unchanged.
  - tracking (keyframe and async): TrackingDetector built its GpuDetector with GpuDetector(ctx, config), i.e. ALWAYS the default lane, ignoring stream.
    The async background thread also runs gpu_->Detect. Now takes the facade's lane (new trailing `vk::Lane *lane = nullptr`).
  - multi-level: MultiLevelDetector built one GpuDetector per level on the default lane, ignoring stream. Same new trailing parameter; levels run in turn on one thread.
  - keep-alive / pre-warm: GpuDetector makes its own sibling lane (CreateSiblingLane(lane_)) for the helper thread: fine.
  - Context::BeginCommands / SubmitAndWait (default lane forwarders): no caller inside library/src.
  - Other callers of the old constructors (tools/track_eval) are own-context and single detector: unchanged by the default argument.
- Audit of other per-Context mutable state: PipelineCache::Save is under save_mutex_ (and last_synced_hash_ is only touched inside it); the temporary
  file name uses an atomic counter; DispatchProbe and trace state are static with a mutex; Options registry is a const static map.
  Nothing else found that is a data race on the Detect() or constructor path. (Not examined: Vulkan itself.)
- Docs: Detector.h (stream comment, threading rule), ARCHITECTURE 8.2, DECISIONS section 5 row, CHANGELOG Unreleased "Fixed".

## Gate (desktop, build-streams)
- ctest: 47/47 pass (46 plus facade_threads); no golden or gpu_trace file changed (git status clean apart from sources and docs).
- `ctest -R facade_threads --repeat until-fail:20`: passed 20 of 20 (about 0.65 s each).
- Corpus, decimation 2, --iterations 2: found 7146/7348, false 1, error 0.017/0.098 px.
- Orange Pi: unreachable ("Host does not exist"); nothing ran on the Pi.

## Step 3: threaded test in every mode (follow-up)
Group `threads` now runs, on a shared device (expected_streams 2) with detectors on streams 0, 0, 1, 3 (one thread each) plus an extra
detector created and destroyed 6 times meanwhile, six modes; `threads_<mode>` runs one. Checks:
- sync (300 calls), levels (100 calls, levels dec 2 and 1, kSync): every result equals the single-threaded reference ids for that frame (three many-tags frames in rotation).
- pipelined depth 1 and 2 (200 calls): ready results arrive in order; the k-th ready result (and the Flush result) equals the reference for frame k mod 3.
- tracking and tracking async (200 calls): the same one-tag frame repeated (rotating unrelated frames is no sequence); every ready result must have no
  id other than 316 and 316 exactly once in at least half of all calls (818 of 818 on the fixed tree).
Whole group about 3.6 s.

Validation-layer counts (`UNASSIGNED-Threading` lines / other "Validation Error" lines), one run per mode:
| mode | unfixed (e9e6195 library, this test) | fixed |
| --- | --- | --- |
| sync | 20 / 43 | 0 / 0 |
| pipelined1 | 20 / 40 | 0 / 0 |
| pipelined2 | 13 / 18, process died under the layer | 0 / 0 |
| tracking | 10 / 20, process died under the layer | 0 / 0 |
| trackingasync | 20 / 41 | 0 / 0 |
| levels | 20 / 40, process died under the layer | 0 / 0 |
Unfixed without the layer: threads_pipelined2 failed 2 of 3 runs (exit 1); tracking and levels passed 3 of 3. The layer's other errors on the unfixed tree
include VUID-vkResetFences-pFences-01123 (a fence reset while its queue work is pending). Fixed tree: no error, no wrong result in any mode.
Gate: ctest 47/47, no golden or gpu_trace file changed; `ctest -R facade_threads --repeat until-fail:20` 20 of 20. Library unchanged, so no corpus run.

## Step 4: Orange Pi round (2026-11-19, Mali-G610, one compute queue)
Unfixed tree = e889f8b with the library files of e9e6195 (as on the desktop). Every run limited to 120 s (a hang counts as a failure).

| mode | unfixed, no layer, 3 runs | unfixed, ThreadSanitizer | unfixed, validation layer (sync validation) | fixed, 3 runs / TSan / layer |
| --- | --- | --- | --- | --- |
| sync | 0/3 (2 hung, 1 with 201 wrong) | 68 race reports | 30 errors, abort | 3/3 / 0 / 0 |
| pipelined1 | 0/3 (2 hung, 1 abort) | 4 | 14, abort | 3/3 / 0 / 0 |
| pipelined2 | 0/3 (323 to 393 wrong) | 4 | 30, abort | 3/3 / 0 / 0 |
| tracking | 3/3 | 52 | 24, abort | 3/3 / 0 / 0 |
| trackingasync | 2/3 (1 hung) | 8 | 16, abort | 3/3 / 0 / 0 |
| levels | 0/3 (about 210 wrong) | 6 | 40, abort | 3/3 / 0 / 0 |

Layer errors on the unfixed tree: VUID-vkResetFences-pFences-01123 and VUID-vkQueueSubmit-fence-00064. A hung run (sync, first attempt, without
a time limit) sat for 5 h 50 min: the main thread joins the detector threads, which wait in libmali's fence wait that never completes.
Fixed tree: ctest 48/48, `ctest -R facade_threads --repeat until-fail:20` 20 of 20, ASan+UBSan and TSan clean on unit_tests, every facade
group, vk_eval (dec 1 and 2; default, hull+small+polish, peaks; one thread) and track_eval; validation layer clean on all six threaded modes,
vk_eval and track_eval.
