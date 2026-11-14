# Facade streams: detectors on a shared Device must not share a lane

Branch research/2026-11-facade-streams (base e9e6195). Build dir ext/avk-compare/build-streams (MSVC Release).

## Step 1: threaded test, unfixed code
- Added group `threads` (`facade_threads`) to tests/facade_test.cpp: shared device (expected_streams 2), detectors with stream 0, 0, 1, 3,
  one thread each, 300 Detect() calls over three many-tags frames, plus an extra detector made and destroyed 10 times meanwhile.
- Unfixed tree, plain run, 3 times: passes (about 0.8 s). The race is not visible in the output on this machine (one compute queue family, AMD).
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
