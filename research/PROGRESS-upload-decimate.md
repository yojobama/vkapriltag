# PROGRESS upload-decimate (branch research/2026-11-upload-decimate, base 1879a92)

Idea: at decimation above 1, decimate on the CPU while uploading (read the caller's frame once, write the decimated
image into a host-visible decimated buffer), skipping the full-frame upload and the GPU decimate dispatch.

Build dirs: `$A/build-upload-decimate` (branch), `$A/build-upload-decimate-base` (1879a92 from `git archive`); Pi
`~/w/upload-decimate/{new,base}/build`. Helper scripts (scratchpad): `ud-build.sh`, `ud-cmp.sh`, `ud-piab.sh`, `ud-pitests.sh`.

## Implementation (commits 4902f2e, 68e8c24, 786d06f, 5a3703f)
- `gpu/CpuDecimate.cpp/.h`: `CpuDecimate(src, w, h, d, box, wide, dst)`. Box decimation `(sum + d*d/2) / (d*d)`, row by row, source read once
  sequentially. NEON (aarch64) and SSE2 (x86-64 baseline) for d = 2 and 4; other d: vertical column sums then per-output division
  (compile-time constant for d = 3). Top-left copy mode (`box` off): SIMD at d = 2, scalar otherwise. `wide` writes one uint32 per
  pixel (the no-8-bit-storage layout) via a row buffer. All modes are covered, so no GPU fallback is needed for any mode.
- `PipelinePlan::cpu_decimate` (printed by Describe; `want_cpu_decimate` is the resolve-time request, `FinalisePlan` confirms it):
  on when `decimation > 1`, `APRILTAG_VK_CPU_DECIMATE` is not 0, `decimated_buf_` was allocated mapped and coherent
  (`MemoryKind::DeviceLocalMapped`, new `BufferOutcome::decimated_direct_write`) and there is no transfer-queue lane.
  Otherwise the GPU path is unchanged. `minmax_decimate` (fused decimate + min/max, box off) is turned off when the CPU decimates, so
  `block_minmax_u8` runs on the decimated buffer instead.
- `GpuDetector`: `UploadDecimated` in `DetectFrame` and `DetectUnfused` (no full-frame write or staging copy for plain frames);
  `RecordThreshold` skips the decimate dispatch and `RecordFrameHead` skips the staging copy when `cpu_decimate`. Imported frames
  (host pointer, dma-buf, YUYV) still run their own `decimate` dispatch into the same buffer, keyed as before; `gray_buf_` is still
  allocated (pipelines bind it). Option `APRILTAG_VK_CPU_DECIMATE` (tuning, disable-only).
- Test: `unit_decimate` compares every mode with a reference for 12 sizes, d = 2..8, three patterns, 8-bit and 32-bit outputs.

## Results
- 1. Micro-benchmark (`decbench.cpp`, hot cache, best of 7x200, desktop, MSVC /O2), us per frame:
  | size | memcpy | d2 box | d2 copy | d3 box | d4 box | d4 copy |
  | --- | --- | --- | --- | --- | --- | --- |
  | 640x480 | 4.9 | 8.4 | 2.7 | n/a | 7.3 | 5.9 |
  | 1280x720 | 15.2 | 26.5 | 11.4 | n/a | 21.8 | 17.5 |
  | 1920x1080 | 109 | 56 | 25 | 282 | 47 | 38 |
  (The first d3 version, 9 loads per output, took 915 us at 1080p.) At 640x480 and 1280x720 the cache-resident memcpy is
  faster than the box kernel, but the kernel replaces the memcpy and the GPU dispatch together; see the end-to-end numbers.
- 2. ctest on the desktop: 45/45 (golden detections, facade goldens, unit_plan, options_lint, `gpu_trace_*` regenerated).
  Diff of the RX 9060 XT `gpu_trace` goldens: only `-dispatch decimate_box[_dot]_u8` (12 files) and one `-barrier compute` per
  trace (14 files); `box_decimate_off_d2`: `block_minmax_decimate_u8` -> `block_minmax_u8`. Mali-G610 goldens regenerated on the Pi,
  same pattern. `unit_plan` goldens: a `cpu_decimate` line per case (1 where decimation > 1 and the option is not 0).
  Pi ctest: 45/45.
- 3. Corpus dumps, desktop, decimation 2: all 26 scenario dumps byte-identical to the base build. Decimation 1: four scenarios
  differ by 1 to 2 lines (code path unchanged: run-to-run noise).
- 4. Desktop timing, `ab.sh`, 3 rounds interleaved, corpus at decimation 2 (median over all frames, ms; total / gpu):
  | | base (3 runs) | new (3 runs) |
  | --- | --- | --- |
  | all frames total | 0.349 0.355 0.341 | 0.323 0.308 0.305 |
  | all frames gpu | 0.233 0.236 0.236 | 0.196 0.188 0.188 |
  | 640x480 total | 0.267 0.261 0.261 | 0.257 0.243 0.243 |
  | 1280x720 total | 0.378 0.384 0.376 | 0.346 0.331 0.328 |
  | 1920x1080 total | 0.528 0.535 0.522 | 0.440 0.420 0.410 |
  cpu median about 0.105 in both (the CPU decimation is inside the gpu part of the timing, as the upload was). Found/false/error
  identical (7146/7348, 1 false, 0.017/0.099 px). Gain: total about -10% overall, -6% at 640x480, -12% at 1280x720, -21% at 1920x1080;
  gpu part -19%. The mapped (BAR) write over PCIe is not a loss here.
- 5. Pi (Orange Pi 5 Plus, Mali-G610), `~/w/upload-decimate/piab.sh ab2 3 2`: 12 interleaved corpus runs (base new new base ...), decimation 2,
  no 1920x1080 frames, medians over all frames (ms; total / gpu):
  base: 0.741 0.763 0.770 0.744 0.761 (and one outlier run, 0.839, with the whole machine slow) / gpu 0.585 0.601 0.594 0.581 0.601;
  new: 0.704 0.714 0.717 0.728 0.711 0.709 / gpu 0.542 0.551 0.557 0.546 0.549 0.549.
  Per size (base about, new about): 640x480 0.70 -> 0.66, 1280x720 0.96 -> 0.87. Gain: total about -5.5%, 640x480 about -6%, 1280x720 about -9%,
  gpu part about -8%. Found 4819/4972, 35 false, error 0.017/0.100 px in every run; all 26 dumps byte-identical to base.
- 6. Desktop gate, same `new` binary with `APRILTAG_VK_CPU_DECIMATE=0` (GPU decimation) against default (CPU), whole corpus dumps
  (`ud-gate.sh`): decimation 4: identical; decimation 2 with `BOX_DECIMATE=0`: identical; decimation 4 with `BOX_DECIMATE=0`: identical;
  `FUSE_SUBMITS=0` (unfused flow): identical; `FORCE_NO_8BIT=1` (32-bit layout): identical except one line of `noise` (about 0.003 px) and
  `PRERECORD=0`: one such line (the known noise frame). Decimation 3 (needs width divisible by 3, so only the 300 1920x1080 frames of
  the corpus, `ud-d3.sh`): box on: one line differs by under 0.001 px (noise), `BOX_DECIMATE=0`: identical.
  Facade tests (`facade_stride`, `facade_pipelined`, ...) pass in ctest: the stride path packs the frame before `Detect`, and the pipelined mode
  waits for the previous frame's GPU work before the next write, as for the grey buffer, so no upload is shared.
- 7. Sequences (10 x 200 frames, `track_eval --modes full,key --decimation 2`, `ud-seq.sh`): the detections of base and new are identical
  in all 20 dumps (only the `# path .. ms` timing comment lines differ).
- 8. Remaining per-frame copy traffic (read, not changed): `FinishFused` memcpy of the selected extents (selected blobs x 32 bytes: a few hundred bytes to
  a few KB); `FramePipeline` copies the line-fit points (8 bytes per point, up to tens of KB on busy frames) once per frame in pipelined mode;
  the facade copies a strided input into a packed buffer (a full-frame memcpy, only when stride != width); `TrackingDetector` async mode copies the full
  frame into `bg_copy_`; the facade's result copy is the detections list (tiny). The full-frame upload was the dominant copy and is gone at decimation above 1.
- Files touched: `gpu/CpuDecimate.cpp/.h` (new), `gpu/GpuDetector.cpp/.h` (upload, `RecordThreshold`, `RecordFrameHead`, buffer outcome),
  `gpu/PipelinePlan.cpp/.h`, `common/Options.cpp`, `library/CMakeLists.txt`, `tests/unit_tests.cpp`, `tests/CMakeLists.txt` (shared: one test group),
  goldens, `PERFORMANCE.md` options table, `DECISIONS.md`, `CHANGELOG.md`. `StageTable.cpp` and the shaders were not needed.
- 9. Pi gate (`ud-pigate.sh`, vk_eval at iterations 1, CPU against `CPU_DECIMATE=0`, whole Pi corpus): decimation 4, d2 `FORCE_NO_8BIT`, d4 `BOX_DECIMATE=0`: all
  26 dumps identical; d2 `BOX_DECIMATE=0`: `noise` one line, `FUSE_SUBMITS=0`: `large-tags` and `noise` one line each, all under 0.003 px (run-to-run noise).
  Pi micro-benchmark (`decbench`, g++ -O2, hot cache) us, memcpy / d2 box / d2 copy / d4 box / d4 copy: 640x480 9.9 / 12.6 / 5.7 / 14.0 / 18.2;
  1280x720 44.3 / 49.7 / 35.6 / 47.2 / 56.2; 1920x1080 122 / 146 / 98 / 125 / 141; d3 box at 1080p 3,043 us (too slow: not vectorised).
  Decision: `want_cpu_decimate` only for decimation 2 and 4; other decimations keep the GPU path (kernels remain correct for them, tested by `unit_decimate`).
- 10. Desktop with `VK_EVAL_GAP_MS=16` (one interleaved round, `ab.sh`): base total median 0.494 (gpu 0.331), new 0.550 (gpu 0.370): 640x480 0.421 -> 0.475,
  1280x720 0.523 -> 0.556, 1920x1080 0.699 -> 0.735; mean 0.612 -> 0.601. A loss in the median at 16 ms gaps (the GPU and the CPU clock down between frames;
  the mapped write over PCIe and the cold source frame cost more then). Two more rounds queued (`ud-ab16b`). The Pi was unreachable
  (connection timed out) from this point: the Pi gap run, the Pi ctest of the d2/d4 restriction and the Pi sequences were not run.
- 11. 16 ms gaps, two more interleaved rounds (total median / mean, ms): base 0.454 / 0.536 and 0.471 / 0.508; new 0.461 / 0.497 and 0.454 / 0.491.
  So at 16 ms gaps the median is neutral (the 0.494 -> 0.550 of item 10 was a noisy single round) and the mean is about 5% lower. Per size the
  medians are within noise (640x480 0.384/0.404 -> 0.377/0.374, 1280x720 0.463/0.481 -> 0.487/0.475, 1920x1080 0.590/0.621 -> 0.608/0.623).
- 12. Rebuilt with the decimation 2 and 4 restriction: desktop ctest 45/45. Pi: unreachable, not rebuilt or re-run after item 9's change (the restriction only
  removes the GPU-path difference for decimation 3 and other values, which the Pi corpus does not use).
