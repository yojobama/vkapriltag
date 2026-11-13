# PROGRESS cpu-tail

Branch `research/2026-11-cpu-tail` from 1879a92. Base source worktree: `$A/src-cpu-tail-base` (1879a92).

## Step 1: go/no-go (head + blob_diff only against the full frame)

Temporary hack (uncommitted, not kept): `HACK_HEAD_ONLY` environment variable makes `RecordFusedTail` stop after the
boundary-count copy (host-read barrier and completion flag set). Builds: `$A/build-cpu-tail` (hack), `$A/build-cpu-tail-base`.

Commands: `ct_gonogo.sh` (desktop, under `benchlock.sh`) and `ct_pigonogo.sh` (Pi, under `flock /tmp/pi_bench.lock sh ~/pi_clean.sh`):
for each of one-tag, many-tags, dense-tags, jpeg and decimation 2 and 1, three rounds of full, head, head, full with
`vk_eval --iterations 5 --warmup 1`; medians of the per-frame `gpu=` and `total=` over all runs (`ct_med.py`).
The head-only frame reports no detections, so `total` there is just the GPU phase.

Medians in ms (gpu = host-timed GPU phase; full -> head only; saving):

| set | machine | dec | full gpu | head gpu | saving (us) |
| --- | --- | --- | --- | --- | --- |
| one-tag | Pi | 2 | 0.681 | 0.520 | 161 |
| one-tag | Pi | 1 | 1.227 | 1.099 | 128 |
| many-tags | Pi | 2 | 0.859 | 0.633 | 226 |
| many-tags | Pi | 1 | 1.305 | 1.177 | 128 |
| dense-tags | Pi | 2 | 1.053 | 0.766 | 287 |
| dense-tags | Pi | 1 | 2.000 | 1.590 | 411 |
| jpeg | Pi | 2 | 1.768 | 1.018 | 750 |
| jpeg | Pi | 1 | 4.601 | 3.150 | 1451 |
| one-tag | desktop | 2 | 0.259 | 0.226 | 33 |
| one-tag | desktop | 1 | 0.330 | 0.284 | 46 |
| many-tags | desktop | 2 | 0.283 | 0.245 | 38 |
| many-tags | desktop | 1 | 0.403 | 0.312 | 92 |
| dense-tags | desktop | 2 | 0.376 | 0.301 | 75 |
| dense-tags | desktop | 1 | 0.512 | 0.473 | 39 |
| jpeg | desktop | 2 | 0.376 | 0.272 | 104 |
| jpeg | desktop | 1 | 0.610 | 0.393 | 217 |

(The desktop was shared with other agents' builds, so absolute times are about 15% above the base figures in the brief.)
Verdict: GO. The Pi saving on one-tag at decimation 2 is 161 us against the 60 us gate. The upper bound ignores the cost of the CPU tail
itself (77 us for 4,400 points in the micro-benchmark) and the second readback wait.

## Step 2 and 3: implementation (in progress)

- `library/src/gpu/CpuTail.cpp` / `internal/vkapriltag/gpu/CpuTail.h`: the host port of hash_group, the extents reduction,
  select_blobs, scatter and sort_points_local. Blobs are numbered by first boundary point; ties in the angle key keep boundary-buffer order.
  The gradient weight W is recomputed from the caller's full-resolution frame (box mean, copy or identity), not from the device-local decimated image.
- `GpuDetector`: `RecordFusedTail` split into `RecordBoundary` + `RecordTailRest` (command stream unchanged); `HeadOnly` and `TailRest`
  recordings (`kHeadOnly`, `kTailRest`); `DetectFused` predicts from the last frame's boundary-point count.
- Options `APRILTAG_VK_CPU_TAIL` (off) and `APRILTAG_VK_CPU_TAIL_MAX_POINTS` (10000). `PipelinePlan` gains `cpu_tail_want`, `cpu_tail_max_points`, `cpu_tail`.
- On unified memory the two boundary buffers become HostVisibleCached when the mode is wanted. Discrete GPUs: mode off for now.
- Only frames uploaded by `Detect(gray)` (not imported frames) take the CPU tail.
