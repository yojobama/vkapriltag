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

### Equivalence test (`tests/cpu_tail_test.cpp`, ctest `cpu_tail`)

Two `GpuDetector`s on the same frames (one with `APRILTAG_VK_CPU_TAIL=1` and the threshold raised so every frame takes the CPU tail), per frame:
the selected blobs (canonical order: smallest point, then size), each blob's extents bit for bit, its sorted point list (a multiset comparison
where equal angle keys swap, counted as "tie reorders"), then `QuadDecode` + `TagDecoder` on both, ids, Hamming distances and corners.

- Desktop (RX 9060 XT, discrete, boundary points reach the host through a staged copy of the first `max_points` entries):
  100 committed frames at decimation 2 and 1: 0 failures, corners bit-identical. 8 blobs differ in presence: a blob whose fill ratio is exactly
  0.5 (13x13 or 15x15 boxes with 26 or 30 points) is rejected by the GPU's division and kept by the exact division on the host. The test
  tolerates a one-sided blob only when its fill ratio is within 1e-6 of 0.5 or 2.5 and counts them ("threshold ties").
- Pi (Mali-G610): every frame differs in the gradient weight W of some points by exactly 1 (GPU one lower): the Mali's `sqrt` of a perfect square
  comes out just below the integer, so `int(sqrt(g))` is one less. (Result of the bias experiment below.)

The first desktop timing with `getenv` inside the per-point weight computation took 1.2 ms (the call is slow on Windows): fixed by reading it once;
it is an experiment only and must not stay (options_lint rejects it).

Desktop, full corpus offline (`cpu_tail_test --dir $A/napa-corpus`, decimation 2 and 1, 8,886 frames, 217,295 selected blobs, 13,562 detections):
0 failures, 1,515 frames with tie reorders (equal angle keys), 863 threshold ties, corners within 0.00275 px (the noise of the unsorted ties).

Pi bias experiment (committed binary with `CT_SQRT_BIAS`, removed since): the Mali's `sqrt` is not simply low: the observed failures at
perfect squares n*n for n in {7, 13, 14, 15, 30, 52, 56, 63, 104, 105, 112, 117, 120, 125, 126, 221} and exact results for
n in {1, 2, 3, 4, 5, 94, 99, 146, 173, 205} are reproduced exactly by `1 / (1 / sqrtf(v))` in float32 (`ct_rsqrt.py`: 35 of the 361
perfect squares fall below n, none of the observed exact ones do). So `sqrt` on the Mali is the reciprocal of the reciprocal square root.
`CpuTailConfig::sqrt_via_rsqrt` (plan field `cpu_tail_rsqrt`, true for ARM vendor id 0x13B5) emulates it.

Desktop track_eval (full,key; off against on) ran for the ten sequences; the SUMMARY lines hold timing only. Recall comparison: see below.
