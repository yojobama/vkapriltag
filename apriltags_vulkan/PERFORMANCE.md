# Performance and portability notes

Reference page for running and measuring the library: how to be sure you are on the GPU (section 1), every environment
switch (section 2, generated), the current measured numbers (section 3), per-device guidance (section 4) and how to
measure and validate (section 5). How the pipeline works is in [ARCHITECTURE.md](ARCHITECTURE.md), why it is built this
way in [DESIGN.md](DESIGN.md), and what was tried, with numbers, in [DECISIONS.md](DECISIONS.md).

The chronological record of the measurements (the old sections 3a to 8be) is frozen in
[archive/PERFORMANCE-history.md](archive/PERFORMANCE-history.md); a section number such as "8ac" in the changelog or in
`research/` refers to that file.

Targets: embedded ARM Mali-G610 (the deployment target), discrete AMD (Radeon RX 9060 XT / RDNA4), and discrete NVIDIA.
The pipeline needs **only core Vulkan 1.1**. Optional extensions (8-bit storage, 64-bit atomics, conditional rendering,
host-pointer import, dma-buf import) and subgroup operations are used when the device has them, and each has a fallback
(the `APRILTAG_VK_FORCE_NO_*` variables in section 2 force them). The fallbacks reproduce the detections of the committed
golden set exactly; the adaptive contrast floor, the fused threshold, compacted labelling and run counting need 8-bit
storage, so without it the fixed contrast floor applies and detections can differ on noisy frames.

## 1. First: make sure you are actually on the GPU

This is by far the largest single performance factor, and it is a
configuration issue rather than a code one.

If no GPU Vulkan driver (ICD) is installed, the Vulkan loader happily hands
back Mesa's **lavapipe/llvmpipe** software implementation. It is functionally
correct and 50-500x slower, which looks exactly like "the GPU port is slow".

The detector **refuses a CPU device by default** and explains what to do:

```
The only Vulkan device available is a CPU/software implementation ...
  * To fix properly: install the Vulkan driver (ICD) for your GPU ...
  * To benchmark or test on the CPU anyway: set APRILTAG_VK_ALLOW_CPU=1
```

Startup always prints the selected device, its relevant limits, and the launch
geometry derived from them. Check that line first. Real output from the two
machines these notes were measured on:

```
apriltag_vulkan: using AMD Radeon RX 9060 XT (RADV GFX1200) (discrete GPU, Vulkan 1.4.354)
  limits: maxComputeWorkGroupInvocations=1024, ..., float64=yes, int64=yes, 8bit_storage=yes,
          subgroup_ballot=yes, subgroup_arithmetic=yes
  chosen geometry: wg1d=256, wg2d=16x16, scan_wg=1024, unified_memory=no, host_cached=yes, timestamps=yes
```

```
apriltag_vulkan: using Mali-LODX (integrated GPU, Vulkan 1.2.165)
  limits: maxComputeWorkGroupInvocations=1024, ..., float64=no, int64=no, 8bit_storage=yes,
          subgroup_ballot=yes, subgroup_arithmetic=yes
  chosen geometry: wg1d=128, wg2d=8x8, scan_wg=1024, unified_memory=yes, host_cached=yes, timestamps=yes
```

Note the Mali line: `float64=no, int64=no`. Nothing in the pipeline requires
either, and that is deliberate (see `archive/OPTIMIZATION_NOTES.md`, item 8).

Sanity checks: `vulkaninfo --summary` should list your GPU; on Linux you need
a working `/dev/dri`.

## 2. Environment knobs

All are optional; defaults are chosen from device limits. Every variable the library reads is a row of
the registry in `library/src/common/Options.cpp`, and the tables below are generated from it
(`vk_eval --list-options`); the `options_lint` test fails if they drift or if the library reads the
environment anywhere else. Each variable is read once, when the object it configures is constructed
(a detector, a decoder, the Vulkan context), or once per process for the few process-wide ones.

Precedence: an environment variable overrides the configuration field it matches. The Precedence
column names the exceptions: `APRILTAG_CPU_THREADS` is used only when `cpu_threads` is 0; switches marked
"can only enable" or "can only disable" never reverse a configured value; and
variables with no configuration field are environment only. For the reasoning behind a switch, search
`DECISIONS.md` for its name.

<!-- options:begin -->
### Features (change what is detected)

| Variable | Type | Default | Configuration field | Precedence | Effect |
| --- | --- | --- | --- | --- | --- |
| `APRILTAG_VK_SMALL_RESCUE` | bool | off | `DetectorConfig::small_tag_rescue` | environment wins | Rescue small tags: label small blobs, fit boxes, refit from the full-resolution image. |
| `APRILTAG_VK_HULL_QUAD` | bool | off | `DetectorConfig::hull_quad` | environment wins | Retry a failed quad fit on the convex hull of the blob (partly covered border). |
| `APRILTAG_VK_QUADFIT` | enum | dp | `DetectorConfig::quad_fit_method` | environment wins | Corner seeding: dp or peaks (other values are ignored). |
| `APRILTAG_VK_REFINE` | enum | profile | `TagDecoder refine_method` | environment wins | Edge refinement: exact or profile (other values are ignored); used when TagDecoderConfig::refine_edges is set. |
| `APRILTAG_VK_POLISH` | bool | off | `TagDecoder::SetPolish` | environment wins | Whole-tag photometric polish of decoded tags. |
| `APRILTAG_VK_PROFILE_FEATURES` | bit mask | 31 |  | environment only | Bit mask of the profile fit's refinements: 1 outliers, 2 coarse to fine, 4 halfway crossing, 8 clip bias, 16 widen blur. |
| `APRILTAG_VK_REFINE_ABORT` | float | -1 |  | environment only | Profile refinement gives a quad up unrefined when a side's usable profile fraction in the first pass is below this; negative: off. |
| `APRILTAG_VK_GREY_MARGIN` | bool | on |  | environment only | Retry an undecoded quad as its inner square (grey margin). |
| `APRILTAG_VK_SIDE_ONE_OUT` | bool | on |  | environment only | Refit a side whose border cells read white, one cell further in. |
| `APRILTAG_VK_THIRD_BIT` | bool | on |  | environment only | Accept a third corrected bit when the three wrong cells touch. |
| `APRILTAG_VK_FAMILY_LIMITS` | bool | on |  | environment only | Reject detections that needed more corrected bits than their family can bear. |
| `APRILTAG_VK_BORDER_PRECHECK` | bool | on |  | environment only | Reject a quad whose border cells are not mostly dark before refining. |
| `APRILTAG_VK_DARK_REJECT` | bool | on |  | environment only | Reject a quad whose payload has fewer white cells than any code. |
| `APRILTAG_VK_EARLY_REJECT` | bool | on |  | environment only | With grey margin on, the border and dark checks reject a quad only if its inner cells fail them too. |
| `APRILTAG_VK_PROBE_MARGIN` | float | 0 (constructor argument) | `TagDecoder probe_margin` | environment wins | Margin gate in grey levels before edge refinement; 0 is off. |
| `APRILTAG_VK_MIN_DIFF` | int | 5 | `DetectorConfig::min_white_black_diff` | environment wins | Minimum white-black difference for a block to be labelled. |
| `APRILTAG_VK_MIN_TAG_PX` | int | 0 | `DetectorConfig::min_tag_pixels` | environment wins | Smallest tag side in full-resolution pixels; raises the minimum cluster size. |
| `APRILTAG_VK_BOX_DECIMATE` | bool | on | `DetectorConfig::box_decimate` | environment wins | Average each decimation block instead of copying its top-left pixel. |
| `APRILTAG_VK_ADAPTIVE_FLOOR` | float | 2.0 | `DetectorConfig::adaptive_floor_scale` | environment wins | Contrast floor as a multiple of the frame's median block range; 0 keeps the fixed floor. |
| `APRILTAG_VK_TRACKING` | bool | off | `TrackingConfig::enabled` | environment wins | Track tags between keyframes instead of running the full detector every frame. |
| `APRILTAG_VK_TRACKING_KEYFRAME` | int | 8 | `TrackingConfig::keyframe_interval` | environment wins | At most this many consecutive frames skip the GPU (at least 1). |
| `APRILTAG_VK_TRACKING_MIN_MARGIN` | float | 10 | `TrackingConfig::min_margin` | environment wins | Smallest decision margin for a tracked tag. |
| `APRILTAG_VK_TRACKING_CONVERGE` | float | 0.05 | `TrackingConfig::converge_px` | environment wins | Corner movement in pixels below which a tracked tag has settled. |
| `APRILTAG_VK_TRACKING_ITERATIONS` | int | 3 | `TrackingConfig::max_iterations` | environment wins | Most decodes per tracked tag (at least 1). |
| `APRILTAG_VK_TRACKING_ASYNC` | bool | off | `TrackingConfig::async` | environment wins | Run the full detector on a background thread while tracked frames return at the tracked cost. |
| `APRILTAG_VK_TRACKING_ASYNC_INTERVAL` | int | 1 | `TrackingConfig::async_interval` | environment wins | The background detector starts at most every this many frames (at least 1). |
| `APRILTAG_VK_TRACKING_ASYNC_COPY` | bool | on | `TrackingConfig::async_copy` | environment wins | Copy each frame for the background detector. |
| `APRILTAG_VK_TRACKING_ASYNC_DROP` | bool | on | `TrackingConfig::async_drop_lost` | environment wins | A lost track stays out of the output until a background result finds it again. |

### Tuning (same detections, different cost)

| Variable | Type | Default | Configuration field | Precedence | Effect |
| --- | --- | --- | --- | --- | --- |
| `APRILTAG_VK_WG` | int | device | `ContextOptions::workgroup_size_override` | environment wins | 1D workgroup size (power of two, clamped to limits); must be positive. |
| `APRILTAG_VK_WG2D` | WxH | device | `ContextOptions::workgroup_size_2d_x/y` | environment wins | 2D workgroup size as <w>x<h>. |
| `APRILTAG_VK_PIPELINE_CACHE` | bool | on | `ContextOptions::use_pipeline_cache` | environment can only disable | Any value other than a non-zero one disables the on-disk pipeline cache. |
| `APRILTAG_VK_CACHE_DIR` | string | platform cache directory |  | environment only | Directory of the on-disk pipeline cache. |
| `APRILTAG_VK_SPIN_WAIT` | bool | off |  | environment only | Poll a fence for up to 2 ms before sleeping in the driver (read once per process). |
| `APRILTAG_CPU_THREADS` | int | 0 (hardware concurrency) | `DetectorConfig::cpu_threads` | configuration wins | Parallelism of the CPU tail, including the calling thread; used only when the configured value is 0. |
| `APRILTAG_CPU_BIG_CORES` | bool | on | `DetectorConfig::cpu_big_cores` | environment wins | Pin workers to the big cores where core capacities differ. |
| `APRILTAG_VK_MAX_POINTS` | int | 0 (dense worst case) | `DetectorConfig::max_boundary_points` | environment wins | Cap on boundary points per frame; must be positive. |
| `APRILTAG_VK_UF_CHUNK` | int | 2 | `DetectorConfig::uf_iterations_per_chunk` | environment wins | Labelling iterations issued per convergence check; must be positive. |
| `APRILTAG_VK_SORT_WG` | int | device maximum (256 integrated, 1024 other) |  | environment only | Workgroup size of the point sort. |
| `APRILTAG_VK_SORT_SPLIT` | int | 1024 (bounded by shared memory) |  | environment only | Blobs above this many points are sorted in runs of this length (power of two). |
| `APRILTAG_VK_SORT_RANK_ALL` | int | 256 |  | environment only | Blobs of at most this many points are sorted by ranking every key; 0 disables. |
| `APRILTAG_VK_FLAG_POLL` | bool | on with device-coherent host memory |  | environment only | Detect completion by polling a flag instead of waiting on a fence. |
| `APRILTAG_VK_KEEPALIVE_MS` | int | -1 (6 integrated, 0 other) | `DetectorConfig::keepalive_ms` | environment wins | Milliseconds between keep-alive dispatches; 0 disables. |
| `APRILTAG_VK_PREWARM_LEAD_US` | int | -1 (3000) | `DetectorConfig::prewarm_lead_us` | environment wins | Microseconds before the next frame is due that a pre-warm burst starts; 0 disables. |
| `APRILTAG_VK_PREWARM_BURST` | int | 2 | `DetectorConfig::prewarm_burst` | environment wins | Dispatches per pre-warm burst (at least 1). |
| `APRILTAG_VK_PREWARM_GAP_US` | int | 1000 | `DetectorConfig::prewarm_gap_us` | environment wins | Microseconds between the dispatches of a burst. |
| `APRILTAG_VK_PREWARM_GROUPS` | int | keep-alive default |  | environment only | Workgroups per pre-warm dispatch (at least 1). |
| `APRILTAG_VK_FUSE_SUBMITS` | bool | on when the readbacks are direct |  | environment can only disable | Record the last three submissions of a frame as one; 0 keeps the unfused flow. |
| `APRILTAG_VK_UF_VERIFY` | bool | off |  | environment only | Run a second merge that checks the labelling converged, with a retry path. |
| `APRILTAG_VK_PRERECORD` | bool | on when fused without a transfer lane |  | environment can only disable | Replay pre-recorded command buffers; 0 records every frame. |
| `APRILTAG_VK_DOT` | bool | on where supported |  | environment can only disable | Integer dot product box decimation; 0 uses the plain shader. |
| `APRILTAG_VK_FUSED_THRESHOLD` | bool | on with 8-bit storage |  | environment can only disable | Fused block min/max and threshold; 0 keeps the separate passes. |
| `APRILTAG_VK_UF_INIT_SCAN` | bool | on where supported |  | environment can only disable | Initialise the labelling with a workgroup scan; 0 uses uf_init and a compression. |
| `APRILTAG_VK_COUNT_RUNS` | bool | on for integrated GPUs |  | environment only | Count blob sizes per run piece (needs the init scan). |
| `APRILTAG_VK_SPARSE` | bool | on | `DetectorConfig::sparse_labelling` | environment wins | Dispatch the labelling over the cells that hold contrast only (needs fused threshold, init scan, count runs). |
| `APRILTAG_VK_MERGE_X4` | bool | on for integrated GPUs |  | environment only | Merge four pixels per thread (needs 8-bit storage and a width divisible by 4). |
| `APRILTAG_VK_MERGE_BAND` | int | 8 discrete, 0 integrated or sparse |  | environment only | Rows merged in bands of this many (power of two) before the band boundaries; 0 is one pass. |
| `APRILTAG_VK_LEAN_TAIL` | bit mask | 3 integrated, 0 other |  | environment only | Leaner tail: bit 1 fixed-grid kernels, bit 2 offsets from select. |
| `APRILTAG_VK_TRACKING_ASYNC_THREADS` | int | 1 | `TrackingConfig::async_threads` | environment wins | Threads of the background decoder; 0 is all cores. |
| `APRILTAG_VK_POLISH_N` | int | 8000 |  | environment only | Most pixels the polish fit uses per tag (read once per process). |
| `APRILTAG_VK_POLISH_BLUREXTRA` | float | 2.0 |  | environment only | Polish: extra blur width, in pixels, added to the window (read once per process). |
| `APRILTAG_VK_POLISH_WINMULT` | float | 0.8 |  | environment only | Polish: window size as a multiple of the blur width. |
| `APRILTAG_VK_POLISH_MINWIN` | float | 0.5 |  | environment only | Polish: smallest window, in pixels. |
| `APRILTAG_VK_POLISH_SMAXM` | float | 1.3 |  | environment only | Polish: largest blur width as a multiple of the starting estimate. |
| `APRILTAG_VK_POLISH_TOL` | float | 0.004 |  | environment only | Polish: convergence tolerance of the parameter step. |
| `APRILTAG_VK_POLISH_ITERS` | float | 6 |  | environment only | Polish: most iterations. |
| `APRILTAG_VK_POLISH_REACH` | float | 1.6 |  | environment only | Polish: how far from the edge, in blur widths, pixels still count. |

### Device forcing

| Variable | Type | Default | Configuration field | Precedence | Effect |
| --- | --- | --- | --- | --- | --- |
| `APRILTAG_VK_ALLOW_CPU` | bool | off | `ContextOptions::allow_cpu_device` | environment can only enable | Permit a software Vulkan device (benchmarking and CI only). |
| `APRILTAG_VK_DEVICE` | int | -1 (best score) | `ContextOptions::device_index` | environment wins | Pick a physical device by index. |
| `APRILTAG_VK_FORCE_NO_8BIT` | bool | off | `ContextOptions::force_no_8bit_storage` | environment can only enable | Use the 32-bit-per-pixel shader family even where 8-bit storage exists. |
| `APRILTAG_VK_FORCE_NO_SUBGROUP` | bool | off | `ContextOptions::force_no_subgroup` | environment can only enable | Use the scalar shader variants even where subgroup operations exist. |
| `APRILTAG_VK_FORCE_NO_INT64_ATOMIC` | bool | off | `ContextOptions::force_no_int64_atomics` | environment can only enable | Use the 32-bit atomic extents reduction. |
| `APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING` | bool | off | `ContextOptions::force_no_conditional_rendering` | environment can only enable | Skip converged compression passes with the in-shader guard. |
| `APRILTAG_VK_MAX_INVOCATIONS` | int | device | `ContextOptions::max_invocations_override` | environment wins | Pretend the device caps workgroups at this many invocations; must be positive. |

### Diagnostics

| Variable | Type | Default | Configuration field | Precedence | Effect |
| --- | --- | --- | --- | --- | --- |
| `APRILTAG_VK_VALIDATION` | bool | off | `ContextOptions::enable_validation` | environment can only enable | Enable the Khronos validation layer if installed. |
| `APRILTAG_VK_TIMESTAMPS` | int | off |  | environment only | Non-zero emits the per-dispatch GPU timestamp breakdown (tracing also enables it). |
| `APRILTAG_VK_TRACE` | string | off |  | environment only | Write a Chrome-trace JSON of the CPU and GPU spans to this file. |
| `APRILTAG_VK_DISPATCH_TIMES` | bool | off |  | environment only | When set to any value, print the median GPU time of every dispatch at exit (read once per process). |
| `APRILTAG_VK_PROBE_STATS` | bool | off |  | environment only | Histogram each unrefined quad's Hamming distance by outcome. |
| `APRILTAG_VK_PROBE_CSV` | string | off |  | environment only | Append one CSV row per quad to this file (also turns on the probe statistics). |
| `APRILTAG_VK_POLISH_DEV` | bit mask | 0 |  | environment only | Development bit mask of the polish fit (read once per process). |

### Platform variables read

| Variable | Type | Default | Configuration field | Precedence | Effect |
| --- | --- | --- | --- | --- | --- |
| `LOCALAPPDATA` | string | unset |  | fallback | Windows: the pipeline cache is kept under %LOCALAPPDATA%\vkapriltag. |
| `XDG_CACHE_HOME` | string | unset |  | fallback | Elsewhere: the pipeline cache is kept under $XDG_CACHE_HOME/vkapriltag. |
| `HOME` | string | unset |  | fallback | Elsewhere, without XDG_CACHE_HOME: the cache is kept under $HOME/.cache/vkapriltag. |
<!-- options:end -->

## 3. Measured performance

**How current these numbers are.** The last whole-corpus timing on both machines is the v1.10.0 validation
(archived section 8az, below). Later changes were timed on the desktop only, one change at a time, and are
listed separately; nothing newer than 1.10.0 has been timed on the Orange Pi as a whole. The 1.7-era table
and the "560 us fixed + 3.3 us per 1k pixels" model that used to stand here are superseded and live in
`archive/PERFORMANCE-history.md` (old section 3).

Machines: the **Orange Pi 5 Plus** (Mali-G610, RK3588 with 4x A76 + 4x A55, Linux, GCC, `Release`, governors on
`performance`, section 4) and the **desktop** (RX 9060 XT, Ryzen 5 5600X, Windows 11, MSVC, `Release`). All
numbers are medians unless said otherwise; the desktop GPU alternates between two clock states from one
process to the next, so its GPU rows move by a few percent between runs.

### 3.1 Whole-frame time, v1.10.0 (archived section 8az)

Per-frame total (upload, GPU, CPU tail), over the whole corpus (4,003 frames on the desktop, 1,116 on the
Orange Pi; `tools/napa_corpus`), median / 95th percentile in ms, best of two rounds per frame:

| | v1.9.0 | v1.10.0 compatibility mode | v1.10.0 defaults |
| --- | --- | --- | --- |
| Desktop, decimation 1 | 0.923 / 2.313 | 0.791 / 1.944 | 0.356 / 0.690 |
| Desktop, decimation 2 | 0.556 / 1.315 | 0.551 / 1.258 | 0.299 / 0.560 |
| Orange Pi, decimation 1 | 2.549 / 6.339 | 2.485 / 6.058 | 1.058 / 2.400 |
| Orange Pi, decimation 2 | 1.230 / 2.809 | 1.191 / 2.723 | 0.743 / 1.499 |

"Compatibility mode" is the 15-variable set that switches every 1.10 behaviour off (see CHANGELOG, "Upgrading").
At 1920x1080 the desktop median is 0.570 ms. Known regression against 1.9.0: candidate-rich frames (JPEG-like
noise) at decimation 2 are 8 to 14% slower on the desktop, because the accuracy defaults refine and retry more
candidates and box decimation exposes more candidates in noise.

### 3.2 Changes since 1.10.0 (desktop, interleaved ABBA, harness `cpu` = quad fit + decode)

| Change | Effect | Source |
| --- | --- | --- |
| Early reject of non-tag candidates | CPU -22% / -20% (clutter-only, decimation 2 / 1), -18% / -9% busy-background, -17% / -5% jpeg; +6% / +4% on dense-tags; total -4 to -7% on clutter | archive section 8be, `research/PROGRESS-early-reject.md` |
| Own decode (replaces patched libapriltag) | CPU -2% to -14% per frame (one-tag -2%, dense-tags -6% / -10%, jpeg -11% / -14% at decimation 2 / 1); the decode attempt alone 1.97x faster | `research/PROGRESS-own-decode.md` |
| Blur level pass | +0.07 to +0.08 ms on blurred frames, +1.5% / +0.6% mean over sharp scenarios | archive section 8bd |
| One shared worker pool | no measurable change (three pools become one) | `research/PROGRESS-one-pool.md` |
| Geometry header, candidate steps, options registry, deletions, own types | within run-to-run noise (about 2%) | the matching `research/PROGRESS-*.md` |

CPU time per frame on the desktop at decimation 2 with the own decode (`vk_eval --iterations 8 --warmup 5`,
`research/PROGRESS-own-decode.md`): one-tag 0.060 ms, many-tags 0.123, small-tags 0.093, clutter-only 0.163,
busy-background 0.191, dense-tags 0.266, jpeg 0.346.

### 3.3 Camera rate (frames with gaps), archive section 8bc

A GPU idle for a few milliseconds is slow to start the next frame. RX 9060 XT, 640x480 one tag, decimation 2, mean
ms per frame: back to back 0.214; 16 ms gaps 0.833 without and 0.415 with the pre-warm; 33 ms gaps 0.841 without and
0.363 with. On the Orange Pi (integrated GPU) the keep-alive is on by default and keeps 33 ms-gap frames within
0.03 to 0.06 ms of back to back. The CPU phase also doubles after a gap (0.042 to 0.085 ms) for a reason that was not
found.

### 3.4 Accuracy on the corpus (26 scenarios, `tools/napa_corpus`)

At the defaults the corpus finds 97.0% of the 7,348 tags at decimation 1 and 97.3% at decimation 2 (7,129 and 7,146),
with no false detections at decimation 1 and one at decimation 2 (the `tag16h5` family scene); median worst-corner
error 0.018 / 0.017 px. v1.9.0 found 94.4% / 92.5% with 3 and 7 wrong ids outside the tag16h5 set. Opt-in
accuracy features (`SMALL_RESCUE`, `HULL_QUAD`, `POLISH`) are in `DECISIONS.md` with their numbers.

The CPU reference NAPA 0.5.1, built with gcc -O3 and run single-threaded on the same corpus on the desktop
(`research/PROGRESS-napa-runner.md`): 97.3% recall, median 0.021 px, p95 0.100 px, frame median 0.094 ms (minimum tag
size 32 px) and 0.273 ms (16 px). NAPA is about 3.6 times faster than vkapriltag at decimation 2 on the desktop
(archive section 8at; vkapriltag's floor is about 0.3 ms against NAPA's 0.03 to 0.12 ms) and 2 to 6 times faster on the
Orange Pi (archive section 8ar); the GPU path pays off where per-pixel work dominates (large frames, decimation 1).
Corner accuracy and recall are on par.

### 3.5 Where the time goes (reading guide)

- Desktop: the GPU phase is about 0.25 to 0.3 ms and the CPU tail dominates on candidate-rich frames.
- Mali: 19 dependent dispatches before the lean tail (about 16 with it, counted from the code, not timed); a chain of 19 trivial dependent dispatches takes 464 to 504 us there and
  82 to 87 us on the RX 9060 XT (archive section 8at), so a flat frame has a floor of about 0.5 to 0.75 ms.
- Decimation is the largest lever: decimation 4 is about 2.1x faster than 2 on the Pi, at a cost in detection range.
  Width and height must each divide by the decimation.
- One dispatch plus a fence wait is 49 to 54 us on the Mali and 57 to 69 us on the desktop; a frame is one submission
  on both machines.
- Per-dispatch GPU timestamps include their own cost (about 26 us per span on the Mali): trust host-timed chains and
  whole-frame times, not the sum of spans.

## 4. Per-device guidance

**Device memory** is sized for the dense worst case by default (4 boundary points per interior decimated pixel):
447 MiB at 1920x1080 before the 8-byte line-fit record (about 15% less now). That is fine on a discrete card and
painful on a unified-memory part. Measured at 1920x1080, detection results identical in all cases:

| `APRILTAG_VK_MAX_POINTS` | Device memory | Boundary points found |
| --- | --- | --- |
| unset (dense worst case) | 447 MiB | 159144 |
| 400000 | ~99 MiB | 159144 (no clamping) |
| 200000 | 56 MiB | 159144 (no clamping) |
| 100000 | ~36 MiB | 100000 (clamped, tag still decoded) |

On Mali or any integrated GPU set `APRILTAG_VK_MAX_POINTS` to about twice the `boundary_points` your scenes report
(`DetectProfile::boundary_points`; the profile line prints it). On discrete cards the default is fine.

**Chosen automatically from the device** (`ARCHITECTURE.md` section 4 lists every fork):

- Workgroup sizes are specialisation constants taken from the device limits; sweeping `APRILTAG_VK_WG` on the Mali
  found the automatic choice (128) within noise of the best.
- Unified memory: the grey frame is written straight into device-local host-visible memory. Discrete GPUs with a
  mappable device-local window (resizable BAR) do the same (-38 us per frame); otherwise there is a staging copy.
- Host-cached readback memory lets the line-fit records and extents be read in place and the whole frame be one
  submission; without it the four-submit flow runs (`ARCHITECTURE.md` section 3).
- Subgroup aggregation survives at one site only, `reduce_extents_hash`, and only on discrete GPUs: on the Mali it
  took the 1.7-era `pipeline_total` from 11.5 to 18.1 ms although the results were bit-correct. The rule that
  separated the sites: aggregate values by key, not bare counters (a counter's contention is what a discrete GPU's
  atomic unit already handles; `uf_final` was 3.5 times slower aggregated, `blob_diff` 1.3 times).
- Integrated GPUs: 4-pixel merge, run-piece counting, compacted (sparse) labelling and the lean tail, a 6 ms
  keep-alive and 256-wide sort workgroups. Discrete GPUs: banded merge, 1024-wide sort workgroups, and
  keep-alive off (the pre-warm covers camera-rate frames).

**Orange Pi 5 Plus.**

- Set the CPU governors of the A76 clusters (policy 4 and 6) to `performance`, or at least a 1.4 GHz floor, in whatever
  starts the detector. The board boots with `ondemand`, which adds 3 to 4 ms to every frame at 30 and 60 fps (the
  cores idle between frames and ramp slowly); `schedutil` is worse (11 to 17 ms). It is a system setting, not
  something the library can do for an application. CPU deep idle, GPU interrupt affinity and the Mali power policy
  made no difference.
- The worker pool pins its threads to the four big cores (`cpu_big_cores`, default on; `APRILTAG_CPU_BIG_CORES=0`
  turns it off): the CPU phase of candidate-rich frames falls 11 to 17% against the unpinned eight threads, and four
  unpinned threads are 20 to 40% worse than eight, so the pinning matters as well as the count.
- Mali has no external GPU profiler; use `hwcnt_run` / `hwcnt_timeline` and `malioc` (section 5).

**Desktop.** With several streams on one RX 9060 XT, spreading them over both compute families
(`ContextOptions::expected_streams`, `Context::StreamSlot`) gives +82% at four streams over one queue; on the Mali
`StreamSlot` returns 0 because one shared queue is faster. Frames with gaps (camera rate) cost about twice as much
without the pre-warm; it is on by default.

**Determinism.** The pipeline is deterministic in everything except the GPU front end's corner placement: boundary
points are appended with an atomic counter, so their order depends on workgroup arrival, and points with equal sort keys
can swap. The decoded tag set and ids are stable; a marginal detection can differ by one dump line and corners by about
1e-3 px between two runs of the same binary (the golden test allows this: `ARCHITECTURE.md` section 11). Changing the
workgroup size can change the candidate quad count by 1 to 2. (An older note claimed exact reproducibility; it was
wrong.)

## 5. Measuring

### Tests and the evaluation harness

Build with the defaults (`cmake -S apriltags_vulkan -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`) and run
`ctest --test-dir build --output-on-failure`; it needs a Vulkan device only for `golden_detections`,
`options_lint` and `installed_package_consumer` (lavapipe works with `APRILTAG_VK_ALLOW_CPU=1`). `ARCHITECTURE.md`
section 10 lists every test.

`vk_eval` (`tools/vk_eval`) is the measuring tool. It runs a list of images through the detector and writes the
detections as a dump:

```
vk_eval --list list.txt --out dump.txt --decimation 2 --iterations 50 --warmup 5 [--threads N] [--family tag16h5]
        [--levels 4b:28:0,2b:14:40,1:8:20] [--small-rescue] [--ref] [--list-options]
```

It prints a `TIME` line per frame and a median. Timing from a harness with fewer than 20 iterations, or while another
process uses the GPU, is noise: interleave A and B in ABBA order and compare medians.

`tools/eval.sh -e <vk_eval> [-c <corpus>] [-- --decimation 1]` runs every scenario of a corpus (default: the committed
10-scenario subset in `tests/corpus`), scores the dumps with `tools/napa_corpus/score_corpus.py` and prints recall,
false detections, median and p95 corner error and the mean frame time against `tests/eval_baseline.json`
(`-u` stores a baseline, `-s` exits non-zero on a regression). The full 26-scenario corpus is generated by
`tools/napa_corpus/gen_corpus.py` (about 3.8 GB, not in the repository) and takes about 40 s per decimation. Sequences
for tracking come from `tools/napa_corpus/gen_sequences.py` and are run with `tools/track_eval` and
`tools/napa_corpus/run_sequences.sh`; `tools/napa_corpus/napa_eval` runs the CPU reference NAPA on the same corpus.

A change is a win only if no scenario gets worse in recall, false detections or p95 error at decimation 1 and 2 beyond
run-to-run noise (a couple of dump lines differ between identical runs: `ARCHITECTURE.md` section 11), and the time
cost is stated.

### Validation against upstream libapriltag (optional)

The default build needs no upstream. With `-DVKAPRILTAG_UPSTREAM_VALIDATION=ON` stock libapriltag v3.4.5 is fetched
and these tools are built: `apriltag_vulkan_validate`, `apriltag_pose_e2e_validate`, `apriltag_pose_validate`,
`apriltag_synth_ground_truth` and `vk_eval --ref`.

```
apriltag_vulkan_validate --data <file-or-dir> [--iterations N] [--pipelined]
                         [--decimation N] [--family tag36h11] [--csv out.csv]
```

`--data` takes a single image or a directory (PGM only in builds without OpenCV). It runs the full pipeline and diffs
the decoded tag IDs and corner positions against the unmodified reference detector. Use `--iterations` for
steady-state numbers: a single `Detect()` is dominated by first-touch page faults across every buffer. `--pipelined`
selects `FramePipeline`.

Three cautions, all learned the hard way here:

- **Verify the two builds use the same `CMAKE_BUILD_TYPE`.** This one cost an
  afternoon: a baseline worktree configured `Release` against the project's
  own `x64-Release` preset, which is `RelWithDebInfo`, reported `quad_decode`
  **+82%** and looked exactly like a serious regression in the change under
  test. MSVC's `RelWithDebInfo` is `/O2 /Ob1`, and `/Ob1` inlines only
  functions declared `inline` or defined in-class - so the DP corner-seeding
  helpers in `QuadDecode.cpp`, called once or twice per point across four
  O(n) passes, stayed real calls. They are marked `inline` now, which is
  worth ~35% of `quad_decode` in that configuration and nothing in `/Ob2`,
  but the lesson generalises: a cross-tree A/B compares toolchain settings
  unless you check. The same trap has a worse Linux form - a build directory
  configured with no `CMAKE_BUILD_TYPE` at all gets **no `-O` flag**, and
  that is exactly what the Orange Pi's working build tree turned out to be,
  so an early round of Mali figures compared a `-O3` baseline against an
  unoptimised branch. `grep CMAKE_BUILD_TYPE build/CMakeCache.txt` on both
  sides before believing anything; an empty value is the dangerous one,
  because nothing warns about it.
- **Verify the binaries actually differ** before believing an A/B —
  `git checkout` carries uncommitted changes onto the new branch, which once
  produced a confident null result from comparing two identical builds.
  `md5sum` them.
- **The Orange Pi's run-to-run variance is ~+/-5%**, enough to invert a
  single-run comparison. Interleave in ABBA order rather than AB (it drifts
  thermally across a session), take many samples, and report min and p25 —
  the mean is dominated by slow outliers.

On the Orange Pi, `perf` works in user mode (`perf record -g`, `perf stat -e cycles:u,instructions:u`
with `perf_event_paranoid` 2), so build the profiling tree with `-g -fno-omit-frame-pointer` and pin the
run to the A76 cores with `taskset -c 4-7`. `valgrind --tool=callgrind` and `heaptrack` are installed,
and GCC 14 carries AddressSanitizer, UndefinedBehaviorSanitizer (add `float-cast-overflow`),
ThreadSanitizer and LeakSanitizer: configure with `CMAKE_BUILD_TYPE=None` and the flags in
`CMAKE_CXX_FLAGS`, `CMAKE_C_FLAGS` and both linker flag variables. Run LeakSanitizer with
`detect_leaks=0` unless the loader's own allocations are being filtered, since the Vulkan loader reports
tens of megabytes at exit.

On Mali, `hwcnt_run <command>` (`tools/mali_counters`, built with
`-DVKAPRILTAG_BUILD_MALI_COUNTERS=ON`) prints every hardware counter over a run, and `hwcnt_timeline
200 out.csv <command>` samples the main ones every 200 us; run the detector paced (`bench_multiqueue
--fps 20`) and align the frames to attribute counters to stages (section 8q). For static shader
analysis, `malioc -c Mali-G610 --vulkan --compute <shader>.spv` from Arm Performance Studio (Linux
x86-64 or Windows host).

For per-dispatch GPU timing, `APRILTAG_VK_TIMESTAMPS=1` adds a timestamp-span
breakdown plus the gaps between spans, separating intra-submit gaps from
submit boundaries, and `APRILTAG_VK_DISPATCH_TIMES=1` (with `APRILTAG_VK_PRERECORD=0`) times every
dispatch on its own. Each timestamp waits for the work before it, which adds about 4 us per dispatch
on the Mali and overstates transfer fills; confirm a saving with an interleaved `Detect()` comparison.

On the RX 9060 XT, RenderDoc adds what the in-app timestamps cannot: every command timed on its own
(fills and copies as well as dispatches) and AMD's GPUPerfAPI counters per dispatch (cache hit rates,
memory and write-unit stalls, wavefronts, bytes fetched and written). A compute-only frame has no
present to mark it, so the validate tool captures one through RenderDoc's in-application API when
built with `renderdoc_app.h` found by CMake:

```
ENABLE_VULKAN_RENDERDOC_CAPTURE=1 APRILTAG_VK_PRERECORD=0 apriltag_vulkan_validate --data img.pgm \
    --iterations 25 --renderdoc-capture 20 --renderdoc-file out/frame
RDOC_CAP=out/frame_capture.rdc RDOC_OUT=frame.tsv qrenderdoc --python tools/renderdoc/rdoc_counters.py
```

Each command is timed in isolation on replay, so transfers that overlap in a real frame read high.
RenderDoc has no counters for the Mali-G610; there, use the Mali tools above.

For the validation layer's synchronisation checks to see compute shaders' buffer accesses, set
`khronos_validation.validate_sync = true` and `khronos_validation.syncval_shader_accesses_heuristic
= true` in a `vk_layer_settings.txt` named by `VK_LAYER_SETTINGS_PATH` (with
`APRILTAG_VK_VALIDATION=1`). Without the second setting only transfers and indirect reads are checked:
a deliberately removed barrier between two dispatches goes unreported. For an independent view on AMD, RADV can emit RGP traces
with `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1` (the per-submit flag is
what makes it work for a headless compute app with no `vkQueuePresentKHR`).
RenderDoc needs an artificial frame boundary for the same reason, and Mali
exposes no external GPU profiler at all — there, attribute cost by external
wall-clock deltas while varying a knob or the input size.

### Tracing and per-quad profiling

`--profile` prints a per-span table (count, total, mean, p50, p99, max; spans nest, so totals
overlap) plus the `TagDecoder` refine breakdown, and `--trace out.json` writes a Chrome Trace Event
file for https://ui.perfetto.dev. Both work in either validate tool and with `--pipelined`, where
`pipeline.gpu_pass`, `pipeline.cpu_tail` and `pipeline.wait_gpu` show which side limits throughput.

- Each pool worker, the GPU-driver thread and a "GPU" track get their own rows. The GPU track is the
  existing `APRILTAG_VK_TIMESTAMPS` spans placed on the host timeline by pinning the first span to the
  first submission's host time, so expect a constant offset of tens of microseconds, not drift.
- Library users call `prof::Enable(true)` (or set `APRILTAG_VK_TRACE=<path>` and call
  `prof::EnableFromEnv()`) **before constructing `GpuDetector`**, which creates the timestamp pool at
  construction, then `prof::Summary()` / `prof::WriteTrace()` while no frame is in flight.
- Cost: a disabled scope is one relaxed atomic load. A before/after comparison on the RX 9060 XT showed
  no difference outside run-to-run noise (median 0.99-1.10 ms vs 0.96-1.04 ms). Enabled costs about
  +0.09 ms per frame (timestamp queries plus roughly 420 events). `VKAPRILTAG_ENABLE_PROFILING=OFF`
  removes the hooks at compile time; that configuration was not built here.
- Keep traces to a few hundred frames: each thread's buffer caps at 262,144 events and drops (and
  reports) the rest.
