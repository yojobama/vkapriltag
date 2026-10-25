# vkapriltag

A Vulkan-compute reimplementation of the [AprilTag](https://github.com/AprilRobotics/apriltag)
detection pipeline, built for embedded and mobile GPUs. Developed and tuned
against an ARM Mali-G610 (Orange Pi 5, RK3588), and verified functionally
identical (against unmodified upstream `apriltag`, tag-for-tag and
corner-for-corner) on a discrete AMD GPU as well.

## What this is

Detection is split across three stages, composed by the caller, plus an
optional fourth for pose:

1. **`GpuDetector`** — the Vulkan compute pipeline. Decimates the input
   frame, thresholds it, runs connected-component labelling, extracts and
   compacts boundary points, groups them into blobs, selects plausible tag
   candidates, and computes per-point line-fit moments. Everything here runs
   on the GPU. On parts where the readback memory is host-cached the whole
   frame is one pre-recorded command buffer and one queue submission, with a
   retry path (recorded the same way) for the rare frame whose labelling has
   not converged; otherwise the counts that size the next dispatch are read
   back mid-frame. `vk::Context` can also create queues on both compute
   families of a device and hand each of several streams a slot
   (`ContextOptions::expected_streams`, `Context::StreamSlot`).
2. **`QuadDecode`** — a CPU tail that turns `GpuDetector`'s per-blob boundary
   points into fitted quad corners. Two corner-seeding methods are available
   (`DetectorConfig::quad_fit_method`): `kDp` (default) seeds corners
   geometrically — mutually-farthest boundary points, then maximum
   perpendicular deviation on each arc — falling back per-blob to `kPeaks`,
   the original windowed line-fit error + peak detection + small
   combinatorial search, whenever DP doesn't cleanly yield 4 valid corners
   (measured ~8% of blobs). Deliberately CPU: this only ever touches a few
   thousand candidate values per frame, and is far simpler to get right as
   scalar C++ than as compute shaders.
3. **`TagDecoder`** — wraps the unmodified, upstream `apriltag` C library's
   own per-family bit-sampling and hamming decode, fed the quads from step 2.
   Optionally (`apriltag_detector_t::refine_edges`, off by default) also runs
   gradient-based corner refinement on each quad first, in the same per-quad
   task and the same order `apriltag_detector_detect()` uses internally.
   Which implementation runs is `APRILTAG_VK_REFINE`'s choice: `exact` (the
   default) is bit-identical to upstream *by construction* but drops the libm
   `modf()` call from the inner loop, worth 25% of this stage on ARM; `fast`
   is a single-precision variant (not bit-identical); `ultrafast` also skips
   refinement for quads that do not decode unrefined (faster, slightly lower
   recall on marginal tags); `upstream` calls
   upstream's actual compiled function, so any suspected corner-accuracy
   regression can be isolated with one environment variable. An opt-in
   margin gate (`probe_margin`, see [Performance](#performance)) skips the
   refinement of quads whose unrefined decision margin shows they cannot be
   tags.

Optionally, a fourth stage:

4. **`PoseEstimator`** — 6-DoF tag pose from the four decoded corners, given
   camera intrinsics and the physical tag size. A rewrite of upstream's
   `apriltag_pose.c` (homography seed, orthogonal iteration, planar-pose
   ambiguity search) in allocation-free fixed-size arithmetic: same
   algorithm, but ~60x faster, because upstream's version is built on
   `matd_op()`, a runtime string-expression interpreter that heap-allocates
   its way through ~1600 parsed expressions per pose.
   Verified against upstream at every stage of the algorithm — see
   [Verifying correctness](#verifying-correctness).

For continuous capture there is also **`FramePipeline`**, which composes the
three detection stages and overlaps frame N's GPU pass with frame N-1's CPU
tail, making a frame cost `max(GPU, CPU)` instead of the sum. It is a
throughput optimization, not a latency one — detections arrive one `Push()`
later, and it needs two frame buffers — so the stages stay independently
usable for callers that must act on the newest frame immediately. See
[Performance](#performance).

This is a port of [frc971's CUDA `GpuDetector`](https://github.com/Team766/apriltags_cuda)
(itself built on AprilRobotics' `apriltag`) to Vulkan compute, so it runs
on hardware without CUDA — the actual target being ARM Mali. Every stage's
output has been checked against unmodified upstream `apriltag` throughout
development, not just at the end: see [Verifying correctness](#verifying-correctness).

## Scope reductions

This port intentionally does less than the CUDA original and upstream
`apriltag`:

- **No lens-distortion handling.** Intrinsics are assumed pinhole, so a
  caller with a wide-angle lens should undistort the detection's corners
  before estimating pose. The CUDA original's `UnDistort` path is not
  ported either.
- **Pose estimation needs calibration data.** `PoseEstimator` is available
  but is not part of `Detect`: it requires camera intrinsics and the
  physical tag size, which only the caller has. Note that translation
  scales linearly with the tag size, so a 1% error there is a 1% range
  error — larger than every other error source in the solver combined.
- **Decimation must be set at detector creation.** `DetectorConfig::decimation`
  accepts any factor that divides the frame evenly (1, 2, 4, ...; default 2),
  but it is a specialization constant baked into the pipelines, so it is fixed
  for a detector's lifetime — changing it means constructing a new detector,
  unlike the CUDA original's per-call `quad_decimate`.
- **Single border polarity per run.** `reversed_border` must match across
  every tag family added to the detector in one run — mixing normal- and
  reversed-border families in a single pass is not supported.

## Requirements

- A Vulkan 1.1+ compute-capable GPU and driver. No optional feature is
  required: 8-bit storage, 64-bit atomics, subgroup operations, conditional
  rendering and host-pointer import are used when the device has them, and
  every one has a fallback path that gives identical detections (each can be
  forced off with an `APRILTAG_VK_FORCE_NO_*` variable, listed in
  `apriltags_vulkan/PERFORMANCE.md`). It has been run on an AMD RX 9060 XT and
  an ARM Mali-G610, and earlier revisions on an Intel iGPU and an NVIDIA
  MX230; other vendors are untested at this revision.
- CMake 3.16+, a C++20 compiler.
- [OpenCV](https://opencv.org/) — optional. `tools/` builds a richer
  validation tool against arbitrary image formats when OpenCV is found, and
  falls back to a PGM-only tool without it.
- V4L2 (Linux only) — only needed for the interactive sample app.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Relevant options (all in `apriltags_vulkan/CMakeLists.txt` /
`library/CMakeLists.txt`):

| Option | Default | Effect |
| --- | --- | --- |
| `VKAPRILTAG_BUILD_APPS` | `ON` | Build the V4L2 camera sample app (Linux only, regardless of this flag). |
| `VKAPRILTAG_BUILD_TOOLS` | `ON` | Build the libapriltag cross-validation tool. |
| `VKAPRILTAG_EMBED_SHADERS` | `ON` | Compile the SPIR-V shader corpus into the library binary, so a deployed build needs nothing on disk beside it (important for e.g. a JNI `.so` extracted from a jar at runtime). Off falls back to loading `.spv` files from an install-relative `SHADER_DIR`. |

## Quick start

```cpp
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/vk/Context.h"

apriltag_vulkan::vk::Context ctx;

apriltag_vulkan::DetectorConfig config;
config.width = width;
config.height = height;
config.tag_width = tf->width_at_border;       // from an apriltag_family_t*
config.reversed_border = tf->reversed_border;
config.normal_border = !tf->reversed_border;

apriltag_vulkan::GpuDetector detector(ctx, config);
apriltag_vulkan::QuadDecode quad_decode(config);
// td: apriltag_detector_t*, families already added. The decimation argument
// must match config.decimation - see TagDecoder's constructor comment - it's
// needed only if td->refine_edges is set (off by default; flip it on to run
// upstream's own gradient-based corner refinement before decode).
apriltag_vulkan::TagDecoder tag_decoder(td, config.decimation);

// Per frame:
detector.Detect(gray_frame);  // tightly packed 8-bit grayscale, width*height bytes
std::vector<apriltag_vulkan::DetectedQuad> quads =
    quad_decode.Decode(detector.last_selected_extents, detector.last_line_fit_points);
zarray_t *detections =
    tag_decoder.Decode(quads, gray_frame, width, height, config.reversed_border);
```

For pose, additionally (once, at setup):

```cpp
#include "vkapriltag/PoseEstimator.h"

apriltag_vulkan::PoseEstimator pose_estimator(
    {.fx = fx, .fy = fy, .cx = cx, .cy = cy},  // pixels, from calibration
    0.1651);                                    // tag border width, metres

// Per frame, after tag_decoder.Decode:
std::vector<apriltag_vulkan::TagPose> poses;
pose_estimator.EstimateAll(detections, poses);  // threaded across detections
```

See `apps/apriltag_vulkan/main.cpp` for a complete example driving this from
a V4L2 camera, and `library/include/vkapriltag/gpu/GpuDetector.h` for the
rest of `DetectorConfig` (cluster-size floors, aspect/fill-ratio filters,
capacity caps, etc. — all have working defaults).

## Verifying correctness

```sh
build/tools/apriltag_vulkan_validate --data path/to/your/images --family tag36h11
```

`--data` accepts either a single image or a directory, in whatever format
OpenCV can load (falls back to PGM-only without OpenCV — see Requirements).
No test images ship with the repo: supply your own, or generate a
resolution-swept set from one source image with `tools/make_corpus`
(`corpus/` itself is gitignored, since it's local/machine-specific — see
`apriltags_vulkan/.gitignore`). For every image, the tool runs this pipeline
and the unmodified upstream `apriltag_detector_detect()` side by side and
compares **both** the decoded tag ID set and the corner positions (RMS pixel
error) between the two — not a visual/manual check. Pass `--iterations N` to
additionally report steady-state timing (see below); a single iteration is
dominated by cold-start costs (page faults, pipeline warm-up) and says
little about per-frame throughput.

`--dump-quads <file>` writes the canonical candidate quads, and `--dump-detections <file>` the
decoded tags after refinement at full precision, so two builds or two configurations can be
compared byte for byte; `--stress <noise|snake|checker|rings>`
runs deliberately hostile frames and `--max-uf <n>` caps the labelling
iterations to force the retry path. `--profile` and `--trace <file>` print and
record the per-stage timing. The same dumps are identical across every
fallback path and workgroup configuration on both development machines.

Two further tools go beyond per-detection corner/ID agreement to the actual
6-DoF pose:

- **`apriltag_pose_e2e_validate --data <dir-or-file>`** runs stock
  libapriltag (detect + `estimate_tag_pose`) and this port (full pipeline +
  `PoseEstimator`) independently end to end on the same image — real
  detection differences (corner localization) flow into the comparison, not
  just the pose solver in isolation. `--ref-refine-edges` / `--our-refine-edges`
  toggle `refine_edges` on either side, so the cost of leaving it off
  asymmetrically (or the effect of turning it on for both) can be measured
  rather than assumed.
- **`apriltag_synth_ground_truth`** (`--refine-edges`, `--contrast <f>`) renders a tag at a *chosen, known* 6-DoF
  pose — built from the tag family's own bit tables, so the render is
  exactly what the decoder itself checks, not a guessed printed-tag
  convention — and compares both pipelines against that known pose, not just
  against each other. This is what can actually say *which* side (if either)
  is closer to correct when the two disagree, which a real photograph (no
  known true pose) cannot.

Pose is verified by a second tool, against upstream `apriltag_pose.c`:

```sh
build/tools/apriltag_pose_validate                        # synthetic + degenerate
build/tools/apriltag_pose_validate --data colorImage.pgm  # + real detections, + timing
```

(`colorImage.pgm` here and below names the specific real single-tag capture
these numbers were measured against on the two development machines — it
isn't shipped with the repo; substitute your own PGM.)

Upstream exposes every stage of its pose algorithm publicly, so this
compares at four levels rather than only the final answer — homography
seed, orthogonal-iteration solution 1, the ambiguity-search solution 2,
and the error-ranked pick — over a 420-case synthetic sweep (distance,
tilt, in-plane rotation, off-axis translation) that also carries **known
ground truth**, plus 8 deliberately degenerate geometries, plus real
detections when `--data` is given. It exits non-zero on failure, so it
works as a regression gate.

Measured agreement with upstream: rotation within **5.2e-7 degrees**,
translation within **3.8e-10** relative after refinement, and zero
disagreements over which ambiguity branch wins. On the real
`colorImage.pgm` detection it is tighter still — 1.9e-13 deg and 2.7e-16.
Absolute accuracy against the synthetic ground truth is *identical* to
upstream's (≤ 0.0032 deg), i.e. the rewrite tracks upstream some four
orders of magnitude more closely than either tracks reality. The one
deliberate divergence is precision in the seed's scale factor, where
upstream uses single-precision `sqrtf`; that shows up as ~1.4e-7 at the
seed and is washed out by iteration.

Iteration is also stopped once the pose stops moving, which upstream never
does — it computes a per-step error and then never compares it.
`DetectorConfig`-style tuning lives on the estimator itself: the default
`convergence_tol` of 1e-8 is the loosest value measured to leave
ground-truth accuracy *identical* to running all 50 iterations, and it
takes **52–54% of the wall clock** off across the synthetic sweep. Setting
it to 0 restores exact upstream iteration counts, which is the
configuration the ladder above is verified against. The saving is
scene-dependent: near-fronto-parallel tags are ill-conditioned enough that
the iteration keeps twitching below the tolerance and still hits the
50-iteration cap, saving nothing — including on `colorImage.pgm` itself.

## Performance

Measured on the two development targets with the 1280x800 corpus image (`tools/make_corpus` output,
one tag in view, 75 candidate quads), decimation 2, medians over 300 to 600 frames. **Every stage
below is labelled with the processor it runs on**: the pipeline is not GPU-only, and the CPU tail
and pose estimation are CPU work.

- **Orange Pi 5** — ARM Mali-G610 MP4 GPU, Rockchip RK3588 CPU (4x Cortex-A76 + 4x Cortex-A55),
  `Release` build, CPU and GPU governors on `performance`.
- **Desktop** — AMD Radeon RX 9060 XT GPU, AMD Ryzen 5 5600X CPU (6 cores), `RelWithDebInfo` build.
  The card alternates between two clock states between processes, so its GPU rows are quoted as a
  range.

**Orange Pi: set the CPU governor.** The board boots with `ondemand` on all clusters, which adds
3 to 4 ms to every frame at camera rates (the cores are idle between frames and ramp up slowly) and
costs 10% of free-running throughput; `schedutil` is worse (11 to 17 ms). Set the A76 clusters
(`policy4`, `policy6`) to `performance`, or at least a 1.4 GHz floor, in whatever launches the
detector. Other settings (CPU idle states, GPU interrupt affinity, Mali power policy) made no
difference at 30 and 60 fps. `apriltags_vulkan/PERFORMANCE.md` section 8i has the comparison.

### Where a frame goes

| Stage | Runs on | Orange Pi 5 | Desktop |
| --- | --- | --- | --- |
| clear (per-frame buffer zeroing) | GPU | 0.048 ms | ~0 ms |
| threshold + decimate | GPU | 0.145 ms | 0.018 ms |
| labelling (includes the blob-size count) | GPU | 0.761 ms | 0.183 ms |
| label finalise (only when a frame retries) | GPU | 0.004 ms | ~0 ms |
| boundary extraction | GPU | 0.276 ms | 0.023 ms |
| hash grouping | GPU | 0.055 ms | 0.017 ms |
| extents | GPU | 0.174 ms | 0.029 ms |
| select + blob scan | GPU | 0.066 ms | 0.017 ms |
| scatter | GPU | 0.070 ms | 0.014 ms |
| sort + line-fit moments | GPU | 0.302 ms | 0.066 ms |
| readback copy | GPU | 0.003 ms | ~0 ms |
| **sum of GPU stages** | **GPU** | **1.90 ms** | **0.37 ms** |

The stage rows are one instrumented frame (`APRILTAG_VK_TIMESTAMPS=1`, which itself costs about 9%
on the Mali). The totals below are without instrumentation:

| | Runs on | Orange Pi 5 | Desktop |
| --- | --- | --- | --- |
| `GpuDetector::Detect()` | GPU | 2.12 ms | 0.38 to 0.55 ms |
| `quad_decode` (DP corner seeding) | **CPU** | 0.20 ms | 0.11 ms |
| `tag_decode` (edge refinement, bit sampling, Hamming) | **CPU** | 0.63 ms | 0.23 ms |
| **Pipeline total, serial** | GPU + CPU | **2.99 ms** | **0.80 to 0.91 ms** |
| **Pipeline total, `FramePipeline`** | GPU \|\| CPU | **2.21 ms** | **0.41 to 0.46 ms** |

`Detect()` also includes the unspanned time (submit, fence wait, host-side readback) that no GPU
timestamp covers. It is small: an empty `vkQueueSubmit` + `vkWaitForFences` round trip measures
19 us on the Mali and 42 us on the RX 9060 XT, and on parts with host-cached readback memory the whole frame
is recorded once and submitted once. Decimation is the largest single lever on the GPU side: `--decimation
4` is about 2x faster than 2 on the Pi at the cost of detection range.

### Throughput

`bench_multiqueue`, frames per second with every stream on the same device, `ultrafast` refinement
(GPU-limited) and `exact` refinement (CPU-tail-limited). "Gate 4" is the margin gate described below.

| | Orange Pi 5 | RX 9060 XT |
| --- | --- | --- |
| 1 stream, `ultrafast` | 439 fps | 3,310 fps |
| 4 streams, `ultrafast`, one queue | 525 fps | 5,271 fps |
| 4 streams, `ultrafast`, queues spread over both compute families | | 9,096 fps |
| 1 stream, `exact` | 432 fps | 2,331 fps |
| 4 streams, `exact` | 523 fps | 3,044 fps |
| 1 stream, `exact`, gate 4 | 439 fps | 3,186 fps |
| 4 streams, `exact`, gate 4 | 524 fps | 4,974 fps |

Spreading streams over queue families (`ContextOptions::expected_streams`, `Context::StreamSlot`)
helps only where the device has two compute families (the desktop: +82% at four streams over one
queue); on the Mali one queue shared by all streams is faster, and the helper then changes nothing.

### The margin gate

With `refine_edges` on, 97% of the CPU refinement time goes to quads that never decode.
`TagDecoder`'s `probe_margin` (a constructor argument, `SetProbeMargin()`, or
`APRILTAG_VK_PROBE_MARGIN`; grey levels, 0 = off, the default) samples each quad's unrefined bits
first and skips refinement when the decision margin is below the gate or the border polarity is
wrong. On the synthetic sweep the gated and ungated runs are byte-identical at gates 4 and 8, at
contrast 1.0, 0.25 and 0.12 and noise sigma 3 and 8; junk quads sit at margin 4 or below and real
tags well above it. It applies only with exactly one tag family, refinement on, and a refine mode
other than `ultrafast`. The risk is a real tag whose unrefined margin is under the gate, so it
stays opt-in. On the Orange Pi it takes `pipeline_total` from 2.99 to 2.59 ms. A Hamming-distance
probe was measured and rejected: it loses 9 to 20% of the tags that only decode after refinement.

### Pose and pipelining

Pose is measured separately, since it is per detected tag rather than per frame. **It runs
entirely on the CPU** — `library/include/vkapriltag/PoseEstimator.h` records why the GPU would be
slower for this computation:

| Stage | Runs on | Orange Pi 5 | Desktop |
| --- | --- | --- | --- |
| upstream `estimate_tag_pose`, per tag | CPU | 0.916 ms | 0.616 ms |
| **`PoseEstimator`, per tag** | **CPU** | **0.016 ms** | **0.013 ms** |
| speedup | | **58x** | **48x** |

`FramePipeline` overlaps the GPU row with the CPU rows instead of running them in sequence (the
last row of the totals table). It raises throughput, not latency: a frame's result arrives one
`Push()` later. The gain depends on resolution; at 640x400 on the Pi it measures neutral, because
the two sides are close in size and the overlap is cancelled by contention for the shared LPDDR
bus. Re-measure at your deployment resolution rather than assuming.

The CPU tails and pose all run on the same `WorkerPool`, sized by `DetectorConfig::cpu_threads` /
`APRILTAG_CPU_THREADS`, so on the Orange Pi they spread across all 8 RK3588 cores by default
(measured 5.4x scaling for `quad_decode`). Results are written by index rather than completion
order, which is what makes that safe (see `TagDecoder`'s class comment).

`apriltag_vulkan_validate` compares against upstream's actual default configuration. That
comparison is not guaranteed to match on every image: a quad this port's corner fitting proposes
but upstream's differently designed quad finder never generates has nothing on the reference side,
which shows up as an ID-set "mismatch" even when every tag both sides find agrees to sub-pixel
precision.

### Tracing and further reading

`apriltag_vulkan_validate --profile` prints the per-stage GPU spans, the CPU-tail split and
blob-size histograms; `--trace out.json` writes a Chrome trace of the CPU and GPU spans.
`apriltags_vulkan/PERFORMANCE.md` covers configuration, per-device guidance and how to measure
fairly; `apriltags_vulkan/OPTIMIZATION_NOTES.md` is the record of what each optimisation did and of
everything measured and rejected, including two Mali-specific results (shared-memory tiling and
subgroup-aggregated atomics are both slower on Valhall, which has no dedicated shared-memory
hardware, though both help on the discrete card).

## Project layout

```
apriltags_vulkan/
  library/     the Vulkan compute pipeline + CPU tail (GpuDetector, QuadDecode,
               TagDecoder, the optional PoseEstimator, and FramePipeline for
               overlapping the GPU pass with the previous frame's CPU tail)
    shaders/   GLSL compute shaders, compiled to SPIR-V at build time
    include/   public headers (vkapriltag::vkapriltag target)
  apps/        interactive V4L2 sample app (Linux)
  tools/       validation tools (ID/corner, end-to-end pose, synthetic
               ground truth, pose solver), the corpus generator, and the
               benchmarks (bench_multiqueue for several streams, bench_barrier
               for the cost of a barrier)
  cmake/       the upstream apriltag patch, shader-embedding machinery
  OPTIMIZATION_NOTES.md   detailed performance history and rejected approaches
  PERFORMANCE.md          how to configure, run and measure it per device
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
attribution: this project is based on frc971's `apriltags_cuda` and
includes software derived from AprilRobotics' `apriltag` (fetched at build
time as an unmodified dependency, patched only to expose three internal
entry points — see `cmake/patches/`).
