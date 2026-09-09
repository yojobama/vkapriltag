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
   on the GPU; the counts of surviving points/blobs are read back only where
   the next dispatch's size depends on them.
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

This is a port of [frc971's CUDA `GpuDetector`](https://github.com/Team766/apriltags_cuda)
(itself built on AprilRobotics' `apriltag`) to Vulkan compute, so it runs
on hardware without CUDA — the actual target being ARM Mali. Every stage's
output has been checked against unmodified upstream `apriltag` throughout
development, not just at the end: see [Verifying correctness](#verifying-correctness).

## Scope reductions

This port intentionally does less than the CUDA original and upstream
`apriltag`:

- **`RefineEdges`** (camera-distortion-based edge refinement) is not ported.
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

- A Vulkan 1.1+ compute-capable GPU and driver.
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
apriltag_vulkan::TagDecoder tag_decoder(td);  // apriltag_detector_t*, families already added

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
# corpus/ is generated locally, not tracked - build it once from any input image:
build/tools/make_corpus --input apriltags_vulkan/grayimage.pgm \
                        --out-dir apriltags_vulkan/corpus

build/tools/apriltag_vulkan_validate --data apriltags_vulkan/corpus --family tag36h11
```

`--data` accepts either a single image or a directory. For every image, the
tool runs this pipeline and the unmodified upstream `apriltag_detector_detect()`
side by side and compares **both** the decoded tag ID set and the corner
positions (RMS pixel error) between the two — not a visual/manual check.
Pass `--iterations N` to additionally report steady-state timing (see
below); a single iteration is dominated by cold-start costs (page faults,
pipeline warm-up) and says little about per-frame throughput.

Pose is verified by a second tool, against upstream `apriltag_pose.c`:

```sh
build/tools/apriltag_pose_validate                        # synthetic + degenerate
build/tools/apriltag_pose_validate --data colorImage.pgm  # + real detections, + timing
```

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

Measured on the two development targets, `colorImage.pgm` (1920x1080),
`APRILTAG_VK_MAX_POINTS=200000`. **Every stage below is labelled with the
processor it actually runs on** — the pipeline is not GPU-only, and the two
tails and pose estimation are all CPU work:

- **Orange Pi 5** — ARM Mali-G610 MP4 GPU, Rockchip RK3588 CPU
  (4x Cortex-A76 + 4x Cortex-A55)
- **Desktop** — AMD Radeon RX 9060 XT GPU, AMD Ryzen 5 5600X CPU (6 cores)

Totals are best/median of 25 iterations; the per-stage GPU rows are single
figures from one instrumented frame (`APRILTAG_VK_TIMESTAMPS=1`).

| Stage | Runs on | Orange Pi 5 | Desktop |
| --- | --- | --- | --- |
| clear (per-frame buffer zeroing) | GPU | 0.08 ms | ~0.00 ms |
| threshold + decimate | GPU | 0.23 ms | 0.02 ms |
| labelling | GPU | 1.55 ms | 0.20 ms |
| label finalize | GPU | 0.49 ms | 0.05 ms |
| boundary extraction | GPU | 0.81 ms | 0.03 ms |
| hash grouping | GPU | 0.10 ms | 0.02 ms |
| extents | GPU | 0.92 ms | 0.05 ms |
| select + blob scan | GPU | 0.05 ms | 0.01 ms |
| scatter | GPU | 0.19 ms | 0.02 ms |
| sort + line-fit moments | GPU | 1.70 ms | 0.11 ms |
| readback copy | GPU | 0.19 ms | 0.02 ms |
| *(unspanned: submit boundaries + host fence/readback, see below)* | both | *1.64 ms* | *0.51 ms* |
| **GPU subtotal** (`GpuDetector`) | **GPU** | **8.11 / 8.37 ms** | **1.28 / 1.29 ms** |
| `quad_decode` (DP corner seeding, default) | **CPU** | 1.58 / 1.91 ms | 0.65 / 0.82 ms |
| `tag_decode` (bit sampling + hamming) | **CPU** | 0.33 / 0.38 ms | 0.08 / 0.10 ms |
| **Pipeline total** (GPU + both CPU tails) | GPU + CPU | **10.08 / 10.73 ms** | **2.04 / 2.20 ms** |

Pose is measured separately, since it is per detected tag rather than per
frame and is not part of the pipeline total above. **It runs entirely on
the CPU** — `library/include/vkapriltag/PoseEstimator.h` records why the
GPU would be slower for this particular computation:

| Stage | Runs on | Orange Pi 5 | Desktop |
| --- | --- | --- | --- |
| upstream `estimate_tag_pose`, per tag | CPU | 0.916 ms | 0.616 ms |
| **`PoseEstimator`, per tag** | **CPU** | **0.016 ms** | **0.013 ms** |
| speedup | | **58x** | **48x** |

Both CPU tails and pose all run on the same `WorkerPool`, sized by
`DetectorConfig::cpu_threads` / `APRILTAG_CPU_THREADS`, so on the Orange Pi
they spread across all 8 RK3588 cores by default (measured 5.4x scaling for
`quad_decode`).

Both runs: 5/5 corpus images match upstream `apriltag` on tag ID and
corner position; `colorImage.pgm` decodes tag `554` with corner RMS
0.1577 px on both.

`quad_decode` dropped from the previous 2.71/2.91 ms (RK3588) once `kDp`
became the default corner-seeding method: DP skips the combinatorial
search's O(C(10,4)) cost entirely rather than shaving a constant factor
off it, so the win scales with how much weaker the CPU is — ~47% on the
RK3588 against ~12% on the Ryzen. `tag_decode` similarly dropped from
0.60/0.61 ms (RK3588) once it was parallelized the same way `quad_decode`
already was; see `TagDecoder`'s class comment for why writing results by
index rather than completion order is what makes that safe.

The "unspanned" row is real time no instrumented stage accounts for, and
what it is has been measured rather than guessed — twice, because the first
answer was wrong. It is *not* per-submission latency in the naive sense:
removing an entire queue submission moved it by 0.05 ms, not the ~0.4 ms
dividing it by the submit count predicted. Nor is it inter-dispatch
barriers, which is what this file previously claimed. Timing each gap
between named spans on the GPU's own clock
(`DetectProfile::gpu_gap_ms`) shows the 8 gaps that sit inside a single
submission total ~0.04 ms — noise — while nearly everything lands at the 3
gaps that cross a submit boundary, and over half of even that is host-side
fence-wait and readback time no GPU timestamp can see. So reducing
dispatch or barrier count has almost nothing left to give here; only
removing a submit boundary's round trip would, and that was tried and
reverted for too small a gain. See `apriltags_vulkan/OPTIMIZATION_NOTES.md` for
the full history of how the pipeline got here — what was tried, what
regressed and why (including two Mali-specific results: shared-memory
tiling and subgroup-aggregated atomics both measured *slower* here despite
helping on the discrete card, because Valhall has no dedicated
shared-memory hardware), and the current list of remaining opportunities.

The tables above are the Orange Pi 5 and Desktop figures and are **not**
affected by the later GPU-agnostic pass, which was measured on different
hardware (an Intel Iris Plus G7 iGPU and an NVIDIA MX230) and has not been
run on either target above. Its own numbers, and the six changes it landed
against the four it measured and rejected, live in their own clearly-labelled
section at the end of `OPTIMIZATION_NOTES.md`. The one result there worth
knowing before touching these shaders: subgroup aggregation and a cheap
early-out are alternative answers to atomic contention, and applying both is
worse than either — so `uf_final` and `reduce_extents_hash` now carry a guard
in their scalar variants that is deliberately absent from their
subgroup-aggregated ones.

## Project layout

```
apriltags_vulkan/
  library/     the Vulkan compute pipeline + CPU tail (GpuDetector, QuadDecode,
               TagDecoder, and the optional PoseEstimator)
    shaders/   GLSL compute shaders, compiled to SPIR-V at build time
    include/   public headers (vkapriltag::vkapriltag target)
  apps/        interactive V4L2 sample app (Linux)
  tools/       libapriltag cross-validation tool, corpus generator
  cmake/       the upstream apriltag patch, shader-embedding machinery
  corpus/      multi-scale validation images, generated by tools/make_corpus
               (gitignored - regenerate rather than expecting a checkout to
               carry it)
  OPTIMIZATION_NOTES.md   detailed performance history and rejected approaches
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
attribution: this project is based on frc971's `apriltags_cuda` and
includes software derived from AprilRobotics' `apriltag` (fetched at build
time as an unmodified dependency, patched only to expose two internal entry
points — see `cmake/patches/`).
