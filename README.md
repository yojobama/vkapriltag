# vkapriltag

A Vulkan-compute reimplementation of the [AprilTag](https://github.com/AprilRobotics/apriltag)
detection pipeline, built for embedded and mobile GPUs. Developed and tuned
against an ARM Mali-G610 (Orange Pi 5, RK3588), and verified functionally
identical (against unmodified upstream `apriltag`, tag-for-tag and
corner-for-corner) on a discrete AMD GPU as well.

## What this is

Detection is split across three stages, composed by the caller:

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

This is a port of [frc971's CUDA `GpuDetector`](https://github.com/Team766/apriltags_cuda)
(itself built on AprilRobotics' `apriltag`) to Vulkan compute, so it runs
on hardware without CUDA — the actual target being ARM Mali. Every stage's
output has been checked against unmodified upstream `apriltag` throughout
development, not just at the end: see [Verifying correctness](#verifying-correctness).

## Scope reductions

This port intentionally does less than the CUDA original and upstream
`apriltag`:

- **`RefineEdges`** (camera-distortion-based edge refinement) is not ported.
- **Pose estimation** (`apriltag_pose.h`) is not wired up — it needs a
  calibrated camera matrix and tag size, which is out of scope here.
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

See `apps/apriltag_vulkan/main.cpp` for a complete example driving this from
a V4L2 camera, and `library/include/vkapriltag/gpu/GpuDetector.h` for the
rest of `DetectorConfig` (cluster-size floors, aspect/fill-ratio filters,
capacity caps, etc. — all have working defaults).

## Verifying correctness

```sh
build/tools/apriltag_vulkan_validate --data apriltags_vulkan/corpus --family tag36h11
```

`--data` accepts either a single image or a directory (the tracked
`corpus/` directory holds a small multi-scale set). For every image, the
tool runs this pipeline and the unmodified upstream `apriltag_detector_detect()`
side by side and compares **both** the decoded tag ID set and the corner
positions (RMS pixel error) between the two — not a visual/manual check.
Pass `--iterations N` to additionally report steady-state timing (see
below); a single iteration is dominated by cold-start costs (page faults,
pipeline warm-up) and says little about per-frame throughput.

## Performance

Measured on the two development targets, `colorImage.pgm` (1920x1080),
`APRILTAG_VK_MAX_POINTS=200000`, best/median of 25 iterations:

| Stage (GPU) | Mali-G610 (Orange Pi 5) | RX 9060 XT (discrete) |
| --- | --- | --- |
| clear (per-frame buffer zeroing) | 0.08 ms | ~0.00 ms |
| threshold + decimate | 0.23 ms | 0.02 ms |
| labelling | 1.55 ms | 0.20 ms |
| label finalize | 0.49 ms | 0.05 ms |
| boundary extraction | 0.81 ms | 0.03 ms |
| hash grouping | 0.10 ms | 0.02 ms |
| extents | 0.92 ms | 0.05 ms |
| select + blob scan | 0.05 ms | 0.01 ms |
| scatter | 0.19 ms | 0.02 ms |
| sort + line-fit moments | 1.70 ms | 0.11 ms |
| readback copy | 0.19 ms | 0.02 ms |
| *(unspanned GPU/driver overhead)* | *1.64 ms* | *0.51 ms* |
| **GPU total** | **8.11 / 8.37 ms** | **1.28 / 1.29 ms** |
| CPU `quad_decode` (DP corner seeding, default) | 1.58 / 1.91 ms | 0.65 / 0.82 ms |
| CPU `tag_decode` (bit sampling + hamming) | 0.33 / 0.38 ms | 0.08 / 0.10 ms |
| **Pipeline total** | **10.08 / 10.73 ms** | **2.04 / 2.20 ms** |

Both runs: 5/5 corpus images match upstream `apriltag` on tag ID and
corner position; `colorImage.pgm` decodes tag `554` with corner RMS
0.1577 px on both.

`quad_decode` dropped from the previous 2.71/2.91 ms (Mali) once `kDp`
became the default corner-seeding method: DP skips the combinatorial
search's O(C(10,4)) cost entirely rather than shaving a constant factor
off it, so the win scales with how much weaker the CPU is — ~47% on
Mali-G610 against ~12% on the discrete card. `tag_decode` similarly
dropped from 0.60/0.61 ms (Mali) once it was parallelized the same way
`quad_decode` already was (see "Project layout" below /
`TagDecoder`'s class comment) — both CPU-tail phases are now threaded via
the same `WorkerPool`, controlled together by
`DetectorConfig::cpu_threads` / `APRILTAG_CPU_THREADS`.

The "unspanned" row is real GPU-side time no instrumented stage accounts
for — mostly inter-dispatch pipeline barriers on Mali's tile-based
architecture, not per-submission latency (measured: removing an entire
queue submission moved it by 0.05 ms, not the ~0.4 ms a naive per-submit
estimate would predict). See `apriltags_vulkan/OPTIMIZATION_NOTES.md` for
the full history of how the pipeline got here — what was tried, what
regressed and why (including two Mali-specific results: shared-memory
tiling and subgroup-aggregated atomics both measured *slower* here despite
helping on the discrete card, because Valhall has no dedicated
shared-memory hardware), and the current list of remaining opportunities.

## Project layout

```
apriltags_vulkan/
  library/     the Vulkan compute pipeline + CPU tail (GpuDetector, QuadDecode, TagDecoder)
    shaders/   GLSL compute shaders, compiled to SPIR-V at build time
    include/   public headers (vkapriltag::vkapriltag target)
  apps/        interactive V4L2 sample app (Linux)
  tools/       libapriltag cross-validation tool, corpus generator
  cmake/       the upstream apriltag patch, shader-embedding machinery
  corpus/      tracked multi-scale validation images
  OPTIMIZATION_NOTES.md   detailed performance history and rejected approaches
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
attribution: this project is based on frc971's `apriltags_cuda` and
includes software derived from AprilRobotics' `apriltag` (fetched at build
time as an unmodified dependency, patched only to expose two internal entry
points — see `cmake/patches/`).
