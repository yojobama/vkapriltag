# vkapriltag

A Vulkan-compute AprilTag detector for embedded and mobile GPUs. The per-pixel work (decimate, threshold,
connected-component labelling, boundary points, grouping, selection, sorting) runs on the GPU in one queue submission;
the quad fit, edge refinement and tag decode run on the CPU. Developed and tuned on an ARM Mali-G610 (Orange Pi 5,
RK3588) and verified on a discrete AMD GPU (RX 9060 XT). It is a port of
[frc971's CUDA `GpuDetector`](https://github.com/Team766/apriltags_cuda), itself built on AprilRobotics'
[`apriltag`](https://github.com/AprilRobotics/apriltag), to Vulkan compute. The library does not link, fetch or install
upstream `apriltag`.

## Documents

| Document | What it is |
| --- | --- |
| [apriltags_vulkan/ARCHITECTURE.md](apriltags_vulkan/ARCHITECTURE.md) | The pipeline as built: GPU stages, submit structure, device-class forks, CPU tail, composites, tests, gotchas |
| [apriltags_vulkan/DECISIONS.md](apriltags_vulkan/DECISIONS.md) | Every idea tried, kept or dropped, with the key number and where it is recorded |
| [apriltags_vulkan/PERFORMANCE.md](apriltags_vulkan/PERFORMANCE.md) | Environment switch reference (generated), current numbers, per-device guidance, how to measure |
| [apriltags_vulkan/DESIGN.md](apriltags_vulkan/DESIGN.md) | Why it is built this way |
| [CHANGELOG.md](CHANGELOG.md) | Per-version changes |
| [apriltags_vulkan/archive/](apriltags_vulkan/archive/) | Frozen history: optimisation notes, performance history, old design notes, old README |

## Pipeline

1. **`GpuDetector`**: the Vulkan compute pipeline. Where readback memory is host-cached a frame is one pre-recorded command
   buffer and one submission; otherwise a four-submit fallback runs. Device-class forks (discrete versus integrated GPU,
   8-bit versus 32-bit storage) are chosen automatically.
2. **`QuadDecode`**: fits a quad to each blob's boundary points (DP corner seeding with a peaks fallback; optional hull and
   small-tag rescue).
3. **`TagDecoder`**: per-candidate steps (early reject, edge refinement, decode, recoveries, family limits), then reconcile. Edge
   refinement is the dense-profile fit by default (`APRILTAG_VK_REFINE=exact` selects the older gradient search). The decode is
   the in-house `OwnDecode`, with family tables generated into the library.
4. **`PoseEstimator`** (optional): 6-DoF pose from the four corners, about 50 times faster than upstream's solver.

Composites: `FramePipeline` (overlap the GPU pass with the previous frame's CPU tail), `TrackingDetector` (opt-in; skips the GPU on
most frames of a slow scene), `MultiLevelDetector` (several decimations; slower than decimation 2, kept as a class).

## Scope

- No lens-distortion handling: undistort corners before pose. Pose needs intrinsics and the physical tag size (translation scales
  linearly with the size).
- Decimation is fixed per detector (default 2); width and height must each divide by it. One border polarity per run. Frames must be
  tightly packed 8-bit grey.
- Defaults favour accuracy: dense-profile refinement, grey-margin and third-bit recoveries, per-family correction limits, early
  reject, block-averaging decimation and a noise-adaptive contrast floor (the last needs 8-bit storage). Opt-in accuracy features:
  `SMALL_RESCUE`, `HULL_QUAD`, `POLISH`. Each switch is listed in the generated table in `PERFORMANCE.md` section 2.

## Requirements

- A Vulkan 1.1+ compute GPU and driver. No optional feature is required; 8-bit storage, 64-bit atomics, subgroup operations,
  conditional rendering, integer dot product and host-pointer import are used when present and each has a fallback. Run on an AMD
  RX 9060 XT and an ARM Mali-G610; other vendors are untested at this revision.
- CMake 3.16+, a C++20 compiler, and `glslangValidator` (Vulkan SDK or `glslang-tools`). The integer dot product decimation shader
  needs glslang 15.2 or later; with an older one it is left out and the plain shader runs (identical detections).
  `-DVKAPRILTAG_DOT_SHADER=ON` makes that an error, `=OFF` leaves it out.
- Nothing is fetched at configure time and no C compiler or AprilTag library is needed. OpenCV is optional (richer validation tool);
  V4L2 is needed only for the sample app (Linux).

## Build and test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| Option | Default | Effect |
| --- | --- | --- |
| `VKAPRILTAG_BUILD_APPS` | `ON` | The V4L2 camera sample app (Linux only) |
| `VKAPRILTAG_BUILD_TOOLS` | `ON` | Evaluation, benchmark and validation tools |
| `VKAPRILTAG_BUILD_TESTS` | `ON` | Unit, golden-detection, decode-differential and lint tests (17 in all; `ARCHITECTURE.md` section 10) |
| `VKAPRILTAG_EMBED_SHADERS` | `ON` | Compile SPIR-V into the library so a deployed build needs no files beside it |
| `VKAPRILTAG_UPSTREAM_VALIDATION` | `OFF` | Fetch stock libapriltag v3.4.5 and build the reference validation tools (`apriltag_vulkan_validate`, the pose tools, `vk_eval --ref`) |
| `VKAPRILTAG_REGEN_VECTORS` | `OFF` | With the option above: rebuild the recorded decode test vectors |

The golden and consumer tests need a Vulkan device (`APRILTAG_VK_ALLOW_CPU=1` runs them on lavapipe). Score a corpus with
`apriltags_vulkan/tools/eval.sh` and measure with `vk_eval` (`PERFORMANCE.md` section 5). If no GPU driver is installed the loader
hands back a software device that is 50 to 500 times slower; the detector refuses it unless `APRILTAG_VK_ALLOW_CPU=1`.

## Quick start

```cpp
#include "vkapriltag/Detector.h"
using namespace apriltag_vulkan;

DetectorOptions options;
options.width = width;  options.height = height;      // tag36h11, decimation 2, refinement on by default
Detector detector(options);                            // opens the GPU

// Per frame (grey: width * height tightly packed bytes, or pass the row stride as a second argument):
const Result &result = detector.Detect(grey);          // valid until the next call
for (const Detection &tag : result.detections) {
  // tag.id, tag.hamming, tag.decision_margin, tag.c (centre), tag.p[4][2] (corners), tag.H (homography)
}
```

Everything else is an option: `options.mode` is `Mode::kSync`, `kPipelined` (the GPU works on this frame while the CPU decodes
the previous one) or `kTracking` (slow scenes skip the GPU on most frames); `options.levels` runs several decimations;
`options.small_tag_rescue`, `hull_quad` and `polish` trade time for recall and precision; `options.tag_size` with
`options.intrinsics` adds a pose per detection (`result.poses`); several cameras share one `Device`. Failures throw
`apriltag_vulkan::Error` with a code. `apps/apriltag_vulkan/main.cpp` is a complete V4L2 example.

Python, and through the same SWIG interface Java and C#, are in [apriltags_vulkan/bindings/swig](apriltags_vulkan/bindings/swig/README.md).

## Performance

Medians over the 26-scenario corpus at the defaults (v1.10.0; later changes are timed in `PERFORMANCE.md` section 3):

| | v1.9.0 | v1.10.0 |
| --- | --- | --- |
| Tags found, decimation 1 / 2 | 94.4% / 92.5% | 97.0% / 97.3% |
| Median corner error (worst corner), decimation 1 | 0.041 px | 0.018 px |
| Frame total, RX 9060 XT, decimation 1 / 2 | 0.923 / 0.556 ms | 0.356 / 0.299 ms |
| Frame total, Orange Pi 5 Plus, decimation 1 / 2 | 2.549 / 1.230 ms | 1.058 / 0.743 ms |

A CPU detector (NAPA) is about 3.6 times faster than this on small frames on the desktop; the GPU path pays off on large frames and at
decimation 1. On the Orange Pi set the A76 CPU governors to `performance` (`PERFORMANCE.md` section 4). Per-version detail for 1.8.0
and 1.9.0 is in [apriltags_vulkan/archive/README-1.10-history.md](apriltags_vulkan/archive/README-1.10-history.md) and the
[changelog](CHANGELOG.md); its tables predate the current defaults and name refinement methods that no longer exist.

## Verifying correctness

The default `ctest` needs no upstream: it compares the in-house decode bit for bit with upstream's results recorded in
`tests/own_decode_vectors.bin`, runs the golden detections and the unit tests. The tools built with
`-DVKAPRILTAG_UPSTREAM_VALIDATION=ON` compare against live upstream: `apriltag_vulkan_validate --data <images>` (tag ids and corner
RMS), `apriltag_pose_validate` (pose solver against upstream at four levels), `apriltag_pose_e2e_validate` and
`apriltag_synth_ground_truth` (end-to-end pose against a known pose). `vk_eval --list-options` prints the switch reference.

## Project layout

```
apriltags_vulkan/
  library/     GpuDetector, QuadDecode, TagDecoder, RefineEdges, OwnDecode, PolishFit, PoseEstimator, FramePipeline,
               TrackingDetector, MultiLevelDetector, the options registry; shaders/ (GLSL), include/ (public headers)
  apps/        interactive V4L2 sample app (Linux)
  tools/       vk_eval, eval.sh, track_eval, the corpus generators and scorers, validation tools, benchmarks, Mali counters
  tests/       unit, golden, differential and lint tests, the committed corpus subset
  cmake/       shader compilation and embedding
  ARCHITECTURE.md  DECISIONS.md  PERFORMANCE.md  DESIGN.md  archive/
research/      surveys and per-topic progress logs of the 2026-11 work
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution: this project is based on frc971's
`apriltags_cuda` and contains code derived from AprilRobotics' `apriltag` (BSD-2-Clause): the tag family code tables and the decode
(`library/src/OwnDecode.*`). Upstream is used only by the optional validation tools, and its licence text is reproduced in NOTICE.
