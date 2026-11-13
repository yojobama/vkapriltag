# Changelog

All notable changes to vkapriltag. Per-version detail with measurements for 1.8.0 and 1.9.0 is in the
"What changed" sections of [README.md](README.md); the measurements behind every change are in
[apriltags_vulkan/PERFORMANCE.md](apriltags_vulkan/PERFORMANCE.md).

## Unreleased

### Changed (API)

- **The library no longer uses libapriltag's types.** `Detection` (id, hamming, decision margin, centre, corners,
  `double H[9]`, `const Family *`), `Family` (the built-in tag tables plus an explicit `max_hamming`; `Family::ByName`),
  `Image` (data, width, height, stride) and `Quad` replace `apriltag_detection_t`, `apriltag_family_t`, `image_u8_t`
  and `struct quad`. `TagDecoder` and `TrackingDetector` take a `TagDecoderConfig` (families, `refine_edges`,
  `decode_sharpening`) instead of an `apriltag_detector_t *`; `Decode()`, `FramePipeline::Push()/Flush()` and
  `TrackingDetector::Detect()` return `DetectionList` (`std::vector<Detection>`), valid until the next call;
  `PoseEstimator::EstimateAll` takes a `std::span<const Detection>`; `setup_tag_family` returns a `const Family *`.
  Output is unchanged. The patched-upstream decode (`APRILTAG_VK_OWN_DECODE=0`) is removed; upstream is only needed by
  the differential test and the reference validation tools (`VKAPRILTAG_UPSTREAM_VALIDATION`).

### Changed (build)

- **The project no longer depends on libapriltag.** A default configure fetches nothing (no network, no C
  compiler needed), applies no patches, and the library, the tests, the install tree and the `.deb` packages hold
  nothing from AprilRobotics' library; `find_package(vkapriltag)` needs only Vulkan and Threads. The decode is the
  in-house `OwnDecode` and the family tables are generated into `library/src/families/`; both derive from upstream
  and `NOTICE` says so and carries upstream's BSD-2-Clause licence.
- `VKAPRILTAG_UPSTREAM_VALIDATION` now defaults to `OFF`. With `ON`, stock upstream v3.4.5 is fetched (excluded from the
  `all` target and the install rules) for the reference validation tools (`apriltag_vulkan_validate`,
  `apriltag_pose_validate`, `apriltag_pose_e2e_validate`, `apriltag_synth_ground_truth`, `vk_eval --ref`), which are no
  longer installed. The parity workflow turns it on for the parity gate and runs the default `ctest` first.
- The differential decode test (`own_decode_*`, `families_match_upstream`) compares the library with upstream's results
  recorded in `tests/own_decode_vectors.bin` (4.8 MB: 3120 random and 2974 corpus quads, 2500 reconcile sets and the
  family table hashes), still bit for bit and with the `comparison_can_fail` self-test. `-DVKAPRILTAG_REGEN_VECTORS=ON`
  (with the validation option) patches upstream and builds `gen_decode_vectors` to rewrite the file.
- New tests: `installed_package_consumer` (installs to a temporary prefix, checks it holds nothing from upstream, builds
  and runs a `find_package(vkapriltag)` consumer on a corpus frame) and a stricter `deps_lint`.

### Removed (build)

- The FetchContent of upstream in the default build, `cmake/patches/` (the decode patches and `ApplyPatches.cmake` now
  live in `tools/upstream/patches/` and are used only to regenerate the test vectors; the decode no-alloc patch and the
  `-std=gnu17` / `-ffp-contract=off` pins for the apriltag target are gone from the default build), and the project's
  C language.

### Added

- **Pre-warm for frames that arrive at camera rate.** A GPU idle for a few milliseconds takes about 0.5 ms
  longer to start the next frame (640x480 one tag at 16 ms gaps: 0.83 ms per frame against 0.21 back to
  back). `Detect()` now learns the frame period and a helper thread submits two trivial dispatches 3 ms before
  the next frame is due, giving 0.36 to 0.42 ms; back-to-back frames and irregular cadences are unaffected.
  `DetectorConfig::prewarm_lead_us`, `prewarm_burst`, `prewarm_gap_us` (`APRILTAG_VK_PREWARM_LEAD_US=0` turns it
  off), and `GpuDetector::Prewarm()` for an application that knows a frame is about to arrive.
  `TrackingDetector` pre-warms before each keyframe (Orange Pi, keyframe 1.11 -> 0.87 ms at 16 ms gaps).

### Performance

- **Early reject of non-tag candidates with grey margin on.** The border pre-check and the dark reject were
  skipped whenever grey margin was on (the default), so every candidate was refined. They now run on the
  unrefined quad and reject it only if its inner W cells (what the grey margin retry decodes) fail them too;
  quads under 1.5 pixels per cell, coarse boxes and quads the small rescue would refit are kept. No tag is lost
  on the corpus (decimation 1 and 2, with and without the rescue and polish options) or the ten sequences; the
  CPU part of the frame falls 5 to 22% on clutter, busy background and jpeg frames and rises 0 to 6% on
  frames full of real tags. `tag16h5` false detections fall (3 to 1 at decimation 2).
  `APRILTAG_VK_EARLY_REJECT=0` restores the previous behaviour.

### Accuracy

- **Motion-blurred edges are placed by the halfway crossing.** When more than a quarter of the profile fit's
  first-pass profiles run into the window ends, each side is refitted from half a cell outside to half a cell
  inside, at the mid level between its white and black (exact for any symmetric blur). Worst-corner error on
  the fast `napa-seq` sequences (decimation 2, median / p95): fast-1tag-1280 0.360 / 1.463 to 0.059 / 0.114
  px (NAPA 0.047 / 0.117), fast-2tag-640 0.537 / 1.747 to 0.138 / 0.701; corpus `blur` p95 0.162 to 0.068,
  `motion-blur` 0.456 to 0.181; no other scenario changes. Costs about 0.07 ms on blurred frames, nothing
  measurable on sharp ones. `APRILTAG_VK_PROFILE_FEATURES=15` (bit 16 off) restores the previous output.

### Removed

- **The `fast`, `ultrafast` and `upstream` edge-refinement methods.** `APRILTAG_VK_REFINE` now accepts only
  `exact` and `profile`; any other value is ignored and the default (`profile`) runs. The `refine_edges`
  export in the upstream patch is gone with `upstream`.
- **A/B-only environment switches whose choice is settled.** `APRILTAG_VK_FIND_MODE`,
  `APRILTAG_VK_MERGE_FLAG_MODE`, `APRILTAG_VK_DEC1_ALIAS`, `APRILTAG_VK_UF_INIT_RUN`, `APRILTAG_VK_THRESH_RUN`,
  `APRILTAG_VK_SORT_CHUNK`, `APRILTAG_VK_REDUCE_X4` and `APRILTAG_VK_GRAY_MAPPED` are ignored; each keeps the
  behaviour it had by default on its device. The naive union-find walk, the sorting-network path of
  `sort_points_local`, and the one-point-per-thread extents shaders `reduce_extents_hash.comp` and
  `reduce_extents_hash_atomic64.comp` (with `reduce_extents_hash_body.glsl`) are deleted.

- **Tracking experiments that lost.** `TrackingConfig::roi_search`, `capture`, `capture_speed_px` and
  `max_speed_px` (`APRILTAG_VK_TRACKING_SEARCH`, `_CAPTURE`, `_CAPTURE_SPEED`, `_MAX_SPEED`) are gone, with
  `Path::kSearch` and `kCapture`, the matching `Stats` fields and the `keycap` and `asynccap` modes of
  `track_eval`. The keyframe and asynchronous modes are unchanged; the decimated start of background tracks
  (`coarse_pad_frac`, `coarse_converge_px`, formerly `capture_pad_frac`, `capture_converge_px`) stays.
- **Worker spinning, cold inline batches and caller pinning.** `cpu_spin_us`, `cpu_inline_max`,
  `cpu_inline_idle_us`, `cpu_pin_caller` and `APRILTAG_CPU_SPIN_US`, `_SPIN_YIELD`, `_SPIN_WORKERS`,
  `APRILTAG_CPU_INLINE`, `_INLINE_IDLE_US`, `APRILTAG_CPU_PIN_CALLER` are ignored; none beat the default on the
  Orange Pi or the desktop. `WorkerPool::Warm`/`WarmAll`, `KeepAlive::Options::on_warm` and the `warm_pool`
  argument of `NoteFrame` (which only served spinning) are removed. Big-core pinning
  (`cpu_big_cores`, `APRILTAG_CPU_BIG_CORES`) and `APRILTAG_CPU_THREADS` stay.
- **Dead code.** The unused `QuadCorners` type, and the dead `fuse` branches and duplicated timestamp
  post-processing in the unfused GPU flow (the flow itself stays, as the fallback for devices without
  host-cached memory).

### Changed

- **One options registry.** Every `APRILTAG_*` variable is a row of `library/src/common/Options.cpp` and is read
  once, where the detector, decoder or context is constructed; the switch reference in `PERFORMANCE.md` is
  generated from it (`vk_eval --list-options`) and a test fails if the library reads the environment elsewhere.
  Precedence is one rule, the environment overrides the configuration, with `cpu_threads` (a configured value
  wins) and the switches that can only enable or only disable listed in the table. No behaviour changes with
  the environment as it was.

### Fixed

- **Setting `DetectorConfig::small_tag_rescue` in code enables the whole rescue.** `TagDecoder` had no
  configuration input for it and discarded the boxes; it now takes the configured value through a new
  trailing constructor argument (`TrackingDetector` passes it; other callers construct `TagDecoder` as before).
- **The keep-alive works on Windows.** Its wait rounded up to the 15.6 ms timer tick; it now uses a
  high-resolution waitable timer, so `APRILTAG_VK_KEEPALIVE_MS=1` to `4` keeps a discrete GPU awake.

## 1.10.1 (2026-11-04)

Release packaging only; the library is unchanged from 1.10.0.

### Fixed

- **The `.deb` packages include the integer dot product decimation shader.** The 1.10.0 packages were
  built with Ubuntu 24.04's `glslang-tools` 15.1, which cannot compile `GL_EXT_integer_dot_product`, so
  they ran the plain decimation shader (identical detections, about 30 us slower per frame on the Mali
  with box decimation at decimation 2 or 4). The release workflow now builds glslang 16.2.0 on both
  runners (amd64 and arm64, Ubuntu 24.04 so the packages still install on Debian 13 and Ubuntu 24.04 and
  later), configures with `-DVKAPRILTAG_DOT_SHADER=ON` so a missing shader fails the build, and checks
  that the package contains `decimate_box_dot_u8.comp.spv`.

## 1.10.0 (2026-11-04)

Output changes at the defaults: detections, corner positions and the work per frame differ from 1.9.0.
Every new default has an environment switch (and, where one exists, a `DetectorConfig` field) that
turns it off; see "Upgrading" below.

### Accuracy

- **Edge refinement uses a profile fit by default.** `RefineEdgesMethod::kProfile` measures each side of a
  quad every ~1.5 px with a dense edge profile and fits a strength-weighted line, with outlier dropping,
  coarse-to-fine passes, a halfway-crossing locator for blurred edges and a clipped-exposure bias
  correction (`APRILTAG_VK_PROFILE_FEATURES`, all four on). Median worst-corner error on the 24-scenario
  corpus falls from 0.041 to 0.018 px (decimation 1) and from 0.043 to 0.017 px (decimation 2).
  `APRILTAG_VK_REFINE=exact` (or `fast`, `ultrafast`) restores the 1.9.0 refinement.
- **The decoder retries more and rejects more.** A quad that fails to decode is retried as its inner
  square when a grey quiet zone fooled the border (`APRILTAG_VK_GREY_MARGIN`), a side fitted one cell
  too far out is refitted (`SIDE_ONE_OUT`), a third corrected bit is accepted when the three wrong cells
  touch (`THIRD_BIT`, with a block-split lookup), correction limits apply per family (`FAMILY_LIMITS`),
  and quads whose border is not mostly dark (`BORDER_PRECHECK`) or whose payload is darker than any valid
  code (`DARK_REJECT`) are dropped before refinement. Each switch set to `0` turns one off.
- **Tags found on the corpus rise** from 94.4% to 97.0% (decimation 1) and from 92.5% to 97.3% (decimation
  2); wrong IDs in the tag16h5 set fall from 1,739 to 0 and from 779 to 3. Tags with a grey margin
  (98%+ from 88% / 48%), tags partly covered and low-contrast tags gain most.
- **Decimation 2 and above averages each block** instead of taking its top-left pixel
  (`DetectorConfig::box_decimate`, `APRILTAG_VK_BOX_DECIMATE=0` restores it): small tags at decimation 4
  are found 71.8% -> 84.2% of the time and blur costs less corner error. It adds wrong IDs in a tag16h5
  scene unless the family limits (now on) are kept.
- **A noise-adaptive contrast floor** (`DetectorConfig::adaptive_floor_scale = 2`,
  `APRILTAG_VK_ADAPTIVE_FLOOR=0` off) raises the 4x4-block contrast threshold to twice the frame's median
  block range, so sensor noise is not labelled. A tag whose edge contrast is under about twice the noise's
  block range is lost (1 to 3 of about 700 degraded tags in the harshest synthetic sets, none in the real
  images). It needs 8-bit storage; without it the fixed floor applies.
- **Fixes found by an edge-case audit** that 1.9.0 does not have: blobs above the shared-memory sort split
  are sorted at any size up to 65,536 points, line-fit moments are taken relative to the blob's first point
  in 64 bits, the fused tail bounds its point threads by the buffer capacity, odd frame sizes upload,
  oversized or zero-iteration configurations are rejected, dispatches stay within
  `maxComputeWorkGroupCount`, a throw in `Detect` leaks nothing, `FramePipeline` checks the frame size,
  `PoseEstimator` rejects a non-positive or non-finite tag size or focal length and an SVD of rank below
  2 gives no pose, `WorkerPool` serialises callers, carries exceptions to the caller and bounds its size,
  `LoadGrayPgm` checks the header against the file, and the V4L2 capture and dma-buf test no longer busy
  wait.

### Performance

Median per-frame total over the corpus (4,003 frames on the RX 9060 XT, 1,116 on the Orange Pi 5 Plus), 1.9.0
-> 1.10.0 at the defaults: desktop 0.923 -> 0.356 ms at decimation 1 and 0.556 -> 0.299 ms at decimation 2;
Orange Pi 2.549 -> 1.058 ms and 1.230 -> 0.743 ms. At 1,920 x 1,080 the desktop median is 2.014 -> 0.570 ms.

- **Labelling and boundary passes run over compacted active cells** on integrated GPUs
  (`DetectorConfig::sparse_labelling`, `APRILTAG_VK_SPARSE`): flat regions launch nothing.
- **A leaner tail on integrated GPUs** (`APRILTAG_VK_LEAN_TAIL`, 3 there, 0 elsewhere): the point and blob
  kernels are grid-stride loops dispatched over a fixed grid, `select_blobs` places each blob's points and
  sets the sort's group count, so three dispatches (two argument kernels and the scan) are gone.
- **A GPU keep-alive on integrated GPUs** (`DetectorConfig::keepalive_ms`, `APRILTAG_VK_KEEPALIVE_MS`): a
  one-thread dispatch every 6 ms while frames arrive keeps the GPU out of its low-power state between camera
  frames (Orange Pi at 33 ms gaps: 0.89 to 1.02 -> 0.66 to 0.69 ms); it stops 100 ms after the last frame.
- **Completion by polling a flag** the last command sets, instead of waiting on the fence
  (`APRILTAG_VK_FLAG_POLL`): on by default with `VK_AMD_device_coherent_memory` (the RX 9060 XT: 0.194 ->
  0.145 ms on a one-tag 640x480 frame); `=1` forces it elsewhere (Orange Pi: 68 to 88 us per frame),
  outside the specification there.
- **Integer dot product in box decimation** (`VK_KHR_shader_integer_dot_product`, enabled when present,
  `APRILTAG_VK_DOT=0` disables): the pass takes 93 -> 62 us on the Mali at decimation 2.
- **Faster decode:** quad refits run in the parallel loop, decode has no per-quad heap allocations, the
  profile fit's tap positions are computed in one vectorisable loop (28% faster), and the third-bit match
  looks codes up by bit block.
- **Known slower case:** candidate-rich frames (JPEG-compressed scenes) at decimation 2 are slower than in
  1.9.0 at the defaults, by 14% (`fast` + margin gate 4) and 8% (`ultrafast`) on the RX 9060 XT and within
  5% either way on the Orange Pi, from the profile fit and the extra candidates the contrast floor and box
  decimation expose. With the old refinement the frames are level with 1.9.0.

### Added

- `RefineEdgesMethod::kProfile`, `MultiLevelDetector` (strided levels with size bands; measured slower
  than decimation 2, kept as a class), `BorderSample`, `vk::CompletionFlag`,
  `Context::CreateSiblingLane`, the `has_integer_dot_product` and `has_device_coherent_memory` caps, and
  `DetectorConfig` fields `adaptive_floor_scale`, `sparse_labelling`, `box_decimate`, `keepalive_ms`,
  `max_tag_pixels`.
- Environment switches: `APRILTAG_VK_ADAPTIVE_FLOOR`, `BOX_DECIMATE`, `DOT`, `FLAG_POLL`, `GREY_MARGIN`,
  `KEEPALIVE_MS`, `LEAN_TAIL`, `PROFILE_FEATURES`, `SIDE_ONE_OUT`, `SPARSE`, `SPIN_WAIT`, `THIRD_BIT`,
  `FAMILY_LIMITS`, `BORDER_PRECHECK`, `DARK_REJECT`, `MIN_DIFF`.
- `tools/napa_corpus`: a labelled 24-scenario test corpus generator (`gen_corpus.py`), a scorer
  (`score_corpus.py`), a runner and a NAPA comparison runner (`napa_run.py`); the benchmark harness takes a
  frame gap (`--gap-ms` / `VK_EVAL_GAP_MS`) to time at a camera rate.

### Build

- The integer dot product decimation shader is compiled only when `glslangValidator` supports
  `GL_EXT_integer_dot_product` (glslang 15.2 or later; Ubuntu 24.04's `glslang-tools` 15.1 does not).
  `VKAPRILTAG_DOT_SHADER=AUTO` (default) probes, `ON` requires it, `OFF` leaves it out; without it the
  plain decimation shader runs and detections are identical. The `.deb` packages are built on Ubuntu
  24.04 (glibc 2.39) so they install on Debian 13 and Ubuntu 24.04 and later, and so they use the plain
  shader.

### Upgrading

To approximate 1.9.0's behaviour set `APRILTAG_VK_REFINE=exact APRILTAG_VK_PROFILE_FEATURES=0
APRILTAG_VK_THIRD_BIT=0 APRILTAG_VK_FAMILY_LIMITS=0 APRILTAG_VK_BORDER_PRECHECK=0 APRILTAG_VK_DARK_REJECT=0
APRILTAG_VK_GREY_MARGIN=0 APRILTAG_VK_SIDE_ONE_OUT=0 APRILTAG_VK_BOX_DECIMATE=0 APRILTAG_VK_ADAPTIVE_FLOOR=0
APRILTAG_VK_SPARSE=0 APRILTAG_VK_LEAN_TAIL=0 APRILTAG_VK_KEEPALIVE_MS=0 APRILTAG_VK_FLAG_POLL=0
APRILTAG_VK_DOT=0`. The output is not byte-identical to 1.9.0's even then, because the audit fixes above
change which tags are found (about 3% of dump lines on the corpus, all extra correct detections).
Library users set the matching `DetectorConfig` fields.

### Tried and not kept

Persistent-kernel tail, calling-thread CPU tail, shuffle and shared-memory variants of the labelling kernels,
fused threshold and init scan, fused dispatch arguments, multi-level detection as a default, and four device
extensions (subgroup size control, uniform control flow, scheduling controls, subgroup shuffle) gave no gain
or a loss on the Mali; the measurements are in PERFORMANCE.md sections 8ac to 8ay.

### Release checks

Both machines (RX 9060 XT, Windows, MSVC; Orange Pi 5 Plus, Linux, GCC): the library, apps and tools build
in Release; the validation tools against upstream libapriltag, the pose tools and the synthetic ground truth
pass (21 of 21 checks on each; there are no `ctest` tests); the Khronos validation layer with
synchronisation validation reports nothing in 75 runs on each; AddressSanitizer on the desktop (75 runs) and
AddressSanitizer with UBSan and ThreadSanitizer on the Orange Pi (75 runs each) report nothing. Details in
PERFORMANCE.md section 8bb.
