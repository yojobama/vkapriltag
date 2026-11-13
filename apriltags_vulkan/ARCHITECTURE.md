# Architecture

The pipeline as it is now (after the 2026-11 restructuring), verified against the code. This is the single
current-truth page: [DESIGN.md](DESIGN.md) says why it is built this way, [DECISIONS.md](DECISIONS.md) records what was
tried and what was dropped, [PERFORMANCE.md](PERFORMANCE.md) holds the numbers, the generated options reference and the
measuring notes. The documents in [archive/](archive/) are frozen and contain statements that this page overturns.

Every statement below is checkable in the code; files are named relative to `apriltags_vulkan/library/` unless a path
starts with `tests/`, `tools/` or `apps/`, and functions are named instead of line numbers.

## 1. Overview

```
gray frame (8-bit, width x height, tightly packed)
  |
  v   GpuDetector            GPU: decimate, threshold, label, boundary points, group, select, sort
  |     last_line_fit_points (a view of host-cached device memory: 8 bytes per point, in blob and angle order)
  v   QuadDecode             CPU: per-blob quad fit (DP seeding, peaks fallback, optional hull and small-tag rescue)
  |     std::vector<DetectedQuad>
  v   TagDecoder             CPU: per-candidate steps (reject, refine, decode, recover), reconcile, stable sort by id
  |     const DetectionList &   (std::vector<Detection>)
  v   PoseEstimator          CPU, optional: 6-DoF pose per detection
```

- The stages are independent classes the caller composes (section 9.3 shows the sequence). Three composites wrap them:
  `FramePipeline` (overlap the GPU pass with the previous frame's CPU tail), `TrackingDetector` (skip the GPU on most
  frames of a slow scene) and `MultiLevelDetector` (several decimations).
- The GPU front end does the per-pixel work and ends with a few thousand sorted boundary points; everything after that
  is scalar, branchy CPU code over a handful of candidates (see `DESIGN.md`).
- Nothing from AprilRobotics' library is used at build or run time: the decode is the in-house `OwnDecode`, the family
  tables are generated sources (section 6.5).
- One worker pool serves the quad fit, the decode and the pose estimator of a pipeline (section 8.1).

## 2. File map

| Area | Files |
| --- | --- |
| GPU front end | `gpu/GpuDetector.cpp`, `include/vkapriltag/gpu/GpuDetector.h` (`DetectorConfig`, `GpuDetector`, `DetectProfile`), `shaders/*.comp`, `shaders/*.glsl`, `include/vkapriltag/gpu/Types.h` (host mirrors of the GLSL structs) |
| Vulkan layer | `vk/Context.cpp` (device selection, `DeviceCaps`, lanes), `vk/Buffer.cpp` (`MemoryKind`), `vk/ComputePipeline.cpp` (pipelines, barriers, indirect dispatch), `vk/PipelineCache.cpp`, `vk/QueryPool.cpp`, `vk/CompletionFlag.cpp`, `vk/Shader.cpp`, embedded SPIR-V (`cmake/GenerateEmbeddedShaders.cmake`) |
| Pre-warm | `gpu/KeepAlive.cpp`, `shaders/keepalive.comp` |
| Quad fit | `gpu/QuadDecode.cpp`, `SmallQuad.cpp` (small-tag rescue refit) |
| Decode | `TagDecoder.cpp` (candidate steps), `RefineEdges.cpp`, `BorderSample.cpp`, `PolishFit.cpp`, `OwnDecode.cpp` (sampling, lookup, reconcile), `Family.cpp` and `src/families/*.cpp` (generated tables, `tools/gen_families.py`), `Geometry.h` (shared CPU geometry), `DecodeSwitches.h` |
| Composites | `FramePipeline.cpp`, `TrackingDetector.cpp`, `gpu/MultiLevelDetector.cpp`, `PoseEstimator.cpp` |
| Support | `common/WorkerPool.cpp`, `common/Options.cpp` (the options registry), `common/Profiler.cpp`, `common/pgm_io.cpp` |
| Public headers | `include/vkapriltag/` (`Detector.h`, `Detection.h`, `Family.h`, `Pose.h`, `Version.h`); the facade is `src/Detector.cpp`, the stages are in `src/internal/vkapriltag/` |
| Harness and tests | `tools/vk_eval`, `tools/eval.sh`, `tools/track_eval`, `tools/napa_corpus/`, `tests/` |

## 3. GPU frame flow

`GpuDetector::Detect(gray)` runs the stages below. The sizes: `decimated_width_ = width / decimation` (width and height
must each divide by the decimation and the decimated frame must be at least 3 pixels each way; the constructor
throws otherwise); a 4x4 block grid on the decimated frame; one pixel of the decimated frame is one labelling cell.

### 3.1 Stages in order

`GpuDetector::RecordFrameHead` records the first five, `RecordFusedTail` the rest. The shader names are the files in
`shaders/`; a `_u8` file is the 8-bit-storage variant of the same shader (section 4.2).

| # | Stage | Shader(s) | What it does |
| --- | --- | --- | --- |
| 0 | Clears | buffer fills (not dispatches) | Zero the counters, the convergence flag, the hash table, the cursors and (without count runs) the blob sizes; one compute-and-transfer barrier |
| 1 | Decimate | `decimate_box(_dot)_u8` (default, decimation above 1), `decimate_u8` (copy a block's top-left pixel, `box_decimate` off), none at decimation 1 with 8-bit storage (the grey buffer is read in place: `alias_decimated_`) | Reduce the frame by the decimation; box decimation averages each block (the dot variant sums a row with one integer dot product on devices that have it) |
| 2 | Block min and max | `block_minmax_u8` (or `block_minmax_decimate_u8`, which also decimates, when box decimation is off) | Min and max of each 4x4 block of the decimated frame |
| 3 | Noise floor | `noise_floor` (one workgroup) | Median block range times `adaptive_floor_scale` (2.0) gives the frame's contrast floor, at least `min_white_black_diff` (5) |
| 4 | Threshold | `threshold_block_u8` (fused block filter plus threshold), or `block_filter` and `threshold` without 8-bit storage | 3x3 min/max over blocks, then each pixel becomes 0, 255 or 127 (ambiguous); with sparse labelling it also lists the 64 x 4 pixel cells that hold contrast; `cell_args` then writes the indirect arguments |
| 5 | Labelling | `uf_init_scan` (each pixel points at its run start), `uf_merge` (one pass; `uf_merge_x4_u8` on integrated GPUs), then `uf_count_runs` (integrated) or `uf_compress_count` (discrete) | Union-find over the thresholded image; blob sizes are counted here, capped at `min_cluster_pixels` |
| 6 | Boundary points | `blob_diff(_u8)` | Up to 4 boundary points per interior pixel (E, SE, S, SW) between a black and a white pixel whose blobs are at least `min_cluster_pixels` large, appended to one compacted buffer with an atomic counter; each point carries the two region roots (rep0, rep1) |
| 7 | Group | `hash_group` | An open-addressing hash table on the (rep0, rep1) pair gives each distinct pair a dense id |
| 8 | Extents | `reduce_extents_hash_subgroup` (discrete with subgroup ballot, arithmetic and shuffle) or `reduce_extents_hash_atomic64_x4` or `reduce_extents_hash_x4` | Per pair: bounding box, point count, gradient moments |
| 9 | Select | `select_blobs` | Keeps pairs whose size, bounding-box aspect (`aspect_max` 8), fill ratio (0.5 to 2.5) and border polarity fit a tag; at most `max_blobs` survive |
| 10 | Offsets | `scan_blob_counts` (not run with the lean tail's bit 2: `select_blobs` writes the offsets) | Prefix sum of the selected blobs' point counts, and the sort's indirect arguments |
| 11 | Scatter | `scatter_index_points` | Each surviving point goes to the next free slot of its blob's contiguous range |
| 12 | Sort | `sort_points_local` (blobs up to `local_sort_split_` = 1024 points, whole, in shared memory; up to 256 points ranked in one pass), `sort_points_large` runs then merge (larger blobs, up to `kLargeSortCap` = 65,536 points) | Order each blob's points by angle around its centre and write the 8-byte `RawLineFitPoint` records (x2, y2 in doubled decimated coordinates, gradient weight W, blob index); a blob over the cap is written unsorted and counted in `DetectProfile::oversized_sort_blobs` |
| 13 | Host read | none | `HostReadBarrier`; the host reads `last_line_fit_points` and `last_selected_extents` from host-cached memory |

`uf_final` (a separate blob-size pass) exists but the default frame does not run it: blob sizes are counted inside stage 5
(`RecordFusedTail(speculative=true)` skips it; only the retry path's `TailRerun` runs it). `uf_compress` and `uf_init`
serve the verification path and the devices without subgroup arithmetic (section 4.1).

### 3.2 Dispatch counts

Counted from `RecordFrameHead` and `RecordFusedTail` for the default configuration (box decimation, adaptive floor, no
`UF_VERIFY`, decimation 2); fills and copies are not dispatches.

| Device class | Dispatches | Composition |
| --- | --- | --- |
| Discrete (RX 9060 XT) | 19 | threshold 4 (decimate, block min/max, noise floor, threshold), labelling 4 (`uf_init_scan`, `uf_merge` twice for the banded merge, `uf_compress_count`), argument builders 2 (`build_indirect_args` twice), then boundary, hash, extents, select, scan, scatter and three sorts (9) |
| Integrated (Mali-G610) | 16 | threshold 5 (adds `cell_args`), labelling 3 (`uf_init_scan`, one indirect `uf_merge`, `uf_count_runs`), then boundary, hash, extents, select, scatter and three sorts (8; the lean tail drops both argument builders and the scan) |

At decimation 1 with 8-bit storage the decimate dispatch disappears (18 and 15). The three sort dispatches always
exist; the large-blob ones are indirect and run zero groups when no blob exceeds 1024 points.

### 3.3 Submit structure

- **Fused flow (the default where both readbacks are host-visible and host-cached).** `fused_submits_` is true when
  `line_fit_points_buf_` and `selected_extents_buf_` are `HostVisibleCached` (`GpuDetector::CreateBuffers`;
  `APRILTAG_VK_FUSE_SUBMITS=0` forces the unfused flow). The whole frame is one command buffer and one queue
  submission: `DetectFused` calls `FirstAttempt`, which records `RecordFrameHead` plus `RecordFusedTail`. Counts that
  size later dispatches never leave the device (`build_indirect_args` writes indirect arguments; the lean tail uses
  fixed-grid kernels instead). The host then reads the counters and the two payloads in place (`FinishFused`).
- **Pre-recorded command buffers.** With `prerecord_` (fused, no transfer-queue upload; `APRILTAG_VK_PRERECORD=0` turns it
  off for per-dispatch timing and capture) the command buffer is recorded once per (kind, imported-frame key, chunk)
  and replayed (`FindRecorded`, `RecordPersistent`).
- **Completion.** `lane_.SubmitPersistentAndWait` waits on a fence, or on a flag the shader writes
  (`vk::CompletionFlag`, `RecordSet`) where the device has device-coherent host memory (AMD) or `APRILTAG_VK_FLAG_POLL=1`.
- **Verification and retry (off by default).** One merge pass joins every component, so the default frame does not
  verify the labelling and always completes in one submission. With `APRILTAG_VK_UF_VERIFY=1` the frame ends its labelling
  with a verification merge that sets `uf_changed_buf_`; if the flag is set, `DetectFused` submits continuation chunks
  (`Continuation`, up to `max_uf_iterations` = 64) and then `TailRerun`. The `conditional rendering` predicate on
  `uf_compress` (`predicated_compress_`) is used only in this verification path.
- **Unfused flow (`DetectUnfused`, the fallback for devices without host-cached readback memory).** Four submissions
  with a host readback between them, recorded every frame: (1) clears, threshold and the first labelling chunk, (2)
  extra labelling chunks if verifying, then blob sizes and boundary extraction (reads the boundary point count), (3)
  hash group, extents, select, scan and scatter (reads the blob and point counts that size the readback), (4) sort and the
  staging copies. It records the same stages with host-sized dispatches.
- **Transfer-queue upload.** With a second `vk::Lane` (a transfer-only queue), the staged frame is copied there and the
  first compute submission waits on a semaphore (`upload_semaphore_`); this path does not pre-record.
- **Extra submissions.** The pre-warm and keep-alive helper (section 8.3) submits trivial dispatches on a sibling lane
  of the same queue; they are not part of the frame.

### 3.4 Capacities and failure modes

`CreateBuffers` sizes everything from the configuration. Boundary points: `qbp_capacity_` is the dense worst case (4 per
interior decimated pixel) unless `max_boundary_points` (`APRILTAG_VK_MAX_POINTS`) caps it; blobs: `max_raw_blobs` 65,536
pairs (hash table size is the next power of two) and `max_blobs` (auto: 2048 at 1080p and decimation 2, scaled with the
decimated pixel count). When a limit is hit nothing throws: `DetectProfile` reports `hash_probe_drops`,
`selected_blob_drops` and `oversized_sort_blobs`. Which blobs are dropped when `selected_blob_drops` is non-zero depends on
GPU scheduling, so detections are then not reproducible.

## 4. Device-class forks

`ResolvePlan` and `FinalisePlan` (`gpu/PipelinePlan.cpp`) resolve every fork from `vk::DeviceCaps` (`vk/Context.cpp`), the
configuration and the options registry into one `PipelinePlan` (printed by `PipelinePlan::Describe`); the second step runs once
the readback buffers exist. `CreatePipelines` then builds each pipeline from its `StageDesc` in `gpu/StageTable.cpp` (shader
chosen by the plan, buffers by `Buf` role, push-constant size, workgroup size, specialisation constants). The
startup line printed by `vk::Context` reports the device, its limits and the launch geometry. Only two combinations
are measured as a class: discrete (RX 9060 XT, RADV) and integrated (Mali-G610, libmali); the other combinations are
reached through the `APRILTAG_VK_FORCE_NO_*` switches and are covered by the golden test (section 10).

### 4.1 The table

| Fork | Discrete default | Integrated default | Condition and override |
| --- | --- | --- | --- |
| Shader family | `_u8` | `_u8` | `storageBuffer8BitAccess`; `FORCE_NO_8BIT` selects the 32-bit family (section 4.2) |
| Fused threshold, adaptive floor | on | on | needs the `_u8` family; `FUSED_THRESHOLD=0`, `ADAPTIVE_FLOOR=0` |
| Box decimation with integer dot product | on | on | `box_decimate` and decimation 2 or 4, device has integer dot product, the build has the shader (`VKAPRILTAG_DOT_SHADER`); `DOT=0` |
| `uf_init_scan` | on | on | subgroup arithmetic, at most 64 subgroups per workgroup, not `UF_VERIFY`; `UF_INIT_SCAN=0` falls back to `uf_init` plus `uf_compress` |
| Count runs (`uf_count_runs`) | off | on | integrated and `uf_init_scan`; `COUNT_RUNS` |
| Sparse (compacted) labelling | off | on | `sparse_labelling` and fused threshold, `uf_init_scan`, count runs; `SPARSE` |
| Merge | banded: 8-row bands, then across them (2 dispatches) | 4 pixels per thread (`uf_merge_x4_u8`), one pass | `MERGE_BAND`; `MERGE_X4` (needs `_u8` and width divisible by 4); the convergence flag uses a read-guarded global `atomicOr` on unified memory (`merge_flag_mode`) |
| Extents reduction | subgroup variant | `atomic64_x4` if 64-bit atomics, else `x4` (32-bit) | subgroup ballot, arithmetic and shuffle, not integrated; `FORCE_NO_SUBGROUP`, `FORCE_NO_INT64_ATOMIC` |
| Lean tail | 0 | 3 | bit 1: fixed-grid kernels instead of `build_indirect_args` (needs fused flow and the x4 reduction); bit 2: `select_blobs` writes the offsets, no `scan_blob_counts`; `LEAN_TAIL` |
| Sort workgroup | up to 1024 | 256 | `SORT_WG`; split size 1024, rank-all 256 (`SORT_SPLIT`, `SORT_RANK_ALL`) |
| Fused submits, pre-recording | on | on | both readbacks host-visible and cached; `FUSE_SUBMITS`, `PRERECORD` |
| Completion flag | on (AMD) | off | device-coherent host memory; `FLAG_POLL` |
| Keep-alive | off | 6 ms | `keepalive_ms` negative selects 6 on integrated, 0 elsewhere (`KEEPALIVE_MS`) |
| Pre-warm | on | on | `prewarm_lead_us` negative selects 3000 (`PREWARM_LEAD_US=0` off) |
| Grey frame memory | `DeviceLocalMapped` if the device has a mappable window, else staged | `DeviceLocalMapped` (unified) | `gray_direct_write_`; staging fallback `upload_staging_` |
| Workgroups | 1D 256, 2D 16x16 (RX 9060 XT) | 1D 128, 2D 8x8 (Mali) | from device limits in `Context.cpp`; `WG`, `WG2D`, `MAX_INVOCATIONS` |

The other forks are not device-class dependent: the retry path, the transfer-queue upload and host-pointer and dma-buf
import (`ImportHostFrame`, `ImportDmaBufFrame`, with `decimate_yuyv` reading the luma of a YUYV frame) depend on API
use or on extensions.

### 4.2 The two shader families

With `storageBuffer8BitAccess` the decimated and thresholded images are one byte per pixel and the `_u8` shaders read them
as bytes or as words. Without it they are 4 bytes per pixel in the plain shaders, and the following are unavailable: the
adaptive floor, the fused threshold, sparse labelling, count runs, `uf_merge_x4` and the aliasing of the decimated image
at decimation 1. The 32-bit family is therefore a different pipeline, not only a different encoding; detections can
differ on noisy frames because the contrast floor is the fixed one. Both test devices have 8-bit storage; the 32-bit
family is exercised only through `FORCE_NO_8BIT`.

## 5. Memory, buffers and ingest

`vk::MemoryKind` (`vk/Buffer.h`) names where a buffer lives. `GpuDetector::CreateBuffers` uses:

| Buffer | Kind | Size |
| --- | --- | --- |
| `gray_buf_` | `DeviceLocalMapped` (written by the host directly); `upload_staging_` (`HostVisible`) only when the device-local memory is not host-visible | width x height bytes, rounded to words |
| `counter_staging_` | `HostVisibleCached` | 64 bytes (seven 4-byte counter slots: the labelling flag, boundary count, selected count, point count, raw blobs, hash drops, oversized sort blobs) |
| `selected_extents_buf_`, `line_fit_points_buf_` | `HostVisibleCached` (read in place by the host) | `max_blobs` x 32 bytes; `ipoint_capacity_` x 8 bytes |
| `readback_staging_` | `HostVisibleCached`, grows on demand | used only by the unfused flow when a payload buffer is not host-visible |
| decimated, thresholded images | `DeviceLocal` | 1 byte per pixel with 8-bit storage, else 4 |
| `parent_buf_`, `blob_size_buf_` | `DeviceLocal` | 4 bytes per decimated pixel each |
| boundary points, keys, hash table, extents, indices, scratch | `DeviceLocal` | `qbp_capacity_` x 4 and x 8, hash table 4 bytes per slot, `IPoint` 12 bytes per point |
| indirect-argument buffers | `DeviceLocal` with the indirect usage | 12 to 36 bytes |
| completion flag | `DeviceCoherentHost` (else `HostVisible`) | one word |

`GpuDetector::device_bytes()` and
`DescribeSizing()` report the total. Frames must be tightly packed; a stride is supported in the CPU stages
(`Image`) but not in the GPU upload or the tracking crops.

Ingest paths: `Detect(gray)` copies into `gray_buf_` (or staging, or the transfer lane); `ImportHostFrame` registers
host memory aligned to `min_host_pointer_alignment` so the GPU reads it in place (the pointer is the key later passed to
`Detect`); `ImportDmaBufFrame` does the same with a dma-buf descriptor, optionally a YUYV frame.

## 6. CPU tail

### 6.1 QuadDecode (`gpu/QuadDecode.cpp`)

`QuadDecode::Decode(line_fit_points)` groups the points into per-blob spans (sorted by blob index and angle) and fits each
blob in parallel on the worker pool; results are collected in span order so the output does not depend on thread
scheduling.

- **DP seeding (`quad_fit_method = kDp`, the default).** Four corner seeds from the two mutually farthest points and the
  point of maximum perpendicular deviation on each arc; each segment's line is fitted from its integer moments. If a
  segment exceeds `max_line_fit_mse` (10) the blob falls back to **peaks** (`kPeaks`): windowed line-fit error, a 7-tap
  Gaussian filter, peak detection and a combinatorial search over the 10 best peaks. `APRILTAG_VK_QUADFIT=peaks` forces
  peaks. Moments use `int64` with an exact 128-bit fallback for very large blobs.
- **`UpdateFitQuad`** refits the four lines, intersects neighbours (rejects a determinant below 0.001), checks the area
  (at least `0.95 * max(3, tag_width / decimation)^2`), convexity and winding (`cos_critical_rad` 0.98), and scales the corners
  to full resolution.
- **Hull quad (opt-in `hull_quad` / `HULL_QUAD`).** For a blob with no valid quad, or whose quad leaves more than 2% of its
  points farther than 2 decimated pixels from every side, and of at least 20 decimated pixels, `FitQuadHull` runs: convex hull,
  Douglas-Peucker, the four long edges, then per side the line of greatest support. At most 6 such fits run per frame
  (`kMaxHullAttempts`, the largest blobs first, in a second parallel pass). It exists for tags with a partly covered border.
- **Small-tag rescue (opt-in `small_tag_rescue` / `SMALL_RESCUE`, decimation above 1).** The GPU lowers `min_cluster_pixels`
  to 8 and the contrast floor to at least 8; `QuadDecode` turns blobs with no quad, a mean gradient of at least 60 and a
  box up to 40 px into `DetectedQuad::coarse` boxes (merged within 8 px, at most 64, dropped if inside a fitted quad), scores
  boxes and small undecoded quads by distance from a 14 px side, and lets only the best 4 per frame (`kMaxRescueRefits`)
  reach the refit in `TagDecoder`. This is one concept read by three components; `ResolveSmallTagRescue` in
  `common/Options.cpp` serves all of them.

### 6.2 TagDecoder candidate steps (`TagDecoder.cpp`)

`TagDecoder::Decode(quads, image, reversed_border)` builds a `DecodeEnv` once (switches, the first family's border width,
the third-bit index), runs `ProcessCandidate` for every quad in parallel, merges the per-quad lists in quad order,
then `own::Reconcile` and `own::SortById` (stable: equal ids keep reconcile's order). `RunSteps` runs these steps:

| Step | What | Default | Switch |
| --- | --- | --- | --- |
| `NormaliseCandidate` | a coarse box is refitted by `FitSmallQuad` (Otsu, largest dark component, minimum-area rectangle of its hull); without the rescue it is dropped | n/a | `SMALL_RESCUE` |
| `MarginGateRejects` | skip quads whose unrefined decision margin is below `probe_margin` grey levels; with one family only | off (0) | `PROBE_MARGIN`, `PROBE_STATS` |
| `PreRefineRejects` | reject quads whose border is not mostly dark (fewer than 22/28 of its cells darker than the midpoint) or whose payload has fewer white cells than any code (less two cells of correction), on the unrefined quad; with grey margin on, only if the inner W cells fail too; kept: coarse boxes, quads under 1.5 px per cell, quads the small rescue would refit | on | `BORDER_PRECHECK`, `DARK_REJECT`, `EARLY_REJECT` |
| `RefineCandidate` | edge refinement (section 6.3) | profile | `REFINE`, `TagDecoderConfig::refine_edges` |
| `DecodeCandidate` | sample the cells through the homography, look the code up | n/a | none |
| `TryInnerSquare` | if undecoded and the inner W cells of the (W+2)-cell outline (a grey quiet zone) decode unrefined, refine that inner quad and decode it | on | `GREY_MARGIN` |
| `TrySideOneOut` | refit a side whose border cells read white one cell further in; keep it if the same id decodes with a darker border | on | `SIDE_ONE_OUT` |
| `TryThirdBit` | accept a third corrected bit when the three wrong cells touch (`MatchThreeTouchingBits`), families with minimum distance 9 or more | on | `THIRD_BIT` |
| `TrySmallRescue` | refit an undecoded small quad (side up to 40 px, gradient at least 60) from the full-resolution image, then decode | off | `SMALL_RESCUE` |
| `ApplyFamilyLimits` | drop detections that needed more corrected bits than the family tolerates (`MaxCorrectedBits`), except the third-bit match | on | `FAMILY_LIMITS` |
| `PolishFound` | whole-tag photometric refit of each detection (section 6.4) | off | `POLISH` |

Grey margin, side-one-out, third bit and the margin gate read the first family only; the pre-refine checks run only with
exactly one family. A single border polarity per run (`reversed_border`) applies to every family.

### 6.3 RefineEdges (`RefineEdges.cpp`)

`RefineEdgesMethod::kProfile` is the default and is what every caller uses; `kExact` is upstream's gradient search in
double precision and is also the fallback when the first profile pass fails (more than 96 taps, or a degenerate side fit). `APRILTAG_VK_REFINE=exact`
selects it.

The profile fit (`RefineEdgesProfile`, `ProfilePass`): along each side a profile of bilinear taps (0.25 px apart, from
`decimation + 1` px either side) is taken every 1.5 px; the edge is the centre of mass of the gradient lobe around the
strongest step; a strength-weighted total-least-squares line is fitted through the points (`FitSide`). The refinements are
bits of `APRILTAG_VK_PROFILE_FEATURES` (default 31, read once per process): bit 1 outlier drop, bit 2 coarse-to-fine passes
(narrower windows, up to three), bit 4 halfway-crossing placement for ramps wider than 0.6 cell, bit 8 clip bias
(`ClipBias`, `GrowQuad`), bit 16 the level pass (`LevelPass`) which, when more than 25% of the first-pass profiles touch a
window end (a blurred quad), refits each side from half a cell outside to half a cell inside at the mid grey level.
`PROFILE_FEATURES=15` reproduces the output before the level pass.

### 6.4 PolishFit (opt-in)

`PolishDetection` (`PolishFit.cpp`) refits a decoded tag's homography to every cell edge of its known code: Gaussian blur per
axis, fitted black, white and outside levels, clipping, optional motion blur segment, Gauss-Newton with a 16-column normal
system, in float. The working set is capped (`POLISH_N`, 8000 pixels); a larger tag keeps its profile fit unless it is
blurred or clipped. It runs after decode on decoded tags only. Knobs: the `APRILTAG_VK_POLISH_*` variables.

### 6.5 Decode, families and reconcile (`OwnDecode.cpp`, `Family.cpp`)

`own::Decoder` samples a quad against every family of its border polarity: a homography solved by Gaussian elimination, a
grid of cell samples with a sharpening kernel (`TagDecoderConfig::decode_sharpening`, 0.25) and a grey model fitted from
the border, the decision margin, then a code lookup through a four-block index that returns the lowest code within
`max_hamming` (default 2, 0 to 3) over the first rotation that has one. `own::Reconcile` removes the lesser of two
overlapping detections of one tag. The families (`Family::ByName`: `tag36h11`, `tag25h9`, `tag16h5`, `tagCircle21h7`,
`tagCircle49h12`, `tagStandard41h12`, `tagStandard52h13`, `tagCustom48h12`) are generated into `src/families/*.cpp` by
`tools/gen_families.py`; a decoder takes at most 8 families, at most 32 cells wide and 63 bits, else `Decode()` throws
`std::invalid_argument`. The code derives from AprilRobotics' `apriltag` v3.4.5 (BSD-2-Clause, see `NOTICE`) and is checked
bit for bit against the results recorded from upstream (section 10).

### 6.6 Shared helpers

`Geometry.h` (namespace `geom`): bilinear sampling, side sampler taps, line and corner intersection, homography
projection, 3x3 inverse and Gaussian solve, monotone-chain hull, side and bounds helpers. `BorderSample.cpp` reads the
border and quiet-ring grey levels from a homography. `SmallQuad.cpp` is the rescue refit. These are deliberately not
shared with the integer moment arithmetic of `QuadDecode` (`FitLine`) or the float inner loops of `PolishFit`, whose
rounding they would change.

## 7. Composites and pose

### 7.1 FramePipeline

`FramePipeline(detector, quad_decode, tag_decoder)` overlaps the GPU pass of frame N with the CPU tail of frame N-1: each
slot owns a thread that runs `GpuDetector::Detect`; `Push` hands over a frame and decodes the one before it on the caller's
thread. Results arrive one `Push` late (two with a second `GpuDetector` that has its own `vk::Lane`, which doubles the device
memory). The line-fit records are copied out of the in-place view before the next `Detect`. `Flush()` drains it. The frame
buffer must stay valid until `depth()` further calls have returned.

### 7.2 TrackingDetector (opt-in)

`TrackingDetector(ctx, decoder_config, detector_config, TrackingConfig)` owns a `GpuDetector`, a `QuadDecode` and
decoders. With `TrackingConfig::enabled` (`APRILTAG_VK_TRACKING`, default off), `Detect` runs the full GPU pipeline on a
keyframe and, on the frames between (up to `keyframe_interval` = 8 consecutive), re-detects the previous frame's tags on
the CPU alone: the corners are predicted by constant velocity and fed through `TagDecoder` (refine and decode, up to
`max_iterations` = 3 until they move less than `converge_px`); a track is accepted if the same id decodes with a margin of
at least `min_margin` (10) and 0.3 of its previous margin, its centre moved at most half a side and its side changed at
most 1.4x. A failure sends the frame to the full detector and backs off (`skip_`). New tags appear only on full frames.
`TrackingConfig::async` (`TRACKING_ASYNC`) instead runs the full detector on a background thread while tracked frames
return at the tracked cost; background results join as tentative tracks, measured on the current frame from a
box-decimated start (`CoarseStart`) before they are reported, and a lost track stays out of the output until the background
finds it again. Tracked frames never reach `GpuDetector::Detect`, so the detector calls `NoteFrame()` and
`ExpectGpuFrame()` to keep the pre-warm's cadence (section 8.3). Not thread-safe: call `Detect` from one thread.

### 7.3 MultiLevelDetector

`MultiLevelDetector(ctx, base, levels)` runs a `GpuDetector` and `QuadDecode` per `LevelSpec` (decimation, box or copy,
minimum and maximum tag side in pixels), coarsest first, and merges their quads: a quad of a finer level is dropped if
its centre is within a third of a side and its side within a quarter of one already taken (`SameTag`). It returns quads, not
detections. It is kept as a class; measured slower than decimation 2 (`DECISIONS.md`).

### 7.4 PoseEstimator

`PoseEstimator(intrinsics, tagsize_metres, cpu_threads, convergence_tol, pool)`: homography seed, orthogonal iteration and
the planar-ambiguity second minimum, in fixed-size arithmetic, ported from libapriltag's `apriltag_pose.c`. `EstimateAll`
runs the detections in parallel. Iteration stops early at relative change 1e-8 (0 restores the full 50 iterations). Pinhole
intrinsics only: undistort corners first. It is not part of `Detect`; it needs intrinsics and the tag size.

## 8. Threading, keep-alive and pre-warm

### 8.1 One worker pool

`WorkerPool` (`common/WorkerPool.cpp`) is a persistent pool; work is handed out by an atomic index and the caller drains
batches too. `ParallelFor(count, fn(i, slot))` returns when all items are done; `slot` indexes per-thread scratch. The
pool is shared by `std::shared_ptr` among stages that run one after another: `MakeWorkerPool(config)` builds it and passes it
to `QuadDecode`, `TagDecoder` and `PoseEstimator` (`TrackingDetector` and `MultiLevelDetector` do this internally). A call from
inside the pool's own task throws `std::logic_error`; calls from two threads serialise (`contended_calls()`). The
asynchronous tracking background gives its quad fit and decoder pools of their own, because the tracked frame and the
background run concurrently. Thread count: a non-zero `cpu_threads` wins over `APRILTAG_CPU_THREADS`
(`ResolveThreadCount`), else the hardware count, or on a CPU with unequal core capacities (Linux
`/sys/devices/system/cpu/cpu*/cpu_capacity`) the number of the highest-capacity cores, to which the workers are pinned
(`cpu_big_cores`, `APRILTAG_CPU_BIG_CORES`, default on; the background tracking decoder never pins). There is no spinning, no inline
batches and no process-wide pool registry. The pool is used by the CPU tail and pose only; the GPU and the driver threads are
separate.

### 8.2 The Vulkan lanes

`vk::Context` creates the device, the compute queue(s) and a pipeline cache; a `vk::Lane` is one queue's submission state
(a command pool, a ring of command buffers and their fences) and is used by one thread at a time. A detector submits through its lane (`ctx.default_lane()` by default). `ContextOptions::expected_streams`
and `Context::StreamSlot` spread several streams over two compute families where the device has them (discrete only; one
shared queue is faster on the Mali). `CreateSiblingLane` gives the pre-warm helper its own lane on the same queue.

### 8.3 Keep-alive and pre-warm (`gpu/KeepAlive.cpp`)

A GPU idle for a few milliseconds is slow to start the next frame. `KeepAlive` is a thread that submits one-thread
dispatches (`keepalive.comp`) on its own lane: periodically every `keepalive_ms` while frames arrive (stops 100 ms after the
last `Touch`; default 6 ms on integrated GPUs, off elsewhere), and predictively, a burst of `prewarm_burst` (2) dispatches
`prewarm_lead_us` (3000) before the next frame is due, armed only when the last four or more inter-frame intervals are within 25% of
their median and the median is below 200 ms. `Detect()` calls `Touch`; `Prewarm()` submits a burst at once (for an
application that knows a frame is coming). On Windows the wait uses a high-resolution waitable timer.

## 9. Options, public API and build

### 9.1 The options registry

Every `APRILTAG_*` environment variable is a row of `OptionTable()` in `common/Options.cpp` (82 rows: 79 variables plus the three
platform variables the cache directory reads) with type, default text, category (feature, tuning, device forcing, diagnostic,
platform), precedence and the configuration field it overrides. `Options::FromEnvironment()` copies them once; a detector,
decoder, tracker or context takes one copy at construction, and a few process-wide statics (`ProfileFeatures`, the polish knobs,
`WaitFence`'s spin) take one at first use. The precedence rule: the environment overrides the configuration, except
`cpu_threads`, switches that can only enable or only disable, and variables with no configuration field. Reading an unregistered
name throws. The reference table is generated (`vk_eval --list-options`) into `PERFORMANCE.md` section 2; the `options_lint` test
fails if the library reads the environment anywhere else, if a source names an unregistered variable, or if the table drifts.

### 9.2 The public interface (`include/vkapriltag/`)

Only these headers are installed (`deps_lint` enforces it, and that they include no Vulkan header):

| Header | Contents |
| --- | --- |
| `Detector.h` | `Detector` (pimpl; `Detect`, `Flush`, `ImportHostFrame`, `ImportDmaBufFrame`, `Reset`, `Describe`), `DetectorOptions`, `Mode`, `Level`, `Result`, `Timing`, `Device`, `DeviceOptions`, `Error`, `ErrorCode` |
| `Detection.h` | `Detection` (family, id, hamming, `decision_margin`, `H[9]`, centre `c`, corners `p`), `DetectionList` |
| `Family.h` | `Family` (`ByName`, `All`) |
| `Pose.h` | `CameraIntrinsics`, `TagPose`, `TagPosePair` |
| `Version.h` | `VKAPRILTAG_VERSION_*` (2.0.0; CMake checks it against the project version) |

The stages (`GpuDetector`, `QuadDecode`, `TagDecoder`, `FramePipeline`, `TrackingDetector`, `MultiLevelDetector`, `PoseEstimator`,
the Vulkan layer in `vk/`, `common/`) live in `library/src/internal/vkapriltag/`. They are not installed; in-tree tools and tests
reach them through the `vkapriltag::internal` target. `facade_test` and the installed-package consumer link only
`vkapriltag::vkapriltag`.

The corner order of a `Detection` is the order the decode reports. A `Result` is owned by the `Detector` and valid until the next
call; the Python binding returns copies.

### 9.3 The per-frame call sequence

```cpp
DetectorOptions options;
options.width = w;  options.height = h;
Detector detector(options);                         // family names, decimation, mode, features and pose are options
const Result &result = detector.Detect(grey);       // detections, poses (when options.tag_size > 0), timing
```

`Detector::Impl` (`src/Detector.cpp`) composes the internal stages by mode:

| Mode | Stages |
| --- | --- |
| `kSync` | `GpuDetector` then `QuadDecode` then `TagDecoder` |
| `kPipelined`, depth 1 or 2 | `FramePipeline` over the same three (depth 2 gives the second `GpuDetector` its own `Lane` on the same queue: a lane is used by one thread at a time) |
| `kTracking` (keyframe or `tracking_async`) | `TrackingDetector` |
| `levels` set (kSync only) | `MultiLevelDetector` then `TagDecoder` |

One `WorkerPool` serves the quad fit, the decode and the pose estimator. `PoseEstimator` runs on the result when `tag_size` is positive.
Combinations the stages cannot do (levels with tracking or pipelining, families of different border width) throw
`Error(ErrorCode::kUnsupported)`. `DetectorOptions::stream` picks a queue of a shared `Device`; a device with fewer queues than
streams shares its first queue.

### 9.4 Build and dependencies

CMake 3.16, C++20, Vulkan headers and loader, `glslangValidator` (the integer dot product shader needs glslang 15.2 or later;
`VKAPRILTAG_DOT_SHADER=AUTO|ON|OFF`). A default configure fetches nothing and needs no C compiler. Options: `VKAPRILTAG_BUILD_APPS`,
`VKAPRILTAG_BUILD_TOOLS`, `VKAPRILTAG_BUILD_TESTS` (default on), `VKAPRILTAG_EMBED_SHADERS` (SPIR-V compiled into the library, default
on), `VKAPRILTAG_UPSTREAM_VALIDATION` (default off: fetches stock libapriltag v3.4.5 into `tools/upstream` for the reference tools;
excluded from `all`, install and packages) and `VKAPRILTAG_REGEN_VECTORS` (also patches it to rewrite the decode test vectors).
`find_package(vkapriltag)` needs only Vulkan and Threads. `.github/workflows/release.yml` builds the `.deb` packages (glslang 16.2.0)
and checks they contain the dot product shader and nothing from upstream; `vkapriltag-parity.yml` runs the default `ctest` on
lavapipe, then the upstream parity gate.

## 10. Tests

`ctest --test-dir <build> --output-on-failure` (`tests/CMakeLists.txt`); 17 tests:

| Test | What it checks | Needs a GPU |
| --- | --- | --- |
| `unit_decode`, `unit_geometry`, `unit_geom`, `unit_edges`, `unit_quadfit`, `unit_options`, `unit_steps`, `unit_pool` | `tests/unit_tests.cpp` groups: third-bit matching and error connectivity; helper known values; edge locators on rendered squares; quad fit on rasterised rings (dp and peaks); registry parsing and precedence; the candidate steps; the worker pool contract | no |
| `own_decode_random`, `own_decode_reconcile`, `own_decode_corpus`, `families_match_upstream` | the decode bit for bit against upstream's results recorded in `tests/own_decode_vectors.bin` (3120 random and 2974 corpus quads, 2500 reconcile sets, family table hashes) | no |
| `own_decode_comparison_can_fail` | the comparison fails when the decode is sabotaged (`WILL_FAIL`) | no |
| `golden_detections` | `tests/golden_test.py`: `vk_eval` over the committed subset (`tests/corpus`, 10 scenarios x 5 frames, 640x480 PNG) in five profiles (`tests/profiles.json`: d1, d2, `rescue-hull`, `polish`, `features15`) against `tests/golden/<profile>/`: same ids and Hamming distances per frame, corners within 0.02 px, at most 2 differing frames per profile | yes |
| `gpu_trace_*` | `tests/gpu_trace_test.cpp`: the recorded GPU work of the first frame (every dispatch with its grid or indirect offset and a hash of its push constants, barriers, fills, copies, submits) per device and profile, compared with `tests/gpu_trace/<device>/<profile>.txt` (RX 9060 XT and Mali-G610 committed; other devices skip). `GPU_TRACE_UPDATE=1 ctest -R gpu_trace` rewrites them |
| `facade_*` | `tests/facade_test.cpp` (linked against the public target only): every mode, pose, strided input, a shared device and the errors; `golden_detections_facade` runs the five golden profiles through `vk_eval --facade` |
| `swig_python` | `bindings/swig/swig_test.py` (when built with `VKAPRILTAG_BUILD_SWIG=ON`): a corpus frame through the Python module, numpy and bytes input, stride, pose, pipelined mode and the exceptions |
| `unit_plan` | `tests/unit_tests.cpp` group `plan`: the resolved `PipelinePlan` of a discrete and an integrated device class under 21 variations against `tests/plan/<class>.txt`, and the stage table's consistency |
| `options_lint` | no `getenv` outside the registry, every named variable registered, `PERFORMANCE.md` table equals `vk_eval --list-options` | needs `vk_eval` built |
| `deps_lint` | no upstream types or includes outside the validation files, `VKAPRILTAG_UPSTREAM_VALIDATION` defaults off | no |
| `installed_package_consumer` | installs to a temporary prefix, checks nothing from upstream is in it, builds a `find_package(vkapriltag)` consumer and runs it on a corpus frame (id 316) | yes |

Regenerate the goldens after an intended behaviour change, from a build of that change:
`python tests/golden_test.py --vk-eval <build>/tools/vk_eval --update`. Regenerate the decode vectors with
`-DVKAPRILTAG_UPSTREAM_VALIDATION=ON -DVKAPRILTAG_REGEN_VECTORS=ON` in a separate build directory
(`tools/upstream/gen_decode_vectors.cpp`). Regenerate the options table with `vk_eval --list-options` between the
`options:begin`/`end` markers of `PERFORMANCE.md`. Score a corpus with `tools/eval.sh` and the harness described in
`PERFORMANCE.md` section 5. The optional validation tools (`apriltag_vulkan_validate`, `apriltag_pose_validate`,
`apriltag_pose_e2e_validate`, `apriltag_synth_ground_truth`, `vk_eval --ref`) need `-DVKAPRILTAG_UPSTREAM_VALIDATION=ON`.
`tools/check_links.py` checks every relative link in the Markdown files.

## 11. Invariants and gotchas

- **What is deterministic.** The CPU tail is deterministic for equal input: candidates are decoded in parallel but merged in quad order
  and ordered by id with a stable sort, so thread count does not change the output. The GPU front end is not bit-stable: boundary
  points are appended with an atomic counter, so run to run a marginal detection can differ by one dump line and corners by about
  1e-3 px (observed in every A/B of 2026-11, also base against base); ids and Hamming distances do not change in practice. This is why
  the golden test has a tolerance and a noise allowance and why dumps cannot be required to be byte-identical across runs.
- **Async tracking is not reproducible.** Which frame a background result lands on depends on thread timing, so recall of
  `TrackingConfig::async` is bimodal on some sequences (for example 84% or 95% on the same input, `track_eval`); `full` and `key`
  modes are reproducible.
- **Floating point.** `QuadDecode.cpp` and `OwnDecode.cpp` are compiled with `-ffp-contract=off` on GCC and Clang, so their results do not
  depend on which multiply-adds a compiler fuses; MSVC does not fuse by default. The other CPU files (`RefineEdges.cpp`, `PolishFit.cpp`)
  are not pinned. `own_decode_*` passed on aarch64 with GCC.
- **One border polarity per run**, one decimation per detector (a specialisation constant), width and height divisible by the decimation,
  tightly packed frames, and the decoded image for `TagDecoder` must be the same full-resolution frame the GPU saw.
- **The unrefined paths read the first family only** (section 6.2): a multi-family detector gets fewer of the early rejects.
- **Labelling connectivity.** `uf_init_scan` and `uf_merge` join left and down neighbours, so white regions are 4-connected; upstream
  unions white diagonally (`research/00-SUMMARY.md`, finding 0.1). The corpus agrees tag for tag; no one has measured the difference
  on adversarial images.
- **Sparse and dense paths give the same detections** (golden test with `SPARSE=0`), as do the other fallbacks on the golden set; the 32-bit
  shader family can differ on noisy frames (section 4.2).
- **Stale header comments.** `GpuDetector.h` still describes the "four-submit layout" in `kGpuGapCrossesSubmit` and labels the unfused
  submits 1 to 5 in a few code comments; the unfused flow has four submissions.
- **Environment variables are read at construction.** Change nothing in the environment after creating the detector; two detectors in one
  process share the process-wide statics (`PROFILE_FEATURES`, the polish knobs).
- **Large frames.** `2 * (width / decimation)` must be at most 16383 (packed point coordinates); `max_raw_blobs` at most 2^22.
