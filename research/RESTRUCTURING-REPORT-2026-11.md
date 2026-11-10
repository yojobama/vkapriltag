# vkapriltag: restructuring for maintainability and speed

Research report, 2026-11-11, on branch `research/improvements-2026-11` at `27f77ea`. Nothing in the
library was changed for it. It answers two questions: can the project be made easier to maintain, and
can restructuring (rather than more tuning) make it faster.

## 1. Summary

**Maintainability: yes, substantially, and mostly by deleting and consolidating rather than rewriting.**
The library is small (14,100 lines of C++ and 2,700 of GLSL) but its behaviour is spread over 79
environment switches, 37 configuration fields, two complete GPU frame flows, five edge-refinement
methods and three overlapping documents totalling 470 KB. There are no automated tests below a single
CI parity run, so every old path is kept "for comparison" because nothing else guards a change.

**Speed: restructuring gives about 10 to 25% on the Orange Pi and less on the desktop. It does not close
the gap to NAPA on ordinary frames.** The reason is structural and already recorded in
`PERFORMANCE.md` 8as: a frame pays for a chain of 16 to 19 dependent GPU dispatches whatever its
content. The GPU stage alone is 0.59 ms on the Pi against NAPA's 0.16 ms whole frame. Further shader
tuning has reached 5 to 10% per idea with most ideas losing. The larger wins come from choosing *which
front end runs* per frame, which the current structure makes hard.

Recommended order (details in section 6):

| # | Step | Kind | Effort | Expected effect |
|---|---|---|---|---|
| 1 | Bring the corpus harness and scorers into the repo; add `ctest` golden-output and unit tests | safety net | M | Makes every later step checkable |
| 2 | Delete paths nothing needs: the unfused frame flow, three refinement methods, dead switches | deletion | S-M | About -1,000 lines, -20 switches |
| 3 | One options registry, resolved once | consolidation | M | Removes 84 scattered `getenv` sites and three process-wide statics |
| 4 | Split the CPU decode into a candidate pipeline with a shared first sample | restructure + speed | M | CPU tail -25 to -45% on cluttered frames (measured upper bound) |
| 5 | GPU: a resolved pipeline plan and a stage table instead of flags | restructure | L | Removes the 113- and 89-complexity functions; one flow |
| 6 | One facade and an own detection type; narrow the patched-libapriltag dependency | API | M-L | 7 objects down to 1; no patched DLL to ship |
| 7 | A front-end interface (GPU blobs, tracker, rescue boxes, later a CPU front end) | architecture | L | Lets the cheap path run when it suffices |
| 8 | Run the pipeline's tail on the CPU for sparse frames on integrated GPUs | speed | M | Pi about -0.1 ms on the median frame (estimate from a micro-benchmark) |
| 9 | Replace the three large documents with an architecture page, a generated switch reference and a decision log | docs | M | One place for the current truth |

## 2. How this was assessed

- Size, churn and complexity: `wc`, `git log --name-only`, `lizard` (cyclomatic complexity), `cppcheck`.
- Reading: I read the GPU side (`GpuDetector.h/.cpp`, the shader variants, `vk/`) and the build and CI
  files. Two Opus 5.5 agents, read-only, covered the CPU side with the public API and mined the three
  large documents; I spot-checked their main claims in the code (marked "checked" below).
- Measurements made for this report (desktop RX 9060 XT / Ryzen 5600X, and Orange Pi 5 Plus with the
  service stopped): per-frame boundary-point counts over the corpus, a CPU micro-benchmark of the GPU
  tail's work, upstream libapriltag on the Pi, and the CPU cost of the decode retries.

## 3. Where the project stands

| | |
|---|---|
| Library C++ | 14,067 lines, 483 functions |
| Shaders | 61 files, 2,656 lines, two storage families (`uint` and `uint8_t` per pixel) |
| Functions with cyclomatic complexity over 30 | 17 |
| Worst functions (complexity / lines) | `TagDecoder::Decode` 141 / 381, `GpuDetector::DetectUnfused` 113 / 422, `GpuDetector::CreatePipelines` 89 / 209, `PolishDetection` 83 / 235, `QuadDecode::Decode` 76 / 186, `TrackingDetector::TrackFrame` 67 / 120 |
| Churn | `GpuDetector.cpp` is touched by 119 of 422 commits, its header by 95 |
| Environment switches read by the library | 79 distinct names at 84 `getenv` sites (36 in `GpuDetector.cpp`) |
| `DetectorConfig` fields | 37, plus separate `TagDecoder` constructor arguments and `TrackingConfig` |
| Automated tests | none (`enable_testing()` with no `add_test`); CI runs one parity tool on lavapipe with `refine_edges` off |
| Documents | `PERFORMANCE.md` 226 KB, `OPTIMIZATION_NOTES.md` 123 KB, `DESIGN.md` 123 KB, eight research files of 60 to 80 KB |
| `cppcheck` | 225 findings: 130 unused struct members, 17 uninitialised members, 22 style; the 17 "out of bounds" in `PoseEstimator.cpp` are false positives (checked) |

## 4. Why it stopped getting faster

### 4.1 The cost is the chain, not the pixels

Whole corpus, median per frame, from the comparison run on 2026-11-10:

| | Desktop | Orange Pi |
|---|---|---|
| vkapriltag, decimation 2 | 0.311 ms (GPU 0.18, CPU 0.12) | 0.743 ms (GPU 0.59, CPU 0.15) |
| Empty frame | 0.14 ms | 0.43 ms |
| NAPA, default | 0.091 ms | 0.156 ms |

`PERFORMANCE.md` 8as and 8at give the mechanism: about 19 dependent dispatches (16 after the lean tail)
each wait for the previous one, and a trivial dependent dispatch costs about 25 us on the Mali and 4.5 us
on the RX 9060 XT (19 of them: 464 to 504 us and 82 to 87 us). The document ledger shows what has been
tried against that floor: fused submits, pre-recorded command buffers, fused threshold, lean tail,
persistent tail, fused arguments, barrier changes, flag polling. Most of the kept ideas were worth 2 to
8% each and several lost. The documents' own conclusion is "each worth about 5 to 10% on the Mali, not 3 times".

### 4.2 What the frames actually contain

Counts per frame at decimation 2 over the 4,203 corpus frames (new measurement):

| | Median | 90th percentile | Largest scenario (jpeg) |
|---|---|---|---|
| Boundary points | 4,452 | 14,772 | 81,594 median |
| Raw blobs | about 12 | | 414 |
| Blobs passing selection | about 3 | | 167 |
| Points sorted | about 1,500 | | 31,152 |

80% of frames have under 10,000 boundary points. For these, the seven dispatches after boundary
extraction (group, extents, select, scatter, three sorts) move a few thousand records each.

### 4.3 The CPU tail spends much of its time on candidates that are not tags

`TagDecoder::Decode` refines every candidate with the profile fit before any rejection: the two cheap
pre-refine rejectors (border pre-check, dark reject) are skipped whenever grey margin is on, and it is on
by default (`TagDecoder.cpp:403`, `:423`; checked). A candidate that then fails to decode is retried by
grey margin (a second homography and sampling) and by the third-bit lookup (a third, resampling the quad
that was just sampled). Desktop, CPU part of the frame in ms, defaults against those two retries off:

| Scenario | Default | Retries off |
|---|---|---|
| clutter-only | 0.20 to 0.25 | 0.11 to 0.12 |
| jpeg | 0.47 | 0.31 to 0.36 |
| busy-background | 0.235 | 0.171 |
| one-tag | 0.061 | 0.066 |

So on frames with many non-tag candidates 25 to 45% of the CPU tail is retry work on rejects. Turning
the retries off loses recall; restructuring them (section 6.4) should keep the recall and most of the
saving.

### 4.4 A CPU port of the same algorithm is not the answer

Upstream libapriltag on the Pi at decimation 2 takes 21.5 ms on one core and 9.2 ms on four for a
one-tag frame, against 0.67 ms for the GPU path. NAPA is 60 to 130 times faster than that with a
different algorithm that reads a fraction of the pixels. A competitive CPU front end therefore means a
different front-end algorithm, not this pipeline moved to the CPU.

## 5. Maintainability findings

### 5.1 GPU side

- **Two frame flows.** `DetectFused` (the default on both test devices and on lavapipe) and
  `DetectUnfused` (422 lines). The second still carries `const bool fuse = false;` and about ten dead
  `if (fuse)` branches (`GpuDetector.cpp:1630`). `PERFORMANCE.md` 8g says of it "No measured device is
  stuck on it" and that its dead branches go "when it does". The timestamp post-processing is duplicated
  between the two (about 50 lines, 12 differing).
- **The capability matrix is resolved by hand in one function.** `CreatePipelines` derives about 20
  booleans (`sparse_`, `count_runs_`, `merge_x4_`, `uf_init_scan_`, `threshold_block_`, `lean_tail_`,
  `reduce_x4_`, `extents_atomic64_`, ...) from device type, 8-bit storage, subgroup features, 64-bit
  atomics, host-cached memory, unified memory, decimation and about 25 environment overrides, with
  dependencies between them written as nested conditions. The same booleans are tested again in every
  `Record*` function. Nothing prints or tests the resolved combination.
- **Only two combinations are ever exercised** (discrete with everything, Mali). The 32-bit shader
  family, the plain and atomic64 extents variants and the unfused flow are reached only through
  `FORCE_NO_*` switches in hand-run matrices. Each device class is measured on one device.
- **Stages are described three times**: the pipeline object with its binding list in `CreatePipelines`,
  an anonymous push-constant struct at each call site (30 inline structs), and the dispatch in one or two
  record functions. Adding a binding means editing all three in step, plus the shader.
- **`ComputePipeline` binds buffers at construction**, so the same shader needs a second pipeline to read
  a different counter (`build_indirect_args` twice).
- **Public header.** `GpuDetector.h` (610 lines) exposes every buffer, pipeline and recording enum, and
  also defines `DetectorConfig` and `DetectedQuad`, so CPU-only code includes `vulkan.h`.

### 5.2 CPU side

- **`TagDecoder::Decode` interleaves 17 steps for 11 features** in one lambda with five early exits.
  Features communicate through shared state: the size of the per-quad detection array, `quad_original.H`
  and a per-slot scratch shared by three features. "Copy the quad, null H, sample, free" appears seven
  times, and the homography matrix is freed by hand in ten places.
- **Five refinement methods; one is used.** Every caller uses the profile fit. `kUpstream` and
  `kUltraFast` (with its own gate inside `Decode`) are reachable only through an environment variable.
  `RefineEdgesT<double>` stays live as the profile fit's fallback.
- **The same "sample along a side" loop exists four times** (`RefineEdgesT`, `RefineEdgesFast`,
  `ProfilePass`, `LevelPass`). Also duplicated: bilinear sampling (5 copies), homography projection (7),
  line fit from moments (4), corner from two lines (4), convex hull (2), 3x3 inverse (4).
- **Three corner conventions and two precision changes**: `DetectedQuad` (double) to upstream
  `struct quad` (float) to detection (double), with corner k mapped to 3-k between types.
- **Three thread pools per frame** (quad fit, decode, pose), four or five with tracking, each waking and
  joining separately; the Pi profile in 8af counts 48 futex calls per frame for 3 candidates.
- **Composites do not compose.** `FramePipeline`, `MultiLevelDetector` and `TrackingDetector` each wrap
  the stages differently (owning or not, stopping at quads or detections, `Detect` / `Decode` / `Push`).
  There is no pipelined tracking or multi-level tracking.

### 5.3 Configuration

- **Precedence differs by switch**: the environment beats the configuration for most, the configuration
  beats the environment for `cpu_threads`, and two header comments state the opposite of the code
  (`QuadDecode.cpp:752`, `RefineEdges.h:25`).
- **One concept, several readers.** Small-tag rescue is read in `GpuDetector`, `QuadDecode` and
  `TagDecoder`; `TagDecoder` has no configuration field for it, so setting
  `DetectorConfig::small_tag_rescue` in code turns on the GPU and quad-fit halves while `TagDecoder`
  discards the boxes (`TagDecoder.cpp:341`; checked). Decimation has three sources.
- **Process-wide state**: `APRILTAG_VK_PROFILE_FEATURES` and two polish switches are function statics;
  seven polish constants are read before `main`; the worker-pool registry lets one detector's keep-alive
  wake every pool in the process. Two detectors in one process cannot be configured differently.
- **Documentation gap**: `PERFORMANCE.md` section 2 says it lists every variable the library reads; about
  40 are missing, including every switch added since 1.10.

### 5.4 Dependency on patched libapriltag

Four patches expose `quad_decode_index`, `reconcile_detections`, `refine_edges` and a code probe, and
remove allocations. Costs: a network fetch at configure time; the patch script continues with a warning
when hunks fail; a patched `apriltag.dll`/`.so` is built shared and linked publicly, and the installed
package config can resolve to a distribution's unpatched library (inferred, not tested); public headers
return upstream `zarray_t*` of `apriltag_detection_t*`. Features that only read family 0 (third bit, dark
reject, pre-check, grey margin, side one out, the profile fit's cell size) make multi-family support
partial.

### 5.5 Tests and harness

- No unit or integration tests. CI never runs any refinement method as such, grey margin, third bit,
  rescue, hull, polish, tracking, `FramePipeline` or `MultiLevelDetector`.
- The harness every measurement in this research used (`vk_eval`, the build wrappers, the comparison
  scripts) lives outside the repository in `ext/avk-compare`; the corpus is 3.8 GB and also outside.
- Consequence: old paths are kept as switches because a byte-identical A/B against the old path is the
  only regression check available.

### 5.6 Documents

The mined ledger found 20 contradictions or uncorrected statements, among them: submit collapsing
rejected twice and then shipped, with section 3 still saying it "is not worth it"; preprocessing fusion
deleted in one file and the default in another; `DESIGN.md` still describing `label_pixels` and the
Batcher sort, both removed; sections 8ag to 8an cited but absent; the README naming `exact` as the default
refinement. The current defaults are spread over at least 15 chronological sections and the three
"reference" sections (switch table, numbers, what is left) are stale.

## 6. Proposals

### 6.1 Safety net first

- Move `vk_eval`, `eval_common.h` and the scoring and comparison scripts into `tools/`, with one command
  that runs a corpus and prints recall, false detections, corner error and time against a stored baseline.
- Commit a small corpus subset (about 50 frames, a few MB, generated by `gen_corpus.py` with fixed seeds)
  and a golden detection dump per supported pipeline profile; run it in CI on lavapipe as a `ctest`.
- Unit tests for the pure pieces that already exist: `MatchThreeTouchingBits`, `ErrorsConnected`,
  `IntersectLines`, `FitSide`, the quad fit on synthetic point rings, the edge locators on rendered
  edges of known position and blur. The CPU tail needs no GPU to test.
- Effect: steps 6.2 to 6.8 become checkable without the hand-run matrices.

### 6.2 Delete what nothing needs

| Candidate | Lines / switches | Condition |
|---|---|---|
| `DetectUnfused` and `APRILTAG_VK_FUSE_SUBMITS`, `UF_VERIFY`, `UF_CHUNK`, `DeviceLocalReadback` | about 550 lines | Decide that host-cached readback memory is required (both devices and lavapipe have it) |
| `kUpstream`, `kUltraFast` and its gate; `kFast` (106 lines) if the 1.10 "fast" configuration is retired | about 150 to 250 lines, one patch hunk | Keep `kExact` as fallback and the profile fit |
| 32-bit-per-pixel shader family and `FORCE_NO_8BIT` | 14 wrapper shaders, the `pick()` logic | Decide that `storageBuffer8BitAccess` is required; without it the adaptive floor, sparse labelling and count runs are already unavailable and detections differ |
| A/B-only switches for settled choices: `FIND_MODE`, `MERGE_FLAG_MODE`, `DEC1_ALIAS`, `UF_INIT_RUN`, `THRESH_RUN`, `SORT_CHUNK=0`, `PRERECORD=0`, `REDUCE_X4`, `GRAY_MAPPED` | about 15 switches | Covered by golden tests instead |
| Experiments the logs call not worth it: tracking ROI search, capture, `max_speed_px`; worker spinning and inline batches; `MultiLevelDetector` ("measured slower, kept as a class") | not counted | Keep on branches |
| Unused: `QuadCorners`, `alive` in `TrackFrame`, per-frame extents copy with no CPU consumer, 130 unused struct members reported by `cppcheck` | small | |

Each of the first three is a product decision about supported devices, not only a clean-up; the
documents name no real device that needs them.

### 6.3 One options registry

One table of options (name, type, default, per-device-class default, category: feature / tuning /
diagnostic / A-B, documentation string), parsed once into an immutable `ResolvedOptions` held by the
detector. Environment overrides are applied in that one place with one precedence rule. The switch
reference in the documentation is generated from the table. This removes the statics, the three readers
of small-tag rescue and the two copies of `DetectorConfig`.

### 6.4 CPU tail as a candidate pipeline (maintainability and speed)

Replace the body of `TagDecoder::Decode` with ordered steps over one candidate record:

1. normalise the candidate (including the coarse-box refit);
2. one unrefined sample (homography, code, margin), shared by the margin gate, border pre-check, dark
   reject and, later, grey margin;
3. refine;
4. decode, returning the raw code so that the third-bit lookup does not resample;
5. recovery strategies in order (grey margin, third bit, rescue), each "try, return a detection or not";
6. improvers (side one out, polish);
7. acceptance (family limits); instrumentation as a wrapper.

Speed: the measurements in 4.3 bound the gain at 25 to 45% of the CPU tail on cluttered frames. How much
survives depends on whether a reject test can be made safe for grey-margin tags (the current reason the
rejectors are off); that needs an experiment on the grey-margin and low-contrast scenarios. Secondary
gains: one pool and one parallel region per frame from blob to detection, removing a wake and join and
the serial gap between quad fit and decode.

A shared `SideSampler` plus separate edge locators (lobe centroid, halfway crossing, level crossing)
would replace the four copies in `RefineEdges.cpp`; a small geometry header (homography, projection,
line intersection, hull) would replace the copies listed in 5.2.

### 6.5 GPU: a resolved plan and a stage table

- **`PipelinePlan`**: a plain struct produced by a pure function of (device capabilities, options). It
  names the profile (for example `discrete`, `integrated`, `portable`) and holds every variant choice.
  It is printed at start-up and unit-tested without a GPU, including the dependencies between choices
  that are now nested conditions.
- **Stage table**: each stage declared once as data: shader name by plan, bindings by buffer role, a
  named push-constant struct, dispatch kind (fixed grid, indirect, per cell list), barrier and timing
  span. A generic recorder walks the table. The frame is then "head, optional retry, tail" over the same
  table, and the cut points for submits are data too.
- **Descriptor sets separate from pipelines**, so one shader module serves several binding sets.
- This is the largest single step. It should come after 6.1 and 6.2, when there is one flow to convert
  and a golden test to hold it.

### 6.6 One facade, own detection type, narrower upstream dependency

- A `Detector` that owns the families, the Vulkan context or lane, the plan, one worker pool and the
  stages; input as pointer plus stride or an imported frame; output as a library-owned span of
  detections (id, family, hamming, margin, centre, corners, homography as `double[9]`) and optionally
  poses. Modes (synchronous, pipelined, tracking) as options of the same object, so they compose.
- Internal headers (`vk/*`, `KeepAlive`, `Types.h`, `QuadDecode`, `RefineEdges`, `PolishFit`) move out of
  the installed include directory.
- Upstream: own the homography, bit sampling, code lookup and overlap reconciliation (about 400 lines;
  the bucket index and rotation helpers already exist), and link upstream unpatched only for the family
  tables and as the reference in the validation tools. This removes the patched shared library, the
  hand-declared prototypes, the `struct quad` layout dependence and the manual `matd` frees. It must be
  bit-compatible with the current decode; the golden tests check that.

### 6.7 A front-end interface

Three producers of candidates already feed the same refine-and-decode tail, each wired separately: GPU
blobs (`QuadDecode`), tracker predictions (`TrackingDetector`) and rescue boxes (`FitSmallQuad`). Giving
them one interface, "produce candidate quads or boxes for this frame, given the previous result", makes
the tail independent of the GPU and gives one place to choose the front end per frame:

- tracker only when all tracks hold (0.13 to 0.18 ms on the Pi's slow sequences against 0.58 full);
- GPU on keyframes, on loss and when the frame is busy;
- later, a CPU sparse front end for small frames (6.9).

It also fixes two things seen in the comparison: keyframe mode twice as slow as full mode under fast
motion on the desktop at camera rate, and async mode's 9.8 ms of CPU per frame on the four-tag sequence;
both come from the tracker driving a full `TagDecoder` and the GPU from outside.

### 6.8 The pipeline's tail on the CPU for sparse frames (integrated GPUs)

Not found in any of the documents as tried. After boundary extraction, read the compacted points on the
CPU (the buffer can be host-visible on unified memory) and do grouping, extents, selection, sort and
moments there, when the frame is sparse.

| Boundary points | CPU cost, one A76 core | CPU cost, Ryzen 5600X |
|---|---|---|
| 2,400 | 49 us | 20 us |
| 4,400 (median frame) | 77 us | 32 us |
| 8,800 | 161 us | 63 us |
| 25,000 | 436 us | 164 us |
| 81,600 (jpeg median) | 1,809 us | 761 us |

These are from an unoptimised single-thread micro-benchmark on synthetic points
(`tail_bench.cpp`, hash grouping, `std::sort` by angle, weighted prefix moments). The GPU tail it
replaces is seven dependent dispatches: about 175 us of fixed cost on the Mali by the 25 us figure, about
32 us on the desktop. So the expected gain is about 0.1 ms on the Pi's median frame (13% of 0.74 ms),
nothing on the desktop, and a loss above roughly 10,000 points, where the GPU tail must stay. The choice
can follow the previous frame's count. This is an estimate; the real GPU tail cost on the current build
should be host-timed first.

### 6.9 A CPU sparse front end (research, uncertain)

The only route to NAPA-level times on small frames is a front end that does not touch every pixel and
has no fixed chain. Section 4.4 shows it cannot be the present algorithm on the CPU. It would be new
work (NAPA is GPL-3.0 or commercial and its code cannot be used), and its recall on the hard scenarios is
unknown. With 6.7 in place it could be developed and compared as one more front end without touching
the rest. I would not start it before 6.1 to 6.7.

### 6.10 Documents

- `ARCHITECTURE.md`: the current pipeline per profile (stage list, submits, buffers, invariants).
- A switch reference generated from the options registry.
- A decision log: one dated entry per idea with outcome, key number, commit or branch, and what
  superseded it. The mined ledger is its seed.
- One "current numbers" page per platform, regenerated at each release by the in-repo harness.
- Move sections 8b to 8bd of `PERFORMANCE.md`, `OPTIMIZATION_NOTES.md` and the two root idea files to
  `archive/` unchanged.

## 7. What restructuring will and will not buy

| | Desktop | Orange Pi | Confidence |
|---|---|---|---|
| Candidate pipeline (6.4) | CPU tail up to -25 to -45% on cluttered frames; little on clean ones | same shape | Upper bound measured; realised share unknown |
| One pool, one parallel region (6.4) | small | tens of microseconds per frame | Inferred from the futex count |
| CPU tail for sparse frames (6.8) | none | about -0.1 ms median | Micro-benchmark estimate |
| Front-end choice (6.7) | large on slow scenes (tracking already shows it) | same | Measured for tracking |
| Stage table, plan, facade, options (6.3, 6.5, 6.6) | none expected | none expected | These are for maintainability |
| Closing the gap to NAPA on ordinary frames | no | no | Needs 6.9 |

## 8. Not verified

- Whether any real target lacks 8-bit storage or host-cached memory; the documents name none.
- The packaging consequences of the patched shared library (inferred from the CMake files, not tested
  on an installed package).
- The realised gain of 6.4 and 6.8; both figures are bounds or estimates.
- The 20 document contradictions were found by the mining agent with section citations; I checked the
  ones quoted in 5.1 and 5.6 only.

## Appendix: commands and raw data

- Complexity: `python -m lizard library/src library/include -l cpp --csv`.
- Static analysis: `cppcheck --enable=warning,performance,portability,style --std=c++20 -I include src`.
- Per-frame counts: a scratch copy of `vk_eval` printing `DetectProfile` fields with `VK_EVAL_STATS=1`.
- Retry cost: `vk_eval --iterations 20 --warmup 2 --decimation 2` with and without
  `APRILTAG_VK_GREY_MARGIN=0 APRILTAG_VK_THIRD_BIT=0`.
- Upstream on the Pi: `vk_eval --ref --decimation 2 --threads 1|4`, pinned to cores 4-7.
