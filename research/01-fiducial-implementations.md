# 01 - Fiducial detector implementations (GPU/FPGA/NPU/CPU)

Entry format: Title / URL / Technique / Reported speed-up / Accuracy-determinism / Applicability to vkapriltag / Verification.
Cache root: C:/Users/yojob/AppData/Local/Temp/claude-research/

## Status

Covered so far:
- [x] 971 CUDA detector lineage: git history of RealtimeRoboticsGroup_aos `frc/orin` (+ old `frc971/orin`), frc971/bos `third_party/971apriltag`, FRC-4143 GpuDetectorJNI, Team766 apriltags_cuda — diffed file by file.
- [x] NVIDIA VPI 4.1 AprilTag docs + perf JSON (Orin, Thor)
- [x] rapidtag source
- [x] SpectrumJetson README (old vs new 971 numbers)
- [x] halide-apriltag source
- [x] OpenCV aruco_detector.cpp / aq.cpp source
- [x] upstream AprilRobotics/apriltag perf commits 2022-2026
- [x] WPILib apriltag fork patches + PhotonVision AprilTagPipeline params
- [x] Isaac ROS benchmark tables (cuAprilTags)
- [x] cached FPGA folder: Frappe, Flottmann, Tola, Zhang, ICVIP, Samarin
- [x] wykvictor/AprilTag-GPU (no CUDA inside)
- [x] argustag (source); web search found no OpenCL/Metal/WebGPU AprilTag ports besides vkapriltag itself
- [x] cached algo folder (aruco_nano, apriltags_tas; whycon only README), web literature sweep for GPU/FPGA/NPU papers
- [x] OpenCV 4.7+ ArUco speed numbers: web search found no benchmark figures (only API docs); nothing to add beyond the PV docs' "~2x" line
- [x] EagleEye (Pi 5, temporal ROIs), Limelight 4 Hailo changelog
- [x] PhotonVision ML ROI PRs #2410/#2604, PV docs throughput claims, photonvision-tools stream-cost measurement, CubVision

Next up (queue): (research complete for this pass; remaining optional items)
- `learned\` papers: titles scanned only (STag, ChromaTag, DeepTag, YoloTag, DeepArUco++ ...); learned detectors are left to category 02 unless time permits

---

## Part 1 — What changed in the newer 971 CUDA detector

**Lineage established by diffing (whitespace/CRLF-insensitive):**
- `FRC-Team-4143_GpuDetectorJNI/frc971/orin/*` = aos commit `ca912dd75` (2024-08-11, "Remove dead code from GPU apriltag algorithm") + 4143's own decimate=1 support and gray-input entry points. `points`, `labeling` and `line_fit_filter` match `ca912dd75` to within 0–37 diff lines. This is "the Aug-2024 copy".
- `Team766_apriltags_cuda/src/apriltag_gpu.cu` is closest to aos `c123b7fe3` (2024-01), i.e. OLDER than the 4143 copy. vkapriltag's ancestor is therefore the early-2024 971 kernel set.
- `frc971_bos/third_party/971apriltag/*` (vendored 2025-08-31, last touched 2026-05) = aos HEAD kernels (threshold.cc: 7 diff lines, line_fit_filter: 10, labeling: 6) with the NEON path removed from `Detect`, CHECKs turned into `absl::Status`, and `min_white_black_diff = 4` in `src/localization/gpu_apriltag_detector.cc`.
- `git diff ca912dd75 HEAD` on the detector files: **the GPU kernels are essentially unchanged.** `labeling_allegretti_2019_BKE.cc` 14 changed lines (size_t -> uint32), `line_fit_filter.cc` 27 lines (drop `__host__`, namespace), `points.h` 10 lines (namespace). All real change is in `apriltag.cc` (host orchestration), `threshold.cc` (front end) and `cuda.h` (memory classes).

**SpectrumJetson confirmation (web, README of https://github.com/Spectrum3847/SpectrumJetson, fetched 2026-09-26):** "Old detector: FRC-Team-4143's copy (GpuDetectorJNI), which dates from about August 2024"; "New detector: built from frc971/bos". "the new detector found tags exactly as well as the old one, and 30–40% faster (1.7 ms per frame instead of 2.4–3.0 ms)"; "The GPU finds the tags in about 2 ms per frame and is only about 12% busy (peaks under 25%)"; 2 cameras x 120 fps at 1280x800, ~15 ms latency; limits were exposure and PhotonVision frame read/decode. [verified from primary source]. Combined with the diff: the 0.7–1.3 ms gained comes from host orchestration (971-3, 971-5, 971-7) and 32-bit/fp32 arithmetic (971-4), NOT from a new kernel algorithm (971-8), and not from min_white_black_diff. bos uses 4, and the old 4143 JNI hard-codes `tag_detector->qtp.min_white_black_diff = 5` in `GpuDetectorJNI.cc:60` (checked in the cached clone), so neither path passes a PhotonVision value through.

Relevant commits (all Austin Schuh unless noted, `frc/orin`):
`86f0ac3f4` 2025-01-31 32-bit types (Kevin Jaget / team 900) · `7d9118bd1` 2025-02-03 bypass gray->gray · `6b083c84a` 2025-02-01 selectable image type · `c30ec48c6` 2025-02-09 detect from unified memory · `8e7d67437` 2025-02-09 async memcpys + explicit sync (Kevin Jaget) · `551e9edd8` 2025-02-16 UnifiedMemory class · `922568c79` 2025-02-17 NEON threshold · `421551b09`/`76d8f2165` 2025-02-22 "Speed up tag detection by probably about 2x" · `8736ba62a` 2026-03-30 "Remove extra host memory copy".

### 971-1. Raise `min_white_black_diff` 5 -> 20 ("the gray rejection is a massive speedup") — 971 / aos commit 76d8f2165, 2025-02-22
- URL: https://github.com/RealtimeRoboticsGroup/aos (commit 76d8f21653d1e15f94882514e1eddd61b246e959; `frc/vision/single_node_camera_config.json`, `frc/orin/gpu_apriltag.cc`)
- Technique: Not an algorithm change. The commit makes `min_white_black_diff` and `min_cluster_pixels` flags and sets `--min_white_black_diff=20` for all four Orin detectors (default stays 5). A larger value marks more 4x4 blocks as 127 (ambiguous), which removes them from union-find merging, boundary extraction, sorting and fitting — so every downstream stage shrinks. Commit message: "Enable the NEON threshold operation, and reject more pixels as gray. The gray rejection is a massive speedup." The title claims "probably about 2x" for the combination with NEON (971-2).
- Reported speed-up: "probably about 2x" (author's estimate, no numbers in the commit), Jetson Orin, 4 cameras, MONO8. Not separated between the two changes.
- Accuracy / determinism impact: Deterministic. Still bit-identical to upstream libapriltag *run with the same qtp.min_white_black_diff*, since upstream has the same parameter. It is a recall trade-off: low-contrast / dim / motion-blurred tags lose edge blocks. frc971/bos (the 2025-26 robot code) actually runs `min_white_black_diff = 4`, so the bos deployment does NOT get this win.
- Applicability to vkapriltag: `GpuDetector.h:48` default `min_white_black_diff = 5`. Already a config field (pushed as `thresh_pc.min_diff`). Action: expose it through PhotonVision (PV's AprilTag pipeline does not surface this knob as far as the cached docs show — to verify) and measure GPU-phase time vs. value on the Mali corpus. Expected to help both Mali and RDNA most in labelling/blob_diff/extents/sort, which scale with the number of non-127 pixels. Cheap to bound: sweep 5/10/15/20 on the existing benchmark corpus, record per-span times + detection recall.
- Verification: [verified from primary source] (commit diff read; the 2x figure is the author's own "probably").

### 971-2. Threshold + decimate moved to the CPU with NEON (GPU "too full") — aos commit 922568c79, 2025-02-17
- URL: https://github.com/RealtimeRoboticsGroup/aos (`frc/orin/neon_threshold.cc`, 340 lines)
- Technique: Two-pass NEON implementation of decimate(2) + 4x4 (decimated) block min/max + 3x3 block filter + threshold. Pass 1 loads 16 bytes from each of 4 even source rows (vertical decimation by loading every other row), `vuzp1_u8` to take even bytes (horizontal decimation), writes the decimated rows, and keeps a running vertical min/max; horizontal 3-block filter is done with a one-block-delayed scalar rolling window and the min/max stored packed as `uint16 (min | max<<8)` per block. Pass 2 does the vertical 3-block filter reading three packed rows and thresholds 2 blocks x 4 rows at a time with `vcgt_u8` after `vuzp1/2_u32` shuffles; uniform blocks are written as `vdup_n_u8(127)`. Outputs go straight into CUDA unified memory ("so it gets moved over to the GPU in the background"). Only MONO8/YUYV.
- Reported speed-up: commit message: "The GPU was too full, this moves it back to the CPU" and "This gives us a 3x speedup over the original aprilrobotics CPU code." No absolute ms in the commit (VLOG prints per pass).
- Accuracy / determinism impact: Intended bit-identical (there is `threshold_test.cc` comparing it against the CUDA path). Note `vcgt_u8` gives 0xFF/0x00 which matches the 255/0 encoding.
- Applicability to vkapriltag: The motivation (GPU saturated by 4 cameras on one Orin) is a multi-camera throughput argument, not latency. For vkapriltag on RK3588 the threshold span is small relative to labelling, and Mali is the bottleneck while 4 A76 cores idle during the GPU phase — so moving decimate+threshold to NEON on the A76s *could* shorten the GPU phase by the threshold span and overlap it with the previous frame's GPU work (frame pipelining already exists). Cost: the CPU tail (~1.2 ms) competes for the same big cores; with unified memory there is no copy. Bound cheaply: take the existing Mali per-span timings for decimate/block_minmax/block_filter/threshold (the "threshold" span) — that is the maximum saving; time the 971 NEON code as-is on an A76 at 1280x800. On RDNA it is not worth it: the RX 9060 XT runs this stage far faster than a desktop CPU would, even though the upload would shrink 4x (thresholded image is W/2 x H/2). Mali only.
- Verification: [verified from primary source] (source read; 3x claim is the commit author's, unmeasured here).

### 971-3. Decimate kernel iterates over OUTPUT pixels and no longer writes a full-res gray image — aos commits 7d9118bd1 / 6b083c84a (Feb 2025), threshold.cc
- URL: https://github.com/RealtimeRoboticsGroup/aos `frc/orin/threshold.cc`
- Technique: The Aug-2024 `InternalCudaToGreyscaleAndDecimateHalide` walked every full-resolution pixel (grid-stride over W*H), wrote the full gray image AND the decimated image, then the host did `gray_image_device_.MemcpyAsyncTo(&gray_image_host_)` (full-res D2H copy) for decode. The new `InternalCudaToGreyscaleAndDecimate<FORMAT>` is templated on input format, launches over (W/2)*(H/2) outputs and reads only the sampled pixel. For MONO8 input the gray conversion and the D2H gray copy are skipped entirely (`gray_image_host_ptr_ = image`). In the 4143 copy, `DetectGrayHost` still copied the gray image H2D *and back* D2H every frame.
- Reported speed-up: none quoted. Bandwidth: removes 1 full-res read, 1 full-res write and 1 full-res D2H copy (~1 MB each at 1280x800) per frame.
- Accuracy / determinism impact: none (same sampled pixels).
- Applicability to vkapriltag: **Already has it.** `decimate.comp` is dispatched 2D over the decimated size, reads one packed word per output, and the host keeps its own gray frame for decode (no gray readback). Upload is a single memcpy of the 8-bit frame (or direct write into a device-local-mapped buffer on unified memory).
- Verification: [verified from primary source]

### 971-4. 32-bit index/size types everywhere + float instead of double for blob centroid / dot product — aos 86f0ac3f4 (Kevin Jaget, team 900), 2025-01-31
- URL: https://github.com/RealtimeRoboticsGroup/aos commit 86f0ac3f4; `frc/orin/apriltag_types.h`, `line_fit_filter.h`
- Technique: `size_t`/`ssize_t` -> `apriltag_size_t` (uint32) in all kernels (BlobDiff write addresses, threshold indexing, labeling counts, SelectBlobs). `MinMaxExtents::cx()/cy()/dot()` switched from `double` to `float` math (Orin GPU fp64 is 1/64 rate). Also `__host__` removed from several line-fit methods. Commit: "Kevin says this is faster ... I'm not surprised it helps."
- Reported speed-up: not quantified.
- Accuracy / determinism impact: the double->float change in `dot()`/`cx()` alters the rounding of the reversed-border test and theta sort key, so 971 is not bit-exact to upstream here (971 never claimed that).
- Applicability to vkapriltag: **Already has it** — shaders are uint32 throughout; `select_blobs.comp` states it deliberately uses no GLSL `double`; the extents path uses int64 atomics only for exact integer sums. Nothing to port.
- Verification: [verified from primary source]

### 971-5. Host-sync hygiene: pinned async count readbacks, event sync, scan moved off the legacy default stream, memset on a side stream — aos 8e7d67437 (2025-02-09) + 421551b09
- URL: https://github.com/RealtimeRoboticsGroup/aos commit 8e7d67437 ("Make apriltag memcopies async and use explicit synchronization ... helps group CUDA kernel calls and reduce context switches ... While we are here, move a lone kernel onto the stream too.")
- Technique: In the Aug-2024 code two of the six per-frame count readbacks (`num_compressed_union_marker_pair`, `num_quads`) used synchronous `cudaMemcpy` into pageable stack variables, and `cub::DeviceScan::InclusiveScan` was launched **without a stream argument**, i.e. on the legacy default stream, which implicitly synchronises with every other blocking stream. The new code reads all counts via `cudaMemcpyAsync` into `cudaMallocHost` pinned buffers, records an event and `Synchronize()`s only right before the value is needed on the host (to size the next CUB call), and puts the scan on `stream_`. The `union_markers_size` memset moved to a separate `memset_stream_` so it overlaps the threshold. The host still round-trips ~6 times per frame (compact count, num_quads, num_selected_blobs, num_peaks, num_peaked_quads, quad copy) because CUB needs host-side sizes.
- Reported speed-up: not quantified in the commit.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: **Already has a stronger version.** vkapriltag sizes dispatches on device (`build_indirect_args.comp`, `DispatchIndirect`), fuses the tail into one submission (`fused_submits_`), and the only remaining host round trip is the union-find convergence flag, which is already handled speculatively (tail recorded optimistically, re-run only if not converged) and seeded with last frame's iteration count. The `FillZero`s are already in the same command buffer. Nothing further to port; this item, together with 971-3 and the `8736ba62a` bug below, is the likely explanation of most of the "new 971 is 30-40% faster than the Aug-2024 4143 copy" observation on a Jetson where the GPU was only ~12% busy — the old copy was host-sync/latency bound, not kernel bound.
- Verification: [verified from primary source]

### 971-6. Regression and fix: per-frame debug readback of the min/max image in the GPU threshold path — aos 8736ba62a, 2026-03-30 ("Remove extra host memory copy. Adam found this while trying to optimize.")
- URL: https://github.com/RealtimeRoboticsGroup/aos commit 8736ba62ae8b007a03f02e35239d6db1f76a5375
- Technique: From Feb 2025 to Mar 2026, `TypedThreshold::ThresholdAndDecimate` allocated a `HostMemory` (cudaMallocHost) and did a synchronous `MemcpyTo` of the whole min/max block image every frame, followed by a debug-print loop. This only affected the GPU threshold path (971 production ran `--use_neon`, so they didn't see it). The fix deletes it.
- Reported speed-up: not quantified. A per-frame cudaMallocHost + synchronous copy is typically ~0.1–1 ms on Orin.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: Not applicable (no such code). Relevant as a caveat: anyone benchmarking a 971 checkout from that window with the GPU threshold path (e.g. frc971/bos, which removed the NEON branch and vendored in 2025-08) got a slower-than-necessary GPU front end. Checked: frc971/bos first vendored the **4143 copy** (file names `971apriltag.cu`, `threshold.cu`) on 2025-08-31 and replaced it with the aos-HEAD version (post-8736ba62a, no debug copy) in PR #139 "update autin tag", 2026-04-03. So bos itself is an "Aug-2024 copy -> 2026 aos" upgrade, which is probably the comparison behind the 30-40% observation.
- Verification: [verified from primary source]

### 971-7. Unified (managed) memory for camera frames — aos c30ec48c6 / 551e9edd8 / 426aa690a, Feb 2025
- URL: https://github.com/RealtimeRoboticsGroup/aos `frc/orin/cuda.h` (`UnifiedMemory<T>` = `cudaMallocManaged`), `frc/vision/cuda_camera_image_callback.cc`
- Technique: `Detect(image, image_device)` accepts an image already in GPU-accessible memory, skipping the H2D `MemcpyAsyncFrom`; camera callback produces frames in managed memory. On Jetson (unified DRAM) this removes one full-res copy.
- Reported speed-up: not quantified.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: **Mostly already has it** — `gray_direct_write_` writes the frame straight into a `DeviceLocalMapped` buffer on unified-memory devices (one memcpy). The remaining step (zero copy: import the camera/decoder dma-buf via VK_EXT_external_memory_dma_buf) is covered in the survey (B1/B3) and belongs to the PhotonVision layer. Mali only.
- Verification: [verified from primary source]

### 971-8. What did NOT change (negative result) — 971 CUDA kernels ca912dd75 -> HEAD
- URL: as above.
- Technique: The BKE labelling (Allegretti 2019), BlobDiff (shared-memory tiled boundary extraction), CUB radix sort of `QuadBoundaryPoint` keys, CUB ReduceByKey extents, DeviceSelect filtering, InclusiveScanByKey line-fit moments, GPU line-fit error filter / peak NMS / FitQuads — all unchanged apart from integer widths. No new stages, no new CUB primitives, no CUDA graphs, no multi-stream kernel concurrency beyond the memset.
- Reported speed-up: n/a
- Accuracy / determinism impact: n/a
- Applicability to vkapriltag: The 30-40% figure should NOT be read as evidence of a better kernel algorithm to port. The only structural 971-vs-vkapriltag differences that remain are ones vkapriltag already chose differently: (a) 971 does peak NMS + quad fitting on the GPU (`FitQuads`), vkapriltag does quad fit on the CPU and overlaps it with the next frame; (b) 971 uses a global radix sort + reduce-by-key, vkapriltag uses hashing + per-blob local sort; (c) 971 uses BKE on the 3-valued image (vkapriltag's notes reject BKE for needing binary 8-connected input). If anything is worth revisiting it is (a): moving quad fit (and its CPU tail) to the GPU on RDNA where the GPU phase is 0.66 ms — see Conclusions.
- Verification: [verified from primary source]

---

## Part 2 — NVIDIA VPI / Isaac ROS

### VPI 4.1 AprilTag detector: CPU and PVA backends only, measured tables — NVIDIA, docs generated 2026-06-03
- URL: https://docs.nvidia.com/vpi/algo_apriltags.html (cached `fiducial/vpi_apriltags.html`), perf data `perf_tegra234_apriltag.json` (Jetson AGX Orin), `perf_tegra264_apriltag.json` (Jetson AGX Thor). The four `perf_{orin,agx_orin,thor,agx_thor}_apriltag.json` files in the cache are identical NVIDIA "Page Not Found" HTML — useless.
- Technique: VPI splits the detector into decimation(2) -> adaptive threshold -> CCL -> gradient cluster generation -> quad fitting -> decoding. The first four run on the **PVA** (Programmable Vision Accelerator, a VLIW/SIMD DSP) or CPU; **quad fitting and decoding are always CPU**. There is **no CUDA backend** for AprilTag in VPI. Pose estimation is CPU only (SVD + fixed-iteration reprojection refinement, picks the better of 2 candidates). Tables are "6 tags", s1/s2/s4/s8 = number of parallel VPI streams.
- Reported speed-up (detect, 6 tags, 1 stream / 4 streams):

  | Device | Res | CPU s1 | CPU s4 | PVA s1 | PVA s4 |
  |---|---|---|---|---|---|
  | AGX Orin | 1920x1080 | 4.46 ms | 11.4 ms | 8.9 ms | 21.48 ms |
  | AGX Orin | 1280x720 | 2.75 ms | 5.8 ms | 4.5 ms | 9.916 ms |
  | AGX Orin | 960x540 | 1.92 ms | 3.63 ms | 2.59 ms | 5.84 ms |
  | AGX Thor | 1920x1080 | 2.66 ms | 6.8 ms | 8.63 ms | 23.65 ms |
  | AGX Thor | 1280x720 | 1.71 ms | 3.4 ms | 4.30 ms | 10.90 ms |
  | AGX Thor | 960x540 | 1.25 ms | 2.2 ms | 2.56 ms | 6.417 ms |

  Pose estimate (CPU, Orin): 1 tag 0.075 ms, 6 tags 0.110 ms, 64 tags 0.48 ms.
- Accuracy / determinism impact: VPI says it "may be used as a drop-in replacement for the reference detection algorithm"; no bit-exactness claim.
- Applicability to vkapriltag: Two useful data points. (1) NVIDIA's own **multithreaded CPU** path (12-core A78AE Orin, 14-core Neoverse V3AE Thor) beats its DSP offload at every size — the offload of front-end stages to an accelerator does not pay for itself when quad fit + decode stays on the CPU and data must move. That mirrors vkapriltag's Mali situation (GPU phase 3.76 ms vs. CPU tail 1.2 ms). (2) A reference point for Orange Pi 5: vkapriltag at 1280x800 d2 = 3.76 ms GPU + ~1.2 ms CPU tail (pipelined) vs VPI-CPU on AGX Orin 2.75 ms at 1280x720 on a much bigger CPU. No technique to port; the benchmark numbers are useful for the PhotonVision pitch. The RK3588 NPU is analogous to the PVA; this table is weak evidence against offloading threshold/CCL to it.
- Verification: [verified from primary source] (numbers copied from the cached NVIDIA JSON; settings behind "6 tags" — decimation, family, hamming — not stated in the JSON).

---

## Part 3 — Other implementations

### rapidtag: what the "leaner pipeline" actually is — Sujith Christopher, v0.1.x, Jul–Sep 2026
- URL: https://github.com/SujithChristopher/rapidtag (cached `fiducial/cv/rapidtag`, HEAD 4dfcafc 2026-09-25; read `src/detector.rs`, `src/imgproc.rs`, `src/contours.rs`, `src/affinity.rs`, README)
- Technique: rapidtag is a pure-Rust port of **OpenCV's ArUco detector** (`cv::aruco::ArucoDetector::detectMarkers`, CORNER_REFINE_NONE) used with DICT_APRILTAG_36h11 — i.e. adaptive-mean threshold at 3 window scales + Suzuki-Abe contour tracing + approxPolyDP, not the AprilTag 3 union-find/line-fit pipeline. The "leaner pipeline" is: (1) adaptive threshold **fused with the contour tracer's input**: the threshold writes the foreground mask directly into a 1-px zero-padded `i8` label buffer (no 0/255 image, no re-scan); (2) the box mean uses a one-row running column sum + 1D prefix (working set a few KB, not a multi-MB integral image), integer compare `2*sum + area < 2*area*(src+C)`, branchless vectorisable interior loop; (3) contour labels collapsed to an `i8` alphabet {0,FG,POS,NEG} because hierarchy is unused — "shrinks this buffer 4x (the dominant cost is streaming it, not the tracing)"; padded buffer removes bounds checks; integer direction indices; (4) trace + size filter fused so tiny noise contours are rejected before any allocation; (5) the 3 threshold scales run on 3 cores via rayon, batch mode flattens (frame x scale) parallelism; (6) big.LITTLE auto-pinning of the rayon pool to the fast cores, plus a `performance` governor recommendation.
- Reported speed-up: README, 1280x800 mono 36h11 dual-camera data: 1.57x vs OpenCV single camera realtime, up to ~3.4x batch. On Radxa Dragon Q6A (QCS6490, 4xA55 + 4xA78): pinning 68 -> 211 fps single 1280x800 detect; `performance` governor 123 -> 211 fps (~1.7x). Corners match OpenCV to 0.0000 px.
- Accuracy / determinism impact: bit-exact to OpenCV's ArUco output (claimed and checked with `tests/crosscheck.py`); not the same algorithm as libapriltag.
- Applicability to vkapriltag: The algorithmic tricks do not transfer (different detector). Two things do: (a) **core pinning + governor** for the CPU tail on RK3588 (4xA76 + 4xA55) — survey item B4 already covers pinning; the governor finding is new evidence: bursty per-frame work never keeps `schedutil` clocked up. vkapriltag's CPU tail is ~1.2 ms bursts + the Vulkan submit thread, exactly this pattern; also relevant to the **Mali GPU devfreq governor** (`/sys/class/devfreq/fb000000.gpu/governor`), where the same bursty-load problem applies. Bound cheaply: run the benchmark with `performance` governor on both CPU policy6/policy4 and GPU devfreq vs. default. (b) the "fuse the producer into the consumer's padded layout" idea is the same one vkapriltag rejected for preprocessing fusion; no new evidence.
- Verification: [verified from primary source] (source + README read; speed numbers are the author's).

### halide-apriltag: Halide JIT threshold for libapriltag — Eeshwar Krishnan, 2025-10-21
- URL: https://github.com/Eeshwar-Krishnan/halide-apriltag (cached `fiducial/cv/halide-apriltag`, single commit b5a5231)
- Technique: Replaces libapriltag's `threshold()` with a Halide pipeline: 4x4 tile min/max (RDom), 3x3 neighbour min/max, per-pixel select 127/255/0; scheduled `tile(64,32).parallel(yo).vectorize(xi,16)`, intermediates `compute_root` + vectorised 16. `apriltag_timing.cpp` compares baseline vs Halide detect time. Comment: "GPU scheduling can be layered on later".
- Reported speed-up: none published in the repo (the timing tool prints numbers but no results are committed).
- Accuracy / determinism impact: Edge handling uses clamp/`repeat_edge` and maps pixels beyond the last full tile to the last tile; whether this is bit-identical to upstream's partial-tile handling is not verified.
- Applicability to vkapriltag: Low. vkapriltag's threshold is already 3 GPU passes; Halide is interesting only as a way to generate a NEON CPU threshold (see 971-2) with auto-scheduling. aos also uses Halide (`rules_halide`, `resize_generator.cc`) for YOLO preprocessing, not for AprilTag.
- Verification: [verified from primary source] (no numbers to verify).

### OpenCV objdetect ArUco (4.7+ `ArucoDetector`) and its AprilTag quad-threshold port — OpenCV, source read from cache
- URL: https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/aruco_detector.cpp and `.../apriltag/apriltag_quad_thresh.cpp` (cached `fiducial/cv/aruco_detector.cpp`, `aq.cpp`)
- Technique: `detectMarkers` runs the adaptive-threshold scales with `parallel_for_(Range(0, nScales))`. With `useAruco3Detection` it computes `fxfy = minSideLengthCanonicalImg / (minSideLengthCanonicalImg + max(W,H)*minMarkerLengthRatioOriginalImg)`, resizes the image by that factor for segmentation, builds a pyramid and does corner refinement (forced to CORNER_REFINE_SUBPIX) on the closest pyramid level. `aq.cpp` is OpenCV's port of the AprilTag 2 quad-threshold (used for CORNER_REFINE_APRILTAG): single-threaded, zarray-based, chained-hash cluster map (with a "XXX lousy hash function" comment), unionfind. No acceleration beyond upstream.
- Reported speed-up: none in the source. The ArUco3 paper figures (up to ~40x with multi-scale + temporal) are already noted in the brief. No new OpenCV 4.7+ benchmark numbers in the cache (see web entries below).
- Accuracy / determinism impact: Aruco3 downscaling changes detections (min marker size relative to image), so it is not bit-identical to anything upstream-AprilTag.
- Applicability to vkapriltag: Nothing to port for the bit-exact path. The only transferable idea is the ArUco3 *temporal* scale choice (use the previous frame's smallest tag to pick decimation), which breaks the libapriltag parity gate unless offered as an opt-in mode.
- Verification: [verified from primary source] (source only; no speed numbers)

### Frappe: fiducial detection on the Raspberry Pi Zero VideoCore IV QPU + VPU — Jones & Hauert, J. Real-Time Image Processing 20:119, Oct 2023
- URL: https://doi.org/10.1007/s11554-023-01373-w (cached `fiducial/fpga/frappe.txt`)
- Technique: An ArUco-compatible detector split across the RPi Zero's units. The 12 QPUs (the GL ES 2 GPU's shader cores, 16-lane SIMD) run two passes over 64x32 tiles: pass 1 computes intensity, gradients and discretised angles; pass 2 does Canny edge thinning and Shi-Tomasi corners. The VPU (dual-core 16-lane vector DSP) runs **FIND_EMPTY_TILES**: it scans 16x16 blocks for any edge pixel and builds a mask, at about 1 ms per frame — "on a typical image we save more than 5 ms in contour tracing". The ARM traces contours only on non-empty tiles, then does quads and decoding. ADAPTIVE_SCALE (ArUco3-style) picks the scale from the smallest fiducial seen in the previous frame. ARM, VPU and QPU share zero-copy VCSM buffers.
- Reported speed-up: 640x480 at >60 Hz on a Pi Zero, "five times faster than the standard ArUco library". The slowest Frappe mode is 4.8x faster than the fastest ArUco mode. 4 cameras x 640x480 at 30 Hz, never over 33 ms/frame.
- Accuracy / determinism impact: a different algorithm (Canny edges) from both ArUco and AprilTag. Detection is comparable; slightly worse than ArUco on one sequence.
- Applicability to vkapriltag: The **empty-tile mask** is the idea that transfers. `threshold` can produce a per-tile "has any non-127 pixel" bit at almost no cost, since it already knows each 4x4 block's max-min against `min_white_black_diff`. That bit can then either (a) let whole workgroups exit early in `uf_merge`/`uf_compress`/`blob_diff`, or (b) drive an indirect dispatch over non-empty tiles only. On Mali, dispatch and barrier cost scale with in-flight work, so skipping uniform tiles saves work in proportion to the 127 area. That area is large on typical FRC frames, and larger still with a higher min_white_black_diff (see 971-1). Bit-identical: skipped pixels are 127 and contribute nothing anyway. Cheap bound: log the fraction of 127 4x4 blocks and of all-127 16x16 tiles on the corpus. If more than ~50% of tiles are uniform, prototype a one-load workgroup early-out in uf_merge and blob_diff. Check first whether the current shaders already exit per-invocation on 127; a per-thread early-out still pays the per-workgroup launch/scheduling cost.
- Verification: [verified from primary source] (paper text read)

### FPGA AprilTag 3 front end on Zynq with HLS (TPSS-CCL) — Marcel Flottmann, bachelor thesis, Hochschule Osnabrück / DLR, Aug 2019
- URL: cached `fiducial/fpga/flottmann.txt` (no public URL recorded)
- Technique: Zynq-7000 SoC. Decimate, the 4x4 min/max + 3x3 filter threshold (2-bit segment image over AXI video stream) and a streaming union-find CCL run in HLS IP blocks at 150 MHz. The CCL is "TPSS-CCL": the first pass writes a provisional label equal to the position of the component's first pixel, merges point at the smallest start index, and the output is a 22-bit label mask. Gradient clusters, quad fit and decode stay on the ARM. The FPGA works on the next frame while the CPU finishes the current one.
- Reported speed-up: CPU-only per-stage baseline on test image 1: decimate 4.95 ms, threshold 6.27 ms, unionfind 9.41 ms, make clusters 7.37 ms, fit quads 1.42 ms, decode 0.70 ms (30.17 ms total). On a busy image (41): clusters 33.8 ms, fit quads 99.85 ms (167 ms total). FPGA version: **71 fps vs 37 fps CPU (~2x)** on simple images. On busy images both collapse (13 vs 6 fps) because the CPU tail dominates.
- Accuracy / determinism impact: the thesis reports differing accuracy on some images vs. the CPU version; not bit-exact.
- Applicability to vkapriltag: Confirms the general shape: accelerating decimate/threshold/CCL alone gives ~2x, until CPU clustering and quad fit dominate on cluttered frames. For vkapriltag this supports keeping clustering and sort on the GPU (already done). It also suggests measuring the CPU tail on cluttered frames, since the 1.2 ms figure is probably corpus-dependent. The min-index union rule is the same as vkapriltag's atomicMin root rule. Nothing new to port.
- Verification: [verified from primary source]

### FPGA-accelerated AprilTag on UAVs (AprilTag 1 gradient pipeline) — Ethan Tola (RIT MS thesis, Mar 2021) and Raymond Zhang (RIT MS paper, May 2018)
- URL: cached `fiducial/fpga/tola.txt`, `zhang.txt` (RIT, advisor D. Kaputa)
- Technique: Both accelerate the *original* AprilTag (1) pipeline. Gaussian blur, gradient magnitude/direction and edge extraction run on the FPGA, the rest in software. Zhang uses a Snickerdoodle Zynq with HDL IP cores. Tola uses fixed-point HDL, Kirsch filters and CORDIC, plus an FPGA connected-component variant ("Ravven Tag CCA").
- Reported speed-up: Zhang: the gradient-magnitude IP is 13.9x faster than software and 2.65x faster than the Snickerdoodle CPU; CPU per-stage baseline includes Gaussian blur 38.4 ms and gradient theta 32.0 ms. Tola: "Ravven Tag CCA" 27.0 ms total, 37 fps peak / 30 fps reliable at 752x480, 19.26x vs "April Tag Baseline" (AprilTag 1 C code).
- Accuracy / determinism impact: fixed-point quantisation changed the edge sets noticeably. Tola's quantised MagTheta variant ended up slower because it produced many more edges, until a threshold was retuned (1225.93 -> 220.19 ms for those stages).
- Applicability to vkapriltag: Not applicable. AprilTag 2/3 replaced this gradient pipeline with the threshold/union-find one vkapriltag uses. The one reminder: quantisation that changes the edge set can make later stages slower (the reverse of 971-1).
- Verification: [verified from primary source]

### Single-scan run-length contour detection for markers on FPGA — Sufi, Rincón, Barba, López (U. Rajshahi / UCLM), ICVIP (year not in the cached text)
- URL: cached `fiducial/fpga/icvip.txt`
- Technique: Streaming single-pass, run-length based contour extraction for ArUco-style markers on a Zedboard (Zynq-7000). It handles single-pixel critical cases in the binarised image, and has a pre-process core (binarisation) and a contour core.
- Reported speed-up: "about 35-70 fps for 640x480".
- Accuracy / determinism impact: n/a (different detector).
- Applicability to vkapriltag: The run-length idea is what vkapriltag's `uf_init` already does (it pre-joins horizontal runs). Nothing new.
- Verification: [verified from primary source]

### Samarin et al., "Fiducial Marker Detection Using FPGAs" — UNB TR-13-227, Sep 2013
- URL: cached `fiducial/fpga/samarin.txt`
- Technique: The FPGA only binarises the camera image and ships the binary image to a PC, which detects markers and computes pose.
- Reported speed-up: not extracted (old, minimal offload).
- Accuracy / determinism impact: n/a
- Applicability to vkapriltag: none.
- Verification: [verified from primary source] (abstract only)

### Upstream AprilRobotics/apriltag performance commits, 2022–2026 (git log of cached clone, HEAD b7c0ebe 2026-08-07)
- URL: https://github.com/AprilRobotics/apriltag (cached `fiducial/apriltag`). vkapriltag pins **v3.4.5 (94be783, 2025-08-29)** in `apriltags_vulkan/library/CMakeLists.txt`. 57 commits have landed since.
- Technique (commit, date, author: effect):
  - `870f228` 2022-12-29 Austin Schuh: split `gradient_clusters` work more finely than the core count so fast cores steal from slow ones on big.LITTLE. "This saved 5-10ms of processing time on my Rock Pi 4b and reduced the outliers."
  - `e1b143c` 2023-06-15 Bouke van der Bijl: batch range-copy in `merge_clusters` instead of per-element `zarray_add_all`. "reduces the total runtime for an image that's 2000x3000 by 20%".
  - `08465f4` 2023-10-26 Bouke: unionfind split into parent/size arrays, memset init. `fbc4cf2` 2023-10-26: parallelise the threshold steps ("Speeds up detection on large images").
  - `2295d19` 2023-12-28 Austin Schuh: skip unused lineparam calculation in `quad_segment_maxima`. Fit quads 8.896 -> 8.429 ms on his test box.
  - `c598bd8` 2023-12-29 Austin Schuh: **stop adding duplicate points to clusters**. Cheaper, and it fixes a sort-order dependence where duplicates with equal slope ended up non-adjacent and were never de-duplicated. The 971 GPU code comments on exactly this ("Aprilrobotics has a *3 instead of a *2 here since they have duplicated points"). Output-affecting; already in v3.4.5.
  - `717dab5` 2025-10-07 Joris van Vugt: **path halving in unionfind** (same idea vkapriltag measured at -17% labelling at d1 on RDNA). `f10450e`: stack buffer for small `ptsort` arrays (no malloc). `ad4a17f`: accumulate line-fit stats in locals, not a struct via memcpy.
  - `7d18fc5` 2026-01-09 Sidd: **pigeonhole quick_decode**. The code is split into 4 chunks of ceil(nbits/4) bits; any codeword within Hamming <=3 must match one chunk exactly. So a 4 x 2^chunk bucket index (prefix-summed offsets + id lists) plus `popcount64` verification replaces the old open-addressing table of every code at every Hamming variant. It rejects `maxhamming > 3`.
  - `02674c1` 2026-02-28 Gold856: `min_cluster_pixels` floored at 24 in place of a redundant second check. `ee70301` 2026-03-21 iabdalkader (OpenMV): interleave unionfind parent+size in one struct for cache locality. `8a5657a` 2026-04-21: relaxed atomics in `unionfind_get_representative` (TSan-clean path halving under threads).
  - `d686c00` 2026-05-07 Bouke: **remove `sched_yield()` from workerpool task pickup** — "Seems not needed and has a big performance impact."
- Reported speed-up: as quoted above; no consolidated benchmark.
- Accuracy / determinism impact: All performance-only except `c598bd8` (in v3.4.5) and `6db6448` 2026-02-13 (colocated-corner divide-by-zero guard; can change output only in degenerate quads). Pigeonhole decode returns the first codeword within maxhamming, so it is identical to the old table whenever the family's minimum distance exceeds 2*maxhamming (true for 36h11 at hamming <=2).
- Applicability to vkapriltag:
  - (a) **CPU tail.** The v3.4.5 `workerpool.c` still has `sched_yield()` at line 84 of the fetched source. vkapriltag uses its own `WorkerPool` for quad decode and pose, so the tail itself is unaffected. Any path that still calls upstream `workerpool_run` (e.g. the CPU reference detector used for the parity gate and benchmarks, or PhotonVision's CPU baseline) inherits the yield. Bumping the pin past `d686c00` (or cherry-picking it) makes the CPU baseline faster and more honest to compare against.
  - (b) **Pigeonhole decode**: vkapriltag's CPU tail DOES call upstream `quad_decode_index` -> `quick_decode_codeword` (TagDecoder.cpp:125), so a post-v3.4.5 pin also changes detector-creation time and memory (the 36h11 hamming-2 table is large). Per-lookup cost on A76 still needs measuring.
  - (c) GPU-side ideas are already in-tree: path halving (found), min-index roots, run pre-joining.
  - Bumping the pin requires re-running the bit-exact gate. `02674c1` and `6db6448` should be output-neutral on normal data. Both devices (CPU-side).
- Verification: [verified from primary source] (commit messages and diffs read; the sched_yield impact is unquantified upstream)

### WPILib apriltag fork + PhotonVision's hard-coded detector parameters — WPILib (allwpilib, via aos third_party), PhotonVision main
- URL: allwpilib `upstream_utils/apriltag.py` (pins upstream commit `3806edf38ac4400153677e510c9f9dcb81f472c8`) + `upstream_utils/apriltag_patches/0001..0008` (cached inside `RealtimeRoboticsGroup_aos/third_party/allwpilib`). PhotonVision `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java` (fetched raw from GitHub main 2026-09-26, saved as `fiducial/pv_AprilTagPipeline.java`).
- Technique: WPILib's fork has 8 patches. The only performance-relevant one is **0003 "Make orthogonal_iteration() exit early upon convergence"** (Tyler Veness / Matt Morley, 2023-01): `orthogonal_iteration` gains `min_improvement_per_iteration` and breaks when `|error - prev_error|` falls below it ("The current approach wastes iterations doing no work. Exiting early can give lower latencies and higher FPS."). The rest are warning/portability fixes. WPILib's `QuadThresholdParameters` defaults are **minClusterPixels = 300** (changed in 2025), maxNumMaxima 10, **criticalAngle 45°**, maxLineFitMSE 10, minWhiteBlackDiff 5. **PhotonVision overrides these explicitly**: `minClusterPixels = 5` ("5 was the default minClusterPixels in WPILib prior to 2025; increasing it causes detection problems when decimate > 1"), `criticalAngle = 45°`, `maxLineFitMSE = 10`, `minWhiteBlackDiff = 5`, `deglitch = false`. It also exposes threads, refineEdges, quadSigma (blur) and decimate as user settings.
- Reported speed-up: no numbers in patch 0003.
- Accuracy / determinism impact: 0003 changes pose output slightly (fewer iterations); the detections themselves are unaffected.
- Applicability to vkapriltag:
  - (1) **Parameter parity with the actual deployment.** vkapriltag's defaults are `min_cluster_pixels = 24` and `cos_critical_rad = 0.98` (~cos 11°, upstream's default). PhotonVision runs `criticalAngle = 45°` (cos = 0.707), which rejects many more candidate quads in the CPU quad-fit tail. So the PhotonVision integration must pass 45°, and the parity gate and benchmarks should also be run at PV's parameters. The CPU-tail number under PV settings is probably lower than the 1.2 ms measured at upstream defaults (to measure). `min_white_black_diff` is hard-coded to 5 in PV, so the 971-1 speedup would need a new PV setting.
  - (2) Pose: **already has it** — `PoseEstimator.cpp` implements orthogonal iteration with a convergence tolerance (`kDefaultConvergenceTol`, step counted at convergence), i.e. the WPILib 0003 idea.
  - Both devices (CPU side).
- Verification: [verified from primary source]

### Isaac ROS AprilTag (cuAprilTags, closed-source CUDA) benchmark tables — NVIDIA isaac_ros_benchmark, updated 2026-09-21 (Isaac ROS 5.0.0)
- URL: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark (README tables; fetched 2026-09-26), https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag
- Technique: `isaac_ros_apriltag` wraps NVIDIA's closed-source cuAprilTags CUDA library behind NITROS zero-copy ROS 2 transport. The algorithm is not published. The benchmark is "AprilTag Node" (detector alone) and "AprilTag Graph" (with rectification), both at 720p.
- Reported speed-up (Node, 720p: max fps / latency at 30 Hz input): AGX Thor T5000 326 fps / 3.2 ms; AGX Thor T4000 248 / 4.3 ms; **AGX Orin 189 / 5.3 ms**; Orin Nano Super 8GB 104 / 9.6 ms; DGX Spark 555 / 1.5 ms; RTX 5090 596 / 1.0 ms; RTX 5070 591 / 1.4 ms. Graph: AGX Orin 186 fps / 6.5 ms, RTX 5090 596 / 1.7 ms.
- Accuracy / determinism impact: not documented; cuAprilTags is not claimed to be libapriltag-exact.
- Applicability to vkapriltag: These are external reference points only. Latencies include ROS 2/NITROS transport, so they upper-bound the kernel time. Comparisons: vkapriltag on the RX 9060 XT (0.66 ms GPU phase at 1280x800, bit-exact) is in the same class as cuAprilTags on an RTX 5070 (1.4 ms node latency at 720p). The 971 detector on Orin (1.7 ms at 1280x800, SpectrumJetson) is ~3x faster than Isaac ROS's 5.3 ms at 720p on AGX Orin. The ~590–596 fps ceiling on every desktop GPU suggests a host/transport limit, not a GPU limit (the same lesson as SpectrumJetson's 12% GPU busy). No technique to port.
- Verification: [verified from primary source] (README table via WebFetch; the fetch summariser reproduced the table, not the raw markdown)


### wykvictor/AprilTag-GPU — "Accelerating ... apriltag using CUDA", 2016 (negative result)
- URL: https://github.com/wykvictor/AprilTag-GPU (cloned to `fiducial/wykvictor`, HEAD 9e0554d 2016-10-09)
- Technique: This is the Kaess C++ port of AprilTag **1** (Gaussian/gradient/edge pipeline; `Edge.cc`, `Gridder.h`, `UnionFindSimple.h`). The only CUDA content is `cuda_add_library` in `src/CMakeLists.txt`, globbing `*.cc` and `*.cu`, and the repository contains **no `.cu` files** and no `cv::cuda` calls. The README only covers building with CUDA 7.5 and OpenCV-with-CUDA. GitHub issue #1 is titled "GPU not used".
- Reported speed-up: none.
- Accuracy / determinism impact: n/a.
- Applicability to vkapriltag: none. Record it so that nobody cites it as prior GPU AprilTag work.
- Verification: [verified from primary source] (clone inspected; issue #1 seen only as a search-result title)

### PhotonVision "ML-accelerated AprilTag": NPU YOLO11 ROI crop, then the CPU detector on crops — PRs #2410 (TheJohnFogarty, 2026-03-26, closed) and #2604 (spacey-sooty, 2026-09-10, open draft)
- URL: https://github.com/PhotonVision/photonvision/pull/2410, https://github.com/PhotonVision/photonvision/pull/2604 (bodies and comments read via `gh`)
- Technique: Stage 1 runs a YOLO11 detector (`apriltagV4-yolo11.rknn` on the RK3588 NPU, `.tflite` on the QCS6490/Rubik Pi 3) that outputs tag bounding boxes. Stage 2 pads each box (40 px by default), crops it, runs the WPILib AprilTag detector on the crop and maps corners and homography back by translation (plus inverse scale when "Adaptive Tag Resizing" shrinks close tags to about 200 px). Duplicate IDs are resolved by the highest decision margin. If the NPU finds nothing, it falls back to a full-frame detect (configurable). #2604 re-implements this through a decoupled `CropPipe`.
- Reported speed-up: **no numbers published** ("Test evidence ... Coming later"). A tester reports that frame rate drops and latency rises close to tags, and that it "appeared to function well at a distance". A reviewer asked whether it beats quad fitting once the NPU copy is counted.
- Accuracy / determinism impact: not bit-identical to the full-frame detector. Tags the NPU misses are lost when fallback is off, and the crop borders change the threshold tiles, and so the edges, compared with a full-frame detect.
- Applicability to vkapriltag: This is the competing direction for the PhotonVision Orange Pi 5 slot, and it attacks the same CPU-bound full-frame detect that vkapriltag replaces. Two implications. (1) The NPU and vkapriltag's Mali GPU are separate units that share only DRAM, so they could be combined. The combination makes no sense for latency. PV's own RKNN benchmark (`docs/source/docs/benchmarks/rknn-model-benchmarks.md`, OPi5 RK3588, int8, COCO val) gives a **YOLOv11 total of 22,988.68 ± 2,355.97 µs** per inference. That is about 6x vkapriltag's whole 3.76 ms GPU phase, before the crop detects even start. This model is the COCO one, so the AprilTag model's size and input resolution may differ (unchecked). vkapriltag should therefore beat the ML-ROI path on latency and match libapriltag exactly. (2) Any ROI mode in vkapriltag (for example temporal ROIs from last frame's detections) breaks the parity gate just as this PR does, so it must be opt-in, like ArUco3's temporal mode. PV's reviewers will likely compare vkapriltag against this pipeline, so benchmark both on the same OPi5 once #2604 publishes numbers.
- Verification: [verified from primary source] (PR text; there are no performance numbers to verify)

### PhotonVision's own throughput claims and a stream-cost measurement — PV docs (cached clone 1f419c9, 2026-09-22) and wifijt/photonvision-tools PR #1 (2026-09-14)
- URL: `docs/source/docs/quick-start/common-setups.md`, `quick-configure.md`, `apriltag-pipelines/detector-types.md` in https://github.com/PhotonVision/photonvision; https://github.com/wifijt/photonvision-tools/pull/1
- Technique / numbers: The PV docs claim that the Orange Pi 5 4GB "Supports up to 2 object detection streams, along with 2 AprilTag streams at 1280x800 (30fps)", and that the Raspberry Pi 5 supports "up to 2 AprilTag streams at 1280x800 (30fps)". The recommended OPi5 + OV9281 settings are 1280x800, **decimate 2**, 3D mode, MultiTag. The ArUco pipeline type is "~2x higher fps and ~2x lower latency than the AprilTag pipeline type, but is less accurate". photonvision-tools PR #1 measured a Pi 5 with an OV9281 at 1280x800 (freshly restarted service, 3 interleaved 45-s cycles): **streams on 60.1 fps / 42.0 ms latency, streams off 80.5 fps / 32.7 ms**. The stream encoder costs about a quarter of the throughput. Two cameras with threads=1 each gave 31.3 + 52.6 = 84.0 fps total. (A first measurement, later retracted as degraded by a long-running process, showed 55.9 vs 60.1 fps.)
- Accuracy / determinism impact: n/a (measurements).
- Applicability to vkapriltag: (1) This sets the bar: PV's documented OPi5 target is only 30 fps x 2 streams at 1280x800 d2, and vkapriltag's GPU phase (3.76 ms, about 266 fps GPU-bound) exceeds that by a large margin. The end-to-end limit on OPi5 will be capture, MJPEG decode and the MJPEG stream encoder, not the detector. This matches SpectrumJetson's "GPU 12% busy" finding. (2) Actionable for the PV integration: measure vkapriltag end to end with streams off and on. The stream (JPEG encode of the annotated frame, CPU) competes for the same A76 cores as the CPU tail, so pinning (survey B4) matters more than further GPU-phase cuts. The Pi 5 figure (80.5 fps with streams off, CPU AprilTag) is a useful OPi5-CPU-class reference.
- Verification: [verified from primary source] (docs are from the cached clone; the PR numbers came via WebFetch's summary of the PR page)

### CubVision (Robocubs, FRC 1701): OpenCV ArUco on Orange Pi 5, "90fps @ 8ms"
- URL: https://github.com/Robocubs/CubVision (README via WebFetch)
- Technique: A Northstar-style Python pipeline that uses OpenCV's ArUco module with the AprilTag dictionary on an Orange Pi 5.
- Reported speed-up: "90fps @ 8ms" on an OrangePi 5. Resolution, decimation and tag count not stated.
- Accuracy / determinism impact: ArUco detector, not libapriltag-exact.
- Applicability to vkapriltag: reference point only. OpenCV ArUco on the OPi5 CPU reaches about 8 ms per frame. vkapriltag at 3.76 ms GPU plus a pipelined 1.2 ms tail should beat it while staying libapriltag-exact, which is the pitch against the "ArUco is 2x faster" PV docs line.
- Verification: [secondary/unverified] (README claim, no methodology)

### argustag: Argus CSI camera -> EGLStream -> CUDA -> Isaac cuAprilTags (nvAprilTags) — pietroglyph (FRC), 2021 (last commit 2022-11-03)
- URL: https://github.com/pietroglyph/argustag (cloned to `fiducial/cv/argustag`; read `argus_camera.cpp`, `main.cpp`)
- Technique: Libargus captures from the CSI ISP into an `EGL_STREAM_MODE_MAILBOX` EGLStream, which keeps only the latest frame. A CUDA consumer (`cudaEGLStreamConsumerConnect` / `AcquireFrame` / `cudaGraphicsResourceGetMappedEglFrame`) receives the NV12 frame as CUDA arrays. The "zero-copy" is not complete: the Y and CbCr planes are `cudaMemcpy2DFromArray`-copied into linear device buffers (the TODO says a surface-reading kernel would be faster), then `cudaNV12ToRGBX` converts them to RGBA. The reason is that the closed-source **nvAprilTags API only accepts `uchar4` RGBA input** (`nvAprilTagsImageInput_t`). The tag detect runs on its own CUDA stream.
- Reported speed-up: none in the repo.
- Accuracy / determinism impact: n/a (cuAprilTags is closed and not libapriltag-exact).
- Applicability to vkapriltag: (1) This is the architectural template for survey B1/B3 on RK3588. The camera path (rkisp/rkcif V4L2 or an MPP JPEG decoder) should deliver a dma-buf that is imported with `VK_EXT_external_memory_dma_buf`. vkapriltag's `decimate.comp` then reads the **Y plane of NV12 directly** as gray: no conversion and no copy. vkapriltag already takes 8-bit gray, so it avoids cuAprilTags' forced RGBA expansion (4 bytes per pixel read, versus 1). (2) Mailbox (latest-frame-only) semantics are the right choice for latency in PhotonVision too. Mali only for the dma-buf part.
- Verification: [verified from primary source] (source read; no numbers)

### EagleEye (Scythe Engineering): libapriltag + pose-predicted ROIs with per-ROI decimation, "120 AprilTag fps on Raspi 5" — ElliotScully, Chief Delphi 2026-09-09
- URL: https://www.chiefdelphi.com/t/eagleeye-photonvision-alternative-with-120-apriltag-fps-on-raspi-5/523976; https://github.com/Scythe-Engineering/EagleEye-Vision-System (cloned to `fiducial/cv/eagleeye`, 2fb666c 2026-09-17; read `src/rust_implementations/modules/temporal_acceleration/src/lib.rs` and `src/main_operations/modules/apriltags/apriltag_detector.py`)
- Technique: The detector is plain libapriltag through `pupil_apriltags` (pinned 1.0.4.post11). The speed comes from **"temporal acceleration"**, a Rust module. It takes the previous camera pose (`last_pose_world_from_camera`) and the field map's known 3D tag corners, projects every tag through the camera matrix and distortion, pads each box (`padding_factor=0.35`), and caps the set at `max_regions=20` of at least `min_region_size_px=16`. Only those ROIs are searched. **Decimation is chosen per ROI**: ROIs under `small_roi_max_px=32` use a decimate-1 detector, ROIs of at least `large_roi_min_px=96` use `large_roi_decimate=3.0`, and the rest use the default `quad_decimate=2.0`. A full-frame search (with its own thread count) is the fallback. There is also a synthetic Blender replay benchmark (`benchmarks/`).
- Reported speed-up: forum post: single camera AprilTags **120 FPS at 1280x800**, and 3 cameras at 60 FPS each at 1280x800, on a Raspberry Pi 5. No latency, baseline or methodology quoted in the post. The PV-on-Pi-5 reference in this file is 80.5 fps with streams off (photonvision-tools PR #1).
- Accuracy / determinism impact: not libapriltag-exact. It misses tags that are outside the predicted ROIs (anything not in the field map, or after a pose jump), and ROI crops change the threshold tiling.
- Applicability to vkapriltag: This is the strongest CPU competitor in the FRC niche, and it is a **tracking** design, not a faster detector. vkapriltag's full-frame bit-exact detector at 3.76 ms on Mali (about 266 fps GPU-bound) already beats the claimed 120 fps without any priors. The transferable, opt-in idea is **per-region decimation** (decimate 1 for tiny far tags, 3 for big near tags). A GPU version would run `decimate`/`threshold` over a small set of ROI rectangles via indirect dispatch. That breaks the parity gate, so it can only be an "ROI mode" layered on top. Rank it low for vkapriltag, and treat it as competitive context for the PhotonVision pitch: publish end-to-end fps at 1280x800 with streams off, to compare like for like.
- Verification: [verified from primary source] for the source-code mechanism; [secondary/unverified] for the fps numbers (forum post via WebFetch summary, no methodology)

### Limelight 4 (Raspberry Pi CM5 + Hailo-8/8L): "Hailo-accelerated AprilTags", up to 50% at 1x downscale — Limelight changelog, OS 2026.1 (2026-04-17)
- URL: https://docs.limelightvision.io/docs/docs-limelight/software-change-log (via WebFetch); Hailo/Limelight CES 2025 press release (search result only)
- Technique: Closed source. Limelight 4 is a CM5 with an optional Hailo NPU. The changelog says: "Hailo-Accelerated AprilTags have returned. With Hailo 8L, teams will see up to a 50% performance improvement at 1x downscaling". Search snippets add "diminishing returns at 2x downscaling and beyond", and a known issue that Hailo-accelerated AprilTag framerates are too low with downscaling enabled. Which stage runs on the NPU is not disclosed. Given the NPU's nature, it is most plausibly a learned ROI/candidate finder, or the threshold as a conv layer. 2024.0, 2024.5 and 2026.0 only say "Higher FPS AprilTag pipelines".
- Reported speed-up: "up to 50%" at 1x downscale, Hailo 8L vs. CM5 CPU. No absolute fps.
- Accuracy / determinism impact: unknown.
- Applicability to vkapriltag: This is weak but consistent evidence with VPI-PVA (Part 2) and the PV ML-ROI PR: NPU/DSP offload helps only at full resolution (decimate 1) and loses its advantage at decimate 2, which is PV's recommended setting and vkapriltag's headline configuration. This supports **not** pursuing the RK3588 NPU for any detector stage.
- Verification: [secondary/unverified] (changelog seen through the WebFetch summariser; the "diminishing returns" wording is from a search snippet only)

### ArUco Nano v6: "visited-aware" contour tracing — R. Muñoz-Salinas et al., SoftwareX 2026 (paper S2352711026001822)
- URL: https://github.com/rmsalinas/aruco_nano (cached `fiducial/algo/aruco_nano`; `aruco_nano.h` read, 709 lines); paper link from the README, not opened
- Technique: A single-header ArUco detector. Adaptive threshold (box filter 15, offset 3) is followed by a new contour tracer, `visitedAwareTracingContour`, that marks each traced pixel `VISITED=100` in the padded binary image. It counts how many already-visited pixels a trace re-enters and drops the contour once revisits exceed `maxTimesRevisited` (5% of contour length). This avoids re-tracing the same boundary from many start points, which is what `cv::findContours` spends its time on. The bit sampling tries plain, then local-adaptive, then Otsu.
- Reported speed-up: README: "Up to 6.5x faster than standard OpenCV ArUco (single-threaded) and 2x faster than the multi-threaded implementation". Hardware not stated in the README.
- Accuracy / determinism impact: a different detector from libapriltag (it claims a higher F1 on hard datasets).
- Applicability to vkapriltag: none directly. The "mark visited, abort duplicated work" idea is the contour-tracing analogue of what union-find already does. It is another data point that the ArUco-family CPU detectors PhotonVision offers keep getting faster, so vkapriltag's pitch has to be "bit-exact AprilTag 3 quality at ArUco-class speed".
- Verification: [verified from primary source] (source read; speed claim is the author's README)

### Kallwies et al., apriltags_tas: accuracy, not speed — UniBw TAS, ICRA 2020 (cached `fiducial/algo/apriltags_tas`, 7239eb7 2026-05-08, and `kallwies.html`)
- URL: https://github.com/UniBwTAS/apriltags_tas; https://ieeexplore.ieee.org/document/9197427
- Technique: A ROS wrapper of the AprilTag 1 C++ port with corner-refinement improvements for localisation accuracy.
- Reported speed-up: none. It is an accuracy paper.
- Accuracy / determinism impact: better corner accuracy; not libapriltag-exact.
- Applicability to vkapriltag: none for speed. Listed so it is not re-read.
- Verification: [verified from primary source] (README)

### Web literature sweep for 2018–2026 GPU/FPGA/NPU AprilTag accelerators — negative result
- URL: searches run on 2026-09-26: "AprilTag 3 FPGA accelerator ... union-find hardware", "AprilTag GPU accelerated detection paper ... embedded", "apriltag OpenCL/Metal/WebGPU/Vulkan".
- Technique / findings: No peer-reviewed GPU AprilTag 3 detector paper turned up. The GPU implementations that exist are 971/Team766/FRC-4143 (CUDA), closed-source cuAprilTags, and vkapriltag itself. wykvictor/AprilTag-GPU (and its fork sabahmax-inc/AprilTag-GPU) contain no GPU code. The FPGA work is the theses already covered (Flottmann 2019, whose ResearchGate German title is "Entwurf und High-Level-Synthese einer FPGA-basierten Hardwarebeschleunigung des AprilTag-Algorithmus auf einem Zynq-SoC"; Tola; Zhang) plus generic streaming CCA such as Union-Retire (J. Imaging 8(4):89, 2022; a single-pass raster CCA that saves up to 36% of memory resources), which is an FPGA streaming design and does not map to GPU union-find. The one adjacent embedded-GPU paper is Chang et al. 2025 (arXiv 2506.07164), which reports >7.3x over OpenCV-CUDA for ORB FAST/Harris on a Jetson TX2 using binary-encoded candidate tests. That is a feature detector, and its "binary-level encoding" parallels vkapriltag's packed-word decimate/threshold, which already exists.
- Reported speed-up: see above.
- Accuracy / determinism impact: n/a.
- Applicability to vkapriltag: The state of the art for GPU AprilTag 3 is the 971 lineage, and vkapriltag has already moved past it on the host-orchestration axis. Further gains must come from the CCL/atomics/scan literature (categories 03–06), not from other fiducial implementations.
- Verification: [verified from primary source] for the arXiv abstract and search listings; the Union-Retire figure is from a search snippet [secondary/unverified]

---

## Conclusions

The newer 971 CUDA detector's 30–40% gain over the Aug-2024 copy came from host-sync, copy and memory hygiene (971-3/5/6/7), which vkapriltag already has in a stronger form (device-side indirect sizing, fused tail submit, speculative convergence). The GPU kernels did not change (971-8). No other open GPU AprilTag 3 implementation exists (wykvictor contains no GPU code; cuAprilTags is closed). No peer-reviewed GPU/FPGA AprilTag 3 paper offers a technique vkapriltag lacks. The remaining wins from this category are therefore **parameters, work avoidance, and the system around the detector**, not new kernel algorithms.

Ranked ideas:

| # | Idea | Expected uplift | Accuracy / determinism risk | Device | Cheap bound before building |
|---|---|---|---|---|---|
| 1 | **Expose `min_white_black_diff` in PhotonVision and sweep it** (971-1). 971 runs 20 on all Orin cameras, calling it "a massive speedup". PV hard-codes 5; the 4143 JNI hard-codes 5; bos uses 4. | Unknown. Every stage after threshold scales with the non-127 pixel count, so it is plausibly large for labelling, blob_diff, extents and sort (971's "probably about 2x" combines it with NEON and was never measured). | Bit-identical to libapriltag *at the same value*. Trades recall on low-contrast or blurred tags. | both | Sweep 5/10/15/20 on the corpus, recording per-span GPU times and detection recall. No code change. |
| 2 | **Empty-tile mask plus workgroup early-out** (Frappe FIND_EMPTY_TILES). Threshold (or block_filter) writes one bit per workgroup tile: "all 127". `uf_init`/`uf_merge`/`uf_compress`/`blob_diff`/`label_pixels` workgroups test it once and exit, or an indirect dispatch covers only live tiles. Checked in-tree: the shaders test 127 only **per invocation** (`uf_init` join_left, `uf_merge` select, `blob_diff` `c0==1` return), with no tile-level skip. | Proportional to the uniform-tile fraction, and it compounds with #1. On Mali it saves per-workgroup scheduling and dependent loads, not just ALU. | Bit-identical (127 pixels never merge and never emit boundary points). The tile bit must be computed exactly. | Mali mainly; RDNA smaller | Log the fraction of all-127 16x16 (and workgroup-sized) tiles over the corpus at mwbd 5 and 20. Prototype only if it is above ~40–50%. |
| 3 | **Benchmark and gate at PhotonVision's real parameters**: criticalAngle 45° (cos 0.707), **minClusterPixels 5**, maxLineFitMSE 10, mwbd 5, decimate 2. vkapriltag's defaults are 0.98 / 24. | Measurement, not a speed-up. minClusterPixels 5 admits many more small blobs into select/sort/fit, so the headline 3.76 ms may **understate** the PV workload, while 45° shrinks the CPU quad tail. | none | both | Rerun the benchmark and the parity gate with PV's values. Do this before any optimisation work, to target the right spans. |
| 4 | **Frequency governors** (rapidtag): `performance` for the A76 policy and the Mali devfreq (`fb000000.gpu`). Bursty per-frame work never ramps `schedutil`. | rapidtag measured 123 -> 211 fps (~1.7x) on a QCS6490 CPU. For vkapriltag, plausibly tens of percent end to end. | none | Mali/RK3588 | A/B the benchmark under each governor. |
| 5 | **System path around the detector** (argustag, photonvision-tools, SpectrumJetson, PV docs). Import the camera/MPP-decoder NV12 dma-buf and read the **Y plane directly** in `decimate` (no conversion, no copy); use latest-frame mailbox semantics; pin the stream encoder away from the CPU-tail cores. The PV stream encoder alone cost about 25% throughput on a Pi 5 (80.5 -> 60.1 fps), and SpectrumJetson's GPU was only 12% busy. | End to end, likely larger than any remaining GPU-phase cut once vkapriltag is inside PV. | none | Mali | Measure PV + vkapriltag end to end at 1280x800 with streams on and off, and record the capture/decode/encode spans. |
| 6 | **NEON decimate+threshold on the idle A76s** (971-2, `neon_threshold.cc`). This takes the threshold span off Mali and overlaps it with the previous frame's GPU work. | At most the Mali threshold span. | Must be bit-exact (971 has a threshold_test). It competes with the CPU tail for the big cores. | Mali only | Time 971's NEON code on an A76 at 1280x800 and compare it with the Mali threshold span. |
| 7 | **Bump the upstream pin past v3.4.5.** This gets `d686c00` (no `sched_yield` in workerpool; affects the CPU reference/baseline path) and pigeonhole `quick_decode` (smaller decode tables, faster detector creation). | Small, CPU side. Mostly makes baselines honest. | Re-run the bit-exact gate. | both | Diff the timings of the CPU baseline before and after. |
| 8 | **(Speculative) GPU peak-NMS + quad fit on RDNA**, as 971 does. | Only if the CPU tail exceeds the 0.66 ms GPU phase on the desktop. | Bit-exactness is hard (971's float fits are not exact). | RDNA | Compare the desktop CPU tail against the GPU phase first. |
| 9 | **Opt-in ROI mode**: temporal / pose-predicted ROIs with per-ROI decimation (EagleEye: decimate 1 below 32 px, 3 above 96 px; PV ML-ROI PR). | Large on sparse frames (EagleEye claims 120 fps on a Pi 5 CPU). | **Breaks the parity gate**: it loses tags outside the ROIs. It must be opt-in and off by default. | both | Not needed for the pitch, since vkapriltag's full-frame speed already exceeds these trackers' claims. |

**Not recommended:** NPU/DSP offload of any detector stage on RK3588. Three independent data points agree:
- VPI's PVA backend loses to VPI's own multithreaded CPU path at every resolution (Orin 1280x720: PVA 4.5 ms vs CPU 2.75 ms).
- Limelight's Hailo AprilTag acceleration gives "up to 50%" only at 1x downscale, with diminishing returns at 2x.
- PV's RKNN benchmark puts one YOLOv11 inference on the RK3588 NPU at about 23.0 ms, roughly 6x vkapriltag's entire GPU phase.

The PV ML-ROI PR (#2604) is therefore a competitor on accuracy trade-offs, not on latency.

**Positioning numbers:**

| Detector | Platform | Figure |
|---|---|---|
| vkapriltag (bit-exact) | OPi5 Mali, 1280x800 d2 | 3.76 ms GPU + ~1.2 ms pipelined CPU tail |
| vkapriltag (bit-exact) | RX 9060 XT | 0.66 ms |
| PV docs' own OPi5 claim | OPi5 | "2 AprilTag streams at 1280x800 (30fps)" |
| PV CPU | Pi 5, streams off | 80.5 fps |
| EagleEye (temporal ROI) | Pi 5 | 120 fps (claimed) |
| CubVision (ArUco) | OPi5 | 90 fps @ 8 ms (claimed) |
| 971 CUDA | Orin, 1280x800 | 1.7 ms |
| Isaac ROS cuAprilTags | AGX Orin, 720p | 5.3 ms |
| Isaac ROS cuAprilTags | RTX 5070 | 1.4 ms |

**Could not verify:**
- The magnitude of 971's "probably about 2x" (never measured by its author).
- Isaac ROS latency composition (transport vs. kernel).
- Whether halide-apriltag's edge handling is bit-identical.
- The fps claims of EagleEye, CubVision and Limelight (no methodology; forum/README/changelog seen through a summariser).
- Which stage Limelight runs on the Hailo.
- The input size of PV's shipped `apriltagV4-yolo11.rknn` (the 23 ms figure is for the COCO YOLOv11 benchmark model).
- The Union-Retire memory-saving figure (search snippet only).
- The `learned\` papers were only title-scanned, and are left to the algorithmic category.
