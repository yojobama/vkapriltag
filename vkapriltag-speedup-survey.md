# vkapriltag speed-up survey — 2026-09-25

> **Superseded in part by `research/00-SUMMARY.md` (2026-09-26).** That file
> merges eight category reports (`research/01`–`08`, 219 source entries). It
> adds about 20 exact options not listed here, and one correctness finding to
> resolve first: **vkapriltag unions white pixels 4-connected, while upstream
> libapriltag unions them 8-connected.** Rankings below that conflict with
> it: B1/B3 (camera import) now come with specific hazards, and a desktop
> single-submission path (00-SUMMARY B1) is likely worth more than anything
> in section 3 for the RX 9060 XT.

A list of every remaining option found for making vkapriltag faster, drawn from
three places: the tree's own record of what has been tried
(`apriltags_vulkan/OPTIMIZATION_NOTES.md`, `PERFORMANCE.md`,
`vkapriltag-optimization-ideas.md`), a search of recent literature and
implementations, and the Vulkan extension lists of both target GPUs. Each
option has an expected uplift and an accuracy class.

**Read the "Already settled" table (section 6) before proposing anything.**
Most of the obvious ideas were measured on this pipeline, and many of them
came out slower.

## 0. What the pipeline is, and where the time goes

vkapriltag is a Vulkan-compute port of the AprilTag 3 detector, derived from
FRC 971's CUDA detector (via Team 766's standalone copy). It is built for
PhotonVision on the RK3588 / Mali-G610 (Orange Pi 5) and verified against
upstream libapriltag tag-for-tag and corner-for-corner.

- **GPU:** decimate → adaptive threshold → union-find labelling → boundary points → hash grouping → extents → blob selection → per-blob sort + line-fit moments.
- **CPU tail:** quad fit → upstream tag decode → optional edge refinement and pose.

Deployment numbers (Mali-G610, 1280x800, decimation 2, from `PERFORMANCE.md`):

| | Mali-G610 | RX 9060 XT |
| --- | --- | --- |
| GPU phase | 3.76 ms | 0.66 ms |
| quad_decode + tag_decode (CPU) | 0.44 + 0.79 ms | 0.09 + 0.29 ms |
| Serial / `FramePipeline` | 5.00 / 3.52 ms | 1.04 / 0.78 ms |

Measured facts that constrain every estimate below:

- **Only about 15% of Mali GPU time is DRAM bandwidth.** Removing *all* memory traffic would buy about 0.5 ms.
  - `labelling`, the largest span (~27%), is limited by dependent loads (chasing `find()` chains), not bandwidth.
  - `extents` and `sort` are limited by atomics and latency. They don't move at all across a 4x memory-clock sweep.
- **Mali has no dedicated shared memory**: `shared` is backed by L2. Tile-local and shared-memory schemes lost there every time they were tried.
- **Subgroup variants are 57% slower on the G610.** They are excluded outright on integrated GPUs.
- **Throughput is set by the GPU phase.** Under `FramePipeline` the CPU tail is hidden (3.52 ms per frame vs 3.76 ms GPU), so CPU-side savings only help the serial (lowest-latency) path.

## 1. How to read the uplift and accuracy columns

**Uplift** is the expected change in *GPU total on the Mali-G610 at decimation 2*, unless the row says otherwise.
- "Measured" means an ABBA-interleaved measurement exists, and says on which device.
- Everything else is an estimate. The reasoning is in the item's section.

**Accuracy class:**

| Class | Meaning |
| --- | --- |
| **Exact** | Bit-identical detections, corners and counters. The tree's normal gate applies. |
| **Exact\*** | Same final results, but reached through a different nondeterministic path. Still gate on bit-identity. |
| **Pixel-level** | Changes input pixels (for example a different JPEG decoder). Detections can differ at the noise level. |
| **Recall risk** | Can miss tags the full detector finds (new, fast-moving, small or far tags). |
| **Accuracy risk** | Can move corners or pose, even when the tag set is unchanged. |
| **Parity loss** | Gives up the libapriltag-identity guarantee entirely. |

## 2. Summary: all remaining options, ranked

| # | Option | Stage | Device | Expected uplift | Accuracy | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | **Path halving in `find()`** (ECL-CC "intermediate pointer jumping" / "inline compression") | labelling | all | **Measured on RX 9060 XT:** labelling −17% (d1), −2.7% (d2); GPU total −3.5% (d1), noise at d2. Mali: est. 1–5% GPU total, unmeasured | Exact | New. Bounded today; not in the tree |
| A2 | Conditional rendering for the converged `uf_compress` | labelling | dGPU only | **Measured on RX 9060 XT:** labelling −3.0% (d1), GPU total ≤1% | Exact | Built, **uncommitted** in the working tree |
| A3 | Per-shader wave32/wave64 via `VK_EXT_subgroup_size_control` | labelling, boundary, sort | RDNA (32–64); no-op on Mali (fixed 16) | Unknown, possibly 5–15% on latency-bound shaders on AMD | Exact | New |
| A4 | Record the fused command buffer once, resubmit each frame | CPU submission | all, mainly Mali | Est. 0.05–0.3 ms CPU per frame (latency) | Exact | New |
| A5 | Remove the per-frame `vkCmdFillBuffer` clears (epoch-tagged hash table; zero `blob_size` inside `uf_init`) | clear | all | Est. 1–3% (the `clear` span is 0.05–0.18 ms and 3.7x bandwidth-bound) | Exact | New |
| A6 | Per-iteration convergence flags, so iterations after convergence are skipped (fixes the ratcheting chunk seed) | labelling | all (dGPU via predication) | ~0 in steady state; helps after a hard frame | Exact | New |
| B1 | Zero-copy frame import (`VK_EXT_external_memory_host`, or dma-buf) | upload | RX: host yes, dma-buf no. Mali: dma-buf yes, host not listed | Est. 0.05–0.2 ms (the upload memcpy) | Exact | New |
| B2 | Decimate straight from the YUYV/NV12 luma plane on the GPU | ingest | all | Removes a full-resolution CPU pass in the capture path (est. 0.2–1 ms of CPU) | Exact | New |
| B3 | Hardware MJPEG decode on RK3588 (MPP JPEG decoder → dma-buf) | camera, outside the library | RK3588 | Large end-to-end, for MJPEG USB cameras (CPU JPEG decode is typically several ms per frame) | Pixel-level | New, integration work |
| B4 | Pin CPU-tail workers to the A76 big cores | CPU tail | RK3588 | Unknown, bounded by the ~1.2 ms tail. Helps serial latency only | Exact | New |
| B5 | Global-priority queue (`VK_KHR/EXT_global_priority`) | scheduling | Mali and RX both expose it | 0 throughput; less jitter when the GPU is shared | Exact | New |
| C1 | Temporal ROI: search near last frame's tags, full frame every N frames | whole GPU phase | all | 2–10x on ROI frames (scales with area) | **Recall risk** | Literature |
| C2 | Adaptive decimation from last frame's smallest tag (ArUco3-style) | whole pipeline | all | Up to ~2.1x (decimation 4 measured 2.1x on the Pi) | **Recall risk** + **accuracy risk** | Literature |
| C3 | Coarse detection plus full-resolution corner refinement | pipeline | all | Recovers most of C2's corner error for about +0.1–0.8 ms CPU | Accuracy risk (smaller than C2) | Upstream feature |
| C4 | Cheap rejection of candidate quads before `tag_decode` | CPU tail | all | ≤0.79 ms of Mali CPU; serial path only | **Recall risk** unless proven conservative | New |
| C5 | Learned detector on the RK3588 NPU (YoloTag family) | replaces pipeline | RK3588 NPU | Paper: 55 vs 24 fps on its hardware | **Parity loss** | Literature, not recommended |

The three that are both new and exact, worth doing first: **A1** (measure on the
Mali), **A5**, and **A4**. A2 is ready to commit for desktop GPUs.

## 3. Tier A — exact GPU-side changes

### A1. Path halving inside `find()` — measured today, the largest new exact item

`uf_merge_body.glsl`'s `find()` walks to the root without touching the path.
All flattening happens in the separate `uf_compress` pass after each merge. So
within a merge pass, every thread whose walk crosses a freshly hooked root
re-walks the same chain.

The standard fix in GPU union-find is to rewrite each visited node's parent to
its grandparent during the walk:

- ECL-CC calls it "intermediate pointer jumping" and reports it as the fastest of four variants in all but one case.
- The 2024 TPDS CCL survey calls it "inline compression" and notes that BUF and BKE, the fastest 2D GPU labellers, both use it.

The experiment:

```glsl
uint find(uint n) {
  uint p = parent[n];
  if (p != n) {
    uint prev = n, next = parent[p];
    while (p > next) { parent[prev] = next; prev = p; p = next; next = parent[p]; }
  }
  return p;
}
```

Why it is safe here:
- Every parent pointer satisfies `parent[x] <= x` (atomicMin hooking, left-neighbour init), so the rewrites never create a cycle.
- A plain store that loses a race with an `atomicMin` is covered by `doUnion`'s existing retry: that thread sees `old != b` and re-unions through `old`.
- The final root is still the component's minimum index, so labels are canonical and everything downstream is unchanged.

**Measured on the RX 9060 XT** (two binaries, ABBA, 16 rounds × 300 iterations,
1280x800, median of per-round paired deltas):

| | `labelling` | GPU total |
| --- | --- | --- |
| decimation 1 | **−17.0%** (faster in 14/16 rounds) | **−3.5%** (14/16) |
| decimation 2 | **−2.7%** (16/16) | +0.2% (noise) |

**Correctness:** bit-identical to the baseline over 36 configurations
(decimations 1/2/4 × 8-bit storage on/off × workgroup geometries auto, 2x2
and 6x6 × chunk size 1 and default). `uf_iterations` is unchanged.

**Mali:** unmeasured. It should help there, because `labelling` is
dependent-load bound (bandwidth ratio 1.30x). But the extra writes are the
cost the survey warns can cancel the gain, and Mali shares one LPDDR bus with
the CPU. Extrapolated estimate: 1–5% GPU total at decimation 2, more at
decimation 1.

**Accuracy:** Exact.

**Next step:** one ABBA pass on the Pi.

### A2. Conditional rendering for the converged `uf_compress` — done, uncommitted

`VK_EXT_conditional_rendering` skips the last compression of each labelling
chunk when the convergence flag is zero. This recovers the part of
`PERFORMANCE.md` §6c's ceiling that the in-shader guard cannot.

- **Measured (RX 9060 XT):** labelling −3.0% (decimation 1, faster in 15/16 rounds); GPU total ≤1%.
- **Availability:** absent on the Mali-G610 (not in the Android driver report either), so it only matters for desktop users.
- **Accuracy:** Exact. Bit-identical over 60 configurations.
- **Caveat:** the validation layer does not model the predicate read. A deliberately broken barrier also passed validation.
- **Status:** in the working tree, not yet committed.

### A3. Per-shader subgroup size on RDNA

The RX 9060 XT reports subgroup sizes 32 to 64 and supports `requiredSubgroupSize`
for compute. The driver's default for compute is usually wave64. Wave32 halves
the divergence cost of branchy, latency-bound loops such as `find()` chains,
the hash probe, and the odd-even merge network.

- **Mali:** a no-op (`min == max == 16`), as the tree's extension survey already says.
- **Accuracy:** exact for every shader without subgroup operations. `reduce_extents_hash_subgroup` is exact at any size.
- **Uplift:** unknown until measured. It is one pipeline-creation flag per shader, so it's cheap to sweep.

### A4. Pre-recorded command buffers

On the fused (unified-memory) path the whole frame is a fixed sequence:
every tail dispatch is indirect, and the only variable is the chunk count. So
the command buffer can be recorded once per `(chunk, speculative)` pair and
resubmitted each frame.

- **Saves:** per-frame `vkCmd*` recording (~60 dispatches plus barriers and push constants) and its driver validation.
- **Doesn't save:** GPU time.
- **Measured so far:** `cpu_begin_ms` on AMD is 0.05 ms. Recording time itself is not split out in `DetectProfile`, so measure that first; on the A76 with libmali it is plausibly 0.1–0.3 ms.
- **Helps:** serial latency, and CPU headroom for the pipelined tail.
- **Accuracy:** Exact.

### A5. Stop clearing buffers every frame

The `clear` span is nine `vkCmdFillBuffer`s. Its bandwidth ratio is 3.68x,
almost pure DRAM writes, at 0.048 ms (memory clock pinned at 2112 MHz) to
0.177 ms (idle clock). The biggest targets:

- **`blob_size_buf_`** (`W*H*4`, 1 MB at decimation 2 and 4 MB at decimation 1). `uf_init` already writes every pixel's `parent[]` in one streaming pass, so it can write `blob_size[i] = 0` alongside. That turns a separate transfer pass into extra bytes in a pass that is already running.
- **`hash_owner_buf_`** (256 KB). Tag each slot with the frame number, so a slot from an older frame reads as empty. The frame counter wraps harmlessly if the check is equality.

- **Uplift:** estimated 1–3% GPU total on Mali, from the `clear` span's size.
- **Accuracy:** Exact.
- **Risk:** the epoch tag steals bits from the owner index, and `max_raw_blobs` is bounded at 2^22, so budget the bits carefully.

### A6. Per-iteration convergence flags

`chunk = max(uf_iterations_per_chunk, last_uf_iterations_)` and
`uf_iterations` starts at `chunk`, so the seed never shrinks. After one hard
frame, every later frame runs that many merge and compress passes. Only each
chunk's last pass reads a freshly cleared flag, so the passes after
convergence are never skipped. The chunk=8 measurement for A2 confirmed this.

**Fix:** one flag per iteration, all zeroed in the frame clear. Iteration k+1
is then gated on flag k: predicated on desktop GPUs, and via an in-shader
merge guard on Mali.

- **Uplift:** ~0 in steady-state video (the corpus converges in 2 iterations). Worth it only if real scenes show the seed climbing.
- **Accuracy:** Exact.

## 4. Tier B — ingest and system integration (exact, mostly outside the GPU phase)

### B1. Zero-copy frame import

Today `Detect()` memcpys the caller's gray frame into a host-visible buffer.
On unified memory that's one full-frame CPU copy, about 1 MB at 1280x800.

- **RX 9060 XT:** exposes `VK_EXT_external_memory_host`. Import the caller's pointer directly and skip the staging copy.
- **Mali-G610:** the Android driver report (r38p1, 110 extensions) lists `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd` and `VK_EXT_image_drm_format_modifier`, but **not** `VK_EXT_external_memory_host`. The libmali g24p0 list on the Pi (130 extensions) needs checking. The Pi-side path is to import the V4L2/rkisp dma-buf.

- **Uplift:** estimated 0.05–0.2 ms, the upload share of `total_ms`.
- **Accuracy:** Exact.
- **Risks:** stride and alignment rules for imported memory; the caller must keep the buffer alive until the frame finishes; host-pointer alignment (`minImportedHostPointerAlignment`).

### B2. Decimate from the camera's native format

`decimate.comp` point-samples. At decimation 2 it reads a quarter of the pixels.

- **Today (demo capture app):** the CPU first extracts luma from YUYV (a full-resolution strided loop), then the library copies it again.
- **Proposed:** a decimate variant that reads YUYV luma, or the NV12 Y plane with its stride, directly. Combined with B1 this removes both CPU passes.
- **Catch:** `TagDecoder` samples the full-resolution gray image on the CPU. For NV12/GREY the Y plane already is that image; for YUYV the decoder needs a gray copy, or a strided sampler of its own.
- **Accuracy:** Exact, since the same luma bytes are used.

### B3. Hardware MJPEG decoding on RK3588

A Jetson deployment of the 971 CUDA detector (SpectrumJetson) found the GPU
about 12% busy. Its real frame-rate limit was JPEG decoding and the USB camera
pipeline.

- **Hardware:** the RK3588's JPEG decoder is rated 1080p at 280 fps and outputs dma-bufs through MPP. Paired with B1, it gives a zero-copy path from camera to GPU.
- **Scope:** this lives in the PhotonVision/camera layer, not the library.
- **Accuracy:** Pixel-level. A different JPEG decoder can round pixels differently from libjpeg-turbo, so detections can differ at the noise level. Libapriltag parity would then be checked on decoded frames, not on the camera stream.

### B4. Pin CPU-tail workers to the big cores

`WorkerPool` spawns `hardware_concurrency()` threads (8 on the RK3588: 4×A76 +
4×A55) with no affinity, handing out items from an atomic counter.

- **Problem:** an A55 that takes the last large blob holds up the whole batch.
- **Evidence elsewhere:** rapidtag reports 68 → 211 fps from pinning to fast cores on a big.LITTLE Qualcomm part. That's a different workload, so treat it as a direction, not a number.
- **Earlier sweep here:** `APRILTAG_CPU_THREADS` found 6 threads best on an older tree.
- **Proposed:** pin to the A76s (4 threads) and compare against unpinned 4, 6 and 8.
- **Accuracy:** Exact.

### B5. Queue priority

Both GPUs expose `VK_KHR_global_priority` (the G610 in the Android report).
When PhotonVision shares the GPU with streaming or UI work, a high-priority
compute queue reduces tail latency.

- **Throughput:** no change.
- **Permissions:** on Linux, realtime priority may need privileges.
- **Accuracy:** Exact.

## 5. Tier C — algorithmic changes that trade accuracy for speed

These change what the detector *finds*, so they break the tree's bit-identity
gate by design. Each needs its own acceptance test: recall and corner RMS
measured on video, not on still images.

### C1. Temporal ROI tracking

Search only windows around last frame's tags, with a full-frame pass every N
frames or whenever a tag is lost.

- **Uplift:** GPU time scales with pixels (`560 us + 3.3 us per 1k source pixels` on Mali). A 25% ROI takes the variable part to a quarter, roughly 3.9 → 1.4 ms by that model. The fixed cost stays.
- **Precedent:** common in practice (OpenMV dynamic ROI, drone and surgical-robot pipelines). The upstream library has an open request for it (AprilRobotics/apriltag #105).
- **Recall risk:** a tag that newly enters the frame is found only at the next full pass; fast motion can leave the ROI.
- **Accuracy risk:** none for the tags that are found, provided the ROI includes the full quad plus the border margin that `refine_edges` samples.
- **GPU implementation:** a crop plus offset on upload. Everything downstream is size-agnostic, but the pipelines are sized per resolution, so budget one detector instance per ROI size class.

### C2. Adaptive decimation, ArUco3-style

Romero-Ramirez et al. (2018) choose the processing scale from the smallest
marker in the previous frame. They report up to 40x speed-up over the
state-of-the-art detector of the time, over 1000 fps on 4K, without
parallelization.

- **Measured here:** decimation 4 is 2.1x faster than 2 on the Pi, but candidate quads drop from 75 to 17 on the reference image.
- **Recall risk:** small or far tags vanish at the coarse scale.
- **Accuracy risk:** corners are fitted on the coarse image, unless C3 is added.

### C3. Detect coarse, refine at full resolution

Upstream's `refine_edges` already re-fits edges against the full-resolution
image after decimated detection. Pairing C2 with refinement always on
recovers most of the corner accuracy.

- **Cost:** `tag_decode` goes up (0.79 ms with refinement off on Mali; `refine_edges` is the single most expensive CPU function, see `PERFORMANCE.md` §4).
- **Recall:** C2's recall risk remains.

### C4. Pre-decode rejection of candidate quads

The reference frame yields 75 candidate quads for one real tag, and upstream
decode runs a homography and bit sampling on each. A cheap test first, for
example border contrast against `min_white_black_diff`, could remove most of
them.

- **Ceiling:** 0.79 ms of Mali CPU, and it only helps the serial path, since `FramePipeline` hides it.
- **Recall risk:** unless the test is proven to reject only quads that decode itself would reject. Gate on the tag set and on corner RMS.

### C5. Learned detectors

YoloTag (2024) reports 55 fps against AprilTag's 24 fps on its UAV hardware.
DeepTag reports 3 fps. On the RK3588 such a model would run on the NPU,
leaving the GPU free.

- **Parity loss:** gives up the libapriltag corner parity this project is built around.
- **Different failure modes:** false positives from a detector, not a decoder. Pose accuracy then depends on the model's keypoint head.
- **Recommendation:** not for this project; listed for completeness. The tree's literature scan already rejected it on the same grounds.

## 6. Already settled — do not re-investigate

Measured in the tree, with numbers in the cited file.

| Idea | Result | Where |
| --- | --- | --- |
| Tile-local / shared-memory union-find | Mali: 3.58 → 5.32 ms at 16x16 (much slower) | OPTIMIZATION_NOTES §"Measured and rejected" |
| BUF / BKE / Playne block-based CCL | Not applicable: 4-connected, three-valued input | same |
| HA4's strip and warp structure | Run-level half taken (−7 to −13% labelling); the warp-intrinsic half is contra-indicated | third pass, items 7 and 9 |
| Subgroup aggregation on Mali | 11.5 → 18.1 ms pipeline total | PERFORMANCE §6 |
| Subgroup aggregation of bare counters (dGPU) | Retired at 2 of 3 sites (scalar 24–72% faster) | third pass, item 6 |
| Fusing decimate+minmax and filter+threshold | +5.9% / +5.4% / +7.8% Mali; deleted | third pass |
| Vectorizing `uf_merge` / `blob_diff` to 4 px/thread | +6% / +11% Mali | item 9 |
| fp16 coordinates or moments | Lossy: half-pixel error above 2048; the covariance cancels | third pass, item 1 |
| Tiled image + `texelFetch` / precomputed `W` | +29–35% threshold span | second pass, A7 and A11 |
| Separable 3x3 filter, barrier-free sort stages | Bounded below 0.03 ms | second pass, A9 and A10 |
| `extract_blob_counts` indirect | ~4 us ceiling | third pass |
| Collapsing submits (on dGPU) | ≤0.05 ms; the fused path already does it on unified memory | items 5 and 8 |
| `VK_KHR_16bit_storage`, `shader_float16_int8`, `scalar_block_layout` | Traffic-only; capped by the 15% bandwidth share | third pass, item 4 |
| `VK_KHR_synchronization2` | Intra-submit gaps total 0.067 ms per frame | same |
| `VK_KHR_buffer_device_address`, `push_descriptor`, `descriptor_buffer` | Host overhead only, already small | same |
| Pinning the memory-controller governor | Variance only; steady state identical | PERFORMANCE §3a |
| Workgroup-size tuning on Mali | Sweeps narrower than the noise | ideas file |
| Mesa PanVK driver | 141 ms vs 11 ms per frame | ideas file |

## 7. Vulkan extension survey

The RX 9060 XT column is from local `vulkaninfo`: AMD proprietary driver
26.8.1, Vulkan 1.4.349, 239 extensions. The Mali column is from vulkan.gpuinfo.org
report 37373: Mali-G610 MC4, **Android** driver r38p1, 110 extensions. The Pi's
libmali g24p0 exposes about 130 (per `OPTIMIZATION_NOTES.md`), so "no" in the
Mali column means "not in the Android report", not a confirmed absence. Rerun
`vulkaninfo` on the Pi to settle these.

| Extension / feature | Could speed up | RX 9060 XT | Mali-G610 | Verdict |
| --- | --- | --- | --- | --- |
| `VK_EXT_conditional_rendering` | skip converged labelling passes | yes | no | **A2, built** |
| `VK_EXT_subgroup_size_control` | wave32 for latency-bound shaders | yes (32–64) | yes (fixed 16) | **A3**, dGPU only |
| `VK_EXT_external_memory_host` | zero-copy frame upload | yes | not in report | **B1** |
| `VK_EXT_external_memory_dma_buf` + `KHR_external_memory_fd` + `EXT_image_drm_format_modifier` | V4L2 / MPP zero copy | no (Windows) | yes | **B1 / B3** on the Pi |
| `VK_KHR_global_priority` / `VK_EXT_global_priority` | latency jitter | yes | yes | **B5** |
| `VK_KHR_shader_atomic_int64` | fewer extents atomics | yes | yes (libmali) | adopted |
| `VK_KHR_8bit_storage` | image traffic | yes | yes | adopted |
| `VK_KHR_pipeline_executable_properties`, `VK_AMD_shader_info` | register and spill statistics while tuning | yes | not in report | diagnostic, use it while doing A1/A3 |
| `VK_KHR_shader_clock` | in-shader timing of `find()` loops | yes | not in report | diagnostic |
| `VK_EXT_device_generated_commands` | GPU-chosen dispatch sequence (labelling loop) | yes | no | Low. Conditional rendering plus indirect dispatch covers it; NVIDIA users report poor DGC performance |
| `VK_KHR_copy_memory_indirect` | device-sized readback copies | yes | no | Low. The fused path reads in place already |
| `VK_KHR_video_decode_queue` | camera decode | yes | no | No MJPEG profile exists; not applicable (B3 uses MPP instead) |
| `VK_KHR_sampler_ycbcr_conversion` | sample NV12 camera frames | yes (core 1.1) | yes | Only with an image path, which A7 rejected; B2 uses buffers instead |
| `VK_KHR_16bit_storage`, `shader_float16_int8`, `EXT_shader_float8`, `KHR_shader_bfloat16` | narrower data | yes | 16-bit and f16 yes | Rejected: traffic-capped and lossy |
| `VK_KHR_cooperative_matrix`, `shader_integer_dot_product` | matrix and dot-product maths | yes | dot product yes | No matrix-shaped work in the pipeline |
| `VK_KHR_shader_subgroup_rotate`, `shader_quad_control`, `shader_maximal_reconvergence` | subgroup algorithms | yes | not in report | Subgroups lose on Mali; dGPU gains bounded (item 6) |
| `VK_KHR_workgroup_memory_explicit_layout`, `zero_initialize_workgroup_memory` | shared-memory layout | yes | zero-init only | Shared memory is the wrong tool on Mali |
| `VK_KHR_shader_expect_assume`, `shader_untyped_pointers`, `EXT_shader_replicated_composites` | codegen hints | yes | no | Negligible |
| `VK_EXT_memory_priority`, `pageable_device_local_memory` | residency on dGPU | yes | no | Working set fits comfortably; no effect |
| `VK_KHR_pipeline_binary`, `EXT_pipeline_creation_cache_control` | startup time | yes | cache control yes | Startup only; the pipeline cache already cuts it 14x |
| `VK_EXT_host_image_copy` | image uploads | yes | no | No images in the pipeline |
| `VK_EXT_shader_object` | graphics state | yes | no | Compute gains nothing |
| `VK_KHR_maintenance5` through `9` | API cleanups | yes | 4 only | No performance effect on this pipeline |
| `VK_ARM_scheduling_controls` | limit shader cores per queue | — | libmali only | Would *reduce* throughput; useful only to leave cores for other GPU work |
| `VK_ARM_shader_core_builtins` | per-core privatization of extents | — | libmali only | K=4 privatization already measured (−18.5% vs −20.3% at K=8); no gain expected |
| Disabling robustness | bounds checks | n/a | n/a | Already off (`robustBufferAccess` is never requested) |

## 8. Suggested order

1. **A1 on the Pi.** It's exact, measured on desktop, and aimed at the largest Mali span. Apply the ECL-CC `find()`, run one ABBA session at decimations 1/2/4, and run the usual bit-identity matrix.
2. **Commit A2** (desktop-only, exact, already verified).
3. **A5, then A4.** Both are exact and cheap. Measure the recording cost before A4.
4. **B1 + B2** once the PhotonVision integration's frame format is fixed (NV12 from rkisp vs YUYV/MJPEG from USB); B3 if the cameras are MJPEG.
5. **C1/C2** only as an opt-in mode with its own recall test on recorded match video. They are the only options here with multiples rather than percentages, and the only ones that change what the detector finds.

## Sources

- [ECL-CC: A High-Performance Connected Components Implementation for GPUs (Jaiganesh & Burtscher, HPDC 2018)](https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf)
- [A State-of-the-Art Review with Code about Connected Components Labeling on GPUs (Bolelli et al., IEEE TPDS 2024)](https://federicobolelli.it/media/publications/pdfs/2024tpds.pdf)
- [YACCLAB benchmark](https://github.com/prittt/YACCLAB)
- [HA4 — A new Direct Connected Component Labeling and Analysis Algorithm for GPUs (Hennequin & Lacassagne, GTC 2019)](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9111-a-new-direct-connected-component-labeling-and-analysis-algorithm-for-gpus.pdf)
- [Speeded up detection of squared fiducial markers (Romero-Ramirez et al., 2018)](http://andrewd.ces.clemson.edu/courses/cpsc482/papers/RMM18_speededAruco.pdf)
- [YoloTag: Vision-based Robust UAV Navigation with Fiducial Markers (2024)](https://arxiv.org/pdf/2409.02334)
- [Team766/apriltags_cuda — standalone FRC 971 CUDA detector](https://github.com/Team766/apriltags_cuda)
- [SpectrumJetson — 971 detector deployment notes (new vs old detector, GPU utilization, JPEG bottleneck)](https://github.com/Spectrum3847/SpectrumJetson)
- [CUDA AprilTag detection with PhotonVision — Chief Delphi, 2025](https://www.chiefdelphi.com/t/cuda-apriltag-detection-with-photonvision/483803)
- [rapidtag — Rust AprilTag/ArUco detector, big.LITTLE pinning results](https://github.com/SujithChristopher/rapidtag)
- [OpenMV forum — AprilTag speed-up with dynamic ROI](https://forums.openmv.io/t/apriltag-speed-up-with-dynamic-roi/2054)
- [AprilRobotics/apriltag #105 — ROI request](https://github.com/AprilRobotics/apriltag/issues/105)
- [Arm GPU Best Practices Developer Guide — compute shading](https://support.arm.com/documentation/101897/latest/Compute-shading/Workgroup-sizes)
- [VK_ARM_scheduling_controls reference](https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_scheduling_controls.html)
- [VK_EXT_device_generated_commands proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_device_generated_commands.html) and [NVIDIA forum: poor DGC performance](https://forums.developer.nvidia.com/t/extremely-poor-vk-ext-device-generated-commands-performance/324189)
- [VK_EXT_external_memory_dma_buf](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_dma_buf.html)
- [vulkan.gpuinfo.org report 37373 — Mali-G610 MC4 (Android r38p1)](https://vulkan.gpuinfo.org/displayreport.php?id=37373)
- [ffmpeg-rockchip — MPP decoder with dma-buf zero copy](https://github.com/nyanmisaka/ffmpeg-rockchip/wiki/Decoder)
- [YUV sampling in Vulkan (Maister)](https://themaister.net/blog/2019/12/01/yuv-sampling-in-vulkan-a-niche-and-complicated-feature-vk_khr_ycbcr_sampler_conversion/)
