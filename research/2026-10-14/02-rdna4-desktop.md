# 02 - RDNA4 desktop (AMD RX 9060 XT, Windows, AMD proprietary Vulkan driver)

Status: complete

## Top conclusions (ranked by expected gain)

- **What limits ~3,000 fps single-stream (~333 us per frame):** derived from in-tree numbers, see
  the budget section.
  - About **115 us (~35%) is submit/launch/completion round trip** that a single in-flight frame
    cannot hide.
  - About **217 us is one frame's command buffer on one hardware pipe**. Of that, roughly **90 us
    (~40%) is idle hardware**: barrier drains, tiny dispatches, and latency-bound tails. Another
    queue's frame can fill that idle time (7,836 fps measured with `alternate`).
  - Host overhead is now only a few us (import plus pre-recording).
  - Wave size and VGPR tuning can only touch the remaining ~130 us of real shader work.

1. **F2 - multi-stream: stop putting every stream on family 0.** 4 streams give ~4,600 fps because
   family 0's 8 queues behave like one pipe. In-tree measurements: queue-per-stream on the
   compute-only family 1 gives 5.2k, and `alternate` across families with import gives **7.8k
   (+70%)**. **Pursue** as the library's lane-allocation default.
2. **F1 - prefer the compute-only family for single streams too.** The default picks the first
   compute family, which is graphics+compute. In-tree data shows family 1 faster by +4% to +24% for
   one stream. It is a few lines. **Pursue after one confirming ABBA.**
3. **F3 - split the ~115 us round trip** with `VK_AMD_buffer_marker` (TOP/BOTTOM_OF_PIPE markers
   polled on QPC) and `VK_KHR_calibrated_timestamps`. That shows whether the cost is submit-side or
   completion-side, and so whether it can be cut (wait on a marker, or system settings). Ceiling
   ~20-28% of single-stream frame time. **Measure first; diagnostic, cheap.**
4. **F6 - give the GPU consumers of `selected[]` a VRAM copy.** The shipped fused path left
   `scatter_index_points` (one 32-byte read per boundary point), `extract_blob_counts` and `sort`
   reading `selected[]` across PCIe. The fused-path change measured +40 us of GPU-side cost at d2.
   A dual write in `select_blobs` is exact and tiny. **Measure first:** the attribution in the tree
   is odd.
5. **F4 - bounded union-find chains (research A8) pay more on RDNA4 than on Mali.** Labelling is
   52% of RDNA4 GPU time against 27% on Mali, and on a wide GPU it is limited by the deepest chain
   (max depth 199 at d2, roughly 20-40 us of serial L2 latency). Estimated 15-40 us. **Measure
   first** on both devices; the agnostic agents own it. Add a `shader_clock` tail histogram on RDNA4.
6. **F7 - cut fills, dispatches and barriers:** 6 counter fills to 1, `blob_size` zeroed in
   `uf_init`, a single-workgroup blob scan, `merge_extents` folded into `select`. About 1.6 us of
   drain plus launch per boundary, ~10-25 us per frame. **Measure first.**
7. **F8 - sort tail:** one 256-thread workgroup per blob means the largest blob (up to 2048 points,
   66 barrier rounds) probably sets the span. Bin large blobs to a 1024-thread instance. **Measure
   first** (blob-size histogram plus per-workgroup `shader_clock`).
8. **F5 - wave32:** for the hot labelling passes, which are latency-bound at full occupancy, wave32
   halves lanes in flight and is likely *slower* (~1.6x on 03's wave model). Sweep only the
   low-occupancy dispatches (sort, hash_group, extents trio). First dump LLPC's actual choices
   (**F9**, a one-off diagnostic). **Low priority.**
9. **Rejected:**
   - F10 SAM/ReBAR: it is off here, BAR types stay CPU-uncached, and BAR upload at best ties import.
   - F11 work graphs: `VK_AMDX_shader_enqueue` is not exposed. DGC cannot bind compute pipelines
     (`PipelineBinding = None`), and the frame has no host round trips left anyway.
   - F12 global and memory priority: zero on an idle GPU.
   - F13 packed, dot and new-atomic ISA features, and WMMA/RT (agent 06): no matching workload.
   - F14 double-buffered detector: throughput only. Real cameras do not need it, and 2 detectors
     already provide it.

## Device facts (from `vulkaninfo`, this PC, 2026-10-14)

Driver: AMD proprietary 26.8.1 (LLPC), Vulkan 1.4.349, deviceID 0x7590 (Navi 44).

- **Queue families:** 0 = graphics+compute+transfer (8 queues), 1 = compute+transfer (8 queues),
  2 = transfer only (1), 3 = video encode, 4 = video decode. Timestamps valid (64 bits) on 0/1/2.
  `timestampPeriod` = 10 ns.
- **Memory heaps:** heap 0 = 15.67 GiB VRAM, heap 1 = 47.9 GiB system RAM, **heap 2 = 256 MiB
  device-local + host-visible**. That is the legacy 256 MiB BAR window: **Smart Access Memory /
  Resizable BAR is OFF on this PC** (with it on, heap 2 would be ~the whole 16 GiB). Types: 2 =
  `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (BAR, CPU-uncached/write-combined), 3 =
  `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` (system RAM), 4-7 = the `DEVICE_COHERENT_AMD |
  DEVICE_UNCACHED_AMD` variants. No type is both device-local and host-cached (confirms
  PERFORMANCE.md 3c).
- **Subgroups:** default `subgroupSize = 64`, `minSubgroupSize = 32`, `maxSubgroupSize = 64`,
  `subgroupSizeControl = computeFullSubgroups = true`, `requiredSubgroupSizeStages` includes
  compute. Ops: basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered, quad, rotate.
- `maxComputeSharedMemorySize = 32768` (the driver reports 32 KiB per workgroup, not 64 KiB).
- Present and relevant: `VK_KHR_pipeline_executable_properties` (`pipelineExecutableInfo=true`),
  `VK_AMD_shader_info`, `VK_EXT_subgroup_size_control`, `VK_KHR_shader_atomic_int64`
  (buffer + shared), `VK_EXT_shader_atomic_float(2)`, `VK_KHR_shader_integer_dot_product`
  (8-bit packed accelerated), `VK_KHR_shader_clock` (subgroup + device clock),
  `VK_EXT_conditional_rendering`, `VK_EXT_device_generated_commands`, `VK_KHR_copy_memory_indirect`
  (graphics + compute queues), `VK_EXT_external_memory_host` (4096-byte alignment),
  `VK_EXT_global_priority(_query)`, `VK_EXT_memory_priority`, `VK_EXT_pageable_device_local_memory`,
  `VK_KHR_timeline_semaphore`, `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`,
  `VK_EXT_shader_float8`, `VK_EXT_descriptor_heap`, `VK_KHR_device_address_commands`.
- **Absent:** `VK_AMDX_shader_enqueue` (work graphs). Only `VK_AMDX_dense_geometry_format` is
  exposed from the AMDX family on this driver.
- **DGC is compute-capable but cannot switch pipelines:**
  `supportedIndirectCommandsShaderStagesPipelineBinding = None` and `...ShaderBinding = None`.

## Where the ~0.33 ms single-stream frame goes (derived from in-tree numbers)

Every input below is a number already in the tree. Only the subtraction is new.

- **Single stream:** `bench_multiqueue --import`, `ultrafast`, pre-recorded, 1 stream: 2,988-3,009 fps,
  i.e. **~333 us per frame** (PERFORMANCE.md 8g).
- **Host overhead:** pre-recording removed recording (22 us) and `begin_commands` (5 us), and import
  removes the 54 us memcpy (8d, 8e, 8f). The 81 us "host work outside submit_wait" measured in 8e was
  made of exactly those three, so with import plus pre-recording only **a few us** of host work remain
  per `Detect()`. This is an estimate: the 58 us figure in 8g was taken on the staged-copy path. Either
  way, the GPU-driver thread spends nearly the whole frame inside submit+fence.
- **GPU execution of one frame on one hardware queue:** 4 streams on family 0 saturate at **~4,600 fps,
  ~217 us per frame**. With several submissions queued back to back, host and launch latency are
  hidden, so ~217 us is the execution time of one frame's command buffer on one pipe, including its
  barrier drains.
- **Round-trip latency:** the difference, **333 - 217 = ~115 us per frame (~35%)**, is submit-to-start
  plus end-to-fence latency, which one in-flight frame cannot hide. 8e found the same quantity
  independently: 137 us of "submit to first span plus last span to fence return", of which only 42 us
  is an empty submit+fence round trip. 8e also showed that spinning on the fence does not recover it,
  so it is not OS wake-up latency. The two candidates are the GPU-side end-of-submission flush and
  submission scheduling (WDDM/KMD/firmware).
- **Idle GPU time inside the 217 us:** with queues on *both* compute families (`alternate`, 4 streams,
  import) the same GPU reaches **7,836 fps, ~128 us per frame** (8d). So roughly **90 us (~40%) of the
  217 us is idle hardware** that another frame can fill. That idle time consists of barrier drains
  (about 31 barriers at ~1.6 us each, ~50 us; section 3), tiny dispatches (about 27 dispatches and
  10 fills per frame, counted below), and latency-bound passes with low occupancy or long tails.
- **Summary:** at 3,000 fps single-stream the frame is about one third launch/completion latency,
  one quarter idle GPU inside the command buffer, and under half real shader execution. Host
  overhead is now small. Shader-level tuning (wave size, VGPRs) can only touch the last part.

**Dispatch count per fused frame** (read from `GpuDetector.cpp:671-852`, chunk = 2, `max_blobs`
auto = 2048 at 1280x800 d2 (`GpuDetector.cpp:166-174`), `scan_wg` 1024):
- Head: 9 `vkCmdFillBuffer`, then decimate, block_minmax, block_filter, threshold, uf_init,
  uf_compress, 2 x (uf_merge + uf_compress), plus one more fill. That is 10 dispatches.
- Tail: uf_final, label_pixels, blob_diff, build_qbp_args, hash_group, build_indirect_args,
  init_extents, reduce_extents, merge_extents, select_blobs, extract_blob_counts, 3 scan dispatches,
  scatter, build_sort_args and sort. That is 17 dispatches.
- Total: **27 dispatches, 10 fills, 6 small copies and about 31 barriers in one submission.** Every
  dispatch is followed by a full compute-to-compute barrier (`ComputePipeline::DispatchRaw`, line
  186-205), so the hardware never overlaps two of them.

## Findings

### F1. Prefer the compute-only queue family (family 1) by default on AMD discrete — RDNA4 (any AMD dGPU)
- **Mechanism:** `Context::CreateLogicalDevice` takes the *first* family with `COMPUTE_BIT`
  (`Context.cpp:468-470`). On this driver that is family 0, the graphics+compute ("universal") family.
  Its 8 queues behave like one hardware pipe. PERFORMANCE.md 8c, `ultrafast`: one queue per stream on
  family 0 gives nothing over sharing one queue (4 streams 3,852 vs 3,944 fps). On family 1
  (compute-only, the async-compute hardware queues) the same layout scales: 5,240 vs 3,793. The tree's
  data also favours family 1 for a **single** stream:
  - 8b (GPU pass only): 2,380 vs 2,293 fps, **+3.8%**.
  - 8c (with tail): `exact` 2,095 vs 1,687, **+24%**; `ultrafast` 2,121 vs 1,905, **+11%**.

  These come from different tables, different sessions and an older tree, and the clock-state caveat
  in 8g applies. Still, all three point the same way. A plausible mechanism (not verified): a dispatch
  and barrier on the compute micro-engine is cheaper than on the graphics ring, whose barriers and
  end-of-submission flushes cover graphics caches and stages as well.
- **Expected gain:** single stream +4% to +24% (in-tree, unreplicated). For multi-stream,
  `per-stream` on family 1 gave 5.2k against about 3.9k fps on family 0 (+35%, in-tree 8c).
- **Cost / risk:** a few lines: prefer a family with `COMPUTE` and without `GRAPHICS`, falling back to
  the first compute family, with `ContextOptions::queue_family` still overriding. Three checks:
  timestamps are valid on family 1 (64 bits, per vulkaninfo); `VK_EXT_conditional_rendering` must
  work on the compute queue (spec-legal, needs verifying with the existing bit-identity matrix);
  Mali has only one family, so nothing changes there. NVIDIA was not measured.
- **Prior art:** 8b/8c measured both families, but only as a multi-stream layout question. Nobody
  changed the single-stream default.
- **Verdict: pursue, after a confirming measurement:** `bench_multiqueue --import --streams 1
  --family 0` vs `--family 1`, ABBA over at least 12 rounds. Then `apriltag_vulkan_validate --profile`
  with the family forced, comparing `gpu.submit_wait` and the per-span table to see whether the gain
  is in spans (barriers) or in launch latency.

### F2. Why 4 streams give ~4,600 and not ~12,000 fps — RDNA4, multi-stream throughput
- **Mechanism:** as F1 explains, the default `per-stream` mode of `bench_multiqueue` puts every
  stream on family 0, which serializes. Every stream's frame is a strictly serial chain of about 27
  barrier-separated dispatches, so on one pipe the frames run one after another. The gain from 1 to 4
  streams (~3,000 to ~4,600) is purely hiding the ~115 us round trip (see the budget above). The
  hardware still has headroom: `alternate` (queues on both families) with import reaches 7,836 fps at
  4 streams and 7,688 at 8 (8d), about 2.6x one stream.
- **Expected gain:** at 4 streams, 4,600 to about 7,800 fps (+70%, in-tree 8d) just by
  defaulting multi-stream setups to `alternate`, or to queue-per-stream on family 1 (5.2k).
- **Cost / risk:** library-side, give `vk::Lane` allocation a policy of round-robin across family 1
  queues first, then family 0. The bench already implements `alternate`.
- **Why the remaining gap is not linear even then:** each frame contains about 50 us of barrier drain
  and several sub-5 us dispatches. With only 2 hardware pipes effectively in use, the pipes cannot
  overlap more than about 2 frames' bubbles. The 8-stream number (7,688) equals the 4-stream number,
  so the ceiling is the 2-pipe structure plus the PCIe ingest (decimate reads the imported frame across
  PCIe: +21 us of `threshold` span per frame, 8d). Whether more family-1 queues map to more hardware
  queues is driver-internal and unpublished. The 5.2k vs 7.8k gap suggests the compute family alone
  does not use all of the hardware.
- **Verdict: pursue** as a default policy for multi-camera use. Measure with `bench_multiqueue
  --import --streams 4|8 --mode per-stream --family 1` vs `--mode alternate`.

### F3. Split the ~115 us launch/completion latency with buffer markers before trying to remove it — RDNA4 (Windows)
- **Mechanism:** the largest single item in the single-stream budget is the ~115-137 us per frame
  that no GPU span covers (budget above; PERFORMANCE.md 8e). 8e's fence-spin experiment already
  rules out host wake-up. What is not known is how much lies **before** the GPU starts
  (submission, KMD, firmware scheduling, command fetch) and how much **after** it finishes
  (end-of-pipe flush, fence write, KMD). The driver exposes two tools that answer this with no
  shader changes:
  - `VK_AMD_buffer_marker`: `vkCmdWriteBufferMarkerAMD` with TOP_OF_PIPE at the start of the
    pre-recorded command buffer and BOTTOM_OF_PIPE at the end, into a small host-coherent buffer. A
    host thread spins on the mapped words with QPC stamps while `SubmitPersistentAndWait` blocks.
    This yields submit-to-start-seen, end-seen and fence-return on one host clock.
  - `VK_KHR_calibrated_timestamps` (exposed) maps the existing timestamp spans onto QPC exactly,
    instead of the tracer's current "pin the first span to the submit time" offset
    (PERFORMANCE.md 9).
- **What it would unlock:**
  - (a) If end-seen is well before fence return, wait on the marker for the readback and keep the
    fence only for buffer reuse. This needs an explicit COMPUTE to HOST barrier
    (`SHADER_WRITE` to `HOST_READ`) before the marker, because the current `HostReadBarrier`
    (`ComputePipeline.cpp:265`) covers only transfer writes and relies on the fence's implicit
    memory dependency for shader writes.
  - (b) If most of the gap is submit-to-start, the levers are system-level: Windows
    hardware-accelerated GPU scheduling on/off, the queue family (F1), and possibly SAM (F10).
  - (c) A GPU-side wait on a host-written flag (a pre-submitted command buffer whose first dispatch
    polls device-uncached memory, memory types 5 and 7) would take the KMD out of the per-frame path
    entirely. It is **not spec-guaranteed** (host writes during execution are not made visible
    without a submission), it occupies a queue, and it risks a TDR. This is a last resort, not a plan.
- **Expected gain:** none from the measurement itself. The ceiling for (a)+(b) is the ~70-95 us
  above the 42 us empty round trip, about 20-28% of single-stream frame time. That is an in-tree
  bound, and the achievable share is unknown.
- **Cost / risk:** about 40 lines, behind an env var, diagnostic only. Markers are AMD-only.
  Calibrated timestamps are portable (Mali g24p0 lacks them).
- **Prior art:** 8e (fence spin, rejected) and 08-vulkan-extensions.md section 14 (buffer_marker as
  "diagnostic only", in the four-submit context). That context is gone. With one submission per
  frame, the round trip is now the largest remaining non-shader cost.
- **Verdict: measure first** (cheap, high information). This decides whether any latency work on
  Windows is worth doing.

### F4. On RDNA4 the labelling span is critical-path (chain-depth) bound: bounded chains pay more here than on Mali — RDNA4 mainly, agnostic change
- **Mechanism:** `labelling` is **172 of ~330 us of summed GPU spans** on the RX 9060 XT (52%;
  PERFORMANCE.md 8e). On the Mali it is 27%. The parent array at decimation 2 is 1 MB, and the
  RX 9060 XT (Navi 44, 2 shader engines) should have about 4 MB of L2. RDNA4 doubled L2 to 2 MB per shader
  engine (8 MB on the 4-SE RX 9070 XT, per Chips and Cheese); 4 MB here is my inference for
  2 SEs. It also has 32 MB of Infinity Cache (AMD spec). So these passes cannot be bandwidth-bound: their time is
  dependent-load latency. `uf_init.comp` links every pixel to `i-1`, so after init a run of length
  L is a chain of depth L. The post-init `uf_compress` walks it with one dependent load per hop
  (`uf_compress.comp:25-29`).
  - Research 03's simulation on `grayimage.pgm` at d2 gives a **maximum chain depth of 199**
    (375 at d1) and a mean of 7.6.
  - On a wide GPU with 4,000 wave64s at d2 and about 1,024 resident, the dispatch's tail is set by
    its deepest chains, not by total work.
  - At roughly 100-200 ns per dependent L2/MALL hit, the RDNA3 Infinity Cache measured
    128-161 ns scalar and 150-199 ns vector; no RDNA4 numbers were found. A 199-hop chain is
    therefore **~20-40 us of pure serial latency in one dispatch**. That is my estimate.
  - Mali, by contrast, is throughput-bound across 4 cores and sees mostly the total-work term.
  - The segmented init from research 03 idea 2 (A8 in 00-SUMMARY: `parent[i] = max(run_start,
    i - x%K)`) cuts the **maximum** depth from 199 to 25 (K=16) or 13 (K=32), and the
    wave64-lockstep latency proxy by 79-84% (03's simulation). Its main payoff on RDNA4 is the
    critical path, which Mali barely has.
- **Expected gain:** on RDNA4, an estimated 15-40 us of the 172 us labelling span (5-12% of GPU
  frame time). Variant 2b, which drops the post-init compress, also removes one dispatch and one
  barrier (~3 us). These are estimates from simulation, not measurements.
- **Cost / risk:** owned by the device-agnostic agent (01/04). It is exact: links point left, so
  root = min index holds. The in-tree *unbounded* run-start scan lost on Mali (2.91 vs 2.57 ms,
  OPTIMIZATION_NOTES.md "Computing run starts directly"). A bounded K removes the O(L) per-thread
  tail that sank it.
- **Device-specific angle worth adding to that A/B:** on RDNA4 use `VK_KHR_shader_clock`
  (`shaderSubgroupClock = true`) for a one-off per-wave cycle histogram of `uf_compress` and
  `uf_merge`. It shows directly whether the span is a long tail of a few waves (critical path, so
  bounded chains win) or uniform (throughput, so they win less). Mali has no shader clock.
- **Verdict: measure first, on both devices.** RDNA4 is the device where this should show most
  clearly. Use the `labelling` span ABBA with K as a spec constant, at d1 and d2.
- Sources: https://chipsandcheese.com/p/amds-rdna4-architecture-video (RDNA4 L2 doubled per SE, no
  L1); https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving (RDNA3 Infinity
  Cache latency and power-state effects).

### F5. Wave32 vs wave64: low expected value for the hot passes; sweep only the low-occupancy ones — RDNA4
- **Mechanism (why the "wave32 for divergent code" rule mostly does not apply here):** no pipeline
  sets `requiredSubgroupSize` (`ComputePipeline.cpp:79-92`), so the LLPC heuristic chooses, and the
  driver's reported default is 64.
  - **Labelling passes:** the hot passes (`uf_compress`, `uf_merge`, `uf_final`) are latency-bound
    *at full occupancy*. Every pipeline reaches the 16-waves-per-SIMD cap with zero spill
    (PERFORMANCE.md 3). With the wave-slot count fixed, wave32 halves the lanes in flight per slot
    and doubles the number of waves (8,000 vs 4,000 at d2). The divergence it saves is modest.
    Research 03's lockstep model gives a mean deepest lane of 22.6 hops per wave64
    (90,335 wave-steps / 4,000 waves) against 14.2 per wave16, so wave32 lands around 18.
    Latency-bound time scales roughly with wave-steps divided by resident waves: about 18 x 8,000
    against 22.6 x 4,000, i.e. **wave32 would be ~1.6x slower** for these passes on this model.
  - **Low-occupancy dispatches** are different: `sort_points_local` (one 256-thread workgroup per
    selected blob, about 75 at d2, so about 300 wave64s on 64 SIMDs), `hash_group`, and the small
    `select`/`init`/`merge_extents`/scan dispatches. There the GPU is under-filled anyway, and
    wave32 can shorten the per-workgroup critical path (cheaper `s_barrier` rounds, less
    padding-lane waste in the sort network).
  - **2D streaming passes** (decimate, minmax, filter, threshold): keep wave64. AMD's guidance is
    wave64 with 8x8 tiles for 2D stores.
- **Expected gain:** labelling, none or negative (model estimate). Sort and small dispatches: a few
  us in total, since the sort span is ~70 us after A3 (0.135 x 0.51 per PERFORMANCE.md 3c and
  OPTIMIZATION_NOTES 15), and most of it is probably the largest blob's tail (F8).
- **Cost / risk:** chain a `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` behind an env var
  per pipeline. Exact for every shader except `reduce_extents_hash_subgroup`, whose uvec4 ballot
  handles both sizes but which must go through the bit-identity matrix.
- **Prior art:** 00-SUMMARY B2 and 08 #4 (proposed, never run). The full-occupancy argument above
  is new and lowers its priority.
- **Verdict: measure first, and narrowly.** First dump the wave size LLPC actually chose per
  pipeline (F9). Then sweep only `sort_points_local`, `hash_group` and the extents trio with ABBA on
  their spans. Do not spend time on the labelling passes.
- Source: https://gpuopen.com/learn/rdna-performance-guide/ (wave32/wave64 guidance; as cited in
  research/08 section 2).

### F6. Keep `selected[]` in VRAM for the GPU consumers and write a host copy alongside — RDNA4 (any dGPU)
- **Mechanism:** the RDNA4 fused path came from moving `selected_extents_buf_` and
  `line_fit_points_buf_` to `HostVisibleCached` system RAM (PERFORMANCE.md 3c). That shipped with
  a measured GPU-side price at d2: `extents` span **+32 us** (0.032 to 0.064) and `sort` **+8 us**.
  `selected[]` is not only written by `select_blobs.comp`; it is **read back across PCIe by three
  later shaders**:
  - `extract_blob_counts.comp:22` (`selected[i].count`);
  - `scatter_index_points.comp:53` (one 32-byte `selected[sel]` read *per boundary point*,
    18k-65k points);
  - `sort_points_local_body.glsl:86` (`selected[blob].count`, workgroup-uniform).

  A per-point read of system memory over PCIe costs on the order of 1 us of latency on an L2 miss.
  Whether these lines stay in GPU L2 for HOST_CACHED, non-`DEVICE_UNCACHED` memory on this driver
  is not documented. The change: `select_blobs` writes `e` twice, to a device-local `selected_dev[]`
  that the three consumers bind, and to the host-visible `selected[]` that `QuadDecode` reads. The
  extra write is at most `max_blobs` x 32 bytes, about 2.4 KB at d2. `line_fit_points_buf_` is
  write-only on the GPU (streamed, coalesced 8-byte records) and can stay where it is.
- **Caveat on the evidence:** the +32 us was attributed to the `extents` span (init, reduce,
  merge_extents), and none of those shaders touch either buffer. Either the timestamp attribution
  bleeds across the select boundary, or the cost has a different cause (allocation placement,
  cache policy). Re-profile at HEAD before building.
- **Expected gain:** up to ~40 us per frame of the ~217 us GPU frame if the cost is read-side
  (in-tree bound). Plausibly less.
- **Cost / risk:** one extra buffer and binding (about 20 lines). Exact, since the values are
  identical. Neutral on Mali, where both copies would be the same kind of memory, so gate it on
  `!unified_memory` or let it be harmless.
- **Prior art:** research/08 section 9 "Option A" warned about exactly this read-side cost, and Option
  A shipped anyway because the submit savings dwarfed it. Option B (`VK_KHR_copy_memory_indirect`,
  exposed on this driver for the graphics and compute families) is the heavier alternative. The
  dual write gets the same result with no extension.
- **Verdict: measure first:** `apriltag_vulkan_validate --profile` at HEAD, looking at the
  `extents`, `blob_scan`, `scatter` and `sort` spans. Then a throwaway experiment that allocates
  `selected_extents_buf_` as `DeviceLocal` and bypasses the `fused_submits_` gate
  (`GpuDetector.cpp:290-312`). This gives wrong host output, but bounds the GPU-side cost of this one
  buffer. ABBA, 16 rounds.

### F7. Fewer fills, dispatches and barriers in the frame: RDNA4 pays a fixed ~1.6 us drain per barrier plus launch per dispatch — RDNA4 mostly, agnostic change
- **Mechanism:** PERFORMANCE.md 3 measures a fixed ~1.6 us per barrier on RDNA4 (about 31 per frame,
  ~48 us). Each barrier also ends a dispatch, so the next one starts on an empty GPU and pays wave
  launch and ramp. With about 27 dispatches and 10 fills (budget above), many frame-tail dispatches
  are smaller than their own overhead: three 1-thread `build_indirect_args`, `init_extents`,
  `merge_extents`, `select_blobs`, `extract_blob_counts` and 3 scan dispatches over 2,048 entries
  (the whole `blob_scan` span measures ~10 us, PERFORMANCE.md 8 item 4). Concrete, exact
  consolidations, all agnostic code:
  - (a) Put the six 4-byte counters (`qbp_counter`, `selected_counter`, `uf_changed`,
    `raw_blob_counter`, `hash_drop_counter`, `oversized_sort_counter`) in one buffer at distinct
    offsets, so the clear is one `vkCmdFillBuffer` instead of six (`GpuDetector.cpp:689-697`). On
    AMD each fill is a separate CP DMA or internal blit; the driver chooses which, and that choice is
    unpublished.
  - (b) Zero `blob_size[i]` in `uf_init` (same index as `parent[i]`, coalesced) and drop the 1 MB
    fill. Its first consumer, `uf_final`, comes after many barriers.
  - (c) A single-workgroup scan when `max_blobs <= scan_wg * 4`, which covers the 2048 default at
    d2: one dispatch instead of three, and `extract_blob_counts` folded into its load.
  - (d) `merge_extents` folded into `select_blobs`, which then reads the copies and folds them
    itself.
- **Expected gain:** about 1.6 us of drain plus about 1-3 us of launch per removed boundary
  (in-tree barrier figure; launch cost estimated). (a)-(d) remove roughly 6-8 boundaries, for
  **about 10-25 us per frame, 5-10% of the ~217 us GPU frame**. That is an estimate. On Mali the
  same changes are smaller in relative terms there, but barrier cost scales with in-flight work.
- **Cost / risk:** owned by the agnostic agents (01/04). Each item is exact and a one-ABBA change.
  (b) adds 1 MB of writes to `uf_init`, which on Mali costs DRAM bandwidth (`clear` is 3.68x
  bandwidth-sensitive there), so it may be neutral on Mali.
- **Prior art:** PERFORMANCE.md 8 item 4 rejected right-sizing `extract_blob_counts` (~4 us total).
  That was about *capacity*, not dispatch count. The two pass fusions in PERFORMANCE.md 8 item 2 were
  *full-image* fusions that lost on bandwidth grounds and do not apply to these tiny dispatches.
- **Verdict: measure first,** starting with (a) and (c), which are the cheapest. Use the `clear` and
  `blob_scan` spans plus `Detect()` ABBA, and remember that the spans exclude barrier drains, so
  GPU total is the real check.

### F8. The sort span on RDNA4 is probably the largest blob's tail; bin large blobs to a 1024-thread variant — RDNA4
- **Mechanism:** `sort_points_local` runs one 256-thread workgroup per selected blob
  (`local_sort_cap_` is clamped to 256 at `GpuDetector.cpp:194-196`, although the device allows
  1024). At d2 a blob can hold up to 2048 points (`local_sort_virtual_cap_`), which means 66
  barrier-separated rounds with 4 comparators per thread per round. Small blobs finish in a few
  rounds. About 75 workgroups on 32 CUs means the dispatch's duration is set by the one or two
  biggest blobs while the rest of the GPU idles. Mean points per selected blob is ~240 at d2
  (145,568 readback bytes / 8 / ~75 blobs; PERFORMANCE.md 6b). The distribution's tail is unknown.
  OPTIMIZATION_NOTES 15's -49% sort win on RDNA (and nothing on Mali) fits a wide GPU whose time is
  dominated by per-round work in the long workgroups.
- **Option:** have `build_sort_args` (or `select_blobs`) split selected blobs into two indirect
  lists by `count > T` (for example T = 512), and dispatch a 1024-thread instance of the same shader
  for the large list. Same network, same output, so it is exact.
- **Expected gain:** unknown until the blob-size histogram and per-workgroup timing are measured.
  Ceiling: the sort span is ~70 us. If a single 2048-point blob is the critical path, 4x the threads
  could cut its rounds' work by about 4x, which bounds the gain at roughly 20-40 us (estimate).
- **Cost / risk:** moderate. It needs a second pipeline, a second indirect arg and a compaction.
  The order of `RawLineFitPoint` output is unchanged because blob offsets come from the scan.
- **Prior art:** 00-SUMMARY A4 (pack several small blobs per workgroup, Mali-motivated) is the
  other half of the same length binning, and was never measured.
- **Verdict: measure first:** dump the blob-size histogram (A4's own gate). On RDNA4, also take a
  `VK_KHR_shader_clock` (subgroup clock) per-workgroup duration histogram of `sort_points_local`
  to confirm a tail. Build only if one or two workgroups dominate.

### F9. One-off diagnostic: dump LLPC's per-pipeline wave size, VGPRs, LDS and spills on the Windows driver — RDNA4
- **Mechanism:** the "16 waves/SIMD, 12-48 VGPRs, zero spill" statement in PERFORMANCE.md 3 was
  most likely taken on RADV (that section's startup sample shows `RADV GFX1200`). The Windows
  driver uses a different compiler (LLPC). `VK_KHR_pipeline_executable_properties`
  (`pipelineExecutableInfo = true`) reports the dispatched `subgroupSize` and register statistics.
  `VK_AMD_shader_info` adds `numUsedVgprs`, LDS bytes, scratch and ISA disassembly. The ISA answers
  questions 08 left open: whether `atomicMin` in `doUnion` compiles to a no-return
  `buffer_atomic_umin`, whether workgroup-uniform reads (`selected[blob].count`,
  `changed_flag`) become scalar `s_load`, and whether `min(min(a,b),c)` becomes `v_min3`.
- **Expected gain:** none directly. It gates F5 and checks the occupancy claim on this driver.
- **Cost / risk:** about 60 lines behind an env var, plus `VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR`
  at creation. It must not be on by default, because it can change pipeline-cache keys.
- **Verdict: do it once,** before any F5 sweep. (Radeon GPU Profiler on Windows is the
  no-code alternative and also shows per-dispatch occupancy and barrier timing; it needs a frame
  boundary, i.e. `VK_EXT_frame_boundary`, which this driver exposes.)

### F10. Smart Access Memory / Resizable BAR: currently off on this PC, and not a lever for readbacks — RDNA4
- **Facts:** heap 2 is 256 MiB, so ReBAR/SAM is disabled in firmware. Even with it on, every
  `DEVICE_LOCAL | HOST_VISIBLE` type stays **uncached for CPU reads** (types 2 and 6), and the only
  host-cached type is system RAM (types 3 and 7). Readbacks therefore belong in system RAM either
  way, which is what the tree already does (PERFORMANCE.md 3c).
- **Upload:** the 1 MB frame already fits in the existing 256 MiB BAR. A direct
  write-combined CPU write into BAR VRAM would move the ingest cost from the GPU (import: +21 us of
  `decimate` reading across PCIe) to the CPU (a ~1 MB WC store burst over PCIe). The in-tree bound
  for "no GPU copy, host memcpy kept" is 7,687 fps at 4 streams, against 7,836 for import (8d).
  So BAR upload ties with import at best for throughput and is worse for single-stream latency.
- **Possible indirect effect:** with SAM on, the driver may place internal allocations (command
  chunks, descriptors) in CPU-visible VRAM, which could shorten command-processor fetch at submit
  time. This is unpublished and only worth a look if F3 shows the latency is submit-side.
- **Verdict: reject as a code change.** A firmware toggle plus `bench_multiqueue --import` ABBA is
  an optional system experiment, only after F3.

### F11. Work graphs (`VK_AMDX_shader_enqueue`) and device-generated commands — reject on this driver
- **Facts:** `VK_AMDX_shader_enqueue` is **not exposed** by 26.8.1; only
  `VK_AMDX_dense_geometry_format` is. `VK_EXT_device_generated_commands` is exposed with compute, but
  `supportedIndirectCommandsShaderStagesPipelineBinding = None`, so DGC cannot switch compute
  pipelines on the GPU. A DGC sequence can only issue dispatches for a fixed pipeline, which is what
  `vkCmdDispatchIndirect` plus `build_indirect_args.comp` already do.
- **Structural reason, beyond availability:** the frame is a chain of full-image passes separated
  by global barriers (labelling convergence, then boundary, grouping, extents, select, scan,
  scatter, sort). A work graph removes host round trips and lets producers feed consumers, but this
  frame has **no** host round trips left (one submission), and each stage needs all of the previous
  one's output. The only data-driven part, "run another labelling chunk if not converged", is
  already handled by speculative recording plus conditional rendering, and retries never occur
  in practice (8g). D3D12 work graphs exist on RDNA3+, but porting the detector to D3D12 is out of
  scope, and the work-graph benefit would be the same small one.
- **Prior art:** research/08 sections 14 and Conclusions (DGC and AMDX rejected).
  The new evidence here is the exact DGC binding limits on this driver.
- **Verdict: reject.**

### F12. Queue global priority and memory priority — RDNA4, defensive only
- **Mechanism:** `VK_EXT/KHR_global_priority` (with query) is exposed. A HIGH-priority compute
  queue only helps when something else is using the GPU (a game, a compositor, an encoder), and
  does nothing on an idle desktop. It may also get the queue a different hardware queue slot, which
  would interact with F1/F2. `VK_EXT_memory_priority` / `pageable_device_local_memory` only matter
  under VRAM oversubscription; the working set is ~41 MiB against 16 GiB.
- **Expected gain:** 0 on an idle machine (consistent with research/08 #8).
- **Verdict: reject for performance;** optional robustness for a desktop shared with other GPU work.

### F13. RDNA4-specific ALU features (packed math, integer dot, new atomics) — no tangible benefit
- **Packed 16-bit / 8-bit math:** the only ALU-heavy passes are decimate, minmax, filter and
  threshold, which together are ~13 us at d2 (`threshold` span; PERFORMANCE.md 8d), and they are
  load- and launch-bound, not ALU-bound. Nothing to gain.
- **Integer dot product** (`integerDotProduct4x8BitPackedUnsignedAccelerated = true`): decimation
  is point sampling, and min/max are not dot products. Covered in depth by agent 06; no use found.
- **Atomics:** the RDNA4 hot atomics are already native: 32-bit `atomicMin` in `doUnion`, and
  64-bit atomics exist but RDNA4 takes the subgroup extents path, which beat the atomic64 one
  (PERFORMANCE.md 6). Float atomics are rejected in-tree because order-dependent sums break
  bit-identity.
- **Subgroup ops:** cheap on RDNA, but the tree measured that aggregating bare counters loses on
  this card (`uf_final` -72% and `blob_diff` -24% when *removed*; PERFORMANCE.md 6). The surviving
  rule, "aggregate values by key, not counters", already holds.
- **WMMA / cooperative matrix and the ray accelerators:** see agent 06. This frame has no GEMM
  shape and no BVH shape, so nothing here suggests otherwise.
- **Verdict: reject.**

### F14. Two frames in flight per stream (double-buffered detector) — throughput only, RDNA4 mostly
- **Mechanism:** `Detect()` is synchronous: one submission, then a fence wait
  (`Lane::SubmitPersistentAndWait`, `Context.cpp:1136-1156`), and every buffer is single-instanced
  in `GpuDetector`. So frame N+1 cannot be queued until frame N's fence returns, and the ~115 us round
  trip is paid serially every frame. Two buffer sets with two pre-recorded command buffers, submitted
  ping-pong (ideally on two family-1 queues), would let one camera's next frame queue behind the
  current one. That is exactly what 2 streams do today.
- **Expected gain:** for a camera that delivers frames faster than ~3,000 fps, throughput up to
  about the 2-stream number (in-tree: 2 streams on one queue, `ultrafast`, ~3,850 fps in 8c; ~4,100
  `alternate`). Latency gain: none. Real cameras run at 30-240 fps, so for one camera this buys
  nothing in practice.
- **Cost / risk:** about 2x device memory, plus a ring of command buffers keyed by buffer set
  (the pre-recording cache already supports keys). It is the same thing as running two
  `GpuDetector`s on one `Context`, which `bench_multiqueue` already does.
- **Verdict: reject as library work.** Callers who need throughput should run several detectors
  on family 1 / `alternate` (F2).

### F15. Measurement hygiene specific to this card
- RDNA3/RDNA4 drop Infinity Fabric and shader clocks when work is sparse. Chips and Cheese measured
  RDNA3 Infinity Cache latency rising from 128 ns to 161 ns, and VRAM latency more than doubling,
  in the low state. A serial benchmark (validate: GPU idle during the CPU tail) therefore runs the
  latency-bound labelling passes in a different clock and fabric state than a pipelined or
  multi-stream one. That explains at least part of why `Detect()` (~0.5 ms in validate) differs so
  much from 1/fps in `bench_multiqueue` (~0.33 ms), and also 8g's ±20% between processes.
- For GPU-side A/Bs on this card, prefer `bench_multiqueue --streams 1 --import` (GPU kept busy) or
  many ABBA rounds. For host-side changes, use `gpu.detect - gpu.submit_wait` (8g). Adrenalin's
  "minimum frequency" tuning can pin clocks for an A/B session; use it only for measurement.
- Source: https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving

