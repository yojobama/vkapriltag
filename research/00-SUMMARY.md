# Research summary — all eight categories

> **Update, 2026-09-27: Tier 1 and part of Tier 2 measured on the RX 9060 XT**,
> on branch `perf/rdna4-pass` (`main` untouched). See
> `apriltags_vulkan/OPTIMIZATION_NOTES.md`'s "fourth pass" and items 11-16 for
> full writeups. Summary: **A1 and B1 shipped with large wins** (−4% and
> −20 to −27% GPU total respectively, and they compose). **A2 shipped
> neutral** (aimed at Mali; nil-to-mixed here). **A6, A9, A13, A14 measured
> and rejected or ruled unnecessary** - notably, three independent
> contention-mitigation ideas for the extents stage (lane-based copy
> selection, struct padding, and a barrier-removal change to `uf_merge`'s
> convergence flag) all lost, and one important scope correction: **this
> device runs the subgroup-aggregated `reduce_extents_hash` variant by
> default**, so several planned items (lane selection, `gx_sum`/`gy_sum`
> packing) only ever applied to the non-default scalar fallback. A3 (sort
> comparator enumeration) was bounded but not attempted - the real waste is
> 13-22%, not the ~40% estimated below, and no safe closed-form formula fell
> out for every round within budget. Nothing has been run on Mali.

Merged 2026-09-26 from the `## Conclusions` of `01`–`08` (219 source entries
in total). Each item cites the category file where its evidence lives.
Baseline: Mali-G610 GPU phase 3.76 ms (1280x800, decimation 2, ~1.2 ms CPU
tail), RX 9060 XT 0.66 ms.

**Nothing below has been measured on the Mali.** Two items have RDNA
measurements (A1, B3). Every other uplift is an estimate from the literature,
from the in-tree cost model (`560 us + 3.3 us/1k px`), or from one agent's
simulation, and says so.

## 0. Findings that change what should be measured first

**0.1 White-pixel connectivity differs from upstream.** Verified in this
session against `_deps/apriltag-src/apriltag_quad_thresh.c`.
- **Upstream:** `do_unionfind_line2` unions white (`v == 255`) pixels diagonally (`DO_UNIONFIND2(-1,-1)` / `(1,-1)`), so white is 8-connected and black 4-connected. The frc971/Team766 ancestor does the same (a modified BKE).
- **vkapriltag:** `uf_init` joins only the left neighbour and `uf_merge` only the down edge, so white is 4-connected. Two white regions touching only at a corner are one blob upstream and two here.
- **Why it hasn't been caught:** the corpus still matches libapriltag tag-for-tag, and the tree's bit-identity gate compares against vkapriltag's own baseline.
- **What to do:** diff the connected-component partition against upstream's on the CPU over the corpus and some adversarial images, and decide whether parity matters. If it does, frc971's 2x2-block variant becomes relevant again: 8-connected white, 4-connected "domino" black, 127 as singletons. That reopens the in-tree rejection "2x2 block methods don't apply". *(03)*

**0.2 Benchmark at PhotonVision's real parameters before optimising.**
PhotonVision uses `minClusterPixels 5`, critical angle 45°, `maxLineFitMSE 10`
and `min_white_black_diff 5`. vkapriltag's defaults are 24 / 0.98 / 24-ish.
With 5, many more blobs reach select, sort and fit, so 3.76 ms may understate
the real workload and shift which spans matter. *(01, 02)*

**0.3 Profile the Mali properly.** `PERFORMANCE.md`'s "no external profiler"
is wrong for this GPU. Before building any shader change:
- Run Arm's offline compiler (`malioc -c Mali-G610`) over every shader and flag work registers > 32 (half occupancy) or spills.
- Capture hardware counters on the Pi (Streamline/gatord or libGPUCounters), especially `LoadStoreUnitCyclesAtomicAccess` and `LoadStoreL2ReadBeats` for `reduce_extents_hash`.
- A ~50-line G610 atomic microbenchmark would settle the extents ideas below. No public Valhall atomic data exists. *(06, 07)*

## 1. Ranked options

Accuracy classes are as in `vkapriltag-speedup-survey.md`: **Exact** =
bit-identical output.

### A. Exact GPU changes, deployment target first

| # | Change | Stage | Expected (Mali unless noted) | Evidence / bound first | Src |
| --- | --- | --- | --- | --- | --- |
| A1 | **Path splitting in `find()`** (ECL-CC Jump4, plain non-`coherent` stores). Put `FIND_MODE` = naive / split / halve / jump2 behind a spec constant and A/B all four in one session | labelling | RDNA **measured** −17% labelling at d1, −2.7% at d2. Mali est. 1–5% GPU total | Jump2 (one store per find) may suit Mali's shared bus better; it was 18% faster on ECL-CC's only long-path graph. Atomic or `coherent` stores cost 12–55% | 03, 04 |
| A2 | **127 fast path in `uf_final` / `label_pixels`**. Skip ambiguous pixels: their label is always 0 when `min_cluster_pixels ≥ 2` | uf_final, label_pixels | 0.04–0.08 ms (1–2%) | Simulation: 127 is 24% of pixels at d2 and 45% at d1, and up to 73–87% of `uf_final`'s atomics. Change A is ~5 lines. Categories 03 and 06 reached it independently | 03, 06 |
| A3 | **Enumerate active comparators in `sort_points_local`** rather than guarding slots. Today every 16-wide warp is half empty for p < 16 | sort | 1–4% GPU total | Simulation: 0.57–0.64x warp executions. The same network means bit-identical output. ~15 lines; one A/B | 05 |
| A4 | **Pack several small blobs per sort workgroup** (length binning) | sort | 2–4% GPU total, *if* most blobs are ≤ 128 points | Dump the blob-size histogram. Run a wrong build that skips blobs ≤ 64 points to get the ceiling | 05 |
| A5 | **Drop `shared wg_changed` + 2 `barrier()`s in `uf_merge`**. Use `if (merged && changed_flag == 0u) atomicOr(...)` | labelling | est. 2–10% labelling | Arm BP 3.4 §9.2–9.3 anti-pattern. **Caution:** the shared aggregation replaced a per-pixel global-atomic flood (see the shader comment). The read guard should prevent that flood, but measure. One-line A/B | 07 |
| A6 | **Extents contention, in order:** (a) pad `MinMaxExtentsGpu` 32 → 64 B, so two blobs stop sharing a line; (b) choose the copy by lane (`gl_LocalInvocationID.x & 7`), not by workgroup; (c) per-shader-core copies via `gl_CoreIDARM` (`VK_ARM_shader_core_builtins`, on g24p0). Map the sparse core IDs 0/2/16/18 through `shaderCoreMask`, not `& 7`; (d) non-power-of-two copy stride | extents | ceiling ~0.16 ms (~5%): today 0.221 ms vs a 0.062 ms plain-store bound | Take the counter capture (0.3) first to tell atomic-issue-bound from line migration. (a) and (b) are one-line changes | 06, 07, 08 |
| A7 | **Pack `gx_sum` / `gy_sum` into one biased int64 `atomicAdd`** | extents | 0.02–0.03 ms | The in-tree rejection A8 was only about 32-bit width; the int64 path has since shipped. Tally the diagonal-point fraction first | 06 |
| A8 | **Bounded segmented run-start init**: `parent[i] = max(run_start, i − x%K)` with K = 8/16/32, optionally dropping the post-init compress | labelling | est. −5 to −15% labelling | Simulation: post-init compress work −77 to −84%, chain depth 7.6 → 1.05–1.5. Not the rejected unbounded scan, but init gets costlier and may still lose on Mali. Fallback: inline compression in `uf_compress` | 03 |
| A9 | **Skip all-127 tiles.** Threshold writes one bit per tile; labelling and boundary workgroups exit on it. Compounds with D1 | labelling, boundary | proportional to the uniform-tile fraction | Log the all-127 tile fraction at `mwbd` 5 and 20. Build only if above ~30–50% | 01, 02 |
| A10 | **Lock-step two-chain walk in `doUnion`**: two independent loads in flight instead of two serial chains | labelling | unknown; the only idea aimed at load latency | `UNION_MODE` spec constant alongside A1. Drop if < 3% of `uf_merge` | 04 |
| A11 | **Merge writes the next `uf_compress`'s indirect args** (the Mali stand-in for conditional rendering) | labelling | ~0.7% ceiling | One A/B | 08 |
| A12 | **Delete the blob-offset scan chain**: one `atomicAdd` range allocator in `select_blobs` | blob_scan | ~10 µs (0.3%) | Read the Mali `blob_scan` span first | 05 |
| A13 | **Deterministic tie-break for equal `theta_key`** in the sort | sort | 0 (costs a few µs) | Removes the documented ±1–2 candidate-quad jitter. Count ties first | 05 |

### B. Desktop (RDNA4) only

| # | Change | Expected | Evidence / bound | Src |
| --- | --- | --- | --- | --- |
| B1 | **Turn on the single-submission frame tail.** Put `selected_extents_buf_` and `line_fit_points_buf_` in host-cached *system* memory (type 3), or copy them out with `VK_KHR_copy_memory_indirect` | ≥0.13 ms of 0.66 (3 boundaries × 42 µs) | Verified this session: no device-local + host-cached type exists (type 2 = `0x7` uncached BAR, type 3 = `0xe` system). Force the type, set `fused_submits_`, ABBA vs `APRILTAG_VK_FUSE_SUBMITS=0`. Re-opens "submit collapsing on dGPU": that rejection predates the device-side sizing that now ships | 08 |
| B2 | Per-pipeline wave32/wave64 sweep (`requiredSubgroupSize`), after dumping choices and VGPRs with `pipeline_executable_properties` | unknown | env-gated create-info, ABBA | 08 |
| B3 | Conditional rendering for the converged `uf_compress` | **measured** −3% labelling | already built, uncommitted | survey A2 |

### C. System and ingest (largest end-to-end effect once inside PhotonVision)

| # | Change | Expected | Risk / bound | Src |
| --- | --- | --- | --- | --- |
| C1 | **Hardware MJPEG decode + zero-copy.** Pipeline: MPP `mjpeg_rkmpp` → NV12 dma-buf pool → import each fd once as a storage buffer → `decimate` reads the Y plane (new `Detect(fd, offset, pitch)`). Jellyfin already does MPP → Mali OpenCL import on this SoC | several ms of A76 decode per frame, plus the 0.22 ms upload | Vendor (BSP) kernel only: mainline has no RK3588 JPEG decoder. Storage-buffer dma-buf import on libmali is **unproven** (only OpenCL is). **Don't** import UVC vmalloc buffers: no cache maintenance, so a silent stale-data hazard. Bound: `cv::imdecode` time vs `ffmpeg -hwaccel rkmpp`, then `vkGetPhysicalDeviceExternalBufferProperties` | 01, 07, 08 |
| C2 | **Frequency governors**: `performance` on the A76 policy and the Mali devfreq | rapidtag measured 123 → 211 fps on a QCS6490 | none; A/B | 01 |
| C3 | **Stream encoder and frame handling**: latest-frame mailbox; pin the encoder away from the CPU-tail cores | PhotonVision's encoder alone cost ~25% throughput on a Pi 5 (80.5 → 60.1 fps) | measure end to end with streams on and off | 01 |
| C4 | **Driver hygiene**: A/B libmali g24p0 vs g29p1; re-measure PanVK on Mesa ≥ 25.2 (the 13x-slower result was Mesa 25.0, likely missing int64 atomics and 8-bit storage) | unknown | ~10 min each | 07 |
| C5 | **Overlap frames on libmali's second queue** | throughput only | Bound with no code: run two validator processes at once | 07 |
| C6 | **NEON decimate + threshold on a spare A76** (frc971 did this because "the GPU was too full") | at most the Mali threshold span | +1 frame latency; frc971's code handles decimation 2 only. Time it on one pinned A76 | 01, 06 |
| C7 | Pin CPU-tail workers to the A76s; bump the libapriltag pin past v3.4.5 (no `sched_yield`, pigeonhole decode) | small, CPU side | — | 01, survey B4 |

### D. Recall and parameter trades (change what is found)

| # | Change | Expected | Risk | Src |
| --- | --- | --- | --- | --- |
| D1 | **Expose and sweep `min_white_black_diff`** (5/10/15/20). 971 runs 20 ("a massive speedup") | unknown, plausibly large for every stage after threshold | Output identical to libapriltag *at the same value*; costs low-contrast recall | 01 |
| D2 | **Field-map-predicted ROIs** with a full-frame fallback in the same frame, packed into one atlas | 2–3x on tracked Mali frames (EagleEye: 3.48 vs 20.6 ms on a Pi 5 CPU) | Exact per found tag (traced in upstream); misses tags outside ROIs. Bound exactness on CPU with aligned crops; recall with EagleEye's 7-clip dataset | 02 |
| D3 | **API: per-frame `(crop list, decimation)`**, policy stays in the caller | enabler for D2 and D4 | keeps the parity gate intact per crop | 02 |
| D4 | Per-frame / per-ROI decimation | decimation 4 = 2.1x vs 2 (measured) | range loss; EagleEye saw little gain from adaptive over static | 02 |

## 2. Rejected with new evidence

- **NPU offload of any stage:** YOLOv8n takes about 13.6 ms and YOLO11 about 23 ms on the RK3588 NPU, against 3.76 ms for the whole Mali GPU phase. NVIDIA's own PVA backend loses to its CPU path. *(01, 02)*
- **Learned detectors:** slower, with about 3x worse corners. **Edge, voting and optical-flow front ends:** slower, and optical-flow corners aren't exact. *(02)*
- **Decoupled look-back scans:** no forward-progress guarantee; the Mali-G77 fails LOBE; a hang is `DEVICE_LOST`. **Decoupled fallback:** only wins at 2^25 elements. **Global radix sorts:** unsupported at subgroup size 16. *(05)*
- **Graph-CC alternatives:**
  - Round-synchronous methods (SV, FastSV, Liu-Tarjan): pass count grows with image diameter.
  - Random or rank linking: breaks root = min index.
  - Lock-based hooking: needs forward progress.
  - Afforest's skip-the-giant-component: already covered by `uf_init`. *(04)*
- **Other hash-table schemes:** linear probing with one CAS is confirmed optimal at 1–2% load. Cuckoo and Robin Hood need forward progress. *(06)*
- **Alternative adaptive thresholds:** all break bit-identity and cost more. *(06)*
- **Extensions:** DGC, descriptor buffer, push descriptors, maintenance5–9, cooperative matrix, float atomics, `VK_ARM_scheduling_controls` (dispatch controls not exposed), ARM tensors/data_graph (absent). *(08)*

## 3. Corrections to the category files

- **`MinMaxExtentsGpu` is 32 bytes**, per the `static_assert` in `Types.h`. `08` says 48, from an older `OPTIMIZATION_NOTES` figure. So two blobs share each 64-byte line, which strengthens A6(a).
- **`APRILTAG_VK_MIN_TAG_PX` is committed** (`b0e38ac`), not a working-tree change as `02` says. The only uncommitted code is the conditional-rendering work (B3).
- **Two ideas were found independently by two categories:** A2 (03, 06) and A6 (06, 07, 08).

## 4. Still unverified (would change rankings)

- **Missing Mali data:**
  - Any measurement of A1–A13.
  - Whether Mali shared-memory cost is per cache line or per warp instruction (sizes A3).
  - The blob-size histogram (A4).
  - The 127 and all-127-tile fractions on real FRC frames (A2, A9).
- **Driver facts to check on the Pi:**
  - g29p1 extension exposure (strings only).
  - libmali dma-buf import as a storage buffer.
  - `shaderCoreMask`.
  - `queueCount` on g24p0.
- **Performance gaps:**
  - A76 software MJPEG decode time.
  - PanVK performance on current Mesa.
- **Correctness:** whether the white-connectivity gap (0.1) changes any detection.
