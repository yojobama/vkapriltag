# Performance and portability notes

Targets: embedded ARM Mali-G610 (the deployment target), discrete AMD
(Radeon RX 9060 XT / RDNA4), and discrete NVIDIA. The pipeline needs **only
core Vulkan 1.1**. Optional extensions (8-bit storage, 64-bit atomics,
conditional rendering, host-pointer import, dma-buf import) and subgroup
operations are used when the device has them, and every one has a fallback
that gives byte-identical detections (section 8n); the `APRILTAG_VK_FORCE_NO_*`
variables in section 2 force each fallback.

This file covers configuration, per-device behaviour, and how to measure.
For the history of what was tried and what was rejected, with numbers, see
`OPTIMIZATION_NOTES.md` — that file is the record of *why* the pipeline looks
like this; this one is the record of *how to run and measure it*.

## 1. First: make sure you are actually on the GPU

This is by far the largest single performance factor, and it is a
configuration issue rather than a code one.

If no GPU Vulkan driver (ICD) is installed, the Vulkan loader happily hands
back Mesa's **lavapipe/llvmpipe** software implementation. It is functionally
correct and 50-500x slower, which looks exactly like "the GPU port is slow".

The detector **refuses a CPU device by default** and explains what to do:

```
The only Vulkan device available is a CPU/software implementation ...
  * To fix properly: install the Vulkan driver (ICD) for your GPU ...
  * To benchmark or test on the CPU anyway: set APRILTAG_VK_ALLOW_CPU=1
```

Startup always prints the selected device, its relevant limits, and the launch
geometry derived from them. Check that line first. Real output from the two
machines these notes were measured on:

```
apriltag_vulkan: using AMD Radeon RX 9060 XT (RADV GFX1200) (discrete GPU, Vulkan 1.4.354)
  limits: maxComputeWorkGroupInvocations=1024, ..., float64=yes, int64=yes, 8bit_storage=yes,
          subgroup_ballot=yes, subgroup_arithmetic=yes
  chosen geometry: wg1d=256, wg2d=16x16, scan_wg=1024, unified_memory=no, host_cached=yes, timestamps=yes
```

```
apriltag_vulkan: using Mali-LODX (integrated GPU, Vulkan 1.2.165)
  limits: maxComputeWorkGroupInvocations=1024, ..., float64=no, int64=no, 8bit_storage=yes,
          subgroup_ballot=yes, subgroup_arithmetic=yes
  chosen geometry: wg1d=128, wg2d=8x8, scan_wg=1024, unified_memory=yes, host_cached=yes, timestamps=yes
```

Note the Mali line: `float64=no, int64=no`. Nothing in the pipeline requires
either, and that is deliberate — see item 8 in `OPTIMIZATION_NOTES.md`.

Sanity checks: `vulkaninfo --summary` should list your GPU; on Linux you need
a working `/dev/dri`.

## 2. Environment knobs

All are optional; defaults are chosen from device limits. This table lists
every variable the library actually reads — if a knob is not here, it does not
exist (`APRILTAG_VK_SORT`, documented in older revisions of this file, was
removed along with the radix/bitonic sorts it selected).

| Variable | Effect |
| --- | --- |
| `APRILTAG_VK_ALLOW_CPU=1` | Permit a software Vulkan device (benchmarking/CI only). |
| `APRILTAG_VK_DEVICE=<n>` | Pick a physical device by index instead of by score. |
| `APRILTAG_VK_VALIDATION=1` | Enable `VK_LAYER_KHRONOS_validation` if installed. |
| `APRILTAG_VK_WG=<n>` | Override the 1D workgroup size (power of two, clamped to limits). |
| `APRILTAG_VK_MAX_INVOCATIONS=<n>` | Pretend the device caps workgroups at `n`. Testing aid: reproduces a constrained part's launch geometry on desktop hardware. |
| `APRILTAG_VK_MAX_POINTS=<n>` | Cap boundary points per frame. The main device-memory lever - see below. |
| `APRILTAG_VK_UF_CHUNK=<n>` | Labelling iterations issued per convergence check (default 2); only with `APRILTAG_VK_UF_VERIFY=1`. |
| `APRILTAG_VK_UF_VERIFY=1` | Run a second `uf_merge` that checks the labelling converged, with the retry path, as in 1.8.0 (one merge joins every component, so it is off by default). See section 8t. |
| `APRILTAG_VK_UF_INIT_SCAN=0` | Initialise `parent[]` with `uf_init` and a compression instead of `uf_init_scan`'s workgroup max-scan. See section 8t. |
| `APRILTAG_VK_FUSED_THRESHOLD=0` | Keep decimation, block min/max, block filter and threshold as separate passes. See section 8t. |
| `APRILTAG_VK_MERGE_X4=0\|1` | `uf_merge` with 4 pixels per thread (default: on integrated GPUs only). See section 8t. |
| `APRILTAG_VK_DISPATCH_TIMES=1` | Print the median GPU time of every dispatch when the detector is destroyed (one detector per process). See section 9. |
| `APRILTAG_VK_GRAY_MAPPED=0` | On discrete GPUs, upload frames through a staging buffer and a GPU copy instead of writing them into mappable device memory. See section 8u. |
| `APRILTAG_VK_MERGE_BAND=<n>` | Merge rows in bands of `n` (a power of two) before the band boundaries; `0` merges in one pass (default: 8 on discrete GPUs, 0 on integrated). See section 8u. |
| `APRILTAG_VK_REDUCE_X4=0` | One boundary point per thread in the atomic extents reductions instead of 4. See section 8u. |
| `APRILTAG_VK_SORT_RANK_ALL=<n>` | Blobs of at most `n` points are sorted by ranking every key in one pass (default 256; `0` disables). See section 8u. |
| `APRILTAG_VK_COUNT_RUNS=0\|1` | Count blob sizes per run piece and find roots in two hops (default: integrated GPUs). See section 8u. |
| `APRILTAG_VK_MIN_TAG_PX=<n>` | Geometric prefilter: minimum tag size in pixels. |
| `APRILTAG_VK_PRERECORD=0` | Pre-recorded command buffers are on by default for the fused single-submission frame; `0` records every frame instead. See sections 8f and 8g. |
| `APRILTAG_VK_UF_INIT_RUN=<n>` | Pixels of a row each `uf_init` thread initialises (default 4). See section 8q. |
| `APRILTAG_VK_THRESH_RUN=<n>` | Pixels of a row each `threshold` thread covers (default 4). |
| `APRILTAG_VK_DEC1_ALIAS=0` | At decimation 1, copy the grey image into the decimated buffer instead of reading it in place. |
| `APRILTAG_VK_SORT_CHUNK=<n>` | Rank-merge sort run length (a power of two, default 16); `0` selects the Batcher sorting network. See section 8k. |
| `APRILTAG_VK_SORT_WG=<n>` | Workgroup size of the point sort (default: the device maximum, at most 256 on integrated GPUs and 1,024 on others). See section 8w. |
| `APRILTAG_VK_SORT_SPLIT=<n>` | Blobs of more than `n` points are sorted in runs of `n` by several workgroups and merged (a power of two; default 1,024 at decimation 1, off otherwise; `8192` turns it off). See section 8r. |
| `APRILTAG_VK_PROBE_MARGIN=<grey levels>` | Margin gate before edge refinement (default off); overrides `TagDecoder`'s `probe_margin`. See section 8l. |
| `APRILTAG_VK_PROBE_STATS=1`, `APRILTAG_VK_PROBE_CSV=<file>` | Per-quad probe histograms and one CSV row per quad (analysis aid). |
| `APRILTAG_VK_TRACE=<file>` | Write a Chrome-trace JSON of the CPU and GPU spans. See section 9. |
| `APRILTAG_VK_QUADFIT=dp\|peaks` | Corner-seeding method. `dp` is the default. |
| `APRILTAG_VK_DP_PEAKS_CHECK=1` | Keep a DP-seeded quad only if the peak search also accepts its blob (`DetectorConfig::dp_peaks_check`; off by default). See section 8z. |
| `APRILTAG_VK_GPU_QUADFIT=1` | Run QuadDecode's DP fit on the GPU (`fit_quad_dp.comp`; needs fp64, so discrete GPUs only; off by default). See section 8z. |
| `APRILTAG_VK_REFINE=exact\|fast\|ultrafast\|upstream` | Edge-refinement implementation; only consulted when the caller sets `td->refine_edges`. See section 4. |
| `APRILTAG_VK_FORCE_NO_SUBGROUP=1` | Force the scalar shader variants even where subgroup ops are available. |
| `APRILTAG_VK_FORCE_NO_8BIT=1` | Force the 32-bit-per-pixel shader variants even where `VK_KHR_8bit_storage` is available. |
| `APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1` | Force the 32-bit-atomic extents reduction even where `VK_KHR_shader_atomic_int64` is available. See section 3b. |
| `APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING=1` | Skip converged `uf_compress` passes with the in-shader guard even where `VK_EXT_conditional_rendering` is available. See section 6c. |
| `APRILTAG_VK_FUSE_SUBMITS=0` | Keep the four-submit frame even where the tail could be recorded as one. See section 3c. |
| `APRILTAG_VK_WG2D=<w>x<h>` | Override the 2D workgroup size. |
| `APRILTAG_VK_TIMESTAMPS=1` | Emit the per-dispatch GPU timestamp-span breakdown. |
| `APRILTAG_VK_PIPELINE_CACHE=0` | Disable the on-disk pipeline cache. |
| `APRILTAG_VK_CACHE_DIR=<path>` | Where that cache lives. |
| `APRILTAG_CPU_THREADS=<n>` | Parallelism for the CPU tail, including the calling thread. Default = `hardware_concurrency()`; 1 = serial. |

## 3. Measured performance

The 1280x800 corpus image (`tools/make_corpus`), one tag, 75 candidate quads, decimation 2,
otherwise default config, medians over 300 to 600 iterations; Orange Pi 5 with the governors
pinned to `performance` (section 8i) and LumenVision stopped, `Release`; desktop `RelWithDebInfo`.
The RX 9060 XT alternates between two clock states from one process to the next, so its GPU rows
are a range. The per-stage GPU table, the throughput tables and the comparison with 1.6.0 are in
`README.md` and sections 8k to 8m.

| Stage | Runs on | Orange Pi 5 (Mali-G610) | Desktop (RX 9060 XT) |
| --- | --- | --- | --- |
| `GpuDetector` | GPU | 2.04 ms | 0.38 to 0.55 ms |
| `quad_decode` | CPU | 0.19 ms | 0.11 ms |
| `tag_decode` | CPU | 0.63 ms | 0.22 ms |
| **Pipeline total, serial** | GPU + CPU | **2.87 ms** | **0.77 to 0.91 ms** |
| **Pipeline total, `FramePipeline`** | GPU \|\| CPU | **2.2 ms** | **0.40 to 0.46 ms** |

The GPU phase dominates on the deployment target (71% of a serial frame) and
is essentially all real shader execution. Two things it is *not*:

- **Not submit-bound.** An empty `vkQueueSubmit` + `vkWaitForFences` round trip
  measures **19 us on Mali** and **42 us on the RX 9060 XT** — the Mali is the
  *cheaper* of the two, having no PCIe hop. A frame is one submission on parts
  with host-cached readback memory (section 3c), so submit cost is under 1% of a
  Mali frame; collapsing the remaining boundaries was tried and is not worth it
  (`OPTIMIZATION_NOTES.md`).
- **Not barrier-bound on the discrete part.** A barrier costs a fixed ~1.6 us
  on RDNA4 regardless of in-flight work, and is free when there is nothing to
  drain; all ~31 of a frame's barriers total ~48 us. On Mali the cost instead
  *scales* with in-flight work (2.6 us between trivial dispatches, 18.7 us
  between large ones), so barrier count matters there and only there.

What the Mali GPU phase *is* bound by is memory traffic across its full-image
passes (ten now that `label_pixels` is gone). Sweeping the input over 320x200 /
640x400 / 1280x800 gave a consistent marginal rate (measured before the
section 8k stage changes, which lowered both terms; the shape holds):

```
GPU time ~= 560 us fixed + 3.3 us per 1k source pixels
```

so 86% of it is pixel-proportional at full resolution. That also makes
**decimation the largest single lever available**: `--decimation 4` measures
**2.1x faster** than the default 2 on the Pi and still decodes correctly, at
the cost of detection range (candidate quads drop from 75 to 17 on this
image). It is a range/throughput trade, not a free win. Note that width and
height must each divide evenly by the decimation factor.

Occupancy is not a limiter anywhere: on the RX 9060 XT every pipeline reaches
the 16-waves/SIMD cap with 12-48 VGPRs and zero scratch spill.

## 3a. How much of the GPU phase is DRAM bandwidth? 15%

Measured directly, by pinning the memory controller with the `userspace`
devfreq governor and sweeping it, interleaved, on the Orange Pi 5 Plus
(`grayimage.pgm`, 1280x800, decimation 2, GPU pinned at 1 GHz):

| DMC | GPU total | `clear` | `label_pixels` | `uf_final` | `labelling` | `boundary` | `extents` | `sort` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 528 MHz | 4.980 | 0.177 | 0.361 | 0.242 | 1.198 | 0.394 | 0.436 | 0.392 |
| 1068 MHz | 3.924 | 0.099 | 0.195 | 0.129 | 0.979 | 0.372 | 0.433 | 0.375 |
| 1560 MHz | 3.574 | 0.062 | 0.143 | 0.099 | 0.938 | 0.364 | 0.428 | 0.434 |
| 2112 MHz | 3.457 | 0.048 | 0.116 | 0.091 | 0.921 | 0.361 | 0.432 | 0.397 |
| **528/2112 ratio** | **1.44x** | **3.68x** | **3.12x** | **2.66x** | 1.30x | 1.09x | **1.01x** | **0.99x** |

Fitting `t = C + K/f` to the endpoints gives `C = 2.95 ms` clock-independent
and `K/f = 0.51 ms` at 2112 MHz. So at the deployment memory clock:

> **Only ~15% of GPU time is DRAM-bandwidth-proportional.** Eliminating
> *every* byte of memory traffic would take the GPU phase from 3.46 ms to
> about 2.95 ms. Halving traffic buys ~7%.

This is the number to check any "reduce memory traffic" proposal against, and
the per-span ratios say where such a proposal could possibly pay:

- **Worth targeting:** `clear` (3.68x — it is almost pure DRAM writes,
  `vkCmdFillBuffer` over `blob_size_buf_` and the hash table),
  `label_pixels` (3.12x) and `uf_final` (2.66x). Together 0.26 ms.
- **Provably not worth targeting:** `extents` (1.01x) and `sort` (0.99x) do
  not move *at all* across a 4x bandwidth range, despite being the second and
  third largest spans. They are atomic- and latency-bound. Any traffic
  optimization aimed at them is dead on arrival, which is measured, not
  argued.
- `labelling`, the largest span at 27%, is only 1.30x — mostly dependent-load
  *latency* (chasing `parent[]` chains), not bandwidth. That is the same
  conclusion `OPTIMIZATION_NOTES.md` item 8 reached from the other direction.

**A caution on the governor.** The board these notes were taken on idles at
`dmc_ondemand` / 528 MHz, and an idle reading invites the conclusion that
pinning the memory controller is worth ~32%. It is not: `dmc_ondemand` ramps
to 2112 MHz under sustained load, and measured head-to-head, steady-state
throughput is identical (3.152 ms ondemand vs 3.156 ms pinned). What pinning
buys is the *variance* `OPTIMIZATION_NOTES.md` describes — the first frames
of a burst run at the low clock while the governor catches up.

## 3b. The extents reduction: contention, not bandwidth

Section 3a's per-span table has one entry that reads as a dead end and was
actually the biggest win available: `extents` at **1.01x**. A span that does
not move at all across a 4x memory-clock range, while being the second
largest in the frame, is not waiting on memory *bandwidth*. It is waiting on
atomics.

`reduce_extents_hash.comp` issues up to eight atomics per boundary point into
that point's raw-blob accumulator. At decimation 2 that is ~65k points over
388 blobs - about **170-way contention on every counter**. Replacing the
atomics with plain stores (a deliberately incorrect build, to bound the
payoff) took the span from 0.434 to 0.062 ms.

Three changes close most of that gap. All are bit-identical:

**1. Read before the atomic, for min/max.** `atomicMin` with a value already
>= the stored one is a no-op, so testing first is exactly equivalent. A
blob's extents stop moving after its first handful of points, so nearly all
four of those atomics are skipped. Worth **-28%** of the span on its own.

Note this is the *opposite* of what section 6's "contention relief does not
compose" warns about. That warning is about stacking a guard on top of
subgroup aggregation; here there is no aggregation to stack on, because
integrated parts take the scalar path.

**2. Privatize the accumulator 8 ways.** Each workgroup accumulates into its
own copy of the extents array and a new `merge_extents.comp` folds the copies
back before anything reads them, cutting contention by the replication
factor. Only the first 4096 blob indices are replicated - far more than any
real frame produces - so the memory cost is ~1 MiB rather than 8x3 MB, and
blobs past that fall back to the shared slot, correct but contended.

Measured on Mali: K=2 -13.1%, K=4 -18.5%, **K=8 -20.3%**, K=16 -20.0%,
K=32 -20.6% (GPU total, against an unprivatized build). K=8 and K=16 are
within noise of each other head to head, so K=8 wins on memory.

**3. `VK_KHR_shader_atomic_int64`, where available.** After the first two the
span is at the atomic *throughput* floor, not the contention floor, so the
remaining lever is issuing fewer of them. `count` and `pxgx_plus_pygy_sum`
are the only two fields every point touches, and `MinMaxExtentsGpu` now
places them in one naturally aligned 64-bit word, so one `atomicAdd` does
both. This is exact, not approximate: the low half (`count`) cannot carry
into the high half, being bounded by the boundary-point capacity, and the
high half accumulates modulo 2^32 exactly as the 32-bit atomic did. Nothing
unpacks afterwards - the two struct fields *are* the two halves.

`reduce_extents_hash_atomic64.comp` is selected when the device reports the
extension plus `shaderBufferInt64Atomics` plus core `shaderInt64`;
`reduce_extents_hash.comp` is the unconditional fallback. On the Mali-G610,
A/B'd on one binary with `APRILTAG_VK_FORCE_NO_INT64_ATOMIC`:
**`extents` 0.221 ms with, 0.252 ms without — -12 to -13%** on the span,
~1% of the frame, reproduced across two sessions.

### Net effect on the deployment target

Stock `7587f1b` against everything in the third pass, on the Mali-G610,
ABBA-interleaved, min of 12, **both binaries built `Release`**, timestamps
off:

| | GPU total | `pipeline_total` | device memory |
| --- | --- | --- | --- |
| decimation 1 | **-13.9%** | -12.5% | 185 -> 155 MiB |
| decimation 2 | **-14.2%** | -11.2% | 48 -> 42 MiB |
| decimation 4 | **-15.4%** | -15.5% | 15 -> 13 MiB |

Detections are identical to stock throughout: 3 decimations x 5
configurations on Mali, and 8 configuration axes x 3 decimations x the
5-image corpus on the RX 9060 XT.

A caution on which numbers these are. They were taken with
`APRILTAG_VK_TIMESTAMPS` unset. The instrumentation is not free - on this
part it costs about 9% of GPU total (3.16 ms against 2.90 ms on the same
build) - so the per-span tables above and the totals here are not measuring
the same configuration, and the spans should be used for attribution
rather than quoted as deployment performance.

## 3c. One submission for the frame's tail

Section 3a's profile has a line that is not a span at all. Between the
timestamp that ends one submission and the one that starts the next, the GPU
is idle waiting for the host, and on Mali those gaps totalled **0.44-0.61 ms
per frame** - 14-19% of GPU total, and larger than every span except
labelling.

The frame used to be four submissions, split by three host readbacks: the
labelling convergence flag, the boundary-point count, and the selected-blob
and point counts. Each readback forced `vkQueueSubmit` + `vkWaitForFences`
before the next dispatch could even be recorded.

Bounded first, by a build that reuses the *previous* frame's counts and
records everything as one submission - exact for a repeated still image, and
it reported the same counters and corner RMS. That measured GPU total
**-7.9 to -10.1%** with timestamps off, so the ceiling was real and not an
instrumentation artifact.

Two of the three readbacks are now gone. `build_indirect_args.comp` writes
three `VkDispatchIndirectCommand`s from device-side counters, and
`hash_group`, `reduce_extents_hash`, `scatter_index_points` and
`sort_points_local` take their bounds from a buffer instead of a push
constant (`count_from_buffer`), so the whole tail is recorded in the same
submission that produces the counts it needs. The last host dependency,
staging the readbacks, went away by reading `selected_extents_buf_` in place
the way the line-fit buffer already was.

Measured on the Mali-G610 against the previous commit, ABBA, min of 12-16:

| | GPU total | `pipeline_total` |
| --- | --- | --- |
| decimation 1 | -2.0% | -2.8% |
| decimation 2 | -4.0% / -4.4% | -3.6% / -4.3% |
| decimation 4 | **-7.9%** | -5.5% |

The gain is larger on smaller frames because the cost removed is per-submit
and roughly fixed, so it is a bigger share of a shorter frame.

**Requires both readbacks to be direct**, i.e. a memory type that is both
host-visible and host-cached, which is what unified-memory parts give. A
discrete card without resizable BAR used to keep the four-submit path here,
and the RX 9060 XT was exactly that case - see "Closing the RDNA4 fallback"
below for why it no longer is.

### The last boundary: closed

One boundary used to remain here, before `uf_final`, because the host still
read the labelling convergence flag to decide whether to run another
union-find chunk - measured on its own at **0.28 ms, ~9% of the frame**.

It is closed now, with the retry this section used to describe as future
work: record the whole frame assuming the chunk count *last* frame needed
converges this frame too (true for essentially every frame once steady
state is reached), and if the flag says it did not, discard everything and
redo it - the labelling chunks resuming from where they left off, everything
downstream re-run once `parent[]` is genuinely converged. `clear`
re-initializes every buffer that needs it, so a redo is simply correct.

Two bugs surfaced while shipping this, both now fixed and both recorded in
`OPTIMIZATION_NOTES.md` because they generalize past this one change: a
shader (`label_pixels.comp`) that destructively rewrites its own input had
to be gated so it never runs before convergence is confirmed - running it
early corrupted `parent[]` for the retry's own `uf_merge`/`uf_compress`
calls, which hung the GPU (`VK_ERROR_DEVICE_LOST`, observed on the
Mali-G610). And a **missing compute-to-transfer barrier**, present in every
version of this code that has ever copied `uf_changed_buf_` back to the
host - including the original, still-shipping unfused catch-up loop - had
been invisible because the copy was always immediately followed by a full
submit boundary; fusing the tail behind the same copy gave the GPU's
scheduler real work to overlap it with, exposing the race.

Measured on the Mali-G610 against the state with the earlier two boundaries
already closed: GPU total -0.7% (decimation 1) to -5.2% (decimation 4). On
the RX 9060 XT it used to measure as noise - the fused path never engaged
there - but see the next section for why that changed.

### Closing the RDNA4 fallback: `HostVisibleCached` instead of `DeviceLocalReadback`

The fallback above assumed the fused path needed a memory type that is
`DEVICE_LOCAL` *and* host-cached, and a discrete card without resizable BAR
has no such type - confirmed directly from a raw `vulkaninfo` memory-type
dump on the RX 9060 XT (AMD proprietary driver 26.8.1): the resizable-BAR
heap (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) is explicitly
`DEVICE_UNCACHED_AMD`, and the only host-cached type
(`HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`) is plain system RAM, not
device-local.

`selected_extents_buf_` and `line_fit_points_buf_` were allocated with
`vk::MemoryKind::DeviceLocalReadback`, which *requires* `DEVICE_LOCAL` and
only prefers host-cached - so on this device it fell back to the
uncached BAR type, `extents_direct_read_`/`linefit_direct_read_` came back
false, and `fused_submits_` stayed permanently off. Switching both buffers
to `vk::MemoryKind::HostVisibleCached` - which requires only `HOST_VISIBLE`
and prefers `HOST_CACHED`, with no device-local requirement - lands them on
the host-cached system-RAM type instead, at the cost of every GPU-side write
to them (`select_blobs.comp`, `sort_points_local.comp`) now crossing PCIe
rather than staying in VRAM. It is not a Mali/RDNA fork: on a unified-memory
part the same request still finds the single ideal device-local +
host-visible + cached type, since the "required" bits are a subset check,
not an exact match, so nothing changes there.

Measured on the RX 9060 XT, ABBA-interleaved, 16 rounds x 300 iterations,
1280x800:

| | GPU total (median) | `extents` span | `sort` span |
| --- | --- | --- | --- |
| decimation 1 | **-19.8%** (16/16 rounds) | not separately measured | not separately measured |
| decimation 2 | **-21.3%** (16/16 rounds) | +101% (0.032 -> 0.064 ms) | +6.3% (0.127 -> 0.135 ms) |

The write-side cost is real and exactly where expected - `extents` roughly
doubles and `sort` picks up a smaller penalty - but both are a few hundredths
of a millisecond against a ~0.9 ms frame, and the three submission
boundaries this removes are worth far more: about 0.19 ms, well above the
~42 us empty-submit-round-trip figure this file quotes elsewhere, because a
real submission also drains in-flight work rather than firing on an idle
queue. Net: a clear win, larger than `OPTIMIZATION_NOTES.md`'s own >=0.13 ms
estimate for this change.

Bit-identical over 36 configurations (decimations 1/2/4 x
`APRILTAG_VK_FORCE_NO_8BIT` on/off x workgroup geometries auto/2x2/6x6 x
`APRILTAG_VK_UF_CHUNK` 1/default) - the only observable difference is
`submits` dropping from 4 to 1, which is the point of the change, not a
side effect of it.

### The sort network: Batcher's odd-even mergesort

`sort_points_local.comp` ran a naive bitonic network over each blob's
points, padded up to the next power of two. Replaced with Batcher's
odd-even mergesort over the same padded capacity: same number of
barrier-synchronized rounds for a given cap (so no change in
synchronization cost), 13-21% fewer total compare-exchanges across the cap
range this shader actually sees (16..4096), since it never compares indices
that provably cannot be out of order. The comparator schedule was proven
correct standalone (zero-one principle, exhaustive at cap=8/16, random
trials at cap=32/64/128) before being ported to GLSL - see
`OPTIMIZATION_NOTES.md` item 9 for the full writeup, including two other
shader-vectorization ideas tried alongside it and rejected.

Measured, ABBA, largest corpus image, GPU total (min-of-60 per run):

| | Mali-G610 (8 ABBA rounds) | RX 9060 XT (4 ABBA rounds) |
| --- | --- | --- |
| decimation 1 | **-2.2%** | **-5.7%** (one outlier run excluded) |
| decimation 2 | **-2.7%** | **-3.5%** |
| decimation 4 | **-3.3%** | not distinguishable from noise (n=4, overlapping) |
| `sort` span alone (RX 9060 XT, decimation 1) | -29%, clean and non-overlapping over 10 runs each | |

Mali holds a consistent, non-overlapping win at all three decimations - an
earlier pass of this table, with only 2 ABBA rounds at decimation 2/4,
wrongly showed the gain vanishing or reversing at decimation 4; that was
sampling noise from too few rounds, not a real effect, and is superseded by
the 8-round numbers above. On the RX 9060 XT the win shrinks with
decimation as expected (smaller blobs, less for the sort network to do),
and at decimation 4 the span is small enough relative to desktop-GPU
scheduling noise that 4 ABBA rounds cannot resolve it - Mali is the
deployment target this project optimizes for, so this was not chased
further with more samples.

## 4. Edge refinement (`APRILTAG_VK_REFINE`)

Upstream libapriltag's `refine_edges` is the single most expensive CPU
function in the pipeline. `library/src/RefineEdges.cpp` provides three
implementations:

- **`exact`** (default) — identical arithmetic in identical double precision,
  with `modf()` rewritten as `trunc()` plus a subtract. This is bit-identical
  to upstream *by construction*, not by measurement: `modf(x, &i)` is defined
  to return `x - trunc(x)` and both results are exactly representable. What it
  removes is a non-inlinable libm call from the innermost loop.
- **`fast`** — single-precision refinement, not bit-identical to `exact`. Each
  sample's search profile is evaluated once per position (the far gradient tap
  is the near tap eight steps earlier), the weighting pass is branch-free with
  independent accumulators, and the line fit uses moments centred on the edge
  midpoint, which removes the cancellation that previously forced double.
  Measured on the RX 9060 XT (`grayimage.pgm`, 75 quads, 1 tag): `tag_decode`
  0.27 -> 0.24 ms; recall and pose accuracy on the synthetic sweep match
  `exact`.
- **`ultrafast`** — `fast`, plus `TagDecoder` refines only quads that already
  decode unrefined. `tag_decode` -> 0.095 ms on the same image. The gate costs
  recall on the synthetic sweep (663 vs 687 decoded of 768 with
  `--refine-edges`) because marginal small tags that only decode after
  refinement are rejected.
- **`upstream`** — calls upstream's compiled function. Use it to isolate any
  suspected corner-accuracy regression to this code.

Measured `exact` against `upstream`: **`tag_decode` -25% on the Orange Pi**
(1.05 -> 0.79 ms), against **-4.7% on x86** (1.99 -> 1.90 ms single-threaded).
The entire win is deleting the libm call, and glibc's aarch64 `modf` costs far
more relative to the surrounding work than x86's `__modf_avx`. `fast` adds ~1%
on x86 and nothing on ARM — the loop is memory-latency bound (eight scattered
byte loads per step), not arithmetic bound, so narrowing precision cannot help.

A caution this measurement earned: a `perf` profile attributed 55% of *cycles*
to `refine_edges`, which badly overstated the wall-clock opportunity. After
being made 6% faster it was still 65% of cycles. Confirm any cycle-share
finding with a controlled wall-clock A/B.

## 5. Frame pipelining (`FramePipeline`)

`Detect` -> `QuadDecode` -> `TagDecoder` runs strictly serially, so the GPU
idles through the whole CPU tail and vice versa. `FramePipeline` runs the GPU
pass for frame N on its own thread while the caller decodes frame N-1, making
a frame cost `max(GPU, CPU)` rather than the sum — the last two rows of the
table in section 3 (2.87 -> 2.2 ms on the Orange Pi, about 0.85 -> 0.43 ms on
the desktop).

Three things callers must know:

- **It is throughput, not latency.** An individual frame takes just as long and
  now arrives one `Push()` later.
- **It needs two frame buffers.** The in-flight GPU pass reads the pushed frame
  asynchronously while the tail samples the previous one.
- **It is resolution-dependent.** At 1280x800 it is a clear win on both
  machines; at 640x400 on the Pi it measures neutral, because the CPU tail and
  GPU phase are close enough in size that the overlap gain is cancelled by
  contention for the shared LPDDR bus. An earlier attempt on an older tree,
  when the GPU phase was 5.70 ms rather than ~3.8, measured a 21% *regression*
  on Mali. Re-measure at your deployment resolution rather than assuming.

It also interacts with the in-place line-fit readback: `last_line_fit_points`
is a non-owning `std::span` into memory the next `Detect()` overwrites, so the
pipelined path must copy it. Serial callers keep the zero-copy path. Zero-copy
readback and pipelining are in tension and do not compound.

ThreadSanitizer is clean. Both the serial and pipelined paths report exactly
one race, identical in both, entirely inside Mesa (`pthread_barrier_destroy`
vs `pthread_barrier_wait` in the driver's own disk-cache threads, spawned from
pipeline-cache creation). No project symbol appears in it.

## 6. Per-device guidance

**Device memory** is sized for the dense worst case by default (4 boundary
points per interior decimated pixel). That is 447 MiB at 1920x1080 - fine on a
discrete card, painful on a unified-memory part. (The figures below predate
the packed `RawLineFitPoint` in section 6b, which takes roughly another 15%
off the total; the *relative* effect of `APRILTAG_VK_MAX_POINTS` is
unchanged.)

Measured at 1920x1080 (detection results identical in all cases):

| `APRILTAG_VK_MAX_POINTS` | Device memory | Boundary points found |
| --- | --- | --- |
| unset (dense worst case) | 447 MiB | 159144 |
| 400000 | ~99 MiB | 159144 (no clamping) |
| 200000 | 56 MiB | 159144 (no clamping) |
| 100000 | ~36 MiB | 100000 (clamped, tag still decoded) |

**Recommendation:** on Mali/integrated, set `APRILTAG_VK_MAX_POINTS` to ~2x
the `boundary_points` your scenes actually report (the profile line prints it).
200000 gives 56 MiB at 1080p with no loss of fidelity. On discrete cards the
default is fine.

Other automatic per-device behaviour:

- **Workgroup sizes are specialization constants**, chosen at runtime from
  `maxComputeWorkGroupInvocations` / `maxComputeWorkGroupSize` /
  `maxComputeSharedMemorySize`. Nothing is hardcoded. Sweeping
  `APRILTAG_VK_WG` on Mali found the automatic choice (128) already within
  noise of the best value, so there is no per-device tuning left to do here.
- **Unified memory** is detected; on integrated parts the grayscale frame is
  written straight into device-local host-visible memory and the staging copy
  is skipped entirely.
- **`HOST_CACHED`** readback memory is requested when available, and where the
  memory type allows it the line-fit records are read in place rather than
  staged. The startup line reports which path is in use.
- **Subgroup aggregation survives at exactly one site**,
  `reduce_extents_hash`. It used to be applied at three, gated together, and
  measuring the three separately on the RX 9060 XT says that was wrong at two
  of them (min of 12, three sessions, each reproducing within 1%):

  | site | aggregated | plain atomics | |
  | --- | --- | --- | --- |
  | `uf_final` | 0.1035 ms | **0.0291 ms** | scalar **-72%** |
  | `blob_diff` | 0.0437 ms | **0.0330 ms** | scalar **-24%** |
  | `reduce_extents_hash` | **0.0702 ms** | 0.1028 ms | subgroup **-45%** |

  Retiring the two losers is worth **GPU total -5.3%** on that card and
  changes nothing on Mali, which never took them. For `uf_final` this only
  confirms a case `OPTIMIZATION_NOTES.md` had already built from the other
  side: on an MX230 the scalar variant with its saturating guard beat the
  aggregated one 3x, and the note says only that a bigger discrete part to
  re-test on was not available.

  What separates the survivor is *what* it aggregates. `reduce_extents_hash`
  reduces per-point **values** across lanes sharing a key, collapsing eight
  atomics per point into eight per distinct key per subgroup. The two retired
  ones only ever aggregated a **counter** — one `atomicAdd` per lane becoming
  one per subgroup — which is precisely the contention a modern discrete
  part's atomic unit already handles well, so the ballot sequence bought
  nothing and cost its own issue slots. That is the transferable rule:
  aggregate values by key, not bare counters.

- **The surviving variant** is selected when the device advertises ballot,
  arithmetic *and* shuffle — **and is not an integrated GPU**. Integrated
  parts are excluded outright regardless of what they report: on the
  Orange Pi 5's Mali-G610, which advertises all three and produces
  bit-correct results with them, enabling them took `pipeline_total` from
  **11.5 ms to 18.1 ms**, with `reduce_extents_hash_subgroup.comp`'s
  `extents` span alone going 0.91 -> 6.04 ms. Valhall has no dedicated
  hardware for these patterns, so the ballot/shuffle sequences cost more
  than the plain atomics they replace — the same class of result as the
  tile-local union-find rejection. On a discrete part (RX 9060 XT,
  subgroupSize=64) they are a small repeatable win, which is why the
  exclusion is specific to integrated GPUs rather than a blanket disable.
  See the comment on `subgroup` in `GpuDetector::CreatePipelines()`.

  (An earlier revision of this file claimed subgroup variants were "worth
  ~2.7%" on Mali and were gated on capability alone. Both halves were
  wrong — they are a large regression there, and they have been excluded
  by device type since. Corrected because that sentence was load-bearing:
  it is exactly the line that makes subgroup stream compaction look like
  an obvious next optimization when it is in fact among the most firmly
  ruled out on this target.)

  Note also that subgroup aggregation and cheap atomic early-outs are
  *alternative* answers to the same contention, and applying both is worse
  than either — which is why the guards in `uf_final` /
  `reduce_extents_hash` are deliberately absent from their
  subgroup-aggregated variants.

### 6b. The line-fit record is packed to 8 bytes

`RawLineFitPoint` - one entry per selected boundary point, and the largest
per-frame readback in the pipeline - carries `(x2, y2, W, blob_index)`. It
used to spend a full 32-bit word on each. Every one of the four has a hard
bound that something else already enforces, so all four fit in two words with
room left over:

| field | bound | why | bits |
| --- | --- | --- | --- |
| `x2`, `y2` | <= 16383 | the constructor rejects `2*(width/decimation) > 16383`, the same bound `PackXY` relies on | 14 each |
| `W` | <= 361 | `int(sqrt(gx*gx + gy*gy)) + 1` over 8-bit gradients: `sqrt(2*255^2)` truncates to 360 | 10 |
| `blob_index` | < 65536 | `max_blobs`, clamped to `max_raw_blobs`; the host rejects `max_raw_blobs > 2^22` | 22 |

This is **exact, not a precision trade** — the accessors return the same
integers the four-field struct did, and output is bit-identical. Measured on
the RX 9060 XT, `grayimage.pgm`, ABBA-interleaved:

| | decimation 1 | decimation 2 |
| --- | --- | --- |
| readback bytes/frame | 836848 -> **425760** (-49%) | 286768 -> **145568** (-49%) |
| device memory | 185 -> **154 MiB** (-17%) | 48 -> **41 MiB** (-15%) |
| `readback_copy` span | -84% to -89% | -20% |
| GPU total | -3.3% to -8.6% | -5.3% to -10.4% |

The byte and memory figures are exact and reproducible. **The span and total
figures are timings, and this machine's spread across four measurement
sessions is wider than the effect being measured** — hence the ranges rather
than a single number.

**On the Mali-G610 target the GPU-time effect is nil**, measured over three
further interleaved sessions: GPU total +1.1% / -0.7% / +0.7%, i.e. noise
either side of zero. `pipeline_total` is consistently about -1%, which is the
CPU tail reading half as many bytes. Device memory is 48 -> 41 MiB there,
exact.

That is not a disappointment, it is section 3a's model working: on a
unified-memory part the line-fit readback is a direct host-cached read
(`readback_copy` = 0.006 ms, no PCIe hop to shorten), and `sort` — where the
halved writes land — measures a bandwidth sensitivity of **0.99x**, i.e. none
at all. The change is worth keeping on that target for the 15% memory
footprint, which section 6 explains matters on a unified-memory part, not for
GPU time. The discrete-card speedup is real and is a PCIe effect.

Worth recording explicitly: **fp16 would not have worked.** Its 11-bit
mantissa makes integers above 2048 round to even, and `x2` reaches 3839 at
1080p / decimation 1 — a half-pixel coordinate error on the right-hand side
of the frame, feeding straight into corner positions. Integer packing has no
such cliff. Storing the *moments* as fp16 is further out still: the CPU
rebuilds `Mxx/Mxy/Myy` in native `int64` precisely because the covariance is
a near-total cancellation, the same reason `exact` refinement keeps
its line fit in double.

### 6c. `uf_compress` skips its read pass when the labelling has converged

`uf_compress.comp` already guarded its *store* (rewriting an identical value
is pure write traffic). It still streamed the whole `parent[]` array to
discover there was nothing to write — 1 MB at 1280x800 / decimation 2, on the
most common case in steady-state video, where labelling converges in the
first chunk and every later pass is pure verification.

It now reads the convergence flag first and returns immediately if the
preceding `uf_merge` joined nothing. That is exact, not approximate: every
chunk ends with a compression, so `parent[]` is already flat entering the
merge, and a merge that changed nothing leaves it flat. The flag is honoured
only for in-chunk compressions, via a push constant — the compression that
runs straight after `uf_init` must always run, and is the one
`OPTIMIZATION_NOTES.md` item 8 measured at 1.0 ms.

Measured on the Mali-G610, ABBA-interleaved, min of 24, three sessions:
**`labelling` -2.1% / -2.2% / -2.4%, GPU total -0.5% / -0.7% / -0.9%.**
Bit-identical at decimations 1/2/4 on both machines.

Small, and deliberately quoted small. The ceiling, measured by removing the
dispatch outright, is `labelling` -4.4% / GPU total -1.4%; the guard captures
about half because the invocations still launch and each pays one scalar
load. Closing that gap needs `vkCmdDispatchIndirect` with a device-computed
group count of zero, which costs an extra dispatch and barrier — and on Mali
a barrier is 2.6-18.7 us against the ~30 us remaining, so it is not obviously
positive. Not attempted.

**Where `VK_EXT_conditional_rendering` exists, the dispatch is now skipped
outright** - no extra dispatch and no extra barrier. The predicate is
`uf_changed_buf_` itself, and the merge's existing barrier just names the
conditional-rendering stage too. The Mali-G610 does not expose the
extension, so there the in-shader guard above is still what runs. Only the
*last* compression of each chunk is predicated: every earlier one reads a
flag that has been accumulating since the previous clear, which is nonzero
in practice, and a predicate that never skips is not free. Predicating
every iteration measured **+5.3% `labelling` at `APRILTAG_VK_UF_CHUNK=8`**
(slower in 10 of 12 rounds) before it was narrowed.

Measured on the RX 9060 XT (four-submit path), A/B'd on one binary with
`APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING`, ABBA-interleaved, 16 rounds of
300 iterations, 1280x800. Figures are the median per-round paired delta of
each run's median `labelling` span:

| | `labelling` | GPU total |
| --- | --- | --- |
| decimation 1 | **-3.0%** (15/16 rounds) | -0.6% (10/16, noise) |
| decimation 1, chunk 1 | **-3.0%** (16/16) | -0.9% (14/16) |
| decimation 2 | -1.1% (12/16) | -0.4% (10/16, noise) |
| decimation 2, chunk 8 | -0.6% (10/16, noise) | 0.0% |

That is on top of the in-shader guard, not instead of it. GPU total barely
moves because this path is dominated by its submit round trips. Bit-identical
over decimations 1/2/4, `APRILTAG_VK_FORCE_NO_8BIT` on and off, workgroup
geometries auto/8x8/32x8/2x2/6x6 and `APRILTAG_VK_UF_CHUNK` 1 and the
default. Clean under `VK_LAYER_KHRONOS_validation` with synchronization
validation. That last point says less than it looks like, though: a
deliberately broken build that dropped the predicate stage from the merge's
barrier was *also* reported clean, so the layer does not model the
predicate read, and that barrier rests on the spec rather than on the tool.
The single-submit path with the extension (a unified-memory desktop part)
has not been run - no such device was available.

Verified launch geometries (all produce identical detections):

| Simulated limit | wg1d | wg2d | scan_wg |
| --- | --- | --- | --- |
| default (1024) | 256 | 16x16 | 1024 |
| 512 (Mali-G610) | 256 | 16x16 | 512 |
| 256 | 256 | 16x16 | 256 |
| 128 (Vulkan minimum) | 128 | 8x8 | 128 |

## 7. A note on determinism

At default settings the pipeline is **exactly** reproducible: consecutive runs
give identical boundary point, blob, point and candidate-quad counts.

Changing the workgroup size does perturb the *candidate quad* count by +/-1-2.
This is inherent and pre-existing: boundary points are appended with an atomic
counter, so their order depends on workgroup arrival order, and points whose
sort keys are exactly equal can therefore end up in a different order. That
shifts the windowed line fit marginally at the margins of the peak-detection
threshold. The decoded tag set was unaffected in every configuration tested.

If you ever need bit-exact reproducibility across launch geometries, the
compaction in `blob_diff.comp` would need an ordered (scan-based) output index
rather than `atomicAdd`, at some cost.

## 8. What is left

`OPTIMIZATION_NOTES.md` holds the full list with numbers, including everything
already tried and rejected — read it before proposing work, since most of the
obvious ideas are in the rejected column. Where it stands on the Orange Pi: 2.04 ms of
GPU work and about 0.85 ms of CPU tail per frame (2.87 ms serial). Open items, largest first:

- **CPU edge refinement.** `tag_decode` is 0.63 ms, and 97% of the refinement time goes to
  quads that never decode. The margin gate (section 8l) removes most of it with no change to the
  detections on the sweep, but it is opt-in because a real tag below the gate would be lost.
- **Labelling** is 40% of the GPU stages; only a structurally different algorithm would move it
  (items 1 to 3 below).
- **Ingest.** Hardware JPEG decode into a dma-buf is outside the library and needs stride and
  offset on `ImportDmaBufFrame` (section 8d).
- **Quads that never decode** (section 8y): the DP corner seeding turns many non-tag blobs into
  convex quads that libapriltag's peak search would reject (131 against 54 on a busy 1200x800
  frame), and each costs a refinement. Fewer of them means a different quad set.
- **The remaining CPU-tail items from profiling** (section 8p): the frame upload and the copy of the
  line-fit points, and the `matd` allocations in upstream's homography.

Measured and closed: the `blob_diff` append atomic and the small-blob sort (section 8n), dispatch
folding, run-merging before the 64-bit atomics, deferring the post-fill barrier (section 8k), and
the CPU idle, interrupt and Mali power settings (section 8i). The earlier conclusions, still valid:

0. ~~**The one remaining submit boundary**, before `uf_final`~~ **Closed** -
   all three of the original four-submit frame's mid-frame readbacks are now
   gone on unified-memory parts, via the retry described in section 3c.
   Finding and fixing the two bugs that surfaced on the way (a shader that
   destructively rewrote its own input, and a missing compute-to-transfer
   barrier present since the very first version of this readback) is
   recorded in `OPTIMIZATION_NOTES.md` item 8, since both generalize past
   this one change.
1. **A better connected-components algorithm.** `labelling` is still the
   largest span at ~28% of the Mali GPU phase, which caps any rewrite there.
   The cheapest part of HA4 - merging per run rather than per pixel - is
   taken (`OPTIMIZATION_NOTES.md` item 7 of the third pass, -7 to -13% of the
   stage). What remains of HA4 is its compact per-strip run arrays and
   warp-per-strip labelling, which add an extraction pass and lean on exactly
   the warp intrinsics four separate measurements here have rejected. Note that BUF /
   Playne-Equivalence **do not apply**: they require every foreground pixel of
   a 2x2 block to be connected, which holds for binary 8-connected labelling,
   while this pipeline is 4-connected and three-valued (127 merges with
   nothing). Tile-local union-find was tried on Mali and was much slower —
   Valhall has no scratchpad, so `shared` is backed by L2 and dependent shared
   loads are not cheaper than global. A simpler idea - just vectorizing
   `uf_merge` to 4 pixels/thread, no algorithm change, purely fewer
   invocations/barriers for the same work - was also tried and **measured
   worse**: correctness-clean, but +6% GPU total on the Mali-G610 (see
   `OPTIMIZATION_NOTES.md` item 9). Fewer, fatter threads is not free on this
   part once the per-thread control flow grows to reconstruct what the
   scheduler used to give for free.
2. ~~**Reducing the number of full-image passes.**~~ **Largely closed by
   section 3a.** The claim that "this is where the Mali time actually is" came
   from an input-size sweep, which shows time scaling with *pixels* — and then
   assumes that scaling is DRAM traffic. A memory-clock sweep separates the
   two, and says only ~15% of GPU time is bandwidth-proportional. Two fusions
   were written to test the idea directly; both lose on both devices
   (see `OPTIMIZATION_NOTES.md`). Pass-count reduction can still help by cutting *launches* and
   *latency* — which is what the `uf_compress` guard in section 6c does — but
   the bandwidth framing was the wrong model and should not drive more work.
3. **Not the labelling convergence check.** Removing the second merge entirely
   — a deliberately incorrect build, used to bound the ceiling — measured
   **zero** on the Pi. `uf_merge` already performs no writes and no atomics on
   a converged pass, so it is effectively a read-only verification pass
   already, and there is nothing for a cheaper convergence proof to reclaim.
4. **Not `extract_blob_counts`' capacity-sized dispatch.** It is the one
   dispatch still sized by a capacity rather than a device-side count, so it
   reads like the obvious remaining instance of the `init_extents` /
   `select_blobs` fix (which is *done* — both go through
   `build_indirect_args.comp`). Bounded directly by shrinking the capacity 8x
   with `--max-blobs 512` on the RX 9060 XT: `blob_scan` moved 0.01044 ->
   0.00632 ms, so the entire capacity-proportional cost of that dispatch
   **and its whole scan chain** is ~4 us, of which right-sizing the dispatch
   alone recovers only a part. That matches the independent estimate already
   in the constructor's `max_blobs` comment (+2.2 us for a 24x capacity
   increase), and the slack is deliberate: it buys reproducibility, since
   overflowing `max_blobs` makes detections depend on GPU scheduling order.
   Not worth an indirect-args slot, a buffer fill, and the loss of the
   invariant that the scan's grand total sits at
   `blob_point_offsets[max_blobs - 1]`.

## 8b. Concurrent streams on separate queue families

`tools/bench_multiqueue` runs N independent `Context` + `GpuDetector` pairs, one per thread, on
the same GPU, each optionally pinned to a queue family (`ContextOptions::queue_family`). GPU pass
only, `grayimage.pgm` (1280x800, decimation 2), 300 iterations per stream, 6 rounds with the
order reversed on alternate rounds. RX 9060 XT: family 0 is graphics+compute (8 queues), family 1
is compute-only (8 queues). Median of rounds:

| Streams | Family 0 | Family 1 | Alternating 0/1 |
| --- | --- | --- | --- |
| 1 | 2,293 fps | 2,380 fps | - |
| 2 | 3,791 | 3,844 | 4,345 |
| 4 | 3,770 | 5,179 | 6,147 |
| 8 | 3,855 | 5,679 | 6,083 |

A single stream leaves the GPU under-used: four streams alternating between the two families give
2.7x the throughput, at 0.64 ms median per frame against 0.42 ms alone. Family 0 saturates near
1.7x; the compute-only family scales further. Caveats: the streams are separate `VkDevice`s, not
several queues of one device, so this shows the hardware overlaps independent work rather than
what one shared device would achieve; other GPUs (Mali, integrated) were not measured. With the
CPU tail on and one decode thread per stream, throughput scales with CPU cores instead (387 fps
for one stream, 2,529 for eight).

On the Orange Pi 5 Plus (Mali-G610, one graphics+compute family with 2 queues, DRAM and GPU
governors pinned to `performance` for the run and then restored) the same experiment shows almost
no gain, so the GPU is already saturated by one stream:

| Streams | GPU-pass fps (median of 6 rounds) | Frame median |
| --- | --- | --- |
| 1 | 365 | 2.76 ms |
| 2 | 390 | 2.93 ms |
| 4 | 377 | 3.21 ms |

With the CPU tail on and one decode thread per stream: 149 / 267 / 313 fps for 1 / 2 / 4 streams,
limited by the 8 CPU cores.

## 8c. Several cameras on one device: queue-per-stream, with frame pipelining

`vk::Lane` (a queue plus its own command pool and fence ring) lets several `GpuDetector`s share one
`VkDevice`; `ContextOptions::queues_per_family` and `use_secondary_compute_family` create the
queues. `tools/bench_multiqueue` now builds one `Context` and, per stream, a `Lane`, `GpuDetector`,
`QuadDecode`, `TagDecoder` and `FramePipeline` fed by its own thread, so the CPU tail is hidden as in
a real multi-camera setup. Modes: `single` (all streams share queue 0), `per-stream` (one queue per
stream on the primary family), `alternate` (queues interleaved across both compute families).
Every result is checked against the reference tag IDs (no mismatches in any run below). Median of
4 rounds, order reversed on alternate rounds, `grayimage.pgm`, CPU threads split evenly per stream.

RX 9060 XT, total frames/s (12 logical CPUs):

| Streams | Mode | `exact` refine | `ultrafast` refine |
| --- | --- | --- | --- |
| 1 | single | 1,687 | 1,905 |
| 2 | single / per-stream / alternate | 2,779 / 2,815 / 2,793 | 3,814 / 3,854 / 4,117 |
| 4 | single / per-stream / alternate | 2,943 / 2,933 / 2,930 | 3,944 / 3,852 / 6,064 |
| 8 | single / per-stream / alternate | 2,696 / 2,641 / 2,692 | 3,878 / 3,747 / 5,908 |

- With `exact` refinement the CPU tail is the bottleneck (about 2.9k fps) and queue layout makes no
  difference.
- With the tail made cheap (`ultrafast`) the GPU limits. On the graphics+compute family (0), one
  queue per stream gives no gain over sharing one queue (about 3.9k fps). Using the second,
  compute-only family (1) helps, and there a queue per stream matters (table below); `alternate`
  across both families reaches about 6.0k fps. The earlier multi-device result (8b) came from the
  same effect.
- Paced at 60 fps per stream- Paced at 60 fps per stream, 32 streams (1,920 fps total) were sustained with either refine mode.
  The reported latency and late-frame counts are not reliable on Windows: results return one
  `Push()` late by design and `sleep_until` granularity is coarse.

Compute-only family (1) alone, same setup (`--family 1`), total fps:

| Streams | Mode | `exact` refine | `ultrafast` refine |
| --- | --- | --- | --- |
| 1 | single | 2,095 | 2,121 |
| 2 | single / per-stream | 2,774 / 2,707 | 3,671 / 3,724 |
| 4 | single / per-stream | 2,932 / 2,961 | 3,793 / 5,240 |
| 8 | single / per-stream | 2,650 / 2,667 | 3,715 / 5,233 |

With a cheap tail, a queue per stream on family 1 reaches 5.2k fps (1.4x over one shared queue),
close to the 6.0k of alternating across both families; on family 0 the same layout gave nothing.

Orange Pi 5 Plus (Mali-G610: one family, 2 queues; governors pinned then restored), total fps:

| Streams | Mode | `exact` refine | `ultrafast` refine |
| --- | --- | --- | --- |
| 1 | single | 351 | 340 |
| 2 | single / per-stream | 442 / 385 | 444 / 381 |
| 4 | single / per-stream | 444 / 420 | 446 / 432 |

The GPU saturates at about 440 fps with one queue; a queue per stream is no better and is slightly
worse (the two queues contend for the same hardware). At 30 fps per stream with `ultrafast`, 12
streams (360 fps total) were sustained.

## 8d. Frame ingest: batching, a transfer queue and zero-copy import (experiment)

All runs use `bench_multiqueue` (one device, `FramePipeline` per stream, `ultrafast` refine so the
GPU is the limit), ABBA-ordered, median of 6 rounds, every result checked against the reference
detections (no mismatches).

**Batching** (several cameras tiled into one frame and detected in one dispatch) was tried with a
throwaway mosaic mode and removed: it raised a lone stream's cameras/s by 1.7x (RX 9060 XT) and
1.27x (Mali-G610), but concurrent streams already reach the same saturation level, so it adds
nothing to a multi-camera setup and nothing for a single camera.

**Transfer-only queue** (RX 9060 XT; `--async-upload`: the staged copy runs on family 2, the first
compute submission waits on a semaphore). Total fps, alternating compute families:

| Streams | staged copy in submission | transfer-queue upload | host-pointer import |
| --- | --- | --- | --- |
| 1 | 2,112 | 1,842 | 2,672 |
| 4 | 6,129 | 6,593 | 7,836 |
| 8 | 5,975 | 7,678 | 7,688 |

The transfer queue helps once several streams are in flight (+8% at 4, +28% at 8) and costs about
10% for a single stream (extra submission), so it is opt-in (`--async-upload`,
`ContextOptions::use_transfer_family` plus the `GpuDetector` overload taking a transfer `Lane`).
It is a fallback for discrete GPUs where import is unavailable: the driver lacks
`VK_EXT_external_memory_host`, or the caller's buffer cannot be imported (host-pointer import needs
the pointer and size aligned to `minImportedHostPointerAlignment`, 4096 on this card, which arbitrary
buffers such as JNI or `cv::Mat` data are not). The GPU-side semaphore that orders the copy before
the first compute submission is worth 5-7% at 1-4 streams and 1.5% at 8 over waiting for the copy's
fence on the CPU (1,829 vs 1,715 fps at 1 stream, 6,381 vs 6,069 at 4, 7,636 vs 7,520 at 8).
Removing the copy entirely bounds the gain: with the GPU copy skipped, 4 streams reached 7,687 fps,
and with the host memcpy skipped too, 8,901. The Mali has no transfer-only family and writes frames
straight into mapped device memory, so this option does not apply there.

**Zero-copy import.** The RX 9060 XT imports the caller's aligned buffer with
`VK_EXT_external_memory_host` (`Buffer::ImportHostPointer`, `GpuDetector::ImportHostFrame`): +28%
at 4 streams, +29% at 8, +26% for one stream. On the Orange Pi the Mali driver (g24p0) does **not**
expose `VK_EXT_external_memory_host`; it exposes `VK_EXT_external_memory_dma_buf` and
`VK_KHR_external_memory_fd`, so `Buffer::ImportDmaBuf` / `GpuDetector::ImportDmaBufFrame` import a
dma-buf instead (`tools/dmabuf_import_test`). A dma-heap buffer produced identical quads to the copy
path and cut `Detect()` from 2.79 to 2.60 ms. UVC capture buffers (V4L2 `mmap`, `VIDIOC_EXPBUF`)
import successfully as well, and `decimate_yuyv` reads the luma out of the YUYV frame on the GPU.
The catch is cache coherency: uvcvideo fills its buffers with the CPU and its vmalloc dma-buf
exporter has no CPU-access sync, so `DMA_BUF_IOCTL_SYNC` does nothing and the GPU read stale data
(0-39% of frames matched the copy path). Cleaning the CPU cache first with `dc civac` over the buffer
(EL0-legal on arm64) made 100 of 100 frames identical at 640x480, 1280x720 and 1920x1080, for a
flush cost of 0.03 / 0.08 / 0.17 ms. Host time per frame, copy path (CPU luma extraction, upload,
`Detect`) versus import (`Detect`): 2.07 vs 1.52 ms at 640x480, 3.86 vs 2.98 at 1280x720 and 8.66 vs
6.02 at 1920x1080 (governors pinned).

**Re-measured with the tracer (RX 9060 XT, `apriltag_vulkan_validate --import`).** The tool detects
the same frame from a plain buffer (staged copy) and from an aligned, imported copy, ABBA-ordered in
one process (4 rounds x 2 x 500 frames per path, three repeats; GPU spans need
`APRILTAG_VK_TIMESTAMPS=1` or `--profile`). `scale_1` (1280x800), median ms:

| | copy | import | |
| --- | --- | --- | --- |
| `Detect()` | 0.565 | 0.484 | -14.2 to -14.4% |
| host memcpy | 0.054 | 0.000 | -100% |
| GPU span `threshold` | 0.013 | 0.035 | +21 us |
| `pipeline_total` | 1.02 | 0.94 | -7.6 to -7.9% |

At 640x400 the gain is smaller: `Detect()` -6.1%, `pipeline_total` -3.4%. The saving is the memcpy
(55 us) plus the staging-to-device copy, minus 21 us that `decimate` now spends reading the frame
across PCIe, which is why `threshold` grows. The staging copy is recorded before the `clear` span
starts, so no span shows it. Output is bit-identical: every frame's quads match the first
copy-path frame's with a worst corner deviation of 0 px on both paths, and 5 of 5 corpus images
match libapriltag.

`bench_multiqueue --import` (`ultrafast` refine, `single` mode, 1,500 frames per stream, ABBA, three
rounds, `grayimage.pgm`), total fps, median: 1 stream 2,079 -> 2,773 (+33.4%), 4 streams 4,421 ->
4,806 (+8.7%), 8 streams 4,437 -> 4,845 (+9.2%), no mismatches. The multi-stream gain is below the
+28% above, but that table used alternating queue families and this one a shared queue, so the
baselines differ.

Determinism note from building this check: the GPU does not emit candidate quads in a fixed order.
Across frames of the same image the 75 quads are identical in value but arrive in a different
order, so comparing frames needs an order-insensitive match, not a byte compare.

Combining: import subsumes the transfer queue (no staged copy is left to move), so the practical
combination is import plus alternating families, with the transfer queue as the fallback when
import is unavailable.

## 8e. Where the CPU tail goes (measured with the tracer)

`apriltag_vulkan_validate --profile` now reports, per `TagDecoder::Decode()`, how refinement time
splits between quads that go on to decode and quads that never do (`TagDecoder::Stats`). RX 9060 XT,
1280x800 `scale_1`, decimation 2, `exact` refine, 12 threads, 100 iterations, one tag in the scene:

| | |
| --- | --- |
| candidate quads / decoded | 75 / 1 |
| CPU time in the tail, summed over threads | 2.75 ms (parallelism x9.7, wall 0.28 ms) |
| refine, quads that never decode | 2.32 ms = **96.8% of refine time** |
| refine, the quad that decodes | 0.08 ms |
| final `quad_decode_index` | 0.35 ms |
| busiest thread vs mean | 0.256 ms vs 0.250 ms |

Single-threaded `tag_decode` by refine mode on the same image (median of 400): `upstream` 2.29 ms,
`exact` 1.79, `fast` 1.43, `ultrafast` 0.23. The thread sweep scales almost linearly (1.78 ms at 1
thread, 0.90 at 2, 0.46 at 4, 0.31 at 8, 0.27 at 12), so this is compute, not pool wake-up cost, and
the load is already balanced, so scheduling (largest-first ordering, finer chunks) has nothing to
recover on this scene.

What follows from it: on a cluttered scene the tail is almost entirely refinement of rejects, and
`ultrafast` is the only mode that avoids it, at the recall cost recorded in section 4. Any gate that
rejects hopeless quads before refining is therefore worth more than a faster refine kernel. The scene
here is one image; the 96.8% figure will be lower on a scene with many real tags, and
`--profile` will say by how much.

### Vulkan overhead per frame (RX 9060 XT, tracer, median of 180 frames)

One `Detect()` is 611 us: compute spans 380 us (62%), gaps between dispatches 14 us (2%), submit to
first span plus last span to fence return 137 us (22%), and host work outside the submit wait 81 us
(13%: frame memcpy, command recording, readback). The 137 us is the part no span covers; the earlier
empty submit+fence round trip on this card is 42 us, so roughly 95 us of it is unexplained (GPU
front-end start-up and end-of-submission flushes are candidates, not measured).

**Rejected: spinning on the fence.** Polling `vkGetFenceStatus` for up to 2 ms before blocking in
`vkWaitForFences` did nothing: 8 interleaved runs of 500 frames each, `Detect()` median 0.60 ms both
ways, all matching the reference. The 137 us is therefore not OS wake-up latency, and the experiment
was not kept.

### Candidates not yet tried

Measured only where stated; everything else is a hypothesis with its reason and its risk.

Device-agnostic (CPU tail):

- **A cheaper, recall-preserving gate than `ultrafast`'s.** Skip refinement when the unrefined quad's
  four edges show no gradient (mean cross-edge contrast below a threshold), instead of requiring a
  full unrefined decode. Not bit-identical to `exact`; needs the synthetic sweep
  (`apriltag_synth_ground_truth`) to pick a threshold that keeps the 687/768 recall.
- **Bit-identical refine kernel.** `exact` is scalar double with upstream's operation order, so only
  lane-wise vectorisation across samples keeps it identical (no FMA contraction, same four-tap
  order). AVX2 gives 4 double lanes on x86 but NEON gives 2 on the Orange Pi, so the deployment target
  would see little. `fast` (float) already removes 20% on x86.
- **`final quad_decode_index` is 0.35 ms CPU for 75 quads** (4.7 us each). The likely cost is the homography and
  `matd` allocations upstream makes per call, but that was not profiled; confirm before patching
  upstream to reuse buffers.

Device-specific (not measurable on the hardware available when this was written):

- **Mali-G610:** `labelling` is the largest GPU span on the RX 9060 XT too (172 us of ~330 us summed).
  The notes already record tiling and vectorised merges as losses on Valhall; the untested remainder is
  workgroup-size/occupancy sweeps of `uf_merge` per `APRILTAG_VK_UF_CHUNK`, which the tracer's GPU
  track can now time per frame on the Pi.
- **Mali, dma-buf ingest:** section 8d measured the copy path against dma-buf import at 1.52 vs 2.07 ms
  host time; whether the sample app's V4L2 path uses `ImportDmaBufFrame` was not checked.
- **RDNA4:** the GPU phase is ~0.33 ms of a ~1.0 ms frame, so GPU shader work is no longer the limit;
  the tail and the host-side `submit_wait` (0.54 ms median, mostly waiting on the GPU) are.

## 8f. Pre-recorded command buffers: the first version

**Gate.** Before building anything, the cost of recording was derived from the tracer (median of
370 frames, RX 9060 XT, 1280x800): command recording about 22 us plus `begin_commands` (fence wait,
reset, begin) about 5 us, 27 us in all, 5.4% of a 503 us `Detect()` (5.6% once the frame copy is
skipped, section 8d). In pipelined mode the GPU-driver thread idles only 20 us between frames and the
GPU pass (557 of 577 us) is the limit. So the ceiling for removing recording is about 5%, and it
applies only to the fused single-submission path.

**What it does.** The fused first attempt is recorded once into a persistent command buffer
(`Lane::BeginPersistentCommands`) and replayed on later frames (`Lane::SubmitPersistentAndWait`),
cached per (input buffer, labelling chunk size), the only things that vary for that attempt. This
first version replayed by passing a null command buffer through `Detect()` so that every recording
call did nothing; section 8g replaced that with an explicit structure and also caches the retry
path. Not combined with the transfer-queue upload. The host phase timings in `DetectProfile` shrink
on a replay, since there is nothing left to record.

**Results** (RX 9060 XT, ABBA with the env toggle in one binary):

| | off | on | |
| --- | --- | --- | --- |
| `Detect()`, staged copy, 4 rounds x 500 frames | 0.605 ms | 0.577 ms | -4.7% |
| `Detect()`, imported frame | 0.536 | 0.508 | -5.2% |
| `pipeline_total`, staged copy / imported | 1.022 / 0.952 | 0.995 / 0.921 | -2.6% / -3.3% |
| `bench_multiqueue --import`, 1 stream (fps) | 2,699 | 2,958 | +9.6% |
| 4 streams / 8 streams (fps) | 4,502 / 4,521 | 4,470 / 4,520 | -0.7% / 0% |

The `Detect()` saving (28 us) matches the gate's 27 us estimate. The single-stream throughput gain is
larger than that (the 12 runs per setting do not overlap: off 2,565-2,715, on 2,875-3,006); a plausible
cause is less driver work competing with the 12 tail threads, which was not verified. With four or more
streams the GPU is saturated and there is no gain. Output is unchanged: 5 of 5 corpus images match
libapriltag, and `bench_multiqueue` reports no mismatches in any run.

**Synchronisation.** `apriltag_vulkan_validate --import` and `bench_multiqueue --streams 4 --mode
per-stream --import` ran under `VK_LAYER_KHRONOS_validation` with synchronisation validation enabled:
no validation errors and no `SYNC-HAZARD` messages, the same as with pre-recording off. The layer was
confirmed loaded.

This was first left off by default; section 8g measures the Orange Pi and turns it on.

## 8g. Pre-recording, completed: explicit structure, retry replay, and the Orange Pi

**What changed.** `GpuDetector::Detect()` no longer passes a null command buffer around. It is now a
shared upload prologue followed by one of two flows. The fused flow (`DetectFused`) is built from
pure recording functions (`RecordFrameHead`, `RecordFusedTail`, `RecordUfContinuation`,
`RecordTailRerun`: no host state, no clocks, so they can be cached), a cache lookup
(`FirstAttempt`, `Continuation`, `TailRerun`), and `FinishFused` for everything after the last
submission. The 12 null-handle guards are gone. The continuation and tail-rerun buffers are cached
like the first attempt, so with pre-recording a fused frame never records, including a frame whose
labelling did not converge. `DetectUnfused` is the previous flow moved out intact: its dispatches are
sized from host-read counts, so it cannot be pre-recorded, and its `fuse` branches are dead code that
goes when it does. This is therefore not a net reduction in lines: the structure is cleaner, but the
file is larger until the unfused flow is removed. `DetectProfile` gained `record_ms` and `replayed`,
and the tracer a `gpu.record` span. **Pre-recording is now on by default**; `APRILTAG_VK_PRERECORD=0`
records every frame through the same functions. Cached buffers are freed with the detector.

**Why default on.** The rule set before measuring was: on only if the Orange Pi shows no regression
and the desktop keeps its win. Both held (tables below), and nothing regressed anywhere.

**Verification** (RX 9060 XT and Orange Pi 5 / Mali-G610, each against a baseline build of the
previous commit made with the same `CMAKE_BUILD_TYPE`, binaries confirmed different by md5):
- `--dump-quads` (canonical sorted quads at full precision, every frame) is byte-identical to the
  baseline in 15 of 15 cases on each machine: four corpus sizes with pre-recording off and on, the
  unfused path (`APRILTAG_VK_FUSE_SUBMITS=0`), forced retries (`--max-uf 1`, off and on), and four
  stress frames (`--stress noise|snake|checker|rings`).
- Retry replay exercised on purpose: with `--max-uf 1` every frame needs two submissions. With
  pre-recording the ring `begin_commands` span disappears and `gpu.record` falls from a 21 us mean to
  a 0.2 us median lookup (one 387 us first-frame recording).
- `apriltag_synth_ground_truth` passes with identical numbers on both builds on both machines
  (desktop 658/767, Orange Pi 659/767).
- `VK_LAYER_KHRONOS_validation` with synchronisation validation: 0 errors and 0 hazards for the
  default, retry, stress, unfused and pipelined paths and a 4-stream `bench_multiqueue`, on both
  machines. On the Orange Pi the layer prints no notice of its own, so it was confirmed loaded from the
  loader's debug output and shown to be reporting by enabling best-practices checks, which produced a
  message.
- Real retries do not occur: every corpus image and every stress pattern converged in exactly 2
  labelling iterations on both machines (0 of 100 frames retried), so `--max-uf 1` is the only way
  the retry path runs.

**A measurement trap worth knowing.** The same binary measured 0.55 ms `Detect()` in one process and
0.68 ms in the next: the GPU's clock state shifts by up to 20% between runs (`submit_wait` moved from
480 to 598 us). A single ABBA pass briefly showed a spurious "-15%" for a change that does nothing. The
host overhead, `Detect()` minus time inside submit+wait, is not affected and repeats to a few
microseconds, so use it (from `--profile`: `gpu.detect` minus `gpu.submit_wait`) for host-side
changes, and many interleaved runs for anything GPU-side.

**Host overhead per `Detect()`** (median of 6 interleaved runs of 300 frames, 1280x800; base = the
previous commit, 0/1 = pre-recording off/on):

| | base 0 | base 1 | new 0 | new 1 |
| --- | --- | --- | --- | --- |
| RX 9060 XT | 81.5 us | 59.2 us | 83.2 us | 58.0 us |
| Orange Pi (Mali-G610) | 193 us | 127 us | 197 us | 126 us |

The restructuring is performance-neutral on the path that matters (new 0 against base 0, and new 1
against base 1, are within the run spread); pre-recording itself saves 22 us on the desktop and
66 us on the Orange Pi.

**Orange Pi detail** (Armbian 26.8.2, kernel 6.1.115-vendor-rk35xx, Mali g24p0, Vulkan 1.3.276, all
CPU clusters, the DRAM controller and the GPU pinned to `performance`, `Release`):

| | base 0 | base 1 | new 0 | new 1 |
| --- | --- | --- | --- | --- |
| `Detect()` median | 2.790 ms | 2.701 ms | 2.780 ms | 2.694 ms |
| `pipeline_total` median | 4.292 | 4.204 | 4.286 | 4.202 |
| 1 stream (fps) | 361 | 380 | 361 | 381 |
| 2 / 4 streams (fps) | 444 / 445 | 445 / 444 | 446 / 445 | 444 / 445 |

So on the Orange Pi pre-recording is -3.2% on `Detect()`, -2.1% on `pipeline_total` and +5.3% for one
stream; with two or more streams the GPU is the limit (about 445 fps) and nothing changes. On the
desktop (`bench_multiqueue --import`, `ultrafast`): 1 stream 2,774 / 2,775 fps off, 2,988 / 3,009 on
(+7.7% / +8.4%); 4 streams about 4,600 either way; no mismatches in any run.

**Stage 2 gate: not needed.** Making the unfused path obsolete (sizing every dispatch on the device so
it too can be pre-recorded) was gated on Orange Pi data:
- G1: the Mali takes the fused path already (`unified_memory=yes, host_cached=yes`, one submission per
  frame, grey upload written directly).
- G2: 0 retried frames out of 100 on every corpus image and stress pattern.
- G5: the unfused path costs +11.5% on `Detect()` (3.09 vs 2.77 ms) and +8.1% on `pipeline_total` on
  the Mali, so any device stuck on it would pay that.

No measured device is stuck on it. The unfused path stays as the fallback for a device without
host-visible, cached readback memory (the fused path requires both readbacks to be read in place);
reconsider only if such a device turns up.

**Tooling added for this.** `apriltag_vulkan_validate` gained `--dump-quads <file>` (diff one build
against another), `--stress <noise|snake|checker|rings>` and `--max-uf <n>`, and it prints how many
frames needed more than one submission and the labelling-iteration histogram.

## 8h. Orange Pi baseline of the current code, and what a barrier costs

**Setup.** Orange Pi 5 at `15ef4da` (`Release`), 1280x800 corpus image at decimation 2, 300 frames,
all CPU clusters, the DRAM controller and the GPU on `performance`, **and the LumenVision service
stopped** (`systemctl stop lumenvision.service`; it runs at boot and uses about two CPU cores, and it
was running during the 8g measurements). One tag in the image, 75 candidate quads, 0 retried frames
(2 labelling iterations per frame).

**GPU stage table** (`APRILTAG_VK_TIMESTAMPS=1`, last iteration; the first complete per-stage table
for the current code on Mali, and the first one that includes `hash_group`, `scatter` and
`blob_scan`):

| stage | ms | share of GPU total (2.67) |
| --- | --- | --- |
| labelling (`uf_init`/`uf_merge`/`uf_compress`) | 0.730 | 27.3% |
| boundary (`blob_diff`) | 0.363 | 13.6% |
| sort (`sort_points_local`) | 0.335 | 12.5% |
| extents | 0.211 | 7.9% |
| threshold | 0.147 | 5.5% |
| `label_pixels` | 0.095 | 3.6% |
| `uf_final` | 0.087 | 3.3% |
| scatter | 0.082 | 3.1% |
| `hash_group` | 0.059 | 2.2% |
| blob scan | 0.051 | 1.9% |
| clear | 0.049 | 1.8% |
| select | 0.015 | 0.6% |
| readback copy | 0.003 | 0.1% |
| sum of spans | 2.227 | |
| inter-span gaps | 0.121 (+0.020 at the submit boundary) | |

The remaining GPU time between the sum of spans and `submit_wait` (0.324 ms in total, 12%) is the
gaps plus launch and completion. These shares agree with the estimates the 2026-10-14 research notes used
(labelling ~30%, sort ~13%, boundary ~12%), so its ranking stands.

**Whole pipeline**: `GPU total` median 2.663 ms (timestamps off) / 2.674 (on), `quad_decode` 0.295,
`tag_decode` 0.731, `pipeline_total` **3.700 ms**. The same image measured 4.20 ms in 8g with the
service running, so that background process cost about 0.5 ms (12%) of `pipeline_total` and nothing
measurable on `Detect()` (2.66 against 2.70 ms). CPU-tail numbers from before this section carry
that contention; the A/B deltas in 8g are unaffected because each pair was interleaved.

`tag_decode` breakdown (last iteration): 75 quads, 1 decoded, wall 0.73 ms, CPU 4.73 ms (parallelism
6.4x); refine spends 92.7% to 97.7% of its time on quads that never decode (two runs; 8e measured
96.8% on the desktop); busiest thread 0.656 ms against a mean of 0.592 ms.

**What a barrier costs: `bench_barrier`.** The inter-span gap metric cannot see barriers (each is
recorded before the span-end timestamp), so "intra-submit fusion has little left" was never
measured. `bench_barrier` records K independent dispatches of S elements into one persistent command
buffer and replays it with no barrier, an execution-only barrier, or the library's compute-to-compute
memory barrier between them; the slope of frame time against K (16 and 144) is the cost per dispatch,
and the difference between variants is the barrier. Variants are interleaved round-robin, 7 rounds of
60 replays, and the rounds agree to 0.3%.

| elements per dispatch | RX 9060 XT: execution barrier | memory barrier | Mali-G610: execution barrier | memory barrier |
| --- | --- | --- | --- | --- |
| 1 (latency only) | +0.96 us | +0.81 us | +0.78 us | +1.72 us |
| 16,384 | +1.06 | +0.68 | +0.89 | +1.80 |
| 262,144 | +0.47 | +1.85 | +0.95 | +2.34 |
| 1,048,576 | +0.28 | +4.64 | +0.97 | +2.37 |

So a barrier costs about 1 to 2 us on either GPU for small and medium passes: execution drain about
1 us everywhere, plus a cache-flush part that grows with the data written (0.7 to 4.6 us on the
desktop, 1.4 us flat on the Mali). A frame has 33 barriers: about **80 us (3% of GPU time) on the
Mali** and **30 to 100 us on the desktop**, where the ~217 us frame is mostly small, latency-bound
passes (see the 2026-10-14 research notes on branch `research/perf-survey-2026-10`). That bounds everything that removes barriers
or dispatches: on the Mali, deleting a third of them is worth about 1%, so they only pay when
combined with real work removed; on the desktop they are worth testing first.

## 8i. Orange Pi: CPU frequency policy and wake-up settings

Everything in 8c to 8h was measured with the CPU governors pinned to `performance`. That is not how
the board runs: it boots with `ondemand` on all three clusters, and the LumenVision service holds
the A55 cluster at 1.8 GHz. This section measures what that costs and which settings matter.

**Method.** `bench_multiqueue`, one stream, `grayimage.pgm`, paced at 30 and 60 fps (a camera) and
free-running; GPU and DRAM on `performance`; LumenVision stopped; ABBA across 4 to 5 rounds with
every setting applied and verified between runs (`pi_env.sh` records the originals and restores
them). The latency column is the reported latency minus the frame period, because the pipeline
returns a frame's result one `Push()` later by design; it is the extra time a frame spends in the
detector, so a constant of about 1 ms is the floor with the CPU always fast.

| setting | 30 fps | 60 fps | free-running fps |
| --- | --- | --- | --- |
| `ondemand` (the board's default) | 4.7-4.9 ms | 4.2-4.3 ms | 349 |
| `ondemand`, `up_threshold` 20 | 3.2 ms | 3.7 ms | 390 |
| `ondemand`, `up_threshold` 20, 2 ms sampling | 3.4 ms | 3.7 ms | 387 |
| `ondemand`, A76 floor 1.416 GHz | 1.9 ms | 2.0 ms | 347 |
| `schedutil` | 17.2 ms | 11.1 ms | 332 |
| **`performance`** | **1.07 ms** | **1.07 ms** | **384-387** |
| `performance`, process pinned to the A76 cores (`taskset -c 4-7`, 4 CPU threads) | 1.14 ms | 1.13 ms | **408** |

- At camera rates the board's default governor adds **3 to 4 ms to every frame**, more than the
  whole `Detect()` call (2.7 ms): the CPU idles between frames and `ondemand` samples every 10 ms
  at a 95% threshold, so each frame's CPU tail runs while the cores climb from 600 MHz. `schedutil`
  is far worse (11 to 17 ms). Tuning `ondemand` recovers a third; a 1.4 GHz floor on the A76
  clusters recovers most of it (1.9 to 2.0 ms) at the cost of a higher idle clock; only
  `performance` reaches the floor.
- Free-running, `performance` is +10% fps over the default, and pinning to the A76 cores adds
  another +6% (408 fps) by keeping the tail off the A55 cores. Pinning does nothing for paced
  latency.
- **Deployment recommendation:** set the A76 clusters (policy 4 and 6) to `performance`, or at least
  a 1.4 GHz floor, in whatever starts the detector; it is a system setting, not something the
  library can safely do for an application.

Settings that **did not matter** (paced: within the 0.15 ms run spread, so no effect found):

| setting | free-running fps | paced latency |
| --- | --- | --- |
| baseline `performance` | 383 | 1.06-1.27 ms |
| CPU deep idle (`cpu-sleep`, 220 us exit) disabled | 401 (+4.8%) | no consistent change |
| GPU interrupts (CPU0, an A55) moved to the A76 cores | 383 | none |
| both of the above | 397 | none |
| Mali `power_policy=always_on` (default `coarse_demand`) | 382 | none |

So the fence-wake and GPU power-down hypotheses from the research notes are **not** supported on
this load at 30 and 60 fps: the GPU shows no power-up penalty, and moving the interrupt does
nothing. Disabling the deep idle state helps only a free-running stream (about +5%), which is not
what a camera does, so it is not worth recommending. The LumenVision service itself costs about
0.5 ms of `pipeline_total` (8h) while it is running.

## 8j. Desktop: queue families for one stream and for several

`bench_multiqueue --import` with `ultrafast` refine (GPU-limited), RX 9060 XT (family 0 graphics +
compute, family 1 compute-only), pre-recorded command buffers on, 3,000 frames per run, ABBA, zero
mismatches in every run.

One stream, 12 rounds: family 0 **3,253 fps**, family 1 3,121 fps (-4.1%). The compute-only family
helped a single stream in the tables of section 8c (+10% to +24%); that no longer holds with
pre-recording and import, so the single-stream default stays on the first compute family.

Several streams, median of 5 to 6 rounds, total fps:

| streams | all on family 0, one queue | family 0, queue per stream | family 1, queue per stream | alternating, `mode=auto` |
| --- | --- | --- | --- | --- |
| 2 | 4,576 | 4,551 | 5,842 | 5,593 / 5,598 |
| 4 | 4,527 | 4,513 | 7,582 | 8,325 / 8,332 |
| 8 | 4,487 | 4,486 | 7,350 | 8,171 / 8,178 |

(`alternating` is the existing `--mode alternate`; the second figure is the new library helper,
below.) Sharing one queue, or giving every stream its own queue on the graphics family, tops out at
about 4,500 fps; the same card reaches 8,300 fps with the streams spread over both families, +82%
at four streams and +84% at eight. With two streams, both on the compute-only family is 4% faster
than one on each.

**Library change.** `ContextOptions::expected_streams` (default 1) makes the Context also create the
second compute family and one queue per stream spread over the families, and `Context::StreamSlot(i)`
returns the recommended queue slot for stream `i`. With `expected_streams` 1, or on a device with a
single compute family, it changes nothing: `StreamSlot` returns 0, because on the Mali a queue per
stream is slower than a shared one (8c). On the Orange Pi `--mode auto` therefore matches
`--mode single` (2 streams 446 against 446 fps, 4 streams 447 against 447, 1 stream within the run
spread, no mismatches). `bench_multiqueue --mode auto` exercises it and matches `alternate` on the
desktop. `VK_LAYER_KHRONOS_validation` with synchronisation validation reports 0 errors and 0 hazards
for `auto`, `alternate` and `single` at 4 streams (the layer was confirmed loaded by its own startup
message).
## 8k. GPU stage changes from the 2026-10 research: -13.5% `Detect()` on the Mali

Five changes to the GPU stages, each bit-identical to the commit before it, each measured on the
Orange Pi against the previous commit (ABBA, 10 interleaved runs of 400 frames, governors pinned,
LumenVision stopped, 1280x800 at decimation 2). The Orange Pi `Detect()` median after each:

| step | change | `Detect()` median | against the previous step |
| --- | --- | --- | --- |
| 0 | start (`perf/full-prerecord` plus the 8h baseline) | 2.668 ms | |
| 1 | `blob_diff` reads the 1-byte threshold codes and gathers `parent[]`/`blob_size[]` only for pixels with an opposite-coloured neighbour; the `label_pixels` pass (a full rewrite of `parent[]`) is deleted | 2.603 ms | -2.4% |
| 2 | the first attempt's last merge is a verification merge with nothing after it (the guarded compression that followed it was a no-op when it joins nothing) | 2.576 ms | -1.0% |
| 3 | the last real compression also counts blob sizes (`uf_compress_count`), so the first attempt skips `uf_final` | 2.549 ms | -1.2% |
| 4 | `blob_diff` reserves a pixel's boundary slots with one atomic increment instead of up to four | 2.337 ms | **-7.8%** |
| 5 | `sort_points_local`: rank-merge sort (runs of 16 ranked, then merged by binary search) in place of the Batcher network | 2.311 ms | -0.9% |

Original against final in one interleaved run (12 rounds): `Detect()` **2.671 -> 2.311 ms (-13.5%)**,
`pipeline_total` 3.688 -> 3.331 ms (-9.7%), host overhead unchanged at 112 us.

**Throughput** (`bench_multiqueue --mode single`, 3 rounds of 800 frames, no mismatches):

| streams | Orange Pi, before | after | RX 9060 XT (`--import`, `ultrafast`), before | after |
| --- | --- | --- | --- | --- |
| 1 | 379 fps | 452 fps (+19%) | 3,070 | 3,357 (+9.3%) |
| 2 | 447 | 524 (+17%) | | |
| 4 | 447 | 524 (+17%) | 4,734 (one queue) | 5,227 (+10.4%) |
| 4, alternating families | | | 8,361 | 9,137 (+9.3%) |

On the RX 9060 XT `Detect()` falls from 0.533 to 0.473 ms in the clock state both runs share (-11%;
its clock bimodality of 0.53 or 0.59 ms makes the plain median unreliable, 16 rounds each).

**Per-stage spans on the Orange Pi** (timestamps on, last iteration), before and after:

| stage | before | after |
| --- | --- | --- |
| labelling | 0.730 | 0.768 (now includes the blob-size count) |
| `uf_final` | 0.087 | 0.004 (only the tail rerun counts) |
| `label_pixels` | 0.095 | deleted |
| boundary (`blob_diff`) | 0.363 | 0.276 |
| extents | 0.211 | 0.170 |
| scatter | 0.082 | 0.066 |
| hash group | 0.059 | 0.051 |
| sort | 0.335 | 0.251 |
| GPU total | 2.67 | 2.31 |

The three stages after `blob_diff` (extents, scatter, hash group) also got faster with the one-atomic
change; the cause is not established (the points of one pixel are now contiguous in the buffer they
read).

**The sort.** With the network removed outright (a bound build, wrong output) the sort span fell from
0.374 to 0.066 ms, so the network cost about 0.31 ms. Replacing it by ranking alone was only a win up
to 256 points per blob (ranking a blob of n points costs n^2 shared-memory reads: 128 -1.1%, 256
-1.1%, 512 +1.9%, 1024 +18%). The rank-merge sort sorts runs of 16 by ranking and merges them pairwise,
each point binary-searching its merged position in the other run, so a blob needs about
log2(count/16)+1 barriers instead of 28 to 55. Run length sweep (8 rounds): network 2.34 ms, run 4
2.32, 8 2.31, 16 2.305, 32 2.306, 64 2.34, 128 2.41, 256 2.54. The sort workgroup size stays at 256
(128 is equal, 64 is +7%, 32 is +22%). Both sorts give the same permutation because every packed key
carries its local index. `APRILTAG_VK_SORT_CHUNK=0` restores the network, `APRILTAG_VK_SORT_WG`
sets the workgroup size. The histogram `--profile` now prints shows why the gain is modest: of 136
selected blobs, 99 have at most 128 points (7,204 of 17,650 points) and 10 have more than 256.

**Tried and dropped.** Deferring the transfer-to-compute barrier after the per-frame fills so that
decimate and threshold overlap them (research item F3): 2.3103 against 2.3126 ms, a null; the Mali
runs the fills in order. Not built: the dispatch diet (folding `init_extents`, `merge_extents` and the
count scan into their neighbours) is bounded by the barrier measurement in 8h to about 8 dispatches at
roughly 3.5 us each, near 1% of the Mali GPU time, for several shader rewrites; and run-merging
before the 64-bit extents atomics was already tried as subgroup aggregation and retired on
integrated GPUs (OPTIMIZATION_NOTES item 6).

**Verification** (each step, both machines): the 15-case canonical-quad matrix (four corpus sizes
with pre-recording on and off, the unfused path, forced retries on and off, four stress frames) is
byte-identical to the original in 15 of 15 cases; also `APRILTAG_VK_UF_CHUNK` 1, 3 and 4 (chunk 1
exercises the count inside the head compression and a real retry), the 32-bit storage variant and the
sort run length 32 to 2048. `VK_LAYER_KHRONOS_validation` with synchronisation validation: 0 errors
and 0 hazards for the default, forced-retry, pipelined, stress, unfused and 4-stream runs on both
machines (the Mali layer confirmed loaded by a best-practices control message). `apriltag_synth_
ground_truth`: 658/767 (RX 9060 XT) and 659/767 (Orange Pi), unchanged.

## 8l. Probing a quad before refining it: a margin gate (opt-in)

96.8% of the CPU refine time is spent on quads that never decode (8e). The research notes proposed
sampling the unrefined quad's bits and refining only quads within a Hamming distance of a code. This
section measures that idea with a probe built into `TagDecoder` and finds a better signal.

**Tooling.** `cmake/patches/apriltag-expose-code-probe.patch` exposes `quad_sample_code()` (upstream's
bit sampling for a quad, up to the code lookup) and `apriltag_min_hamming()`. With
`APRILTAG_VK_PROBE_STATS=1` `TagDecoder` histograms, for every candidate quad, the Hamming distance
from its unrefined bits to the nearest code, split by whether the full refine + decode accepted it;
`APRILTAG_VK_PROBE_CSV=<file>` appends one row per quad (distance, decision margin, perimeter,
accepted). `apriltag_synth_ground_truth --contrast <f>` scales the frame's dynamic range about
mid-grey. Data below: the synthetic sweep (768 cases, blur 0.7, noise sigma 3, `--refine-edges`,
74k candidate quads of which 705 decode) and the corpus photographs.

**A Hamming cutoff does not work.** Many tags that decode only decode after refinement; their
unrefined bits are far from any code:

| skip refinement when the nearest code is farther than | accepted tags lost | junk quads skipped |
| --- | --- | --- |
| 3 | 20.3% | 99.9% |
| 5 | 18.9% | 99.1% |
| 7 | 14.6% | 82.5% |
| 9 | 9.1% | 43.6% |
| 10 | 0% | 37.2% |

Zero loss needs a cutoff of 10 of 36 bits, which skips only 37% of the junk (random 36-bit words sit
around distance 8 from the nearest of 587 codes x 4 rotations). The earlier estimate of "98% of
clutter rejected at distance 4-5" holds for the junk, but not for the tags it would lose.

**The decision margin does.** The same sampling yields upstream's decision margin (the contrast
between the white and black bit cells, in grey levels). Junk quads sit at the noise floor; real tags
do not:

| contrast | accepted: min / median margin | junk: p99 / max margin | skip margin < 4: tags lost, junk skipped |
| --- | --- | --- | --- |
| 1.0 | 74.4 / 226 | 3.7 / 127 | 0, 99.7% |
| 0.5 | 36.9 / 114 | 3.7 / 64 | 0, 99.7% |
| 0.25 | 16.5 / 57 | 3.7 / 32 | 0, 99.8% |
| 0.12 | 8.1 / 26 | 3.7 / 18 | 0, 99.7% |

Polarity failures (negative margin) are 37% of the junk and were never an accepted tag. On the corpus photographs the accepted tags have margin 101 and the junk reaches 24,
so a gate at 4 skips 87% of the junk there and a gate at 8 skips 91%.

**The gate.** `TagDecoder::SetProbeMargin(m)` / `APRILTAG_VK_PROBE_MARGIN=<grey levels>` (off by
default): when edge refinement is on and exactly one family is registered, each quad is sampled
unrefined and rejected without refinement if its margin is below `m` (or its polarity is wrong).
Every quad that passes is refined and decoded as before, so **detections are identical to an ungated
run unless a real tag's unrefined margin is below the gate**. The per-case CSV of the synthetic
sweep is byte-identical to the ungated run at gate 4 and 8 for contrast 1.0, 0.25, 0.12 and noise
sigma 3 and 8 (and 8 with contrast 0.25) on the RX 9060 XT, and at gate 4 on the Orange Pi. The risk is a real tag below the gate: a tag needs a swing of under 4 grey levels
(it is 8 at contrast 0.12 in the sweep), and a higher noise level only raises the junk margins, which
keeps the gate safe and makes it skip less.

**Result on the Orange Pi** (`--profile`, 1280x800 corpus image, 8 interleaved runs of 400 frames,
governors pinned, LumenVision stopped, `APRILTAG_VK_PROBE_MARGIN` 0 against 4): `pipeline_total`
**3.331 -> 2.915 ms (-12.5%)**, `Detect()` unchanged (2.305 against 2.312 ms).


## 8m. Against v1.6.0

Version 1.7.0 against v1.6.0 (`5f578a0`). Both sides were built from clean trees with the same
build type (RelWithDebInfo on the RX 9060 XT, Release on the Orange Pi) and run interleaved,
1280x800 at decimation 2. "Gate 4" is 1.7.0 with `APRILTAG_VK_PROBE_MARGIN=4`.

**Orange Pi 5** (Mali-G610; governors pinned, LumenVision stopped; 6 rounds of 400 frames):

| | v1.6.0 | 1.7.0 | 1.7.0, gate 4 |
| --- | --- | --- | --- |
| `Detect()` median | 2.487 ms | 2.118 ms (-14.8%) | 2.120 ms |
| `pipeline_total` median | 3.497 ms | 3.125 ms (-10.6%) | 2.706 ms (-22.6%) |
| 1 stream, `ultrafast` | 387 fps | 439 (+13%) | 445 (+15%) |
| 2 streams, `ultrafast` | 447 | 522 (+17%) | 524 |
| 4 streams, `ultrafast` | 446 | 525 (+18%) | 524 |
| 1 stream, `exact` refine | 350 | 432 (+23%) | 439 (+25%) |
| 4 streams, `exact` refine | 447 | 523 (+17%) | 524 |

**RX 9060 XT** (5 rounds; `bench_multiqueue --import`, 3,000 frames):

| | v1.6.0 | 1.7.0 | 1.7.0, gate 4 |
| --- | --- | --- | --- |
| `Detect()` median | 0.573 ms | 0.480 ms (-16%) | 0.460 ms |
| `pipeline_total` median | 0.994 ms | 0.889 ms (-10.6%) | 0.695 ms (-30%) |
| 1 stream, `ultrafast` | 2,760 fps | 3,310 (+20%) | 3,313 |
| 4 streams, one queue, `ultrafast` | 4,745 | 5,271 (+11%) | 5,280 |
| 4 streams, alternating families, `ultrafast` | 8,347 | 9,096 (+9%) | 9,086 |
| 1 stream, `exact` refine | 2,278 | 2,331 (+2%) | 3,186 (+40%) |
| 4 streams, `exact` refine | 3,011 | 3,044 (+1%) | 4,974 (+65%) |

With `exact` refinement the CPU tail is the limit, so only the margin gate moves the throughput; with
`ultrafast` the GPU is the limit and the GPU changes carry it. The gate does nothing for `ultrafast`,
which already has its own unrefined-decode gate.

**Results are unchanged.** `apriltag_synth_ground_truth` prints the same figures on v1.6.0 and the
1.7.0 build, with and without `--refine-edges` (658/767 and 687/768 decoded, identical mean and
worst rotation and translation errors against ground truth and against libapriltag), every
validate run reports MATCH, and no `bench_multiqueue` run had a mismatch. Everything else is
verified against the previous commits in 8g, 8k and 8l (canonical quad dumps, validation layer).
1.7.0's defaults differ from v1.6.0's in pre-recorded command buffers, the 8k GPU stage
changes (including the rank-merge sort) and the removal of `label_pixels`, all on; the margin gate and
`expected_streams` are opt-in.

## 8n. The two remaining GPU items, measured, and the fallback matrix

**`blob_diff` counter (A6) and the small-blob sort (A8).** Bound builds on the Orange Pi (3 interleaved
rounds of 300 frames, governors pinned, LumenVision stopped, output wrong by design):

| build | span | original | bound |
| --- | --- | --- | --- |
| `blob_diff` without the append atomic (one thread sets the counter) | boundary | 0.275 ms | 0.264 ms |
| blobs of up to 128 points left unsorted | sort | 0.30 ms | 0.22 ms |
| same | GPU total (median) | 2.299 ms | 2.244 ms |

The atomic is worth at most 0.011 ms (0.5% of the GPU time), inside the run spread: **A6 is rejected**.
Not sorting the small blobs bounds A8 at 2.4%. The one buildable form, ranking every blob of up to 128
points in one pass (no merge rounds), measured slower: sort 0.245-0.265 -> 0.273-0.321 ms and GPU
median 2.30 -> 2.30-2.39 ms, so **A8 is rejected** (the n^2 reads cost more than the barriers saved,
as in 8k). The desktop was not measured.

**Fallback matrix.** The 15-case canonical-quad matrix (as in 8g) is byte-identical to the golden
dumps in 15 of 15 cases for each setting below on both machines, and `apriltag_synth_ground_truth` is
unchanged (658/767 RX 9060 XT, 659/767 Orange Pi):

| setting | what it forces |
| --- | --- |
| `APRILTAG_VK_FORCE_NO_8BIT=1` | 32-bit image storage instead of `u8` shaders |
| `APRILTAG_VK_FORCE_NO_SUBGROUP=1` | no subgroup extents reduction |
| `APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1` | no 64-bit extents atomics |
| `APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING=1` | `uf_compress` skip decided in the shader |
| all four together | the minimum feature set |
| `APRILTAG_VK_MAX_INVOCATIONS=256` / `128` | smaller workgroups (desktop: 1D 128, 2D 8x8) |
| `APRILTAG_VK_WG=64` + `APRILTAG_VK_SORT_WG=64`, `APRILTAG_VK_WG2D=8x8`, `APRILTAG_VK_SORT_CHUNK=0` | workgroup shapes, sorting network |

The Orange Pi's Mali has no float64, no conditional rendering and no host-pointer import, so its
default run already covers those paths. With `VK_LAYER_KHRONOS_validation` and synchronisation
validation on the desktop, the default, `--max-uf 1` (forced retries), pre-recording off, the `rings`
stress frame and a 4-stream `bench_multiqueue` run give 0 errors and 0 hazards for the default,
all-four-fallbacks, `MAX_INVOCATIONS=128` and sorting-network settings. Only one device (the
RX 9060 XT) is installed on the desktop and no software device (lavapipe, SwiftShader) is available,
so Vulkan 1.1 hardware from other vendors remains untested. A positive control for the layer was
inconclusive with the current `VK_LAYER_VALIDATE_*` settings; with the older `VK_LAYER_ENABLES` form
the layer reported that it had read the setting.

## 8o. The hypotenuse in the quad fit

`std::hypotf` is called in `FitLineError` and `FitLine` (`QuadDecode.cpp`), about 3,900 times per frame
on the 1280x800 corpus image. A profiling hint suggested `sqrtf(x*x + y*y)`, which is 8 times faster on
an A76. Three forms were compared: `hypotf`, the double form (`(float)sqrt((double)x*x + (double)y*y)`,
"A") and the float form ("B").

**Profile (Orange Pi, `perf record` on the A76 cores, `APRILTAG_CPU_THREADS=4`, `exact` refinement).**
Of all CPU samples `hypotf` is 2.2%, because 68% is edge refinement. Of the quad-fit work itself
(the `QuadDecode::Decode` task, `ReadMomentsWindow` and `hypotf`) it is about 12%, which matches the
hint's 14%. After the change `hypotf` is 0.15% (the remainder is the double `hypot`).

**Micro-benchmark** (2 x 10^6 moment-style inputs and 2 x 10^6 wide-range inputs, `-O3`, one core):

| | `hypotf` | A (double) | B (float) |
| --- | --- | --- | --- |
| A76, ns per call | 7.36 | 3.06 (2.4x) | 0.91 (8.1x) |
| A55, ns per call | 34.5 | 10.2 (3.4x) | 5.3 (6.5x) |
| RX 9060 XT host (MSVC), ns per call | 2.6 | 1.95 (1.3x) | 1.08 (2.4x) |
| results differing from `hypotf` | **0** | **0** | 14.5% / 5.0% (1 ulp) |

**Output.** A reproduces the canonical quad dumps byte for byte (15 of 15 on both machines), the
synthetic sweep (658/767 and 659/767; 687/768 with `--refine-edges`) and every validate tool. B changes
all 15 dumps on both machines (a 1 ulp change in the eigenvalue and the normal moves corners), so it is
not used. A is the shipped form.

**Speed** (ABBA, 8 rounds of 400 frames, 16 runs per variant, medians; Pi governors pinned, LumenVision
stopped):

| | A76, 4 threads | A76, 1 thread | RX 9060 XT, 12 threads | RX 9060 XT, 1 thread |
| --- | --- | --- | --- | --- |
| `quad_decode`, A against `hypotf` | -2.9% (0.2747 -> 0.2667 ms) | -2.3% | -3.2% (0.1176 -> 0.1138 ms) | -0.9% (within spread) |
| `quad_decode`, B against `hypotf` | -7.5% | -5.3% | -6.9% | -2.7% |
| `pipeline_total`, A | -0.1% | -0.2% | -1.0% | -0.3% |

`perf stat` over 400 frames on one core: A76 cycles 4,149 -> 4,136 million (-0.3%), instructions 7,068
-> 7,021 million (-0.7%); A55 cycles 9,471 -> 9,403 million (-0.7%, about 0.09 ms of CPU time per
frame). The saving is real but small: about 0.01 ms per frame on an A76 and 0.09 ms on an A55, under 1%
of `pipeline_total` with `exact` refinement. It is larger where the tail runs on the A55 cores or
refinement is off (`ultrafast`).

**Correctness.** With the double form, GCC 14 AddressSanitizer + UBSan (including `float-cast-overflow`
and signed overflow) report nothing on the corpus, the four stress frames, `--max-uf 1`, decimation 1
and both synthetic sweeps; ThreadSanitizer reports nothing with 8 CPU threads and pipelined. LeakSanitizer
reports 36 MB, all reachable from the Vulkan loader's instance creation, plus 664 bytes allocated by
upstream's `apriltag_detector_create` in the validate tool; none involves this change.

## 8p. Profiling the CPU tail on the Orange Pi: three exact changes, and what is left

**Method.** `bench_multiqueue` (the library alone, one stream with `FramePipeline`) in two
configurations: `ultrafast` refinement at decimation 1 on a 1280x720 frame (the common use; 430
selected blobs, 47,990 boundary points, 6.5% of blobs fall back from DP to the combinatorial search)
and `exact` refinement at decimation 2 on `grayimage.pgm`. Profiling build with `-g
-fno-omit-frame-pointer`; `perf record -g` and `perf report --sort srcline`, `perf annotate`, `perf
stat` (user-mode counters), `perf record -e sched:sched_switch` (off-CPU), `heaptrack`; governors
pinned, LumenVision stopped. `valgrind --tool=callgrind` stops with an illegal instruction inside
the Mali driver, so instruction counts come from `perf stat` instead.

**What the profile showed.** With `ultrafast` at decimation 1, 77% of all CPU samples were the quad
fit, and 49% of cycles were back-end stalls (IPC 0.93). Three causes, each fixed without changing
any output:

| change | where | Orange Pi | RX 9060 XT |
| --- | --- | --- | --- |
| the prefix sums are accumulated in scalars and stored field by field; the struct copy through the stack made every 16-byte load overlap the previous 8-byte stores, which defeats store-to-load forwarding (two instructions held 21% of the quad-fit samples) | `FitQuadForBlob` | `quad_decode` -9% (decimation 1), -11% (decimation 2) | |
| the combinatorial search fits every ordered pair of peaks once (at most 90 fits) instead of refitting three of the four segments inside the loops (up to about 500 fits per fallback blob); `ReadMomentsWindow` is inlined so its 48-byte result stays in registers | `FitQuadForBlob` | `quad_decode` a further -15% (decimation 1), -21% (decimation 2) | |
| `exact` refinement reuses each bilinear tap: the tap at n - 1 is the n + 1 tap of eight steps earlier, computed from bit-identical operands, so it is kept in an eight-entry ring (as `RefineEdgesFast` already did) | `RefineEdgesT` | `tag_decode` -12% (0.720 -> 0.635 ms) | `tag_decode` -16% |
| the DP corner search's four argmax passes run as four independent chains, merged with the lower index winning ties; squared distances are computed in `int64` (exact for these coordinates) and the perpendicular test squares in double as before | `FindDpCornerIndices` | `quad_decode` a further -3.8% (decimation 1), -3.1% (decimation 2), -5.2% on one A76 thread; CPU time -4.8% | `quad_decode` -3.9% (decimation 1) |
| the first three together | | CPU time per frame -24% (`ultrafast`, decimation 1), -13% (`exact`, decimation 2); `pipeline_total` 3.13 -> 2.99 ms | `exact` throughput +12% (1 stream), +17% (4 streams); `ultrafast` unchanged (GPU-limited) |

Over 2,000 frames with `ultrafast` at decimation 1, task-clock fell from 9,316 to 7,104 ms, cycles
by 26%, and IPC rose from 0.93 to 1.16. Frame rate does not change there because the GPU pass (5.2
ms at decimation 1) is the limit; the saving is CPU time and per-frame latency.

**Verification.** The 15 canonical quad dumps are identical on both machines, as are the decimation-1
dumps and the dumps with `APRILTAG_VK_QUADFIT=peaks` (every blob through the combinatorial search).
`apriltag_vulkan_validate --dump-detections <file>` (new) writes every decoded tag's id, margin,
centre and corners at full precision: identical before and after for `exact` and `upstream`
refinement at decimations 1 and 2 on four images, and `exact` still equals upstream's
`refine_edges` bit for bit. The synthetic sweep's per-case CSV is identical with `--refine-edges`
at contrast 1.0 and 0.25. GCC AddressSanitizer + UBSan (with `float-cast-overflow`) report nothing
on the corpus in `exact` and `ultrafast`, decimation 1, peaks-only, the `noise` and `rings` stress
frames and the synthetic sweep; ThreadSanitizer reports nothing with 8 CPU threads, pipelined and
peaks-only.

**Found and not changed** (largest first; shares of all CPU samples after the changes):

- Tried and rejected: finding each blob's end by galloping and bisection instead of scanning all
  48,000 points (`QuadDecode::Decode`, 7.4% of samples). Output is identical and the Pi gains 1 to 3
  points of `quad_decode`, but on the RX 9060 XT `quad_decode` becomes 18% slower with 12 threads:
  the sequential scan is what brings the read-back points into the cache, and without it each worker
  takes those misses itself. Its samples were mostly that first touch, not the comparisons.
- `memcpy`, 5.3%: the frame upload into the mapped buffer on the GPU-driving thread, before the
  submit (on the critical path when the GPU is the limit; dma-buf import avoids it, section 8d), and
  `FramePipeline`'s copy of the line-fit points (double-buffering the read-back avoids it).
- `exact`, decimation 2: refinement is still 72%; the rest of its cost is the sequential
  `Mn += weight * n` accumulation, which must stay in order for bit-identity, and the tap
  arithmetic. The margin gate (section 8l) removes most of it: `pipeline_total` 2.99 -> 2.59 ms.
- About 1,000 heap allocations per frame (`heaptrack`, `ultrafast` decimation 1; 1,400 with
  `exact`), mostly upstream's `matd` matrices for the homography in `quad_decode`, each 16 to 72
  bytes; about 1% of samples in `calloc`/`free`.
- Off-CPU time is the worker pool and `FramePipeline` waiting on condition variables and the GPU
  fence; no lock contention showed up.

## 8q. Profiling the Mali GPU with Arm's tools

**Tools.** The Mali Offline Compiler (`malioc`, Arm Performance Studio 2026.5; the Linux build, run
under WSL) for static analysis of every shader's SPIR-V for the Mali-G610; Arm's libGPUCounters for
the G610's hardware counters on the Orange Pi (`tools/mali_counters`: `hwcnt_run` prints all 270
counters over a command's run, `hwcnt_timeline` samples selected counters every 200 us); the Mali
driver's sysfs controls under `/sys/devices/platform/fb000000.gpu` and the Rockchip devfreq nodes.
Streamline was not needed: the timeline sampler gives the same per-stage view headless.

**Static analysis.** Every shader is load/store-bound, with almost no arithmetic, no register
spilling and full occupancy, except `merge_extents` (64 registers, 50% occupancy, a small
dispatch). Longest load/store paths: `scan_block` 30 cycles, `blob_diff_u8` 29, `sort_points_local_u8`
22, `merge_extents` 18, `block_minmax_u8` 17, `init_extents` 16.

**Hardware counters** (`ultrafast`, decimation 1, 1280x720, 2,000 frames). The GPU reads 27.4 MB and
writes 21.9 MB of DRAM per frame, about 53 times the frame's size; the shader cores report no
instruction to issue for most of their active cycles (waiting on memory), the FMA pipe is idle (0.2%)
and every load is a partial-width access. The level-3 TLB misses about 800 times per frame
(negligible, so the large-page pool is not worth enabling). Sampling every 200 us at 20 fps and
aligning the frames places the traffic: about 9 MB of writes in the first 0.8 ms (clears, threshold,
labelling initialisation), about 17 MB of reads and 8 MB of writes in labelling, the atomics in the
boundary, hash and extents stages, and high starvation with little traffic in the sort.

**Changes** (each identical in output: the 15 canonical quad dumps on both machines, about 100
further configurations including the 32-bit path, stress frames, forced retries, `--import`,
`--pipelined` and dma-buf import, the synthetic CSV; validation layer with synchronisation validation
clean on both machines):

| change | Orange Pi | RX 9060 XT |
| --- | --- | --- |
| `uf_init` initialises 4 pixels of a row per thread and points each at the start of its run within the chunk, so the following compression walks chains 4 times shorter (runs of 8, 16, 32 and 64 were slower: too few threads, serial loops) | labelling -13.5% (decimation 1), -11% (decimation 2); `Detect()` -5.4% and -4.3%; `ultrafast` throughput +5.7% (decimation 1), +3.0% (decimation 2) | labelling about -11% |
| at decimation 1 with 8-bit storage the threshold and sort stages read the grey buffer as the decimated image and the copy pass is skipped (imported frames are decimated into the grey buffer instead) | `Detect()` -2.6% (decimation 1) | |
| `threshold` covers 4 pixels per thread, reading each 4x4 block's min/max once | threshold span 0.35 -> 0.28 ms (decimation 1); with the alias `Detect()` -4.0% (decimation 1), -1.5% (decimation 2), throughput +3.7% | |

Together on the Orange Pi: `Detect()` 5.21 -> 4.78 ms at decimation 1 (-8%) and 2.12 -> 2.04 ms at
decimation 2; load/store issue cycles -15%, starvation -10%, DRAM reads -3% and writes -6% per frame.
`APRILTAG_VK_UF_INIT_RUN`, `APRILTAG_VK_THRESH_RUN` and `APRILTAG_VK_DEC1_ALIAS=0` restore the old
behaviour for comparison.

**Measured and not changed.** Workgroup size (1D 64 to 512, 2D 8x8 to 32x8): within 1% everywhere.
The sort's workgroup: 128 and 64 threads are 50% and 120% slower at decimation 1 and 512 or 1,024 no
faster, because its time is set by the largest blobs. The GPU (1 GHz) and DDR (2112 MHz) clocks were
already at their maximum with the `performance` governors, and all four shader cores are enabled
(`core_mask` 0x50005). Labelling remains the largest stage (39% of the GPU spans at decimation 2) and
is bound by DRAM traffic on a 4-byte-per-pixel `parent[]` array that does not fit in the L2.
The Mali has no host-pointer import (`VK_EXT_external_memory_host`), so `bench_multiqueue --import`
warns and runs without it there; dma-buf import is the Mali path.

## 8r. Profiling round 3: the sort's largest blob, and two frames in flight

**Profile.** Every profiler from sections 8p and 8q again, on both machines, in two configurations:
`ultrafast` with margin gate 4 at decimation 1 on a 1280x720 frame (the common use) and `exact` at
decimation 2. Every configuration is limited by the GPU (Orange Pi: 5.0 ms of GPU work in a 5.6 ms
serial frame at decimation 1). The Mali's own utilisation counter read 82% to 93% with one pipelined
stream, so the GPU idled between frames. The sort was 19% of the GPU spans at decimation 1 with
little memory traffic and high starvation in the counters. The DRAM controller was 34% loaded (46.8 MB
per frame). On the CPU, the 6.5% of blobs that fall back from the DP corner search took about 60% of
the quad fit's time, and `heaptrack` counted 1,000 to 1,400 allocations per frame (upstream's `matd`).
callgrind still stops inside the Mali driver, `perf annotate` crashes on this `perf` build, and AMD's
Radeon GPU Analyzer was not run.

**The sort is set by its largest blob.** One workgroup sorts each blob, so the one blob of more than
2,048 points at decimation 1 runs on a single core while the rest of the GPU waits. Bound first, by
leaving blobs above a cap unsorted (wrong output): cap 2,048 took the sort span from 863 to 550 us and
`Detect()` down 7.8% on the Orange Pi; cap 1,024 down 10%; on the RX 9060 XT the sort fell from 132 to
54 us. Halving the shared memory (the sorting network in place of the rank-merge) changed nothing, so
the limit is the serial tail, not occupancy.

**Changes** (each identical in output: the 15 canonical quad dumps on both machines, 60 configurations
against the previous commit on each machine at decimations 1, 2 and 4 with four stress frames, every
split length from 64 to 2,048 at decimations 1, 2 and 4, the split on the fused, unfused,
non-pre-recorded and retry paths; dma-buf import at decimations 1 and 2, grey and YUYV; validation layer
with synchronisation validation clean on both machines; ThreadSanitizer, AddressSanitizer and UBSan
clean on the new paths):

| change | Orange Pi | RX 9060 XT |
| --- | --- | --- |
| blobs of more than 1,024 points are listed by the per-blob sort, sorted in runs of 1,024 by one workgroup each, and merged by whole-GPU rounds of one thread per point (`sort_points_local_body.glsl` modes 1 and 2); on by default at decimation 1 only | sort span 824 -> 579 us; `Detect()` 4.77 -> 4.49 ms (-5.8%); `pipeline_total` 5.38 -> 5.08 ms; 1 stream 202 -> 215 fps; 4 streams on one queue 217 -> 232 fps | sort span 131 -> 100 us; `Detect()` 0.66 -> 0.59 ms (-10%); `pipeline_total` -7%; 1 stream +6%; 2 and 4 streams on one queue +6% and +7%; **2 and 4 streams spread over both compute families -5% and -10%** |
| `FramePipeline` takes an optional second `GpuDetector` (with its own `vk::Lane`) and alternates frames between them, so the next GPU pass starts without waiting for the host; detections arrive two `Push()` calls late; opt-in | Mali utilisation 93% -> 99% (decimation 1) and 83% -> 99% (decimation 2); 1 stream: decimation 1 215 -> 233 fps, decimation 2 `exact` 479 -> 555, `ultrafast` gate 4 435 -> 555 | 1 stream: decimation 1 2,183 -> 2,843 fps, decimation 2 `exact` 2,413 -> 2,630, `ultrafast` gate 4 2,604 -> 4,480 |

Split length: 512 was 0.7% better than 1,024 on the Orange Pi at decimation 1 and 5% worse on the
desktop; 2,048 gained little and 256 lost (more merge rounds). At decimation 2 (a 2,048-point cap and
one round) the split cost 2% on the Orange Pi, so it stays off there. With streams spread over both
compute families of a discrete GPU the other streams already hide the long sort and the split's extra
dispatches are pure cost: set `APRILTAG_VK_SORT_SPLIT=8192` there.

Depth 2 is a throughput change: it doubles the device memory and, at full rate, the push-to-result
latency (`bench_multiqueue`, free-running) goes from 9.2 to 12.9 ms on the Orange Pi at decimation 1
and from 0.91 to 1.05 ms on the desktop. With several streams it changes nothing measurable; they
already keep the GPU busy. `bench_multiqueue --depth 2` and `apriltag_vulkan_validate --pipelined
--pipeline-depth 2` select it; the pipelined path of the validate tool now writes `--dump-quads` and
`--dump-detections`, which match the serial path's byte for byte at depths 1 and 2.

**Measured and not shipped:**

- **The quad refit inside the parallel loop** (it runs serially after the per-blob fits): `quad_decode`
  -25% on the desktop at decimation 1 with identical output there, but on the Orange Pi it changed the
  last bit of the corner coordinates in 40 of 60 configurations. GCC on AArch64 fuses `a * b + c`
  into multiply-adds by default (`-ffp-contract=fast`), and moving the code changes which ones; an
  out-of-line function did not restore them. It could ship with `-ffp-contract=off` for the library,
  which would change every GCC build's dumps once.
- **Fitting the largest blobs first:** `quad_decode` +6% on the Orange Pi at decimation 1, -5% at
  decimation 2; desktop 0% and -12%. Output identical; not worth the mixed result.
- **Zeroing `blob_size` in `uf_init`** in place of the per-frame fill: identical output, no gain
  (`Detect()` 4.55 ms against 4.51 ms with the fill).
- **Skipping the boundary points of clusters that selection rejects** (71% of the 166,000 points at
  decimation 1): the most it could save is the downstream stages' share, about 0.44 ms; knowing the
  clusters first needs a counting pass, and `blob_diff` with its point stores removed (a bound build)
  still takes 573 of its 749 us. Not built.

**Two earlier problems fixed alongside.** The transfer-queue upload (`--async-upload`) wrote each
frame into the grey buffer with nothing on that queue ordering it after the previous frame's copy;
synchronisation validation reported a write-after-write hazard (10 per 100 frames, also in 1.7.0). A
transfer-to-transfer barrier now opens each upload command buffer, and the report is gone. The
validate tool now releases its detectors, the reference detections and the tag family per image, so
LeakSanitizer is clean on it.

## 8s. Against v1.7.0

Version 1.8.0 (sections 8o to 8r) against v1.7.0 (`2626760`). Both sides built the same way (`Release`
on the Orange Pi, `RelWithDebInfo` on the RX 9060 XT), run alternately: six rounds of 1,000 frames of
`apriltag_vulkan_validate` (medians of the round medians) and three rounds of `bench_multiqueue` (800
frames per round on the Orange Pi, 3,000 on the desktop), with the Orange Pi's governors on
`performance` and LumenVision stopped. Three configurations: decimation 1 on a 1280x720 frame with
`ultrafast` refinement and margin gate 4, and decimation 2 on the 1280x800 corpus image with `exact`
and with `ultrafast` at gate 4. "Depth 2" is `FramePipeline` with a second detector (opt-in).

Orange Pi 5:

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 5.185 -> 4.505 ms (-13%) | 2.110 -> 1.996 ms (-5.4%) | 2.113 -> 2.013 ms (-4.7%) |
| `quad_decode` | 0.675 -> 0.459 ms (-32%) | 0.289 -> 0.190 ms (-34%) | 0.286 -> 0.190 ms (-34%) |
| `tag_decode` | 0.127 -> 0.124 ms | 0.722 -> 0.637 ms (-12%) | 0.169 -> 0.173 ms |
| `pipeline_total` | 6.001 -> 5.089 ms (-15%) | 3.125 -> 2.830 ms (-9.4%) | 2.582 -> 2.385 ms (-7.6%) |
| 1 stream | 187 -> 216 fps (+15%) | 418 to 457 -> 442 to 464 fps (noisy) | 437 -> 475 fps (+8.7%) |
| 1 stream, depth 2 | 233 fps (+25%) | 556 fps (+29%) | 555 fps (+27%) |
| 4 streams, one queue | 199 -> 232 fps (+16%) | 523 -> 554 fps (+6%) | 524 -> 555 fps (+6%) |
| latency, 1 stream (depth 1 / depth 2) | 10.7 -> 9.26 / 12.9 ms | 4.6 -> 4.4 / 5.4 ms | 4.6 -> 4.2 / 5.4 ms |

RX 9060 XT:

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 0.673 -> 0.587 ms (-13%) | 0.471 -> 0.455 ms (-3.4%) | 0.467 -> 0.445 ms (-4.7%) |
| `pipeline_total` | 1.000 -> 0.882 ms (-12%) | 0.852 -> 0.787 ms (-7.6%) | 0.685 -> 0.652 ms (-4.8%) |
| 1 stream | 2,041 -> 2,151 fps (+5%) | 2,179 -> 2,339 fps (+7%) | 2,553 -> 2,544 fps (0%) |
| 1 stream, depth 2 | 2,893 fps (+42%) | 2,601 fps (+19%) | 4,466 fps (+75%) |
| 4 streams, one queue | 2,920 -> 3,130 fps (+7%) | 2,942 -> 3,544 fps (+20%) | 4,734 -> 4,941 fps (+4%) |
| 4 streams, both compute families | 3,713 -> **3,573 fps (-4%)** | 2,895 -> 3,455 fps (+19%) | 7,065 -> 7,276 fps (+3%) |

The one loss is the sort split (8r) with streams spread over both compute families at decimation 1;
with `APRILTAG_VK_SORT_SPLIT=8192` the same run gives 3,946 fps (+6% over 1.7.0). With several streams
on the Orange Pi depth 2 changes nothing; the streams already keep the GPU busy.

Output: the quad dumps of the two versions are byte-identical in 60 configurations on each machine
(five images, decimations 1, 2 and 4, no stress and the `noise`, `rings` and `snake` stress frames),
the 15 canonical dumps match, and `bench_multiqueue` reported no mismatches in any run above.

Sanitisers on the release tree: AddressSanitizer with UBSan (including `float-cast-overflow`) and
LeakSanitizer, and ThreadSanitizer, on the Orange Pi (GCC 14); AddressSanitizer on the RX 9060 XT
(MSVC). The runs covered the corpus at decimations 1, 2 and 4, four stress frames, the retry path,
`FramePipeline` at depths 1 and 2, the unfused, non-pre-recorded and split-sort paths, every
optional-feature fallback, the four refine modes with the margin gate, `bench_multiqueue` (1 and 4
streams, both queue families, depth 2, transfer-queue upload, host-pointer import), `bench_barrier`,
`dmabuf_import_test` (grey and YUYV), the pose tools, `make_corpus` and the synthetic sweep (658/767
on the desktop and 659/767 on the Orange Pi, 687/768 with `--refine-edges`, as before). The library had no reports. Three test tools leaked and are fixed: the pose
end-to-end tool and the synthetic sweep did not release libapriltag's detections, and
`dmabuf_import_test` did not release its tag family. LeakSanitizer runs with `leak:_dl_init`
suppressed, which covers the Mali driver's allocations when it is loaded.

## 8t. Profiling round 4: every dispatch timed, one merge pass, fewer and wider passes

**Method.** `APRILTAG_VK_DISPATCH_TIMES=1` (new) writes a GPU timestamp after every dispatch and prints
each one's median when the detector is destroyed; bound builds then removed part of a stage's work to
see what the rest costs. Two configurations as in 8s: `ultrafast` with margin gate 4 at decimation 1
on a 1280x720 frame, and `exact` at decimation 2 on the 1280x800 corpus image. Timings are interleaved
`apriltag_vulkan_validate` runs against v1.8.0 (six rounds of 500 frames, medians), governors on
`performance`, LumenVision stopped.

**Per dispatch on the Orange Pi, decimation 1** (median us; each timestamp adds about 4 us):

| stage | v1.8.0 | round 4 |
| --- | --- | --- |
| clears | 161 | 161 |
| threshold (decimate, block min/max, filter, threshold) | 85 + 59 + 157 | 52 + 111 |
| labelling | `uf_init` 225, `uf_compress` 291, `uf_merge` 576, `uf_compress_count` 407, verification `uf_merge` 383 | `uf_init_scan` 233, `uf_merge` (4 pixels per thread) 432, `uf_compress_count` 420 |
| `blob_diff` | 757 | 710 |
| grouping and extents | argument builds 17 + 21, `hash_group` 123, `init_extents` 26, reduction 344, `merge_extents` 26 | argument builds 17 + 21, `hash_group` 144 (initialises the extents), reduction 339 |
| selection and offsets | `select_blobs` 17, count copy and scan chain 76, sort arguments 17 | `select_blobs` 43 (folds the copies), one-workgroup scan 28 |
| scatter, sort | 140, 632 | 141, 631 |
| sum | 4,539 | 3,481 |

At decimation 2 the sum fell from 1,988 to 1,560 us.

**Changes** (each identical in output: the 15 canonical quad dumps on both machines, 95 configurations
against v1.8.0 per machine including two images whose decimated width is not a multiple of 4, 42 record
paths and 44 fallback combinations, the synthetic CSVs at contrast 1.0 and 0.25 with and without
`--refine-edges`):

| change | Orange Pi `Detect()`, dec. 1 / dec. 2 |
| --- | --- |
| one `uf_merge` pass: each union retries its `atomicMin` until it hooks a root, so one pass joins every component; the verification merge never found a change in 105 configurations of v1.8.0 (`APRILTAG_VK_UF_VERIFY=1` keeps it) | 4.71 -> 4.32 / 2.20 -> 2.03 ms |
| `uf_init_scan`: run starts within a workgroup's row segment from a subgroup max-scan, so no compression precedes the merge | -> 4.12 / 1.99 |
| six fewer dispatches: extents initialised by `hash_group`, copies folded by `select_blobs`, one-workgroup scan that also writes the sort's arguments (a tiny dispatch with its barrier costs about 10 us here: 8 and 16 extra cost 78 and 160 us) | -> 4.09 / 1.93 |
| decimation folded into `block_minmax`, `block_filter` into a per-block `threshold` | -> 4.05 / 1.90 |
| `blob_diff`: 4-adjacent candidate neighbours share one root and one blob-size test | -> 4.01 / 1.89 |
| word-wide loads and stores in the 4-pixel threshold and labelling passes | -> 3.86 / 1.84 |
| `uf_merge` with 4 pixels per thread and word loads, on integrated GPUs | -> 3.73 / 1.83 |

Against v1.8.0 (`Detect()` and `pipeline_total` from the interleaved validate runs; throughput and latency from `bench_multiqueue`, three rounds of 800 frames on the Orange Pi and 3,000 on the desktop, interleaved, depth 1 unless stated):

| Orange Pi | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 4.71 -> 3.73 ms (-21%) | 2.19 -> 1.83 ms (-17%) | |
| `pipeline_total` | 5.32 -> 4.38 ms (-18%) | 3.04 -> 2.67 ms (-12%) | |
| 1 stream | 216 -> 271 fps (+26%) | 494 -> 538 fps (noisy) | 480 -> 536 fps (noisy) |
| 1 stream, depth 2 | 233 -> 302 fps (+29%) | 554 -> 691 fps (+25%) | 554 -> 691 fps (+25%) |
| 4 streams, one queue | 230 -> 299 fps (+30%) | 533 -> 672 fps (+26%) | 532 -> 674 fps (+27%) |
| latency, 1 stream (depth 1 / depth 2) | 9.25 -> 7.37 / 12.9 -> 9.93 ms | 4.04 -> 3.72 / 5.40 -> 4.33 ms | |

| RX 9060 XT | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 0.597 -> 0.534 ms (-11%) | 0.462 -> 0.432 ms (-6.5%) | 0.462 -> 0.432 ms (-6.5%) |
| `pipeline_total` | 0.894 -> 0.836 ms | 0.789 -> 0.762 ms | 0.664 -> 0.636 ms |
| 1 stream | 2,103 -> 2,239 fps (+6%) | 2,333 -> 2,240 fps (-4%, noisy) | 2,483 -> 2,525 fps (+2%) |
| 4 streams, one queue | 3,136 -> 3,426 fps (+9%) | 3,525 -> 3,521 fps | 4,927 -> 5,411 fps (+10%) |
| 4 streams, both compute families | 3,582 -> 4,102 fps (+15%) | 3,524 -> 3,463 fps (-2%) | 7,306 -> 7,325 fps |

**Validation layer.** Its synchronisation checks ignore compute shaders' buffer accesses unless
`syncval_shader_accesses_heuristic` is set (section 9): with it unset, a barrier removed on purpose
between `hash_group` and the reduction went unreported, so the "clean" results of earlier sections
covered transfers and indirect reads only. With it set, three reports remained, all for disjoint byte
ranges of an indirect-argument buffer (one, between the split sort's rounds, also in v1.8.0). The blob
counts' dispatch arguments now have their own buffer and the sort's runs and merge passes are compiled
with the large-blob list read-only; both machines report nothing across 84 configurations each (three
images, decimations 1 and 2, seven paths and fallbacks, plain and `snake` frames), with no extra
barriers. AddressSanitizer with UBSan and LeakSanitizer, and ThreadSanitizer, report nothing on the
Orange Pi (62 runs each, including `--pipelined --pipeline-depth 2`, `--max-uf 1`, `bench_multiqueue`
and `dmabuf_import_test`).

**Measured and not shipped** (Orange Pi, decimation 1, per dispatch):

- `blob_diff` with 4 pixels per thread and word loads: 755 -> 1,072 us (61 registers, half
  occupancy), and 1,168 us with one atomic per pixel at 32 registers. Its time is in the candidate
  pixels (removing everything after the candidate test leaves 219 of 755 us), not in the code loads.
- `uf_compress_count` counting each horizontal run once with its length capped at
  `min_cluster_pixels`: 409 -> 615 us (434 us even without the length loop); with 4 pixels per
  thread reusing a run's root: 422 -> 820 us (and 47 -> 57 us on the RX 9060 XT).
- `uf_merge` with 4 pixels per thread on the RX 9060 XT: 92 -> 124 us, so it stays on integrated GPUs.
- Zeroing `blob_size` at run starts in `uf_init` in place of the fill: the fill's 133 us left the
  dispatch timings but `Detect()` moved only 0.01 ms; the timestamp after each fill overstates it.
- 4 or 16 extents copies instead of 8; sort chunks of 8, 32 and 64 and the sorting network; sort
  workgroups of 64 and 128 with split lengths of 256 to 1,024. None better.
- Bounds for what is left: the sorting itself is 320 of the sort's 424 us (emitting the points in
  input order instead); the unions are 350 of `uf_merge`'s 578 us; counting is about 120 of
  `uf_compress_count`'s 409 us.

## 8u. Profiling round 5: RenderDoc on the desktop, and five more changes

**Instrumentation.** Everything from section 8t again, plus RenderDoc 1.45 on the RX 9060 XT (a
capture of one frame through the in-application API, then a replay script that times every command
and reads AMD's GPUPerfAPI counters per dispatch; see section 9), `perf`, `heaptrack` and the Chrome
trace for the CPU tail on the Orange Pi, and the Mali hardware counters against v1.8.0.

**What RenderDoc showed on the desktop.**

- Every frame began with a 38 us GPU copy of the 0.9 MB grey frame from the staging buffer, a
  transfer the in-app spans had attributed to "clear". The card has a 256 MiB device-local window the
  host can map (no resizable BAR needed), so the frame is now written there directly.
- `uf_merge` was the largest stage (83 us at decimation 1, 49 to 101 us at decimation 2 depending on
  the run) with a 93% L2 hit rate and no memory stalls: latency-bound on union-find chains, not
  bandwidth. A tall component's run roots hook one row onto the next, so later finds walk one hop per
  row. Merging inside bands of 8 rows first, then across the band boundaries, keeps chains short.
- `hash_group` and `scatter` spend 57% to 66% of their cycles in memory stalls (random gathers); the
  4-points-per-thread versions tried below did not help them.

**Mali counters** (per frame, v1.8.0 against round 4): load/store issue cycles -31%, partial reads
-33%, instruction issue starvation -19% to -22%, L2 read lookups -14%. Every shader is still
load/store-bound.

**CPU tail.** `perf`: the quad fit is 54% of CPU samples with `ultrafast` at decimation 1 and `exact`
refinement 72% at decimation 2, as in round 3. The trace shows `quad.decode_all` at about 500 us for
1.8 ms of fit work: the slowest blob (up to 540 us for a 2,904-point blob that falls back to the
combinatorial search) is not the limit, and 4 threads on the A76 cores are as fast as 8 on all cores,
so the A55 cores add nothing. Thread counts of 4 to 8 and core pinning measured no better. The Orange
Pi stays GPU-bound for throughput.

**Changes** (each identical in output: canonical dumps, 95 configurations against v1.8.0, 42 record
paths and 80 fallback combinations per machine, synthetic CSVs, validation layer with shader-access
synchronisation analysis):

| change | where | effect |
| --- | --- | --- |
| frames written into mappable device memory on discrete GPUs (`APRILTAG_VK_GRAY_MAPPED=0` keeps the staging copy) | RX 9060 XT | 4 streams +13% to +32%, 1 stream +2% to +4%; 4 streams with `exact` at decimation 2 (CPU-bound) -2% |
| `uf_merge` in two phases, inside 8-row bands then across them, on discrete GPUs | RX 9060 XT | `uf_merge` 95 -> 59 us (decimation 1), 86 -> 32 us (decimation 2); `Detect()` -11% and about -29%; 4 streams +10% |
| extents reduction with 4 boundary points per thread, combining a run of points in the same blob before its atomics (atomic variants) | Orange Pi | reduction 341 -> 223 us; `Detect()` 3.73 -> 3.63 ms, 1.83 -> 1.78 ms (RX 9060 XT without subgroups: 98 -> 20 us) |
| blobs of up to 256 points sorted by ranking every key, four keys per shared load | Orange Pi | sort 425 -> 386 us; `Detect()` 3.63 -> 3.58 ms at decimation 1, unchanged at decimation 2 |
| blob sizes counted per run piece and roots found in two hops, in place of `uf_compress_count`, on integrated GPUs | Orange Pi | `uf_compress_count` 420 us replaced by `uf_count_runs` 303 us, the `blob_size` fill gone, `uf_init_scan` +63 us, `blob_diff` +50 us; `Detect()` 3.59 -> 3.44 ms, 1.79 -> 1.78 ms |

Against v1.8.0 and round 4 (section 8t), interleaved:

| Orange Pi | v1.8.0 | round 4 | round 5 |
| --- | --- | --- | --- |
| `Detect()`, dec. 1 / dec. 2 `exact` | 4.72 / 2.19 ms | 3.73 / 1.84 ms | 3.44 / 1.78 ms |
| `pipeline_total`, dec. 1 / dec. 2 | 5.33 / 3.05 ms | 4.38 / 2.68 ms | 4.09 / 2.62 ms |
| dec. 1, 1 stream / depth 2 / 4 streams | 216 / 233 / 230 fps | 271 / 302 / 298 fps | 299 / 330 / 324 fps |
| dec. 2 `exact`, depth 2 / 4 streams | 554 / 533 fps | 689 / 665 fps | 714 / 700 fps |
| dec. 2 `ultrafast`, depth 2 / 4 streams | 554 / 538 fps | 689 / 672 fps | 716 / 695 fps |
| latency, dec. 1, 1 stream (depth 1 / depth 2) | 9.24 / 12.9 ms | 7.38 / 9.93 ms | 6.66 / 9.09 ms |

| RX 9060 XT, against v1.8.0 | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 0.598 -> 0.482 ms (-19%) | 0.462 -> 0.358 ms (-23%) | 0.460 -> 0.362 ms (-21%) |
| `pipeline_total` | 0.897 -> 0.777 ms | 0.790 -> 0.689 ms | 0.663 -> 0.564 ms |
| 1 stream | 2,046 -> 2,403 fps (+17%) | 2,257 -> 2,371 fps (+5%) | 2,485 -> 2,600 fps (+5%) |
| 4 streams, one queue | 3,118 -> 3,966 fps (+27%) | 3,486 -> 3,431 fps (-2%) | 4,965 -> 8,375 fps (+69%) |
| 4 streams, both compute families | 3,526 -> 5,095 fps (+44%) | 3,445 -> 3,416 fps (-1%) | 7,251 -> 8,238 fps (+14%) |

With `exact` refinement at decimation 2 the desktop is limited by the CPU tail, so the GPU savings do
not show in its throughput. AddressSanitizer with UBSan and LeakSanitizer, and ThreadSanitizer, report
nothing on the Orange Pi (80 runs each, including the new paths forced on and off), and the validation
layer reports nothing on either machine.

**Measured and not shipped:**

- 4 points per thread in `hash_group` (143 -> 170 us) and `scatter` (141 -> 228 us) on the Mali: only
  the atomic-bound reduction gains.
- Banded merging on the Mali: the in-band pass alone costs 370 us against 433 for one pass; the merge
  there is limited by per-pixel work, not chain length.
- `uf_init_scan` over whole rows (one workgroup per row looping over chunks), which would remove the
  segment-boundary pointers: 233 -> 374 us on the Mali.
- Run counting on the RX 9060 XT: 15 to 20 us slower per frame.
- The sort split at decimation 2 on the desktop (lengths 256 to 1,024): no gain over the run-to-run
  spread, which is ±40% for that dispatch there.
- Rank-all thresholds of 64, 512 and 1,024 (all worse than 256).
- The 4-points-per-thread reduction in place of the subgroup one on the desktop: 25 -> 20 us at
  decimation 1 but 11.5 -> 16 us at decimation 2.

## 8v. Rounds 4 and 5 against v1.7.0

The branch head (sections 8t and 8u) against v1.7.0 (`2626760`), measured as in section 8s: six
interleaved rounds of 1,000 frames of `apriltag_vulkan_validate` (medians of the round medians) and
three rounds of `bench_multiqueue` (800 frames per round on the Orange Pi, 3,000 on the desktop), the
same build type on both sides (`Release` on the Orange Pi, `RelWithDebInfo` on the RX 9060 XT),
governors on `performance` and LumenVision stopped. v1.7.0 has no second detector, so its depth-2
column is compared with its one-stream rate. Output is byte-identical: the 15 canonical dumps, 95
configurations per machine (five images and two of awkward width, decimations 1, 2 and 4, plain and
four stress frames) and 36 record paths (unfused, not pre-recorded, `--max-uf 1`, pipelined at depths
1 and 2, `--import`) against v1.7.0's serial path; no `bench_multiqueue` mismatches.

Orange Pi 5:

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 5.248 -> 3.227 ms (-39%) | 2.108 -> 1.602 ms (-24%) | 2.117 -> 1.593 ms (-25%) |
| `quad_decode` | 0.699 -> 0.480 ms (-31%) | 0.286 -> 0.180 ms (-37%) | 0.284 -> 0.176 ms (-38%) |
| `tag_decode` | 0.132 -> 0.141 ms | 0.720 -> 0.629 ms (-13%) | 0.169 -> 0.153 ms |
| `pipeline_total` | 6.088 -> 3.852 ms (-37%) | 3.120 -> 2.423 ms (-22%) | 2.579 -> 1.931 ms (-25%) |
| 1 stream | 187 -> 299 fps (+59%) | 434 -> 604 fps (+39%) | 435 -> 599 fps (+38%) |
| 1 stream, depth 2 | 329 fps (+75%) | 715 fps (+65%) | 715 fps (+64%) |
| 4 streams, one queue | 197 -> 325 fps (+65%) | 505 -> 688 fps (+36%) | 508 -> 699 fps (+38%) |
| latency, 1 stream (depth 1 / depth 2) | 10.65 -> 6.69 / 9.11 ms | 4.61 -> 3.31 / 4.19 ms | 4.59 -> 3.33 / 4.19 ms |

RX 9060 XT:

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 0.682 -> 0.488 ms (-28%) | 0.503 -> 0.369 ms (-27%) | 0.501 -> 0.370 ms (-26%) |
| `quad_decode` | 0.254 -> 0.217 ms | 0.119 -> 0.111 ms | 0.118 -> 0.109 ms |
| `tag_decode` | 0.079 -> 0.082 ms | 0.273 -> 0.220 ms (-19%) | 0.096 -> 0.092 ms |
| `pipeline_total` | 1.016 -> 0.789 ms (-22%) | 0.893 -> 0.713 ms (-20%) | 0.715 -> 0.577 ms (-19%) |
| 1 stream | 1,973 -> 2,363 fps (+20%) | 2,052 -> 2,351 fps (+15%) | 2,428 -> 2,538 fps (+5%) |
| 1 stream, depth 2 | 2,836 fps (+44%) | 2,390 fps (+16%) | 4,005 fps (+65%) |
| 4 streams, one queue | 2,885 -> 3,948 fps (+37%) | 2,849 -> 3,367 fps (+18%) | 4,648 -> 8,138 fps (+75%) |
| 4 streams, both compute families | 3,673 -> 5,018 fps (+37%) | 2,823 -> 3,393 fps (+20%) | 7,108 -> 8,261 fps (+16%) |

The loss section 8s recorded (4 streams over both compute families at decimation 1 with the sort
split) is gone: +37% with the split and +43% without it (`APRILTAG_VK_SORT_SPLIT=8192`, 5,242 fps).

## 8w. Profiling round 6: what is left, and one change

**Instrumentation.** Per-dispatch GPU times (`APRILTAG_VK_DISPATCH_TIMES`), the Mali hardware counters
and `malioc` on the Orange Pi; RenderDoc with AMD's counters on the RX 9060 XT; the Chrome trace of the
CPU tail; the validation layer with shader-access synchronisation analysis; ASan, UBSan and LSan on
both machines (MSVC on the desktop, GCC 14 on the Orange Pi) and TSan on the Orange Pi.

**Where the Mali time goes** (decimation 1, medians in us): `blob_diff` 760, `uf_merge` 435, the sort
383 + 142 + 23 + 38, `uf_count_runs` 307, `uf_init_scan` 296, the extents reduction 227, `hash_group`
142, `scatter` 142, the threshold 119 + 50, `select_blobs` 44, the scan 28. `malioc` puts every
shader at full occupancy except `select_blobs` (64 registers, half occupancy, 44 us). Other findings:

- GPU timestamps cost about 0.2 ms per frame on the Mali: `Detect()` without them is 3.23 ms at
  decimation 1 and 1.59 ms at decimation 2. The spans and dispatch times read high by that much there.
- Small buffer fills are free (ten extra fills measured no difference), `vkQueueSubmit` takes 19 us,
  and spinning on the fence instead of waiting gained nothing.
- The CPU tail (`quad_decode`, about 500 us at decimation 1) is about 59 us of serial span grouping
  before the workers start, 280 us in parallel and a 160 us serial tail, of which `update_fit` is about
  100 us. Waking a worker takes about 120 us on an A55 core and 60 us on an A76.

**On the RX 9060 XT** (RenderDoc): with frames written into mappable device memory (section 8u),
`block_minmax` reads them over PCIe in 16 us at 63% memory stall, against 3 us from a device-local
copy; the 38 us copy saved still outweighs it. The second merge phase takes 23 us at decimation 2;
`hash_group` and `scatter` stay about 60% memory-stalled. The card runs in two clock states about 1.45x
apart, so single desktop runs are noisy.

**Change** (identical output: canonical dumps, 95 configurations, 42 record paths and 84 fallback
combinations on the desktop; integrated GPUs are unaffected):

| change | where | effect |
| --- | --- | --- |
| point sort in workgroups of up to 1,024 threads on discrete GPUs (256 kept on integrated GPUs, where larger was slower) | RX 9060 XT | sort at decimation 2 (one large blob, no split) 47 to 68 -> about 25 us, at decimation 1 about 102 -> 88 us; `Detect()` about -3.6% at decimation 1 |

**Measured and not shipped:**

- `blob_diff` in two passes, the candidate pixels listed with one counter increment per subgroup and
  then processed densely: Mali 759 -> 833 us (368 + 446 us); slower on the desktop as well.
- A shortcut in `blob_diff`'s size test through the piece lengths: 759 -> 838 us on the Mali.
- Merge bands nested over several levels on the desktop: no net gain over bands of 8.
- The final line fits in parallel on the worker pool (MSVC): `quad_decode` -7% at decimation 1 but
  +7% at decimation 2.
- Sort splits of 128 to 512 points on the desktop: slower.

**Sanitisers and validation.** Clean: MSVC ASan on the desktop (86 runs), GCC ASan, UBSan (with
`float-cast-overflow`) and LSan on the Orange Pi (92 runs) and TSan there (23 runs, 8 CPU threads and
pipelined runs), covering the stress frames, `--max-uf 1`, every record path and the environment
switches for the fallbacks.

The remaining Mali stages are each load/store-bound at full occupancy, and the changes tried on them
this round were slower; what remains on the CPU side is the serial part of the quad fit.

## 8x. Version 1.9.0 against v1.8.0

Version 1.9.0 (sections 8t to 8w) against v1.8.0 (`5400c69`). Both sides built the same way
(`Release` on the Orange Pi, `RelWithDebInfo` on the RX 9060 XT), run alternately: six rounds of 1,000
frames of `apriltag_vulkan_validate` (medians of the round medians) and three rounds of
`bench_multiqueue` (800 frames per round on the Orange Pi, 3,000 on the desktop), governors on
`performance` and LumenVision stopped on the Orange Pi. No `bench_multiqueue` mismatches.

Checks before the release, on both machines: the 15 canonical dumps; quad dumps and detections
against v1.8.0 in 95 configurations (seven images, decimations 1, 2 and 4, plain and four stress
frames), 42 record paths and 84 fallback configurations, all identical; the four synthetic sweeps give
the same CSVs (659/767 and 687/768 on the Orange Pi, 658/767 and 687/768 on the desktop); no messages
from the validation layer with synchronisation validation and shader-access analysis (forced
fallbacks and every labelling and sort switch, with the best-practices positive control reporting);
AddressSanitizer clean on the desktop (84 runs); AddressSanitizer, UBSan and LeakSanitizer (88 runs)
and ThreadSanitizer (23 runs) clean on the Orange Pi.

Orange Pi 5 (decimation 1 on `hd720.pgm`, decimation 2 on the 1280x800 corpus image; depth 2 on
`grayimage.pgm`):

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 4.492 -> 3.230 ms (-28%) | 2.003 -> 1.582 ms (-21%) | 2.001 -> 1.591 ms (-20%) |
| `quad_decode` | 0.462 -> 0.482 ms | 0.188 -> 0.176 ms | 0.182 -> 0.173 ms |
| `tag_decode` | 0.127 -> 0.141 ms | 0.635 -> 0.632 ms | 0.170 -> 0.151 ms |
| `pipeline_total` | 5.085 -> 3.862 ms (-24%) | 2.837 -> 2.405 ms (-15%) | 2.369 -> 1.922 ms (-19%) |
| 1 stream | 213 -> 300 fps (+41%) | 442 -> 588 fps (+33%) | 445 -> 571 fps (+29%) |
| 1 stream, depth 2 | 233 -> 329 fps (+41%) | 556 -> 716 fps (+29%) | 555 -> 717 fps (+29%) |
| 4 streams, one queue | 230 -> 326 fps (+42%) | 534 -> 692 fps (+30%) | 533 -> 693 fps (+30%) |
| latency, 1 stream (depth 1 / depth 2) | 9.37 -> 6.66 / 12.85 -> 9.10 ms | 4.53 -> 3.38 / 5.39 -> 4.19 ms | 4.50 -> 3.51 / 5.40 -> 4.18 ms |

RX 9060 XT (decimation 1 on `hd720.pgm`, decimation 2 on `grayimage.pgm`):

| | dec. 1, `ultrafast`, gate 4 | dec. 2, `exact` | dec. 2, `ultrafast`, gate 4 |
| --- | --- | --- | --- |
| `Detect()` | 0.602 -> 0.474 ms (-21%) | 0.462 -> 0.349 ms (-24%) | 0.445 -> 0.325 ms (-27%) |
| `quad_decode` | 0.222 -> 0.220 ms | 0.109 -> 0.110 ms | 0.110 -> 0.111 ms |
| `tag_decode` | 0.079 -> 0.082 ms | 0.223 -> 0.219 ms | 0.095 -> 0.093 ms |
| `pipeline_total` | 0.905 -> 0.778 ms (-14%) | 0.792 -> 0.704 ms (-11%) | 0.648 -> 0.533 ms (-18%) |
| 1 stream | 2,086 -> 2,305 fps (+11%) | 2,349 -> 2,319 fps (-1%) | 2,474 -> 2,848 fps (+15%) |
| 4 streams, one queue | 3,141 -> 4,145 fps (+32%) | 3,525 -> 3,451 fps (-2%) | 4,944 -> 8,344 fps (+69%) |
| 4 streams, both compute families | 3,583 -> 5,144 fps (+44%) | 3,397 -> 3,384 fps (0%) | 7,411 -> 8,426 fps (+14%) |

With `exact` refinement at decimation 2 the desktop is limited by the CPU tail, and its throughput is
unchanged within the run-to-run spread.

## 8y. Against vulkan-apriltag, and the quad refit in the parallel loop

**The comparison.** vulkan-apriltag (github.com/carbotaniuman/vulkan-apriltag, GPL-3.0, one commit)
is another Vulkan port of the AprilTag front end: full resolution only, tag36h11 only, quads fitted
on the GPU and decoded with libapriltag's own `quad_decode`. Both were driven by the same harness
(same images, the same serial timing loop of upload, GPU, readback and CPU decode, no GPU timestamps,
`Release` builds, interleaved rounds), with libapriltag on the CPU as the reference. Its code was only
built and run, not used: the changes below come from the measurements.

- Accuracy: on the synthetic sweep (384 frames each at contrast 1, contrast 0.25 and noise sigma 8
  with blur 1.2) vulkan-apriltag decodes exactly what libapriltag does at `quad_decimate` 1, and so
  does vkapriltag at decimation 1 (two more tags at contrast 0.25), with corner errors within 0.001
  px of each other. On ten real images vulkan-apriltag matches libapriltag's 24 tags within 0.005
  px; vkapriltag matches them within 0.04 px and also decodes a 12 px tag libapriltag misses.
- Speed at the same resolution: on the Orange Pi vkapriltag is 1.4x to 2.2x faster (4.1 to 7.5 ms
  against 5.7 to 16.5 ms per frame; its GPU part grows to 15.9 ms on a frame with eight tags). On the
  RX 9060 XT the two tie on one-tag frames (0.85 ms) and vulkan-apriltag is 8% to 21% faster on
  frames with five to eight tags: its CPU part is 0.15 to 0.23 ms against our 0.34 to 0.75 ms.

**Why our CPU part was longer.** vkapriltag hands about twice as many quads to `TagDecoder` as
libapriltag produces (131 against 54 on `field2.png` at decimation 1, 52 against 25 on `hd720.pgm`).
Counting where each blob is rejected, in our quad fit and in a counting copy of libapriltag's:

| field2.png, decimation 1 | libapriltag | vkapriltag |
| --- | --- | --- |
| blobs fitted (after the size and border-polarity tests) | 709 | 837 |
| rejected by area | 284 | 430 |
| rejected by the angle and convexity test | 299 | 253 |
| quads | 54 | 131 |

The DP corner seeding (item 25) takes the farthest points and the points of largest deviation as
corners, so a ball or other round blob becomes a tidy convex quad that passes the angle test, where
libapriltag's error peaks usually do not. Removing those quads changes the quad set, so it was not
done. Fitting only the blobs that end up as quads (simulated by remembering them from the first of
repeated frames) saves 40 to 110 us of `quad_decode` on the desktop: a GPU pre-filter would be worth
little. The time was elsewhere: about 120 us of serial refit after the parallel fits at decimation 1.

**Change: the refit runs inside the parallel loop.** Each blob's lines, corners, area and angle
tests now run in the task that fitted it; survivors are still collected in blob order. Measured in
section 8r at -25% on the desktop, it was held back because GCC on AArch64 fuses `a * b + c` into
multiply-adds by default and moving the code changed which ones, and so the last bit of some corners.
`QuadDecode.cpp` is now compiled with `-ffp-contract=off` on GCC and Clang, so its arithmetic no
longer depends on the code's layout.

Output: identical on the RX 9060 XT (15 canonical dumps, 95 configurations, 42 record paths and 84
fallbacks against v1.9.0; synthetic CSVs). On the Orange Pi 234 of 331 dump files change, all by at
most 1.1e-13 px, with the same quads, ids and Hamming distances everywhere; record paths and fallbacks
match the new build's own default output, the synthetic CSVs are unchanged, and ASan, UBSan, LSan
and TSan are clean (88 and 23 runs; MSVC ASan on the desktop, 84 runs).

Interleaved against v1.9.0, six rounds (1,000 frames on the desktop, 600 on the Orange Pi), `exact`
refinement:

| | RX 9060 XT `quad_decode` | RX 9060 XT `pipeline_total` | Orange Pi `quad_decode` | Orange Pi `pipeline_total` |
| --- | --- | --- | --- | --- |
| field2.png (1200x800), dec. 1 | 0.445 -> 0.334 ms (-25%) | 1.444 -> 1.312 ms (-9%) | 0.913 -> 0.742 ms (-19%) | 6.878 -> 6.704 ms (-2.5%) |
| field2.png, dec. 2 | 0.137 -> 0.118 ms (-14%) | 0.677 -> 0.651 ms (-4%) | 0.186 -> 0.152 ms (-18%) | 2.429 -> 2.391 ms (-1.6%) |
| hd720.pgm, dec. 1 | 0.214 -> 0.163 ms (-24%) | 0.869 -> 0.819 ms (-6%) | 0.482 -> 0.403 ms (-16%) | 4.097 -> 4.025 ms (-1.8%) |
| hd720.pgm, dec. 2 | 0.169 -> 0.153 ms (-10%) | 0.761 -> 0.721 ms (-5%) | 0.223 -> 0.200 ms (-10%) | 2.285 -> 2.250 ms (-1.5%) |
| grayimage.pgm / corpus 1280x800, dec. 1 | 0.231 -> 0.176 ms (-24%) | 0.910 -> 0.851 ms (-6%) | 0.501 -> 0.416 ms (-17%) | 4.380 -> 4.288 ms (-2.1%) |
| grayimage.pgm / corpus 1280x800, dec. 2 | 0.110 -> 0.094 ms (-14%) | 0.725 -> 0.644 ms (-11%) | 0.176 -> 0.153 ms (-13%) | 2.398 -> 2.383 ms (-0.6%) |

The Orange Pi figures include whatever turning off multiply-add fusion in `QuadDecode.cpp` costs;
it was not measured on its own.

## 8z. Four more items: tuned builds, allocation-free decode, a peak check, the quad fit on the GPU

Four of the suggestions left after section 8y, each behind a switch so one binary covers every
combination. Benchmarked alone and cumulatively (2, 2+3, 2+3+5, 2+3+5+6) in three refinement states:
S1 `exact` without the margin gate, S2 `ultrafast` with `APRILTAG_VK_PROBE_MARGIN=4` (the gate has
no effect under `ultrafast`, which already decodes every quad unrefined first), S3 `fast` with a gate
of 4. Images `field2.png` (1200x800, five tags), `hd720.pgm` and the 1280x800 corpus image at
decimations 1 and 2; all variants interleaved in rotating order, six rounds of 1,000 frames on the
desktop and 600 on the Orange Pi.

- **Item 2, `VKAPRILTAG_CPU_TUNE`** (CMake, empty by default): `-mcpu=<cpu>` on GCC and Clang
  (`cortex-a76.cortex-a55` for the RK3588), `/arch:<arch>` with `/fp:contract` on MSVC (`AVX2`),
  for the library and libapriltag; `QuadDecode.cpp` keeps fusion off. The binaries then need that
  CPU. Output identical to the generic build on both machines (all 331 dump files).
- **Item 3, allocation-free decode** (a fourth libapriltag patch; benchmarked behind a switch, now
  unconditional): `quad_update_homographies()` keeps only
  `matd_inverse()`'s singularity test instead of building the unused inverse, `quad_decode_index()`
  and `quad_sample_code()` share the quad instead of copying it and both homographies per family,
  and `quad_decode()`'s two scratch buffers live on the stack. Identical output.
- **Item 5, `dp_peaks_check`** (`APRILTAG_VK_DP_PEAKS_CHECK=1`, off): a DP-seeded quad is kept only
  if the peak search would also accept its blob, as libapriltag's fit does. It halves the quads
  (160 -> 79 on `hd720.pgm` at decimation 1, 133 -> 67 on `field2.png`) but loses 1 of 369 decoded
  tags on the contrast-0.25 synthetic set in S1 and S3 (a tag under 20 px); S2 and the real images
  are unchanged. On the three synthetic sets and ten real images it removed 64% to 69% of the quads
  that never decode and lost 2 of 2,038 decoded tags, both among the smallest (93 and 117 px^2).
- **Item 6, `APRILTAG_VK_GPU_QUADFIT=1`** (off): `fit_quad_dp.comp` runs QuadDecode's DP path on the
  GPU, a seeding pass with one workgroup per blob over a device-local copy of the sorted points and a
  fit pass with four invocations per blob. Float divisions and square roots are done in double and
  refined to be correctly rounded, nothing is contracted, and any test within 1e-4 of its limit goes
  to the CPU, so the output is identical to the CPU's (all 331 dump files). Needs `shaderFloat64`,
  `shaderInt64` and `VK_KHR_shader_subgroup_extended_types`, so not on the Mali-G610.

Change in `pipeline_total` against the base build, the range over the three images (negative is
faster):

| RX 9060 XT | 2 | 3 | 5 | 6 | 2+3 | 2+3+5 | 2+3+5+6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1, dec. 1 | -2.3 to -1.8% | -0.4 to +0.4% | -0.4 to +5.7% | +4.2 to +4.8% | -3.1 to -2.6% | -3.0 to +4.8% | +2.7 to +9.3% |
| S1, dec. 2 | -2.0 to +7.6% | -0.4 to +4.4% | -10.5 to -3.6% | +4.6 to +8.2% | -3.1 to -1.3% | -12.3 to -3.7% | -6.2 to +2.2% |
| S2, dec. 1 | -3.5 to -0.5% | +0.3 to +1.3% | +9.6 to +14.5% | +5.0 to +7.0% | -3.3 to -2.2% | +5.2 to +11.3% | +11.4 to +16.2% |
| S2, dec. 2 | -5.3 to -1.4% | -3.4 to +0.2% | -2.7 to +6.9% | +4.8 to +7.4% | -5.5 to -2.8% | -5.0 to +3.3% | +3.7 to +11.8% |
| S3, dec. 1 | -4.2 to -1.9% | -0.1 to +1.3% | +1.4 to +10.7% | +5.4 to +6.3% | -4.8 to -2.6% | -3.9 to +8.9% | +3.6 to +15.3% |
| S3, dec. 2 | -5.8 to -1.6% | -0.3 to +5.2% | -8.5 to +5.2% | +4.0 to +7.6% | -7.3 to -3.1% | -13.1 to +1.4% | -6.1 to +8.9% |

| Orange Pi 5 | 2 | 3 | 5 | 6 (not available) | 2+3 | 2+3+5 |
| --- | --- | --- | --- | --- | --- | --- |
| S1, dec. 1 | -0.3 to +0.5% | -0.3 to +1.2% | -1.0 to -0.3% | | -0.5 to -0.4% | -1.3 to -0.6% |
| S1, dec. 2 | -0.3 to 0.0% | -0.2 to 0.0% | -6.3 to -2.8% | | -0.7 to -0.2% | -6.5 to -2.9% |
| S2, dec. 1 | -0.4 to +0.1% | -0.3 to -0.1% | +2.6 to +3.9% | | -0.6 to +0.5% | +2.2 to +3.5% |
| S2, dec. 2 | -0.7 to -0.2% | -0.3 to +0.2% | +3.4 to +6.7% | | -0.7 to +1.3% | +3.9 to +5.1% |
| S3, dec. 1 | -0.3 to -0.2% | -0.5 to -0.1% | -0.3 to +2.1% | | -0.7 to +0.3% | -0.4 to +1.9% |
| S3, dec. 2 | -0.3 to +1.2% | -1.1 to -0.5% | -4.0 to +4.5% | | -1.1 to +0.6% | -4.1 to +3.5% |

The desktop's two clock states make single cells noisy (item 2's +7.6% and item 3's +4.4% and +5.2%
are each one image); the consistent results are item 2 at -1.4% to -5.8% on the desktop
(`tag_decode` -5% to -18%), item 3 at `tag_decode` -1% to -6% on the Orange Pi, and items 2+3 at -1.3%
to -7.3% on the desktop in every cell.

Item 5 saves refinement but spends a peak search on every DP quad that passes the checks: it pays only
with `exact` refinement at decimation 2 (-3% to -12%) and loses under `ultrafast`, which already
skips refining quads that do not decode. Item 6 adds 0.05 to 0.18 ms of GPU work against 0.02 to
0.13 ms of CPU, so serial frames are 4% to 8% slower on the desktop; where the desktop is limited by
the CPU tail it raises throughput instead (`bench_multiqueue`, `exact`, decimation 2: 4 streams 3,428
-> 3,709 fps, +8%; one stream at depth 2 2,598 -> 2,844 fps, +9%), and where the GPU is the limit it
costs (decimation 1, 4 streams: 2,651 -> 2,300 fps, -13%).

Defaults: item 3 always on (its switch's lazily read flag was itself a data race under TSan, so it was
removed once the output proved identical); items 2, 5 and 6 opt-in. Checks: the validation layer with synchronisation and
shader-access analysis reports nothing with every switch on (both machines); MSVC ASan clean with
every switch on (84 runs); on the Orange Pi ASan, UBSan, LSan and TSan clean with and without the peak
check.

## 8aa. The edge-case fixes and what they cost

Seventeen issues from an audit of the code (wrapped int32 line-fit sums, unchecked sizes and
counts, races, leaks on error paths, a busy-waiting capture loop), fixed one commit each. Every fix
was timed against its parent, interleaved, in two states (`exact` without the margin gate, `fast`
with a gate of 4) on `field2.png`, `hd720.pgm` and the 1280x800 corpus image at decimations 1 and
2: six rounds of 1,000 frames on the desktop, four of 400 on the Orange Pi with the performance
governor and LumenVision stopped. Below, the median over those 12 configurations of the change in
`quad_decode` and `pipeline_total`; desktop configurations scatter by about +-5% between processes,
the Orange Pi's by about +-1%.

| Fix | Desktop quad_decode | Desktop total | Pi quad_decode | Pi total |
| --- | --- | --- | --- | --- |
| 1. int64 line-fit moments (first form) | +12.0% | +3.7% | +1.7% | +0.1% |
| 4. WorkerPool: serialised, exceptions, size | +0.7% | +0.7% | +0.3% | -0.1% |
| 5. Size checks | +1.7% | +0.4% | -0.1% | -0.2% |
| 6. Fused tail bounded by capacity | -0.3% | +0.6% | -0.1% | +0.1% |
| 7. FramePipeline size checks | +0.2% | +0.5% | +0.0% | -0.1% |
| 8. Fence reset at submit | -2.3% | -0.7% | -0.1% | +0.2% |
| 9. Workgroup-count limits, exception safety | -1.5% | -0.9% | -0.2% | +0.1% |
| 11. Dispatch-time probe ownership | -0.1% | -0.2% | +0.4% | +0.2% |
| 12. Sort shared-memory budget | -0.1% | -0.1% | -0.0% | -0.0% |
| 13. Device selection | +0.5% | +1.3% | -0.2% | -0.1% |
| 14. Constructor cleanup, whole-size ranges | +1.3% | +0.9% | -0.6% | -0.0% |
| 15. Pipeline-cache lock, lane waits | +1.0% | +0.9% | -0.2% | -0.1% |
| 16-17. Tools and capture | +0.7% | +0.8% | +0.0% | +0.0% |
| All, final (with the follow-up to 1) | -0.6% | -0.1% | +0.2% | -0.0% |

Fixes 2, 3 and 10 (pose and PGM input) are outside the frame loop. Only fix 1 cost anything, and only
on the desktop. Two things made its first form slow under MSVC: the RelWithDebInfo build (`/Ob1`)
inlines only functions marked `inline`, and the whole blob switched to 128-bit covariances once its
totals crossed the int64 bound, which large background blobs at decimation 1 do; running in parallel,
one such blob sets the frame's tail (`APRILTAG_CPU_THREADS=1` showed no cost at all). The follow-up
chooses 128 bits per line fit, keeps the peak search's windows (at most 41 points, which cannot
overflow) in int64, and lets the DP path sum its four segments directly rather than build the prefix
sums; `quad_decode` is then back within noise on both machines, and the GCC build of the CPU fit is
faster than before in a stand-alone benchmark (-12% DP, -37% peak search on synthetic blobs).

With the opt-in GPU quad fit (`APRILTAG_VK_GPU_QUADFIT=1`) the desktop's GPU time is 3.0% higher
(`pipeline_total` +3.1%): `fit_quad_dp.comp` now sums six int64 moments per segment where it summed
three int64 and four int32, which the fix needs for the same reason as the CPU path.

Output is unchanged: the dump matrix (331 files, including two odd-sized crops) is identical to the
base on the desktop with and without the GPU fit, and on the Orange Pi (291 files).

## 8ab. Blobs of any size sorted

Blobs above the sort's capacity (2048 points at decimation 2, 4096 at 1) used to pass through
unsorted, so a tag of about 1,000 px at decimation 2 (1,300 px at 1) was never detected: a
synthetic 3000x3000 frame with one tag36h11 lost every tag from 1,000 px to 2,600 px, which the
reference detector found. Blobs above 1024 points are now sorted in runs of 1024 and placed by one
ranking pass, up to 65,536 points (see DESIGN.md). All of those tags are found, at decimations 1
and 2, with corners within 0.0003 px of libapriltag's; the dump matrix is unchanged (331 files on
the desktop with and without the GPU fit, 291 on the Orange Pi), since the order of every blob the
old sort handled is the same.

Against the previous head, interleaved, the three images at decimations 1 and 2 in two states:

| | Desktop GPU | Desktop total | Pi GPU | Pi total |
| --- | --- | --- | --- | --- |
| Decimation 1 | -4.7% to -7.1% | -2.3% to -5.1% | -1% | -1.0% |
| Decimation 2, frames with a blob over 1024 points | +2.8% to +3.4% | +0.8% to +0.9% | +1.6% to +3.0% | +0.7% to +2.0% |
| Decimation 2, other frames | within noise | within noise | +0.2% to +0.3% | 0.0% |

Decimation 1 is faster because one ranking pass replaces two merge rounds. At decimation 2 the
cost is the sorting of large background blobs that used to be skipped; with a 2048-point limit and
run it was 0.2 ms on the Mali-G610 (+10% of a frame), hence 1024 at every decimation.

## 9. Measuring

```
apriltag_vulkan_validate --data <file-or-dir> [--iterations N] [--pipelined]
                         [--decimation N] [--family tag36h11] [--csv out.csv]
```

`--data` takes a single image or a directory (PGM only in builds without
OpenCV). It runs the full pipeline and diffs the decoded tag IDs *and* corner
positions against the unmodified reference detector. Use `--iterations` for
steady-state numbers: a single `Detect()` is dominated by first-touch page
faults across every buffer. `--pipelined` selects `FramePipeline`, so both
paths can be A/B'd from one binary.

Three cautions, all learned the hard way here:

- **Verify the two builds use the same `CMAKE_BUILD_TYPE`.** This one cost an
  afternoon: a baseline worktree configured `Release` against the project's
  own `x64-Release` preset, which is `RelWithDebInfo`, reported `quad_decode`
  **+82%** and looked exactly like a serious regression in the change under
  test. MSVC's `RelWithDebInfo` is `/O2 /Ob1`, and `/Ob1` inlines only
  functions declared `inline` or defined in-class - so the DP corner-seeding
  helpers in `QuadDecode.cpp`, called once or twice per point across four
  O(n) passes, stayed real calls. They are marked `inline` now, which is
  worth ~35% of `quad_decode` in that configuration and nothing in `/Ob2`,
  but the lesson generalises: a cross-tree A/B compares toolchain settings
  unless you check. The same trap has a worse Linux form - a build directory
  configured with no `CMAKE_BUILD_TYPE` at all gets **no `-O` flag**, and
  that is exactly what the Orange Pi's working build tree turned out to be,
  so an early round of Mali figures compared a `-O3` baseline against an
  unoptimised branch. `grep CMAKE_BUILD_TYPE build/CMakeCache.txt` on both
  sides before believing anything; an empty value is the dangerous one,
  because nothing warns about it.
- **Verify the binaries actually differ** before believing an A/B —
  `git checkout` carries uncommitted changes onto the new branch, which once
  produced a confident null result from comparing two identical builds.
  `md5sum` them.
- **The Orange Pi's run-to-run variance is ~+/-5%**, enough to invert a
  single-run comparison. Interleave in ABBA order rather than AB (it drifts
  thermally across a session), take many samples, and report min and p25 —
  the mean is dominated by slow outliers.

On the Orange Pi, `perf` works in user mode (`perf record -g`, `perf stat -e cycles:u,instructions:u`
with `perf_event_paranoid` 2), so build the profiling tree with `-g -fno-omit-frame-pointer` and pin the
run to the A76 cores with `taskset -c 4-7`. `valgrind --tool=callgrind` and `heaptrack` are installed,
and GCC 14 carries AddressSanitizer, UndefinedBehaviorSanitizer (add `float-cast-overflow`),
ThreadSanitizer and LeakSanitizer: configure with `CMAKE_BUILD_TYPE=None` and the flags in
`CMAKE_CXX_FLAGS`, `CMAKE_C_FLAGS` and both linker flag variables. Run LeakSanitizer with
`detect_leaks=0` unless the loader's own allocations are being filtered, since the Vulkan loader reports
tens of megabytes at exit.

On Mali, `hwcnt_run <command>` (`tools/mali_counters`, built with
`-DVKAPRILTAG_BUILD_MALI_COUNTERS=ON`) prints every hardware counter over a run, and `hwcnt_timeline
200 out.csv <command>` samples the main ones every 200 us; run the detector paced (`bench_multiqueue
--fps 20`) and align the frames to attribute counters to stages (section 8q). For static shader
analysis, `malioc -c Mali-G610 --vulkan --compute <shader>.spv` from Arm Performance Studio (Linux
x86-64 or Windows host).

For per-dispatch GPU timing, `APRILTAG_VK_TIMESTAMPS=1` adds a timestamp-span
breakdown plus the gaps between spans, separating intra-submit gaps from
submit boundaries, and `APRILTAG_VK_DISPATCH_TIMES=1` (with `APRILTAG_VK_PRERECORD=0`) times every
dispatch on its own. Each timestamp waits for the work before it, which adds about 4 us per dispatch
on the Mali and overstates transfer fills; confirm a saving with an interleaved `Detect()` comparison.

On the RX 9060 XT, RenderDoc adds what the in-app timestamps cannot: every command timed on its own
(fills and copies as well as dispatches) and AMD's GPUPerfAPI counters per dispatch (cache hit rates,
memory and write-unit stalls, wavefronts, bytes fetched and written). A compute-only frame has no
present to mark it, so the validate tool captures one through RenderDoc's in-application API when
built with `renderdoc_app.h` found by CMake:

```
ENABLE_VULKAN_RENDERDOC_CAPTURE=1 APRILTAG_VK_PRERECORD=0 apriltag_vulkan_validate --data img.pgm \
    --iterations 25 --renderdoc-capture 20 --renderdoc-file out/frame
RDOC_CAP=out/frame_capture.rdc RDOC_OUT=frame.tsv qrenderdoc --python tools/renderdoc/rdoc_counters.py
```

Each command is timed in isolation on replay, so transfers that overlap in a real frame read high.
RenderDoc has no counters for the Mali-G610; there, use the Mali tools above.

For the validation layer's synchronisation checks to see compute shaders' buffer accesses, set
`khronos_validation.validate_sync = true` and `khronos_validation.syncval_shader_accesses_heuristic
= true` in a `vk_layer_settings.txt` named by `VK_LAYER_SETTINGS_PATH` (with
`APRILTAG_VK_VALIDATION=1`). Without the second setting only transfers and indirect reads are checked:
a deliberately removed barrier between two dispatches goes unreported. For an independent view on AMD, RADV can emit RGP traces
with `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1` (the per-submit flag is
what makes it work for a headless compute app with no `vkQueuePresentKHR`).
RenderDoc needs an artificial frame boundary for the same reason, and Mali
exposes no external GPU profiler at all — there, attribute cost by external
wall-clock deltas while varying a knob or the input size.

### Tracing and per-quad profiling

`--profile` prints a per-span table (count, total, mean, p50, p99, max; spans nest, so totals
overlap) plus the `TagDecoder` refine breakdown, and `--trace out.json` writes a Chrome Trace Event
file for https://ui.perfetto.dev. Both work in either validate tool and with `--pipelined`, where
`pipeline.gpu_pass`, `pipeline.cpu_tail` and `pipeline.wait_gpu` show which side limits throughput.

- Each pool worker, the GPU-driver thread and a "GPU" track get their own rows. The GPU track is the
  existing `APRILTAG_VK_TIMESTAMPS` spans placed on the host timeline by pinning the first span to the
  first submission's host time, so expect a constant offset of tens of microseconds, not drift.
- Library users call `prof::Enable(true)` (or set `APRILTAG_VK_TRACE=<path>` and call
  `prof::EnableFromEnv()`) **before constructing `GpuDetector`**, which creates the timestamp pool at
  construction, then `prof::Summary()` / `prof::WriteTrace()` while no frame is in flight.
- Cost: a disabled scope is one relaxed atomic load. A before/after comparison on the RX 9060 XT showed
  no difference outside run-to-run noise (median 0.99-1.10 ms vs 0.96-1.04 ms). Enabled costs about
  +0.09 ms per frame (timestamp queries plus roughly 420 events). `VKAPRILTAG_ENABLE_PROFILING=OFF`
  removes the hooks at compile time; that configuration was not built here.
- Keep traces to a few hundred frames: each thread's buffer caps at 262,144 events and drops (and
  reports) the rest.
