# Performance and portability notes

Targets: embedded ARM Mali-G610 (the deployment target), discrete AMD
(Radeon RX 9060 XT / RDNA4), and discrete NVIDIA. The pipeline needs **only
core Vulkan 1.1** and enables **zero optional device features**.

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
| `APRILTAG_VK_UF_CHUNK=<n>` | Labelling iterations issued per convergence check (default 2). |
| `APRILTAG_VK_MIN_TAG_PX=<n>` | Geometric prefilter: minimum tag size in pixels. |
| `APRILTAG_VK_QUADFIT=dp\|peaks` | Corner-seeding method. `dp` is the default. |
| `APRILTAG_VK_REFINE=exact\|fast\|upstream` | Edge-refinement implementation; only consulted when the caller sets `td->refine_edges`. See section 4. |
| `APRILTAG_VK_FORCE_NO_SUBGROUP=1` | Force the scalar shader variants even where subgroup ops are available. |
| `APRILTAG_VK_FORCE_NO_8BIT=1` | Force the 32-bit-per-pixel shader variants even where `VK_KHR_8bit_storage` is available. |
| `APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1` | Force the 32-bit-atomic extents reduction even where `VK_KHR_shader_atomic_int64` is available. See section 3b. |
| `APRILTAG_VK_FUSE_SUBMITS=0` | Keep the four-submit frame even where the tail could be recorded as one. See section 3c. |
| `APRILTAG_VK_WG2D=<w>x<h>` | Override the 2D workgroup size. |
| `APRILTAG_VK_TIMESTAMPS=1` | Emit the per-dispatch GPU timestamp-span breakdown. |
| `APRILTAG_VK_PIPELINE_CACHE=0` | Disable the on-disk pipeline cache. |
| `APRILTAG_VK_CACHE_DIR=<path>` | Where that cache lives. |
| `APRILTAG_CPU_THREADS=<n>` | Parallelism for the CPU tail, including the calling thread. Default = `hardware_concurrency()`; 1 = serial. |

## 3. Measured performance

`grayimage.pgm` (1280x800), decimation 2, otherwise default config, medians
over 300-400 iterations. **These are not the same conditions as the tables in
`README.md`**, which use `colorImage.pgm` (1920x1080) with
`APRILTAG_VK_MAX_POINTS=200000` — do not compare the two directly.

| Stage | Runs on | Orange Pi 5 Plus (Mali-G610) | Desktop (RX 9060 XT) |
| --- | --- | --- | --- |
| `GpuDetector` | GPU | 3.76 ms | 0.66 ms |
| `quad_decode` | CPU | 0.44 ms | 0.09 ms |
| `tag_decode` | CPU | 0.79 ms | 0.29 ms |
| **Pipeline total, serial** | GPU + CPU | **5.00 ms** | **1.04 ms** |
| **Pipeline total, `FramePipeline`** | GPU \|\| CPU | **3.52 ms** | **0.78 ms** |

The GPU phase dominates on the deployment target (75% of a serial frame) and
is essentially all real shader execution. Two things it is *not*:

- **Not submit-bound.** An empty `vkQueueSubmit` + `vkWaitForFences` round trip
  measures **19 us on Mali** and **42 us on the RX 9060 XT** — the Mali is the
  *cheaper* of the two, having no PCIe hop. All four of a frame's submits are
  ~1.5% of a Mali frame. Collapsing them is not worth the complexity; it was
  tried and reverted (`OPTIMIZATION_NOTES.md`).
- **Not barrier-bound on the discrete part.** A barrier costs a fixed ~1.6 us
  on RDNA4 regardless of in-flight work, and is free when there is nothing to
  drain; all ~31 of a frame's barriers total ~48 us. On Mali the cost instead
  *scales* with in-flight work (2.6 us between trivial dispatches, 18.7 us
  between large ones), so barrier count matters there and only there.

What the Mali GPU phase *is* bound by is memory traffic across its ~11
full-image passes. Sweeping the input over 320x200 / 640x400 / 1280x800 gives
a consistent marginal rate:

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

Stock `7587f1b` against everything on this branch, on the Mali-G610,
ABBA-interleaved, min of 12, **both binaries built `Release`**:

| | GPU total | `pipeline_total` | device memory |
| --- | --- | --- | --- |
| decimation 1 | **-11.2%** | -9.2% | 185 -> 155 MiB |
| decimation 2 | **-11.0%** | -7.5% | 48 -> 42 MiB |
| decimation 4 | **-12.5%** | -9.8% | 15 -> 13 MiB |

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
discrete card without resizable BAR keeps the four-submit path, and the
RX 9060 XT here does exactly that - it is the fallback that stays verified,
not a dead branch.

### What is left of this

One boundary remains, before `uf_final`, because the host still reads the
labelling convergence flag to decide whether to run another union-find chunk.
Measured on its own it is **0.28 ms, ~9% of the frame** - now the largest
single non-span item.

Removing it exactly (without guessing an iteration count and risking
under-converged labels) wants a retry rather than a speculation: record the
whole frame with the chunk count the *previous* frame needed, read the
convergence flag with all the other counters at the end, and if it says the
labelling did not converge, redo the frame with a larger count. `clear`
re-initializes every buffer, so a redo is simply correct, and
`last_uf_iterations_` already adapts after one frame - so a scene that needs
more iterations pays double once, not every frame. Not attempted here.

## 4. Edge refinement (`APRILTAG_VK_REFINE`)

Upstream libapriltag's `refine_edges` is the single most expensive CPU
function in the pipeline. `library/src/RefineEdges.cpp` provides three
implementations:

- **`exact`** (default) — identical arithmetic in identical double precision,
  with `modf()` rewritten as `trunc()` plus a subtract. This is bit-identical
  to upstream *by construction*, not by measurement: `modf(x, &i)` is defined
  to return `x - trunc(x)` and both results are exactly representable. What it
  removes is a non-inlinable libm call from the innermost loop.
- **`fast`** — additionally narrows the innermost sampling loop to float. The
  per-edge line fit stays double regardless: it builds a covariance from raw
  second moments, a near-total cancellation that float would destroy.
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
a frame cost `max(GPU, CPU)` rather than the sum — the 5.00 -> 3.52 ms and
1.04 -> 0.78 ms rows in section 3.

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
a near-total cancellation, the same reason section 4's `fast` refinement
keeps its line fit in double.

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
obvious ideas are in the rejected column. The short version:

0. **The one remaining submit boundary**, before `uf_final`, worth **0.28 ms
   (~9%)**. Two of the original three are gone (section 3c); this one is held
   open by the host readback of the labelling convergence flag, and section
   3c sketches the retry-based way to close it exactly. Note that the old
   "collapsing submits is not worth it" verdict in `OPTIMIZATION_NOTES.md`
   was measured when the frame was 11.5 ms and the same absolute cost was
   ~1%; everything else has since got ~4x faster and this had not, which is
   why it was worth re-measuring rather than inheriting.
1. **A better connected-components algorithm.** `labelling` is now the
   largest span at ~29% of the Mali GPU phase, which caps any rewrite there. Note that BUF /
   Playne-Equivalence **do not apply**: they require every foreground pixel of
   a 2x2 block to be connected, which holds for binary 8-connected labelling,
   while this pipeline is 4-connected and three-valued (127 merges with
   nothing). Tile-local union-find was tried on Mali and was much slower —
   Valhall has no scratchpad, so `shared` is backed by L2 and dependent shared
   loads are not cheaper than global.
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

For per-dispatch GPU timing, `APRILTAG_VK_TIMESTAMPS=1` adds a timestamp-span
breakdown plus the gaps between spans, separating intra-submit gaps from
submit boundaries. For an independent view on AMD, RADV can emit RGP traces
with `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1` (the per-submit flag is
what makes it work for a headless compute app with no `vkQueuePresentKHR`).
RenderDoc needs an artificial frame boundary for the same reason, and Mali
exposes no external GPU profiler at all — there, attribute cost by external
wall-clock deltas while varying a knob or the input size.
