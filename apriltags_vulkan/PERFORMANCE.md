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
discrete card, painful on a unified-memory part.

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
- **Subgroup variants** are selected when the device advertises ballot /
  arithmetic support. On Mali they are worth ~2.7%. Note that subgroup
  aggregation and cheap atomic early-outs are *alternative* answers to the
  same contention, and applying both is worse than either — which is why the
  guards in `uf_final` / `reduce_extents_hash` are deliberately absent from
  their subgroup-aggregated variants.

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

1. **A better connected-components algorithm.** Union-find is ~20-24% of the
   Mali GPU phase, which caps any rewrite there. Note that BUF /
   Playne-Equivalence **do not apply**: they require every foreground pixel of
   a 2x2 block to be connected, which holds for binary 8-connected labelling,
   while this pipeline is 4-connected and three-valued (127 merges with
   nothing). Tile-local union-find was tried on Mali and was much slower —
   Valhall has no scratchpad, so `shared` is backed by L2 and dependent shared
   loads are not cheaper than global.
2. **Reducing the number of full-image passes.** This is where the Mali time
   actually is (section 3). Individual fusions each measured ~1-2%, so it only
   pays as a sustained campaign, not a one-off.
3. **Not the labelling convergence check.** Removing the second merge entirely
   — a deliberately incorrect build, used to bound the ceiling — measured
   **zero** on the Pi. `uf_merge` already performs no writes and no atomics on
   a converged pass, so it is effectively a read-only verification pass
   already, and there is nothing for a cheaper convergence proof to reclaim.

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

Two cautions, both learned the hard way here:

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
