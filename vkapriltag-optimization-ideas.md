# vkapriltag optimization ideas

> **STATUS — reviewed against the tree, 2026-09-19. Read this box first.**
>
> This document was written without `apriltags_vulkan/OPTIMIZATION_NOTES.md`
> in hand. That file is the 750-line record of what has already been tried on
> this pipeline, with numbers. Against it, **most of the list below is already
> implemented, already measured and rejected on the deployment target, or
> aimed at a configuration that is not deployed.** Per-item verdicts are
> inline, marked `VERDICT:`.
>
> Three framing errors affect the whole document:
>
> 1. **The profile is at decimation 1; the default and deployment config is
>    decimation 2.** `PERFORMANCE.md` section 3 measures the same image on the
>    same board at **3.76 ms**, not 9.5 ms. Since GPU time is
>    `560 us + 3.3 us per 1k source pixels`, decimation 1 inflates exactly the
>    pixel-proportional full-image passes that A1/A2/A3 target, by ~4x. The
>    41.6% / 22.3% span weights the whole priority ordering rests on are
>    therefore not the weights the deployed configuration has.
> 2. **The stated precondition is already met.** The bold caveat "before
>    acting on any item below, add per-dispatch timestamps" is stale:
>    `7587f1b`, the commit profiled, already emits **13** spans under
>    `APRILTAG_VK_TIMESTAMPS=1` (`kSpanClear` ... `kSpanReadbackCopy` in
>    `GpuDetector.h`), not the 5 tabulated here. The finer attribution this
>    document asks for is available from the same binary.
> 3. **Some figures are not from this configuration.** A1's "480k decimated
>    pixels" is not 1280x800 at decimation 1, which is 1.024M. The work
>    counters quoted elsewhere do reproduce exactly, so the profile is real —
>    but individual numbers in the prose are not all from it.
>
> **What came out of this review and was actually built** (all bit-identical
> to what they replace, verified across decimations 1/2/4, the 5-image
> corpus, both storage widths, and five workgroup geometries):
>
> - **B3, as integer packing rather than fp16** — `RawLineFitPoint` is now
>   8 bytes instead of 16. Readback **-49%** and device memory **-15 to -17%**
>   (both exact, not timings); GPU total -3 to -10% across four sessions on an
>   RX 9060 XT, a spread wider than the effect, so treat the sign as the
>   result. **Shipped on.**
> - **A2, both fusions** — `decimate`+`block_minmax` and
>   `block_filter`+`threshold`. **Shipped off**, behind
>   `APRILTAG_VK_FUSE_PREPROCESS`: they measure a consistent *loss* on the
>   only device available to test (+12.7% on the span), for a mechanism that
>   should invert on the bandwidth-bound target. See `PERFORMANCE.md`
>   section 3a for the one-command A/B to settle it on the Pi.
> - **A6 was bounded and dropped** — see its entry.

Candidate algorithmic changes for the vkapriltag GPU detector, split by whether
they work on any Vulkan 1.1 device or depend on optional features.

## Measurement context

Everything below is grounded in a profile taken on real hardware, not estimated:

| | |
|---|---|
| Board | Orange Pi 5 Plus (RK3588, Mali-G610 MP4, 4 shader cores) |
| Driver | ARM libmali `g24p0` on the vendor kbase driver, Vulkan 1.3.276 |
| OS | Armbian-unofficial 26.5.2 trixie, vendor kernel 6.1.115-vendor-rk35xx |
| GPU clock | devfreq governor `performance`, pinned 1000 MHz |
| vkapriltag | `main` @ `7587f1b` |
| Workload | `grayimage.pgm` 1280x800, tag36h11, decimation 1, 20 iterations |
| Command | `apriltag_vulkan_validate --pgm apriltags_vulkan/grayimage.pgm --family tag36h11 --iterations 20 --decimation 1` |

Chosen geometry for that run: `wg1d=128, wg2d=8x8, scan_wg=1024,
unified_memory=yes, host_cached=yes, timestamps=yes`.
Work counters: `boundary_points=189634, raw_blobs=1257, uf_iterations=2,
submits=4, blobs=458, points=51386`.

### Baseline profile

GPU total median **9.5 ms**, `pipeline_total` median **11.7 ms**, versus the CPU
libapriltag reference at **43-45 ms** on the same image.

| Span | ms | % of frame | Dispatches covered |
|---|---|---|---|
| threshold+label | 3.60 | 41.6% | decimate, block_minmax, block_filter, threshold, uf_init, uf_compress, uf_merge |
| boundary | 2.13 | 22.3% | blob_diff over interior pixels |
| sort+group | 1.93 | 20.2% | hash_group, build_indirect_args, init_extents, reduce_extents_hash, select_blobs, extract_blob_counts, scatter_index_points |
| linefit | 1.30 | 13.6% | sort_points_local + line fit |
| upload + readback | 0.22 | 2.3% | |

> **Caveat.** These are the five instrumented spans. Attribution of cost to
> individual shaders *within* a span is inference, not measurement. Before
> acting on any item below, add timestamps around the individual dispatches -
> the `vk::QueryPool` machinery for this already exists. It is entirely
> possible that, say, `uf_merge` is a rounding error inside `threshold+label`
> and `block_minmax` is the whole cost.

---

## A. Core Vulkan 1.1 - portable to every device

### A1. Tile-local connected components instead of iterative global union-find
**Targets the 41.6% span. Expected to be the single biggest win.**

Today labelling is `uf_init` -> `uf_compress` -> N x (`uf_merge` + `uf_compress`)
-> `uf_final` -> `label_pixels`, where each is a full pass over all 480k
decimated pixels using global atomics and parent-pointer chasing. Even at the
observed `uf_iterations=2` that is roughly seven full-image passes.

Instead, label each 8x8 or 16x16 tile completely in shared memory - one pass, no
global atomics, no pointer chasing across DRAM - then merge only across tile
borders. Border work is O(perimeter) rather than O(area). This is the standard
GPU CCL formulation (Komura-equivalence, BUF/BKE family) and typically yields
2-3x on this stage.

> **VERDICT:** **Already tried on Mali; much slower, and worse the bigger the tile.**
> `OPTIMIZATION_NOTES.md` measured `threshold+label` at 3.58 ms flat vs 4.02
> (4x4), 4.62 (8x8), 5.32 (16x16), 6.63 (32x32). The tiling worked as intended
> - at 16x16 it cut global merge/compress from 2.70 to 1.46 ms - but the tile
> pass itself cost 3.00 ms to save 1.24. Valhall has no scratchpad: `shared`
> is backed by L2, so a dependent shared load is not cheaper than a global
> one. Separately, the BUF/BKE citation does not transfer: those schemes
> require every foreground pixel of a 2x2 block to be connected, which holds
> for binary 8-connected labelling, while this pipeline is 4-connected and
> three-valued (127 merges with nothing), so the 4x node reduction is simply
> unavailable. Would plausibly win on a discrete part - where the whole GPU
> phase is already 0.66 ms.


### A2. Fuse the four 2D preprocessing dispatches
**Targets the 41.6% span.**

`decimate` -> `block_minmax` -> `block_filter` -> `threshold` are four separate
dispatches, each of which round-trips a full image through global memory.
`decimate` and `block_minmax` fuse naturally: compute the per-block min/max
while the decimated tile is still in registers or shared memory. `block_filter`
and `threshold` fuse the same way. On a bandwidth-bound integrated GPU this
approaches halving the stage's memory traffic.

> **VERDICT:** **Correct mechanism, and now built - but it measures a loss on the
> only hardware available.** Both fusions are in the tree
> (`decimate_minmax.comp`, `threshold_filter.comp`), bit-identical, behind
> `APRILTAG_VK_FUSE_PREPROCESS`, **off by default**. On an RX 9060 XT the
> `threshold` span goes +8.1% (minmax), +4.1% (threshold), +12.7% (both) -
> additive, so it is a result and not noise. That card's L2 holds the block
> grid and the decimated image, so the DRAM round trips being removed were
> never paid, while the shared staging and barriers added are. The target is
> the opposite kind of part, which is the argument for expecting the reverse
> there - but `OPTIMIZATION_NOTES.md` already estimates individual fusions at
> ~1-2% on Mali, so calibrate to that, not to "halving". One env var settles
> it; see `PERFORMANCE.md` section 3a.


### A3. Fold boundary detection into the labelling pass
**Targets the 22.3% span.**

`blob_diff` is a separate full-image pass, but "is this pixel on a blob
boundary" is a purely local 4-neighbour label comparison. If A1 is adopted, the
tile already holds labels in shared memory, so boundary points can be emitted
there and the dedicated pass mostly disappears.

> **VERDICT:** **Depends on A1, which is rejected.** Also overtaken: `blob_diff`
> has already been cut twice - per-pixel labels took `boundary` 3.25 -> 1.71
> ms, and folding the threshold into `parent[]`'s spare bits took another 33%
> on Intel. It now measures ~0.20 ms on both second-pass devices. Note too
> that changing the emit order shifts the candidate-quad count by +/-1-2
> (`PERFORMANCE.md` section 7), which costs the bit-identical A/B gate the
> last two optimization passes relied on.


### A4. Remove the mid-frame CPU readback
**Structural. Helps latency more than throughput.**

`ReadCounterSlot(kSlotUfChanged)` (GpuDetector.cpp) reads a convergence counter
back to the CPU mid-frame to decide whether to run more union-find iterations.
That round trip is what forces the frame to be split into four
`SubmitAndWait` calls, each a `vkQueueSubmit` followed by a blocking
`vkWaitForFences`.

Do the convergence test on the GPU and make surplus iterations no-ops driven by
`DispatchIndirect`. The required machinery (`build_indirect_args.comp`,
`DispatchIndirect`) is already used elsewhere in the same file. This collapses
four submits toward one.

> **VERDICT:** **Tried twice, reverted twice, and the premise is measured at
> ~0.05 ms.** An empty submit+fence round trip is 19 us on Mali (the Mali is
> *cheaper* than the RX 9060 XT's 42 us, having no PCIe hop). `bcfa3dc`
> merged two submissions via device-side indirect dispatch, `ecaeaae`
> reverted it, `f436eb3` records why: removing a whole submission moved the
> unspanned residual by 0.05 ms, not the 0.37 ms that dividing the residual
> by submit count predicted. Separately, the convergence check this targets
> is itself free - deleting the second merge entirely, as a deliberately
> incorrect build, measured **zero** on the Pi.


### A5. Overlap the CPU tail with the next frame's GPU work
**~15% throughput, and it need not cost latency.**

`pipeline_total` (11.7 ms) minus `GPU total` (9.5 ms) is roughly 2 ms of
CPU-side `quad_decode` + `tag_decode` during which the GPU is idle. Overlapping
frame N's GPU work with frame N-1's CPU tail reclaims that. Unlike classic frame
pipelining this does not inherently add latency, because the two stages occupy
different resources.

Note there is already a `perf/frame-pipelining` branch in the vkapriltag repo,
so this may be partly explored already.

> **VERDICT:** **Already done and merged.** `library/src/FramePipeline.cpp`;
> `PERFORMANCE.md` section 5 measures 5.00 -> 3.52 ms on the Pi and 1.04 ->
> 0.78 ms on the desktop. The 11.7 ms `pipeline_total` quoted here is the
> serial path. Two caveats that section records: it is resolution-dependent
> (neutral at 640x400 on the Pi, and an earlier attempt on an older tree
> measured a 21% *regression*), and it forces a copy of
> `last_line_fit_points`, so zero-copy readback and pipelining do not
> compound.


### A6. Size dispatches by actual counts rather than capacities
Several dispatches are sized by worst-case capacity rather than the real count,
for example `extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, ...)`.
Moving these onto `DispatchIndirect` with GPU-computed counts (already done for
`init_extents` and `select_blobs`) avoids launching invocations that immediately
exit. With only four shader cores, wasted launches are not free.

> **VERDICT:** **Bounded at ~4 us and dropped.** This was initially ranked the best
> item on the list, on the strength of the 0.25 ms `OPTIMIZATION_NOTES.md`
> attributes to right-sizing `init_extents` / `select_blobs`. That figure
> does not transfer: those two run over `max_raw_blobs` (65536) reading
> 48-byte structs, and they are *already* indirect. The one dispatch left,
> `extract_blob_counts`, runs over `max_blobs` (2048-4046, ~32x smaller)
> writing 4 bytes per invocation. Bounded directly by shrinking the capacity
> 8x with `--max-blobs 512`: `blob_scan` moved 0.01044 -> 0.00632 ms, so
> ~4 us is the entire capacity-proportional cost of the dispatch *and its
> whole scan chain*. The constructor's `max_blobs` comment already says the
> same thing (+2.2 us for a 24x capacity increase) and explains that the
> slack is deliberate - it buys reproducibility, since overflowing
> `max_blobs` makes detections depend on GPU scheduling order. Rejected.


### A7. Manually pack pixels four-per-`uint`
A portable fallback that captures most of the bandwidth benefit of 8-bit storage
(B2) without requiring the extension, for devices that lack it.

> **VERDICT:** **Fallback-only work for hardware not in play.**
> `VK_KHR_8bit_storage` is core in Vulkan 1.2 and present on both the Mali
> and the desktop part. The 32-bit path already exists as the fallback and is
> exercised by `APRILTAG_VK_FORCE_NO_8BIT`. Byte-granular writes from
> separate invocations would also need read-modify-write or a layout where
> each invocation owns a whole word - which is the reason the u8 path uses
> the extension instead.


---

## B. Depends on optional features

### B1. Subgroup stream compaction - the highest-value optional item
**Requires `SUBGROUP_FEATURE_BALLOT` + `SUBGROUP_FEATURE_ARITHMETIC` in compute.
Targets the 22.3% and 20.2% spans.**

`scatter_index_points` and the boundary-point emit path perform a global
`atomicAdd` per point to reserve output slots. With a subgroup ballot plus a
subgroup prefix sum, each subgroup performs **one** atomic on behalf of all its
lanes - at the G610's subgroup size of 16, that is 16x fewer global atomics.
Atomic-heavy compaction is usually where this transformation pays best.

The codebase already has `_subgroup` variants for `blob_diff`, `uf_final` and
`reduce_extents_hash`, plus capability gating in `vk/Context.cpp`
(`has_subgroup_ballot`, `has_subgroup_arithmetic`, `has_subgroup_shuffle`) and
an `APRILTAG_VK_FORCE_NO_SUBGROUP` escape hatch, so the pattern and the fallback
discipline already exist.

> **VERDICT:** **The single most firmly ruled-out item here, and it is ranked
> highest.** `GpuDetector::CreatePipelines()` excludes subgroup variants on
> integrated GPUs *outright*, regardless of reported capability, because on
> this exact Mali-G610 - which advertises all three features and produces
> bit-correct results with them - enabling them took `pipeline_total` from
> **11.5 ms to 18.1 ms**, with `reduce_extents_hash_subgroup.comp`'s
> `extents` span alone going 0.91 -> 6.04 ms. Read the comment above the
> `subgroup` flag; it is 25 lines and it is about precisely this idea.
>
> Two premises are also wrong. `blob_diff`'s emit path is *already*
> ballot-aggregated where enabled (`AGGREGATE_APPEND_COUNTER` in
> `blob_diff_body.glsl`). And `scatter_index_points`' atomic is a **per-blob
> cursor**, not one global counter — lanes in a subgroup hit different keys,
> so aggregating it needs a partitioned reduce-by-key loop, which is exactly
> the construct that measured 6x worse in `reduce_extents_hash` on this part.
>
> The likely source of the confidence: `PERFORMANCE.md` section 6 used to
> claim subgroup variants were "worth ~2.7%" on Mali and gated on capability
> alone. Both halves were wrong. That sentence has now been corrected.


### B2. 8-bit storage
**Requires `VK_KHR_8bit_storage` (core in Vulkan 1.2). Already implemented.**

The `*_u8.comp` shader variants, selected when the device supports it, keep the
decimated image as `u8` instead of `u32` - 4x less bandwidth. Already gated
behind `force_no_8bit_storage`.

### B3. 16-bit storage / `shaderFloat16`
**Requires `VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8`.**

`line_fit_points_buf_` is multi-megabyte. Storing point coordinates and line-fit
moments as fp16 or int16 halves that traffic. Precision must be validated
against the existing libapriltag parity test before adopting.

> **VERDICT:** **Right target, wrong mechanism — and implemented as integer
> packing, which needs no extension at all.** `RawLineFitPoint` is now 8
> bytes instead of 16: `x2`/`y2` in 14 bits each (the constructor already
> rejects `2*(width/decimation) > 16383`), `W` in 10 (it is
> `int(sqrt(gx^2+gy^2))+1` over 8-bit gradients, so <= 361), `blob_index` in
> 22. Exact, no `VK_KHR_16bit_storage`, no precision question to validate.
> Measured: readback -49% and device memory -15 to -17% (exact); GPU total
> -3 to -10% across four sessions, a spread wider than the effect itself.
> **Shipped on.**
>
> The fp16 half of the suggestion is the one genuinely accuracy-degrading
> idea in this document, by two separate mechanisms. fp16's 11-bit mantissa
> makes integers above 2048 round to even, and `x2` reaches 3839 at 1080p /
> decimation 1 - a half-pixel coordinate error on the right of the frame,
> landing straight in corner positions. And fp16 *moments* are further out
> still: the CPU rebuilds `Mxx/Mxy/Myy` in native `int64` precisely because
> the covariance is a near-total cancellation - the same reason
> `PERFORMANCE.md` section 4's `fast` refinement keeps its line fit in
> double even after narrowing everything around it.


### B4. Subgroup arithmetic for block min/max reductions
**Requires `SUBGROUP_FEATURE_ARITHMETIC`.**

Replaces the shared-memory-plus-barrier reduction in `block_minmax` with
`subgroupMin` / `subgroupMax`.

> **VERDICT:** **Dead on the target for B1's reason** (subgroup variants are
> disabled outright on integrated GPUs), and near-zero on discrete: the whole
> `decimate`+`threshold` stage is 0.86 ms of Mali's 2.59 ms labelling stage,
> and 0.013 ms of an RX 9060 XT frame. Note also that today's
> `block_minmax.comp` has no shared-memory-plus-barrier reduction to replace
> - each invocation reduces its own 4x4 window in registers. The shader this
> describes is the *fused* one added under A2.


### B5. Subgroup sort for the per-blob point sort
**Requires `SUBGROUP_FEATURE_SHUFFLE`. Low priority - est. 2-4% total.**

`sort_points_local_body.glsl` is a workgroup-wide bitonic network in shared
memory with a `barrier()` between each of the ~log2(cap)*(log2(cap)+1)/2 stages.
Within a subgroup, lanes run in lockstep and can exchange values via
`subgroupShuffleXor(v, j)` with no barrier and no shared memory; a bitonic
network maps directly onto that (shuffle, then min/max select).

The limit here is that the G610's subgroup size is 16 while blobs average ~112
points (51386 points / 458 blobs), so a subgroup-only sort does not cover a
blob. A hybrid - sort 16-wide chunks by shuffle, then merge the chunks through
shared memory - cuts barriers substantially but does not eliminate them.

Because the whole `linefit` span is only 13.6% of the frame, and the sort is
only part of that span, even a free sort caps out around 5-8% total. This is
the lowest-value item on the list.

> **VERDICT:** **Correctly self-ranked last, and its own ceiling is still 5-10x
> too generous.** Already bounded in `OPTIMIZATION_NOTES.md` as item A9, by
> the deliberately-incorrect-build method: removing **all** bitonic
> `barrier()` calls measured `sort` at 0.1198 vs 0.1321 ms on an MX230 and
> 0.3989 vs 0.4205 on Intel — a ceiling of 9% and 5% of a small span, i.e.
> 0.012-0.022 ms, not 5-8% of the frame. It would also depend on local
> invocation indices mapping to subgroup lanes contiguously, which is not
> guaranteed without `VK_EXT_subgroup_size_control`. Rejected there; nothing
> has changed.


### B6. Timeline semaphores
**Requires `VK_KHR_timeline_semaphore` (core in Vulkan 1.2).**

Expresses the dependencies in A4 and A5 more cleanly than the current
fence-per-submit approach, without full CPU stalls. Not strictly required for
either.

---

## Suggested order

> **SUPERSEDED.** This ordering was built on the three framing errors in the
> status box, and it puts the two most firmly rejected items (A1, B1) first.
> Replaced by the list below.

~~1. **A1 + A2 + A3** together - they attack 64% of the frame, are portable,
   and partly reinforce each other (A3 depends on A1's tile pass existing).~~
~~2. **B1** - best optional-feature win, and the gating pattern already
   exists.~~
~~3. **A4 + A5** - structural; mostly latency and pipeline-occupancy rather
   than raw GPU time.~~
~~4. **A6, B4** - cleanup tier.~~
~~5. **B5** - lowest value.~~

~~Before any of it: **add per-dispatch timestamps**~~ - already emitted, 13
spans, under `APRILTAG_VK_TIMESTAMPS=1`.

## Revised order

1. **Run the A/B that is already set up.** `APRILTAG_VK_FUSE_PREPROCESS=1`
   against `0` on the Orange Pi, ABBA-interleaved (`PERFORMANCE.md` section
   3a has the loop). No code, no rebuild, one environment variable. It
   decides whether pass-count reduction — the thing `PERFORMANCE.md` names as
   where the Mali time actually is — is worth pursuing further, and until it
   runs everything else in that direction is speculation. If it wins, flip
   the default and keep fusing; if it loses, the whole A1/A2/A3 family is
   closed and the notes should say so.
2. **Confirm the packed record on the target.** It is on by default and
   measured only on a discrete card. It should help *more* on a
   unified-memory part, where the readback crosses the same LPDDR bus as
   everything else, but "should" is what this document was already too full
   of.
3. **The DP corner-seeding fallback rate.** The only substantive item here
   that nothing has measured, and the only one that touches accuracy — so
   gate it on the corner RMS, not the tag ID set. Measure at the deployed
   decimation, where the rate is 2.94%, not the 6.99% quoted at decimation 1.
4. **Nothing else on this list.** A4, A5, A6, B1, B2, B4, B5 are done,
   rejected with numbers, or bounded below a microsecond-to-millisecond
   threshold that does not justify the code. A1 and A3 are rejected *on this
   hardware* and would need a different target to revisit.

And before adding to this document: read
`apriltags_vulkan/OPTIMIZATION_NOTES.md` first. Most of what looks obvious
about this pipeline has already been measured, and roughly half of it
measured backwards.

---

## Non-Vulkan item

The same run reported:

```
DP corner seeding: 32/458 blobs fell back to the combinatorial search (6.9869%)
```

That fallback is CPU-side, inside `quad_decode`. Reducing the fallback rate
shrinks the ~2 ms CPU tail directly, and is independent of any GPU work above.

> **VERDICT:** **Genuinely unexplored — and the one item here where "optimizing"
> directly trades accuracy.** The combinatorial search exists *because* DP
> seeding failed on those blobs; making DP accept more cases means accepting
> seeds it currently judges bad. So this one has to be gated on the corner
> RMS, not just on the decoded tag ID set - the ID set survives corner errors
> that matter. Note the rate is configuration-dependent: 6.99% at decimation
> 1, but 2.94% at the deployed decimation 2 and 0% at decimation 4 on the
> same image.


---

## Already ruled out - do not re-investigate

These were tested on hardware and found not to be the problem. Recorded so the
same ground is not covered twice.

- **Workgroup size tuning.** Sweeping `APRILTAG_VK_WG` over 32/64/128/256/512/1024
  moved `threshold+label` only between 3.49 and 3.65 ms, and sweeping
  `APRILTAG_VK_WG2D` over 4x4/8x8/16x16/32x32/16x8/8x16/32x8/64x4 moved it only
  between 3.57 and 3.72 ms. Run-to-run noise at fixed settings is 3.56-3.74 ms.
  **Both sweeps are narrower than the noise floor** - the auto-chosen
  `wg1d=128, wg2d=8x8` is already optimal on libmali.

- **GPU clock / devfreq governor.** Pinning `performance` is worth roughly 20%
  and removes most frame-time variance, and is now done by the image. Beyond
  that, clocks are not a limiter.

- **Barrier cost in the bitonic sort.** A theory that the per-stage `barrier()`
  in `sort_points_local` was pathological on Mali. Disproved: the identical
  shader takes 91.8 ms under Mesa PanVK and 1.18 ms under libmali. It was driver
  code generation, not barriers.

- **Mesa PanVK as a driver choice.** 141 ms/frame versus libmali's 11 ms on
  identical hardware and binary, i.e. slower than the 43 ms CPU detector. Worst
  single shader 78x slower, adaptive threshold 11.5x slower, and PanVK exposes
  neither host-cached memory types nor timestamp queries. PhotonVision images
  therefore ship libmali on the vendor kbase driver.
