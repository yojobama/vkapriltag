# Optimization notes: grouping, sorting and line fitting

Measured end to end on the intended deployment target — an Orange Pi 5
(RK3588, Mali-G610, 8 GB LPDDR), Ubuntu 6.1.0-1025-rockchip, Arm proprietary
Vulkan driver `g6p0-01eac0`, Vulkan 1.2.165 — running the tracked
`colorImage.pgm` at 1920x1080 through `apriltag_vulkan_validate`.

Every number below is best-of-20-to-25 with `APRILTAG_VK_MAX_POINTS=200000`, and
every step was checked against the unmodified libapriltag CPU detector: the
candidate quad count stayed at **97** and the decoded tag set at **[554]**
throughout.

## Summary

| Step | Change | GPU total |
| --- | --- | --- |
| — | baseline (`87ae653`) | 39.48 ms |
| 1 | bitonic network sized per blob | 30.59 ms |
| 2 | 16-byte line-fit record instead of 48 | 27.87 ms |
| 3 | rank only roots that can own a boundary point | 23.14 ms |
| 4 | hash grouping replaces the (rep0, rep1) sort | 15.34 ms |
| 5 | 256-thread local sort + CPU tail cleanup | 14.31 ms |
| 6 | delete the now-unreachable sort machinery | 14.09 ms |
| 7 | per-pixel labels remove `blob_diff`'s random gathers | 12.51 ms |
| 8 | horizontal runs pre-joined in `uf_init` | **11.53 ms** |

Per-stage, baseline versus now:

| Stage | Before | After |
| --- | --- | --- |
| upload | 0.22 | 0.22 |
| **threshold + label** | **3.58** | **2.59** |
| **boundary** | **3.24** | **1.73** |
| **sort + group** | **15.83** | **3.01** |
| **linefit** | **13.41** | **3.05** |
| readback | 3.25 | 1.09 |
| GPU total | 39.48 | 11.53 |
| CPU `quad_decode` | 4.15 | 3.7 |
| **pipeline total** | **44.3** | **16.1** |

Device memory also drops, since the sort's scratch is gone: 55 -> 41 MiB at
`APRILTAG_VK_MAX_POINTS=200000`, and 261 MiB at the default dense sizing.

## A note on the line fit itself

`FitLine()` already *is* eigenvector (PCA) line fitting: it forms the scatter
matrix `Cxx`/`Cxy`/`Cyy` from the moment sums and takes the smaller eigenvalue
and its eigenvector in closed form. There is no better formulation to switch
to, and at O(1) per call it was never the cost. The time was in the machinery
that *fed* it — the global sort used to group the points, and a bitonic
network that ignored how many points each blob actually had.

---

## 1. Size the per-blob bitonic network to the blob

`sort_points_local.comp` sorted `kLocalCap` (2048 on this device) virtual
slots for **every** blob, regardless of its point count. The mean blob on a
1080p frame has 233 points. A 2048-slot network is 66 stages; a 256-slot one
is 36, over 8x fewer elements per stage.

`count` is uniform across the workgroup, so the derived `cap` is too and every
`barrier()` is still reached by every invocation. Slots in `[count, cap)` hold
`0xFFFFFFFF` sentinels and sort to the end either way, so the result is
bit-identical.

This was a latent regression rather than a missing optimization: `linefit` on
this device was **worse** with the segmented shared-memory sort (13.4 ms) than
with the flat radix sort it replaced (10.4 ms), purely because of the fixed
network width.

> linefit 13.41 -> 4.62 ms

## 2. Ship only the independent line-fit quantities

`RawLineFitPoint` was 48 bytes carrying `Mx`, `My`, `W` and 64-bit `Mxx`,
`Mxy`, `Myy` split into hi/lo halves. All six are exact functions of
`(x2, y2, W)`, so 32 of every 48 bytes conveyed no information — and producing
them required the `imulExtended` hi/lo dance that exists only because GLSL core
has no `int64`.

The record is now `{ x2, y2, W, blob_index }` (16 bytes) and the CPU
reconstructs the moments in native `int64` during the prefix sum it was
already performing. Readback drops from 4.93 MB to 1.64 MB per frame and
`compute_line_fit_points.comp` writes a third of the bytes.

> readback 3.25 -> 1.09 ms, linefit 4.62 -> 4.08 ms

## 3. Rank only the roots a boundary point can name

`mark_roots.comp` flagged every union-find root, and the resulting root
**count** is what sets the sort key width and therefore the radix pass count.
On this frame that count was **153886** — nearly a third of all decimated
pixels — because it included every single-pixel speck and every ambiguous
region.

But `blob_diff.comp` refuses to emit a point whose blob is smaller than
`min_cluster_pixels`, so those roots are unreachable from the key. Filtering
the flag by `blob_size` drops the count to **663**: 18 bits of key become 10,
and 5 radix digits per word become 3.

This is a four-line change and was the cheapest win in the set. (Step 7 later
retires the dense-id machinery altogether — once the grouping is a hash, the
key only has to be distinct, not narrow — but this step is what made the sort
cheap enough for the hash's margin over it to be measured honestly.)

> num_roots 153886 -> 663, sort+group 15.78 -> 10.90 ms

## 4. Group with a hash table instead of sorting

This is the main algorithmic change, and the answer to "is there a better
algorithm for the group+sort part": **the sort was solving a harder problem
than the pipeline poses.**

Nothing downstream consumes the *order* of the boundary points — only their
*grouping* — because `sort_points_local.comp` re-sorts each selected blob's
points by angle around its centroid immediately afterwards. Grouping by an
unordered key is a hash-table problem, not a sort problem. Where an LSD radix
sort needs `2 * ceil(bits/4)` stable passes, each reading and rewriting three
words per element, a hash insert needs one pass.

`hash_group.comp` does open addressing with linear probing over a table sized
to 4x `max_raw_blobs` (25% load factor; linear probing averages well under two
probes there). The insert is lock-free **and spin-free**, which is
load-bearing:

- A slot holds `owner + 1`, where `owner` is the index of the boundary point
  that claimed it. `0` means empty, so `vkCmdFillBuffer(0)` resets the table.
- A thread that loses the `atomicCompSwap` gets the winner's index back and
  can compare keys **immediately**, because the claimant's key was written by
  `blob_diff.comp` in an earlier, barriered dispatch and is therefore already
  visible.
- No thread ever waits on another thread's later write. Vulkan does not
  guarantee independent forward progress between workgroups, so a spin here
  could deadlock on a conformant implementation. This formulation cannot.

Two threads with the same key always converge on the same slot: they follow
the same probe sequence and stop at the first slot whose owner's key matches,
and a thread can never skip past an empty slot it would have claimed.

Downstream, `mark_slots.comp` plus an inclusive scan over the table replaces
`mark_heads.comp` plus a scan over every boundary point;
`reduce_extents_hash.comp` does the same arithmetic as `reduce_extents.comp`
keyed off the slot; and `scatter_index_points.comp` replaces
`rewrite_index_points.comp`, placing each point with a per-blob cursor instead
of an offset into a sorted array.

Net: roughly 30 dispatches (10 radix digit passes x 3, plus `gather`,
`mark_heads` and the qbp scan chain) collapse to 4.

> sort+group 10.90 -> 3.06 ms

## 5. Local-sort workgroup width, and the CPU tail

Once the bitonic network is sized to the blob (step 1) the typical network is
~256 slots wide, so `local_sort_cap_`'s previous choice of 1024 threads left
three quarters of its lanes idle in every stage while still reserving the full
shared-memory allocation. Measured on Mali-G610:

| threads | linefit |
| --- | --- |
| 64 | 5.71 ms |
| 128 | 3.83 ms |
| **256** | **3.08 ms** |
| 512 | 3.08 ms |
| 1024 | 4.03 ms |

On the CPU side, `FitQuadForBlob` heap-allocated four vectors per blob (several
hundred blobs per frame) and used `%` for every circular index — roughly a
dozen integer divisions per boundary point across the windowed-error loop, the
7-tap filter and peak detection. The scratch is now `thread_local` and grows
monotonically, and the wrap-around is conditional subtraction, which is exact
because every offset is within one period.

> linefit 4.08 -> 3.18 ms, quad_decode 4.15 -> 3.58 ms

## 6. Delete the unreachable sort machinery

With the sort gone, `radix_histogram`, `radix_scan_hist`, `radix_scatter`,
`bitonic_sort`, `bitonic_local`, `fill_max_key`, `gather_generic`,
`mark_heads`, `reduce_extents`, `rewrite_index_points`, the qbp scan chain and
`DetectorConfig::sort_algorithm` (with its `APRILTAG_VK_SORT` override) were
all dead. Removing them reclaims 14 MiB of device memory — `qbp_sorted_buf_`
alone is 6.4 MB at 200k points — and the key arrays no longer need to be
padded to a power of two, since only the grouping hash reads them.

`DetectProfile`'s `qbp_sort_n` / `ipoint_sort_n` / `sort_passes` are replaced
by `raw_blobs`, the number of distinct (rep0, rep1) pairs the frame produced.

## 7. One spatially-local label per pixel

`blob_diff.comp` needed two things per pixel and per neighbour: which blob it
belongs to, and whether that blob is big enough to matter. It got them with
`blob_size[parent[n]]` — six **random** gathers into a 2 MB array for each of
the ~515k interior pixels, plus two more per emitted point to translate roots
into dense ids through `root_dense_id[]`.

`label_pixels.comp` now precomputes `1 + root, or 0 if the blob is too small`
into a per-pixel array. That is exactly **one** random gather per pixel, and
every read in `blob_diff.comp` afterwards is a neighbouring address. It also
collapses the shader's two rejection tests into one comparison.

The dense-id machinery disappears with it. Dense ids existed only to keep the
sort key narrow; a hash key just has to be distinct, so `mark_roots.comp` and
its full-image inclusive scan (a pass plus a multi-level scan over 518400
elements, every frame) are gone. The hash table's scan gets its own,
much smaller chain.

> boundary 3.25 -> 1.71 ms

## 8. Pre-join horizontal runs in the init pass

The labelling stage is not arithmetic-bound, it is bound on dependent global
loads. Two ablations pin that down. Splitting the stage shows `uf_merge` is
essentially all of it — and running `uf_merge` *without* the compression
passes is **worse**, 5.24 ms against 3.60, which is the signature of chain
walking rather than of pass count:

| labelling passes run | threshold + label |
| --- | --- |
| none | 0.86 ms |
| compress only | 1.04 ms |
| merge + compress (was) | 3.60 ms |
| merge only | 5.24 ms |

`uf_init.comp` now points each pixel at its left neighbour when the two are
the same non-ambiguous value, so every horizontal run is already one component
before any merge pass runs. That is exactly the structure the first `uf_merge`
pass used to build for the right-hand edges — with all-singleton input,
unioning (i, i+1) hooks `parent[i+1] = i`, because hooking is by `atomicMin`
and i < i+1. Building it directly is a pure streaming write: no atomics, no
`find()` walks, no contention, where the merge pass paid two `find()`s, an
`atomicMin` and a retry loop per horizontal edge. `uf_merge.comp` is then
responsible only for the down edges, halving its union work; between them the
two still cover every edge exactly once, and the "a component's root is its
minimum index" invariant is preserved, because a run's leftmost pixel is its
minimum index.

One detail is load-bearing, and was worth 1.0 ms on its own: **a compression
pass has to run between the init and the first merge.** Without it the
vertical unions walk the run chains the init just built, paying an
O(run length) `find()` each, and the whole change measures as an exact wash
(3.577 ms against a 3.575 ms baseline). With it, 2.57 ms.

> threshold+label 3.58 -> 2.59 ms

---

## Measured and rejected

Recording these so they don't get re-litigated.

**Collapsing the four queue submissions into one.** The frame reads three
device-side counts back to the host, each costing a submit + fence. Replacing
them with `vkCmdDispatchIndirect` looked like an obvious win. It is not: an
empty `BeginCommands`/`SubmitAndWait` round trip measures **0.019 ms min /
0.029 ms median** on this device over 500 samples. Four of them is ~0.12 ms of
a 12.5 ms frame. Not worth the refactor.

**Right-sizing `init_extents` / `select_blobs` with an extra readback.** Those
two are dispatched over `max_raw_blobs` (65536) x a 48-byte struct, about 6 MB
of traffic for the ~734 blobs a real frame has. Ablating the dispatch width
down to 2048 confirmed the saving is real but small: sort+group 3.03 -> 2.79
ms. Buying the exact count with an extra submit measured a **wash** — the
round trip and its command recording cost about what the traffic does. The
0.25 ms is still there for the taking via `vkCmdDispatchIndirect`, which
avoids the round trip; it just needs the count passed to the shaders through a
binding rather than a push constant, since push constants are host-side.

**Pipelining the CPU tail behind the next frame's GPU stage.** Implemented as
`PipelinedDetector`: a 1-deep double-buffered handoff so frame N+1's GPU stage
(`GpuDetector::Detect`) runs on a background thread's tail work for frame N
(`QuadDecode` + `TagDecoder`) concurrently, instead of the strictly serial
`Detect -> Decode -> Decode` every caller used before. `GpuDetector::Detect()`
is already fully synchronous (four `SubmitAndWait`s, host copies complete
before it returns), so no device buffer needed double-buffering — only two
host-side things did: `last_selected_extents`/`last_line_fit_points` (copied
out before the tail runs, since the next `Detect()` overwrites them) and the
raw grayscale frame `TagDecoder` samples from (double-buffered, since the
caller may start capturing the next frame into the same buffer while the tail
is still reading it). Verified race-free — a ThreadSanitizer build ran 300
pipelined frames across the full corpus with zero reported races, and every
frame decoded identically to the serial path in both a normal and a TSan
build (5/5 corpus matches, corner RMS bit-identical).

It is nonetheless a **clear throughput regression on the Mali-G610/RK3588**,
at every corpus scale, confirmed after ruling out two obvious confounds:

| image | serial `pipeline_total` (median) | pipelined throughput/frame |
| --- | --- | --- |
| 320x200 | 2.30 ms | 2.31 ms |
| 480x304 | 4.46 ms | 4.32 ms |
| 640x400 | 3.14 ms | 3.55 ms |
| 960x600 | 7.21 ms | 6.38 ms |
| 1280x800 | 6.77 ms | 8.18 ms |

Two of five scales look flat-to-slightly-better; the largest (1280x800, the
most representative of real deployment) is **21% worse**. The GPU stage
*itself* measured slower when run concurrently with the previous frame's tail
(`gpu_ms_median` 5.70 -> 7.84 ms at 1280x800) — the regression is not
overhead from spawning a thread per frame (measured separately at 0.087 ms
average spawn+join on this device, negligible against multi-millisecond
frames).

Ruled out:
- **DRAM controller governor.** `dmc_ondemand` was still active at 528 MHz of
  a 2112 MHz maximum (see the deployment note above) — pinning it to
  `performance` and re-measuring changed nothing material (serial 6.71 ms vs.
  pipelined 8.05 ms at 1280x800, essentially the same gap).
- **`QuadDecode`'s WorkerPool oversubscribing the 4xA76+4xA55 cores** while
  the main thread also needs CPU time to service the GPU driver's fence wait.
  Sweeping `APRILTAG_CPU_THREADS` from 1 to 8 found a shallow minimum at 6
  threads (8.03 ms) — still worse than serial's 6.77 ms at every thread
  count tested.

Working theory (not independently confirmed): this SoC has unified CPU/GPU
memory over a shared LPDDR bus, and/or a Vulkan driver whose
`vkWaitForFences` does not yield the CPU cheaply while blocked. Either way,
running CPU-heavy work concurrently with a GPU submission is not free the way
it would be on a discrete card with its own VRAM and an otherwise-idle CPU
during the wait — the same class of platform-specific result as the
tile-local union-find rejection below. `PipelinedDetector` and its
`--pipelined` validate-tool flag are kept in the tree (branch
`PipelineCpuTail`, not merged) since the mechanism itself is correct and
might pay off on different hardware or once the tail is small enough that
contention no longer dominates; do not enable it by default on this target.

**Tile-local union-find in shared memory.** The obvious answer to a stage
bound on dependent global loads is to move the pointer chasing into shared
memory: one workgroup per tile, resolve every component that fits inside the
tile locally, leave only the cross-tile seams to the global passes. For a
16x16 tile that is ~94% of all union operations. It is **much slower**, and
gets worse the bigger the tile:

| tile | threshold + label | tile pass alone |
| --- | --- | --- |
| none (flat, per pixel) | 3.58 ms | — |
| 4x4 | 4.02 ms | 1.29 ms |
| 8x8 | 4.62 ms | 2.09 ms |
| 16x16 | 5.32 ms | 3.00 ms |
| 32x32 | 6.63 ms | 4.36 ms |

The tiling did work as intended — at 16x16 it cut the global merge/compress
work from 2.70 ms to 1.46 ms — but the tile pass itself cost 3.00 ms to save
1.24 ms. The reason is architectural: Mali (Valhall) has no dedicated
shared-memory scratchpad the way NVIDIA and AMD parts do, it backs GLSL
`shared` with L2. A dependent load in shared memory is therefore not
meaningfully cheaper than one in global memory, and all that remains is the
extra pass and its barriers. **This is a Mali-specific result** — the same
change would plausibly win on a discrete GPU, so do not port it blind.

**Computing run starts directly instead of chaining then compressing.** Since
step 8 needs a compression pass anyway, `uf_init.comp` could scan backwards
through `thresholded` to find each run's first pixel and write a flat parent
in a single pass, saving a 2 MB read and a 2 MB write. The backward scan reads
known addresses, so unlike path compression it is not a dependent chain.
Measured 2.91 ms against 2.57 ms for chain-then-compress: the O(run length)
per-thread scan costs more than the pointer chase plus a separate flat pass.
Rejected.

**An equal-root early-out before `doUnion` in `uf_merge`.** `parent[]` is flat
on entry, so testing `parent[i] != parent[i + width]` costs the two loads
`find()` would have issued anyway and skips every union on the
convergence-check pass. Measured 2.550 ms against 2.569 ms — inside the noise,
and it encodes a fragile ordering invariant in the shader. Rejected.

**`sqrtf(a*a+b*b)` instead of `hypotf` in `FitLine`/`FitLineError`.** Worth
3.67 -> 3.38 ms on the CPU tail with identical output, but `hypotf` is
overflow-safe and the arguments are `float` casts of int64 covariance terms.
The margin before `a*a` overflows is only ~450x at realistic blob sizes, so
this trades 0.29 ms for an overflow cliff in a numerically delicate fit.
Rejected.

## Behavioural changes to be aware of

- **`MinMaxExtentsGpu::starting_offset` is now always 0.** There is no sorted
  array to offset into. `QuadDecode` never read it, but it is a public struct
  field.
- **Order within a blob is now atomic-arrival dependent.** This is the same
  class of nondeterminism the boundary-point append order in `blob_diff.comp`
  already had, and it is overwritten by the angular sort a dispatch later. It
  is more prevalent than before, so points with exactly equal `theta_key`
  could order differently between runs and shift the candidate quad count by
  +/-1 at the margins of peak detection — the same +/-1 the workgroup-size
  matrix already showed. The decoded tag set was unaffected in every
  configuration tested.
- **Validated on one image.** The hash table sizing (4x `max_raw_blobs`,
  128-probe cap) should be exercised against scenes with substantially more
  blobs before this is trusted in the field.

## Deployment note: the DRAM controller governor

Unrelated to the code, but it dominates measurement stability on RK3588:

```
$ cat /sys/class/devfreq/dmc/governor
dmc_ondemand
$ cat /sys/class/devfreq/dmc/cur_freq
528000000          # of a 2112000000 maximum
```

The CPU and GPU governors were already `performance`; the memory controller
was not. Pinning it did not move the median much, but it collapsed the spread:
worst-case frame went from 81 ms to ~20 ms. For a latency-sensitive workload
this is worth making persistent.

```
echo performance > /sys/class/devfreq/dmc/governor
```

## Remaining opportunities, largest first

Where the 16.1 ms now goes: CPU `quad_decode` 3.7, sort+group 3.01, linefit
3.05, threshold+label 2.59, boundary 1.73, readback 1.09, tag_decode 0.66,
upload 0.22.

1. ~~**Overlap the CPU tail with the next frame's GPU work.**~~ **Attempted and
   rejected — see "Measured and rejected" below.** It regresses throughput
   ~20-25% on this hardware. The claim below that this "hides essentially all
   of it — about 25% of end-to-end latency" was also wrong on its own terms
   even setting the regression aside: pipelining raises *throughput*, not
   per-frame *latency* (frame N's result is returned one frame later); the
   two are easy to conflate but are not the same claim.

2. **Further work on labelling — but not the obvious kinds.** The stage is
   now 2.59 ms, of which 0.86 ms is decimate + threshold and ~1.7 ms is the
   union-find. Two approaches are already measured and rejected above
   (shared-memory tiling, direct run-start scanning). Note also that the
   classic block-based schemes (BUF, Playne-Equivalence) do **not** apply
   here: they rely on all foreground pixels of a 2x2 block being connected,
   which holds for binary 8-connected labelling. This image is 4-connected and
   three-valued (0 / 255, with 127 merging with nothing), so a 2x2 block is
   not guaranteed to be a single component and the 4x node-count reduction is
   simply not available. What is left is reducing pass count: the second merge
   exists only to observe convergence, so a cheaper convergence proof would
   save most of a pass.

3. **`vkCmdDispatchIndirect` for `init_extents` / `select_blobs`** — 0.25 ms,
   measured. See "Measured and rejected" for why the readback version of this
   is a wash and indirect dispatch is not.

4. ~~**Pack `thresholded_buf_` and `decimated_buf_` to one byte per pixel.**~~
   **Done**, and the premise here was stale: it claimed "no 8-bit storage
   extension is available", but `VK_KHR_8bit_storage` is core as of Vulkan
   1.2 and the tree now carries `_u8` variants of every consumer, selected at
   runtime from `DeviceCaps::has_8bit_storage` with the 32-bit path as the
   fallback. `blob_diff` dropped off the consumer list entirely — see the
   GPU-agnostic pass below, which folds the threshold into `parent[]`.

5. ~~**`uf_final`'s blob-size histogram**~~ **Done** — see "A4a" in the
   GPU-agnostic pass below. The saturating form is exact, not approximate,
   because the count itself has no reader.

## Pipeline caching (startup latency, not per-frame)

Everything above is steady-state per-frame time. Separately,
`GpuDetector::CreatePipelines()` builds ~30 `VkPipeline`s (plus a
capacity-dependent number of scan-chain stages) once at construction, and
until now every one of those was a full SPIR-V -> ISA compile with
`vkCreatePipelineCache`'s cache argument hardcoded to `VK_NULL_HANDLE` - i.e.
no caching at all, on every process start.

`vk::Context` now owns a `vk::PipelineCache` (`library/src/vk/PipelineCache.
cpp`) that persists a `VkPipelineCache` to disk across runs and hands it to
every `vk::ComputePipeline`'s `vkCreateComputePipelines` call. The on-disk
file is keyed to `vendorID`/`deviceID`/`pipelineCacheUUID` (so a driver
update or a different GPU never gets fed stale data - the spec defines
`pipelineCacheUUID` to change exactly when compiled pipeline data would stop
being valid) plus an FNV-1a hash of the compiled `.spv` corpus (so a shader
rebuild during development doesn't feed the driver last week's binaries
either). `GpuDetector` flushes it right after `CreatePipelines()` rather than
relying solely on `~Context()`, since a camera-loop binary is more often
killed than shut down cleanly. Disable with `APRILTAG_VK_PIPELINE_CACHE=0`;
override the cache directory with `APRILTAG_VK_CACHE_DIR=<path>` (default:
`%LOCALAPPDATA%\vkapriltag` / `$XDG_CACHE_HOME/vkapriltag`).

Measured on the Windows desktop dev box (AMD Radeon RX 9060 XT, Vulkan
1.4.349) with a throwaway harness that just constructs `Context` +
`GpuDetector` and exits, timing `CreatePipelines()`:

| run                                             | `CreatePipelines()` |
|--------------------------------------------------|---------------------:|
| truly cold (no app cache, no prior driver cache)  |            240.7 ms |
| warm (app cache hit)                              |             16.6 ms |

A ~14x reduction on this GPU. Two things worth knowing before generalizing
that number:

* AMD's own driver keeps a persistent shader cache underneath ours. Once
  *anything* had compiled these shaders on this machine, even runs with
  `APRILTAG_VK_PIPELINE_CACHE=0` came back at ~18 ms - the driver-level cache
  alone was already doing most of the work here. The 240 ms number is only
  visible on the very first compile a machine ever does. This doesn't make
  the app-level cache redundant: it's the layer that's actually there on
  drivers with no such cache of their own (Mesa/Panfrost on the Orange Pi
  target above is the case that matters), and it's unaffected by whatever a
  given driver does or doesn't do underneath it.
* This targets pipeline *creation*, not first-dispatch latency. Some drivers
  defer final codegen to first use, so a residual first-frame cost can
  survive a warm pipeline cache. Not measured here; a follow-up would be a
  throwaway warm-up dispatch during construction, only if profiling on the
  actual Mali target shows it's still worth shaving.
* A warm run makes zero writes to the cache file (checked via mtime): saving
  is skipped whenever the retrieved cache data hashes the same as what was
  loaded, so steady-state use touches the filesystem only on the first run
  after a shader rebuild or driver update.
* Feeding the driver a corrupted or hand-edited cache file falls back
  cleanly to an empty cache and a fresh compile (verified by truncating a
  cache file to garbage bytes) - a bad cache can slow a run back down to
  the cold-path cost, but never breaks detection.

## Reproducing

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF
make -j8
APRILTAG_VK_MAX_POINTS=200000 \
  ./tools/apriltag_vulkan_validate --pgm colorImage.pgm --iterations 25
```

---

# A second pass: GPU-agnostic optimizations

Everything above was measured on the Mali-G610 deployment target (and
cross-checked on an RX 9060 XT). **This section was not.** It was measured on:

- **Intel Iris Plus G7** (i7-1065G7, integrated, unified memory, 8-bit
  storage present, subgroup variants disabled by the integrated-GPU exclusion)
- **NVIDIA MX230** (Pascal GP108, discrete, ~512 KB L2, subgroup variants
  enabled)

No figure in this section belongs in the Mali tables above, and none of it has
been run on Mali. The two devices between them do cover both memory
topologies and both sides of the subgroup switch, so every code path here is
exercised somewhere - but "unmeasured on the tuning target" applies to all of
it.

Test image `grayimage.pgm` (1280x800), default config. Per-span figures are
min over 12-15 runs of 6 iterations, because the value the validate tool
prints comes from a **single** instrumented frame and is far noisier than the
best-of-25 totals beside it.

## The bar, and what "bit-identical" means here

Every change in this pass had to show a measured win on at least one of the
two GPUs, and to leave output bit-identical. That second condition turned out
to be stronger than expected: ten repeat runs of the baseline were
byte-identical on every counter *and* on the corner RMS to six figures, so the
`theta_key` tie nondeterminism the "Behavioural changes" section warns about
does not manifest on this frame. Every step below therefore holds
`boundary_points`, `raw_blobs`, `selected_blobs`, `points`,
`candidate_quads`, `hash_probe_drops`, `uf_iterations`, the decoded ID set and
the corner RMS exactly constant, at decimations 1/2/4, with and without 8-bit
storage and subgroups, plus 5/5 on the generated corpus.

Two harness fixes were needed first: the OpenCV validate variant always
returned 0 (so the only variant supporting corpus mode could not gate
anything), and `kSpanLabelFinalize` bracketed `uf_final` and `label_pixels`
together - which nets out exactly the two changes that move them in opposite
directions. The span count also lived as four independent hardcoded literals
with the enum's own count sizing none of them; there is now one
`kNumGpuStages` and a `static_assert`.

## Summary

| Item | Change | Intel | MX230 |
| --- | --- | --- | --- |
| A5 | `uf_merge`: drop the per-pixel integer divide | labelling flat | labelling -1.0% |
| A6 | interleave the two hash-key arrays into one `uvec2` | boundary -13% | flat |
| A4a | `uf_final`: saturating counter, not a histogram | uf_final **-59%** | (scalar -55%) |
| A3 | fold the threshold value into `parent[]`'s spare bits | boundary **-33%** | boundary -4% |
| A8-lite | skip the provably no-op atomics in the extents reduction | extents -2.6% | (scalar -3.3%) |
| A2 | read the line-fit records in place where the memory type allows | readback_copy **-60%** | flat (fallback) |

`pipeline_total` best over the series: Intel **9.02 -> 8.03 ms**, MX230
**4.14 -> 4.23 ms**. The MX230 figure sits inside that device's own
run-to-run spread (its totals wandered 3.79-4.25 ms across steps with no
relation to what changed), so the honest reading is: a solid ~11% on the
integrated part, nothing measurable on the discrete one. That asymmetry is the
theme.

Two of these also **reduce** the shader corpus. A3 retired
`blob_diff_u8.comp` and `blob_diff_u8_subgroup.comp` - `blob_diff` was
parametrized on the 8-bit axis crossed with the ballot axis purely because it
read `thresholded` directly, and it no longer reads it at all - moving that
axis to `label_pixels`, where it costs two variants instead of doubling two
into four.

## The finding that generalizes: contention relief does not compose

Two independent items here (A4a, A8-lite) both helped the **scalar** shader
variant and **hurt** the subgroup-aggregated one, for the same reason. This is
the most transferable result in this pass.

Subgroup aggregation and a cheap early-out are alternative answers to the same
problem - too many contended atomics. Applying both is worse than either,
because aggregation has already collapsed the atomics to roughly one per
distinct key per subgroup, so the guard almost never fires and always costs a
test (and, for A4a, a dependent load of a location other subgroups are
concurrently updating). `uf_final` span on the MX230, min of 12:

| | ms |
| --- | --- |
| scalar, unconditional (was) | 0.1812 |
| scalar + saturating guard | **0.0809** |
| aggregated, unconditional (was) | 0.2478 |
| aggregated + saturating guard | 0.4024 |

So both guards apply to the scalar variants only, and the aggregated variants
keep their unconditional atomics with the numbers recorded inline. Since
integrated parts take the scalar path - Mali included - the wins land on the
path that matters for the deployment target.

Worth flagging because it contradicts a claim in the tree: **scalar +
saturating (0.0809) beats aggregated (0.2478) by 3x on the MX230, and
scalar-unconditional already beat aggregated there (0.1812)**. The comment in
`GpuDetector::CreatePipelines()` calling subgroup aggregation "a small but
real and repeatable win" on discrete GPUs was measured on an RX 9060 XT and
does not hold on this much smaller Pascal part. Retiring
`uf_final_subgroup.comp` is *not* done - that hardware is not available to
re-test, and this file is full of platform-specific inversions - but the case
for it is now on record.

## Measured and rejected

**A1: collapse the four queue submissions.** Not attempted, because it
already was. `bcfa3dc` merged the boundary+grouping submissions via
device-side indirect dispatch (4 -> 3, verified bit-identical), `ecaeaae`
reverted it, and `f436eb3` records why: removing a whole submission moved the
unspanned residual by **0.05 ms**, not the ~0.37 ms that dividing the residual
by submit count predicted. Going to a single submit would plausibly buy
0.1-0.15 ms for a lot of device-side-sizing machinery. The reverted commit is
on record if that judgement ever changes.

**A11: precompute the per-pixel gradient weight `W`.**
`sort_points_local`'s `ComputeLineFitPoint` takes four taps into the decimated
image per boundary point, and visits points in *angular* order around each
blob's perimeter, so they are scattered. `W` is a pure function of the image
at `(ix, iy)`, so it was hoisted into `threshold.comp` - which already sweeps
the same image coalesced - leaving one gather per point instead of four. That
also deleted `sort_points_local_u8.comp`, since the shader stopped reading the
decimated image at all. **Clear regression on both devices:**

| | threshold | sort | net |
| --- | --- | --- | --- |
| MX230 | 0.135 -> 0.175 (+29%) | 0.134 -> 0.130 (-3%) | **+0.036 ms** |
| Intel | 0.250 -> 0.338 (+35%) | 0.425 -> 0.417 (-2%) | **+0.080 ms** |

The flaw is the ratio of work: `sort_points_local` needs `W` at ~17000 point
locations, and the precompute produces it for all 256000 pixels - a 15x
overcompute, plus a 1 MB buffer to write and read back. And the four taps it
replaced were nearly free: removing three of four bought only 2-3%, because
the decimated image is 256 KB at this resolution and simply lives in L2.

**A7: back the decimated image with a tiled image and `texelFetch`.** Dropped
on A11's evidence rather than on speculation. A design pass established that
after A3 the *only* genuinely layout-sensitive consumer left is exactly this
gradient gather: `uf_merge`/`uf_init` are dispatched 1D over the linear index,
so a workgroup reads its own row and the row below as two perfectly coalesced
streams and the row stride costs nothing; `block_minmax`'s 4x4 windows are
disjoint, so global traffic is one read per pixel regardless of layout; and
`decimate`/`threshold` are pure raster sweeps that an optimally-tiled layout
would make *worse*. A11 then measured that one remaining gather to be worth
2-3%. A read-only tiled replica - which is the portable form, since `R8_UINT`
`STORAGE_IMAGE` is not mandatory but `SAMPLED_IMAGE | TRANSFER_DST` is, so it
would be filled by `vkCmdCopyBufferToImage` and only ever sampled - would add
a per-frame copy to chase less than that. Not worth a `vk::Image` abstraction
plus a descriptor-layer change.

**A8: pack `gx_sum` and `gy_sum` into one 32-bit `atomicAdd`.** The field
widths do not exist. `reduce_extents_hash` runs *before* `select_blobs`, so it
accumulates over raw blobs with no size filter - `max_cluster_pixels` (default
100000) is applied later and does not bound it - and a single raw blob can own
up to `qbp_capacity_` points (2,061,616 dense at 1080p/decimation 2). A biased
sum then needs ~19-22 bits per field, so two cannot share a word; 16/16 fails
too. Silent overflow in the border-polarity term would surface as an
occasionally undetected tag on an untested scene, in exchange for 1 of 8
atomics. A 64-bit `atomicAdd` has the width but needs
`VK_KHR_shader_atomic_int64`, which is not core. The zero-guard subset
(A8-lite) shipped instead.

**A9: drop the bitonic network's barriers for sub-subgroup stages.** Bounded
before building: an intentionally-incorrect build with **all** bitonic
`barrier()` calls removed measured `sort` at 0.1198 against 0.1321 on the
MX230, and 0.3989 against 0.4205 on Intel - a ceiling of 9% and 5% of a small
span, or 0.012/0.022 ms. A9 could capture at most the ~75% of stages with
`j < subgroupSize`, and would depend on local invocation indices mapping to
subgroup lanes contiguously - not guaranteed without
`VK_EXT_subgroup_size_control`'s full-subgroup guarantee, and Intel picks
SIMD8/16/32 per shader, so the threshold would have to be a runtime
`gl_SubgroupSize` branch. Not worth a portability assumption for 0.01-0.02 ms.

**A10: separable 3x3 min/max in `block_filter`.** Also bounded first:
reducing the 3x3 window to 1x1 (incorrect) put the *entire* cost of that
gather at 0.030 ms on Intel and 0.015 ms on the MX230. Separable saves 3 of 9
loads, so ~1/3 of that, while adding a dispatch, a barrier and an intermediate
buffer's write+read - and on Intel a trivial pass over 16000 elements costs
about as much as the 0.010 ms it would save. Cannot win.

## A note on method

Three of those rejections were settled by building a deliberately **incorrect**
shader to bound the payoff before writing the correct one: removing all the
barriers, shrinking a window to 1x1, forcing a guard to never fire. Each took
one build and a few minutes, and each killed a change that would otherwise
have taken an afternoon to write and then revert. Worth doing first whenever
the mechanism is "this access is expensive" - on this pipeline the answer was
repeatedly that it is not, because the working set at 1280x800 with 8-bit
storage fits in L2. Re-measure at 1080p and decimation 1 before trusting that
on a real target.

One trap worth naming: the obvious cheap A/B for A4a was to leave the guard
in and set the floor unreachably high so it never fires. That is **not** a
proxy for the original code - it measures load-plus-atomic where the original
was atomic-only, and on Intel it read 3.22 ms against the true baseline's
0.74 ms. The guard has to actually be compiled out.

## Where the time goes now

Single instrumented frame, `grayimage.pgm` at 1280x800, default config:

| Stage | Intel Iris Plus | MX230 |
| --- | --- | --- |
| clear | 0.418 | 0.039 |
| threshold + decimate | 0.265 | 0.136 |
| labelling | 1.355 | 1.208 |
| uf_final | 0.344 | 0.251 |
| label_pixels | 0.075 | 0.082 |
| boundary | 0.211 | 0.201 |
| hash_group | 0.189 | 0.056 |
| extents | 2.027 | 0.284 |
| select | 0.020 | 0.005 |
| blob_scan | 0.086 | 0.023 |
| scatter | 0.237 | 0.053 |
| sort + line-fit | 0.525 | 0.131 |
| readback_copy | 0.034 | 0.098 |
| **sum of spans** | **5.787** | **2.567** |
| GPU total (best/median of 25) | 7.47 / 8.25 | 3.79 / 4.06 |
| `quad_decode` (CPU) | 0.37 / 0.47 | 0.29 / 0.40 |
| `tag_decode` (CPU) | 0.10 / 0.17 | 0.09 / 0.14 |
| **pipeline_total** | **8.03 / 9.04** | **4.23 / 4.58** |

The obvious next target on the Intel part is **`extents` at 2.03 ms** - 35% of
its whole frame, and 7x what the same stage costs on the MX230. An outlier
that large is more likely a driver or access-pattern pathology specific to
that part than anything about the algorithm, and it is also the noisiest span
there (min 1.80-1.90 across repeats, excursions past 4 ms). Nothing in this
pass explains it; it was not chased because the deployment target is not an
Intel iGPU.

---

# A third pass: measured on the deployment target

Measured first on a **Windows desktop, AMD Radeon RX 9060 XT (RDNA4,
discrete, Vulkan 1.4.349)**, then - unlike the first two passes - **on the
Orange Pi 5 Plus / Mali-G610 deployment target itself** (Armbian 26.5.2,
vendor kernel 6.1.115, libmali, GPU pinned 1 GHz). `grayimage.pgm` at
1280x800.

Having the target available overturned the pass's original premise and one
of its shipped changes, and produced a result that matters more than either:
**a direct measurement of how much of the GPU phase is DRAM bandwidth at
all** (item 0).
Both prior passes reasoned about Mali memory traffic from an input-size
sweep; that sweep shows time scaling with pixels, which is not the same claim
and, it turns out, is not mostly bandwidth.

Baseline: GPU total 1.38 ms at decimation 1, 0.89 ms at decimation 2.

## 0. How much of the GPU phase is DRAM bandwidth? 15%, not 86%

Pinning the memory controller with the `userspace` devfreq governor and
sweeping it, interleaved (decimation 2, GPU pinned at 1 GHz):

| DMC | GPU total | `clear` | `label_pixels` | `uf_final` | `labelling` | `boundary` | `extents` | `sort` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 528 MHz | 4.980 | 0.177 | 0.361 | 0.242 | 1.198 | 0.394 | 0.436 | 0.392 |
| 1068 MHz | 3.924 | 0.099 | 0.195 | 0.129 | 0.979 | 0.372 | 0.433 | 0.375 |
| 1560 MHz | 3.574 | 0.062 | 0.143 | 0.099 | 0.938 | 0.364 | 0.428 | 0.434 |
| 2112 MHz | 3.457 | 0.048 | 0.116 | 0.091 | 0.921 | 0.361 | 0.432 | 0.397 |
| **528/2112** | **1.44x** | **3.68x** | **3.12x** | **2.66x** | 1.30x | 1.09x | **1.01x** | **0.99x** |

`t = C + K/f` fits the endpoints with `C = 2.95 ms` and `K/f = 0.51 ms` at
2112 MHz, and predicts the two interior points to within 2%. So **~15% of GPU
time is bandwidth-proportional**; removing *all* traffic would buy 0.51 ms of
3.46. Halving it buys ~7%.

The per-span ratios are the more useful half of this. `extents` (1.01x) and
`sort` (0.99x) - the second and third largest spans - do not move at all
across a 4x bandwidth range: they are atomic- and latency-bound, and traffic
work aimed at them cannot pay. `labelling`, the largest span, is 1.30x, i.e.
mostly dependent-load latency, which is the same conclusion item 8 of the
first pass reached by ablation. What *is* bandwidth-bound is small: `clear`
(3.68x, and nearly pure `vkCmdFillBuffer` traffic), `label_pixels` (3.12x)
and `uf_final` (2.66x), together 0.26 ms.

PERFORMANCE.md section 3's "86% of it is pixel-proportional" is not wrong,
but it is an input-size scaling, and it has been read as a bandwidth claim
(including by this file's own "remaining opportunities"). Those are different
things: more pixels means more *invocations*, more dependent loads and more
atomics, not only more bytes.

**Governor caution.** The board idles at `dmc_ondemand`/528 MHz, which makes
pinning look like a 32% win. It is not - `dmc_ondemand` ramps to 2112 MHz
under sustained load, and head to head the steady state is identical (3.152
vs 3.156 ms). Pinning buys the variance reduction described further up this
file, not throughput.

## The bar, and how these were measured

Same bar as the second pass — bit-identical output, plus a measured win — with
the A/B tightened after the first attempt produced a false signal. Measuring
before/after in two separate sessions showed `label_pixels` moving 37% on a
shader neither change touches. Rebuilding both binaries, keeping both, and
interleaving them **ABBA within one session** collapsed that to -0.1%. Every
number below comes from the interleaved harness, reported as a min over 8-20
rounds. Where a span still disagreed with itself between sessions it is
reported as noise rather than as a result, and only the sign that held
everywhere is quoted.

Bit-identity was checked on the full matrix each time: decimations 1/2/4, the
5-image corpus, `APRILTAG_VK_FORCE_NO_8BIT` on and off, and workgroup
geometries 8x8 / 16x16 / 32x8 / 2x2 / 6x6 (the last two exercising the
unfused fallback) - every counter, the decoded ID set and the corner RMS
constant throughout.

## 1. RawLineFitPoint packed into two words - shipped, on

`RawLineFitPoint` is the largest per-frame readback in the pipeline, and it
spent a full 32-bit word on each of `x2`, `y2`, `W`, `blob_index`. Each has a
bound that something else already enforces (see PERFORMANCE.md section 6b for
the table), so all four fit in two words. The packing is exact - the host
accessors return the same integers - so this is a storage change, not a
precision one.

| | decimation 1 | decimation 2 |
| --- | --- | --- |
| readback bytes/frame | 836848 -> **425760** | 286768 -> **145568** |
| device memory | 185 -> **154 MiB** | 48 -> **41 MiB** |
| `readback_copy` span | -84% / -89% | -20% |
| GPU total (RX 9060 XT) | -3.3% / -8.6% | -5.3% / -10.4% |
| GPU total (**Mali-G610**) | — | **+1.1% / -0.7% / +0.7%, i.e. nil** |
| `pipeline_total` (**Mali-G610**) | — | **~-1%** |

The byte and memory figures are exact. **The timings are not**: across four
interleaved sessions GPU total landed anywhere from -3.3% to -10.4%, and at
decimation 2 the `sort` and `labelling` spans wandered 13-33% in *both*
directions - `labelling` on a shader this change cannot affect. So what is
claimed here is the sign, which held everywhere on that machine, and the
exact byte and memory reductions.

**On Mali it buys no GPU time at all**, and item 0 says why: the line-fit
readback there is a direct host-cached read with no PCIe hop to shorten
(`readback_copy` = 0.006 ms), and `sort`, where the halved writes land,
measures a bandwidth sensitivity of 0.99x - none. The discrete-card speedup
was a PCIe effect. It stays on for the 15% memory footprint, which is what
actually matters on a unified-memory part, and `pipeline_total` is
consistently ~1% better there from the CPU tail reading half the bytes.

**fp16 was the obvious-looking alternative and is wrong.** Its 11-bit
mantissa makes integers above 2048 round to even; `x2` reaches 3839 at 1080p
/ decimation 1, i.e. a half-pixel coordinate error on the right of the frame,
landing directly in corner positions. And fp16 *moments* are further out
still - the CPU rebuilds `Mxx/Mxy/Myy` in native `int64` exactly because the
covariance is a near-total cancellation. Integer packing has no cliff at all.

## 2. uf_compress skips its read pass once converged - shipped, on

`uf_compress.comp` already guarded its store. It still streamed all of
`parent[]` to find out there was nothing to store - 1 MB per pass at
decimation 2, on the commonest case in steady-state video. It now reads the
convergence flag first and returns if the preceding merge joined nothing,
which is exact rather than approximate: every chunk ends with a compression,
so `parent[]` is flat entering the merge, and a merge that changes nothing
leaves it flat. A push constant keeps the flag *un*honoured for the
compression right after `uf_init`, which must always run - that is the pass
item 8 above measured at 1.0 ms.

Mali-G610, ABBA, min of 24, three sessions: `labelling` **-2.1% / -2.2% /
-2.4%**, GPU total **-0.5% / -0.7% / -0.9%**. Bit-identical at decimations
1/2/4 on both machines.

The ceiling, from removing the dispatch outright, is `labelling` -4.4% / GPU
total -1.4%. The guard gets about half: the invocations still launch and each
pays one scalar load. Recovering the rest needs `vkCmdDispatchIndirect` with
a device-computed group count of zero, costing a dispatch and a barrier -
2.6-18.7 us on Mali against ~30 us remaining. Not attempted; the sign is not
obvious.

Note what this is and is not: a *launch and latency* saving that also happens
to remove traffic, not evidence for the bandwidth thesis item 0 demolishes.

## 3. The extents reduction: 170-way atomic contention - shipped, on

Item 0's per-span table pointed here by pointing *away* from everything else.
`extents` was the second largest span and moved **1.01x** across a 4x
memory-clock range, i.e. not bandwidth at all. `reduce_extents_hash.comp`
issues up to eight atomics per boundary point into that point's blob
accumulator - ~65k points over 388 blobs at decimation 2, so ~170 points
contending per counter. Bound, by replacing the atomics with plain stores:
span 0.434 -> 0.062 ms.

Three changes, all bit-identical, in the order they were measured:

1. **Read before the atomic on min/max.** Exactly equivalent (an `atomicMin`
   with a value already >= the stored one is a no-op) and a blob's extents
   stop moving after a handful of points. **-28% of the span.**
2. **8-way privatization + `merge_extents.comp`.** Only the first 4096 blob
   indices are replicated, so the cost is ~1 MiB, not 8x3 MB; past that,
   points fall back to the shared slot, correct but contended. Sweep on
   Mali (GPU total vs unprivatized): K=2 -13.1%, K=4 -18.5%, **K=8 -20.3%**,
   K=16 -20.0%, K=32 -20.6%; K=8 and K=16 are within noise head to head, so
   K=8 wins on memory.
3. **`VK_KHR_shader_atomic_int64`** - see item 4.

A trap worth recording. The first two bounds were run *without* the merge
pass, so their results were wrong, and wrong extents meant `select_blobs`
kept different blobs and every downstream span did different work. They
reported GPU total -20.3% where the correct implementation measures -5.4%.
**A bound whose output is wrong can flatter itself through the rest of the
pipeline**, which is a different failure from the usual "incorrect build"
technique this file recommends - there the incorrectness was confined to the
span being measured. Check that a bound's downstream counters still match
before believing its total.

## 4. Optional extensions: a survey, and the one that paid

The G610 under libmali exposes 130 device extensions. Checked against what
items 0 and 4 say the pipeline is actually bound by:

| Extension | Verdict |
| --- | --- |
| **`VK_KHR_shader_atomic_int64`** | **Adopted.** `count` and `pxgx_plus_pygy_sum` are the only two fields every point touches; `MinMaxExtentsGpu` now places them in one aligned 64-bit word so one `atomicAdd` does both. Exact - the low half cannot carry into the high half, being bounded by the point capacity. `extents` **0.221 ms with, 0.252 without (-12 to -13%)**, ~1% of frame. Gated on the extension + `shaderBufferInt64Atomics` + core `shaderInt64` (the G610 has `shaderInt64` but not `shaderFloat64`), with `reduce_extents_hash.comp` as the unconditional fallback and `APRILTAG_VK_FORCE_NO_INT64_ATOMIC` to A/B it. |
| `VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8` | Not pursued. These buy traffic, and item 0 caps *all* traffic work at 15% of GPU time; the specific buffer they would shrink is the line-fit record, whose halving (item 1) measured nil on this part. fp16 would also be lossy - see item 1. |
| `VK_EXT_subgroup_size_control` | Useless here: `minSubgroupSize == maxSubgroupSize == 16`, so there is nothing to control. |
| `VK_EXT_shader_subgroup_ballot` / `vote`, `VK_KHR_shader_subgroup_extended_types` | Already ruled out by measurement, and hard: subgroup variants are a 57% regression on this part (see PERFORMANCE.md section 6). |
| `VK_KHR_synchronization2` | Not pursued. Finer barrier scopes would target the intra-submit gaps, measured at **0.067 ms total** for a whole frame. The gaps that are actually large are at *submit* boundaries, which is a CPU round-trip problem, not a barrier-scope one. |
| `VK_KHR_timeline_semaphore`, `VK_EXT_host_query_reset` | Structural/diagnostic convenience, no per-frame GPU time. |
| `VK_EXT_scalar_block_layout`, `VK_KHR_relaxed_block_layout` | Tighter struct packing, i.e. traffic again - capped by item 0. |
| `VK_KHR_buffer_device_address`, `VK_KHR_push_descriptor` | Descriptor/host-side overhead, not the bottleneck; the host-side submission cost is already measured small. |
| `VK_ARM_scheduling_controls`, `VK_ARM_shader_core_builtins` | Vendor-specific and would break the "core Vulkan 1.1, zero optional features" posture for no identified bottleneck. Not tried. |

The honest summary: **one extension out of 130 addressed a real bottleneck**,
and it is worth about 1% of the frame. The other 8% of this pass came from an
algorithmic change (contention) that needed no extension at all. That ordering
- measure what you are bound by, then look for a tool - is what item 0 bought.

## 5. The frame's tail is one submission - shipped, on where the memory allows

Item 0 measured spans. The gaps between them turned out to matter as much:
between the timestamp ending one submission and the one starting the next,
the GPU is idle waiting for the host, and those gaps totalled **0.44-0.61 ms
per frame** on Mali - larger than every span but labelling.

Bounded first, and the bound is worth describing because it avoids the trap
item 4 fell into. Instead of breaking correctness to measure the ceiling, it
reuses the PREVIOUS frame's counts and records everything as one submission:
for a repeated still image the counts are identical, so the build is exactly
correct and its counters can be checked against the four-submit path. It
reported **GPU total -7.9 to -10.1%** with timestamps off (worth checking
separately - the timestamp instrumentation itself costs ~9% here, 3.16 ms
against 2.90 ms, so profiling numbers must not be quoted as deployment
numbers).

Two of the three readbacks are now gone:

* `build_indirect_args.comp` writes three `VkDispatchIndirectCommand`s from
  device-side counters instead of one, and `hash_group`,
  `reduce_extents_hash`, `scatter_index_points` and `sort_points_local` take
  their bound from a buffer instead of a push constant, so the tail is
  recorded in the same submission that computes the counts it needs.
* `selected_extents_buf_` is read in place, like the line-fit buffer already
  was. Not for the copy's size - it is a few KB - but because a staged copy
  needs `num_selected_blobs` ON THE HOST at record time, which was the last
  thing keeping the readback round trip alive.

Measured against item 4's state, ABBA, min of 12-16: decimation 1 **-2.0%**,
decimation 2 **-4.0/-4.4%**, decimation 4 **-7.9%** GPU total. The gain grows
as the frame shrinks, because what was removed is per-submit and roughly
fixed.

Requires both readbacks to be host-visible AND host-cached, which unified
memory gives and a discrete card without resizable BAR does not - the
RX 9060 XT keeps the four-submit path, so the fallback stays exercised.

Two things this pass got wrong on the way, both worth knowing:

* Push-constant field order. `count_from_buffer` was appended in the shader
  after `count` but in the host struct at the end; the two silently
  disagreed and the frame produced 64856 hash drops and zero blobs. GLSL
  push-constant blocks are positional, and nothing checks them.
* `kGpuGapCrossesSubmit` was a static table describing the four-submit
  layout, so after fusing it still labelled two intra-submit barriers as
  submit boundaries - attributing ~0.2 ms to submissions that no longer
  happened. It is now per-frame state (`DetectProfile::gap_crosses_submit`).
  A diagnostic that hardcodes the structure it measures will lie the moment
  the structure changes.

**What is left**: the boundary before `uf_final`, worth **0.28 ms (~9%)**,
held open by the host readback of the labelling convergence flag. The exact
way to close it is a retry rather than a guess: record the frame with the
chunk count the previous frame needed, read the flag with the other counters
at the end, and redo the frame if it says the labelling did not converge.
`clear` re-initializes everything, so a redo is simply correct, and
`last_uf_iterations_` already adapts after one frame. Not attempted.

## 6. Subgroup aggregation, per site - two of three retired

The subgroup-aggregated variants were gated by one flag covering three
sites. Measuring the three separately on an RX 9060 XT (min of 12, three
sessions, each reproducing within 1%) says the shared gate was wrong at two:

| site | aggregated | plain atomics | |
| --- | --- | --- | --- |
| `uf_final` | 0.1035 ms | **0.0291 ms** | scalar **-72%** |
| `blob_diff` | 0.0437 ms | **0.0330 ms** | scalar **-24%** |
| `reduce_extents_hash` | **0.0702 ms** | 0.1028 ms | subgroup **-45%** |

`uf_final_subgroup.comp` and `blob_diff_body.glsl`'s
`AGGREGATE_APPEND_COUNTER` path are deleted. Worth **GPU total -5.3%** on
that card; nothing on Mali, which never took them.

For `uf_final` this closes a question this file already left open: the second
pass measured scalar+guard beating aggregated 3x on an MX230 and said "the
case for retiring it is now on record" but that the hardware to confirm on a
bigger discrete part was not available. It is now, and it agrees.

**The transferable rule: aggregate values by key, not bare counters.** What
separates the survivor is that `reduce_extents_hash` reduces per-point
*values* across lanes sharing a key, collapsing eight atomics per point into
eight per distinct key per subgroup. The two retired ones aggregated a
*counter* - one `atomicAdd` per lane becoming one per subgroup - which is the
contention a modern discrete part's atomic unit already handles, so the
ballot sequence bought nothing and cost its own issue slots.

`blob_diff` is retired on one device's evidence rather than two. It is
recoverable from this branch's history if a part ever makes the case.

## A scan of the literature, and what it does and does not offer here

Done at the end of this pass, against the two spans that dominate what is
left (`labelling` ~29% on Mali, `sort` ~15%).

**HA4 / FLSL (Hennequin & Lacassagne).** The genuinely relevant find, and the
one the earlier BUF/BKE dismissal does not cover. This file correctly rules
out BUF and Block-based Komura Equivalence because they need every foreground
pixel of a 2x2 block to be connected, which holds for 8-connected binary
labelling and not for this pipeline's 4-connected three-valued input. **HA4
is the 4-connected one**: a hybrid pixel/segment (run-length) algorithm that
splits the image into horizontal strips, gives each strip to one warp, and
merges using only each run's start pixel as a proxy. FLSL, a GPU port of the
LSL SIMD algorithm, then improves on HA4 by reducing memory-access conflicts
on many-core parts.

Two things to weigh before anyone starts. First, `uf_init.comp` already
pre-joins horizontal runs (item 8 of the first pass), so the pipeline has
taken the cheapest part of this idea already. Second, and more seriously,
HA4's efficiency comes from warp intrinsics - and this tree now has three
independent measurements saying those are the wrong tool on the deployment
target: `reduce_extents_hash_subgroup` at 6.6x slower on Mali, and the two
retirements above on a discrete part. HA4 would have to earn its keep through
the run-based structure alone, with the intrinsics replaced by shared memory
that Valhall backs with L2. That is not a reason not to try it; it is a
reason to bound it before writing it.

**Not relevant.** NVIDIA VPI and Isaac ROS AprilTag are CUDA-only and closed,
so they inform nothing portable. The learned detectors (YoloTag, DeepTag,
E2ETag) replace the detector wholesale and give up bit-compatibility with
libapriltag, which is this project's entire correctness gate.

## Measured and rejected (third pass)

**Fusing the preprocessing dispatches.** `decimate` + `block_minmax` and
`block_filter` + `threshold` each collapse into one pass, removing a
full-image read and a whole block-grid round trip respectively and deleting
`minmax_filtered_buf_`. Both were written, verified bit-identical across
decimations 1/2/4, both storage widths and five workgroup geometries - and
both **lose on both devices**.

Mali-G610, GPU total, min of 8 ABBA rounds: minmax fusion **+5.9%**,
threshold fusion **+5.4%**, both **+7.8%**. RX 9060 XT, `threshold` span at
decimation 1: **+8.1% / +4.1% / +12.7%**. The additivity on both parts is
what makes it a result rather than noise.

The expectation was that Mali would invert the discrete result, being the
bandwidth-bound part. Item 0 explains why that was wrong: the fusions target
`threshold`, whose bandwidth sensitivity is 1.34x and which is 4% of the Mali
frame, and they pay for the DRAM traffic they remove with shared-memory
traffic and barriers - on an architecture where `shared` is backed by L2 and
is not cheaper than global, exactly as the tile-local union-find rejection
above found.

They were briefly kept behind an `APRILTAG_VK_FUSE_PREPROCESS` env knob, on
the theory that a future part with a real scratchpad might invert it. That
was the wrong call for a tree with this much dead-path surface already: two
measured-negative results do not earn four shader variants, a `_body.glsl`
pair, a second buffer-allocation path and a branch in the dispatch sequence.
The shaders are deleted; this entry is the record. Recover them from
`perf/extents-contention-and-int64-atomics` history if a device ever makes
the case.


**Right-sizing `extract_blob_counts` with `vkCmdDispatchIndirect`.** The last
dispatch still sized by a capacity rather than a device-side count, and so it
looks like the remaining instance of the `init_extents` / `select_blobs` fix
recorded above. Bounded before building, per the method note below: shrinking
the capacity 8x with `--max-blobs 512` moved the `blob_scan` span 0.01044 ->
0.00632 ms. That ~4 us is the entire capacity-proportional cost of the
dispatch **and its whole scan chain**; right-sizing only the dispatch
recovers a fraction of it, against a 1.38 ms frame. It also agrees with the
figure already sitting in the constructor's `max_blobs` comment (+2.2 us for
a 24x capacity increase). And the slack is load-bearing: it exists so a
scene with unusually many blobs does not overflow `max_blobs`, which would
make detections depend on GPU scheduling order. Buying ~2 us costs an
indirect-args slot, a per-frame buffer fill to zero the tail, and the
invariant that the scan's grand total sits at
`blob_point_offsets[max_blobs - 1]`. Rejected.
