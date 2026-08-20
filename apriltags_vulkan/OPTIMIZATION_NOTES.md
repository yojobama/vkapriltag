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

4. **Pack `thresholded_buf_` and `decimated_buf_` to one byte per pixel.**
   Both store a `uint` per pixel today: `thresholded` holds only 0/127/255,
   and `decimated` holds a `uint8` grayscale value. `thresholded` is streamed
   by `uf_merge` (twice) and `blob_diff`, `decimated` by `threshold` and
   `compute_line_fit_points`. Roughly 6 MB per frame, so ~0.4 ms — but it
   costs pack/unpack ALU in four shaders and no 8-bit storage extension is
   available, so it has to be done by hand. Estimated, not measured.

5. **`uf_final`'s blob-size histogram** is 518400 random `atomicAdd`s with
   heavy contention on large blobs. Only the `>= min_cluster_pixels`
   predicate is ever consumed, so a saturating or hierarchical count would do.
   Part of the 1.71 ms boundary stage; not separately measured.

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
