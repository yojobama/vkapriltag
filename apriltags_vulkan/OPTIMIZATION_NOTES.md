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
| 7 | per-pixel labels remove `blob_diff`'s random gathers | **12.51 ms** |

Per-stage, baseline versus now:

| Stage | Before | After |
| --- | --- | --- |
| upload | 0.22 | 0.22 |
| threshold + label | 3.58 | 3.55 |
| **boundary** | **3.24** | **1.71** |
| **sort + group** | **15.83** | **2.98** |
| **linefit** | **13.41** | **3.07** |
| readback | 3.25 | 1.09 |
| GPU total | 39.48 | 12.51 |
| CPU `quad_decode` | 4.15 | 3.56 |
| **pipeline total** | **44.3** | **16.8** |

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

Where the 16.8 ms now goes: threshold+label 3.55, CPU `quad_decode` 3.56,
linefit 3.07, sort+group 2.98, boundary 1.71, readback 1.09, tag_decode 0.66,
upload 0.22.

1. **Overlap the CPU tail with the next frame's GPU work.** `quad_decode` +
   `tag_decode` is 4.2 ms of the 16.8, and it runs strictly after the GPU
   finishes. Double-buffering the detector so frame N's CPU tail runs during
   frame N+1's GPU pipeline hides essentially all of it — about 25% of
   end-to-end latency for no algorithmic change. This is an application-level
   change (`main.cpp` and the detector's buffer set), not a shader one, and is
   almost certainly the best remaining ratio of win to risk.

2. **Block-based connected-component labelling.** Ablation (`chunk=0` versus
   `chunk=2`) puts decimate + threshold at 0.86 ms and the union-find itself
   at 2.7 ms, i.e. ~1.35 ms per merge/compress iteration over 518400 pixels.
   Both passes are random-access bound on the 2 MB `parent` array. A
   block-based scheme (BUF / Playne-Equivalence, union-find over 2x2 blocks)
   has 4x fewer nodes and is the standard 3-4x win here — call it 2 ms. It is
   also the riskiest change on this list: it is a rewrite of the one stage
   whose correctness everything downstream depends on.

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

## Reproducing

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF
make -j8
APRILTAG_VK_MAX_POINTS=200000 \
  ./tools/apriltag_vulkan_validate --pgm colorImage.pgm --iterations 25
```
