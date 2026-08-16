# Optimization notes: grouping, sorting and line fitting

Measured end to end on the intended deployment target — an Orange Pi 5
(RK3588, Mali-G610, 8 GB LPDDR), Ubuntu 6.1.0-1025-rockchip, Arm proprietary
Vulkan driver `g6p0-01eac0`, Vulkan 1.2.165 — running the tracked
`colorImage.pgm` at 1920x1080 through `apriltag_vulkan_validate`.

Every number below is best-of-25 with `APRILTAG_VK_MAX_POINTS=200000`, and
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
| 5 | 256-thread local sort + CPU tail cleanup | **14.31 ms** |

Per-stage, baseline versus now:

| Stage | Before | After |
| --- | --- | --- |
| upload | 0.22 | 0.22 |
| threshold + label | 3.58 | 3.61 |
| boundary | 3.24 | 3.28 |
| **sort + group** | **15.83** | **3.06** |
| **linefit** | **13.41** | **3.18** |
| readback | 3.25 | 1.09 |
| GPU total | 39.48 | 14.31 |
| CPU `quad_decode` | 4.15 | 3.58 |
| **pipeline total** | **44.3** | **18.7** |

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

This is a four-line change and was the cheapest win in the set.

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

The scan over the hash table reuses `root_scan_chain_`'s block-sum buffers.
That chain was built for `pixels` elements, which is strictly larger than the
table, and the two scans run in different submissions so they never overlap.

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

---

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

## Now-unreachable code

The boundary-point sort is gone, so the radix pipelines (`radix_histogram`,
`radix_scan_hist`, `radix_scatter`), `bitonic_sort`, `bitonic_local`,
`fill_max_key`, `gather_generic`, `mark_heads`, `rewrite_index_points`, the
qbp scan chain and `DetectorConfig::sort_algorithm` (and its
`APRILTAG_VK_SORT` override) are all unreachable. They are left in place here
so this change stays reviewable; removing them reclaims roughly 10 MB of
device memory — `qbp_sorted_buf_` alone is 6.4 MB at 200k points — plus the
pipeline creation cost at startup.

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

## Remaining opportunities (not done here)

1. **`init_extents` and `select_blobs` still dispatch over `max_raw_blobs`**
   (65536) x 48-byte structs — about 6 MB of traffic for 734 real blobs, and
   probably most of the remaining 3.06 ms in sort+group. Wants
   `vkCmdDispatchIndirect` driven by a device-side count.
2. **Four `SubmitAndWait` round trips per frame.** Indirect dispatch would
   also remove the `qbp_count` and `num_selected` host readbacks, collapsing
   the frame to one or two submissions.
3. **`decimated_buf_` stores one `uint` per pixel** — 4x the bytes needed, and
   it is read by both `threshold.comp` and `compute_line_fit_points.comp`.
4. **Delete the unreachable sort machinery** listed above.

## Reproducing

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF
make -j8
APRILTAG_VK_MAX_POINTS=200000 \
  ./tools/apriltag_vulkan_validate --pgm colorImage.pgm --iterations 25
```
