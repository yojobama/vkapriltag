# 01 - GPU compute stages, device-agnostic improvements

Status: complete

Scope: the GPU stages in `apriltags_vulkan/library/shaders` and their orchestration in
`library/src/gpu/GpuDetector.cpp`, stage by stage. Ranked by expected gain on the Orange Pi first.
Nothing here was built or timed (brief rule); every gain is an estimate and says so.

## Top conclusions

Ranked by expected gain on the Orange Pi (Mali-G610), weighted by confidence. All are exact
(bit-identical, or identical up to append order that is already nondeterministic and has zero
genuine ties on the corpus). All are estimates; none was built or timed.

1. **`blob_diff`'s append counter** (finding 2): one global `atomicAdd` per point (~65k per frame
   at d2) on a span that is 1.09x DMC-insensitive. Reserve a pixel's points with one atomic
   (~10 lines); partition K ways only if a gap remains. Up to -4 to -5% if the counter is the cost.
   **Measure first** with the no-atomic bound build; cheapest high-ceiling item in the file.
2. **Delete `label_pixels`** (finding 1): `blob_diff` tests the 1-byte threshold codes first and
   gathers `parent`/`blob_size` only at black/white transitions (~13% of pixels). Removes the most
   bandwidth-sensitive real pass and the destructive `parent[]` rewrite that caused a DEVICE_LOST.
   Est. -2 to -3%. **Pursue.**
3. **Barrier-free rank sort for small blobs** (finding 6): sort keys are unique, so any sort is
   bit-identical; rank-by-counting needs no shared memory and no barriers, the two costs the tree
   measured on Mali. -2 to -5% if most blobs are small. **Measure first**: the blob-size histogram
   is already readable on the host from `selected[].count`.
4. **Drop the in-chunk `uf_compress` passes; `uf_final` flattens** (finding 3). Anchored on the
   measured 1.4% ceiling for one no-op compress. Est. -1.5 to -2%. **Pursue.**
5. **Run-merge before the atomics in `reduce_extents_hash` and `scatter`** (finding 4): cut the
   unconditional 64-bit atomics on an atomic-bound span. Est. -2 to -3%. **Measure first**
   (run-length statistics), after item 1.
6. **Delete `decimate`, fold `uf_init` into `threshold`** by recomputation, not shared memory
   (finding 5). Est. -1 to -2%. Measure first (exact still-image bound exists).
7. **Dispatch diet** (finding 7): fold `init_extents` into the `hash_group` winner and
   `merge_extents` into `select_blobs`, plus research/05's atomic range allocator. ~-1%. Pursue as
   a bundle.
8. **Fold `hash_group` into `blob_diff`** via 64-bit key-in-slot CAS (finding 8). ~-1 to -2%,
   adds a variant axis. Only if the Mali `hash_group` span is >= ~0.04 ms.

Not additive in full (1, 2 and 8 all touch `blob_diff`), but if the bounds confirm, items 1-6
together are on the order of -10% of Mali GPU time. Before any of it: record one `--profile`
span table on the Pi; the tree has none for the current code, and `hash_group` / `scatter` /
`blob_scan` have never been recorded on Mali. RDNA gains are small throughout (its spans are
microseconds); items 2 and 6 are the most likely to show there.

## Working numbers (Mali-G610, 1280x800, decimation 2)

There is no current per-span Mali table in the tree. The newest is PERFORMANCE.md 3a (third pass,
GPU pinned 1 GHz, DMC 2112 MHz, timestamps on, GPU total 3.457 ms): `labelling` 0.921, `extents`
0.432, `sort` 0.397, `boundary` 0.361, `label_pixels` 0.116, `uf_final` 0.091, `clear` 0.048.
Since then: `extents` 0.221 (int64 atomics, PERFORMANCE.md 3b), `uf_final` -12% and `label_pixels`
-18% (127 skip, OPTIMIZATION_NOTES item 12), `labelling` -3% (path splitting, item 10). Today's
`Detect()` is ~2.70 ms with timestamps off (PERFORMANCE.md 8g), of which ~0.13 ms is host overhead.
So the shares used below are approximate: labelling ~30%, sort ~13%, boundary ~12%, extents ~7%,
threshold ~5%, label_pixels ~3%, uf_final ~3%, the rest (hash_group, scatter, select, blob scan,
clear, gaps) ~25%. **First action for whoever measures:** run `--profile` on the Pi once and
replace this table; several rankings below depend on the hash_group / scatter split, which has
never been recorded on Mali.

Work counters for scale: decimation 1 `boundary_points=189634, raw_blobs=1257, blobs=458,
points=51386` (vkapriltag-optimization-ideas.md). Decimation 2: ~65k boundary points, 388 raw blobs
(PERFORMANCE.md 3b), ~17k selected points (OPTIMIZATION_NOTES A11). So **about 73% of boundary
points are discarded at `select_blobs`**, after `hash_group`, `reduce_extents_hash` and
`scatter_index_points` have each processed them.

## Findings

### 1. Delete `label_pixels`: test codes first in `blob_diff`, gather labels only at transitions

- **Idea**: `blob_diff` reads the 1-byte `thresholded[]` for its 6 pixels, exits unless some
  E/SE/S/SW pair straddles black/white, and only then reads `parent[]` (flat root) and
  `blob_size[root]` for the pixels it needs. `label_pixels` and its in-place rewrite of `parent[]`
  disappear. Both devices; the gain is on Mali.
- **Mechanism**: `label_pixels_body.glsl:23-36` streams `thresholded` + `parent` (1 MB at d2),
  gathers `blob_size[r]`, and rewrites all of `parent` (1 MB) for every pixel, so that
  `blob_diff_body.glsl:55-85` can read six packed words per pixel. But a pixel can only emit when
  `c0 + cN == 2` for some N (`blob_diff_body.glsl:96-99`), i.e. at a black/white transition. At d2
  that is roughly boundary_points/2 = ~33k emitting pixels of ~255k interior pixels (~13%; an
  emitting pixel on a straight edge emits 2 points, see finding 2). For the other ~87% the label
  is never consulted, and the codes alone decide. So:
  - common case: 6 byte loads from `thresholded` (cache lines shared with neighbours), exit;
  - transition case: `parent[i]` -> `blob_size[r0]`, then the same for each N with
    `c0 + cN == 2`, plus W and S only when the SW duplicate rule needs `lW`/`lS`
    (`blob_diff_body.glsl:91-92`). Roots are shared by long boundary stretches, so the
    `blob_size` gathers are cache-hot.
  `label_pixels` is the most bandwidth-sensitive real pass in the frame (3.12x across the 4x DMC
  sweep, PERFORMANCE.md 3a), so removing it removes real DRAM traffic, not just a launch.
- **Expected gain** (estimate): the `label_pixels` span, ~0.095 ms at d2 (0.116 ms in 3a minus the
  shipped -18%), i.e. ~3% of Mali GPU time, minus the added dependent gathers in `blob_diff` for
  ~13% of pixels. `boundary` is 1.09x DMC-insensitive, so its cost is not traffic; I expect the
  gathers to overlap with whatever it is bound by (finding 2 argues the append atomic). Net
  **-2 to -3% Mali GPU total**; more at d1 (label_pixels scales with pixels). RDNA: ~0 (the span is
  ~5 us there).
- **Cost and risk**: ~40 lines in `blob_diff_body.glsl` (+ a u8/32-bit wrapper pair, since
  `blob_diff` regains a `thresholded` binding), delete `label_pixels*` (3 files) and one dispatch
  and barrier in `RecordFusedTail` / `DetectUnfused`. Exact by construction: every emit decision
  uses the same `c`, `l` values, computed lazily. Side benefit: `parent[]` is never rewritten in
  place, so the "destructive rewrite before convergence" hazard (OPTIMIZATION_NOTES item 8, bug 1,
  a `VK_ERROR_DEVICE_LOST` on Mali) disappears as a class; `blob_diff` keeps its
  `honour_changed_flag` gate only to avoid wasted work.
- **Prior art in the tree**: first pass item 7 introduced `label_pixels` because `blob_diff` then
  did `blob_size[parent[n]]` for all six neighbours of **every** pixel (boundary 3.25 -> 1.71 ms at
  1080p). This proposal keeps that win (no per-pixel gathers) and removes the pass that bought it.
  A3 (second pass) folded the code into `parent[]` to save the second array; here `blob_diff`
  reads the code array instead of `parent[]` in the common case, which is fewer bytes (1 B vs 4 B
  per load on the u8 path). Not proposed in research/03-06 (checked: 06 and DESIGN.md only discuss
  the fold, not removing the pass).
- **Verdict**: **pursue**. Measure: ABBA `boundary` + `label_pixels` spans and GPU total on the Pi
  at d1/d2/d4 (`apriltag_vulkan_validate --profile`, 12-16 rounds), `--dump-quads` byte-identical
  against the baseline, plus `--stress noise|checker` (dense transitions are this change's worst
  case: on `checker` nearly every pixel is a transition pixel, so check it does not regress there).

### 2. `blob_diff`'s append counter: one atomic per pixel, then (if bounded) K-way partitioning

- **Idea**: reserve all of a pixel's 0-4 points with one
  `atomicAdd(counter, bitCount(want mask))` instead of one `atomicAdd` per point; if a bound shows
  the counter is the bottleneck, additionally split it into K counters with K disjoint output
  regions. Both devices; aimed at Mali.
- **Mechanism**: `append()` (`blob_diff_body.glsl:32-41`) does `atomicAdd(counter, 1u)` per point
  on a single address: ~65k same-address atomics per frame at d2, ~190k at d1. `boundary` is the
  third-largest Mali span (0.361 ms) and is **1.09x** across a 4x DMC sweep, so it is not bandwidth;
  its loads are six cache-shared words per thread (~1.5M lane-loads at d2, which is tens of
  microseconds of LS issue on 4 cores, not 360 us). The tree has already shown how expensive
  same-address atomics are on this part: `extents` with ~170-way contention per address measured
  0.434 ms against 0.062 ms with plain stores (PERFORMANCE.md 3b). A single counter is 65k-way.
  An emitting pixel on a straight edge emits 2 points (horizontal edge: S + SE, SW suppressed as a
  duplicate; vertical edge: E + SE), so per-pixel reservation roughly halves the atomics on that
  address. K-way partitioning (counter = `workgroup_linear_id & (K-1)`, region k sized for the
  dense worst case of the workgroups that map to it) divides the per-address rate by K again;
  consumers (`hash_group`, `reduce_extents_hash`, `scatter_index_points`) then map a dense index
  to (region, offset) through a K-entry prefix written by `build_qbp_args`.
- **Expected gain**: unknown until bounded. If `boundary` is mostly counter serialization (my
  reading of the 1.09x ratio plus the extents precedent), per-pixel reservation alone could take
  ~30-50% off the span, i.e. **-0.1 to -0.15 ms, -4 to -5% Mali GPU total** (estimate). If the
  bound shows the counter is not the cost, both parts are dead and cost nothing to have checked.
- **Cost and risk**: per-pixel reservation is ~10 lines and exact: same point set, different
  arrival order, and the order is already nondeterministic (PERFORMANCE.md 7) with zero genuine
  `theta_key` ties on the corpus (OPTIMIZATION_NOTES item 13). Capacity truncation behaves as
  today (scheduling-dependent which points drop past `qbp_capacity_`). It also improves downstream
  key locality (a pixel's points become adjacent), which item 11 showed the extents privatization
  relies on. K-way partitioning is ~60 lines across four shaders, changes the overflow semantics
  under `APRILTAG_VK_MAX_POINTS` (a region can fill while others have room), so it needs the
  region sizing to stay dense-worst-case by default.
- **Prior art in the tree**: the subgroup-ballot aggregation of this counter was deleted (-24%
  scalar on RX 9060 XT; integrated GPUs never take subgroup variants, PERFORMANCE.md 6).
  research/05 idea 5 proposes shared-memory aggregation (needs barriers, a class that lost on Mali)
  and gives the right bound. Per-thread multi-slot reservation (no subgroup ops, no shared memory,
  no barriers) and counter partitioning are not in the tree or research/.
- **Verdict**: **measure first**, cheaply. Bound: a deliberately incorrect build where
  `pos = (oy * iw + ox) * 4 + k` (no atomic; consumers dispatched over the dense count, empty
  slots marked) gives the ceiling on the `boundary` span; check downstream counters still match
  (the extents trap in OPTIMIZATION_NOTES 3rd pass item 3). If the ceiling is > ~30% of the span,
  build per-pixel reservation and ABBA it on the Pi; consider K-way only if a gap remains.

### 3. Drop the in-chunk `uf_compress` passes; flatten in `uf_final` instead

- **Idea**: in the steady-state chunk (merge, compress, merge-verify, guarded compress), drop both
  compressions and have `uf_final` do a full `find()` with a path-compression store
  (`parent[i] = root`) before counting. Both devices.
- **Mechanism**: `RecordUfChunk` (`GpuDetector.cpp:643-668`) records, for the default chunk of 2:
  `uf_merge` (real unions), `uf_compress` (real flattening), `uf_merge` (verification, joins
  nothing on every corpus/stress frame, PERFORMANCE.md 8g), and a guarded `uf_compress` that
  launches every invocation only to read the flag and exit (no conditional rendering on Mali).
  The only consumers that need a flat `parent[]` are `uf_final` (`uf_final_body.glsl:25`, which
  reads `parent[i]` as the root) and, after it, `label_pixels`/`blob_diff`. `uf_merge`'s `find()`
  works on any valid forest. So the flattening can move into `uf_final`, which already streams
  `parent[i]` for every non-127 pixel: after the post-init compress every pixel points at its run
  start, and the merge only re-hooks run-start roots, so the walk is typically 1-3 cache-hot hops.
  127 pixels are always their own root and are skipped as today.
- **Expected gain** (estimate, anchored on measured ceilings): removing a converged
  `uf_compress` dispatch outright was measured at **labelling -4.4%, GPU total -1.4%** on Mali
  (PERFORMANCE.md 6c ceiling); the guard recovers about half, so the trailing guarded pass still
  costs ~0.7%. The real mid-chunk compress costs at least the same pass overhead plus its walks
  and writes; the walks and writes move into `uf_final`, the pass overhead (launch, 1 MB stream of
  `parent`, barrier) does not. Net **-1.5 to -2% Mali GPU total**, minus a small increase in
  `uf_merge`-verify's `find()` depth (it now walks unflattened paths at overlap starts only, ~12%
  of pixels, with path splitting).
- **Cost and risk**: ~15 lines (`uf_final_body.glsl` gets a find + store; `RecordUfChunk` stops
  emitting compresses for the last two passes of a chunk). Exact: the forest, the roots
  (root = minimum index) and therefore every label are unchanged; stores only ever write an
  ancestor, the same safety argument as path splitting (OPTIMIZATION_NOTES item 10). The
  speculative tail is still safe: `uf_final` on an unconverged forest is a valid compression, and
  the rerun re-clears `blob_size`. Multi-chunk (non-converged) frames keep a compress between
  non-final merges so long merge sequences still get flattened. The 6c "merge that joined nothing
  leaves parent flat" invariant is no longer needed and its comment must go.
- **Prior art in the tree**: research/03 idea 6 proposed the mirror image (fold the count into the
  final compress) and judged it a wash because it gives up the 6c skip; this variant keeps the
  skip's benefit (the trailing pass disappears entirely) and also removes the real mid-chunk pass.
  "Removing the second merge measured zero" (PERFORMANCE.md 8 item 3) is about the merge, not the
  compresses.
- **Verdict**: **pursue**. Measure: ABBA `labelling` + `uf_final` spans and GPU total at d1/d2/d4
  on the Pi; `--dump-quads` identical; `--max-uf 1` (forces the retry path) and
  `APRILTAG_VK_UF_CHUNK=1..4` identical.

### 4. Thread-coarsened run-merge in `reduce_extents_hash`

- **Idea**: each invocation handles 2-4 consecutive points, merges runs of equal `slot` in
  registers (min/max/sums), and issues one set of atomics per run instead of per point. Both
  devices; Mali is where the scalar/int64 path runs.
- **Mechanism**: `extents` is atomic-throughput-bound, not bandwidth (1.01x DMC, PERFORMANCE.md
  3a; plain-store bound 0.062 ms vs 0.221 ms shipped). After the read-before-atomic guards, every
  point still issues two unconditional 64-bit `atomicAdd`s (`reduce_extents_hash_body.glsl:47-56`).
  Consecutive points already share a blob for long stretches (`blob_diff`'s append locality is
  what made workgroup-keyed copies beat lane-keyed ones, OPTIMIZATION_NOTES item 11), and
  finding 2 would make a pixel's 2 points adjacent by construction. A 4-point run of one slot
  then costs 2 atomics instead of 8 and up to 4 guarded min/max probes instead of 16.
- **Expected gain**: the atomics account for at most ~0.16 ms (0.221 - 0.062). If the mean run
  over 4-point windows is ~3, unconditional atomics drop ~3x: **~-0.05 to -0.08 ms, -2 to -3%
  Mali GPU total** (estimate; the run-length distribution is unmeasured).
- **Cost and risk**: ~30 lines; exact (integer min/max/sum are associative and commutative, which
  is already relied on by the 8-way privatization). The item 9 lesson ("fewer, fatter threads
  never paid on Mali") is about reshaping a fixed amount of work; this reduces the dominant work
  item (atomics) itself, the kind of change the same note says does pay. Thread count drops 4x, so
  workgroup-keyed copy selection now spans 4x more points per copy; keep K=8 and re-check.
- **Prior art**: research/06 covers privatization (shipped), lane-keyed copies (rejected),
  per-core copies (`gl_CoreIDARM`, unmeasured), struct padding (rejected). Run-merging before the
  atomic is the CPU/GPU histogram "aggregate before you atomically add" technique without
  subgroup ops; not in the tree.
- **Verdict**: **measure first**. Bound: count, per frame, distinct-slot runs over 4-point windows
  (a counter in the shader, or offline over a dumped `point_slot[]`); build only if the mean run
  is >= 2. Then ABBA the `extents` span. Do it after finding 2, which changes the run statistics.
- **Same technique, second site**: `scatter_index_points.comp:64` does
  `atomicAdd(blob_cursor[sel], 1u)` per selected point, the same "consecutive points, same blob"
  contention shape as `extents` (~17k atomics over ~200 cursors at d2). Run-merging gives one
  `atomicAdd(cursor, run_len)` per run. Alternative that removes the atomic entirely: the
  `count` atomic in `reduce_extents_hash` already returns a unique per-copy rank (the low half of
  the old 64-bit value); store it per point, have `merge_extents` write per-copy bases, and
  scatter computes `pos = offset + copy_base + rank` with no atomic. That costs a 4 B write per
  point in `reduce` (73% of them for discarded blobs), so prefer run-merging unless the scatter
  span turns out large. The Mali `scatter` span has never been recorded; read it first.

### 5. Delete the `decimate` pass and fold `uf_init` into `threshold`, by recomputation (no shared memory)

- **Idea**: (a) point-sampled decimation is an addressing mode: `block_minmax`, `threshold` and
  `sort_points_local`'s gradient taps read `gray[(y*d)*W + x*d]` directly and `decimate` plus
  `decimated_buf_` go away. (b) `threshold` also computes the left neighbour's code (one more byte
  and, at `x % 4 == 0`, one more `minmax_filtered` word) and writes `parent[i]` itself, so
  `uf_init` goes away. Both devices; Mali first.
- **Mechanism**: `decimate_u8.comp:26-28` is a pure subsample, matching upstream
  `image_u8_decimate` for integer factors. Its only consumers are `block_minmax_u8.comp:23`,
  `threshold_u8.comp:36` and `sort_points_local_body.glsl:42-45` (checked in
  `CreatePipelines`). Reading the source directly keeps DRAM traffic about level (at d2 the even
  rows' lines are fetched by two passes instead of being fetched once and the 256 KB result being
  written and read twice) and removes a full dispatch, its barrier and a 256 KB write. At d1
  `decimate` is a 1 MB copy, so the saving is larger. `uf_init_u8.comp:22-24` needs only
  `thresholded[i]` and `thresholded[i-1]`; `threshold` can recompute `i-1`'s code from data it
  already touches, removing another full 2D dispatch, its barrier and a 256 KB re-read.
- **Expected gain** (estimate): the `threshold` span (all four front-end dispatches) is ~0.14 ms,
  ~5% of the Mali frame, with 1.34x DMC sensitivity (research/06). One of four dispatches plus
  `uf_init` (inside `labelling`) is plausibly **-1 to -2% Mali GPU total** at d2, more at d1.
- **Cost and risk**: (a) touches the YUYV import path: `decimate_yuyv` exists because luma sits
  at byte `2x` in YUYV, so `block_minmax`/`threshold` would need a source-format spec constant
  (not new shader files; the tree dislikes variant growth, OPTIMIZATION_NOTES third pass). On the
  RX 9060 XT with host-pointer import, the frame lives across PCIe and `decimate` already costs
  +21 us there (PERFORMANCE.md 8d); reading it twice would cost more, so keep `decimate` on
  discrete parts reading imported memory (one bool, chosen at construction). Exact in both parts:
  same sample positions, same arithmetic.
- **Prior art in the tree**: the third pass fused `decimate`+`block_minmax` and
  `block_filter`+`threshold` **with shared-memory tiles and barriers** and lost +5.4..+7.8% on
  Mali (OPTIMIZATION_NOTES "Fusing the preprocessing dispatches"). That loss was attributed to
  L2-backed `shared` plus barriers. This variant has neither: it deletes a pass and re-reads
  cached source bytes. Not proposed before.
- **Verdict**: **measure first**, cheaply and exactly: for a repeated still image, a build that
  records `decimate` only on the first frame produces identical output (the buffer already holds
  this frame's decimation), so ABBA it against the baseline to get (a)'s ceiling with no
  incorrectness; the gap between spans gives (b)'s ceiling the same way (skip `uf_init` after
  frame 1 is *not* exact, since compress rewrites `parent`; time `uf_init` alone with an extra
  timestamp pair instead).

### 6. `sort_points_local`: barrier-free rank sort for small blobs

- **Idea**: for blobs with `count <= T` (T = 64..256), compute each point's output position as
  its rank `#{j : key_j < key_i}` over the blob's packed keys, in a flat one-thread-per-point
  dispatch with **no shared memory and no barriers**; keep the Batcher network for larger blobs.
  Both devices; aimed at Mali.
- **Mechanism**: the sort span (~0.40 ms, ~13%) is 0.99x DMC-insensitive and the comparator
  enumeration that cut RDNA's span by 27-49% did nothing on Mali (OPTIMIZATION_NOTES item 15),
  which the notes attribute to barriers and latency. Each blob costs
  `log2(cap)(log2(cap)+1)/2` barriered rounds regardless of size (21 rounds for 64 points, 28 for
  128) in a 256-thread workgroup, most of whose warps idle. The packed key
  `(theta_key << 12) | local_index` (`sort_points_local_body.glsl:90-91`) is unique, so **every**
  correct sort produces the same permutation: switching algorithm per blob size is bit-identical
  by construction. Rank-by-counting reads the blob's keys in a loop where all lanes of a warp
  read the same address in the same iteration (a broadcast), and the loads are independent, so
  they pipeline. Work is `sum(n_b^2)`, which is small when restricted to small blobs.
- **Expected gain**: depends entirely on the blob-size histogram, which is not recorded. If, as
  research/05 estimates, most blobs are <= 128 points (mean 112 at d1), the small blobs are most of
  the workgroups and most of the barrier rounds: plausibly **-20 to -40% of the sort span,
  -2 to -5% Mali GPU total** (estimate). RDNA: smaller.
- **Cost and risk**: a second indirect dispatch (small-blob list) or a per-point early-out in one
  flat dispatch over the point count (blob size is in `selected[blob_index].count`); ~50 lines.
  Large blobs keep today's path. Load imbalance is bounded by T.
- **Prior art**: research/05 idea 2 (pack several small blobs per workgroup, still barriered) and
  idea 7 (rank-by-counting only for count <= 32, inside packed workgroups). This variant removes
  barriers and shared memory altogether, which is the specific Mali cost the tree measured; the
  unique-key argument for exactness across algorithms is not stated in the tree.
- **Verdict**: **measure first**. The histogram needs no shader change: `selected_extents_buf_`
  is read in place by the host every frame (`count` per selected blob), so a validate-tool flag
  can print it. Then the research/05 ceiling build (skip the network for `count <= T`, incorrect)
  bounds both this and the packing idea in one run; build this one if the ceiling is >= ~0.08 ms.

### 7. Dispatch diet in the tail (small, exact, cheap)

- **Idea**: remove tiny dispatches and their barriers: (a) the `hash_group` CAS winner writes the
  identity into its new dense id's `extents` entry and private copies, deleting `init_extents`;
  (b) `select_blobs` folds the 8 private copies itself, deleting `merge_extents` (nothing else
  reads the folded canonical array; `scatter` and the host read `selected[]`); (c) research/05
  idea 4, the atomic range allocator in `select_blobs`, deleting `extract_blob_counts` and the
  2-3 scan dispatches. Both devices; Mali, where a barrier costs 2.6 us even between trivial
  dispatches (PERFORMANCE.md 3).
- **Mechanism**: `RecordFusedTail` (`GpuDetector.cpp:731-852`) has ~17 dispatches; about nine are
  tiny (3 `build_*_args`, `init_extents`, `merge_extents`, `select_blobs`, `extract_blob_counts`,
  2x `scan_block`, `scan_add_offsets`). (a)+(b)+(c) remove six of them with their barriers.
- **Expected gain** (estimate): 6 x ~3-5 us = **~20-30 us, ~-1% Mali GPU total**; negligible on
  RDNA (1.6 us barriers).
- **Cost and risk**: (a) ~10 lines, (b) ~15 lines (deletes a shader), (c) as research/05 says.
  All exact; (c) changes `blob_point_offsets` semantics (host reads the total from the allocator's
  counter instead of `offsets[max_blobs-1]`), a documented invariant to update.
- **Prior art**: (c) is research/05 idea 4 (~10 us). (a) and (b) are new.
- **Verdict**: **pursue (bundle)**; one ABBA on the Pi for the bundle, `--dump-quads` identical.

### 8. Fold `hash_group` into `blob_diff` via a 64-bit key-in-slot CAS

- **Idea**: store the 64-bit key `(rep1 << 32) | rep0` in the hash slot (0 = empty is safe since
  `rep1 > rep0 >= 0`), so an insert never needs another thread's earlier write; then `blob_diff`
  can insert at emission time and write `point_slot[pos]` directly. `hash_group`, the `keys[]`
  buffer (8 B per point written and read) and one dispatch+barrier disappear. Devices with
  `shaderBufferInt64Atomics` (the Mali has it; it runs `reduce_extents_hash_atomic64`).
- **Mechanism**: today's insert is spin-free only because losers read `keys[owner]`, written in an
  earlier dispatch (`hash_group.comp:4-8`). Key-in-slot removes that dependency, which is what
  makes fusion legal. The winner still numbers the slot with `atomicAdd(raw_blob_counter)` and
  writes `slot_dense[h]`, consumed only by later dispatches. Read-before-CAS keeps atomics to
  roughly one per distinct key.
- **Expected gain** (estimate): the `hash_group` span, unrecorded on Mali (2.2-3.3% of GPU time on
  the MX230 / Iris tables, OPTIMIZATION_NOTES second pass), minus a hash + one hot 64-bit load per
  point inside `blob_diff`: **~-1 to -2% Mali GPU total**.
- **Cost and risk**: ~40 lines; a third `blob_diff` variant axis (int64 or not) unless the 32-bit
  path keeps today's two-pass scheme. Table clear doubles (64-bit slots, 512 KB at 65536 slots).
  Exact (grouping is unordered; dense ids are already arrival-ordered).
- **Prior art**: research/06 idea 10 proposes key-in-slot CAS inside `hash_group` (low priority,
  to drop a dependent load). The fusion it enables is new.
- **Verdict**: **measure first**: record the Mali `hash_group` span; build only if >= ~0.04 ms.
  Lower priority than 1-3 because it adds a variant axis.

### Addendum to finding 2

Even if Mali already merges same-address atomics within one 16-lane warp instruction, `append()`
is called four times per thread (`blob_diff_body.glsl:107-110`), so a warp issues up to four
atomic instructions on the counter where one would do. Per-pixel reservation cuts the
instruction count 4x -> 1x in that case, independent of how the hardware handles lanes. General
GPU references agree that same-address atomics serialize at L2 and their throughput is set by
latency, and list privatization and aggregation as the remedies (sources below); there is no
public Valhall figure, so the bound build is the only way to size it.

## Considered and not proposed

- **Exact early culling of points before `hash_group`.** 73% of boundary points are discarded at
  `select_blobs`, but every rejection test there (point count, bounding box, border polarity via
  the `dot` sum) needs the per-cluster reduction. No component-level test implies any of them
  exactly (e.g. the count bound `count(A,B) <= 4(|A|+|B|)` is already implied by the pixel floor).
  The optional aspect/fill prefilters are not upstream-exact and stay opt-in. Nothing to add.
- **Smaller per-point records** (e.g. `IPoint` 12 -> 8 B by dropping `blob_index`, which the
  per-blob workgroup already knows). `sort` and `scatter` are bandwidth-insensitive on Mali
  (0.99x, PERFORMANCE.md 3a) and the line-fit halving measured nil there (6b). RDNA-only, tiny.
- **Avoiding the 1 MB `blob_size` clear.** Only root entries need zeroing and roots are always
  run starts, so `uf_init` could zero just run starts; but at d2 the mean run is ~5.6 px, so
  nearly every 64 B line still gets written. Epoch-tagged counters need CAS loops. Not worth it.
- **Shrinking the hash table** below `max_raw_blobs` to cut its 256 KB clear: ~7 us and breaks
  the capacity guarantee.
- **2024-2026 CCL literature.** A search for newer GPU CCL / union-find work turned up nothing
  beyond what research/03 already covers (BUF/BKE 2019-2020, Playne, DLP, coarse-to-fine). The
  remaining labelling levers are research/03's segmented run-start init and inline compression,
  which this round's graph agent (04) owns; findings 3 and 5(b) here compose with both.

## Sources

- Kirk & Hwu, *Programming Massively Parallel Processors*, ch. 9 (histogram: privatization,
  aggregation of consecutive same-bin updates before the atomic, serialization of same-address
  atomics): https://www.oreilly.com/library/view/programming-massively-parallel/9780128119877/xhtml/chp009.xhtml
- NVIDIA Nsight docs on atomic serialization at L2 for the same address:
  https://docs.nvidia.com/nsight-visual-studio-edition/4.6/Content/Analysis/Report/CudaExperiments/KernelLevel/MemoryStatisticsAtomics.htm
- Intel oneAPI GPU optimization guide, memory system (same-address atomics):
  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/gpu-memory-system.html
- CCL search baseline (nothing newer than research/03): Allegretti et al. BUF/BKE
  https://iris.unimore.it/handle/11380/1179616 ; Optimized UF for CCL https://arxiv.org/pdf/1708.08180
- Everything else is cited to files and lines in this tree.
