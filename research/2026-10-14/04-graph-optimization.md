# 04: Graph optimization (frame compute graph, CCL / union-find graph algorithms, ML-compiler ideas)

Status: complete

## Top conclusions (ranked by Orange Pi gain, then desktop)

Ceiling for everything in this file: a second concurrent stream fills ~0.25 ms/frame of GPU idle on
Mali (~10% of the GPU phase) and ~0.06 ms on the desktop (F1). No intra-frame graph change can win
more than that.

1. **Re-plan the labelling tail of the speculative first attempt** (F2): fold `uf_final` into the
   last real `uf_compress`, drop the guarded `uf_compress` after the verification merge, and give
   each merge its own flag slot so the mid-chunk flag fill and its two barriers go. Exact (relies on
   the existing retry path). Estimate -0.06 to -0.09 ms on Mali (2.5-3.5% of GPU), small on RDNA.
   Pursue.
2. **The tree has never measured what its barriers cost** (F1). Every dispatch's barrier is
   recorded *before* the span-end timestamp, so the inter-span "gap" instrumentation that concluded
   "intra-submit fusion has ~0.04 ms left" structurally cannot see barriers. The multi-stream data
   gives a usable ceiling instead: on Mali ~0.25 ms/frame (~10% of GPU) of bubbles exist that a
   second stream fills. Measure with an exact "split test" before and after any fusion work.
3. **Let the per-frame clears overlap compute** (F3): the 1.3 MB of `vkCmdFillBuffer` is
   bandwidth-bound (~0.05 ms on Mali) and serialised at the head by a barrier no early dispatch
   needs. Exact, a few lines. Measure.
4. **Fold the tail's tiny dispatches** (F4): `merge_extents` into `select_blobs`, `init_extents`
   into `hash_group`'s slot-win path, `extract_blob_counts` + 3-dispatch scan into one workgroup,
   and the three `build_indirect_args` dispatches into their producers' last workgroup. ~8 fewer
   dispatches and barriers. Mali ~1-2%, desktop ~3-5% (barriers cost a fixed ~1.6 us there).
5. **Measure first, after the above:** fuse `label_pixels` into `blob_diff`, and insert into the
   grouping hash from `blob_diff` with a 64-bit key CAS so `hash_group` and `keys[]` go (F6, ~1-3%
   on Mali or a loss). Consolidate the counters into one block (F5, small; only alongside F3/F4).
6. **No new CCL / union-find algorithm** since the previous round (F9). With convergence at the
   minimum 2 passes, only per-pass cost is left: F2 here plus research/03 items 2-3.
7. **Auto-tuning** (F10): there is no TVM/IREE path for this kind of code; per-pipeline workgroup
   tuning through the existing specialization constants is cheap, but expect ~0-2% on Mali.
8. **Reject:** megakernels with grid barriers, work graphs (`VK_AMDX_shader_enqueue` is not
   exposed by AMD 26.8.1 and is still provisional), DGC (F8). Running two frames in flight for one
   camera gives +16% throughput on Mali and no latency gain, so keep it as a recipe only (F7).

## Findings

### F0. The per-frame compute graph as recorded today (fused path, pre-recorded, chunk = 2)

Read from `GpuDetector.cpp` `RecordFrameHead` (l.671-729), `RecordUfChunk` (l.636-669),
`RecordFusedTail` (l.731-852) and `ComputePipeline.cpp` (`DispatchRaw` records `vkCmdDispatch`
then the barrier, l.203-204; `Barrier` is always one global `VkMemoryBarrier`, l.238-263).
1280x800, decimation 2: 640x400 = 256,000 px, `max_raw_blobs` 65536, `max_blobs` 2048,
`scan_wg` 1024 on both GPUs. C = compute->compute barrier, CAT = compute+transfer both sides,
IDB = compute -> DRAW_INDIRECT, H = transfer -> host.

| # | Command | Reads | Writes | Barrier after |
| --- | --- | --- | --- | --- |
| 0 | 9 x `vkCmdFillBuffer` (4 B counters x6, `blob_size` 1.0 MB, `hash_owner` 256 KB, `blob_cursor` 8 KB) | - | counters, blob_size, hash_owner, blob_cursor | B1 CAT |
| 1 | decimate | gray | decimated | B2 C |
| 2 | block_minmax | decimated | minmax_unf | B3 C |
| 3 | block_filter | minmax_unf | minmax_f | B4 C |
| 4 | threshold | decimated, minmax_f | thresholded | B5 C |
| 5 | uf_init | thresholded | parent | B6 C |
| 6 | uf_compress (post-init, unguarded) | parent | parent | B7 C |
| 7 | uf_merge a | parent, thresholded | parent, flag | B8 C |
| 8 | uf_compress a (guarded) | parent, flag | parent | B9 C |
| 9 | fill flag (4 B) between B10 CAT and B11 CAT | - | flag | B10, B11 |
| 10 | uf_merge b (verification) | parent, thresholded | (nothing when converged), flag | B12 C |
| 11 | uf_compress b (guarded; early-exits when converged) | flag | - | B13 C, B14 CAT |
| 12 | copy flag -> staging | flag | staging | - |
| 13 | uf_final | thresholded, parent | blob_size (atomics) | B15 C |
| 14 | label_pixels (gated on flag) | thresholded, parent, blob_size | parent (in place) | B16 C |
| 15 | blob_diff (gated) | parent | compacted, keys, qbp_counter | B17 CAT |
| 16 | copy qbp_counter; build_qbp_args (1 thread) | qbp_counter | args[1] | B18 IDB |
| 17 | hash_group (indirect) | keys, hash_owner | point_slot, slot_dense, raw_blob_counter | B19 C |
| 18 | build_indirect_args (1 thread) | raw_blob_counter | args[0] | B20 IDB |
| 19 | init_extents (indirect) | - | extents (+7 private copies) | B21 C |
| 20 | reduce_extents_hash (indirect) | compacted, point_slot, slot_dense | extents (atomics) | B22 C |
| 21 | merge_extents (indirect) | extents copies | extents | B23 C |
| 22 | select_blobs (indirect) | extents | selected, remap, selected_counter | B24 C |
| 23 | extract_blob_counts (2048) | selected, selected_counter | offsets | B25 C |
| 24-26 | scan_block L0 (2 WG x 1024), scan_block L1 (1 WG), scan_add_offsets L0 | offsets | offsets | B26-B28 C |
| 27 | scatter_index_points (indirect) | compacted, point_slot, slot_dense, remap, selected, offsets | index_points, blob_cursor | B29 CAT |
| 28 | 4 x copy (raw_blobs, drops, point total, selected count) | | staging | - |
| 29 | build_sort_args (1 thread) | selected_counter | args[2] | B30 C, B31 IDB |
| 30 | sort_points_local (indirect, 1 WG/blob) | selected, offsets, index_points, decimated | line_fit_points | B32 CAT |
| 31 | copy oversized counter | | staging | B33 H |

Totals per frame: **27 dispatches, 10 fills, 7 copies, 33 pipeline barriers**, plus 26 timestamps
and a query-pool reset when profiling. The dependency graph is a **single chain**: apart from the
fills, the copies and the three 1-thread `build_*_args` dispatches, every dispatch reads what the
one before it wrote. So there is almost no *independent* work to reorder into parallel; the graph
levers are (a) fewer nodes (fusion), (b) fewer or later barriers where the edge is not a real
dependency, and (c) overlapping the chain with something else (fills, another frame).

Real dependencies that are **not** adjacent in the chain (so a barrier could be deferred):
- The fills: the first readers are `uf_merge a` (flag), `uf_final` (`blob_size`), `blob_diff`
  (`qbp_counter`), `hash_group` (`hash_owner`), `scatter` (`blob_cursor`). Nothing in dispatches
  1-6 touches a filled buffer, yet B1 sits in front of `decimate`.
- The copies at #28 read counters finished at #17-#26; only the transfer stage forces B29 to be
  CAT instead of C.
- `build_sort_args` needs only `selected_counter` (#22), not `scatter`.

### F1. The tree's evidence on barrier cost does not measure barriers (both devices; methodology)

- **Idea:** before deciding whether fusion and barrier work is worth anything, fix the evidence.
  The "intra-submit gaps total ~0.04 ms on Mali, so intra-submit fusion has almost nothing left"
  conclusion (commit `8b081d8`, also the basis of OPTIMIZATION_NOTES item 4's sync2 row:
  "intra-submit gaps 0.067 ms total") cannot see barriers at all.
- **Mechanism:** `ComputePipeline::DispatchRaw` records `vkCmdDispatch` *and then* the barrier
  (`ComputePipeline.cpp:203-204`), and the span-end `vkCmdWriteTimestamp` comes after that call
  (`GpuDetector.cpp:739-740` etc.). So each barrier, and the pipeline drain it causes, is inside the
  span of the dispatch before it, and the "gap" between span-end and the next span-start contains
  no command at all. Verified the same at `f29f29c` (the commit that added the gap metric). The
  timestamps also use `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` (`QueryPool.h:33`), i.e. every timestamp
  is itself a drain point, which is the likely reason timestamps cost ~9% of Mali GPU time
  (PERFORMANCE.md 3a: 3.16 vs 2.90 ms) and why spans are attribution-only.
- **What the tree does know:** a Mali barrier costs 2.6 us between trivial dispatches and 18.7 us
  between large ones; RDNA4 ~1.6 us fixed, "~31 barriers total ~48 us" (PERFORMANCE.md s3, origin
  commit `0f16d8f`, method not recorded). With the 33 barriers of F0 that is 0.09-0.6 ms on Mali
  (3.5-23% of a ~2.55 ms GPU phase) and ~50 us on RDNA4 (~15% of its ~0.33 ms GPU phase).
- **A measured ceiling that does exist:** concurrent streams fill exactly these bubbles.
  Orange Pi: 1 stream 381 fps (2.62 ms/frame), 2-4 streams ~445 fps (2.25 ms) (PERFORMANCE.md
  8g). Of the 0.37 ms difference ~0.126 ms is host overhead per `Detect()` (8g) that a second
  stream hides, leaving **~0.25 ms/frame (~10% of the GPU phase) of intra-frame GPU idle or
  under-fill** (barriers, ramp-down tails, launch latency, end-of-submit flush). Desktop: 3,000 vs
  4,600 fps = 0.333 vs 0.217 ms, minus 58 us host = **~0.06 ms**. Every intra-frame graph idea
  below (fusion, fewer barriers, overlap) is bounded by these two numbers; nothing in this file can
  win more than ~10% on Mali.
- **How to measure barrier cost exactly (no deliberately wrong build needed):** the *split test*.
  Pick a large dispatch whose result does not depend on workgroup order (`uf_final`,
  `label_pixels`, `uf_compress`) and record it as two half-range dispatches with a C barrier
  between them; output is bit-identical. Delta GPU total = one in-situ large-dispatch barrier plus
  one launch. Do the same with two halves *without* a barrier (legal for these three, since halves
  touch disjoint pixels and only commutative atomics are shared) to separate launch from drain.
  Run with timestamps **off** (GPU total from `gpu.detect` - host overhead), ABBA, 12+ rounds on the
  Pi, as PERFORMANCE.md 8g prescribes. This converts every estimate below from "2.6-18.7 us" to a
  number.
- **Expected gain:** none by itself; it decides whether F2-F4 are 1% or 5% items.
- **Cost / risk:** ~20 lines behind an env knob, removed afterwards. No accuracy impact.
- **Prior art:** commit `f29f29c`/`8b081d8` (gap metric and its conclusion), PERFORMANCE.md s3,
  research/07 BP-4 (proposed a no-barrier bound, never run).
- **Verdict: measure first** (do this before F2-F4). Also worth a one-line correction in
  OPTIMIZATION_NOTES item 4 (sync2 row) once measured.

### F2. Re-plan the labelling tail of the speculative first attempt (both; Mali first)

- **Idea:** in the pre-recorded first attempt only, (a) do `uf_final`'s counting inside the last
  *real* compression (`uf_compress a`, row 8 of F0), (b) do not record the guarded
  `uf_compress b` after the verification merge at all, and (c) give the verification merge its own
  flag word so the mid-chunk `fill flag` and its two CAT barriers (B10, B11) disappear. Net: -2
  dispatches (`uf_compress b`, `uf_final`), -1 fill, -4 barriers (B10, B11, B13, B15), and one less
  full read of `parent[]` + `thresholded`.
- **Why it is exact.** The steady state is "1 real merge + 1 verification merge": the tree's own
  data says every corpus and stress frame converges in exactly 2 iterations (PERFORMANCE.md 8g), and
  that is a theorem, not luck: `doUnion` retries through the `atomicMin` return value until both
  ends share a root, so every overlap-start union is complete when its thread exits (the ECL-CC
  "hook each edge once" argument, research/04 ECL-CC entry; item 7's induction covers the
  non-start columns). Hence after `uf_merge a` + `uf_compress a`, `parent[]` is flat *and final*
  whenever the verification merge sets no flag, and in exactly that case:
  - `uf_compress a` already holds each pixel's final root `n` in a register (`uf_compress.comp`
    l.27-31), so `if (code != 127 && blob_size[n] < min) atomicAdd(blob_size[n], 1)` there is
    `uf_final_body.glsl` verbatim, minus its two reloads;
  - `uf_compress b` is a no-op by construction (it is guarded on the flag, and `findSplit` performs
    no stores on a flat forest because `p > next` is false at a root).
  When the verification merge *does* set the flag, the attempt is already discarded and
  `Continuation` + `TailRerun` (`GpuDetector.cpp` l.854-877) redo labelling from the current
  `parent[]` (a valid, merely unflattened forest, which `uf_merge` accepts) and then rerun the
  unfused tail including a real `uf_final` with `blob_size` re-zeroed. So the cold path keeps
  today's command buffers and nothing about its correctness changes. The only extra condition:
  `uf_compress a` must not take its "flag == 0, return" early exit when fused (it would skip the
  counting); it can still skip the walk.
- **(c) in detail:** make `changed_flag` a 2-word array indexed by a push constant. Word 0 is set
  by `uf_merge a`, word 1 by the verification merge; both are zeroed by the head's existing fill
  batch. `label_pixels`/`blob_diff` gates and the host copy read word 1. The retry path can keep
  the fill-between-barriers pattern on word 1 (cold).
- **Expected gain (estimate from in-tree numbers):** Mali, at the deployment DMC clock:
  - `uf_final`: PERFORMANCE.md 3a gives 0.242 ms at 528 MHz and 0.091 at 2112 MHz; fitting
    t = C + K/f gives ~0.050 ms of bandwidth-proportional reads plus ~0.041 ms fixed (atomics,
    launch); A2 later took ~12% off at d2. Fusion removes the reads (4 B parent + 1 B thresholded
    per pixel, re-read only to be compared), the launch and one large barrier: **~0.04-0.06 ms**.
  - `uf_compress b`: PERFORMANCE.md 6c measured removing it outright at labelling -4.4% / GPU
    -1.4%, of which the guard already captured half: **~0.02 ms** left.
  - flag fill + B10/B11: a few us.
  - Total **~0.06-0.09 ms, ~2.5-3.5% of the Mali GPU phase**. RDNA4: `uf_final` ~0.016-0.03 ms,
    `uf_compress b` is already predicated off there, ~4 barriers at 1.6 us: ~0.015-0.03 ms
    (~5% of 0.33 ms, low confidence).
- **Cost and risk:** ~60 lines: a fused variant of `uf_compress` (adds the `thresholded` and
  `blob_size` bindings and a `count_mode` push constant), a `RecordFusedTail` flag to skip
  `uf_final`, and `RecordUfChunk` taking "first attempt" into account. Pre-recording keeps each
  variant cached. Risk: `uf_compress a` is a dependent-load walk and gains one atomic per pixel at
  its end; if Mali's LS unit is the limiter that could eat part of the saving. Bit-identity: the
  same saturating count over the same roots; `blob_size[r]` is "only reliable as a test against
  `min_blob_pixels`" by design (uf_final_body.glsl l.4-6), and that test is unchanged. Apply only
  when the recorded chunk is >= 2 (with `APRILTAG_VK_UF_CHUNK=1` every frame retries anyway).
- **Prior art:** research/03 item 6 ("fold the `uf_final` count into the final compress") was
  judged "likely a wash" because it "gives up the 6c skip of the converged compress". That
  objection does not apply here: the fused compress is the *pre-verification* one, which never
  skipped in practice (merge a always merges on a real image), and the converged compress is
  removed rather than skipped. PERFORMANCE.md s8 item 3 (removing the *verification merge* measured
  zero) is a different node and is left alone, because it is the speculation's safety net.
- **Verdict: pursue.** Measure: ABBA on the Pi, 12-16 rounds x 300, d1/d2/d4, timestamps off,
  GPU total and `pipeline_total`; `--dump-quads` byte-identical on all corpus sizes, `--max-uf 1`
  forced retries, and the four `--stress` frames (the retry path must still match).

### F3. Let the per-frame clears overlap the head of the chain (both; Mali first)

- **Idea:** B1 (CAT, `GpuDetector.cpp:699`) forces the 9 fills (1.3 MB, dominated by
  `blob_size` 1.0 MB and `hash_owner` 256 KB) to finish before `decimate` starts, yet no dispatch
  before `uf_merge a` reads or writes a filled buffer (F0). Two exact variants:
  - **(a) Defer the barrier.** Delete B1 and make B7 (after the post-init `uf_compress`) a CAT
    barrier. B2-B6 are COMPUTE->COMPUTE, so per the spec their first synchronization scope does not
    include the TRANSFER-stage fills, and a driver that honours stage masks may run the fills
    alongside `decimate` ... `uf_compress`. Two lines, no shader change.
  - **(b) Zero inside the post-init `uf_compress`** (1D over every pixel, so thread i writes
    `blob_size[i] = 0`, `i < 65536` also writes `hash_owner[i] = 0`, thread 0 the counters). That
    pass is a dependent-load walk (mean chain 7.6, max 199 at d2: research/03 sim), i.e. the one
    place in the frame where a fire-and-forget store stream has idle bandwidth to use. Removes all
    9 fills and B1. `TailRerun` keeps its fills (cold path).
- **Mechanism / numbers:** the `clear` span is 0.048 ms at DMC 2112 and 0.177 ms at 528
  (PERFORMANCE.md 3a): fitting t = C + K/f gives ~0.043 ms of pure DRAM write time and only ~5 us
  of fixed per-command overhead. So the prize is hiding the ~0.04 ms of write traffic under
  latency-bound work, plus one barrier.
- **Expected gain:** Mali up to ~0.04-0.05 ms (1.5-2%) if the overlap happens; (a) is zero if
  libmali drains everything at every barrier regardless of stage mask (unknown, and the point of
  measuring (a) first). RDNA4: `clear` is "~0.00 ms" (README), so only the barrier, ~1.6 us.
- **Cost and risk:** (a) trivial. (b) ~15 lines and extra bindings on `uf_compress`; risk that the
  extra stores slow the walk. Both bit-identical by construction (the zeroes land before their
  first reader through B7 / the existing chain).
- **Prior art:** none in the tree; research/07 BP-4 suggested a barrier audit generally.
  OPTIMIZATION_NOTES "not the per-frame buffer clears" (an old residual analysis) measured the
  clears at 0.08 ms but never tried to overlap them.
- **Verdict: measure first**, (a) then (b). Same ABBA protocol; also check `clear` disappears from
  the profile and `labelling` grows by less than it.

### F4. Fold the tail's small dispatches into their neighbours (both; desktop gains more)

All four are exact and independent of each other. Per-item savings are one dispatch launch plus
one barrier each, i.e. the "trivial dispatch" class: ~2.6 us on Mali, ~1.6 us on RDNA4.

- **F4a. `merge_extents` into `select_blobs`.** Both are indirect over the raw blob count with one
  thread per blob, and `select_blobs` reads only `extents[i]` for its own `i`
  (`select_blobs.comp` l.47). Fold the 8 private copies in registers inside `select_blobs` (the
  loop from `merge_extents.comp` l.24-35) and skip writing the canonical entry back (nobody else
  reads `extents_buf_`). -1 dispatch, -B23, and ~388 x 32 B of writes.
- **F4b. `init_extents` into `hash_group`'s slot-win path.** The thread that wins a slot already
  takes the dense id (`hash_group.comp` l.44-47); it can write the identity record to
  `extents[dense]` and its 7 private copies right there (skipping ids >= `max_raw_blobs`, as
  `reduce_extents_hash` already does). `reduce_extents_hash` runs after B19/B20, so the writes are
  visible. -1 dispatch, -B21.
- **F4c. `extract_blob_counts` + the 3-dispatch scan chain as one single-workgroup dispatch.**
  At the default `max_blobs` = 2048 and `scan_wg` = 1024 the chain is
  `scan_block(2048)`, `scan_block(2)`, `scan_add_offsets(2048)` (`RunInclusiveScan`, l.534-562):
  four dispatches and four barriers for 2048 integers. One workgroup (256 threads x 8 elements:
  sequential per-thread prefix, then a 256-wide shared scan; Arm BP s9.2 prefers <= 64-256 threads
  with barriers) reads `selected[i].count` (0 past `num_selected`) and writes the same inclusive
  scan, keeping the invariant that the grand total sits at `offsets[max_blobs - 1]`. Fall back to
  the chain when `max_blobs` exceeds a cap (it auto-scales to 8192 at 1080p d1). -3 dispatches,
  -3 barriers. Distinct from research/00 A12 (atomic range allocator), which changes the blob layout
  and the readback; this keeps both.
- **F4d. Drop the three 1-thread `build_*_args` dispatches.** Two ways:
  - *Grid-stride consumers:* dispatch `hash_group`, `reduce_extents_hash`, `scatter_index_points`
    (and `init/select` over raw blobs) with a fixed grid sized to fill the GPU and loop
    `for (i = gid; i < count_buf; i += grid)`; `sort_points_local` loops
    `for (b = WorkGroupID; b < n; b += NumWorkGroups)` (uniform per workgroup, so its barriers stay
    legal). The extents privatisation key becomes the virtual workgroup id (`i / wg & 7`) so the
    "same workgroup, same copy" locality that item 11 showed matters is kept. No indirect args at
    all: -3 dispatches, -B18/B20/B30/B31.
  - *Last-workgroup-writes-args* (CUDA's threadFenceReduction pattern; no spinning, so no
    forward-progress assumption): each producer workgroup does `barrier(); if (lid == 0 &&
    atomicAdd(done, 1) == numWG - 1) write args`, and the producer's existing barrier gains the
    DRAW_INDIRECT destination. Cheap for `select_blobs` (a few WGs), but it adds a `barrier()` to
    `blob_diff` and `hash_group`, which Arm BP s9.2 says prevents workgroup split/merge on Mali.
    Formally needs Vulkan-memory-model release/acquire (`vulkanMemoryModel` support on g24p0 not
    checked). Prefer grid-stride.
- **Expected gain:** ~8 dispatches and ~9 barriers in total. Mali: ~0.02-0.05 ms (1-2%) if they
  cost what the "trivial" figure says; desktop: ~9 x 1.6 us + launches = **~0.015-0.02 ms of a
  ~0.33 ms GPU phase (4-6%)**, which matters there because desktop single-stream throughput is
  GPU-latency-bound (PERFORMANCE.md 8e: 380 us of spans in a 611 us `Detect()`).
- **Cost and risk:** F4a/F4b ~20 lines each, low risk. F4c ~60 lines plus the fallback. F4d
  touches five shaders; grid-stride changes nothing in the arithmetic but changes the extents
  copy assignment (exact, merges are associative) and could expose a tail imbalance in `sort`
  (blob sizes vary 16..4096 points), so give `sort` enough workgroups (e.g. 2 x max blobs seen).
- **Prior art:** OPTIMIZATION_NOTES "Right-sizing `extract_blob_counts`" (rejected for a different
  mechanism: an indirect slot plus a tail fill to save ~2 us); research/05 A12; research/00 A11
  ("merge writes the next uf_compress's indirect args").
- **Verdict: measure first, desktop first** (that is where the per-barrier saving is a larger
  share). Order: F4a+F4b (smallest), then F4c, then F4d.

### F5. One counter block instead of 9 fills and 7 copies (both; small)

- **Idea:** the six 4-byte counters, the flag word(s) and any F4d "done" counters live in nine
  separate `VkBuffer`s, each zeroed by its own `vkCmdFillBuffer` and read back by its own
  `vkCmdCopyBuffer` (F0 rows 0, 12, 16, 28, 31). Put them in one 512-byte block (one 64-byte line
  per hot counter, so `blob_diff`'s 65k-append counter shares a line with nothing).
  - **Mali (unified memory):** allocate the block `HostVisibleCached`, like `selected_extents_buf_`
    already is, and read it in place after the fence (invalidate 512 B). That removes all seven
    copies, turns B14/B17/B29/B32 from CAT into C, and the final host barrier becomes
    COMPUTE -> HOST. Zero it in the post-init `uf_compress` (F3b) or with one fill.
  - **RDNA4:** keep it device-local (shader atomics over PCIe would be ruinous) but copy it once at
    the end: 7 copies -> 1. Have the scan (F4c) also write the point total into the block so the
    copy out of `blob_point_offsets_buf_` goes too.
- **Expected gain:** the fixed part of the 9 fills is only ~5 us on Mali (the `clear` fit in F3),
  and copies are similar; so ~5-15 us on Mali (<0.5%), a few us plus 3-4 lighter barriers on
  RDNA4 (~1-2%). Mainly a simplification that makes F3/F4 cleaner.
- **Cost and risk:** ~40 lines of host code; bindings change in every shader that touches a
  counter. Exact.
- **Verdict:** do it only as part of F3/F4, not on its own.

### F6. Fuse across the boundary stage: `label_pixels` into `blob_diff`, and the hash insert into `blob_diff` (both; measure first)

- **F6a. Drop `label_pixels` as a pass.** It exists to give `blob_diff` one packed word per
  pixel (`label_pixels_body.glsl`). `blob_diff` could instead read six 1-byte threshold codes
  first (the u8 buffer), exit unless some neighbour has the opposite code (true for a minority of
  interior pixels: ~65k points from ~254k pixels at d2), and only then read `parent[]` and
  `blob_size[root]` for the pixels involved. Today every interior pixel reads six 4-byte words.
  Removes a full pass that is 3.12x DMC-sensitive (~0.095 ms on Mali, PERFORMANCE.md 3a plus
  item 12), its large barrier, and the destructive in-place rewrite that forced the speculation
  gates (OPTIMIZATION_NOTES item 8). **Against it:** boundary pixels get a three-deep dependent
  chain (code -> parent -> blob_size) where they now have one load, and `blob_diff` is latency- and
  atomic-bound (1.09x DMC); the two rejected `blob_diff` rewrites (+11% on Mali, item 9) are a
  warning, though those reshaped work rather than removing a pass. Estimate: -0.03 to -0.08 ms on
  Mali, or a loss. Exact.
- **F6b. Insert into the grouping hash table from `blob_diff`** and delete `hash_group` as a pass.
  Today a losing `hash_group` thread must read the winner's key from `keys[]` (written by an
  *earlier* dispatch), which is why the insert cannot run inside `blob_diff`. Store the key itself
  in the table as one 64-bit word (`rep0`, `rep1` < 2^30 each; key 0 is impossible because
  rep0 < rep1, so a zero-filled table is empty) and insert with a 64-bit `atomicCompSwap`
  (`shaderBufferInt64Atomics`, present on g24p0 and RDNA4); a loser compares the CAS return
  value directly, so no other invocation's plain write is needed. The winner still takes the dense
  id. Removes the `keys` buffer (8 B/point written and read), the `hash_group` dispatch, B19 and
  the `build_qbp_args` dependency for grouping. Costs: the table doubles to 512 KB (more clear
  traffic, see F3), and `blob_diff` gains a hash probe per point. `hash_group` was 0.10 ms on Mali
  at 1080p d1 in the old README table, so perhaps ~0.03-0.05 ms at 1280x800 d2; the fused probe
  costs part of that back. Estimate -0.01 to -0.04 ms. Exact (dense-id order was already
  arrival-order).
- **Prior art:** none for either; OPTIMIZATION_NOTES' general lesson ("wins came from reducing
  work, never reshaping per-thread work") favours trying them, the item 9 losses argue for a bound
  first.
- **Verdict: measure first, after F2-F4.** Cheapest bound for F6a: a build where `label_pixels`
  is skipped and `blob_diff` does the lookups (it is a correct build, so `--dump-quads` checks it).

### F7. Cross-frame overlap and multi-queue scheduling (both; throughput only)

- **What exists:** `FramePipeline` overlaps the CPU tail of frame N with the GPU pass of N+1
  (PERFORMANCE.md s5). GPU-GPU overlap of two frames is measured through multiple streams:
  Mali 381 -> ~445 fps (+16%), desktop ~3,000 -> ~4,600 fps (+53%), with no gain from a second
  queue on Mali (8c: one family, two queues contend).
- **For one camera** the same gain is available by alternating two `GpuDetector`s (own buffers)
  on one lane from one `FramePipeline`: frame N+1's head fills frame N's bubbles. Cost: 2x device
  memory (~42 -> ~84 MiB at d2) and one more frame of latency.
- **Verdict: reject as a default.** A robot camera at 60-120 fps is nowhere near 381 fps, and
  what matters there is latency, which this cannot reduce. Keep it as a documented recipe for a
  throughput-bound user; multi-camera setups already get it. Its real value here is the ceiling in
  F1: it measures how much idle the intra-frame graph has (~0.25 ms on Mali).

### F8. Megakernels, persistent threads, GPU-driven scheduling, work graphs, DGC (reject)

- **Single persistent megakernel with a software grid barrier:** needs inter-workgroup forward
  progress, which Vulkan does not guarantee; research/05 already records the Mali-G77 failing
  LOBE and a hang being `DEVICE_LOST`. The frame is a 27-node chain of global dependencies (F0), so
  a megakernel would need ~25 grid barriers per frame. Reject.
- **Persistent / grid-stride workgroups without inter-workgroup sync:** safe, and the useful
  part of the idea is F4d (removing indirect-args dispatches). Nothing else in the frame needs it.
- **`VK_AMDX_shader_enqueue` (work graphs):** still provisional ("should not be used in
  production", registry rev. 2, 2024-07-17). `vulkaninfo` on this desktop (AMD proprietary 26.8.1)
  does not list it; no Mali driver has it. Even where present, work-graph nodes suit
  producer-consumer fan-out, while this frame's edges are mostly global reductions (select needs
  every extent, sort needs the whole scan), so each would still be a full-graph barrier. The total
  possible desktop gain is the ~0.06 ms bubble ceiling of F1, which F4 attacks portably. Reject.
- **`VK_EXT_device_generated_commands`:** exposed on this desktop, absent on proprietary Mali;
  research/08 already rejected it (cannot encode the labelling barriers; a 20x dispatch-storm
  slowdown reported on NVIDIA). Nothing new. Reject.
- **Conditional rendering on Mali:** not exposed on g24p0, which is why F2(b) removes the
  converged `uf_compress` statically rather than predicating it.
- Sources: https://docs.vulkan.org/spec/latest/chapters/executiongraphs.html ,
  https://gpuopen.com/learn/gpu-workgraphs-mesh-nodes-vulkan/ ,
  https://www.khronos.org/news/permalink/amd-releases-experimental-vulkan-vendor-extension-to-support-gpu-work-graphs

### F9. CCL / union-find literature since research/03-04, judged against 2-iteration convergence (both)

- **Search:** arXiv API (`connected component labeling` OR `union-find` AND `GPU`, newest first,
  25 hits to 2026-09-10) plus web searches for 2025-2026 GPU CCL and Burtscher-group union-find
  work. **No new dense-2D GPU CCL or GPU union-find technique appeared** since research/03's sweep;
  the only GPU entries are 2510.01592 (3D voxel labelling for legged robots) and 2412.11809 (sparse
  detector hits), both already judged in research/03. ECL-CC v1.1 is still the reference GPU
  union-find.
- **What 2-iteration convergence implies:** the pass count is already at its minimum (one real
  merge, one verification), so every round-synchronous or sampling method (SV, FastSV, Afforest,
  label propagation, Contour) is ruled out more firmly than research/04 argued: they can only *add*
  passes. The remaining levers are all **per-pass cost**:
  - the post-init `uf_compress` walk (mean chain 7.6, max 199 at d2): research/03 items 2 and 3
    (bounded segmented init, inline compression), still unmeasured on Mali;
  - the verification merge, which F2 keeps but strips of its compress, flag fill and barriers;
  - the `uf_final` pass, which F2 folds into the compress.
  F2 is the graph-level complement of research/03's per-pass items; they compose.
- **Connectivity (research/00 item 0.1):** upstream labels white 8-connected; vkapriltag 4/4. If
  parity is chosen, white gets two extra (diagonal) edge checks per pixel in `uf_merge`, so
  labelling gets *more* expensive unless it moves to frc971's 2x2-block BKE variant. That is a
  correctness decision first; from the graph side, it is the only change that could shrink the
  node count (4x fewer nodes with 2x2 blocks), and it is only available if white becomes
  8-connected.
- **Verdict:** no new algorithm to adopt. Pursue F2 plus research/03 items 2/3 on Mali.
- Sources: http://export.arxiv.org/api/query?search_query=all:%22connected%20component%20labeling%22 ,
  https://arxiv.org/abs/2412.11809 , https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/

### F10. ML-compiler-style auto-tuning and fusion search (both; low on Mali)

- **What transfers:** not the compilers. TVM, IREE and XLA tune and fuse dense tensor operators
  from a linalg/tensor IR; this pipeline is 27 hand-written integer GLSL kernels with atomics,
  hashing and pointer chasing, which none of them can express or fuse. CLTune and Kernel Tuner
  target CUDA/OpenCL/HIP, not Vulkan. What transfers is the **method**: a per-kernel search space
  of launch parameters, measured on the target, cached per device.
- **Practical path:** every workgroup size is already a specialization constant
  (`ComputePipeline.cpp` l.63-77). Add per-pipeline overrides (`APRILTAG_VK_WG_<shader>=WxH`), a
  sweep script driving `pi_bench.py`-style ABBA runs with `--profile` spans, and a small table in
  the library keyed by (vendorID, deviceID, driverVersion) holding the winners. Search space:
  1D {64, 128, 256}, 2D {8x8, 16x4, 32x2, 64x1, 128x1 for `uf_merge`'s one-row tiles}, plus the
  existing mode constants (`kFindMode`, `kMergeFlagMode`) and, on RDNA4, wave32/wave64
  (research/00 B2). ~10 kernels x ~5 options x ABBA = a few hours on the Pi.
- **Expected gain:** the global `APRILTAG_VK_WG` sweep on Mali found 128 "within noise of the
  best" (PERFORMANCE.md s6), and Arm's guidance says barrier-free shaders are insensitive to
  workgroup size because Valhall splits and merges them. So on Mali expect ~0-2%, mainly from
  `uf_merge`'s tile shape and `sort`'s 256. On RDNA4 the wave-size choice is the open part; the
  published OpenCL autotuning results (a median 3.79x for stencils over a fixed size) come from
  far less tuned baselines and do not transfer.
- **Cost and risk:** small code, real measurement time; a per-device table to maintain; no
  accuracy effect (every geometry is already verified bit-identical, PERFORMANCE.md s6).
- **Verdict: measure first, low priority on Mali**; worth it on RDNA4 only alongside B2.
- Sources: https://arxiv.org/abs/1703.06503 (CLTune), https://arxiv.org/abs/1511.02490
  (OpenCL workgroup-size autotuning for stencils)

## What was not verified
- The origin and method of the 2.6-18.7 us Mali / 1.6 us RDNA4 barrier costs (commit `0f16d8f`
  states them without a method). F1's split test is the fix.
- Whether libmali lets a TRANSFER-stage fill overlap compute work across COMPUTE-only barriers
  (F3a).
- `vulkanMemoryModel` on Mali g24p0 (true on the desktop per `vulkaninfo`), relevant only to the
  last-workgroup variant of F4d.
- Current per-span Mali numbers: the newest full table in the tree is PERFORMANCE.md 3a, taken
  before items 10-17; all Mali gains above scale from it and from the 8g totals.

