# Design notes

Design decisions and rationale removed from source comments. Measurements live in PERFORMANCE.md and OPTIMIZATION_NOTES.md.

## Shaders


### blob_diff.comp / blob_diff_body.glsl
- The `thresholded` binding is gone: label_pixels.comp folds the three-valued threshold into the spare high bits of the parent[] word, so blob_diff needs no 8-bit-storage axis. This collapsed the former four variants (8-bit x ballot) back to two (scalar / subgroup-aggregated append).
- Boundary points are appended directly to the compacted output with the shared atomic counter, instead of writing a dense 4-plane (width-2)*(height-2) QBPoint array (~66 MB at 1080p, mostly empty) and compacting it in a second pass (~132 MB extra traffic plus a 2M-thread dispatch).
- Reads are six spatially local loads per interior pixel (was twelve). The blob identity, the "big enough" test and the threshold value all come from one parent[] word, replacing six random gathers into blob_size[] plus two per emitted point into root_dense_id[], and then the parallel thresholded[] stencil.
- Append order is nondeterministic; safe because points are grouped by (rep0, rep1) with a hash table and MinMaxExtentsGpu::starting_offset is not consumed by the CPU tail.
- QBPoint.x/.y store the un-decimated coordinate (CUDA `base_x()*2+dx`), because downstream consumers (MinMaxExtentsGpu::cx()/cy() with 0.05118/-0.028581 sub-pixel offsets, line-fit moments, theta) assume the doubled convention.
- Dispatched 2D so (ox, oy) comes from gl_GlobalInvocationID.xy rather than a runtime `%`/`/` by the interior width (the hottest of the converted shaders, up to 4 appends per pixel).
- honour_changed_flag: on the fused fast path's speculative first attempt, if labelling has not converged parent[] still holds raw union-find pointers (label_pixels is gated the same way and has not run). PixelLabel/PixelThreshCode would extract garbage; it would not crash (downstream is clamped) but would waste a full pass appending garbage points, so the shader skips.
- rep_a/rep_b are raw union-find roots, not dense ids: grouping is a hash table, so keys only need to be distinct per blob. They are not part of QBPoint; nothing downstream read them from the compacted point.
- append() takes want_append as a value and is called unconditionally four times. This was originally load-bearing for the subgroup-aggregated variant (subgroup ops in divergent control flow dropped points on the desktop test GPU; see OPTIMIZATION_NOTES.md). With that variant retired it is only stylistic; guarding at the call site would be equally correct.
- The subgroup-aggregated (warp-aggregated atomic) counter variant was removed: it made the `boundary` span 24% slower on an RX 9060 XT and integrated parts never used it. Aggregating a bare counter is already handled well by atomic units. reduce_extents_hash_subgroup aggregates per-point values by key, a different and worthwhile trade (see GpuDetector::CreatePipelines()).
- keys[] is one interleaved uvec2 (single 8-byte store here, single 8-byte probe load in hash_group.comp) rather than two uint arrays.
- The `c0 == 1u` half of the early-out is already implied by `l0 == 0u` when min_cluster_pixels >= 2 (uf_init/uf_merge never join ambiguous pixels, so they are size-1 components); kept explicit so behaviour is unchanged at min_cluster_pixels == 1.
- The SW dedup test is pure equality on codes, preserved by any injective mapping of 0/127/255.
- The three gradient signs are computed unconditionally; monotonic codes make them bit-identical to the value-based ones even when unused.

### block_minmax.comp / block_minmax_u8.comp
- Result packed as (min | max<<8) in one uint so 8-bit storage support is not needed in the base variant; in the _u8 variant it stays packed because it is already 4x smaller than the per-pixel image (one entry per 4x4 block).
- Decimated width/height need not be a multiple of 4 (only the original image dimensions are guaranteed even). Edge-clamping (rather than bounds-check-and-skip) repeats the edge pixel, a harmless way to extend a min/max window past the border.

### build_indirect_args.comp
- Purpose: dispatch init_extents/select_blobs over the actual raw blob count rather than max_raw_blobs (65536 default vs a few hundred to a few thousand real blobs); the count only exists on the device, so building args here avoids a host round trip. Single invocation, negligible cost.
- groupCountX rounds up; consumers already bounds-check against max_raw_blobs (a push constant), so the shader only needs "large enough", not exact.
- Slots 1 and 2 let dispatches previously sized by host readback be issued in the same submission that produced the count, removing three mid-frame SubmitAndWait round trips (~0.47 ms GPU idle per frame on Mali-G610).

### common.glsl
- PackXY: 14 bits per axis bounds supported image size at 16383; GpuDetector's constructor asserts config width/height stay under it.
- Label/code word: folding the threshold into the label_pixels-written word lets blob_diff use six loads from one array instead of twelve from two, at the cost of one extra sequential read in label_pixels.
- Label bit budget: constructor rejects 2*(width/decimation) > 16383, so the decimated grid is at most 8191x8191 < 2^26; 26 bits needed worst case (1080p: 21 bits at decimation 1, 20 at decimation 2), 30 allocated. The host asserts this in CreateBuffers.
- Monotonic code mapping keeps blob_diff's rewritten tests identity-preserving: boundary test v0+vN==255 becomes c0+cN==2 (c0 in {0,2} forces cN = 2-c0, so the (1,1) collision is unreachable); gradient sign is preserved; SW dedup is equality.
- QBPoint: the original struct also carried rep0/rep1 and a validity flag; neither had a reader once the (rep0, rep1) key moved to its own arrays and later shaders only ran over valid points.
- MinMaxExtentsGpu omits starting_offset (no sorted array left to offset into; see reduce_extents_hash.comp) and rep0/rep1 (no CPU consumer).
- Privatised extents: reduce_extents_hash issues up to eight atomics per point into that point's blob accumulator; ~65k points over a few hundred blobs is ~170-way contention. Replacing them with plain stores (an incorrect build, to bound the payoff) cut the `extents` span 0.434 -> 0.062 ms and GPU total 3.43 -> 2.64 ms on Mali-G610. Replicating all max_raw_blobs entries would cost 8 x 3 MB, so only the first kPrivateExtentsBlobs (4096; real frames see 388 raw blobs at decimation 2, 1257 at decimation 1) are replicated; overflow falls back to the contended canonical slot (correct, just slower). Copy 0 being the canonical array leaves select_blobs, extract_blob_counts and the host unchanged.
- Field order: count and pxgx_plus_pygy_sum are incremented by every point, so they are packed into one 64-bit atomicAdd when VK_KHR_shader_atomic_int64 is available; reduce_extents_hash_atomic64.comp explains why this is exact.
- IPoint omits gx/gy (written by scatter_index_points.comp, never read; compute_line_fit_points.comp recomputes the gradient weight from the decimated image).
- RawLineFitPoint: largest per-frame readback (one entry per selected point); Mx/My/Mxx/Mxy/Myy are exact functions of (x2, y2, W), so only those plus blob_index are carried and the host rebuilds moments in int64. Two-word packing is exact because each field's bound is enforced elsewhere (W: |gx|,|gy| <= 255 gives sqrt(2*255^2) = 360.6, truncating to 360; the host static_asserts the 22-bit blob_index ceiling against max_raw_blobs). fp16 would not be exact: its 11-bit mantissa rounds integers above 2048 and x2 reaches 3839 at 1080p/decimation 1.

### decimate.comp / decimate_u8.comp
- Source packed four pixels per uint32 keeps the upload at one byte per pixel (the host used to widen every pixel to uint32 in a scalar loop, 4x the bytes) without depending on an optional device feature; unpacking by shift avoids VK_KHR_8bit_storage. Little-endian order is what memcpy of the source produces on every Vulkan platform.
- Dispatched 2D because Mali (Valhall) has no integer divide; recovering dx/dy via `%`/`/` cost a software divide per pixel.
- kDecimation is a specialisation constant (fixed per detector) so the multiply/divide by it is compile-time on devices with no integer divide (see GpuDetector.cpp).
- _u8: selected at runtime (GpuDetector::ShaderPath) on devices reporting storageBuffer8BitAccess; quarters decimated_buf_ and the traffic of decimate's write, block_minmax's read and sort_points_local's gradient sample. GrayImage stays uint32 because it is the host upload buffer, unrelated to the feature. No optional device feature may be assumed, so the 32-bit path is the default.

### extract_blob_counts.comp
- Runs in the same submission as select_blobs.comp, before the host reads num_selected_blobs, hence the count is read from the GPU counter buffer instead of a push constant.

### hash_group.comp
- Replaces a global (rep0, rep1) sort that existed only to make equal pairs adjacent. Nothing downstream needs point ORDER, only GROUPING (each selected blob's points are re-sorted by angle in sort_points_local.comp), so a single hash pass replaces 2 * ceil(bits/4) stable radix passes that each read and rewrote three words per element.
- Lock-free and spin-free: the losing CAS thread gets the winner's index back and compares keys immediately (the key was written by blob_diff.comp in an earlier, barriered dispatch). Vulkan does not guarantee independent forward progress between workgroups, so a spin waiting on another thread's later write could deadlock. A plain FillZero resets the table because 0 means empty.
- The unique CAS winner assigns the dense 1-based raw blob id with one atomicAdd, replacing mark_slots.comp plus a full scan over the large, mostly-empty hash table. Nothing needs ids in slot order, only distinct and 1-based, so this is a strict replacement.
- keys[] is one uvec2 per point: separate arrays meant two random gathers into two ~800 KB buffers (two cache lines) per probe. std430 gives uvec2 stride/alignment 8 and the descriptor is bound at offset 0, so it is one naturally aligned 8-byte load; blob_diff's append() likewise does one 8-byte store instead of two scattered 4-byte ones.
- Device-side count: the boundary-point total only exists on the GPU, and reading it back forced a mid-frame SubmitAndWait. Taking the bound from a buffer lets the dispatch be indirect in the same submission (GpuDetector's fused_submits_, build_indirect_args.comp); the push constant remains the fallback for the unfused path. Same applies to reduce_extents_hash_body.glsl.
- Hash mix: keys are dense small ids, so a plain xor/shift would cluster badly under linear probing.
- Dropping a point on probe exhaustion matches the defensive capacity clamps elsewhere; it is counted (drop_counter) rather than silent so an undersized table shows up in profiling.

### init_extents.comp / merge_extents.comp
- One invocation per blob writes the kExtentsCopies-1 extra entries, so the work scales with the frame's real blob count (indirect dispatch; a few hundred blobs, not ~94k replicated slots).
- Identity, not zero: min/max fold correctly only against +/-INT_MAX and sums against 0. A never-written copy contributes nothing, which also lets the subgroup reduction variant stay unprivatised and still merge correctly.
- merge_extents runs a few hundred invocations folding 7 entries each, versus the ~65k points x 8 atomics of contention it removes. Because min/max/sum fold associatively and exactly, the result is bit-identical to the unprivatised reduction regardless of how points were distributed across copies.

### label_pixels_body.glsl
- Folds blob identity, "blob big enough" and the threshold into one spatially local per-pixel word written back into parent[]. In-place is safe (each invocation reads/writes only its own entry, no extra buffer to allocate/zero/bind) and cannot leak across frames because next frame's uf_init overwrites every entry.
- Motivation: blob_diff used to do blob_size[parent[n]] for the pixel and five neighbours (six random gathers into a 2 MB array per interior pixel) plus two per emitted point for dense ids; that became one random gather here. Absorbing the threshold read then took blob_diff from twelve loads across two arrays to six from one, at the cost of one extra streaming read of thresholded[i] here.
- min_blob_pixels must match uf_final.comp's, which saturates its counter at the same floor; agreement makes the comparison exact (see uf_final.comp's proof).
- honour_changed_flag: the write is destructive (it discards the raw union-find parent pointer). If it ran before labelling converged, a retry's uf_merge/uf_compress would find()-walk a buffer without valid parent pointers, possibly a cycle that hangs the GPU (seen as VK_ERROR_DEVICE_LOST during development). Skipping leaves an incomplete but valid union-find structure that a retry can keep converging. See GpuDetector's finish_frame on the fused fast path.
- Label +1 bias lets label 0 mean "no usable blob", collapsing blob_diff's two rejections (ambiguous pixel, blob too small) into one comparison.

### label_pixels_u8.comp
- Thresholded is uint8_t; see decimate_u8.comp notes. Parent stays uint32 because it carries the packed word.

### reduce_extents_hash_body.glsl / reduce_extents_hash_atomic64.comp
- Same arithmetic as the original sorted-run reduce_extents; the raw blob index now comes from the hash slot because there is no sorted array (scatter_index_points.comp uses a per-blob cursor, and MinMaxExtentsGpu has no starting_offset).
- atomic64 variant is selected at runtime when the device reports shaderBufferInt64Atomics and shaderInt64.
- ExtentsU64 aliasing: the u64 view only reaches (count, pxgx_plus_pygy_sum) and the struct view only the other six fields, so there is no aliasing hazard.
- Read-before-atomic guard: atomicMin/Max with a value already >= (<=) the stored one is a no-op, so testing first is exactly equivalent and most of the four atomics are skipped. Worth 28% of the span. Unlike the guards OPTIMIZATION_NOTES.md warns about, this is the scalar path (the only one integrated parts take), where a cheap early-out and contention relief compose.
- 64-bit add exactness: adding `(uint64(uint(v)) << 32) | 1` increments the high half by v mod 2^32, identical to the 32-bit atomicAdd, and the low half by one. The low half never carries into the high half because count is bounded by boundary-point capacity (< 2^32). Nothing to unpack afterwards: the struct fields are those halves.
- gx/gy are structurally zero for two of four connection types (E is (+/-1, 0), S is (0, +/-1); only SE/SW carry both), so about half the points would issue an atomic RMW adding zero. Skipping is bit-identical and, unlike a saturating guard, tests a value already in a register.

### reduce_extents_hash_subgroup.comp
- Needs ARITHMETIC beyond what uf_final_subgroup.comp needs because it reduces per-point values, not just counts; SHUFFLE for the same runtime-lane-index reason as uf_final_subgroup.comp (see its subgroupBroadcast vs subgroupShuffle comment).
- Motivation: the scalar version issues 8 atomics per boundary point keyed by raw blob index (~1.27M at 1080p); this reduces all 8 across every lane sharing a key before touching memory.
- No partition/match-any extension is available on the target device, hence the leader-peeling loop. Worst case (every lane a different key) equals an unaggregated pass, so it cannot regress.
- Unlike uf_final.comp's raster-order-correlated pixels, points are indexed by blob_diff's atomic append order and are not obviously key-correlated within a subgroup; effectiveness was measured, not assumed (see OPTIMIZATION_NOTES.md).
- Ballot tracked as uvec4 because subgroup size can exceed 32 (RX 9060 XT reports 64).
- Shares the descriptor layout and push-constant block with the scalar variants (see reduce_extents_hash_body.glsl).
- No `!= 0` guard on gx_sum/gy_sum (unlike the scalar shader): there gx or gy is structurally zero for ~half of points, so the guard skips real atomics; here values are already reduced across a subgroup so zero sums are rare and the test is pure cost. Measured on an MX230 (extents span, min over 12 runs of 6 iterations): 0.2816 ms unguarded vs 0.2919 guarded (+3.7%). Same split as uf_final.comp / uf_final_subgroup.comp.

### scan_block.comp / scan_add_offsets.comp
- Block size is a specialisation constant because the former hardcoded 1024 exceeds maxComputeWorkGroupInvocations on Mali-G610 (512) and Vulkan only guarantees 128; the host picks the largest power of two the device allows.
- shared temp[1024] (4 KiB) is well inside the 16 KiB of shared memory Vulkan guarantees.
- Single-workgroup case is the top ("collapse") level of a multi-level scan, where the caller binds block_sums to the same buffer as output_values; with gl_WorkGroupID.x == 0 the block_sums write would clobber output_values[0], which already holds the correct scanned value.
- scan_add_offsets: workgroup 0 needs no offset; block_offsets is the inclusive scan of block totals (scan_block.comp run on block_sums one level up), so block_offsets[g-1] is exactly the exclusive prefix for block g.

### scatter_index_points.comp
- Replaces rewrite_index_points.comp without the sorted array. The cursor-based placement needs no separate output counter: every point reaching the atomicAdd on blob_cursor[sel] is one of exactly e.count points of that blob (counted the same way by reduce_extents_hash), so the total equals blob_point_offsets_buf_'s last inclusive-scanned entry from extract_blob_counts.comp's scan chain.
- Intra-blob order is atomic-arrival dependent, like the blob_diff append order, and is overwritten by sort_points_local.comp.
- Pseudo-angle: only the sort order around the perimeter is consumed, never the angle value. The "diamond" construction is monotonic in true angle (strictly increasing counterclockwise through the same quadrant crossings as atan2) for one divide and no transcendental: 0->1 across the first quadrant, 1->3 across dx<0 (-dy/d is already monotonic decreasing there), 3->4 across dx>=0, dy<0. Its zero point is the +x axis rather than atan2+PI's; the sort is circular so the seam position does not change the ordering FitQuadForBlob reads on the CPU.
- Key scaling: 20 bits so sort_points_local.comp can pack key and 12-bit local index into one 32-bit shared word. With the 4096-point ceiling that index implies, a full blob still gets ~244 key units per point, so ties stay vanishingly rare at any decimation.
- Device-side count: as in hash_group.comp.

### select_blobs.comp
- remap[] lets rewrite/scatter_index_points cheaply find where (or whether) each boundary point lands in the final list. The atomic counter is clamped so a pathological frame cannot overflow selected[]; max_blobs matches the original kMaxBlobs.
- Geometric prefilters (aspect_max/fill_min/fill_max): defaults are no-ops; see DetectorConfig::aspect_max/fill_min/fill_max for the derivation and corpus measurements.
- Polarity test: no GLSL `double`, since float64 (shaderFloat64) is optional and unsupported on Mali and most mobile GPUs, where using it fails pipeline creation outright. The CUDA original's dot() returns float anyway. Only the sign matters and sum2 is an exact integer, accumulated in int32 like the reduce atomics; a positive scale keeps the sign, so 2*dot clears the 0.5. fp32 rounding of sum2 (ulp ~64 at realistic magnitudes) is far below the correction terms, so it can only disagree with a float64 version for blobs essentially exactly on the polarity boundary.
- Bounding-box test history: the earlier `bbox area < tag_width` compared a pixel area against tag_width (in bit squares, 8 for tag36h11), a unit mismatch that rejected almost nothing. Zero width/height is a hard rejection as it would divide by zero below.

### sort_points_local*.comp
- Wrappers only: the _u8 sibling is chosen by GpuDetector on devices with storageBuffer8BitAccess, the 32-bit one otherwise.

### sort_points_local_body.glsl
- Per-blob local sort replaced a single flat radix/bitonic sort across all selected blobs (which needed a (blob_index, theta) composite key to stop blobs interleaving). Points of each blob are already contiguous (scatter_index_points.comp), so only theta_key within the blob's range is needed.
- The network operates over kLocalCap virtual slots independent of workgroup size (each thread owns kLocalCap/threads slots, strided), so per-blob capacity is sized from the shared-memory budget rather than the thread count; real blobs (e.g. a large tag's border) can exceed the device's max workgroup invocations yet fit in shared memory.
- Oversized fallback: blobs over kLocalCap keep original unsorted order (no lost points, no crash), but FitQuadForBlob consumes points as an ordered perimeter walk, so such a blob yields a geometrically meaningless quad and a tag whose border lands here becomes undetectable (this happened at decimation 1 before the cap scaled with decimation; see local_sort_virtual_cap_). The failure is silent and total, hence the OversizedBlobs counter. A handful per frame is normal (large background structures that pass select_blobs filters and fit junk quads that fail decoding); the counter matters when an expected tag goes missing. select_blobs shape/size filters do not bound point count, so ordinary scenes reach this path; the cap must give headroom above a real tag border at the configured decimation, hence derived from decimation rather than a constant.
- Fused with the line-fit moment computation (formerly compute_line_fit_points.comp): each thread knows which source point lands at its output position, so it samples the decimated image and writes the RawLineFitPoint directly, retiring the 2-3 MB index_points_sorted_buf_ and its traffic.
- The network is Batcher's odd-even mergesort (a bitonic network before). Same barrier-round count for a given cap (log2(cap)*(log2(cap)+1)/2) but ~13-21% fewer compare-exchanges across cap 16..4096. The (p, q, r, d) schedule (standard iterative Batcher construction) was verified standalone before porting: zero-one principle checked exhaustively for cap = 8 and 16, cross-checked against brute-force permutations at cap = 8 (8! cases) and 20000 random permutations each at cap = 32/64/128; q >= p confirmed at every step for cap up to 2^20, so the uint subtraction q - p never wraps.
- Sizing the network to the blob's own cap (not kLocalCap): count is uniform across the workgroup, so cap and log2_cap are too and all barriers are reached by every invocation.
- Key/index packing: 20-bit key + 12-bit index in one word (index is an incidental tiebreaker) halves the network's shared-memory footprint and per-compare traffic. 12 bits rather than 11 because perimeter in decimated points scales as 1/decimation: the ~1200-point border of a 1080p tag at decimation 2 is ~2400 at decimation 1, which overflowed the old 2048-slot ceiling and silently took the unsorted fallback. The host's local_sort_virtual_cap_ derivation only requests the larger ceiling for decimations that need it.
- max_blobs clamp: select_blobs' counter counts blobs that PASSED the filters and can exceed max_blobs while it drops the overflow, so the device-side bound clamps like the host did.
- Oversized-blob counter bumps once per blob because that is the number a caller can act on.

### subgroup_ballot128.glsl
- subgroupBallot returns a uvec4 (up to 128 lanes) and the RX 9060 XT reports subgroupSize=64, so set bits can span .x and .y. Callers that read only .x silently dropped lanes >= 32, a bug invisible on Mali-G610 (subgroup size 16).
- Does not declare GL_KHR_shader_subgroup_ballot itself to avoid a needless requirement in files with no other subgroup calls. FindLSB128 returns 128 for zero, but callers check AnyBits128 first.

### threshold.comp / threshold_u8.comp
- Dispatched 2D for the same reason as decimate.comp (no runtime `%`/`/` by decimated_width).
- block_width is passed explicitly rather than derived as decimated_width/4: decimated_width need not be a multiple of 4 (only source dimensions are guaranteed even), so the host's ceil-rounded GpuDetector::block_width_ is the only correct stride into minmax_filtered[].
- _u8: DecimatedImage and Thresholded change together because this is the one dispatch that reads decimated_buf_ and writes thresholded_buf_, so only one cannot cleanly be 8-bit. The three-valued encoding is unchanged; only the storage width differs. Selected at runtime via GpuDetector::ShaderPath on storageBuffer8BitAccess devices.

### uf_compress.comp
- Store guard: rewriting an identical value is pure write traffic. It matters most for the last compression pass of a chunk, whose only job is to observe convergence after a merge that changed nothing; the guard makes it a read-only chain check instead of a full 2 MB rewrite.
- Read guard (honour_changed_flag): if the preceding uf_merge joined nothing, parent[] is unchanged and was already flat (every chunk ends with a compression), so the pass cannot find work but would stream the whole array to learn that (1 MB at 1280x800 / decimation 2; the common steady-state video case where labelling converges in the first chunk). Reading the flag first reduces that to one scalar load per invocation.
- The flag is not honoured for the compression straight after uf_init.comp: no merge has run so the flag is still zero from the frame's clear, and skipping would be wrong and costly (OPTIMIZATION_NOTES.md item 8: that pass is worth 1.0 ms, since compressing between init and the first merge stops the vertical unions walking the run chains init just built). Hence a push-constant opt-in rather than an unconditional read.
- `original` need not be re-read because parent[i] is written only by its own invocation.

### uf_final.comp
- Was a faithful histogram (one unconditional atomicAdd per pixel, 518400 at 1080p) with worst contention where it hurt most: all pixels of a large tag border or background target one blob_size[] slot. blob_size[] has one reader, label_pixels.comp, asking only `blob_size[r] >= min_blob_pixels`, so once the floor is reached further increments are unobservable and a non-atomic read lets the contended case be a cache-hot load.
- Exactness proof. Let S(r) be the true pixel count of root r and M = min_blob_pixels; blob_size[r] only increments, at most S(r) times, from 0.
  - S(r) >= M gives final >= M: either nothing was skipped (final = S(r) >= M), or some thread skipped, meaning it read a value >= M, and the counter never decreases.
  - S(r) < M gives final < M: the counter never exceeds S(r) < M, so no load returns >= M, no thread skips, and final = S(r).
- The stored value underestimates blobs above the floor; the array is a saturating counter, not a histogram. Do not add a second reader wanting the true size without reverting this.
- Non-atomic read race: Vulkan's memory model calls it a data race, deliberately not marked coherent/volatile. A stale load can only return some previously written value, all <= S(r), so it only underestimates (an unnecessary atomicAdd, never a wrong answer) and both directions of the proof still hold. Coherent would force the load past L1 on every pixel, defeating the purpose, for freshness not needed. hash_group.comp already relies on a plain read of a concurrently atomically updated location (hash_owner[h] used as an index).
- min_blob_pixels must equal label_pixels.comp's or the shader saturates at a different threshold than the one tested and the predicate stops being exact.

### uf_init.comp / uf_init_u8.comp
- Building horizontal runs here replaces what the first uf_merge pass used to do for right-hand edges (with singleton input, unioning (i, i+1) hooks parent[i+1] = i via atomicMin since i < i+1). Directly it is a pure streaming write with no atomics, find() walks or contention, versus two find()s, an atomicMin and a retry loop per horizontal edge; horizontal edges are the majority of same-component pairs in a thresholded image. uf_merge is left with vertical edges only, halving its union work.
- Root convention preserved: a run's root is its leftmost pixel, the minimum index, as atomicMin hooking expects.
- "127 means I'm my own blob" matches the CUDA implementation.
- Dispatched 2D so the row-start test (x > 0) needs no runtime `%` by width (see decimate.comp notes); i is still the linear index into parent[]/thresholded[].
- _u8: Parent stays uint32 because it holds a pixel index that routinely exceeds 255.

### uf_merge_body.glsl
- Only the down edge is considered; with uf_init's horizontal runs the two cover every edge exactly once.
- Run-level merging: down edges between two rows are not independent. Wherever a run in row y sits above a run of the same value in row y+1, every column of the overlap requests the same union of the same two components (uf_init already joined each run), so only the leftmost column needs to perform it. Each redundant union costs two find() walks, the dependent global loads that OPTIMIZATION_NOTES.md's first pass identified as this stage's real cost. The start test costs two extra loads adjacent to ones already made, against a find pair and an atomicMin saved per interior overlap column.
- This is the transferable half of HA4 (Hennequin & Lacassagne), run-based 4-connected GPU CCL, without its warp intrinsics, which three measurements in this tree say are the wrong tool on both device classes here. See "A scan of the literature" in OPTIMIZATION_NOTES.md.
- Exactness: if v[i-1] == v[i] uf_init joined i-1 and i; if v[i-1+W] == v[i+W] it joined those two; once the overlap's leftmost column has unioned its pair, i and i+W are already in one component. Induction over the overlap gives the rest, the per-pass closure is unchanged, and hooking is still atomicMin so a root is still the component's minimum index.
- Convergence flag: previously `atomicAdd(changed_count, 1u)` per pixel pair with a matching neighbour (up to two per pixel, ~1M RMWs per pass on a single 4-byte address, serialising on one cache line and dwarfing the union work), and the result was never read (the host looped a fixed 64 times). Now the flag is set only when a union joined distinct components ("another pass needed") and aggregated in shared memory, so at most one global atomic per workgroup.
- 2D dispatch with 1 x N workgroups: x is needed for the run-start test, and recovering it from a linear index costs a division by a runtime width on parts without an integer divide (Valhall). One-row-tall groups keep consecutive threads on consecutive columns, so the two row streams stay as coalesced as under the 1D dispatch.
- Barrier rule: every invocation must reach both barriers, so bounds and ambiguous-pixel tests select work rather than returning early.
- _u8: Parent stays uint32 because union-find indices routinely exceed 255.

## GPU detector

### Types.h / QuadDecode.h
- MinMaxExtentsGpu drops the CUDA original's starting_offset and rep0/rep1 (no CPU caller reads them); IPoint drops gx/gy (written once, never read) and packs x/y.
- count and pxgx_plus_pygy_sum are adjacent so the GPU accumulates both in one 64-bit atomic.
- RawLineFitPoint packs four values into two words as a storage change only (no precision change); it is the largest per-frame readback, so halving it halves the copy and the sort_points_local writes.
- QuadFitScratch is slot-indexed, not thread_local: the library is dlopen()'d from a JNI shim and thread_local access from a freshly spawned pthread SIGBUSed on aarch64/glibc. It is a free struct so QuadDecode.cpp's file-local FitQuadForBlob can use it.
- CPU tail is scalar/branchy work on little data (a few thousand blobs/points), simpler to verify as C++ than as compute shaders; ports line_fit_filter.cu DoFitLines/DoFitQuads and apriltag_detect.cu UpdateFitQuads. It is the most expensive stage once the GPU is sized properly, hence the parallel per-blob fits (independent; collected in blob order for deterministic output).
- Decode takes a span so GpuDetector can pass a view into host-visible device memory instead of copying.
- The former selected_extents parameter of Decode was never read (the fit needs only blob_index and the array's run structure); GpuDetector::last_selected_extents is still populated.
- A high DP fallback rate (fallbacks/attempts) means DP costs without saving the combinatorial search, because callers fall through to it anyway.
- pool_ is a unique_ptr so a const Decode() can still hand work to the stateful pool.

### GpuDetector config defaults and limits
- decimation: baked into decimate.comp as a specialisation constant (id 3) so the compiler can fold `dx * decimation` into a shift; a smaller factor costs only the extra pixels. Sampling is one representative pixel per block (copy, not average), like CudaToGreyscaleAndDecimateHalide; upstream's 1.5x blended path is not implemented.
- min_tag_pixels is a recall/throughput trade: undersized noise/foliage blobs never reach the CPU quad fit; tags below the size are guaranteed dropped.
- aspect_max = 8: clears views up to ~82 degrees off-normal, rejects stringy blobs (text, foliage). Corpus: real tag aspect 1.0-1.02, background 1-28.
- fill_min/fill_max (0.5/2.5): real tag 1.00-1.01, background ~0.5-1.7 (1st-99th percentile); bounds deliberately wider than the observed spread because the corpus is one photographed scene.
- quad_fit_method default kDp: identical decoded ID sets and corner RMS differences in the third decimal (< 0.1 px, no systematic direction) at every decimation on both dev targets; an algorithmic win (skips the C(10,4) search), so it helps weaker CPUs more. ~8% of blobs fall back to kPeaks at decimation 2. A high fallback rate means DP costs without saving the search.
- max_raw_blobs/max_boundary_points are defensive caps, smaller than the CUDA implementation's dense worst case (tens of MB / ~400 MB at 1080p); max_boundary_points = 0 is the dense worst case and reproduces CUDA behaviour exactly.
- max_blobs = 0: scales with decimated pixel count, anchored at 2048 for 1080p at decimation 2 (the former flat value), so frames at or below that size are unchanged. Slack is cheap (one dispatch plus scan over capacity; the per-frame readback uses the actual count; +2.2 us for 24x capacity, 0.06% of a 3.6 ms frame; a 12 MP frame at decimation 2 qualifies 1878 blobs against 12042 resolved). Clamped to max_raw_blobs because selected blobs are a subset of raw blobs. Overflow is not graceful: select_blobs.comp drops in atomicAdd order, so detections become irreproducible (4032x3024 at decimation 1 returned 3 or 4 tags).
- uf_iterations_per_chunk = 2: uf_compress does full path compression each iteration, so one merge pass already resolves these shapes and the second only observes no change. At 1080p boundary points, blob count, point count and candidate quads are bit-identical for 2 through 64. Scenes needing more cost one extra chunk on the first frame; the adaptive seed (last_uf_iterations_) remembers the count.
- Constructor checks: boundary coordinates are the doubled decimated index (2*width/decimation), so a bare `width <= 16383` test is off by 2x at decimation 1; label_pixels' 30-bit label is implied by that (8191x8191 < 2^26); RawLineFitPoint's 22-bit blob_index and 10-bit W (W <= 361) mean max_raw_blobs <= 2^22, checked because failure would be silent mis-grouping.
- block_width_/block_height_ round up because width/height are only guaranteed divisible by decimation; block_minmax/threshold handle the ragged block (edge-clamped sampling / block_width push constant).
- local_sort_cap_ = min(pow2 floor of max workgroup invocations, 256): the bitonic network is sized to the blob (~256 slots typical), so 1024 threads idle three quarters of lanes. Mali-G610 1080p: 1024 threads 4.03 ms, 512 3.08, 256 3.08, 128 3.83, 64 5.71.
- local_sort_virtual_cap_ is decoupled from thread count so large blobs (a tag's own border) still get angle-sorted rather than hitting the unsorted identity fallback, which gives geometrically meaningless quads (FitQuadForBlob walks points as an ordered perimeter). Perimeter in decimated points goes as 1/decimation (~1200 at decimation 2, ~2400 at 1 for a 1080p tag), so the 2048 ceiling is scaled by 2/decimation but never below the anchor; coarser decimation keeps its measured cost. Hard cap 4096 = 12-bit local index in the shared word; Vulkan guarantees >= 16 KiB shared (this device 32 KiB), so the /4 budget is looser than the index cap.
- Env: APRILTAG_VK_MAX_POINTS caps boundary points (the main memory lever on unified-memory parts); APRILTAG_VK_UF_CHUNK: smaller chunks detect convergence more precisely at the cost of an extra round trip when they guess low.

### GpuDetector diagnostics (DetectProfile)
- oversized_sort_blobs: nonzero is normal (1080p: 4 at decimation 2, 3 at 1, 1 at 4 large background structures that fit junk quads). It is a silent total failure only if an expected tag border lands there; that is how decimation 1 was once broken (~2400-point border vs 2048 slots).
- selected_blob_drops: max_blobs does not otherwise scale with area, while blob count goes as area/decimation^2; contrast local_sort_virtual_cap_, which scales deliberately. Raising max_blobs costs max_blobs * sizeof(MinMaxExtentsGpu) plus a scan chain.
- hash_probe_drops is 0 for every real scene at the current table sizing; watch it if max_raw_blobs/table sizing is tightened.
- gpu_gap_ms: 3 of the 11 gaps cross a submit boundary (Labelling->UfFinal, Boundary->HashGroup, Scatter->Sort); the other 8 are intra-submit barriers. It exists to attribute the "unspanned" residual (cpu_submit_wait_ms - sum(gpu_stage_ms), ~1.5 ms at 1080p on Mali-G610). Findings: not per-submission latency (4 -> 3 submits moved it 0.05 ms, not the predicted 0.37; mostly-fixed cost); not the per-frame buffer clears (~2.3 MB fills, 0.08 ms, span "clear"); not mostly barriers (8 intra-submit gaps total ~0.04 ms); the 3 submit-crossing gaps show ~0.7 ms GPU-visible, the rest is CPU-side fence-wait wake latency/readback/driver overhead invisible to GPU timestamps. Further reduction must remove a submit boundary's fence-wait/readback cost, not fuse shaders.
- kNumGpuStages is the single source for gpu_stage_ms, gpu_gap_ms, kGpuStageNames, kGpuGapCrossesSubmit; previously four literals plus the enum's own count, so a missed edit went undetected. A static_assert ties the enum in (previously a silent out-of-range read).
- Span split: "clear" and "readback_copy" have their own spans because they are non-compute work that otherwise lands in the unattributed gap. uf_final and label_pixels are separate spans because optimisations to them move in opposite directions (atomics removed from uf_final vs work added to label_pixels), which a shared span nets to nothing.
- DetectProfile::gap_crosses_submit reports the layout that ran; with fused_submits_ the boundary->hash_group and scatter->sort boundaries disappear, and the static table would misattribute ~0.2 ms of barrier time to submissions that did not happen.

### GpuDetector buffers and submissions
- fused_submits_: removing the mid-frame round trips saved ~0.47 ms of GPU-clock idle per frame on Mali-G610; needs both direct-read paths, so discrete parts without a cached readback memory type keep the four-submit path.
- selected_extents_buf_ is readback-capable so its size is not needed at record time (a staged copy needs num_selected_blobs, which used to force a mid-frame SubmitAndWait); reading in place removes the last host dependency in the tail.
- line_fit_points_buf_ (~1.6 MB/frame at 1080p) used to be DeviceLocal copied to staging then memcpy'd (two copies on unified memory). Direct read requires host-visible AND cached: reading through an uncached mapping is ~4x slower on Mali (Buffer.h).
- last_line_fit_points is a span (not a vector) so a caller cannot silently undo the zero-copy path.
- qbp_keys_buf_ is sized to the point capacity (no longer a power of two for a bitonic network); interleaved uvec2 because the hash probe compares a whole key, and split arrays cost two random gathers per probe.
- Hash table sized to max_raw_blobs (previously 4x for a 25% worst-case load): a real 1080p frame has ~734 raw blobs of 65536, and an overflowing pathological frame degrades via the probe cap like every other clamp.
- parent_buf_ is repurposed after labelling as the per-pixel label so blob_diff needs no thresholded binding (cut the pipeline variants from 4 to 2; the u8 axis moved to label_pixels). blob_diff also performs the compaction that compact_qbp.comp used to do.
- extents_buf_ privatised accumulators: kExtentsCopies-1 extra copies of the first kPrivateExtentsBlobs entries (see common.glsl ExtentsSlot); ExtentsSlotCount() is the one place turning the constants into a size.
- init_extents/select_blobs dispatch indirectly from raw_blob_counter_buf_; before, they ran over max_raw_blobs (~6 MB wasted traffic), and computing the exact count on the host cost a round trip that made it a wash.
- blob_point_offsets_buf_[max_blobs-1] is used as the point total because entries past the selected blobs are zero-padded, so a static offset known at record time gives the exact total.
- Sort dispatch: rewrite_index_points.comp already packs each blob's points contiguously, so a one-workgroup-per-blob shared-memory sort suffices (no composite key or gather pass); it is fused with line-fit moments so each thread samples the decimated image and writes RawLineFitPoint directly.
- The pipeline cache is flushed in the constructor because a killed camera-loop process would otherwise lose it every time.

### GpuDetector pipeline notes
- Subgroup aggregation only for reduce_extents_hash (RX 9060 XT, min of 12, three sessions): uf_final subgroup 0.1035 vs scalar 0.0291 ms; blob_diff 0.0437 vs 0.0330 ms; extents subgroup 0.0702 vs scalar 0.1028 ms. It wins there because it reduces per-point values across lanes sharing a key (eight atomics per point -> eight per distinct key per subgroup); the retired ones only aggregated a counter, which discrete atomic units handle well. Consistent with OPTIMIZATION_NOTES.md (MX230: scalar with saturating guard beat aggregated 3x).
- Integrated GPUs excluded: Orange Pi 5 Mali-G610 reports all three subgroup capabilities and is bit-correct, but pipeline_total went 11.5 -> 18.1 ms (extents span 0.91 -> 6.04 ms). Same class as tile-local shared-memory union-find in OPTIMIZATION_NOTES.md: Valhall lacks dedicated hardware, so software ballot/shuffle sequences cost more than plain atomics. Windows/RX 9060 XT (subgroupSize 64) gains (best 1.40 -> 1.33 ms).
- SHUFFLE is needed because subgroupBroadcast's lane id must be a compile-time constant in SPIR-V; ARITHMETIC because per-point values (not just counts) are reduced.
- 2D dispatches so shaders recover (x, y) from gl_GlobalInvocationID.xy; Mali (Valhall) has no integer divide instruction. uf_merge uses a one-row-tall 2D workgroup so consecutive threads walk consecutive columns (coalesced) and x is available for the run-overlap test.
- uf_compress after uf_init: without flattening the run chains, vertical unions pay an O(run length) find() each, cancelling the run-based init's savings; honour_changed_flag = 0 there because skipping would drop the pass OPTIMIZATION_NOTES.md item 8 measured at 1.0 ms.
- The convergence flag is cleared before the LAST merge of a chunk; clearing once at the start would conflate "converged" with "changed earlier in this chunk".
- Transfer barrier after uf_compress: the default Compute barrier's dst stage is compute only, so a following vkCmdCopyBuffer was not synchronised. Invisible on the unfused path (an idle submission never ran the copy early); fusing gave the scheduler work to overlap, and APRILTAG_VK_UF_CHUNK=1 reproduced the race every time.
- Fused speculative path: the tail is recorded right after the labelling chunk assuming convergence (collapses the remaining submit boundary, PERFORMANCE.md 3c, on every video frame after the first); if it did not converge, the tail is discarded (not patched), merging resumes from parent[] (safe on a partially converged state), buffers are re-zeroed, and the timestamp pool is reset wholesale (Reset is all-or-nothing; rewriting an index without a reset is the QueryPool hazard), so that one frame's per-span breakdown is lost (wall-clock timing is unaffected). label_pixels_pl_/blob_diff_pl_ gate on honour_changed_flag so neither runs ahead of confirmed convergence.
- The "labelling" span ends after the first chunk; extra convergence chunks appear only in the wall-clock figure since a query cannot be rewritten without a reset.
- uf_iterations/uf_converged are set after the caller resolves the speculative result; setting them inside finish_frame would report the unverified guess.
- Timestamp profiling is opt-in (APRILTAG_VK_TIMESTAMPS=1) so a normal run pays for no query pool.
- Line-fit direct read: invalidate the range when the mapping is non-coherent, since skipping the copy must not skip Buffer::Read()'s invalidate.
- The uf_final min-size floor must equal label_pc's value, or its counter saturation stops matching the predicate.

### QuadDecode.cpp
- ReadMomentsWindow bug fix: the wrap-around branch used to read cs[index0 - 1] unconditionally, unlike the index0 < index1 branch (guarded by index0 > 0). index0 == 0 reaches it whenever ksz == 0 (blobs under 12 points), reading cs[SIZE_MAX]. Found while porting the window read to GPU (compute_window_error.comp has the mirrored guard). index0 == index1 falls into the wrap branch exactly as in the original.
- The three DP helpers are `inline` because MSVC /Ob1 (RelWithDebInfo; CMakeSettings.json x64-Release) inlines only inline-declared functions; without it quad_decode measured ~75% slower than /Ob2 (found while chasing an apparent regression that was two build types).
- FindDpCornerIndices: exact all-pairs diameter would be O(n^2), hence the 2-pass approximation; a triangle or rounded blob still yields some max-distance point per arc, hence the min_perp check (0.0001 * diameter_sq^2, a deliberately low bar: a real corner deviates by ~half the diameter). Consecutive entries just need to trace consecutive arcs; ReadMomentsWindow's wrap branch handles any consecutive pair.
- DP coordinates are approximate integer boundary coordinates, fine because they only seed indices.
- FitQuadForBlob takes a span; the caller previously built a fresh std::vector per blob (heap allocation and copy per blob, several hundred per frame).
- Windowed-error loop: circular indexing by conditional subtraction because the loop nest was issuing about a dozen integer divisions per boundary point.
- Decode scale: a plain multiply by decimation matches upstream apriltag.c. decimate.comp point-samples the top-left pixel of each block (as upstream's image_u8_decimate does), so index c maps to c*d. A `(c - 0.5) * d + 0.5` form is right only for area/box decimation. It was previously used here, giving a corner bias of (d-1)/2 pixels (0 at d=1, -0.5 at d=2, -1.5 at d=4); it survived because the corner check reports RMS without gating and the bias is invisible at d=1. tools/validate_pose_e2e prints the mean signed corner offset per tag because only a signed statistic separates a convention bug from zero-mean noise.

## Library


### WorkerPool
- Pool is persistent rather than thread-per-frame: spawning ~a dozen std::threads costs hundreds of microseconds, negligible against the old ~10 ms tail but not against the ~1.5 ms parallelised tail.
- Work is handed out by an atomic index, not statically partitioned: per-blob cost varies by more than an order of magnitude (the combinatorial quad search runs only for blobs with >= 4 peaks), so an even split would leave threads idle behind one straggler.
- `slot`-indexed scratch is preferred over `thread_local`: the library is loaded via dlopen() from a JNI shim (System.load()), and a dlopen()'d module's `thread_local` variables are not safely accessible from freshly spawned pthreads on every platform (observed SIGBUS/BUS_ADRALN on aarch64/glibc, even without concurrent access). Slots are captured by value in each worker, so no per-thread storage is needed.
- The calling thread also drains batches (slot 0) rather than blocking idle.
- ResolveThreadCount lives in one place so APRILTAG_CPU_THREADS means the same thing for both CPU phases (QuadDecode, TagDecoder), and one setting affects both.

### apriltag_family
- Thin OpenCV-free port of apriltags_cuda's setup_tag_family / teardown_tag_family / print_detections (apriltag_utils.cu).

### pgm_io
- No image-library dependency; mirrors main.cpp's DumpPgm writer.

### Embedded shaders
- SPIR-V is embedded so a library loaded from somewhere with no install prefix (e.g. a JNI shared object extracted from a jar, as PhotonVision does) needs nothing on disk. The disk path (SHADER_DIR) remains right for the sample app and validation tools, which run from a build/install tree.
- Generated by cmake/GenerateEmbeddedShaders.cmake from the same .spv files the install rules ship; with embedding OFF it emits an empty table so callers fall back to SHADER_DIR with no #ifdefs.
- The corpus digest keys the on-disk pipeline cache so a shader rebuild lands on a fresh cache file.
- ShaderSource is implicitly constructible from a path so path-based callers were unaffected.

### PipelineCache
- Purpose: avoid paying the ~30 vkCreateComputePipelines SPIR-V -> ISA compiles on every start; pay once per device per shader build.
- Keying on device IDs, pipelineCacheUUID and the corpus hash is not a correctness requirement (the driver validates the blob header and ignores unknown data); it only stops the cache directory accumulating dead entries.
- Save() skips the write when unchanged, so a warm run touches the filesystem zero times.
- Destroy must precede vkDestroyDevice because member destruction order would run after the device is gone.

### QueryPool
- One Reset() per frame suffices because every submission in a frame runs after the previous fence signalled (Context::SubmitAndWait). Unwritten indices (e.g. a stage skipped because no points survived compaction) are reported unavailable rather than blocking.
- count == 0 gives an inert pool so callers need no #ifdef on devices/builds without the instrumentation.

### ComputePipeline
- BarrierKind exists because the original port emitted the widest barrier (COMPUTE|TRANSFER both sides, all access bits) after every dispatch; with ~570 dispatches/frame that is ~570 full drains, most needing only shader-write -> shader-read ordering.
- Workgroup sizes are specialization constants (ids 0/1/2) rather than baked into GLSL because Vulkan only guarantees maxComputeWorkGroupInvocations >= 128 and parts differ (Mali-G610 tops out at 512, desktop at 1024); the host picks from device limits. Dispatch uses the pipeline's own size so host and shader never disagree.
- The descriptor set is written once because all GPU buffers are allocated up front and reused each frame.
- Extra specialization constants (from id 3) serve device-derived constants such as a shared-memory array length decoupled from the workgroup thread count.
- Zero-element dispatches record nothing because count-driven dispatch sizes make empty frames normal.
- Indirect dispatch needs IndirectDispatchBarrier (compute -> DRAW_INDIRECT stage); a plain Compute barrier does not cover it. See build_indirect_args.comp.
- The host-read barrier is needed since a fence alone does not make device writes host-visible in the Vulkan memory model.

### Buffer / MemoryKind
- MemoryKind (rather than raw VkMemoryPropertyFlags) keeps unified-memory fallbacks in one place.
- HostVisible (write staging): coherent so memcpy needs no flush; upload is insensitive to cache state.
- HostVisibleCached (read staging): COHERENT is not required because some devices (Mali-G610 on Orange Pi 5) have no type both COHERENT and CACHED; requiring both fell through to an uncached type, measured 4x slower for a 1.5 MB per-frame readback. Reads therefore invalidate explicitly when not coherent (Buffer::coherent_).
- DeviceLocalMapped exists on unified-memory parts and discrete cards with resizable BAR; DeviceLocalReadback falls back to DeviceLocal on discrete GPUs without resizable BAR, so a staging path must remain.
- host_cached() is load-bearing: uncached mappings are fine to write through but dramatically slower to read, so zero-copy readback must be conditional on it.
- Host-visible buffers are mapped once for their lifetime (mapping is not free; nothing benefits from unmapping).
- No VMA: a fixed set of long-lived buffers is allocated at startup, so raw vkAllocateMemory suffices; the key property is no per-frame allocation.

### Context / ContextOptions / DeviceCaps
- Env overrides let one binary target a different GPU (or tolerate a software implementation) without a rebuild, across discrete NVIDIA/AMD and embedded Mali parts.
- allow_cpu_device is off by default deliberately: a missing GPU ICD makes the loader return a CPU implementation that is functionally perfect but ~100x slower, indistinguishable from "the GPU port is slow" unless stated loudly.
- force_no_* flags isolate feature axes so they can be A/B'd independently (bisecting a regression, measuring one optimisation without the other).
- WG2D override is a testing aid for sweeping launch geometry (see OPTIMIZATION_NOTES.md workgroup-size item); max_invocations_override lets a desktop GPU exercise the geometry of a constrained part (Mali-G610 reports 512; Vulkan guarantees 128).
- Pipeline cache is keyed to physical device and shader corpus so it never serves stale data after a driver update or shader rebuild.
- DeviceCaps is collected once at startup so no hot path calls vkGetPhysicalDevice* queries.
- float64/int64 are reported only for diagnostics: Mali and many mobile parts lack them, so every shader avoids needing them. 8-bit storage IS used (default assumption of the codebase is the plain 32-bit-per-pixel shaders as fallback).
- Subgroup: core Vulkan 1.1, no extension needed; unlike 8-bit storage it is gated only by what SPIR-V may use. All three subgroup sites (uf_final, reduce_extents_hash, blob_diff counter) share one combined ballot+arithmetic flag for simplicity, even though blob_diff alone needs only ballot + broadcast + elect; ARITHMETIC and BALLOT are usually reported together.
- has_subgroup_shuffle: the reduce-by-key loops elect a leader lane at runtime (findLSB of a ballot); subgroupBroadcast needs a compile-time-constant id (SPIR-V OpGroupNonUniformBroadcast restriction; only Shuffle takes a dynamic id), so a fixed-lane broadcast cannot substitute.
- Launch geometry: Vulkan guarantees only maxComputeWorkGroupInvocations >= 128, so nothing may hardcode 256, let alone the 1024 the scan shaders once assumed (Mali-G610 caps at 512 and failed).
- Context requires only Vulkan 1.1 and no optional features/extensions so it runs on desktop NVIDIA/AMD/Intel, Mali/Adreno and Mesa software implementations.
- Pipeline cache is saved explicitly by GpuDetector after building pipelines so a killed (not cleanly shut down) process keeps the cache.
- SubmitAndWait waits on a fence rather than vkQueueWaitIdle, which serialises every queue on the device.
- CreateLogicalDevice runs before QueryCaps and is the only place that can request/enable the 8-bit storage extension.
- Submission count is exposed because it is worth watching per frame.
- FindMemoryType returns UINT32_MAX so callers with a fallback can test rather than catch.

### RefineEdges / TagDecoder
- Profiling on an RX 9060 XT put upstream's refine_edges() at ~55% of all CPU cycles in the pipeline, plus ~16% inside glibc modf() (called twice per interpolation step): the single most expensive stage, GPU phases included. Hence the reimplementations.
- kExact: modf(x,&i) is defined to return x - trunc(x) and both results are exactly representable, so it is bit-identical to kUpstream by construction while removing a non-inlinable libm call (which also stores through a pointer, blocking vectorisation) from the innermost loop.
- kFast: line-fit builds covariance from raw second moments (Cxx = Mxx/N - Ex*Ex) at image coordinates; Mxx reaches ~1e7 at 1080p and the subtraction is near-total cancellation, so float loses most significant digits. Only the inner loop (8-bit pixel inputs, single ratio Mn/Mcount output) is narrowed to float.
- kExact/kFast are reimplementations that must be kept in sync with apriltag.c's refine_edges; only kUpstream cannot diverge.
- TagDecoder mirrors GpuDetector::DecodeTags()/QuadDecodeTask() of the CUDA original (apriltag_detect.cu). Refinement runs immediately before quad_decode_index, the same order as upstream's quad_decode_task. td->refine_edges defaults to false; enabling it is the caller's choice.
- Pose estimation (apriltag_pose.h, needs calibrated camera matrix/tag size) is intentionally not wired up here.
- Parallel quad_decode_index matches upstream, which calls it concurrently from its own workpool taking td->mutex around the shared write (appending to detections). Per-quad scratch zarrays (per_quad_) make the merge happen in quad order rather than completion order, so output (including which near-duplicate reconcile_detections keeps) is bit-identical to the serial version regardless of thread count.
- `decimation` is needed because upstream's refine_edges() computes its per-edge search radius from td->quad_decimate and TagDecoder cannot learn the GPU decimation otherwise; the constructor sets it once and reads nothing else from that field.
- cpu_threads uses ResolveThreadCount, the same as QuadDecode, so the env var affects both CPU phases identically.
- per_quad_ entries are zarray_truncate'd and reused rather than destroyed and recreated.

### FramePipeline
- Serially the GPU idles during the CPU tail. Measured tail 0.36 ms vs 0.69 ms GPU pass on RX 9060 XT; 1.15 ms vs 3.9 ms on Orange Pi 5 Plus (Mali-G610). Concurrent frame cost is max(GPU, CPU): ~1.3x (AMD), ~1.25x (Pi).
- No device-side double buffering is needed: Detect() ends by copying both result payloads out of the readback buffer into last_selected_extents / last_line_fit_points; the only hazard is Detect() overwriting them next frame, so Push() swaps them into its own pair first. Detect() stays strictly serialised, so all device buffers, descriptor sets and command buffers keep single-threaded access.
- Throughput not latency optimisation: detections come back one Push() late. A control loop needing the newest frame immediately should call GpuDetector/QuadDecode/TagDecoder directly.
- Two frame buffers are required because the GPU reads gray_frame asynchronously; reusing the camera buffer back-to-back corrupts the in-flight frame.
- Selected extents are still exposed although QuadDecode no longer takes them: GpuDetector still populates them, and the detector's copy already belongs to the in-flight frame by the time a caller could read it.
- Extents ping-pong via swap to keep capacity (Detect() would otherwise reallocate every frame); points must be copied.

### PoseEstimator
- Pinhole only: neither this library nor the CUDA original implements undistortion; wide-angle callers should undistort corners first, which matters far more than any solver numerical detail.
- TagPose::error is the Lu 2000 object-space error (same scalar as libapriltag's estimate_tag_pose) and selects between the two ambiguity candidates. Convention verified against libapriltag via tools/validate_pose.
- PoseCandidates mirrors estimate_tag_pose_orthogonal_iteration so each half can be verified independently; EstimateHomography and EstimateOrthogonalIteration(iterations) are exposed so divergences can be localised (tools/validate_pose ladder; iterations = 1 isolates a single step).
- Fixed-size stack arithmetic exists because libapriltag's solver is built on matd_op(), a runtime string-expression interpreter that per call scans the expression, heap-allocates an argument array plus a 2*exprlen garbage array, parses character by character allocating intermediates, copies the result and frees everything. ~16 calls per iteration x 100 iterations = ~1600 interpreted evaluations per pose; cost is interpreter/allocator overhead, not the ~100k FLOPs. Measured on Mali-G610/RK3588: 1.309 ms per tag (libapriltag) vs 0.0175 ms for a 50-iteration solve here.
- CPU rather than GPU: ~1 us of GPU work but needs its own queue submission; a submit boundary measures ~0.46 ms (~0.19 ms GPU clock + ~0.27 ms host fence/readback). The 50 iterations are strictly sequential (each consumes previous R and t) and only 4 points exist to spread over a subgroup. GPU only overtakes a threaded CPU port past ~38 simultaneous tags.
- Iteration cap of 50 matches libapriltag so results are comparable.
- Early exit is a deliberate improvement over the port: libapriltag always runs 50 steps, computing a per-step error it never compares. The test is on the POSE, not the error, because near a minimum the error is quadratically flat in the pose, so an error delta of 1e-12 can hold while the pose is still ~1e-6 from converged.
- Default convergence_tol chosen from a measured sweep (tools/validate_pose, 210 synthetic poses, compared to the fixed-iteration solver and ground truth):

  | tol | rot vs fixed | |dt|/|t| vs fixed | truth rot | truth |dt|/|t| | iters saved |
  |---|---|---|---|---|---|
  | off | 0 | 0 | 0.006098 | 2.598e-06 | - |
  | 1e-12 | 2.7e-10 deg | 1.4e-12 | 0.006098 | 2.598e-06 | 20% |
  | 1e-10 | 1.7e-07 deg | 1.7e-10 | 0.006098 | 2.598e-06 | 33% |
  | 1e-08 | 2.8e-05 deg | 2.2e-08 | 0.006098 | 2.598e-06 | 62% |
  | 1e-06 | 2.2e-03 deg | 2.5e-06 | 0.006578 | 4.752e-06 | 90% |
  | 1e-04 | 3.6e-03 deg | 7.9e-06 | 0.006578 | 8.008e-06 | 96% |

  1e-8 is the last row with accuracy identical to the full 50 iterations (saves 62%). 1e-6 is where accuracy degrades (worst rotation error rises, worst translation error nearly doubles). Tighter than 1e-8 chases digits made meaningless by the input's own corner noise (~2.6e-06 relative); 1e-8 is ~260x below that floor. tol = 0 is the configuration verified for exact libapriltag parity.
- Translation scales linearly with tagsize, so a 1% error is a 1% range error, dwarfing numerical concerns.
- cpu_threads resolution is shared with QuadDecode/TagDecoder so the env var means one thing across every CPU-tail phase.
- EstimateAll is independent per detection; results are written in input order so output is identical regardless of thread count.

### FramePipeline.cpp / TagDecoder.cpp / RefineEdges.cpp
- FramePipeline destructor drains first because joining while Detect() runs would destroy members it uses; GPU errors on the last frame are swallowed because they are unobservable then.
- last_line_fit_points is a non-owning span (into the host-visible readback buffer or the detector's linefit_scratch_) overwritten by the next Detect(). A swap cannot take ownership of a view, so the pipelined path copies (the copy the in-place readback avoids); assign() keeps capacity so it is a memcpy after the first frame. Serial callers keep the zero-copy path; the cost is the price of overlap and far smaller than the CPU tail it hides.
- The patched apriltag entry points (quad_decode_index, refine_edges) are forward-declared as the original CUDA apriltag_detect.cu does; the library is otherwise unmodified upstream.
- Scratch arrays are emptied without destroying elements because every pointer they hold is, after the merge, also held by detections_ (sole destruction owner); destroying both would double-free.
- The reset/merge loops are bounded by quads.size() because per_quad_ never shrinks and entries past it hold destroyed pointers from an earlier, larger frame.
- refine_edges is safe concurrently across quads: it reads td_->quad_decimate (set once before ParallelFor) and writes only the task's own stack quad.
- Homographies (H, Hinv) are freed after quad_decode_index because upstream's apriltag_detector_detect destroys quads itself after the loop, so ownership lands on whoever supplied the quad. Skipping it leaked two matd_t per quad (four allocations, as matd_create callocs header and data separately) per frame: ~100 quads at 30 fps is roughly 2 GB/hour in a continuous camera loop. Unconditionally safe because quad_update_homographies leaves H/Hinv freshly allocated or NULL, and they are NULL on entry.
- Merge in quad order so the result, and which candidate reconcile_detections keeps on overlap, is bit-identical to serial regardless of thread count.
- RefineEdges: bilinear tap order is upstream's verbatim; reassociating changes low bits of g1/g2 and costs kExact its bit-identity. Only the innermost search loop uses `Sample`; everything else is double. atan2f/cosf/sinf are the float forms because upstream uses them despite surrounding double.

### vk/*.cpp
- QueryPool reads two 64-bit words per query (value + availability) so a query never written in a frame (skipped stage, e.g. no boundary points survived compaction) reports unavailable instead of blocking forever or racing WAIT_BIT against a query that will never complete.
- Buffer HostVisibleCached does not require COHERENT: requiring both COHERENT and CACHED on devices with no such type (Mali-G610) silently falls through to uncached memory, measured 4x slower for readback. DeviceLocalReadback: taking it on an uncached type would replace a fast device-to-device copy plus cached memcpy with an uncached read (slower), hence the host_cached() check and staging fallback.
- Shader code buffer is kept alive only until vkCreateShaderModule returns; embedded sources point straight at rodata.
- Flush/invalidate ranges are rounded outward (not clamped) so the whole requested range is covered.

### PipelineCache.cpp
- The corpus hash is not a correctness requirement (the driver keys on each pipeline's full create-info, so a stale entry is just a miss); it only keeps the cache directory from accumulating dead files across rebuilds.
- Cache blobs above a size cap are treated as corrupt: ~30 small compute pipelines never legitimately approach it.
- The FNV-1a framing detects a process killed mid-write so it is not handed to the driver.
- A blob passing our validation can still be refused by the driver (checksum-consistent corruption, driver quirk); fall back to an empty cache rather than failing detector startup.
- Directory scan is skipped when caching is off since the hash only names a file that would not be written.

### Context.cpp
- The device-extension check exists because extensions must be named at vkCreateDevice even when the driver apiVersion is new enough to have folded them into core; the project targets VK_API_VERSION_1_1, so VK_KHR_8bit_storage (core in 1.2) and its dependency VK_KHR_storage_buffer_storage_class still need naming.
- Only the storageBuffer8BitAccess sub-feature (byte buffer element access) is needed, not storageBuffer16BitAccess-style packing. Feature queries only need the physical device so are callable before device creation.
- Int64 atomics need the extension, shaderBufferInt64Atomics and core shaderInt64 (Int64 SPIR-V capability) for reduce_extents_hash_atomic64.comp; checked separately since Mali-G610 reports shaderFloat64 = false but shaderInt64 = true.
- WxH parsing failing falls back to automatic selection rather than silently picking 0x0.
- Instance API version 1.1: nothing uses a 1.2 entry point, and 1.1 keeps older mobile drivers (and older Panfrost) eligible.
- The no-usable-GPU message says exactly what happened because a silent CPU fallback destroys performance.
- Device is created against core Vulkan 1.1 compute with no optional features so Mali/Adreno parts lacking shaderFloat64/shaderInt64/8-bit storage work. 8-bit-storage variants (uint8_t decimated_buf_/thresholded_buf_ instead of a uint32 per pixel) exist purely for memory-traffic wins, with the 32-bit variants as the default/fallback. Int64 atomics follow the same shape.
- features2.features is zeroed but routed through the pNext chain because the spec requires VkDeviceCreateInfo::pEnabledFeatures be NULL when VkPhysicalDeviceFeatures2 is chained. shaderInt64 is the only core optional feature ever enabled, only when a chosen shader variant needs it.
- Subgroup properties need a separate VkPhysicalDeviceProperties2 query since vkGetPhysicalDeviceProperties has no room for extension structs.
- Launch geometry: 1D default 256 on desktop, 128 on Mali-class; bitonic_local.comp stages three uint32 arrays (one element per invocation) in shared memory so the workgroup must fit that budget. Scan block: bigger means fewer levels but must fit invocation limit and shared memory (4 bytes/element).
- 2D on integrated GPUs is 8x8 unconditionally: on Mali-G610 (Orange Pi 5), across decimate/threshold/uf_init/blob_diff (driven by wg2d_ since the A6 move to 2D dispatch), 8x8 beat 16x16 by ~1-2% on pipeline_total over three repeated A/B runs (median 11.616/11.772/11.618 ms vs 11.772/11.808/11.778 ms). A wg1d_ sweep (64/128/256/512 via APRILTAG_VK_WG; uf_merge/uf_compress/uf_final/hash_group/reduce_extents_hash/select_blobs/extract_blob_counts/scatter_index_points/label_pixels) showed no signal within ~0.2 ms noise, so the 128 default for integrated is unchanged. The WG2D override needs no validation because ComputePipeline throws if it exceeds real limits.
- Pipeline cache destroy must precede vkDestroyDevice because member destruction order runs after the function body.
- Fences are created signalled so the first BeginCommands() wait is a no-op.

### PoseEstimator.cpp
- SVD is genuinely required (not a polar-decomposition shortcut): orthogonal iteration's M3 = sum_j (q_j - q_mean) * p_res_j' is always rank deficient because the four tag corners are coplanar (every p_res_j has zero z, so M3's third column is zero). A Newton polar iteration needs M^-1 and collapses to a zero pose on this input. Jacobi on A = M'M gives V and squared singular values; U columns are M*v_i / sv_i; the column for a zero singular value is recovered as the cross product of the other two to keep U orthogonal. matd_svd sorts by descending magnitude and folds the sign into U.
- With more than one rank-deficient direction, later U columns would be filled from earlier ones: still orthogonal, arbitrary, the same freedom as matd_svd.
- Horner instead of libapriltag's polyval (one libm pow() per term): that ran inside a 100-iteration Newton loop inside the recursive root finder, the single largest avoidable cost in the file. Horner is better conditioned, so roots differ from libapriltag in the last bits (see tolerance discussion in tools/validate_pose).
- solve_poly_approx is approximate by construction (drops roots beyond kMaxRoot, fixed 100 iterations, no convergence assertion, unconverged root returned); preserved deliberately since changing it would change which pose comes out. An all-zero linear polynomial yields a non-finite root as in libapriltag.
- Root-finder buffers are stack-based because degree is at most 4 (quartic from FixPoseAmbiguities). They are zero-initialised only to silence GCC's warning (only [0, n_der_roots) is read).
- Orthogonal iteration differences from libapriltag are pure hoisting: (F_j - I) built once (libapriltag re-derives it inside matd_op each iteration), R*p_j computed once per point per iteration and reused by translation, rotation and error steps (libapriltag recomputes three times). Early exit (tol > 0) is the one deliberate behavioural change.
- Convergence compares against the previous step's pose so the first step always runs (nothing to compare; solution 2 enters with t unset). R is orthonormal so an absolute bound is scale-free; t is in metres so bounded relatively.
- std::max is avoided in library source because MSVC windows.h min/max macros (pulled in transitively) would break it unless every consumer defines NOMINMAX.
- Normalisation rejects zero vectors that libapriltag's matd_vec_normalize walks into silently; the Gram-Schmidt step is degenerate when t is parallel to e_x and is rejected.
- libapriltag builds R_t rows by reading MATD_EL(R_t_1, 0, 1) and (0, 2) off a 3x1 column vector (outside its declared range but landing on the right flat offsets in row-major storage), i.e. the vector's components laid out as a row; that intent is what is ported.
- homography_to_pose: libapriltag computes the scale with single-precision sqrtf, perturbing the seed by ~1e-7 relative; since orthogonal iteration is a local optimiser this can occasionally steer it into the other basin of the planar-pose ambiguity, which is why comparison against libapriltag is a tolerance, not an equality. Here the scale is double precision.
- The homography seed's polar decomposition takes the orthogonal factor with no determinant fixup (unlike orthogonal iteration's use), preserved from libapriltag. Camera looks along -Z, so the sign of s requires the tag in front.
- EstimateAll tasks each write one entry by index, so no synchronisation is needed and output is independent of scheduling.

### apps/apriltag_vulkan
- V4L2Capture replaces OpenCV's VideoCapture (used by the CUDA original); YUYV 4:2:2 is requested because nearly all USB/UVC cameras support it natively, and only luma is needed by the detector.
- Capture retry is a simple blocking loop to keep the file free of extra dependencies (rather than select()/poll()).
- main.cpp: TagDecoder is constructed after `config` because its decimation must match config.decimation. A resolution mismatch fails loudly because reflecting it would require reconstructing the detector.
- Pose estimation (apriltag_pose.h, needs calibrated camera matrix/tag size) is out of scope of the app; see README.md for scope reductions versus the CUDA original.

## Tools, build and CI

### Build system (CMake, cmake/*.cmake, CPack)
- Shaders target Vulkan 1.1, not 1.2: nothing needs 1.2 features, and 1.1 keeps older mobile drivers (Mali/Adreno, older Panfrost) eligible. Shaders avoid all optional device features (no shaderFloat64/shaderInt64/8-bit storage) so the SPIR-V loads unmodified on desktop and mobile. Workgroup sizes are specialisation constants chosen at runtime from device limits.
- Shared shader headers are globbed as dependencies because glslangValidator's dependency info is not wired in; without it, editing a shared header leaves every `.comp` that `#include`s it silently stale (Ninja/Make see no timestamp change).
- Embedded shader corpus lets a deployed binary need nothing on disk (e.g. a JNI shared object extracted from a jar, as PhotonVision consumes native code). Both embedded and disk-fallback paths stay compiled; selection is at runtime via `vk::HasEmbeddedShaders()`. With EMBED=OFF the generator still writes an empty table so the build graph and call sites are identical.
- GenerateEmbeddedShaders: the file list is `|`-separated because a `;` list passed via `-D` splits into separate arguments. Files are sorted so emitted order and corpus digest are stable. The regex spells out the group repeat because CMake's regex has no `{n}` (it silently matches nothing). Per-file digests feed the corpus digest since hashing the ~0.5 MB of hex directly is slow. Lines are wrapped because one multi-hundred-kilobyte source line is legal but poorly tolerated. `alignas(4)` is load-bearing: `pCode` is a `const uint32_t*`. An empty corpus emits one inert entry (zero-length arrays are ill-formed) with `kShaderCount` 0. OUTPUT is written unconditionally: skipping the write when contents match would leave it older than its inputs and re-run the generator every build.
- ApplyPatches.cmake: patches are comma-separated because PATCH_COMMAND is stored as a semicolon-joined CMake list by FetchContent/ExternalProject and an escaped `;` does not survive the round trip (it splits into two arguments and silently drops the second patch). It runs via `cmake -P` rather than a chained shell line so idempotency does not depend on cmd.exe vs sh `&&`/`||` grouping. Already-applied patches (a stale reconfigure re-running PATCH_COMMAND) are skipped.
- Upstream apriltag is patched to extract the per-quad family decode and the cross-family duplicate reconciliation from `apriltag_detector_detect()` into `quad_decode_index()`/`reconcile_detections()`; behaviour of `apriltag_detector_detect()` is unchanged. The pure-C library has no CUDA/OpenCV dependency and runs the final CPU decode, as the original CUDA implementation did.
- pthreads_cross patch: upstream assumes any `_WIN32` target lacks pthread.h; MinGW-w64 ships winpthreads, so the emulation typedefs conflict with MinGW's `<pthread.h>` (pulled in via libstdc++ gthr-default.h). The emulation is scoped to "_WIN32 and not MinGW".
- BUILD_EXAMPLES is forced off because upstream defaults it ON and installs apriltag_demo/opencv_demo, which would leak into install/CPack output and add a spurious OpenCV runtime dependency to packages on hosts with OpenCV dev files.
- C standard pinned to gnu17: GCC 15+ makes implicit/incompatible pointer-type calls (e.g. `zarray_vmap(..., free)` in common/getopt.c) hard errors.
- Top-level NOMINMAX/WIN32_LEAN_AND_MEAN: `<windows.h>` (pulled in by apriltag's pthreads_cross.h and the Vulkan headers) defines min()/max() macros that break std::min/std::max; set before add_subdirectory so it also reaches the FetchContent'd apriltag. The policy floor is needed because the fetched apriltag declares a very old cmake_minimum_required().
- CPack scope matches the library install() rules and the release workflow (-DVKAPRILTAG_BUILD_APPS=OFF -DVKAPRILTAG_BUILD_TOOLS=OFF). CPACK_PACKAGE_VERSION is only defaulted when unset because a plain set() would shadow the command-line cache value the release workflow derives from the git tag. CPACK_DEBIAN_PACKAGE_ARCHITECTURE is left unset so it is auto-detected via `dpkg --print-architecture`; correct only if packaging is always native (never cross-compiled).

### CI workflows
- release.yml: native arm64 build (ubuntu-24.04-arm, free on public repos) instead of cross-compilation, because cross-linking Vulkan and FetchContent's apriltag build would need multiarch apt sources for no benefit. Tags are strict vMAJOR.MINOR.PATCH (no -rc/build metadata), failing fast before build work. libopencv-dev is omitted since only tools/ use OpenCV. The .deb contains the library scope only (headers, static lib, SPIR-V corpus, CMake package config).
- vkapriltag-parity.yml: runners have no GPU, so Mesa lavapipe/llvmpipe is used; lavapipe was confirmed to report all needed features (shaderFloat64, storageBuffer8BitAccess, subgroup ballot+arithmetic) and to reproduce bit-identical output to real AMD/Mali hardware for synth_pose_ground_truth. APRILTAG_VK_ALLOW_CPU=1 overrides vk/Context.cpp's refusal of software devices (50-500x slower than hardware).
- The lavapipe ICD path is discovered with `find`, not hardcoded: a hardcoded `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` failed on the ubuntu-latest runner ("Failed to open JSON file") because the layout differed from the published noble package listing (likely noble-updates). Discovery fails loudly and self-diagnosingly on future Mesa packaging changes. The follow-up vulkaninfo step fails clearly if the ICD does not work, instead of confusingly inside the tool's Vulkan setup.

### make_corpus
- Purpose: Phase 0 of the optimisation plan; produces different effective tag pixel sizes to sweep item-2 geometric-prefilter thresholds (min_tag_pixels, aspect_max, fill bounds).
- Caveat: rescaling the whole frame is only a proxy for "tag is farther away" (clutter, noise and blur shrink proportionally too); good enough for checking resolution-relative thresholds and gross regressions, not a substitute for real multi-distance captures.
- Downscale only: upscaling synthesises pixels the source never had (smoother/different edge profile), so it is not a valid stand-in for distance and must not validate item-2 thresholds or item-3 corner-fit changes.
- Dimensions round to even because the OpenCV validate tool skips odd-sized images (decimation halves each dimension). INTER_AREA avoids aliasing when shrinking.

### validate_against_libapriltag (+ validate_common.h)
- Purpose: the "verify against official libapriltag outputs" check, not an eyeballed comparison; hence upstream's default refine_edges = true is matched on our side.
- Phase 0 harness needs: (a) time the whole pipeline (GPU + quad_decode + tag_decode) per iteration since items 2/3 are mostly CPU-tail changes invisible to a GPU-only timer; (b) compare per-tag corner positions, not only ID sets, since item 3 (DP corner seeding) can regress corner accuracy with unchanged IDs.
- Repeated iterations because a single Detect() is dominated by cold-start costs (first-touch page faults, pipeline warm-up); best/median approximates live-camera steady state. Decimation is kept in sync with td_ref->quad_decimate so the "verified against unmodified upstream" claim holds at every tested decimation.
- max_blobs override exists because a frame overflowing max_blobs detects a different tag set each run; raising it until selected_blob_drops reads zero confirms that cause.
- FramePipeline mode is throughput-only (GPU pass and CPU tail overlap, so per-stage vectors stay empty); pushing one buffer repeatedly is safe only because both consumers read it (a live camera must alternate two buffers).
- The GPU span breakdown lives in validate_common.h because the PGM-only build (used on the Mali deployment target, which usually lacks OpenCV) previously printed only the five coarse spans, so profiling there lacked per-dispatch attribution. The host-submission cost is printed alongside because the submit-boundary gaps are GPU idle waiting on exactly that work; comparing them tells whether to chase submits or barriers.
- The residual is deliberately not divided by the submit count: the "cost per round trip" reading was disproved by removing a submission; it is GPU-side time no span covers, dominated by inter-dispatch barriers.
- Exit is non-zero on mismatch and when zero images validated (mistyped --data path, or every image skipped) so scripted runs cannot report success having checked nothing.
- Corner comparison assumes matching corner index correspondence (both detectors derive p[4][2] as homography-refined corners in a fixed winding order), so no nearest-corner search is needed.
- DP counters: a high dp_fallbacks/dp_attempts ratio means DP adds cost without saving the combinatorial search's. CSV fields are unescaped since they are only filenames and numbers.

### synth_pose_ground_truth
- Why: a real photograph has no known pose, so validate_pose_e2e's mutual delta cannot say which side is closer to correct (an 8-degree disagreement is consistent with either side being off).
- Rendering is built from the family's own bit_x/bit_y/codes tables so the tag is decodable by construction and errors are the detector's, not rendering-convention artefacts. `apriltag_to_image()` is unsuitable: it draws a 1px-per-cell perimeter-outline debug thumbnail, not solid cells a camera could photograph. Convention matches quad_decode_index's sampling (sample above threshold sets the bit). tag36h11 has reversed_border == false; reversed-border families are untested elsewhere.
- Per-case pipeline: choose a pose and project the border-square corners (same projection as validate_pose.cpp's MakeSynthCase); solve the exact 4-point homography from bitmap corners (ground truth, not a fit) and warp onto a flat background; run both pipelines as validate_pose_e2e does; compare each against truth (pose and corners in pixels), plus the mutual delta.
- Object +y increases with pixel row (pinhole projection with no sign flip, like bit_y), so obj (-1,+1) maps to the bottom bitmap corner. Getting this backwards renders a vertically mirrored tag; warping still succeeds arithmetically and 100% of frames fail to decode on either side (confirmed by isolating this swap).
- Blur and read noise are added because a crisp noise-free warp can hide last-bit corner-fit ties (e.g. an exact 0.000 deg rotation delta from bit-identical corner reads on a real photograph in validate_pose_e2e's field2 outputs). The RNG seed is fixed because cv::randn() uses OpenCV's global RNG: two runs (real AMD, then lavapipe) were bit-identical, but this tool gates CI and must not depend on an implicit default that may change across OpenCV versions.
- The sweep straddles ~18-50 px, the regime where validate_pose_e2e found its worst real-corpus deltas, plus larger tags; distance is solved from target pixel size at tilt=0 so the sweep is expressed in sensor pixels. Bucketing by truth pixel size follows the real corpus' error dependence. Bitmaps use 32 px/cell so downsampling is always a minification for INTER_AREA.
- VERDICT thresholds gate on the mean, never the worst case: the worst mutual delta is dominated by legitimate pose-ambiguity branch flips at small/oblique sizes (measured up to ~124 deg), which would make the CI gate flaky on correct code.
- OpenCV 5 moved getPerspectiveTransform() into a geometry module not pulled in by opencv.hpp; the include is conditional since older versions lack the header.

### validate_pose
- libapriltag exposes each stage publicly (estimate_pose_for_tag_homography, estimate_tag_pose_orthogonal_iteration with both solutions, both errors and a settable nIters), so each half is checked on its own and a divergence localises to a stage; no libapriltag patch is needed. Ladder: L1 seed, L2 solution 1 + err1, L3 solution 2 + err2 (exercises fix_pose_ambiguities), L4 final pick.
- The synthetic sweep additionally gives absolute error against ground truth, catching the case where both implementations agree and are both wrong. Degenerate geometry: t parallel to e_x, fronto-parallel, extreme range, tiny tag, near-collinear corners. Real detections (--data) keep H matrices in-distribution.
- Rotation angle uses atan2(|axis|, cos-part), not acos((trace-1)/2): acos has an infinite derivative at 1, so a single-ULP trace error becomes ~4e-8 rad (~2.4e-6 deg) of phantom difference, a noise floor visible even when comparing a solver against itself.
- kErrFloor: synthetic cases are fitted exactly, so both error scalars land near zero (~1e-13) and their ratio is noise; relative comparison is applied only above the floor.
- A differing number of ambiguity minima is a legitimate outcome of tiny numerical differences, so it is counted rather than failed. A branch disagreement is benign when the two errors are effectively tied.
- convergence_tol = 0 keeps libapriltag's full 50 iterations so the ladder compares like with like; the early exit is a deliberate deviation measured separately. Timing runs both `est` (parity setting) and `early_est` (default tolerance); the gap is what the early exit buys.
- EstimateAll writes results by index, not completion order, so output must be bit-identical to serial Estimate() at any thread count; "plausible poses" would not catch an ordering bug. The serial reference uses the shipping configuration, single-threaded, to isolate batching.
- Early exit (Phase B): libapriltag always runs the full 50 steps and is the wrong oracle, so it is checked against the fixed-iteration solver and against ground truth (agreeing with the fixed solver means nothing if both drifted). The tolerance is swept so the default follows the measured iterations/accuracy tradeoff. Wall clock is measured rather than iteration count since the seed, per-point F precompute and the quartic ambiguity solve are fixed overhead.
- Early-exit gate: a fixed absolute bound on deviation from the fixed solver would just restate the tolerance, so instead: (1) no accuracy regression against ground truth, (2) deviation an order of magnitude below the accuracy limit the input imposes, plus deviation scaling with the tolerance (catches too-early exit).
- Verdict thresholds are calibrated to measured divergences: the L1 seed differs only because libapriltag computes homography_to_pose's scale with single-precision sqrtf while this port uses double (~1.4e-7 relative translation), so L1 gets a looser translation bound; by L2 translation agrees to ~4e-10. Rotation bound 1e-4 deg vs measured worst ~2.1e-6 deg (~48x margin) and ~30x tighter than the shared error against ground truth (~3.2e-3 deg): the port tracks libapriltag far more closely than either tracks reality. Object-space error is compared absolutely because both scalars sit at ~1e-13.
- GCC reports -Wmissing-field-initializers for brace-init of only a name despite member initialisers, hence the explicit constructor. `sink` is consumed to avoid both dead-code elimination and a zero-length printf (-Wformat-zero-length).

### validate_pose_e2e
- Neither side shares corners/homography with the other (unlike validate_pose, which feeds one shared H to both solvers), so corner-localisation differences flow into the pose comparison, measuring what a user of the whole pipeline gets versus stock libapriltag.
- Both ambiguity solutions are kept: a large rotation delta is usually a branch flip (near-tied object-space errors decided by a sub-pixel corner difference, loser tens of degrees away), not a solver disagreement. Ambiguity is near-tied for small, near-fronto-parallel tags, hence reporting pixel size as the covariate. Comparing against the reference's runner-up separates agreement from branch selection; the gap between all_rot_deg and the resolved delta is damage from branch selection; a small err_gap means the flip is decided by corner noise below detector precision.
- A constant non-zero mean corner offset (not zero-mean scatter) indicates a coordinate-convention difference between detectors.
- Intrinsics default to per-image (fx = fy = width, cx/cy = centre) because the tool measures agreement, not absolute accuracy, as long as both sides get identical values. Default tag size 0.1651 m matches validate_pose.
- refine_edges: libapriltag defaults ON while vkapriltag defaults off (TagDecoder supports it via --our-refine-edges). Both off is the apples-to-apples default; --ref-refine-edges alone measures a real asymmetry; both on is shipping-vs-shipping.
- The reference side calls the orthogonal-iteration entry point directly (estimate_tag_pose runs it and returns only the winner), which costs nothing extra and yields the runner-up; min(err1, err2) reproduces estimate_tag_pose's choice. Duplicate tag IDs (rare) keep the first, since that is a decoder-quality issue orthogonal to pose.

## RDNA4 pass

### uf_final_body.glsl / uf_final.comp / uf_final_u8.comp
- The wrapper defines Thresholded's binding and THRESHOLDED_AT(i), as label_pixels_body.glsl does; the u8 wrapper differs only in the element type (see decimate_u8.comp).
- blob_size[] has one reader, label_pixels.comp, which only tests `blob_size[r] >= min_blob_pixels`. The count itself is never consumed, so an increment after a blob has reached the floor is unobservable. Skipping it after a plain read turns the contended case (every pixel of a large region hitting one slot) into a cache-hot load.
- The stored value is a saturating counter, not a histogram; a second reader that needs true sizes cannot use it.
- Exactness: let S(r) be the true pixel count of root r and M = min_blob_pixels. The counter only increments, at most S(r) times, from 0. If S(r) >= M, either nothing skipped (final = S(r)) or a thread read a value >= M and the counter never decreases (final >= M). If S(r) < M, no load can return >= M, so nothing skips and final = S(r).
- Ambiguous (127, code 1) pixels never merge, so pixel i is its own root and blob_size[i] is at most 1, below any floor of 2 or more. Reading thresholded[i] first and skipping parent[]/blob_size[] for code 1 is exact. GpuDetector floors min_cluster_pixels at 2 for this reason.
- The plain read races with other invocations' atomicAdd and is deliberately not coherent/volatile: a stale load returns some previously written value, all <= S(r), so it can only underestimate, costing an extra atomicAdd and never a wrong result. Coherence would force every load past L1 for freshness the shader does not need. hash_group.comp similarly reads hash_owner[h] non-atomically.
- min_blob_pixels must equal label_pixels.comp's, or the saturation threshold differs from the one tested.

### label_pixels_body.glsl (additions)
- The wrapper defines Thresholded's binding and THRESHOLDED_AT(i), as uf_final_body.glsl does.
- Each invocation reads and writes only its own parent[] entry, so rewriting parent[] in place needs no extra buffer. The next frame's uf_init overwrites every entry, so the packed encoding cannot leak across frames.
- Folding the blob lookup and threshold into one per-pixel word reduces blob_diff to one random gather per pixel instead of a blob_size lookup for the pixel and each neighbour plus two per emitted point; the cost here is one extra streaming read of thresholded[i].
- The +1 bias on the label makes 0 mean "no usable blob", merging blob_diff's ambiguous-pixel and small-blob rejections into one comparison.
- honour_changed_flag: the write is destructive (raw union-find parents are replaced by the packed word). On the speculative first attempt of the fused path, labelling may not have converged; a retry would then walk parent[] pointers that are no longer valid, possibly forming a cycle that hangs the GPU. Returning early leaves a valid, incomplete union-find structure the retry can continue.
- Ambiguous pixels get label 0 without reading blob_size: their counter is 0 or 1, both below the floor (see uf_final_body.glsl).

### reduce_extents_hash_body.glsl
- There is no sorted array to offset into: scatter_index_points.comp places each point with a per-blob cursor, and MinMaxExtentsGpu has no starting_offset field.
- The device-side count (count_buf) lets the dispatch be issued indirectly in the same submission that produced the count, avoiding a mid-frame SubmitAndWait (see fused_submits_ and build_indirect_args.comp). The push constant is the fallback for the unfused path.
- The uint64 view of binding 3 and the struct view address disjoint fields, so there is no aliasing hazard.
- Testing before atomicMin/atomicMax is exact (an atomic with a value not tighter than the stored one is a no-op), and extents stop changing after a blob's first few points, so almost every one of these atomics is skipped. This is worthwhile on the scalar path (the only one integrated parts use), where an early-out and contention relief compose.
- 64-bit count/sum packing: count (+1 per point) and pxgx_plus_pygy_sum (+v per point) are the only fields every point updates, and common.glsl places them in one 64-bit word. Adding `(uint64(uint(v)) << 32) | 1` adds v modulo 2^32 to the high half, as the 32-bit atomicAdd does, and 1 to the low half; the low half cannot carry into the high half because count is bounded by the boundary-point capacity.
- gx_sum/gy_sum share the other 64-bit word. Each contribution is biased by +1 so it lies in {0,1,2} and is non-negative; with raw signed deltas the running low half wraps through 0/0xFFFFFFFF and carries spuriously into gy. The biased low half stays monotonic and below 2^32 (at most twice the point capacity). Every point must contribute, so this path is unconditional; select_blobs.comp subtracts count from each half afterwards.
- On the 32-bit path gx and gy are zero for horizontal and vertical connections (blob_diff_body.glsl emits E as (+/-1, 0) and S as (0, +/-1); only SE/SW diagonals carry both), so about half the gx_sum/gy_sum atomics add zero. Skipping a zero add is bit-identical, and it tests a register value rather than loading one.

### select_blobs.comp
- The polarity test uses no GLSL double: shaderFloat64 is optional and absent on Mali and most mobile GPUs, where a shader using it fails pipeline creation. Only the sign of the dot product is used and `sum2` is an exact integer, so it is accumulated in int32 as the atomics produce it and only the sub-pixel correction is in fp32. Testing 2*dot (sign unchanged by a positive scale) clears the 0.5 and keeps the integer term exact. The fp32 rounding of `sum2` (ulp about 64 at realistic magnitudes) is far below the correction terms, so the result can differ from a float64 evaluation only for blobs essentially on the polarity boundary.
- gx_gy_biased: reduce_extents_hash_atomic64.comp stores gx_sum/gy_sum as (true sum + count) so its packed word never wraps; the scalar and subgroup variants write true sums and need no correction.
- A blob with zero bounding-box width or height is degenerate and would divide by zero in the fill test, so it is rejected outright. The aspect and fill filters are independent and each disabled by 0 (see DetectorConfig::aspect_max/fill_min/fill_max).
- tag_width is in bit squares, not pixels, so it does not bound a pixel-area test.

### sort_points_local_body.glsl (additions)
- The wrapper defines DecimatedImage's binding and DECIMATED_AT(i) (see decimate_u8.comp for the u8 variant).
- scatter_index_points.comp packs each blob's points into one contiguous range, so only theta_key is needed, sorted within the blob's range; no composite (blob_index, theta) key is required.
- The network runs over kLocalCap virtual slots with each thread owning a strided subset, so per-blob capacity follows the shared-memory budget rather than the workgroup thread count. Large real blobs (a large tag's own border) can exceed the device's maximum workgroup size while fitting in shared memory.
- Blobs above kLocalCap keep their original unsorted order. FitQuadForBlob consumes points as an ordered walk of the perimeter, so such a blob yields a geometrically meaningless quad, and a tag whose border lands here is undetectable. The failure is silent and total, so each such blob bumps OversizedBlobs (DetectProfile::oversized_sort_blobs). A handful per frame is normal (large background structures fitting junk quads that fail to decode). select_blobs.comp's filters do not bound point count, so ordinary scenes reach this path; the cap must leave headroom above a real tag border at the configured decimation, hence its derivation from decimation.
- The line-fit moment computation is fused in: once a blob's points are in final order each thread knows which source point lands at its output position, so it samples the decimated image and writes the RawLineFitPoint directly, without an intermediate sorted IPoint buffer.
- theta_key is scaled to 20 bits and kLocalCap is capped at 4096, so key and local index share one word (key high, index low). Comparing packed words sorts by key with the index as a harmless tiebreak, halving the network's shared-memory footprint and per-compare traffic. 12 index bits (rather than 11) are needed because a blob's perimeter in decimated points scales as 1/decimation.
- The network is sized to the blob (`cap` and `log2_cap` are uniform across the workgroup, so every barrier is still reached by every invocation).
- The sorting network is Batcher's odd-even mergesort over the padded `cap` slots: the same number of barrier-separated rounds as a bitonic network (log2(cap)*(log2(cap)+1)/2), with fewer comparators per round. The (p, q, r, d) schedule is the standard iterative construction, checked with the zero-one principle (all binary inputs at cap = 8 and 16, brute-force permutations at cap = 8, random permutations at larger caps); `q - p` never wraps for cap up to 2^20.
- Comparators are enumerated directly instead of visiting every idx and testing `(idx & p) == r`, which leaves a fraction of iterations idle. The valid idx form a period-2p pattern with p valid values per period, truncated by the idx+d < cap bound at T = cap - d. Because p and 2p are powers of two, the full-period count and the position within the period are a shift and a mask, with no runtime divide (the device class has no integer divide instruction). The comparator count and exact idx set match the guard-based form for every round over caps 16..4096.

### uf_merge_body.glsl
- The wrapper declares Thresholded's storage width; Parent stays uint32 because a union-find index routinely exceeds 255.
- Only the down edge is processed: uf_init.comp has already joined horizontal runs, so together the two cover every edge once.
- Run-level merging: wherever a run in row y lies above a run of the same value in row y+1, every column of the overlap requests the same union of the same two components (each row's run is already one component). Only the leftmost column of the overlap needs to perform it; each redundant column would cost two find() walks of dependent global loads. The run-start test costs two extra loads adjacent to ones already made. This is the transferable part of HA4 (Hennequin and Lacassagne's run-based 4-connected GPU CCL), without its warp intrinsics, which are the wrong tool on both device classes here.
- Equivalence: if v[i-1] == v[i] then uf_init joined i-1 and i; if v[i-1+W] == v[i+W] it joined those two; so once the overlap's leftmost column has unioned its pair, i and i+W are already in one component, and induction over the overlap gives the rest. The closure per pass is unchanged, and hooking is by atomicMin so a root is still its component's minimum index.
- changed_flag is set only when a union actually joined two distinct components, so it means "another pass is needed". Counting every pixel pair with a matching neighbour would issue up to two atomics per pixel to a single address, serialising on one cache line.
- Dispatched 2D with a one-row-tall workgroup: the run-start test needs x, and recovering it from a linear index needs a division by a runtime width on parts with no integer divide (Valhall). One-row workgroups keep consecutive threads on consecutive columns, so the two row streams stay coalesced.
- find mode 1 (path splitting, ECL-CC "Jump4" / Jayanti-Tarjan "split") repoints every visited node to its grandparent with a plain, non-atomic, non-coherent store. It is safe because parent[x] <= x always (every hook is atomicMin and uf_init points left or at self), so the grandparent is never greater than the node's current parent and no cycle can form. A store that loses a race with another find() or an atomicMin costs at most one longer future walk; doUnion's retry already tolerates the tree changing under it. Coherence would pay for freshness the shader does not need, the same trade as uf_final's plain read. It is the default on every device class; APRILTAG_VK_FIND_MODE=0 selects the naive walk for comparison.
- Merge flag mode 0 aggregates in shared memory (one atomicOr per workgroup, two barriers) and suits hardware with dedicated shared memory. Mode 1 issues a read-guarded global atomicOr per pixel with no shared memory or barriers; atomicOr is idempotent so correctness is unconditional and the guard only bounds redundant atomics. It suits parts that back `shared` with L2 (Mali Valhall), where the barriers buy nothing. The default follows caps().unified_memory, opposite in direction to the find mode default.

### Conditional rendering (Context, ComputePipeline, GpuDetector)
- VK_EXT_conditional_rendering has no shader variant behind it: it predicates dispatches whose shaders already carry an in-shader early-out (uf_compress.comp's honour_changed_flag), and that early-out is the fallback where the extension is absent.
- If a merge joins nothing, parent[] is untouched and already flat (every chunk ends with a compression), so the following compression has nothing to do. Only the last compression in a chunk is predicated: earlier ones read a flag accumulated since the previous clear, which is nonzero in practice, and a predicate that never skips costs extra because the front end must wait for the flag before issuing the dispatch. With predication the compression runs with honour_changed_flag = 0 since it only runs when the flag is nonzero.
- The ComputeAndPredicate barrier kinds add the predicate-read stage on the destination side; on the source side the predicate read writes nothing, so an execution dependency is enough for write-after-read. The stage bit is invalid without the extension.
- The entry points are loaded through vkGetDeviceProcAddr because the loader does not export them; the feature is dropped if either fails to resolve, so a bad driver costs the optimisation rather than the device.
- uf_changed_buf_ carries VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT only when the feature is present.

### GpuDetector.cpp (additions)
- The packed-coordinate check uses the doubled decimated coordinate (2*width/decimation), which equals width only at decimation 2.
- The label_pixels 2^30 pixel-count check is implied by the 14-bit coordinate limit (8191x8191 is under 2^26); it documents the coupling and fires first if the packing widens. The max_raw_blobs check likewise documents RawLineFitPoint's 22-bit blob_index: W is at most 361 by construction and blob_index is bounded by max_blobs <= max_raw_blobs, and overflow would otherwise mis-group points silently.
- min_cluster_pixels is floored at 2 because a 127 pixel is always a size-1 root whose saturating counter reads 0 or 1; a floor of 0 or 1 would let an ambiguous pixel pass the size test.
- Auto max_blobs scales linearly with the decimated pixel count and is anchored so 1080p at decimation 2 resolves to 2048. Overflow is not graceful: select_blobs.comp drops the excess in atomicAdd order, so detections stop being reproducible (DetectProfile::selected_blob_drops). The headroom is deliberate because blob density is scene dependent; its cost is one dispatch and one scan over the capacity, while the per-frame readback is sized by the actual count.
- The sort workgroup is limited to 256 threads because the network is sized to the blob (typically about 256 slots), so a wider workgroup leaves lanes idle in every stage while still reserving the full shared-memory allocation.
- The sort slot count scales as 2048 * 2 / decimation because a blob's perimeter in decimated points goes as 1/decimation (a 1080p tag border is about 1200 points at decimation 2 and about 2400 at decimation 1). It never drops below the decimation-2 value, so coarser decimations keep the same geometry; it is capped at 4096 by the 12-bit local index and by shared memory.
- selected_extents_buf_ and line_fit_points_buf_ use HostVisibleCached rather than DeviceLocalReadback: the latter requires device-local memory, which on a discrete card without a device-local + host-cached type falls back to plain DeviceLocal (the staged path, keeping fused_submits_ false). HostVisibleCached treats device-local as a preference, so it lands on host-cached system memory on such cards and on the device-local + host-visible + cached type on unified-memory parts; one kind serves both. On a discrete card the shader writes then go to system memory, which can slow them while removing the submission overhead that fused_submits_ cuts. The selected-extents readback is small; the point is that its size is known at record time (staging needs num_selected_blobs on the host), and reading in place removes the last host dependency in the frame tail. The line-fit buffer is the one large readback (about 1.6 MB at 1080p); it must be cached, since reading it through an uncached mapping is much slower than the copy it replaces.
- The hash table is sized to max_raw_blobs itself. A frame that reaches max_raw_blobs distinct pairs drops the excess through hash_group.comp's probe-cap fallback, as the other capacity clamps do, and real frames use a small fraction of the table.
- Subgroup aggregation is used only for reduce_extents_hash: it reduces per-point values across lanes sharing a key, collapsing atomics per point into atomics per distinct key per subgroup. Aggregating only a counter (uf_final, blob_diff) replaces one atomicAdd per lane with one per subgroup, which a discrete part's atomic unit already handles, so the ballot/shuffle sequence costs issue slots for nothing. The variant needs ballot (leader election), arithmetic (value reduction) and shuffle (SPIR-V subgroupBroadcast needs a constant lane id; the leader lane is computed at runtime). Integrated GPUs are excluded regardless of reported support: on Mali-G610 the software-costed ballot/shuffle sequences are far slower than the plain atomics they replace.
- decimation is specialization constant 3 so the compiler can fold `dx * decimation` into a shift for power-of-two values; a push-constant value would need an integer divide, which Valhall lacks. decimate_pl_ therefore takes the decimated dimensions from the host.
- The first uf_compress runs with honour_changed_flag = 0 because no merge has run and the flag is still zero from the clear; honouring it would skip a pass that flattens the run chains before the first merge (otherwise each vertical union pays an O(run length) find()).
- The convergence flag is cleared immediately before the chunk's last merge, so a zero readback means the final pass changed nothing. Clearing once at the start of the chunk cannot tell "converged" from "changed something earlier in this chunk".
- Every copy of the flag after record_uf_chunk needs a ComputeAndTransfer barrier: the default Compute barrier's destination stage is compute, so a following transfer read is not ordered after the write. The race shows up when a fused tail gives the scheduler work to overlap the copy with, and a small chunk (APRILTAG_VK_UF_CHUNK=1) makes it lose reliably.
- The timestamp pool is reset once per frame: queries cannot be rewritten without a reset, later submissions run after this one's fence, and the fused retry resets it again because Reset() clears every query. That frame's per-span breakdown is unavailable, and extra convergence chunks are not attributed to the labelling span.
- finish_frame is a lambda so the fused path can run the tail speculatively and again after a retry; `speculative` gates label_pixels_pl_ and blob_diff_pl_ so neither acts on unconverged labels. uf_iterations and uf_converged are set after the caller resolves the retry because the speculative pass's own guess is wrong on exactly the frames the mechanism handles.
- On a retry parent[] is incomplete rather than invalid, so uf_merge and uf_compress resume from it. The buffers the discarded tail dirtied are re-zeroed (all of the frame's clear set except uf_changed_buf_, which record_uf_chunk re-zeroes).
- init_extents and select_blobs dispatch over the frame's actual raw blob count via arguments built on the device from raw_blob_counter_buf_, avoiding the host round trip; both bounds-check against max_raw_blobs, so the rounded-up group count is harmless.
- The total point count is read from blob_point_offsets_buf_[max_blobs - 1], the final inclusive-scan value: entries past num_selected_blobs are zero-padded, so this static offset equals the sum of all selected blobs' counts.
- select_blobs.comp's counter counts blobs that passed the filters and can exceed max_blobs (it drops the overflow), so the host keeps it unclamped to report selected_blob_drops.
