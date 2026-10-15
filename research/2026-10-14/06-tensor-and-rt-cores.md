# 06: Matrix units (WMMA / cooperative matrix / int dot / Arm tensors) and ray-tracing units

Status: complete

Scope: rigorous, stage-by-stage feasibility study of fixed-function matrix and ray-tracing hardware
for vkapriltag on the RX 9060 XT (RDNA4) and the Orange Pi 5 (Mali-G610 MP4). The previous round
rejected cooperative matrix in one paragraph (`research/08-vulkan-extensions.md` section 6); this
file redoes it stage by stage, with arithmetic. No builds or timing runs; one read-only
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` query via Python `ctypes`.

## Top conclusions (final, ranked)

1. **No tangible benefit from matrix or RT units for the current pipeline, on either device.
   Verdict: reject all 13 stages (section 1 table).** Nothing to measure: there is no candidate
   kernel worth building.
2. **The deployment target has neither unit** (0.3): the Mali-G610 exposes no
   `VK_KHR_cooperative_matrix`, no `VK_KHR_ray_query` / acceleration structures (the names exist
   only in the libmali binary strings, for other GPUs), and no `VK_ARM_tensors` / `data_graph`. So
   WMMA/RT could only ever help the desktop, where 8e already says GPU shader work is not the limit
   (~217-380 us of GPU per frame; the CPU tail and submit/wait dominate).
3. **RX 9060 XT cooperative matrix, queried directly (0.1):** 18 shapes, all 16x16x16, subgroup
   scope; inputs f16, bf16, fp8 (e4m3/e5m2), u8/s8 only. No 1-bit or 4-bit shapes, no fp32/int32
   inputs, no `NV_cooperative_matrix2`. Every exact mapping must go through u8 -> s32 with byte-plane
   splitting.
4. **No stage is a dense (+,x) product that is ALU-bound** (sections 0.5, 1, 2). The threshold is
   min/max (wrong semiring); labelling, extents, sort and boundary are latency/atomic/barrier bound
   with irregular keyed access; the dense-matmul forms that do exist are slower: e.g. union-find as
   boolean closure costs ~134 G MAC = ~1.3 ms at peak vs 172 us for all of labelling today (2.2);
   rank-sort needs ~40x the comparisons of the current network (2.5).
5. **Integer dot product (the one unit the Mali has): no applicable site** without changing
   results (section 1, last paragraph): every multiply-add operand in the tree exceeds 8 bits or
   is not a dot product.
6. **Codebook search is already O(1)** (upstream hash table, 36 MB for tag36h11 at Hamming 2);
   brute-force matmul+argmin would be ~0.06 us but replaces nothing that costs time (2.8). Packed
   `bitCount(q ^ c)` on plain ALU is within ~2x of u8 WMMA because Vulkan has no 1-bit MMA.
7. **Where the answer changes (section 4):** (a) large families (tagStandard52h13, tagCircle49h12)
   at max Hamming 2 need a 6-8 GB table, so GPU brute-force decode becomes worthwhile - but with
   packed popcount (~9 us desktop / ~0.23 ms Mali), not WMMA (a tie at ~9 us); (b) soft-decision
   correlation decoding over such a family is the one case where f16 WMMA beats ALU (~8x, ~15 us vs
   ~0.12 ms), at the cost of libapriltag parity; (c) a learned front end (YOLOv8n-class, ~0.2-0.4 ms
   on desktop WMMA, ~13.6 ms on the RK3588 NPU) only if the goal becomes recall, not speed. Larger
   images change no verdict (4.4).
8. **RT units (section 3): reject.** Published wins (RTNN 2.2-65x, RayJoin 3-28x, kNN up to ~200x)
   need large static primitive sets, many queries and spatial pruning, and are all on NVIDIA; the
   2026 survey found no AMD RT research and no gain for latency-bound graph traversal. This
   pipeline has no spatial search with large N; any BVH would be rebuilt per frame by driver
   compute shaders.

## 0. Hardware facts (verified this session)

### 0.1 RX 9060 XT: cooperative matrix property list (queried directly)

`vulkaninfo` prints only the feature bits (`cooperativeMatrix = true`,
`cooperativeMatrixRobustBufferAccess = true`, stages = compute). The actual list of supported
shapes comes from `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`, which `vulkaninfo` does not
dump, so it was queried with a 60-line Python `ctypes` script against `vulkan-1.dll` (no build;
script in the session scratchpad, `coopmat.py`). Driver 26.8.1 (LLPC), API 1.4.349. **18 entries,
every one M16 N16 K16, scope = subgroup:**

| A x B -> C/Result | saturating variants |
| --- | --- |
| f16 x f16 -> f32 | no |
| f16 x f16 -> f16 | yes and no |
| u8/s8 x u8/s8 (all four sign mixes) -> s32 | yes and no |
| e4m3/e5m2 x e4m3/e5m2 (all four mixes) -> f32 | no |
| bf16 x bf16 -> bf16 | yes and no |
| bf16 x bf16 -> f32 | no |

Consequences that matter for this study:
- **No 32-bit integer or fp32 inputs.** Anything wider than 8-bit integer must be split into
  byte planes (K-slicing) or carried in f16/bf16 (11/8-bit mantissa), which is not exact for the
  values this pipeline sums (see 2.7).
- **No int4 / binary (1-bit) shapes exposed**, although RDNA4 WMMA has an iu4 mode in hardware
  (AMD RDNA4 ISA). So a "popcount via 1-bit tensor op" (the NVIDIA `b1` XOR/AND-popc trick) is not
  available through Vulkan on this card; bits would have to travel as u8 0/1 values (8x the
  storage and 8x the K-dimension).
- **No `VK_NV_cooperative_matrix2`, no `VK_NV_cooperative_vector`** (no `VK_NV_*` extensions at
  all). So no tensor-addressing, reductions, per-element ops or workgroup-scope matrices: only the
  KHR subgroup-scope 16x16x16 MMA, load/store from buffers or shared memory, and element-wise
  access. Shared memory is 32 KiB per workgroup (`maxComputeSharedMemorySize = 32768`).
- Subgroup size 32 or 64 (`minSubgroupSize 32`, `maxSubgroupSize 64`); WMMA on RDNA is native in
  wave32, where a 16x16 f32 accumulator is 8 VGPRs per lane.
- Also exposed: `VK_KHR_shader_integer_dot_product` with all 8-bit and 4x8-bit-packed forms
  `Accelerated = true`, 16/32/64-bit forms **not** accelerated; `shaderBFloat16DotProduct`,
  `shaderFloat8CooperativeMatrix`.

### 0.2 RX 9060 XT: ray tracing

`VK_KHR_acceleration_structure` (rev 13), `VK_KHR_ray_query` (`rayQuery = true`),
`VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_tracing_position_fetch`,
`VK_KHR_ray_tracing_maintenance1`, `VK_EXT_ray_tracing_invocation_reorder` (hint = NONE, i.e. no
hardware reordering) are all exposed. `maxPrimitiveCount = 536870912`. So ray queries from a
compute shader are available on the desktop.

### 0.3 Mali-G610 MP4: no matrix unit, no RT unit, int8 dot only

- `VK_KHR_cooperative_matrix`: **not exposed** on any G610 report (g18p0, g24p0 Linux report
  41734 = the deployment driver, r38p1, r54p2, panvk 25/26). The name appears in the g24p0 and
  g29p1 *binary strings* only because one libmali binary serves 5th-gen parts
  (`research/08-vulkan-extensions.md` section 0 caveat, re-read this session).
- `VK_KHR_ray_query` / `VK_KHR_acceleration_structure`: same situation - present in the binary
  strings, **not exposed on the G610** (same caveat list: "features of other Valhall/5th-gen parts
  that one binary serves: `KHR_acceleration_structure`, `ray_query`, `ray_tracing_pipeline`").
  This matches Arm's product line: hardware ray tracing first shipped in Immortalis-G715 (2022);
  the G610 (2021) has none. **Verified: the deployment target has no RT hardware and no RT API.**
- `VK_KHR_shader_integer_dot_product`: exposed since g18p0 with `8BitUnsigned/Signed` and
  `4x8BitPackedUnsigned` `Accelerated = true` (reports 37373, 41734, 50479; 08 section 6).
- `VK_ARM_tensors` / `VK_ARM_data_graph`: absent from every G610 driver and from both libmali
  binaries' strings (08 table). These target Arm's neural-accelerated GPUs (announced for 2026
  parts), not Valhall.

### 0.4 Throughput of the units, against plain shader ALU (for the arithmetic below)

| Device | Path | Peak | Basis |
| --- | --- | --- | --- |
| RX 9060 XT (32 CU, 3.13 GHz boost) | shader ALU, 32-bit int/fp lane-ops | ~6.4 T/s single-issue (25.6 TFLOPS FP32 counts FMA x2 and VOPD dual issue) | AMD spec table (StorageReview); 32 CU x 64 lanes x 3.13 GHz |
| | `v_dot4` int8 (`OpUDot` 4x8 packed, Accelerated) | ~25.6 T MAC/s | estimate: 4 MAC/lane/clk |
| | WMMA FP16 dense | ~102.6 TFLOPS = ~51 T FMA/s | estimate: AMD's 410 INT8 TOPS headline is the 2:4-sparse figure; dense int8 205 / fp16 102.6 is consistent with 32 CU x 3.13 GHz x 1024 FP16 FLOP/clk/CU (flopper.io lists 103 FP16 / 205 FP8) |
| | WMMA int8 dense | ~205 TOPS = ~102 T MAC/s | as above |
| | Ray accelerators | 32 (one per CU): 8 box tests or 2 triangle tests per clock per CU; traversal loop runs in the shader (`IMAGE_BVH8_INTERSECT_RAY` + LDS stack instructions); BVH **build** is compute shaders in the driver | Chips and Cheese "RDNA 4's raytracing improvements" |
| Mali-G610 MP4 (~1 GHz) | shader ALU | 512 GFLOPS FP32 (4 cores x 64 FMA lanes x 2 x 1 GHz) | Arm community thread arithmetic; matches clpeak-style figures |
| | int8 dot (Accelerated per driver) | unpublished; one user measured int throughput "about a quarter of what I expected" | Arm community forum 55167 (no Arm answer) |
| | matrix unit, RT unit | **none** | 0.3 |
| RK3588 NPU (comparison only) | 3 cores, 6 TOPS int8 (vendor) | YOLOv8n ~13.6 ms vs the whole Mali GPU phase 3.76 ms | `research/00-SUMMARY.md` section 2 |

So the matrix unit's advantage on the desktop is ~16x over scalar ALU and ~4x over int8 `dot4` -
**but only for work that is (a) a dense (+,x) product, (b) at least 16x16x16 in shape after
padding, (c) exact in 8-bit-integer or f16/bf16/fp8 operands, and (d) not dominated by the
irregular gathers / scatters / atomics needed to put the data into 16x16 tiles.** The rest of this
file is checking (a)-(d) per stage.

### 0.5 Where the time is (the ceiling for any unit)

Orange Pi, `grayimage.pgm` 1280x800, decimation 2, timestamps on, DMC 2112 MHz
(`PERFORMANCE.md` 3a): GPU total 3.457 ms = `labelling` 0.921 (27%), `extents` 0.432 (12.5%),
`sort` 0.397 (11.5%), `boundary` 0.361 (10.4%), `label_pixels` 0.116, `uf_final` 0.091, `clear`
0.048, rest (threshold+decimate, hash_group, scatter, select/scan, readback, and at the time
inter-submit gaps) ~1.09. CPU tail: `quad_decode` 0.44 ms, `tag_decode` 0.79 ms (section 3).
Current `Detect()` ~2.70 ms, `pipeline_total` ~4.20 ms (brief).

RX 9060 XT: one `Detect()` ~0.5-0.6 ms, of which compute spans ~380 us (8e); `labelling` is 172 us
of ~330 us summed spans (8e); `threshold` span 13 us (8d). With 4 streams the GPU is saturated at
~4,600 fps, i.e. **~217 us of GPU time per frame**. 8e: "the GPU phase is ~0.33 ms of a ~1.0 ms
frame, so GPU shader work is no longer the limit; the tail and the host-side `submit_wait` are."
Older per-stage desktop shares (README, 1920x1080, before several optimizations): labelling 0.20,
sort+line-fit 0.11, label finalize 0.05, extents 0.05, boundary 0.03, threshold+decimate 0.02 ms.

Two structural facts bound everything that follows:
1. **The deployment target has no matrix unit and no RT unit** (0.3). Only int8 dot exists there.
   Everything WMMA/RT can only ever help the desktop.
2. **No GPU span is ALU-throughput-bound.** `extents` and `sort` do not move across a 4x memory
   clock range and are atomic/latency/barrier bound; `labelling` is dependent-load latency
   (`PERFORMANCE.md` 3a); occupancy is not a limiter on the RX 9060 XT (all pipelines reach the
   16-waves/SIMD cap, section 3). A matrix unit multiplies ALU throughput, which is the one
   resource no stage is short of.

## 1. Stage-by-stage verdict table (matrix units and integer dot)

Data volumes are for 1280x800, decimation 2: 640x400 = 256,000 decimated pixels, ~65k boundary
points, ~388 raw blobs, ~75 candidate quads, 1 decoded tag (`PERFORMANCE.md` 3b, 8e).
"Share" = share of the device's frame time, from 0.5. Details for each row in section 2.

| Stage (code) | Runs on | Share, Mali / RX 9060 XT | Matrix / dot formulation | Exact in the exposed types? | Best-case unit time vs current | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| decimate (`decimate_u8.comp:18-28`) | GPU | small / ~1% | none: integer decimation is point sampling (one byte gather per pixel), zero arithmetic | n/a | nothing to accelerate | reject |
| block min/max + 3x3 filter + compare (`block_minmax_u8.comp:33-42`, `block_filter.comp:18-31`, `threshold*.comp`) | GPU | ~3-6% / ~3% (13 us) | min/max is the (min,max) semiring; WMMA is (+,x) only. A box *sum* would be a matmul, but upstream's threshold is min/max | no mapping | n/a | reject (2.1) |
| union-find labelling (`uf_init/merge/compress/final`) | GPU | 27% + 6% / ~50% | connectivity closure by repeated boolean squaring of tile adjacency | yes (u8 0/1, s32 acc, re-binarize) | ~1.3 ms at peak WMMA for intra-tile closure alone vs 172 us whole labelling today: **>=7.6x slower** | reject (2.2) |
| boundary / blob_diff + hash_group | GPU | ~13% / ~10% | none: per-pixel neighbour compare, compaction, hashing | n/a | n/a | reject |
| extents reduction (`reduce_extents_hash_body.glsl:37-62`) | GPU | 12.5% / ~15% | segmented sum = one-hot(blob) x point-features; min/max fields not mappable | sums only, after byte-plane splitting | MMA part ~1.2 us, but needs points grouped by blob first (they arrive in `blob_diff` append order) and 4 of 8 fields stay atomic | reject (2.3) |
| select / blob offset scan (`scan_block.comp`) | GPU | ~0.3% | TCU scan (Dakkak) | yes for small ints | <=10 us span total (A12) | reject (2.4) |
| scatter_index_points | GPU | ~3% / ~4% | none (scatter) | n/a | n/a | reject |
| per-blob sort (`sort_points_local_body.glsl:99-137`) | GPU | 11.5% / ~10-20% | rank = row-sum of an n x n comparison matrix; matmul radix split (Ascend) | yes | n^2 compares on ALU (1M per 1024-pt blob) vs ~25.6k comparators today: **~40x more ALU work**; the MMA only does the trivial row-sum | reject (2.5) |
| gradient weight W (`sort_points_local_body.glsl:40-47`) | GPU | <1% | `gx*gx+gy*gy` is a 2-term dot | **no**: \|gx\|,\|gy\| <= 255 needs 9-bit signed, s8 dot overflows the operand | 2 multiplies saved per point | reject |
| line-fit prefix moments (`QuadDecode.cpp:279-295`) | **CPU** | part of `quad_decode` 0.44 / 0.09 ms | prefix sum = lower-triangular ones x values (TCU scan) | only after splitting 37-bit products into 5 byte planes and re-carrying in int64 | ~393k values (65k pts x 6 moments): microseconds on plain GPU ALU (`subgroupInclusiveAdd`); the cost today is that it is on the CPU, not ALU throughput | reject for the unit (2.6); "move to GPU" is a different idea |
| windowed line error + 7-tap Gaussian (`QuadDecode.cpp:313-330`) | CPU | (in `quad_decode`) | Toeplitz matmul | **no**: double in upstream; f16/bf16 inputs only | ~7 MAC/point | reject (2.6) |
| homography (upstream `homography_compute2`, 8x9 Gaussian elimination per quad) | CPU | (in `tag_decode`) | batched small solves | **no**: f64 elimination with pivoting; no f32/f64 MMA inputs | ~75 tiny solves | reject (2.7) |
| bilinear tag sampling + sharpen (upstream `quad_decode`: 64 border samples, 2 x 3x3 graymodel solves, 100-cell 3x3 sharpen, 36 bits) | CPU | (in `tag_decode` 0.79 / 0.29 ms) | sharpen is a 3x3 conv (tiny); sampling is a gather | no (double) | ~100 samples per quad, ~7.5k per frame: no arithmetic mass | reject (2.7) |
| codebook search, tag36h11 587 codes (upstream `quick_decode_codeword`: hash probe x 4 rotations) | CPU | ~0 (O(1) lookups) | Hamming = \|a\|+\|b\|-2 a.b as u8 0/1 matmul + argmin | yes | 300 queries x 587 codes x 36 bits = 6.3M MAC = ~0.06 us at peak; replaces 300 hash probes (~tens of us, already parallel) | reject for tag36h11; **worthwhile only for large families / high max-Hamming / soft decoding (section 4)** |

**Integer dot product (both GPUs, including the Mali):** searched for 8-bit dot shapes in every
shader and in the CPU tail. The only multiply-adds of 8-bit-derived values are the gradient weight
(9-bit operands, does not fit s8 / u8 with sign) and the point-sampling decimate (no multiply at
all). The moments (`x2` up to 14 bits, `W` up to 361) and the extents term `x*gx + y*gy` exceed
8 bits; the extents `gx`/`gy` are {-1,0,1} and fit, but they are *summed per blob*, not dotted
with anything, and are already packed two-per-int64-atomic (`reduce_extents_hash_body.glsl:55`). **There is no site where `OpSDot`/`OpUDot` applies without changing results.** This
confirms the 2026-09 verdict for int dot with a full sweep rather than a spot check.

## 2. Matrix units: the findings behind the table

### 2.1 Threshold tiles as matmuls (block min/max, 3x3 filter) - reject, both devices
- **Idea:** compute the 4x4 tile statistics and the 3x3 tile blur with WMMA (RX 9060 XT only).
- **Mechanism:** a box *sum* over 4x4 tiles is `T = R . I . C` with 0/1 row/column selector
  matrices, a textbook matmul (ConvStencil/TCStencil-style). But upstream's adaptive threshold, and
  this tree's bit-identical copy of it, is **min and max** per tile (`block_minmax_u8.comp:33-42`)
  and a 3x3 **min/max** blur (`block_filter.comp:18-31`). WMMA implements the (+,x) semiring only.
  There is no exact (+,x) encoding of max for 8-bit data short of 255 indicator planes
  (`max = sum_t [any v >= t]`), i.e. ~255x the work of the 16 compares it replaces.
- **Expected gain:** none. Even if it mapped, the desktop `threshold` span is 13 us (`PERFORMANCE.md`
  8d), ~6% of a saturated-GPU frame; the Mali has no matrix unit.
- **Cost / risk:** mean-based thresholds were already rejected for breaking bit-identity
  (`research/00-SUMMARY.md` section 2, "Alternative adaptive thresholds").
- **Prior art in tree:** 08 section 6 point (a) said the same in one line.
- **Verdict:** reject. Nothing to measure.
- Sources: ConvStencil PPoPP'24 https://www.microsoft.com/en-us/research/uploads/prodnew/2024/04/ppopp24_ConvStencil.pdf ;
  SPIDER (notes ConvStencil/LoRAStencil/TCStencil spend 2.1-5.9x the minimal FLOPs on a box stencil) https://arxiv.org/html/2506.22035v3

### 2.2 Union-find labelling as boolean matrix closure - reject, with arithmetic
- **Idea:** replace intra-tile union-find with transitive closure of the tile's pixel-adjacency
  matrix by repeated squaring on WMMA (u8 0/1 inputs, s32 accumulate, re-binarize with `min(.,1)`
  between steps), RX 9060 XT only.
- **Mechanism / arithmetic:** a 16x16-pixel tile has 256 nodes; closure needs
  `ceil(log2 256) = 8` squarings of a 256x256 matrix = 8 x 256^3 = **134M MAC per tile**; 256,000
  pixels = 1,000 tiles = **134 G MAC per frame**. At the ~102 T MAC/s dense int8 peak that is
  **~1.3 ms**, against **172 us for the whole of labelling today** (8e), which also does the
  cross-tile merges the closure would still need. An 8x8 tile (64 nodes, 6 squarings, 1.6M MAC per
  tile, 4,000 tiles) is 6.3 G MAC = ~62 us at peak, before loads, re-binarization and cross-tile
  merges - and it would replace only the intra-tile part, which `uf_init`'s run joining already
  makes cheap (labelling's cost is merge chains, `PERFORMANCE.md` 3a: dependent-load latency).
  The dense formulation wastes ~98% of its MACs: a pixel has <= 4 neighbours, so the adjacency is
  ~1.6% dense in a 256-node tile.
- **Literature:** no tensor-core CCL or union-find paper found (search 2020-2026). The nearest are
  bit-MMA BFS (BLEST, 2025: NVIDIA `b1` `m8n8k128` MMA) and tensor-core triangle counting / MIS,
  all on NVIDIA bit or sparse MMA modes that **Vulkan on RDNA4 does not expose** (0.1: no 1-bit or
  4-bit shapes, no `NV_cooperative_matrix2`). Round-synchronous label propagation, the other
  matrix-shaped CC method, was already rejected (pass count grows with image diameter; 00-SUMMARY
  section 2).
- **Expected gain:** negative by >= 7.6x at the 16x16 tile; at best break-even on part of a span
  that is not ALU-bound. Mali: not applicable.
- **Verdict:** reject.
- Sources: BLEST https://arxiv.org/pdf/2512.21967 ; TC-MIS https://arxiv.org/pdf/2605.29604 ;
  ToT triangle counting (PPoPP'25 poster) https://ppopp25.sigplan.org/details/PPoPP-2025-Main-Conference-1/7/POSTER-Triangle-Counting-on-Tensor-Cores

### 2.3 Extents reduction as a one-hot matmul - reject
- **Idea:** compute the per-blob sums (`count`, `gx_sum`, `gy_sum`, `pxgx_plus_pygy_sum`) as
  `S = OneHot(blob)^T x F` on WMMA instead of atomics (`reduce_extents_hash_body.glsl:37-62`).
- **Mechanism / arithmetic:** dense one-hot is 388 blobs x 65,536 points. F = [1, gx, gy,
  x*gx + y*gy]; `gx, gy` are in {-1, 0, 1} (`common.glsl:36`, so they fit s8 exactly) but
  `x*gx + y*gy` is ~12-bit signed and needs 2 byte planes, so F has 5 s8/u8 columns:
  388 x 65,536 x 5 = **~127M MAC = ~1.2 us at peak**. That looks cheap, but:
  1. 4 of the 8 fields are min/max (`:40-43`), which stay atomic or need a separate reduction.
  2. A dense one-hot is 25M bytes per frame (80 us of the card's 320 GB/s just to write it) unless
     built in LDS on the fly, and LDS is 32 KiB per workgroup.
  3. The sparse alternative - group points by blob first, then reduce each blob's contiguous run
     with a ones-vector MMA (Dakkak-style segmented reduction) - needs the points sorted by blob.
     They arrive in `blob_diff` append order; the scatter that groups them runs *after* extents,
     because extents decide which blobs are selected. Sorting 65k points to save atomics costs more
     than the atomics (a global sort was rejected in round 1, 05).
  4. What remains is not the arithmetic: on the RX 9060 XT the subgroup-aggregated atomic path is
     70 us (`PERFORMANCE.md` section 6 table), and the cost is the keyed scatter (`point_slot`,
     `slot_dense` loads, then atomics), not adds.
- **Expected gain:** none demonstrable; the MMA would do the cheapest 1% of the work.
- **Prior art in tree:** contention and int64 packing work, `PERFORMANCE.md` 3b; plain-store bound
  0.062 ms vs 0.434 ms on Mali (no matrix unit there).
- **Verdict:** reject.
- Sources: Dakkak et al., "Accelerating reduction and scan using tensor core units" (ICS'19) https://arxiv.org/pdf/1811.09736

### 2.4 Scans (blob offsets) on the matrix unit - reject
- **Idea:** TCU scan (lower-triangular ones matrix x values) for `scan_block.comp`.
- **Mechanism:** Dakkak et al. report up to 100x for segmented *reduction* and up to 3x for *scan*
  on V100, in fp16, against CUB, **for small segment sizes over large arrays**; fp16 is exact only
  for integers <= 2048. Here the scan is over a few hundred to a few thousand blob counts in one
  dispatch: launch- and barrier-bound, ~10 us total (round-1 A12 bound: "~10 us (0.3%)"). The
  Ascend work (Wroblewski et al. 2025) reports 5-9.6x over vector-only scan on large inputs and
  reaches 37.5% of memory bandwidth - again a large-array result.
- **Expected gain:** <= the 10 us span on the desktop, realistically 0.
- **Verdict:** reject.
- Sources: https://arxiv.org/pdf/1811.09736 ; https://arxiv.org/pdf/2505.15112v2 ;
  TCU-model scan analysis https://arxiv.org/html/2411.17887v1

### 2.5 Per-blob sort - reject
- **Idea:** rank-based sort (`rank_i = sum_j [k_j < k_i]`, ties by index) with the row-sum on WMMA, or
  matmul-assisted radix splits (Ascend: up to 3.3x for radix sort).
- **Mechanism / arithmetic:** the comparison matrix is built element-wise on ALU; for a 1,024-point
  blob that is 1,048,576 compares against Batcher's ~n log2^2 n / 4 = ~25.6k comparators that
  `sort_points_local_body.glsl:99-137` executes. The MMA only does the trivial row-sum. Radix
  splits win on large arrays where the split prefix counts dominate; here each blob is <= 1,024 keys
  in LDS and the span is barrier/latency-bound on Mali and was divergence-bound on RDNA (fixed by
  comparator enumeration, A3, -27 to -49% of the span).
- **Verdict:** reject.
- Sources: https://arxiv.org/pdf/2505.15112v2

### 2.6 Line-fit moments and the corner error filter - reject for the unit
- **Idea:** compute the six cumulative moments (`Mx, My, W, Mxx, Mxy, Myy`, `QuadDecode.cpp:279-295`)
  as a matmul scan, and the 7-tap Gaussian over the windowed error (`:321-330`) as a Toeplitz
  matmul.
- **Mechanism / arithmetic:** both run on the **CPU** today. The moments are int64 because
  `W*x2*x2` reaches 361 x 16,383^2 ~ 9.7e10 (37 bits); in the only exact WMMA type (u8 -> s32) that
  means 5 byte planes per value and an int64 carry pass on ALU. Volume: ~65k points x 6 moments =
  ~393k values per frame. A plain GPU subgroup scan does that in a few microseconds of ALU; the MMA
  cannot make a few microseconds smaller in any way that shows up. The Gaussian is 7 double MACs per
  point in upstream; f16/bf16 inputs would break bit-identity (fp16 corner positions were already
  rejected for exactly this, `research/08` section 6, 16-bit entry).
- **What does matter here:** `quad_decode` is 0.44 ms on the Mali A76 tail. Moving the moment
  prefix sums into `sort_points_local` on the GPU is a legitimate idea (another agent's area), and
  it needs nothing beyond `subgroupInclusiveAdd`.
- **Verdict:** reject for matrix units.

### 2.7 Homography, bilinear tag sampling, sharpening - reject
- **Mechanism:** per candidate quad, upstream (`apriltag.c` `homography_compute2`, `quad_decode`)
  does one 8x9 double Gaussian elimination with pivoting, 64 border samples feeding two 3x3
  least-squares gray models, 36 interior samples, and a 3x3 sharpen over the 10x10 cell grid for
  tag36h11 (`width_at_border 8`, `total_width 10`). That is ~100 bilinear samples per quad, ~7.5k
  per frame for 75 quads - almost no arithmetic, and all of it f64. WMMA exposes no f32/f64 inputs.
  The measured cost is elsewhere: 4.7 us per `quad_decode_index` call, likely `matd` allocations
  (8e), and 96.8% of refine time spent on quads that never decode (8e).
- **Aside, the fixed-function unit that *does* match this shape:** the texture unit's hardware
  bilinear filter. It is not exact either (Vulkan only guarantees `subTexelPrecisionBits >= 4`;
  upstream interpolates in f64), so it fails the parity gate for the same reason.
- **Verdict:** reject.

### 2.8 Codebook search (Hamming nearest code) as matmul + argmin - reject for tag36h11
- **Idea:** decode all candidate quads at once: query bits Q (0/1 u8) x codebook^T, Hamming =
  `|q| + |c| - 2 q.c`, argmin per query, on WMMA (desktop) or int8 dot (both).
- **What upstream actually does:** not a linear search. `quick_decode_init` builds an
  open-addressing hash table of every codeword with up to `maxhamming` bit flips, and
  `quick_decode_codeword` does one probe sequence per rotation (4 per quad). Entries are 16 bytes
  (`uint64 rcode, uint16 id, uint8 hamming, uint8 rotation`, padded), `nentries = 3 x capacity`.
  For tag36h11 at the default max Hamming 2: capacity = 587 x (1 + 36 + 36 x 35) = 761k,
  table = 2.28M entries = **~36 MB**, lookup O(1). So for tag36h11 there is nothing to accelerate:
  300 probes per frame, and the measured decode cost is the sampling / allocation (2.7).
- **Arithmetic if it were brute force:** 75 quads x 4 rotations = 300 queries x 587 codes x 36 bits
  = 6.3M MAC = **~0.06 us at WMMA peak**. But note the competitor is not scalar MACs: with the
  36-bit codes packed in a 64-bit word, `bitCount(q ^ c)` does all 36 "MACs" in ~3-4 lane
  instructions, i.e. ~10x bit-parallelism on plain ALU. 300 x 587 = 176k XOR+popcount pairs = ~0.7M
  lane-ops = ~0.1 us of RX 9060 XT ALU, or ~3 us of Mali ALU (256 G lane-ops/s). **Packed popcount
  already matches the u8 WMMA within ~2x**, because Vulkan exposes no 1-bit MMA (0.1) and the u8
  path spends a byte per bit.
- **Expected gain:** zero for tag36h11 (the lookup is already O(1) and off the critical path).
- **Verdict:** reject for the current configuration; see section 4 for where it changes.
- Sources: upstream `apriltag.c` `quick_decode_init` / `quick_decode_codeword` (vendored in
  `out/build/x64-Release/_deps/apriltag-src/apriltag.c:120-131, 196-260, 295-310`); Li & Su,
  bit-tensor-core BNN (the `POPC(A XOR B)` MMA primitive, NVIDIA only) https://arxiv.org/pdf/2006.16578

## 3. Ray-tracing units (RX 9060 XT only; the Mali has none)

### 3.1 What RT hardware is good at, per the literature
- **Survey (arXiv 2603.28771, 2026, "Ray tracing cores for general-purpose computing"):** wins are
  concentrated in nearest-neighbour search and its variants (best case ~200x for kNN), and in
  problems where BVH traversal *prunes* work; "many short-length rays should be preferred over a
  few large rays". Losses: latency-bound problems ("reducing memory accesses in BFS does not
  improve performance because of latency"; BVH build can exceed the whole traversal and all nodes
  must be visited); FP32-only coordinates; data shuttling between RT and shader cores; 9x memory
  blow-up when scalar keys become triangles; **"no research on ray tracing was found on [AMD]
  GPUs"** - every published speedup is on NVIDIA RT cores, which run the whole traversal in
  hardware. RDNA4 runs the traversal loop in the shader and only accelerates the box/triangle tests
  (8 box or 2 triangle tests per clock per CU, `IMAGE_BVH8_INTERSECT_RAY`), and the driver builds
  BVHs with compute shaders (Chips and Cheese).
- **RTNN (PPoPP'22):** fixed-radius neighbour search, 2.2-65x over GPU neighbour-search libraries,
  only after query scheduling and partitioning - "a naive mapping under-exploits the hardware".
- **RTIndeX (VLDB'23) / cgRX (2024):** database point and range lookups as rays; the follow-up
  lists RX's limits as "significant memory overhead per key, slow range-lookups, and poor
  updateability".
- **RayJoin (ICS'24):** point-in-polygon and segment intersection over millions of polygons,
  3.0-28.3x over high-precision CPU/GPU methods; it has to add software precision checks and work
  to reduce BVH build cost.
- Common preconditions: a **static or slowly changing** primitive set whose BVH build is amortised
  over many queries, **large N** (10^5-10^8 primitives, 10^6+ queries), a **spatial pruning**
  structure to exploit, and tolerance of FP32 geometry.

### 3.2 Candidate RT uses in this pipeline - none qualifies
| Candidate | Where | Why it fails the preconditions |
| --- | --- | --- |
| Point-in-quad / overlap tests | upstream's final de-duplication of overlapping detections (O(D^2) polygon overlap) | D = number of detections, ~1-20: microseconds on CPU; nothing to prune |
| "Which blob owns this point" | after labelling | already known exactly: every point carries its label; an RT lookup would replace a load |
| Boundary-pair grouping (`hash_group.comp`) as an RT index | GPU | keys change **every frame**, so the BVH is rebuilt per frame by driver compute shaders - a build over ~65k primitives is itself a sort/reduction pass at least as expensive as the ~20 us hash insert it replaces; RTIndeX's own follow-up names updateability as its weakness |
| Union-find / CCL as traversal | GPU | the survey's BFS case: latency-bound, all nodes visited, no pruning; no RT-CCL paper exists |
| Nearest-code decode | CPU | Hamming space is 36-52-dimensional binary; RT is 3-D FP32. Only approximate high-dimensional NN (JUNO) maps, and exact decode cannot be approximate |
| Tag sampling | CPU | a gather with known coordinates, nothing to search |
| Pose / multi-tag geometry | CPU | a handful of points |

- **Verdict:** reject on both devices; Mali-G610 has no RT hardware or API (0.3), and on the
  RX 9060 XT no stage contains a spatial search with large N over a reusable primitive set.
  Measuring is unnecessary: there is no candidate kernel to build.
- Sources: https://arxiv.org/html/2603.28771v1 ; RTNN https://arxiv.org/pdf/2201.01366 ;
  RTIndeX https://arxiv.org/abs/2303.01139v2 ; cgRX https://arxiv.org/abs/2406.03965v1 ;
  RayJoin https://github.com/pwrliang/RayJoin ; Chips and Cheese, RDNA 4 raytracing
  https://chipsandcheese.com/p/rdna-4s-raytracing-improvements

## 4. Where the answer changes (named thresholds)

### 4.1 Large tag families or max Hamming >= 3: GPU brute-force decode - but popcount, not WMMA
Upstream's hash table grows as `ncodes x C(nbits, <= maxhamming)` (ordered-pair count in the
code, x3 load factor, 16-byte entries; `apriltag.c:196-215`):

| Family (codes, bits) | max Hamming 1 | max Hamming 2 | max Hamming 3 |
| --- | --- | --- | --- |
| tag36h11 (587, 36) - FRC | 1.1 MB | 36 MB | 1.2 GB |
| tagStandard41h12 (2,115, 41) | 4.3 MB | 171 MB | 6.7 GB |
| tagStandard52h13 (48,714, 52) | 124 MB | **6.3 GB** | infeasible |
| tagCircle49h12 (65,535, 49) | 157 MB | **7.6 GB** | infeasible |

(Computed from the formula; the build also takes `capacity` hash inserts on the CPU.) From roughly
10^4 codes at Hamming 2, or tag36h11 at Hamming 3, the table stops being practical on an Orange Pi
(4-16 GB boards) and brute force becomes the sane decoder. Brute force for tagStandard52h13: 300 queries x
48,714 codes = 14.6M pairs.
- **Packed `bitCount(q ^ c)`** (52 bits in a 64-bit word, ~4 lane-ops per pair): 58M lane-ops =
  **~9 us on the RX 9060 XT, ~0.23 ms on the Mali-G610** (estimates from peak lane rates in 0.4).
- **u8 WMMA** (K padded 52 -> 64): 935M MAC = **~9 us at peak** on the RX 9060 XT. A tie with
  packed popcount, because without a 1-bit MMA mode the unit spends 8 bits per code bit.
- **int8 `dot4`** (both GPUs): 4 code bits per lane-op vs ~13 for packed popcount: 3x **worse**.
So the large-family case is a real reason to move decode onto the GPU, and the right kernel is
plain packed popcount on either device; the matrix unit adds nothing. Only relevant if a user needs
those families; FRC uses tag36h11.

### 4.2 Soft-decision (correlation) decoding over a large family: the one place WMMA wins
If decode were changed from hard bits to maximum-likelihood correlation of the sampled intensities
with +-1 codewords (not upstream's behaviour, so not parity-preserving), it becomes a genuine dense
real matmul: queries x codes x bits.
- tag36h11: 300 x 587 x 36 = 6.3M FMA = ~1 us of fp32 ALU on the RX 9060 XT. No case for WMMA.
- tagStandard52h13: 300 x 48,714 x 52 = **760M FMA = ~0.12 ms fp32 ALU vs ~15 us f16 WMMA** at
  peak (~8x), and ~3 ms on the Mali's ALU (512 GFLOPS), which has no unit. f16 inputs (11-bit
  mantissa) are adequate for 8-bit samples; accumulate in f32.
- Precondition: a recall benefit from soft decoding that justifies leaving libapriltag parity, plus
  a large family. Neither holds for the FRC deployment. **Verdict: measure only if the product
  requirements change.**

### 4.3 A learned front end
A CNN corner/tag detector is the workload these units exist for. YOLOv8n is 8.7 GFLOPs at 640x640
(Ultralytics model card): ~85 us at the RX 9060 XT's dense FP16 WMMA peak, realistically
0.2-0.4 ms at the 20-40% efficiency small-batch convnets reach - comparable to or slower than the
whole current desktop GPU phase (~217-380 us), with ~3x worse corner accuracy (round 1, `02`:
ArUco 0.72 px vs 1.9-2.1 px for learned pipelines). On the Pi there is no matrix unit; the NPU runs
YOLOv8n in ~13.6 ms vs 2.70 ms for the whole `Detect()` (round 1, 00-SUMMARY section 2).
**Worthwhile only if the goal becomes recall under blur/occlusion rather than speed**, and then on
the desktop (WMMA) or a future Arm GPU with neural accelerators (`VK_ARM_tensors` /
`VK_ARM_data_graph`), not on the RK3588.

### 4.4 Much larger images - does not change any verdict
Every stage that scales with pixels (threshold min/max, labelling, boundary, extents atomics, sort)
is either not a (+,x) product or is irregular/latency-bound; the dense closure in 2.2 scales
*worse* than union-find. At 4K or decimation 1 the matrix unit is still mapped onto nothing. The
only array that would get large enough for a TCU scan to matter is a per-pixel scan, and the
pipeline has none (`scan_block` runs over blobs).

### 4.5 RK3588 NPU (comparison point only; another agent covers it)
Even for the one matmul-shaped job above (4.1 brute force, 935M MAC), the NPU's 6 TOPS int8
(~3 T MAC/s) gives ~0.3 ms at peak, before RKNN invocation latency and a host round trip; the Mali
does the same job with packed popcount in ~0.23 ms in the same command buffer. No case.

## 5. Literature summary (non-ML use of matrix and RT units, 2018-2026)

| Work | Unit / hardware | Task | Reported speedup | Preconditions that fail here |
| --- | --- | --- | --- | --- |
| Dakkak et al., ICS'19 (arXiv 1811.09736) | NVIDIA V100 tensor cores, fp16 | segmented reduction, scan | up to 100x (reduction), 3x (scan), small segments | large arrays, fp16-exact values, contiguous segments; our reductions are keyed scatters and our scans are ~10 us |
| Ascend scan (arXiv 2505.15112) | Huawei cube units | scan, radix sort | 5-9.6x scan, 3.3x radix sort | large inputs; our sort is per-blob <= 1,024 keys in LDS |
| ConvStencil PPoPP'24 / SPIDER 2025 | NVIDIA tensor cores (fp64 / sparse) | stencils | outperforms prior stencil frameworks; box stencils cost 2.1-5.9x the minimal FLOPs | needs (+,x) weighted sums; our only stencils are min/max |
| BLEST 2025, TC-MIS 2026, ToT PPoPP'25 | NVIDIA bit / sparse MMA | BFS, MIS (2.8-18.8x), triangle counting (11.6x) | rely on `b1`/sparse MMA modes not exposed by Vulkan on RDNA4 |
| Li & Su, TPDS'21 (arXiv 2006.16578) | NVIDIA bit tensor cores | BNN (`POPC(A XOR B)` MMA) | 77% faster than prior BNN | 1-bit MMA, NVIDIA only |
| RTNN PPoPP'22 | NVIDIA RT cores | fixed-radius NN | 2.2-65x | static point set, many queries, spatial pruning |
| RTIndeX VLDB'23, cgRX 2024 | NVIDIA RT cores | DB point/range lookups | "competitive"; memory overhead, poor updateability | per-frame rebuild here |
| RayJoin ICS'24 | NVIDIA RT cores | point-in-polygon, segment intersection | 3.0-28.3x | millions of polygons; our D ~ 1-20 detections |
| RT survey (arXiv 2603.28771, 2026) | RT cores generally | 30+ problems | kNN best ~200x; BFS no gain | "no research ... on [AMD] GPUs"; latency-bound problems lose |

No paper was found applying matrix or RT units to connected-component labelling, fiducial
detection, or Hamming decode of fiducial codebooks.

