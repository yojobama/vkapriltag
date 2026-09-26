# 05 - Scan, compaction, sort

Entry format: Title / URL / Technique / Reported speed-up / Accuracy-determinism / Applicability to vkapriltag / Verification.
Cache root: C:/Users/yojob/AppData/Local/Temp/claude-research/

## Status

- Covered (21 entries): merrill_lookback, onesweep, progress2109, dfallback_paper.tex, gs_mobile (=Texture3dgs), hou_segsort, b0nes164 GPUSorting+GPUPrefixSums READMEs, Levien 2020+2021 blog, vulkan_radix_sort/VkRadixSort/FidelityFX, Bakunas-Milanowski ordered compaction, Kobus 2023 (abstract only), Vulkan-Docs #2233, Arm BP 3.4 (bp34.txt), optimal-depth networks + merge path (summaries), in-tree tie-order code read, Satish 2009, Vello PR #685, poniesandlight bitonic + own warp simulation (comparator enumeration), CUB DeviceSegmentedSort (source), last-block-done / scan-chain removal, Green merge path (abstract). Also read the azhirnov Valhall notes (mali/azh_valhall.md) and folded them into the entries.
- Conclusions FINAL (2026-09-26). The session's WebSearch budget is used up.
- Next up (if resumed): Stehle-Jacobsen 2017 hybrid radix sort; the Arm community blog on atomics; open the Green 2012 ICS paper. None is expected to change the ranking.

## In-tree baseline (read before the sources)

- `scan_block.comp` + `scan_add_offsets.comp`: classic reduce-then-scan style multi-level
  Hillis-Steele scan (log2(wg) barriered rounds, 2 barriers each) over `max_blobs` entries
  (default capacity; 8x shrink moved `blob_scan` 0.0104 -> 0.0063 ms on RX 9060 XT).
  Mali `blob_scan` span measured 0.086 ms on Intel / 0.023 ms on MX230 in an older table;
  on RDNA the entire chain is ~10 us. Only one scan in the frame (blob point offsets).
- `blob_diff_body.glsl`: unordered atomicAdd append of boundary points (one global counter).
  Subgroup-aggregated append variant was deleted (24% slower on RX 9060 XT; integrated GPUs
  exclude subgroup variants outright). Order is nondeterministic (PERFORMANCE.md section 7).
- `select_blobs.comp`: atomicAdd compaction of blobs passing filters into `selected[]`
  (so blob order - and which blobs survive if > max_blobs - depends on scheduling).
- `scatter_index_points.comp`: per-blob atomic cursor (counting-sort scatter into ranges
  given by the scan). Intra-blob order is arrival-dependent, then overwritten by the sort,
  except ties on equal packed key (key<<12 | local index - index is arrival order!).
- `sort_points_local_body.glsl`: one workgroup per blob, Batcher odd-even mergesort over
  next-pow2(count) packed 32-bit words in shared memory; log2(cap)(log2(cap)+1)/2
  barriered rounds; kLocalCap up to 4096; oversize blobs fall back to unsorted.
  Shipped: -2..-3% Mali GPU total vs bitonic. Barrier-removal ceiling measured on
  MX230/Intel only: 5-9% of the sort span (A9 rejected). `sort` span on Mali =
  ~0.39 ms of 3.46 ms (~11%), bandwidth-insensitive (0.99x over 4x DMC sweep).

## Entries

### Single-pass Parallel Prefix Scan with Decoupled Look-back — D. Merrill, M. Garland (NVIDIA), NVR-2016-002 (2016)
- URL: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back (cached `ccl/scansort/merrill_lookback.txt`)
- Technique: Each workgroup ("partition") publishes a status descriptor {flag X/A/P, aggregate, inclusive_prefix}. After reducing its tile it publishes A+aggregate, then walks predecessors backwards accumulating aggregates until it meets a P (inclusive prefix), publishes P, and does its local scan seeded by that prefix. ~2n traffic (vs ~3n reduce-then-scan, ~4n scan-then-propagate), one pass. Partition IDs must be taken from an atomic counter at activation (not gl_WorkGroupID) or it deadlocks when later groups occupy the machine while earlier ones are unscheduled. The same pass fuses select-if / partition-if / reduce-by-key / RLE: i.e. ORDER-PRESERVING stream compaction in one kernel.
- Reported speed-up: matches memcpy throughput on Kepler/Maxwell for large n; harmonic-mean speed-ups at saturation 1.1x vs StreamScan (chained scan), 1.4x vs MGPU (reduce-then-scan), 2.3x vs Thrust (Tesla C2050 / K40 / M40).
- Accuracy / determinism impact: Scan result is deterministic (integer add). Compaction built on it is ORDER-PRESERVING, hence fully deterministic output order — exactly the property PERFORMANCE.md section 7 says `blob_diff` lacks.
- Applicability to vkapriltag: Paper's own Safety property says it "will run to completion if the system guarantees forward-progress for all processors"; it explicitly assumes a fair scheduler for minimal waiting. Neither Vulkan nor Mali/Adreno/Apple guarantee inter-workgroup forward progress (see later entries). The blob scan is only max_blobs (default thousands of) uints — latency-bound, ~10 us chain on RDNA — so the 2n vs 3n traffic argument is irrelevant here; the gain would be removing 2 dispatches + barriers from the chain (on Mali each barrier 2.6-18.7 us, but these follow tiny dispatches so ~2.6 us). For `blob_diff` the interesting part is ordered compaction, which needs a scan over ~1-2M interior pixels (the dense grid) — that is where decoupled look-back would be used, and where forward-progress risk bites.
- Verification: [verified from primary source]

### Onesweep: A Faster Least Significant Digit Radix Sort for GPUs — A. Adinets, D. Merrill (NVIDIA), arXiv 2206.01784 (June 2022)
- URL: https://arxiv.org/abs/2206.01784 (cached `ccl/scansort/onesweep.txt`)
- Technique: LSD radix sort where each digit-binning pass is ONE kernel: an upfront kernel builds all p digit histograms at once (shared-memory atomics, privatized), a tiny kernel exclusive-scans them, then p chained-scan passes each fuse 256 concurrent decoupled look-back scans (one per bin) with the scatter. Look-back descriptors are single 32-bit words, 2-bit status (Not ready / Local count / Global sum) + 30-bit value, so publish is one store with no fence ordering between flag and value. Tile IDs from a global atomic counter "ensures that tiles are processed" in launch order (the deadlock-avoidance rule from Merrill 2016). ~(2p+1)n memory ops vs ~3pn.
- Reported speed-up: 29.4 GKey/s for 256M random uint32 on A100 (clocks locked 1410 MHz), ~1.5x vs CUB reduce-then-scan LSD; 1.4-1.6x across distributions; up to 36 GKey/s.
- Accuracy / determinism impact: Stable, deterministic. Paper states the reason atomics are NOT used for binning: "their update-order is non-deterministic, preventing them from being used in stable LSD digit binning" and performance is "capricious" under low digit diversity (contention).
- Applicability to vkapriltag: vkapriltag no longer has a global sort (hash grouping replaced it; OPTIMIZATION_NOTES items 4-6: sort+group 15.8 -> 3.06 ms). Two transferable pieces: (1) the packed status|value 32-bit descriptor, which lets a look-back scan run on Vulkan with only 32-bit atomics/stores and no fence between flag and payload (point counts <2^30 fit); (2) the explicit statement that atomic scatter is order-nondeterministic, which is the same root cause as PERFORMANCE.md section 7's +/-1-2 quad jitter. Onesweep itself depends on forward progress (see next entries) and is tuned for n >> 10^6; the per-blob sort here is 16-4096 keys - out of Onesweep's regime.
- Verification: [verified from primary source]

### Specifying and Testing GPU Workgroup Progress Models — T. Sorensen, L. Salvador, H. Raval, H. Evrard, J. Wickerson, M. Martonosi, A. Donaldson, OOPSLA 2021 (arXiv 2109.06132)
- URL: https://arxiv.org/abs/2109.06132 (cached `ccl/scansort/progress2109.txt`)
- Technique: Formalizes progress models (fair; OBE = workgroups that have taken >=1 step are fairly scheduled; HSA = lowest unfinished id guaranteed progress; LOBE = OBE plus every lower id than a started one also gets progress). Synthesizes 483 progress litmus tests and runs them on 8 GPUs / 5 vendors: GF940m, Quadro RTX 4000, Apple A12, A14 (Metal), Intel HD620 (Mesa), Mali-G77 MP11 (Vulkan 1.1, Galaxy S20 Exynos), Adreno 620 (Vulkan 1.1, Pixel 5), Jetson TX1.
- Reported speed-up: n/a (correctness study). Key results: Vulkan and Metal "say almost nothing about relative forward progress guarantees between workgroups". All GPUs passed all weak-LOBE tests EXCEPT Apple A12 and Mali-G77, which failed the same 11 weak-LOBE conformance tests (A12 deterministically on 10/11; G77 non-deterministically, "over half of the runs on all 11 tests"). No violation of weak OBE or HSA observed on any device in the campaign, but Apple supplied a spin-lock (CAS mutex) test that DOES violate OBE on A12 and A14. ~246 tests fail deterministically everywhere under the "chunked" heuristic, consistent with non-preemptive in-order workgroup dispatch. Some tests needed hard reboots after repeated timeouts.
- Accuracy / determinism impact: n/a directly; defines when a spin-waiting scan can hang.
- Applicability to vkapriltag: Decoupled look-back with atomic-ticket tile IDs needs, formally, that predecessor workgroups which have already started keep making progress while a later one spins (OBE-like, for workgroups resident concurrently). Mali-G77 (Valhall, same family as the target's G610) failed LOBE non-deterministically; Mali behaviour under OBE held in their suite, but Apple shows OBE is not universal. Conclusion for vkapriltag: a spin-waiting single-pass scan is NOT portable-safe on Mali/Apple by specification, and on Mali only "empirically usually OK". A hang here is VK_ERROR_DEVICE_LOST (the tree already hit that once on the G610 - OPTIMIZATION_NOTES item 8), so any look-back must have a non-blocking fallback (next entry) or not spin at all.
- Verification: [verified from primary source]

### Decoupled Fallback: A Portable Single-Pass GPU Scan — T. Smith, R. Levien, J. D. Owens, SPAA '25 (July 2025)
- URL: SPAA 2025 proceedings; code lineage = b0nes164/GPUPrefixSums (cached LaTeX source `ccl/scansort/dfallback_paper.tex`; the cached `dfallback.pdf` is only 5.5 KB, i.e. not the real PDF)
- Technique: Chained scan (tile id from an atomic bump counter) with decoupled look-back, but when the lookback thread has spun MAX_SPIN (=4) times on a Not-Ready predecessor, the WHOLE WORKGROUP re-reads that predecessor tile's input and reduces it itself (work stealing), then atomically tries to post it as Ready (skipped if the tile has since gone Inclusive; they use an atomic op that cannot regress state), and continues the look-back. Termination no longer depends on any inter-workgroup progress guarantee. Also: tile state+payload split across 2 or 4 32-bit atomics, each carrying 2 state bits + 16 payload bits; the message is accepted only when all parts show the same state -> works with relaxed 32-bit atomics only, no 64-bit atomics and no memory barriers. Workgroup-wide scan = radix-s Ladner-Fischer over Kogge-Stone subgroup scans, subgroup-size-agnostic (4..128). Config: 256 threads, 16 elem/thread, tile 4096, ~1 KB shared.
- Reported speed-up: 2^25 u32 inclusive prefix sum, WGSL on Dawn. DF vs reduce-then-scan (RTS): Mali-G78 MP20 (Pixel 6a, "no FPG") 2.725 vs 2.025 Gelem/s = 1.346x, and 103.8% of memcpy; Apple M3 1.457x; M1 Max 1.431x; Intel HD620 1.426x; RTX 2080 Super 1.491x; RX 7900 XT 1.331x (113% of memcpy). Mali-G78 had the most unfair scheduling: 4702 fallbacks initiated (57.8% of workgroups), 560 successful insertions, mean 2.5 spins/WG, mean lookback length 3.5. Plain decoupled look-back on the M1 Max without fallback: histogram shows runs slower than RTS and many past the Windows TDR (2 s) threshold, i.e. effectively hangs. Simulated blocking up to 1-in-2 tiles keeps DF >= 85% of its unblocked speed (RTS ~ 66%).
- Accuracy / determinism impact: Exact and deterministic for integer sum; fallback re-computes the same reduction. Enables ordered in-situ compaction ("in-situ compaction ... avoiding redundant writes or secondary compaction passes").
- Applicability to vkapriltag: This is the answer to "is single-pass safe on Mali?": plain decoupled look-back is NOT safe (no FPG on ARM/Apple per Sorensen 2021; Mali-G78 empirically very unfair), decoupled FALLBACK is safe and still 1.35x over RTS on a Valhall Mali (G78, same generation family as G610). But the win is a memory-traffic win (3n -> 2n) at 2^25 elements. vkapriltag's only scan is max_blobs (thousands) of counts: at that size it is 1 tile, so the whole chain could simply be ONE workgroup doing the scan (no look-back needed at all). The real candidate use is an ORDERED boundary-point compaction in blob_diff (a fused select-if over ~1M interior pixels, 4 candidates each), replacing atomicAdd append -> deterministic point order (fixes PERFORMANCE.md section 7) at roughly the same pass count. Cost risk: Mali atomics/latency bound spans; the fallback path re-reads input (parent[] words). Needs subgroup ops for the fast intra-WG part, which are disabled on Mali in-tree (6x slower reduce-by-key) - a shared-memory-only variant would be needed there.
- Verification: [verified from primary source] (LaTeX source; numbers read from the results table)

### Optimizing 3D Gaussian Splatting for Mobile GPUs (Texture3dgs) — M. M. R. Sanim, Z. Shu, ..., W. Niu, B. Ren, G. Agrawal (UGA / William & Mary), arXiv 2511.16298 (Nov 2025)
- URL: https://arxiv.org/abs/2511.16298 (cached `ccl/scansort/gs_mobile.txt`; this is the "GPUSorting mobile notes" file of the brief - it is NOT b0nes164 GPUSorting)
- Technique: Global bitonic sort of 2^20-2^24 key-value pairs laid out in a 2D RGBA texture so each compare-exchange stage's partners fall in the same texture-cache block ("quad"-blocked layout, index transformation per stage), keys normalized to 32 bits, work-group access ranges shaped for the L1 texture cache. Motivation: mobile GPUs' read-only 2.5D texture cache is the best-cached path.
- Reported speed-up: sort alone 1.5-4x vs GPUTeraSort and TFLite-GPU sorts on Snapdragon 8 Gen 2 (Adreno 740 class); 1.10-1.15x vs VKRadixSort (the Vulkan radix sort used by 3dgs.cpp); end-to-end 3DGS up to 1.7x (avg 1.25x) vs 3dgs.cpp. Portability plotted on Adreno 540 and Mali-G57 MC2 (figure only, no numbers in text).
- Accuracy / determinism impact: Sorting network - deterministic.
- Applicability to vkapriltag: Low. Sizes are 10^6+ keys (global sort), vkapriltag sorts <=4096 keys per workgroup in shared memory. The texture-cache angle was already rejected in-tree ("tiled image / texelFetch" on Mali). One useful datum: on Adreno a well-laid-out bitonic network beats a Vulkan radix sort only by 1.10-1.15x at 2^20+ keys, i.e. networks are competitive with radix on mobile even at large n, and at small n the network is the natural choice. Nothing here on Mali specifically.
- Verification: [verified from primary source] (for the Adreno numbers; the Mali portability result is only a figure)

### Fast Segmented Sort on GPUs — K. Hou, W. Liu, H. Wang, W. Feng (Virginia Tech / NBI / NTNU), ICS 2017
- URL: https://doi.org/10.1145/3079079.3079105 (cached `ccl/scansort/hou_segsort.txt`)
- Technique: (1) Bin segments by length with a warp-vote histogram + exclusive scan + atomic binning, then launch a different kernel per bin (unit-bin: 1 thread; warp-bins: reg-sort; block-bins: reg-sort + shared-memory merge; grid-bin: multi-block). 13 bins covered all segments in their data. (2) reg-sort: N elements over M<=32 threads, each thread holding N/M elements in REGISTERS; bitonic compare-exchanges are classified into exch_local (both elements in the same thread's registers - no communication at all), exch_intxn / exch_paral (shfl_xor between lanes), with an in-register transpose for coalesced write-out. Reg-sort handles segments up to 256 pairs on Kepler / 512 on Pascal; up to 2048 via smem-merge (MergePath split points so each thread merges an equal number of outputs); >2048 grid-bin.
- Reported speed-up: vs cub-segsort up to 63.3x (K80) / 86.1x (TitanX) for short segments ("high cost for radix sort on short segments"); vs cusp-segsort up to 12.5x / 16.5x; vs mgpu-segsort (merge-based) up to 3.0x / 3.8x, similar performance in its best regime; power-law segment lengths ~1.9x.
- Accuracy / determinism impact: Deterministic (networks + merge path).
- Applicability to vkapriltag: The per-blob angular sort IS a segmented sort (segments 16..4096, mean ~112 at 1280x800 dec 2 per the ideas doc, 233 at 1080p). Two transferable ideas, both unexplored in-tree: (a) Length binning - today every blob gets a full workgroup (128 threads on Mali, 256 on RDNA) padded to next-pow2; a blob with 20 points still pays a workgroup launch, 15 barriered rounds for cap=32 with 96+ idle threads. Putting several small blobs in one workgroup (one subgroup/lane group per blob, or even thread-per-blob insertion sort for count <= 16) and full workgroups only for big blobs cuts workgroups and barriers. (b) Register-resident per-thread chunks: each thread sorts k consecutive keys in registers (exch_local, no barrier, no shared-memory traffic) and only the cross-thread strides go through shared memory. On Mali shared memory is L2-backed, so the ~2 loads+2 stores per compare-exchange of the current shader are L2 transactions; register-local rounds remove them. CAUTION: the in-tree lesson is that "fewer, fatter threads" lost twice on Mali (+6%, +11%); but those were global-memory latency-bound shaders, while the sort is shared(=L2)-memory-traffic + barrier bound, and the barrier-removal ceiling (5-9% of span) was measured on MX230/Intel, NOT Mali. Needs a Mali-specific bound first (see Conclusions). Subgroup shuffles (exch_intxn) are out on Mali given the in-tree subgroup results; the exch_local part needs no subgroup ops.
- Verification: [verified from primary source]

### GPUSorting and GPUPrefixSums (portable HLSL/WGSL/CUDA sorts and scans) — Thomas Smith (b0nes164), GitHub, 2023-2025
- URL: https://github.com/b0nes164/GPUSorting , https://github.com/b0nes164/GPUPrefixSums (READMEs cached `ccl/scansort/web/gpusorting_readme.md`, `gpuprefixsums_readme.md`)
- Technique: Portable ports of CUB's DeviceRadixSort (reduce-then-scan) and OneSweep (chained scan + decoupled look-back), wave-size-agnostic via runtime logic (tested wave 4/16/32/64). GPUPrefixSums surveys block scans (Kogge-Stone, Sklansky, Brent-Kung, raking reduce-scan, warp-sized-radix Brent-Kung/Sklansky) and contributes Decoupled Fallback (spin-limit + CAS-posted redundant reduction). Also "SplitSort", a hybrid radix-merge SEGMENTED sort, strongest for 16-bit keys and for max segment length < 256 on 32-bit keys (proof of concept; benchmarked with the Kobus et al. segmented-sort suite).
- Reported speed-up: all benchmark results are images (not readable here). Text claims: OneSweep "relies on forward thread-progress guarantees ... tends to run on anything that is not mobile, a software rasterizer, or Apple. Use OneSweep at your own risk"; DeviceRadixSort (reduce-then-scan) "should be used whenever portability is a concern". Decoupled Fallback "should allow devices without forward thread progress guarantees to perform the scan without crashing"; the up-to-date version lives in Vello.
- Accuracy / determinism impact: Radix sorts here are stable/deterministic (no atomic scatter).
- Applicability to vkapriltag: Confirms the rule of thumb for Mali: never ship an unguarded look-back scan; either reduce-then-scan (what scan_block/scan_add_offsets already are) or Decoupled Fallback. SplitSort's regime (segments < 256, narrow keys) matches the per-blob sort (mean ~112-233 points, 20-bit theta key) - worth reading the Zulip write-up if a segmented radix is ever considered, but on Mali a shared-memory radix means shared atomics / scans on L2-backed shared memory, which is the same cost class as the current network.
- Verification: [verified from primary source] for README text; [secondary/unverified] for any numbers (in images only)

### Prefix sum on Vulkan (2020) and Prefix sum on portable compute shaders (2021) — Raph Levien, blog
- URL: https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html ; https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html
- Technique: Decoupled look-back in Vulkan GLSL using the Vulkan memory model (acquire/release atomics at device scope); a "compatibility" tree-reduction (reduce-then-scan) variant with barriers only; discussion of Elias Naur's "scalar progress" fallback (when waiting, do a bit of the predecessor's reduction yourself) - the precursor of Decoupled Fallback.
- Reported speed-up: 2020: 31.2 G elem/s for 64M u32 on GTX 1080 (~262 of 320 GB/s); look-back ~10-15% of total time. 2021: compatibility (tree) mode ~48 G elem/s on RX 5700 XT, "basically the same as memcpy"; decoupled look-back expected ~1.5x over tree reduction in general.
- Accuracy / determinism impact: Deterministic sums.
- Applicability to vkapriltag: States plainly that the look-back "depends on other workgroups making forward progress while it's waiting", Apple mobile GPUs lack occupancy-bound guarantees, and (2021, citing Sorensen) "Apple and ARM exhibit failures of forward progress" so Vulkan can only expose it as an optional property. Also notes decoupled look-back was impossible on Metal without 32-bit packing. Note the 5700 XT result: on a desktop RDNA the barrier-based tree scan already hits memcpy speed for large n, so for a tiny scan like the blob scan the choice of global scan algorithm is irrelevant; what matters is dispatch/barrier count. Subgroup-size pitfalls (Intel reports gl_SubgroupSize 32 regardless; need VK_EXT_subgroup_size_control) match the tree's A9 rejection reasoning.
- Verification: [verified from primary source] (via WebFetch summaries of both posts)

### Vulkan radix sorts: jaesung-cs/vulkan_radix_sort, MircoWerner/VkRadixSort, AMD FidelityFX Parallel Sort — various, 2020-2025
- URL: https://github.com/jaesung-cs/vulkan_radix_sort ; https://github.com/MircoWerner/VkRadixSort ; https://gpuopen.com/fidelityfx-parallel-sort/
- Technique: All three are reduce-then-scan LSD radix sorts (no look-back, hence no forward-progress dependency). vulkan_radix_sort: header-only, 8-bit digits, needs subgroup size 32 or 64 and >= 20 KB shared memory. VkRadixSort: a single-workgroup variant ("good performance for fewer elements (<10k)") and a multi-workgroup histogram+sort pair run 4x (8-bit digits). FidelityFX: 4-bit digits, 8 iterations for 32-bit keys, each = histogram, reduce, scan, global scan, scatter; SM6 wave ops, tuned for RDNA.
- Reported speed-up: vulkan_radix_sort on RTX 5080, N=2^25: 16.26 GItems/s keys-only, 9.89 key-value; CUB OneSweep is 1.39x / 1.19x faster. VkRadixSort single-WG 1M elements 18.973 ms on RTX 3070. FidelityFX page gives no numbers.
- Accuracy / determinism impact: Stable, deterministic.
- Applicability to vkapriltag: None of these is a fit. vulkan_radix_sort explicitly: "mobile GPUs with a subgroup size of 16 or smaller aren't supported" (Mali-G610 is fixed 16). They sort one large array; vkapriltag replaced exactly this kind of global radix sort with hash grouping + per-blob local sort (OPTIMIZATION_NOTES items 4-6), which was a large win. The only relevant data point is VkRadixSort's single-workgroup mode being its recommended path below ~10k keys - consistent with keeping everything per-blob inside one workgroup.
- Verification: [verified from primary source] (READMEs/product page via WebFetch)

### Efficient Algorithms for Stream Compaction on GPUs — D. Bakunas-Milanowski, V. Rego, J. Sang, Y. Chansu (Cleveland State / Purdue), IJNC 7(2), 2017
- URL: https://www.jstage.jst.go.jp/article/ijnc/7/2/7_208/_pdf/-char/en (cached `ccl/scansort/web/bakunas.txt`)
- Technique: (1) Order-preserving compaction in 3 kernels: each warp evaluates the predicate for 32 subgroups of 32 elements, ballot+popc counts, shuffle-reduce -> per-1024-element counts; scan the counts; then each warp re-evaluates and writes at group_offset + subgroup_offset + popc(ballot & lanemask_lt). (2) Non-order-preserving "hybrid": the same in-warp ballot/popc/shuffle-scan, but the group offset comes from ONE atomicAdd per warp per 1024 elements, single kernel.
- Reported speed-up: vs Thrust copy_if on Tesla K40 / Quadro K620 (128M elements): order-preserving 3.7x, hybrid 5.6x faster; hybrid ">120x" vs sequential CPU. So on Kepler the ORDER-PRESERVING version is ~1.5x slower than the atomic hybrid (5.6/3.7). Best block size 128 for all but InK-Compact.
- Accuracy / determinism impact: Method (1) gives deterministic output order; (2) does not (warp-group order depends on atomic arrival, though within a 1024-block order is preserved).
- Applicability to vkapriltag: Gives a price for determinism: an ordered (scan-based) boundary-point compaction costs an extra pass over the predicate domain (~1M interior pixels at 1280x800 dec2 x 4 directions) - i.e. blob_diff runs twice (count then write) or writes a dense per-tile count. `boundary` is 0.36 ms on Mali and only 1.09x DMC-sensitive, so a second evaluation pass could cost up to ~+0.3 ms (~9% GPU) there - a bad trade just for determinism. Also relies on ballot/popc (subgroup ballot), which the tree excludes on integrated GPUs. Conclusion: do NOT make blob_diff ordered; get determinism downstream instead (see Conclusions: deterministic tie-break in the sort).
- Verification: [verified from primary source]

### Faster Segmented Sort on GPUs — R. Kobus, J. Nelgen, V. Henkys, B. Schmidt (JGU Mainz), Euro-Par 2023
- URL: https://doi.org/10.1007/978-3-031-39698-4_45 ; artifact https://springernature.figshare.com/articles/code/Artifact_for_Euro-Par_2023_Paper_Faster_Segmented_Sort_on_GPUs_/23540553
- Technique: Re-tunes Hou et al.'s bin-by-length segmented sort per GPU (optimal bin boundaries / threads-per-segment setups determined from measured runtimes), improves specific segment-length regimes, and adds a keys-only variant (no value payload moved through the network).
- Reported speed-up: average 1.26-1.35x over the original Hou et al. implementation across RTX 4090, V100, A100, GTX 1080 (from abstract; full text paywalled, not read).
- Accuracy / determinism impact: Deterministic.
- Applicability to vkapriltag: Reinforces two points: (a) the best configuration is per-segment-length AND per-GPU - so any binning in sort_points_local should be tuned on the Mali board, not the RX 9060 XT; (b) keys-only helps because payload movement costs - vkapriltag already does the equivalent by packing theta_key|local_index into one 32-bit word (the payload is fetched once after the sort). Nothing Mali-specific.
- Verification: [secondary/unverified] (abstract via search summary only; paper not opened)

### [Roadmap Feedback] Forward Progress Guarantees for Compute Workgroups — KhronosGroup/Vulkan-Docs issue #2233 (devshgraphicsprogramming), opened 2023-09-22
- URL: https://github.com/KhronosGroup/Vulkan-Docs/issues/2233
- Technique: Request that Vulkan guarantee that a started workgroup eventually resumes (needed for inter-workgroup sync such as chained scans / persistent work queues). Mentions UE5 Nanite disabling its ordered-dispatch-dependent path on PC and keeping it on consoles "because only there you can guarantee an ordered dispatch".
- Reported speed-up: n/a.
- Accuracy / determinism impact: n/a.
- Applicability to vkapriltag: Status open, no vendor response (as fetched). Confirms that as of the fetch there is no Vulkan core feature or extension a Mali driver could advertise to make a spin-waiting look-back legal; any single-pass scan in vkapriltag must be correct under zero inter-workgroup progress guarantees (i.e. Decoupled Fallback or no spinning).
- Verification: [verified from primary source] (issue page via WebFetch; I did not see the full comment thread, so "no vendor response" means none in the fetched content)

### Arm GPU Best Practices Developer Guide 3.4 (101897_0304_10), sections 9.2 Workgroup sizes, 9.3 Shared memory, 10.13 Atomics — Arm, 2025
- URL: https://developer.arm.com/documentation/101897/latest (cached `mali/bp34.txt`, lines ~4546-4672, 5267-5310)
- Technique / guidance: Mali (Midgard/Bifrost/Valhall) has NO dedicated shared memory: "system RAM that is backed up by the load-store cache". With barriers or shared memory, the whole workgroup must be co-resident (hardware cannot split/merge it). Recommendations: 64 as baseline workgroup size, "Do not use more than 64 threads per workgroup", "For barriers, smaller workgroups are less expensive", "It can be computationally cheaper splitting an algorithm over multiple shaders when compared to inserting barriers", "Do not copy data from global memory to shared memory on Arm GPUs", subgroup sizes are 16 or smaller so barrier-avoiding warp-synchronous tricks are unsafe. Atomics: contention across shader cores forces L2 snooping; space atomics 64 bytes apart; accumulate in a shared-memory atomic and push one global atomic per workgroup; "If better solutions that use multiple passes are available, then do not use atomics."
- Reported speed-up: none (guidance).
- Accuracy / determinism impact: n/a.
- Applicability to vkapriltag: CONTRADICTED in-tree for the sort: OPTIMIZATION_NOTES item 5 measured sort_points_local on the G610 at 64/128/256/512/1024 threads per blob = 5.71/3.83/3.08/3.08/4.03 ms (older 1080p state), so wider-than-64 workgroups won for one-blob-per-workgroup. The Arm guidance is still informative for WHY the sort costs what it does: every compare-exchange round is shared-memory (= L1/L2 load-store cache) traffic plus a full-workgroup barrier, and the whole 256-thread workgroup is pinned while a blob with cap=32 uses 32 lanes. That points to packing several small blobs into one workgroup (fills idle lanes without shrinking the workgroup) rather than shrinking the workgroup. On atomics: scatter_index_points' per-blob cursor and blob_diff's single global counter are exactly the "contended cache line" case; the guidance "spread / shared-atomic then one global atomic" is the subgroup-free form of warp aggregation (the subgroup form was measured slower in-tree; a SHARED-memory-atomic aggregation for blob_diff's counter has, as far as I can see in the notes, not been measured on Mali - unverified).
- Verification: [verified from primary source]

### Fewer-round sorting networks: optimal-depth networks (Bundala, Codish, Cruz-Filipe, Schneider-Kamp, Závodný; Ehlers & Müller) and merge-path merging (Green, McColl, Bader)
- URL: https://arxiv.org/abs/1412.5302 (Optimal-Depth Sorting Networks, JCSS 2017); https://arxiv.org/pdf/1501.06946 (New Bounds on Optimal Sorting Networks); merge path: Green, McColl, Bader, "GPU Merge Path", ICS 2012 (not opened)
- Technique: SAT-based proofs/synthesis of minimum-depth comparator networks. Depth is the number of barriered rounds in a shared-memory implementation. Merge path: split a merge of two sorted runs into equal-output-size pieces by binary search on the cross diagonal, so each thread merges a fixed number of outputs sequentially; a full sort = in-register chunk sort + log2(n/chunk) merge levels, ~1-2 barriers per level instead of O(log^2 n) rounds.
- Reported speed-up: n/a (depth results). Search summary confirms optimal depth for n=16 is 9 (no 8-layer network exists), vs 10 rounds for Batcher/bitonic at n=16 (log2 16 * 5 / 2 = 10). Known optimal-depth values are established only for small n (<= ~20); for 32..4096 nothing better than Batcher-class O(log^2 n) depth is practical.
- Accuracy / determinism impact: Any comparator network or merge path is deterministic given a total order on keys.
- Applicability to vkapriltag: Optimal-depth networks buy at most 1 round at cap=16 and nothing proven beyond ~20 - negligible (sort rounds for cap=128 are 28; for 2048, 66). The ways to cut ROUNDS materially are structural: (a) register-blocked bitonic (Hou reg-sort exch_local: with k elements per thread in a blocked layout, strides < k need no barrier; k=4 at cap=128 turns 13 of 28 rounds into register-only compare-exchanges) - note this maps to BITONIC's power-of-two strides, not to Batcher odd-even merge's odd strides (d = q - p), so it would partially give back the shipped 13-21% comparator saving; (b) merge path over register-sorted chunks: log2(cap/k) levels (5 at cap=128,k=4) of ~2 barriers each, but each merge step is a dependent shared-memory binary search - exactly the dependent-load pattern that is slow on Mali's L2-backed shared memory (OPTIMIZATION_NOTES tile-local UF finding). Rank-by-counting (each thread counts keys smaller than its own: O(n) broadcast reads, ONE barrier, trivially deterministic) is another small-n option for cap <= 64 - my own suggestion, no source found benchmarking it on GPUs.
- Verification: [secondary/unverified] (depth facts from search-result summaries of the SAT papers; merge path paper not opened; round counts computed by me from the shader's loop structure)

### In-tree check: where the sort's tie order actually comes from — vkapriltag shaders + docs, read 2026-09-26
- URL: `apriltags_vulkan/library/shaders/scatter_index_points.comp`, `sort_points_local_body.glsl`, `blob_diff_body.glsl`, `select_blobs.comp`; PERFORMANCE.md section 7; OPTIMIZATION_NOTES.md "Behavioural changes"
- Technique (code read, not a new technique):
  - scatter_index_points gives each point `rank = atomicAdd(blob_cursor[sel], 1)`, so a point's slot inside its blob's range depends on arrival order.
  - The sort packs `(min(theta_key, 2^20-1) << 12) | slot`. Equal theta_key values are therefore ordered by arrival.
  - After the sort, each output slot emits `RawLineFitPoint{xy2, W, blob}`, with W recomputed from the decimated image at (x2, y2). Two points with the same (x, y) therefore produce the SAME output record, whatever their order.
  - blob_diff does emit such duplicates: the SE and SW diagonal pairs of one 2x2 quad share the midpoint (2x+1, 2y+1), and gx/gy are dropped.
  - So only ties between DISTINCT points change the output. These are points on nearly the same ray from the offset centroid, within 1/250000 of a pseudo-angle unit.
  - select_blobs is also atomic-append. Selected-blob order is nondeterministic, but blob content is deterministic unless more than max_blobs blobs pass. max_blobs is deliberately kept large to avoid that (PERFORMANCE.md section 8, item 4).
- Reported speed-up: n/a.
- Accuracy / determinism impact:
  - The docs report 10 repeat runs byte-identical on the reference frame.
  - Changing the workgroup size moves candidate_quads by +/-1-2. The mechanism is exactly the distinct-point theta_key tie above.
- Applicability to vkapriltag: A cheap fix that makes the order a pure function of the data, with no ordered compaction:
  - After the network, add one pass in which each thread compares the high 20 bits of its word with its right neighbour's. A shared flag records whether any tie exists (one barrier).
  - Only when the flag is set, re-order each tie run by PackXY(src[slot]), e.g. a serial insertion sort by the run's first thread; runs are length 2-3 in practice.
  - Result: order depends only on (theta_key, x, y), and identical (x, y) is harmless as shown above.
  - Cost is one barrier plus one neighbour read per point, about one extra network round (1/28 of the rounds at cap = 128; a few µs on Mali).
  - More index bits would not help: ties need a secondary key, not a wider slot.
- Verification: [verified from primary source] (code read). Tie frequency NOT measured.

### Designing Efficient Sorting Algorithms for Manycore GPUs — N. Satish, M. Harris, M. Garland (NVIDIA), IPDPS 2009
- URL: https://research.nvidia.com/publication/2009-05_designing-efficient-sorting-algorithms-manycore-gpus (PDF cached `ccl/scansort/web/satish2009.pdf/.txt`)
- Technique: Merge sort with t = 256-thread blocks.
  - Tiles of t = 256 elements are sorted in shared memory with Batcher's ODD-EVEN merge sort. The paper chose it over bitonic "because our experiments show that it is roughly 5-10% faster in practice".
  - Sorted tiles are merged by RANK: element a_i of A lands at i + rank(a_i, B), where rank(a_i, B) comes from a binary search in B. All elements are placed in one parallel step with no data-dependent control flow beyond the search.
  - Large merges are split into t-element windows using splitters.
  - The paper argues bitonic "is often the fastest sort for small sequences", but that its O(n log^2 n) work and communication hurt at larger n.
- Reported speed-up (GTX 280): merge sort was the fastest comparison sort in the literature at the time; radix sort was up to 4x faster than GPUSort, >2x faster than other CUDA radix sorts, and 23% faster than a quad-core CPU sort. The 5-10% odd-even-vs-bitonic figure is for the in-shared-memory tile sort.
- Accuracy / determinism impact:
  - Networks are deterministic.
  - A rank merge is deterministic and stable if ties break asymmetrically: lower_bound for A's elements searching B, upper_bound for B's elements searching A.
- Applicability to vkapriltag:
  - (1) This independently corroborates the shipped bitonic -> odd-even change. 5-10% of a CUDA tile sort in 2009, vs the -2..-3% Mali GPU total (~-20% of the sort span) measured in-tree.
  - (2) A concrete fewer-barrier design for large blobs (cap 512-4096):
    - Network-sort 256-slot sub-tiles: 36 rounds.
    - Then log2(cap/256) rank-merge levels, each with ONE barrier. Every element does a binary search of <= log2(cap) dependent shared loads in the sibling run and writes to a ping-pong buffer.
    - At cap = 1024 that is 36 + 2 = 38 barriers instead of 55. At cap = 4096 it is 36 + 4 = 40 instead of 78.
    - Costs: 2x shared memory (ping-pong), which halves the max kLocalCap unless the merge output goes to registers first. The binary search is a chain of dependent L2-backed loads on Mali, the exact pattern that made tile-local UF slow there.
    - Mean blobs (~112-233 points) never reach the merge levels, so this only helps the few big blobs (tag borders at decimation 1). Low priority unless a size histogram shows the big blobs dominate the span. The workgroup-per-blob span is set by the longest workgroups, so this could matter for tail latency at decimation 1.
- Verification: [verified from primary source]

### Vello: shipped scans vs. the single-pass (Decoupled Fallback) PR — linebender/vello main tree + PR #685 (b0nes164, opened 2024, still OPEN at fetch 2026-09-26)
- URL: https://github.com/linebender/vello/pull/685 ; tree `vello_shaders/shader/pathtag_reduce*.wgsl`, `pathtag_scan*.wgsl`, `draw_reduce/leaf`, `clip_reduce/leaf`
- Technique:
  - Vello main still ships TREE (reduce-then-scan) scans: pathtag_reduce -> reduce2 -> scan1 -> scan, and draw/clip reduce + leaf.
  - PR #685 would replace the tree scans with a single-pass decoupled-fallback scan, using `workgroupUniformLoad` for the shared lock to pass WGSL uniformity analysis. It is blocked on toolchain issues: naga lowers `for` into `loop { if .. break }`, so FXC rejects the loop-unrolling/dynamic-indexing cases. On Metal, wgsl->naga->msl ran at ~60% of the glsl->SPIR-V->spirv_cross->msl speed (tracked as wgpu#6521).
- Reported speed-up: none in the PR thread for Vello itself. The ~60% Metal slowdown is a toolchain effect, not an algorithm effect.
- Accuracy / determinism impact: n/a (integer monoid scans).
- Applicability to vkapriltag:
  - The GPUPrefixSums README says the most up-to-date Decoupled Fallback "lives in Vello". In practice it has not been merged into Vello's production path in ~2 years. The only production-quality portable renderer that tried it still runs reduce-then-scan everywhere.
  - vkapriltag compiles GLSL -> SPIR-V directly, so none of the naga/FXC blockers apply. But no production Mali deployment of DF was found anywhere.
  - Keep the existing reduce-then-scan for the blob scan (or shrink it to one workgroup); do not adopt DF.
- Verification: [verified from primary source] (PR thread via gh; tree listing via gh api)

### Thread-per-comparator mapping vs. the in-tree "iterate slots, guard, skip" loop — poniesandlight "Implementing Bitonic Merge Sort in Vulkan Compute" (Tim Gfrerer, poniesandlight.co.uk; date not checked) + own warp simulation
- URL: https://poniesandlight.co.uk/reflect/bitonic_merge_sort/ ; simulation script cached `ccl/scansort/web/oem_warp_sim.py`
- Technique:
  - The standard Vulkan bitonic layout has "n/2 worker threads", and "each worker thread performs one single operation for every pair ... compare-and-swap". Every lane has a live comparator in every round.
  - vkapriltag's loop is `for idx = tid; idx + d < cap; idx += threads) if ((idx & p) == r) {...}`, and the deleted bitonic used `if (partner > idx)`. Both iterate over ALL slots, and only the lanes that pass the guard do work.
  - For p >= 16 the guard selects whole 16-lane warps, which is warp-efficient. For p < 16 (p = 8, 4, 2, 1) the active lanes interleave, and every warp runs the body half-populated.
  - The fix is to enumerate the active comparators directly: c = tid .. cap/2, `idx = ((c >> log2p) << (log2p + 1)) | r | (c & (p - 1))`, then skip if idx + d >= cap.
  - The comparators in one round are disjoint (the shader already relies on that), so the output is BIT-IDENTICAL: same network, same pairs, different lane assignment.
- Reported speed-up: The source gives only "1M random ints in 4 ms" vs 40 ms std::sort, with no hardware given. My simulation counts warp-level (16-wide, Mali) compare-exchange body executions per blob, current vs enumerated:

  | cap | current | enumerated | ratio |
  | --- | --- | --- | --- |
  | 32 | 25 | 15 | 0.60 |
  | 64 | 65 | 37 | 0.57 |
  | 128 | 167 | 95 | 0.57 |
  | 256 | 419 | 243 | 0.58 |
  | 1024 | 2467 | 1507 | 0.61 |
  | 4096 | 13571 | 8707 | 0.64 |

  The comparator counts (1471 at cap = 128) match Batcher's closed form, which checks the schedule replica.
- Accuracy / determinism impact: None. The output is bit-identical by construction.
- Applicability to vkapriltag: Not in the notes as tried. OPTIMIZATION_NOTES item 9 changed the network, not the lane mapping.
  - It REDUCES WORK (~40% fewer warp instructions in the compare body, and a halved loop trip count when cap > threads) instead of reshaping it. That is the pattern the notes say has paid on this Mali ("fewer dispatches, fewer redundant unions"). It is not "fewer, fatter threads".
  - Caveat: the number of distinct 64-byte cache lines touched per round stays about the same, because an enumerated warp spans 32 slots instead of 16. If Mali's load-store cache is line-throughput-bound for shared memory, the gain shrinks to issue/ALU/loop overhead only.
  - Expected uplift: somewhere between ~0 and ~40% of the sort's compare phase. The compare phase is most of the ~0.39 ms sort span, so plausibly -1 to -4% Mali GPU total. RDNA is similar in direction; waves of 32/64 are half-populated in even more rounds.
  - Cheap to bound: ~15-line change, no correctness proof needed beyond "same pairs". Just ABBA it.
- Verification: [verified from primary source] for the poniesandlight mapping. [secondary/unverified] for the uplift: the warp counts are my own simulation of the shader's schedule, not a measurement.

### cub::DeviceSegmentedSort (size-partitioned segmented sort) — NVIDIA CUB 1.15.0 (Nov 2021) onward, now in NVIDIA/cccl
- URL: https://github.com/NVIDIA/cub/discussions/391 (release notes); sources cached `ccl/scansort/web/cub_dispatch_segmented_sort.cuh`, `cub_tuning_segsort.cuh`, `cub_agent_subwarp_merge.cuh` (cccl main, fetched 2026-09-26)
- Technique: A three-way partition of segments by size (a DevicePartition pass writes large/medium and small index lists), then two kernels.
  - LARGE: one 256-thread block per segment, block radix sort with 19-23 items/thread (up to ~5-6K items in shared memory); bigger segments use global-memory radix.
  - MEDIUM and SMALL: `WarpMergeSort` on sub-warp groups inside 256-thread blocks. Medium uses 16-32 threads per segment × 7-11 items/thread; small uses 2-8 threads per segment × 7-9 items/thread. So one block sorts 8-16 medium or 32-128 small segments at once, with items held in registers and merged across lanes.
  - The partition only kicks in above PARTITIONING_THRESHOLD = 300-500 segments (per-arch tuning). Below that, every segment goes to the one-block-per-segment kernel.
- Reported speed-up: "up to 5000x" vs DeviceSegmentedRadixSort for many small segments (release notes, no hardware given).
- Accuracy / determinism impact: Deterministic.
- Applicability to vkapriltag:
  - This is Hou 2017's binning done by the production library, and it validates the shape of idea "pack several small blobs per workgroup".
  - vkapriltag is right at CUB's threshold. The ideas doc gives 458 blobs / 51386 points (mean ~112) for its reference frame, i.e. just BELOW CUB's 500. CUB would not partition that frame on an NVIDIA part with dozens of SMs.
  - The occupancy argument is stronger on a 4-core Mali-G610. By my arithmetic, 458 workgroups of 128 threads are ~14 "waves" of workgroups, assuming ~1024 resident threads per core. That residency is inferred from shaderWarpsPerCore = 64 on G710 in `mali/azh_valhall.md`; the G610 value was not checked.
  - Take a 20-point blob with cap 32. It pins 8 warps, and 6 of them only execute the loop header and the barriers, taking thread slots that other blobs' work could use.
  - The CUB-specific parts do not port to Mali: warp shuffles for the cross-lane merge (subgroup ops, rejected in-tree), and register arrays of 7-11 items (Valhall halves occupancy above 32 registers per thread, per `mali/azh_valhall.md`).
  - The portable form: one sort workgroup, several blobs, each blob's sub-network in its own shared-memory window, one shared barrier per round.
  - Blob-to-workgroup assignment needs a size class. Blob counts already exist before the sort (the blob_point_offsets scan), so a tiny pass, or the existing scan chain, can emit a small-blob list and a large-blob list for two indirect dispatches. That costs one or two dispatches of latency (~2.6 µs+ each on Mali) against the saved workgroups.
- Verification: [verified from primary source] (source code); the 5000x is [secondary/unverified] (release-note claim, no setup given)

### Last-block-done single-pass reduction (threadFenceReduction) — NVIDIA CUDA Samples; applied here to the blob-offset scan chain
- URL: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/threadFenceReduction (description via search summary); in-tree `GpuDetector.cpp` lines ~608-720, 1084-1086; `extract_blob_counts.comp`, `scan_block.comp`
- Technique:
  - Each block reduces its tile and writes a partial. It then fences (`__threadfence`) and takes a ticket with a global atomicAdd. The block that draws ticket == numBlocks-1 knows every other partial is visible and finishes the reduction.
  - Nobody waits on anybody, so this needs NO inter-workgroup forward-progress guarantee. It is safe on Mali/Apple, unlike decoupled look-back.
  - Vulkan form: `memoryBarrierBuffer()` (or a release atomic) before the ticket atomic, `coherent` buffers, and an acquire on the last workgroup's side.
- Reported speed-up: none given (sample only).
- Accuracy / determinism impact: Integer sums, exact and deterministic.
- Applicability to vkapriltag: The blob-offset chain today is `extract_blob_counts` (capacity-sized) + `RunInclusiveScan`. With max_blobs = 2048 and scan_wg = 512 on Mali, that is 4 dispatches separated by 4 barriers: extract, scan_block L0, scan_block L1, add_offsets. It feeds `blob_point_offsets` (base = offsets[b] - count) and the indirect point count. Two options, both forward-progress safe:
  - (a) ONE workgroup does extract + scan. 512 threads × 4 entries serially at max_blobs = 2048, then one shared-memory scan. That is 4 dispatches -> 1, with a bit-identical `blob_point_offsets`.
  - (b) Remove the scan altogether with an atomic range allocator in select_blobs: after `pos < max_blobs`, `start = atomicAdd(point_total, e.count)`, stored in the dead `MinMaxExtentsGpu::starting_offset` field. It "is now always 0" per OPTIMIZATION_NOTES "Behavioural changes".
    - `point_total` directly becomes the indirect point count. That is one extra atomic per selected blob (~458) on a counter that is already contended by `counter`, and it deletes extract_blob_counts and the whole scan chain.
    - Determinism: blob ranges get an arrival-order layout. Blob ORDER is already arrival-order (select_blobs' atomicAdd), so nothing that is deterministic today becomes nondeterministic. The contents per blob are unchanged, but the host-side consumers of blob_point_offsets must switch to starting_offset.
  - Expected uplift: small.
    - On Mali, barriers between trivial dispatches are ~2.6 µs each (PERFORMANCE.md section 3), so (a) saves ~8 µs and (b) ~10 µs plus the dispatches: ~0.2-0.4% of 3.76 ms.
    - On RDNA it is ~4 µs total (the measured `blob_scan` capacity cost plus ~1.6 µs per barrier).
    - Not a headline item, but nearly free to bound: time the `blob_scan` span on the G610 with the existing profiler. The only in-tree Mali-class number is the older Intel 0.086 ms / MX230 0.023 ms table.
  - Note: the notes rejected indirect right-sizing of extract_blob_counts at ~2 µs for its complexity. (b) is different: it deletes passes instead of adding an indirect slot.
- Verification: [secondary/unverified] for the CUDA sample (search summary; the sample page itself was not opened). The in-tree dispatch structure is [verified from primary source] (code read). The uplift is my estimate from the documented barrier costs.

### Merge Path — A Visually Intuitive Approach to Parallel Merging — O. Green, S. Odeh, Y. Birk, arXiv 1406.2628 (2014; GPU version Green, McColl, Bader ICS 2012)
- URL: https://arxiv.org/abs/1406.2628 (abstract only via WebFetch)
- Technique: Split the merge of two sorted arrays into contiguous pairs of sub-sequences with any chosen total size, each of which forms a contiguous run of the output. This is done by a binary search along cross-diagonals of the merge grid. The result is a "synchronization-free, cache-efficient merging (and sorting) algorithm". Each thread then merges a fixed-size output chunk sequentially.
- Reported speed-up: none in the abstract. The ICS 2012 GPU paper was not opened.
- Accuracy / determinism impact: Deterministic and stable with a consistent tie rule.
- Applicability to vkapriltag: This is the building block behind Hou's block-bin smem-merge and CUB's WarpMergeSort. For sort_points_local it would replace the late network rounds of a big blob with log2(cap/chunk) merge levels, each with one diagonal binary search per thread plus a sequential merge of `chunk` outputs. Both are dependent-load chains in L2-backed shared memory on Mali. Same verdict as the Satish rank-merge entry: only worth it for the rare large blobs, and only after a size-histogram bound.
- Verification: [verified from primary source] (abstract only)

## Conclusions

"Sort span" means sort_points_local: ~0.39 ms of 3.46 ms on the Mali-G610 (~11%; ~15% in an older breakdown).

The overall picture:
- No scan or compaction change in this category is a big Mali win. The scan is tiny (max_blobs = 2048 entries), and the only large compaction (blob_diff) is cheapest as the unordered atomic append it already is.
- The remaining leverage is inside sort_points_local: how lanes and workgroups are mapped to comparators and blobs. The changes that reduce work without reshaping per-thread work are the ones the in-tree history says pay on this Mali.

### Ranked ideas

**1. Enumerate active comparators instead of guarding slots in sort_points_local.**
- What: today's loop visits every slot and keeps those with `(idx & p) == r`. For p < 16 that leaves every 16-lane warp half-populated. Map `c = tid .. cap/2` to `idx = ((c >> log2p) << (log2p+1)) | r | (c & (p-1))` instead.
- Why: my simulation of the exact shader schedule gives 0.57-0.64x the warp-level compare-exchange body executions for cap 32-4096 (entry "Thread-per-comparator mapping").
- Expected uplift: 0 to ~40% of the compare phase. Plausibly -1 to -4% of Mali GPU total. The lower end applies if Mali's load-store cache is line-throughput-bound, since distinct lines per round stay about the same. RDNA: same direction.
- Determinism: bit-identical by construction (same comparator pairs; pairs within a round are disjoint).
- Forward progress: n/a (intra-workgroup).
- Device: both.
- Bound: it is a ~15-line change, so the bound is the build. ABBA 8 rounds on the G610 at decimation 1/2/4, per the section 3c protocol.

**2. Several small blobs per sort workgroup (length binning, after Hou 2017, Kobus 2023 and cub::DeviceSegmentedSort).**
- What: today a 20-point blob pins a 128-thread workgroup (8 warps) for 15 barriered rounds while 6 of those warps only run loop headers and barriers.
  - Give blobs with count <= 64 (or <= 128) a sub-range of a shared workgroup: each blob gets its own shared-memory window and lane group, and all blobs in the workgroup share one barrier per round.
  - Large blobs keep one workgroup each.
  - Classification comes free from counts known after select_blobs. Two indirect dispatches (small list, large list), or one dispatch whose first K workgroups are "packed".
- Why: on a 4-core Mali, 458 blobs are ~14 successive waves of workgroups, and idle warps take resident-thread slots.
- Expected uplift: unknown until the size histogram exists. If most blobs are <= 128 (mean 112 at the reference frame), plausibly 20-40% of the sort span, ~2-4% of Mali GPU total. RDNA: smaller.
- Determinism: unchanged (network).
- Forward progress: n/a.
- Device: Mali-first; tune the bins on the board, not the RX 9060 XT (Kobus: the best bins are per-GPU).
- Bound:
  - (a) Dump the per-frame blob-size histogram.
  - (b) Run a deliberately incorrect build that skips the network for count <= 64 blobs. That gives the ceiling on what packing them could save.
- Caution: OPTIMIZATION_NOTES item 5 found 128-256 threads per blob better than 64 on Mali. That argues against shrinking the workgroup; it does not argue against filling it with several blobs.

**3. Deterministic tie-break for equal theta_key.**
- What: after the network, one neighbour-compare pass plus one barrier. Only if a shared "tie seen" flag is set, re-order each tie run by PackXY(src[slot]).
- Why: equal-(x, y) duplicates (from the shared SE/SW diagonal midpoint) emit identical RawLineFitPoints, so their order is harmless. Only distinct points on the same ray matter, and those currently order by atomic arrival (scatter_index_points' per-blob cursor). That is the documented source of the +/-1-2 candidate-quad jitter (PERFORMANCE.md section 7).
- Why not the alternative: this is far cheaper than an ordered blob_diff. Bakunas 2017: the ordered version is ~1.5x slower than atomic append, and needs a second predicate pass, ~+0.3 ms on Mali.
- Expected uplift: 0 (costs ~1 extra round, a few µs on Mali). Payoff: reproducibility across workgroup sizes and devices.
- Device: both.
- Bound: count distinct-point theta_key ties per frame over the corpus. If always 0, it is not needed.

**4. Delete the blob-offset scan chain with an atomic range allocator in select_blobs.**
- What: `start = atomicAdd(point_total, e.count)`, stored in the dead `starting_offset` field. `point_total` becomes the indirect point count.
- Alternative (a'): if bit-identical `blob_point_offsets` must be kept, do extract + scan in ONE workgroup.
- Expected uplift: removes 4 dispatches and 4 barriers (extract, 2x scan_block, add_offsets at max_blobs = 2048, scan_wg = 512). About 10 µs on Mali (~0.3%), ~4-6 µs on RDNA.
- Determinism: blob ranges get arrival order, but blob order is already arrival order. Nothing deterministic today becomes nondeterministic. Host consumers of blob_point_offsets must switch fields. (a') is identical output.
- Forward progress: safe. Nobody waits, unlike decoupled look-back.
- Device: Mali mostly.
- Bound: read the Mali `blob_scan` span from the existing profiler first.

**5. Shared-memory-atomic aggregation of blob_diff's single global append counter** (Arm BP 10.13 recipe; NOT the subgroup variant that was rejected in-tree).
- What: each workgroup counts its boundary points with a shared atomic, then does one global atomicAdd, then hands out offsets.
- Expected uplift: unknown. It depends on whether `boundary` (0.36 ms on Mali, 1.09x DMC-sensitive) is contention-bound.
- Caution: this adds a barrier and shared traffic, the class of change that lost on Mali before (item 9).
- Determinism: still unordered.
- Device: Mali.
- Bound: an incorrect build that replaces the global atomicAdd with `gl_GlobalInvocationID`-derived slots (no contention) gives the ceiling. Spend effort only if the ceiling is > 10% of the span.

**6. Register-blocked chunks / rank-merge for the rare large blobs** (Hou reg-sort exch_local, Satish rank merge, merge path).
- What: this cuts barriered rounds for cap >= 512 (e.g. 55 -> 38 at 1024 with 256-slot tiles plus rank merges).
- Costs: dependent binary-search loads in L2-backed shared memory, 2x shared footprint, and, for the register-bitonic form, giving back part of the shipped odd-even comparator saving.
- Device: Mali (decimation-1 tail blobs).
- Bound: only after idea 2's histogram shows large blobs dominate the sort span. The barrier-removal ceiling was 5-9% of the span on MX230/Intel and was never measured on Mali; measure it on the G610 first.

**7. Rank-by-counting for tiny blobs** (count <= 32).
- What: each thread counts the keys smaller than its own, needing one barrier. It is deterministic because packed keys are unique.
- Only as a sub-case of idea 2. No GPU benchmark source was found; this is my own suggestion.

### Rejected / not worth building

- **Decoupled look-back (single-pass chained scan) without a fallback.** Vulkan gives no inter-workgroup forward-progress guarantee (Vulkan-Docs #2233 still open). Mali-G77 failed LOBE tests (Sorensen 2021). The GPUSorting README warns OneSweep will not run on mobile or Apple. A hang is VK_ERROR_DEVICE_LOST.
- **Decoupled Fallback.**
  - It is safe (Mali-G78: 1.35x over reduce-then-scan at 2^25 elements, with 57.8% of workgroups hitting the fallback). But the win is memory traffic at huge n, and vkapriltag's scan is ~2048 entries.
  - Vello, where the README says DF lives, has still not merged it into production (PR #685 open since 2024).
- **Ordered (scan-based) compaction in blob_diff.** It pays an extra predicate pass (~+0.3 ms on Mali) only for determinism. Idea 3 gets the same determinism for a few µs.
- **Global radix sorts** (Onesweep, FidelityFX, vulkan_radix_sort, VkRadixSort).
  - vulkan_radix_sort explicitly does not support subgroup size <= 16.
  - The architecture already replaced the global sort with hash grouping plus a per-blob sort (OPTIMIZATION_NOTES items 4-6).
- **Optimal-depth networks:** at most 1 round saved at n = 16.
- **Texture-layout bitonic** (Texture3dgs): tiled image/texelFetch is already rejected on Mali, and it is a 10^6-key regime.
- **Subgroup-shuffle sort / sub-warp merge** (CUB WarpMergeSort, Hou exch_intxn): subgroup ops are rejected on Mali in-tree, and A9 bounded barrier removal at 5-9% of the span.

### Could not verify

- Kobus 2023 numbers (abstract only).
- GPUSorting / GPUPrefixSums / SplitSort benchmark numbers (images only).
- The CUB "5000x" claim (no setup given).
- The CUDA threadFenceReduction sample text (search summary only).
- Green 2012 GPU merge-path numbers (not opened).
- The G610's own shaderWarpsPerCore (the ~1024 resident threads per core estimate is inferred from the G710).
- Whether Mali's shared-memory cost is per cache line or per warp instruction. This decides idea 1's size.
- Per-frame distinct-point theta_key tie frequency.
- The blob-size histogram on the corpus.
- The Mali `blob_scan` span (only Intel/MX230 figures are in-tree).
- Whether a shared-memory-atomic aggregation of blob_diff's counter was ever measured on Mali (not found in the notes).
