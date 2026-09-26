# 03 - GPU connected-component labelling (images)

Entry format: Title / URL / Technique / Reported speed-up / Accuracy-determinism / Applicability to vkapriltag / Verification.
Cache root: C:/Users/yojob/AppData/Local/Temp/claude-research/

## Status

- Covered: vkapriltag shaders; Bolelli 2024 TPDS; YACCLAB UF_IC, TUF, KE4, allegretti_2018 (KE8), RADAR, cabaret_2017 (DLP), HA4, BKE_2019 + yonehara_2015 (downloaded); HA4 GTC2019; multivalue1402; Chen 2018; Windisch 2023; generalized1603 (Komura 2016); hybridpix2024; Playne&Hawick 2018 (abstract only, paywalled); OpenCV CUDA CCL; NPP LabelMarkersUF + Chen 2017 UF; FLSL GPU (ICASSP21/GTC21) + BW-FLSL 2022; upstream/frc971 connectivity check; SparseCCL DASIP19; ICIP24 split-merge; FastAtlas 2025 (GLSL ECL-CC); Allegretti 2019 TPDS (IC ablation); Afforest/ECL-CC giant-component analysis; MDPI Algorithms 18(6):344 2025 (HTTP 403, not opened); arXiv API sweep 2024-2026 (no new dense 2D GPU CCL found); own CPU simulation (grayimage.pgm)
- Next up: (done for this run) — optional follow-ups: re-run `ccl/sim/ccl_sim.py` on the generated corpus / on vkapriltag's dumped `thresholded`; Playne&Hawick 2018 full text if a copy turns up; MDPI Algorithms 18(6):344 via a browser. Web-search budget was exhausted in this run.

## Current algorithm in vkapriltag (for reference)

- `uf_init`: parent[i] = i-1 if v[i]==v[i-1] and v!=127 and x>0, else i. Pure streaming write; every horizontal run is a left-linked chain rooted at its leftmost pixel.
- `uf_compress` (unconditional, right after init): full find() per pixel, store only if changed. Flattens the run chains (measured worth 1.0 ms on Mali).
- N x (`uf_merge` + `uf_compress`), N ~ 2 in steady state (the 2nd merge only observes convergence): `uf_merge` considers only the DOWN edge (i, i+W) with v equal and !=127, and only at a run-overlap start (x==0 or v[i-1]!=v or v[i-1+W]!=v). doUnion = find/find/atomicMin(parent[max], min) retry loop; find() is a plain read-only walk (no path compression in-tree; path halving measured but not committed). One shared-memory atomicOr + one global atomicOr per workgroup for the changed flag.
- `uf_final`: saturating blob_size[root] counter (skip atomic once >= min_blob_pixels).
- `label_pixels`: parent[i] <- (blob_size[r]>=M ? r+1 : 0) | code<<30.
- Invariant needed downstream: root = minimum linear index of the component (label = 1+root; canonical, deterministic).
- Pass count per frame at steady state: init, compress, merge, compress, merge, compress(skipped by flag / predicated), final, label_pixels = ~6-7 full-image passes plus a host readback of the flag between chunk and the rest (fused fast path speculates).

---

### A State-of-the-Art Review with Code about Connected Components Labeling on GPUs — Bolelli, Allegretti, Lumetti, Grana (UNIMORE), IEEE TPDS 2024
- URL: https://federicobolelli.it/media/publications/pdfs/2024tpds.pdf (cached `ccl/bolelli2024tpds.txt`; Table 3 re-extracted with `pdftotext -raw -f 15 -l 15` because the -layout text garbles the row labels)
- Technique: Survey of ~20 GPU CCL algorithms, all re-implemented in YACCLAB. Two axes: iterative (NP/LNP/DPL/LE/OLE/STAVA/BRB/8DLS/M8DLS/RASMUSSON/BE; re-run until a flag is clear) vs direct (UF, LBUF, KE, ACCL, DLP, HA4/HA8, C_SAUF/C_BBDT/C_DRAG, BUF, BKE; fixed kernel count, atomics for correctness), and pixel / run / 2x2-block granularity. Mechanisms relevant here:
  - **UF (Oliveira & Lotufo 2010)**: Init (L[i]=i), Merge (union with each half-neighbourhood foreground neighbour, atomicMin hook to smaller root, retry loop — *identical to vkapriltag's doUnion*), Compression. Original is 4-connected, with a shared-memory LocalMerge tile phase + border Merge (the tile phase is what vkapriltag measured and rejected on Mali).
  - **LBUF (Yonehara & Aizawa 2015)**: UF with one image line as the tile: each thread links its pixel to its left neighbour if both foreground — exactly vkapriltag's `uf_init`. LBUF is also where **inline compression** was introduced.
  - **KE (Komura 2015)**: Init links each pixel to its *smallest-index* connected neighbour (not itself); Compression; Reduction = union only with the neighbours NOT chosen at init; Compression. 4-connected originally, 8-connected in Allegretti IPAS 2018.
  - **Inline compression (IC)** (Alg. 1): `find` that, after locating the root, writes `L[id] = root` for the *starting* node (Alg.1 lines 7-12: id<-a; while L[a]!=a: a<-L[a]; L[id]<-a — note the pseudo-code writes inside the loop, i.e. the start node is re-pointed at each step, ending at the root). Both BUF and BKE use it. "the use of IC always improves BUF and BKE performance on 2D datasets"; on 3D real volumes trees are short, IC "hardly save[s] any memory read" and the extra writes can make it slower (why BKE, which compresses twice, loses to BUF in 3D on older GPUs).
  - **HA4/HA8**: run-based, warp ballot to find run starts, union only from run-start pixels, vertical merge "if X or the pixel above is the start of a run"; strips of 32 columns per warp, border merge kernel; final labelling: only run-start thread does find, shuffles to the rest.
  - **DPL** (Hawick 2010): per-row/column directional min-label propagation, 4-connected only, iterative.
  - **BRB, BE, BUF, BKE, C_BBDT/C_DRAG**: 2x2 block based — valid only for 8-connectivity (paper: "because, when considering 8-connectivity, any two foreground pixels in a 2x2 block are connected").
- Reported speed-up (Table 3, Ubuntu, times include cudaMalloc of the output; **all 8-connected binary**):
  - A100: 3DPeS UF 0.100 / KE 0.096 / LBUF 0.099 / HA8 0.117 / BUF 0.102 / BKE 0.094 ms; XDOCS (4853x3387) UF 1.041 / KE 0.957 / HA8 0.906 / LBUF 0.933 / BUF 0.740 / BKE 0.621 ms.
  - Quadro K2200 (Maxwell): XDOCS UF 16.852 / KE 14.388 / HA8 12.445 / LBUF 15.815 / BUF 10.321 / BKE 9.153 ms; Medical UF 1.966 / KE 1.454 / HA8 1.218 / BKE 0.947.
  - Quadro P1000: XDOCS UF 11.238 / KE 11.578 / HA8 8.413 / BKE 7.204.
  - So among the *pixel-level* direct algorithms that could apply to 4-connected input, KE beats plain UF by 0-26% (dataset/GPU dependent, sometimes slower: P1000 XDOCS +3%), and run-based HA8 beats UF by 13-38% on the older GPUs. Paper: "the performance gap between different algorithmic proposals is negligible when using newer architecture and smaller datasets" and "space for further optimization is really small, extremely so" relative to the bare copy.
- Accuracy / determinism impact: all exact. None of the benchmarked algorithms guarantee a specific label numbering in their UF core except those that hook with atomicMin to the smaller root (UF, LBUF, KE, BUF, BKE all do: `union` sets "the root of one tree (usually the one with the smallest value) as the father of the other").
- Applicability to vkapriltag: vkapriltag already = LBUF init + HA4-style run-start vertical merge + UF Merge/Compress, 4-connected, three-valued. The survey's actionable residue: (1) inline compression, which the survey says always wins in 2D (vkapriltag's measured path-halving variant is a cousin — see YACCLAB entry for the exact IC find), (2) KE-style init (link to smallest connected neighbour, i.e. for 4-conn: min(left, up) — this makes the *up* link free at init and leaves Reduction only for the non-chosen edge), (3) no 2x2 block method is admissible. Mali: nothing in the survey measured non-NVIDIA hardware.
- Verification: [verified from primary source] (numbers read from the re-extracted Table 3; all benchmarks 8-connected, NVIDIA only)


### YACCLAB `labeling_UF_InlineCompression.cu` (UF + inline compression) — YACCLAB contributors (Allegretti/Bolelli), 2020
- URL: https://github.com/prittt/YACCLAB (cached `ccl/yacclab/labeling_UF_InlineCompression.cu`)
- Technique: 8-connected UF with a 16x16 shared-memory LocalMerge, GlobalMerge on tile borders only, PathCompression. The inline-compression `FindCompress` is exactly:
  ```
  id = n; label = s_buf[n];
  while (label-1 != n) { n = label-1; label = s_buf[n]; s_buf[id] = label; }
  return n;
  ```
  i.e. only the *starting* node is re-pointed, once per hop (C->B, then C->A), so a concurrent reader of C skips steps. It is **not** path halving/splitting (intermediate nodes are untouched). In this file it is used only for the final in-tile find in LocalMerge; `Union` still uses the plain `Find`. Union is the standard find/find/atomicMin(larger root, smaller root)/retry loop — the same as vkapriltag's `doUnion` apart from the +1 label bias. File comment: "This algorithm performs better than BUF only sometimes (rarely?)".
- Reported speed-up: none in-file; TPDS survey: IC "always improves BUF and BKE" on 2D.
- Accuracy / determinism impact: exact; parent[x] <= x is preserved (each store writes an ancestor), so root = min index.
- Applicability to vkapriltag: as a `find()` inside `doUnion` this is weaker than the path halving already measured in-tree (-17% labelling at d1, -2.7% at d2 on RX 9060 XT), because it only shortens the starting node. The one place IC is strictly better is **inside `uf_compress`**, where the thread owns `parent[i]` anyway: writing `parent[i] = p` at every hop (instead of once at the end) lets *other* threads whose chain passes through i skip ahead during the same dispatch. For the post-init compress, where every pixel of a run walks the same left-chain, pixel i+1's walk lands on pixel i's already-advanced pointer. It costs one extra store per hop (on Mali, an L2 write). Portable: plain loads/stores, no shared memory, no intrinsics; 4-connectivity and three-valued input don't matter. Root-min preserved.
- Verification: [verified from primary source]

### YACCLAB `labeling_TUF.cu` (Tree-based UF: SAUF decision tree picks the neighbours) — YACCLAB contributors
- URL: https://github.com/prittt/YACCLAB (cached)
- Technique: 8-connected UF where each pixel's set of unions follows the SAUF decision tree: if N is foreground, union only with N (W, NW and NE are all 8-adjacent to N, so already joined); otherwise NE, then NW or W. Same LocalMerge/GlobalMerge/PathCompression as UF; plain Find, atomicMin union.
- Reported speed-up: not in the file; TUF is not in the TPDS tables.
- Accuracy / determinism impact: exact, root = min.
- Applicability to vkapriltag: the 4-connected analogue of the SAUF pruning is exactly vkapriltag's run-overlap-start test (skip (i, i+W) when (i-1, i-1+W) is also an equal pair), already shipped (-7 to -13% labelling). Nothing further: in 4-connectivity W and N are not adjacent, so there is no other redundant edge to drop. Already done.
- Verification: [verified from primary source]

### YACCLAB `labeling_KE4.cu` and `labeling_allegretti_2018.cu` (Komura Equivalence, 4- and 8-connected) — Komura 2015; Allegretti, Bolelli, Cancilla, Grana IPAS 2018
- URL: https://github.com/prittt/YACCLAB (cached); papers: Y. Komura, Comput. Phys. Commun. 194 (2015); S. Allegretti et al., "Optimizing GPU-Based Connected Components Labeling Algorithms", IPAS 2018
- Technique: KE4 (4-conn, binary): **Init** `L[i] = up if fg(up) else (left if fg(left) else i)`, i.e. link to the connected neighbour with the *smallest* index, vertical preferred. **Analyze**: plain full find, write root. **Reduce**: for every fg pixel with fg left, `Union(i, i-1)`, unconditionally, even when Init already chose left (after Analyze those are 1-hop, equal-root no-ops). **Analyze** again. Exactly 4 kernels, no convergence loop, no flag. KE8 (Allegretti 2018): Init priority N, NW, NE, W; Reduce unions W and NE; `Find_label` passes the already-loaded label to save one load.
- Reported speed-up: TPDS Table 3 (8-conn KE vs UF): A100 XDOCS 0.957 vs 1.041 ms (-8%); K2200 Medical 1.454 vs 1.966 (-26%); K2200 XDOCS 14.388 vs 16.852 (-15%); P1000 XDOCS 11.578 vs 11.238 (+3%).
- Accuracy / determinism impact: exact. Every init link points to a smaller index and union hooks the larger root under the smaller, so root = min index.
- Applicability to vkapriltag: vkapriltag is already "KE transposed". Its init pre-links the *horizontal* direction (left), and its merge handles the vertical edges only at run-overlap starts. For this image class that is strictly better than KE4: horizontal runs are long, and KE4's vertical-first init leaves every horizontal edge to Reduce and creates column chains of length O(height) for Analyze to walk.
  **Hybrid init (new, cheap):** a run-START pixel (x==0 or v[i-1]!=v) has no left link, so its parent slot is free. Link it UP when v[i-W]==v (and v!=127): `parent[i] = join_left ? i-1 : (join_up ? i-W : i)`. This performs, for free inside the streaming init, the union for every run overlap whose leftmost column is the lower run's start. `uf_merge` can then skip overlap starts where the lower pixel i+W is itself a run start (test `x==0 || v[i+W-1]!=v`; both values are already loaded). Root-min preserved (i-W < i).
  **Risk:** chains get longer. A run start now chains up into the middle of the run above, then left to that run's start, then possibly up again, so the post-init compress walks O(x + y) hops instead of O(x) in large regions with a straight left edge. Example: at the image border x==0 every row's start links up, giving a 400-deep vertical chain. Mitigations: link up only when the pixel above is NOT itself a run start, or only on every k-th row. Bound it on the CPU first: dump `thresholded`, then simulate (a) the fraction of overlap-start unions removed and (b) the max/mean compress chain length before and after.
- Verification: [verified from primary source] (code); speed numbers from TPDS Table 3 [verified from primary source]

### YACCLAB `labeling_RADAR.cu` (Kang 2016, radar-signal CCL) — YACCLAB contributors after Kang et al. 2016
- URL: https://github.com/prittt/YACCLAB (cached)
- Technique: Kalentev/OLE-style label equivalence. Init L[i]=i+1, then loop: **Scan**: each pixel takes the min over its 8 neighbours' labels and, if smaller, does a **non-atomic** `L[label-1] = min(L[label-1], min_label)` and sets a changed byte; the host reads the byte; **Analyze**: full find per pixel. Repeat until nothing changes.
- Reported speed-up: not in the TPDS tables. The OLE family is the slowest class in TPDS Table 3 (A100 XDOCS OLE 2.750 ms vs UF 1.041).
- Accuracy / determinism impact: exact at convergence; iteration count is data-dependent; one host readback per iteration.
- Applicability to vkapriltag: none. It is the iterative, flag-polled design vkapriltag already moved away from. Rejected.
- Verification: [verified from primary source]

### YACCLAB `labeling_cabaret_2017.cu` (DLP, Distanceless Label Propagation) — Cabaret, Lacassagne, Etiemble, IPTA 2017
- URL: https://github.com/prittt/YACCLAB (cached)
- Technique: 16x16 shared-memory tile labelling with a 2x2 gather-scatter mask. DLP-SR takes the min label of the 2x2 mask and writes it non-atomically into each label's *node* (`SetRoot`: `if L[label] > eps: L[label] = eps`); then DLP-R (find); then DLP-RUF using **atomicRUF**: `m = atomicMin(L[label], eps); if eps > m: RUF(eps, m) else if label > m: RUF(m, eps)`. That is a recursive union with **no find() walk**: it hooks at whatever node it is given and recurses on the displaced parent. The South/East border merges use the same atomicRUF; the final Relabeling is a find.
- Reported speed-up: TPDS Table 3: A100 XDOCS DLP 1.378 ms vs UF 1.041 (slower); P1000 XDOCS 10.804 vs 11.238 (-4%).
- Accuracy / determinism impact: exact, root = min (atomicMin, parent <= self).
- Applicability to vkapriltag: the 2x2 mask is 8-connected. It gathers across all four pixels, so it is not valid for 4-conn/three-valued input unless restricted to the one same-valued pair. atomicRUF replaces the plain-load find walk with a chain of atomic RMWs: same dependency depth, but each hop is now an atomic. On Mali, atomics are the path measured as expensive, so this likely loses there. On RDNA it could only help if chains are 1-2 hops, which they are after compress. Not recommended.
- Verification: [verified from primary source]

### YACCLAB `labeling_hennequin_2018_HA4.cu` (HA4, 4-connected, run-based) — Hennequin, Lacassagne, Cabaret, Meunier, DASIP 2018
- URL: https://github.com/prittt/YACCLAB (cached)
- Technique (exact, from the code):
  - Layout: block = 32 x 4, i.e. one warp per row and 4 rows per block, with ONE block column (`grid.x = 1`). The warp loops over the whole row in 32-px steps, carrying `distance_y` (the distance to the run start) across chunks.
  - **StripLabeling**: `__ballot_sync` of the fg bit, then `start_distance = __clz(~(pixels << (32-tx)))`. Only run-start pixels write a label (`L[i] = i - carried_distance + 1`); non-start pixels are *never initialised*. Vertical unions inside the 4-row strip happen at pixels where both are fg and either this pixel or the one above is a run start, between the two **run-start indices** (`labels_index - s_dist`, `labels_index_up - s_dist_up`). So union-find nodes are runs, never interior pixels.
  - **StripMerge**: the same union across every 4th row (the strip border).
  - **Relabeling**: only run-start threads call find; the root is broadcast to the rest of the run with `__shfl_sync(label, tx - s_dist)`.
  - `merge()`: walk both to their roots, then loop `swap so l1>l2; l3 = atomicMin(L[l1], l2); l1 = (l3==l1) ? l2 : l3`. The retry continues from the displaced parent **without re-running find()** (DLP/Rem-style splice).
- Reported speed-up: HA8 (the 8-conn variant), TPDS Table 3: A100 XDOCS 0.906 vs UF 1.041 (-13%); K2200 XDOCS 12.445 vs 16.852 (-26%); P1000 XDOCS 8.413 vs 11.238 (-25%). HA4 itself is not in the 8-conn tables.
- Accuracy / determinism impact: exact, root = min index (atomicMin).
- Applicability to vkapriltag:
  1. Run-overlap-start union: shipped.
  2. Run-start-only nodes: vkapriltag gets the same effect after the post-init compress (every pixel then points at its run start), but pays a full-image compress for it. HA4 pays nothing, because interior pixels are never written.
  3. The retry-without-refind in `merge()` is a free micro-change to `doUnion`: after a failed atomicMin, continue from `old` instead of re-walking from it. vkapriltag already sets `b = old` and then calls `find(b)`, which on a flat array costs one extra load. Negligible.
  4. Relabeling-by-broadcast needs subgroup shuffle + ballot, and runs longer than the subgroup (16 on Mali) need a carry. That is the rejected class on Mali.
- Verification: [verified from primary source]


### A new Direct Connected Component Labeling and Analysis Algorithm for GPUs (HA4 / HA4-64, GTC 2019 talk) — Hennequin & Lacassagne (LIP6 / CERN LHCb), March 2019
- URL: https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9111-a-new-direct-connected-component-labeling-and-analysis-algorithm-for-gpus.pdf (cached `ccl/ha4_gtc2019.txt`)
- Technique: HA4 as in the YACCLAB entry, plus four points the code alone doesn't show:
  - The **merge()** (recursive atomicMin union) is credited to Playne & Hawick. Its correctness argument: `e3 = atomicMin(L[e1], e2)`; if e3 == e1 there was no concurrent write and the loop ends, otherwise merge e3 with e2.
  - The union condition is Playne's "necessary condition" (only 1 of the 2x2 configurations needs a merge). For 4-connectivity that is "both fg and one of them is a segment start".
  - **HA4-64**: two horizontally adjacent pixels "either belong to the same segment or have a different color", so one thread handles 2 pixels with a 64-bit mask.
  - **CCA fused into relabelling**: per segment [x0,x1) on row y the features are closed-form (S = x1-x0, Sy = S*y, Sx = (x1(x1-1) - x0(x0-1))/2), and each segment start does one atomic per feature into its root. The talk also notes "the first statistical moments, computed with atomic addition, are faster than the bounding boxes computed with atomic min and max", and that CCA shows "weak scalability ... (concurrent accesses in atomic operations)".
- Reported speed-up: HA4-64 is "2x faster in average than Playne and Cabaret" on Jetson TX2 (2048x2048 random images, density 0-100%, granularity 1-16); CCL 1.2 Gpx/s on TX2 (g=4), 4.6 Gpx/s on Jetson AGX Xavier, 27.0 Gpx/s on V100. CCA (7 features) 3.4 Gpx/s on AGX. The 4-connected percolation threshold is 64% density: past it, run-based processing time *decreases* ("thanks to the use of segments").
- Accuracy / determinism impact: exact; labels = min linear address.
- Applicability to vkapriltag:
  - **Per-run accumulation for `uf_final`** (new, portable, no intrinsics). `uf_final` only needs the saturated predicate `blob_size[r] >= M` (M = min_cluster_pixels: 24 by default in vkapriltag's GpuDetector.h, possibly raised by the tag-size floor; 5 upstream). A run-start pixel can scan at most M bytes forward in `thresholded` (addresses known, so not a dependent chain; 24 u8 loads at the default, one or two cache lines, or 6 aligned uint loads of packed bytes) to get `len = min(run_length, M)`, then do `if (blob_size[r] < M) atomicAdd(blob_size[r], len)`. Every non-start pixel exits after reading two bytes of `thresholded`, without touching `parent[]` or `blob_size`. The saturating proof in `uf_final.comp` still holds: a start adds min(len, M); if any run has len >= M the blob reaches M; otherwise the sum of all runs equals S(r); and the counter never exceeds S(r). 127-pixels are 1-pixel blobs, which is unchanged. This cuts `uf_final`'s parent[] reads from one per pixel to one per run (plus one per 127 pixel).
  - `uf_final` is 0.091-0.242 ms on the Mali-G610 depending on DMC clock (2.66x bandwidth sensitive, PERFORMANCE.md 3a), i.e. ~2.6% of GPU time at 2112 MHz. Expected win: maybe half of that.
  - The same closed-form per-run moments would matter if blob statistics were ever accumulated per pixel. They are not in vkapriltag (the extents work on boundary points), so no further use.
  - HA4-64's two-pixels-per-thread is the rejected 4 px/thread vectorization class.
- Verification: [verified from primary source]

### A Fast Two Pass Multi-Value Segmentation Algorithm based on Connected Component Analysis — D. Mukherjee (U. Windsor), arXiv 1402.2606, Feb 2014
- URL: https://arxiv.org/abs/1402.2606 (cached `ccl/multivalue1402.txt`)
- Technique: sequential CPU, run-based two-pass CCA for multi-valued (colour) images. In the top-down pass, runs of "similar" values are extracted per row (similarity = user distance: Euclidean colour, gradient orientation, etc.), given provisional labels, and equivalences are recorded (TYPE-I: overlapping similar runs in adjacent rows; TYPE-II: transitive). In the bottom-up pass, equivalent runs are relabelled via `MAKEEQUIVALENT`.
- Reported speed-up: CPU only; the timings table (ms per image size) reads as linear in the number of runs. No GPU numbers.
- Accuracy / determinism impact: exact for an equality predicate. With a distance threshold the relation is not transitive, so results depend on scan order.
- Applicability to vkapriltag: conceptually confirms that multi-value CCL is just "edge iff same value" plus runs as nodes, which is already what vkapriltag does (0/255 join like-valued neighbours; 127 joins nothing). Nothing GPU-specific to port. The one generalisable point is that the 127 class never forms runs, so it is pure per-pixel overhead in every labelling pass. See the "127 fast path" idea in Conclusions.
- Verification: [verified from primary source]


### Efficient Parallel Connected Components Labeling with a Coarse-to-fine Strategy (C2FL) — J. Chen, K. Nonaka, H. Sankoh, R. Watanabe, H. Sabirin, S. Naito (KDDI Research), IEEE Access 6 (2018), arXiv 1712.09789
- URL: https://arxiv.org/abs/1712.09789 (cached `ccl/chen2018.txt`)
- Technique: 4-connected, equality-based (`subimg[tid] == subimg[tid-1]`, so it works on any-valued input), 32x32 shared-memory tiles.
  - **Coarse labelling**: a row scan copies the left neighbour's label, then a column scan overwrites it with the upper neighbour's label (non-atomic, no find). This builds short partial trees that may split one region into several.
  - **Refinement**: one more row scan does an atomic `merge` wherever two horizontal neighbours have different roots.
  - Then a find, a local-to-global index conversion, **boundary analysis** (a merge only on tile-border pixels), and a global root-find.
  - Paper's key observation: "the provisional label of the left pixel and that of the upper pixel are always smaller than the label of a target pixel, while the upper one is always the minimum", so any sequence of such copies keeps parent <= self.
- Reported speed-up: abstract: "outperforms the other approaches between 29% and 80% on average" (GTX 1070, CUDA 8, 4-connected versions of LE, CCLSM, UF, LUF; BE is 8-conn). Table 1 is garbled in the text extraction, so no per-image numbers are quoted here. Also: coarse labelling *increases* iterations for tiles below ~256x256-pixel images ("a local label-equivalence list in a low-resolution image is short"), and it minimises atomics.
- Accuracy / determinism impact: exact; root = min index within tile and globally (larger root -> smaller).
- Applicability to vkapriltag: the tile/shared-memory structure is the class measured and rejected on Mali (3.58 -> 5.32 ms at 16x16). Two transferable points:
  1. It independently validates "pre-link with plain stores in a fixed direction, then fix up with atomics only where roots differ". That is what `uf_init` (left) + run-start `uf_merge` already do.
  2. Its column-copy-after-row-copy is the "hybrid init" idea (see the KE entry) done with a *second* non-atomic pass instead of inside the same init. In vkapriltag's global-memory setting a second pass costs a barrier (2.6-18.7 µs on Mali), so the single-pass hybrid (link run starts up) is the version worth testing.

  Candidate on RDNA only: a 4-connected, three-valued C2FL tile pass in LDS might recover what the tile-UF lost on Mali. It would need a device-class switch, and in-tree evidence says shape changes did not help on Mali. Low priority.
- Verification: [verified from primary source] (method); speed-up figure is the abstract's own claim [verified from primary source]; per-image table not verified (garbled)

### Parallel Algorithm for Connected-Component Analysis Using CUDA (Tree Merge Kernel) — D. Windisch, C. Kaever, G. Juckeland, A. Bieberle (TU Dresden / HZDR), Algorithms 16(2):80, Feb 2023
- URL: https://doi.org/10.3390/a16020080 (cached `ccl/windisch2023.txt`)
- Technique: CCA *after* labelling (labels come from Playne's algorithm). Each thread seeds an object-feature struct (OFS: label, count, bbox, sum x, sum y, plus binary-tree links) from 2 pixels; array size W*H/2, since a chessboard is the worst case for binary 4-conn. Trees are merged pairwise at doubling granularity: `atomicCAS` to claim child slots, non-atomic feature merge for equal labels, `__syncthreads` in shared memory and then `cooperative_groups::this_grid().sync()` globally. Then a per-leaf size walk to the root (atomicCAS only at two-child nodes), and a "balance" pass that moves valid nodes to the array front. The goal is that the host downloads only valid objects (the object count is predicted from the previous frame, with a second copy if that was too few).
- Reported speed-up: aims at transfer volume, not kernel time. Paper: best for medium image sizes (256-1024 px); for larger images the total is similar to HA4; "mere feature calculation using TMK is slower than HA4 for most cases", except near the percolation threshold.
- Accuracy / determinism impact: exact features; the node order in the tree is atomic-order dependent.
- Applicability to vkapriltag: none new. vkapriltag already compacts on the GPU (select_blobs + scan + scatter) and reads back only selected blobs. The grid-wide sync it relies on has no Vulkan equivalent without a persistent-threads forward-progress assumption. The one idea echoed elsewhere — seeding the readback size from the previous frame — is already done in-tree (the `last_uf_iterations_` seeding and the fused speculative path). Rejected.
- Verification: [verified from primary source]


### A generalized GPU-based connected component labeling algorithm — Y. Komura (RIKEN AICS), arXiv 1603.08357, Mar 2016 (extends Komura, Comput. Phys. Commun. 194, 2015)
- URL: https://arxiv.org/abs/1603.08357 (cached `ccl/generalized1603.txt`)
- Technique: restates the 2015 KE algorithm and generalises it to arbitrary nearest-neighbour lists (graphs). The four steps are:
  - (i) **init**: `label[i] = min(connected neighbours with index < i)`, or i if there are none. The chosen neighbour is removed from the "residual list".
  - (ii) **analysis**: pointer-jump until fixed.
  - (iii) **label reduction** for each remaining residual neighbour j: find both roots; if they differ, order them so label_1 > label_2 and loop `label_3 = atomicMin(&label[label_1], label_2)`:
    - label_3 == label_2: done;
    - label_3 > label_2: label_1 = label_3;
    - label_3 < label_2: (label_1, label_2) = (label_2, label_3).

    This is the no-refind splice, the same as HA4's `merge()`.
  - (iv) **analysis**.

  The 2015 paper notes: "The number of such comparisons in this method is 1 for a square lattice", i.e. only ONE direction needs an atomic reduction because init consumed the other. It is explicitly "without conventional iteration": no convergence loop and no flag.
- Reported speed-up: 2015 KE ≈ half the time of Kalentev's label equivalence for Swendsen-Wang (quoted in this paper's intro); this paper reports bond-percolation timings on a TITAN X (Fig. 7, log plot; no numbers extracted).
- Accuracy / determinism impact: exact; the label is the minimum site number of each cluster ("The label of each cluster depends on the minimum site number in each cluster").
- Applicability to vkapriltag: two points.
  1. **A second, independent statement that find/atomicMin/retry union is a DIRECT algorithm**: one reduction pass plus one analysis gives the final labelling. vkapriltag's `record_uf_chunk` still runs a second `uf_merge` purely to observe convergence (the corpus always reports `uf_iterations = 2`, and chunk=1 always converges), plus the changed-flag readback and speculate-and-retry machinery. The in-tree bound says deleting the verification merge is worth **0 on Mali** (it is read-only on a converged image), so the speed payoff is small there. What it unlocks is structural. If labelling is declared direct (init -> compress -> merge -> compress), then:
     - the final compress is known to be final, so `uf_final`'s saturating add can be folded *into* it (it already has root r in a register), which deletes the `uf_final` dispatch and its barrier;
     - the changed flag, its shared-memory barrier pair in `uf_merge`, and the `label_pixels`/`blob_diff` speculative guards become dead code.

     Keep the flag as a debug assertion (or keep it on a `--paranoid` path) for safety.
  2. The no-refind retry (`label_3 > label_2 -> continue from label_3`) is a small change to `doUnion`: on a failed atomicMin, continue hooking the displaced parent directly instead of calling `find(old)`. On a flat array it saves ~1 load per retry; retries are rare. Negligible.
- Verification: [verified from primary source]

### Parallel CPU- and GPU-based connected component algorithms for event building for hybrid pixel detectors — T. Celko, F. Mráz, B. Bergmann, P. Mánek (Charles U. / CTU / UCL), arXiv 2412.11809, Dec 2024
- URL: https://arxiv.org/abs/2412.11809 (cached `ccl/hybridpix2024.txt`)
- Technique: clustering of *sparse, time-stamped* Timepix hits, not dense images. Hits are radix-sorted by time of arrival and split into chunks. Each GPU thread clusters one chunk *sequentially* with its own 256x256 "last hit per pixel" matrix and a union-find forest over hit indices (union keeps the root with smaller ToA; full path compression on every find). A second step merges chunk borders.
- Reported speed-up: "up to 300 million hits per second", "two-order-of-magnitude speedup over compared CPU-based methods".
- Accuracy / determinism impact: exact given the time window; the root is the earliest hit.
- Applicability to vkapriltag: none. One-thread-per-chunk sequential UF is the opposite design point from a dense image, and it needs a per-thread 256x256 scratch matrix. Its min-key root convention (earliest ToA) is the same idea as root = min index. Rejected.
- Verification: [verified from primary source]


### YACCLAB `labeling_allegretti_2019_BKE.cu` (BKE, the algorithm in OpenCV CUDA) — its *Compression* kernel, plus `labeling_yonehara_2015.cu` (LBUF) — Allegretti, Bolelli, Grana, TPDS 2019; Yonehara & Aizawa 2015
- URL: https://raw.githubusercontent.com/prittt/YACCLAB/master/cuda/src/labeling_allegretti_2019_BKE.cu (downloaded to `ccl/yacclab/`, along with `labeling_yonehara_2015.cu`, `labeling_KE_2S.cu`, `labeling_UF_naive.cu`, `labeling_oliveira_2010.cu`, `labeling_BUF_2S.cu`). The YACCLAB repo also has `*_NoInlineCompression.cu` ablation variants of BKE/BUF.
- Technique: BKE runs InitLabeling (2x2 blocks, links each block to its smallest connected neighbour block, stores the remaining connection bits in the unused byte of the block's last pixel) -> **Compression** -> Merge (unions only for the Q/R/S bits Init didn't consume) -> **Compression** -> FinalLabeling. The Compression kernel is:
  ```
  __device__ unsigned FindAndCompress(int* s_buf, unsigned n) {
      unsigned id = n;
      while (s_buf[n] != n) { n = s_buf[n]; s_buf[id] = n; }
      return n;
  }
  ```
  So in the fastest published 2D GPU labeller, **inline compression lives in the flattening pass**, not in the union's find. Each thread rewrites its OWN slot at every hop, so any other thread whose chain passes through that slot during the same dispatch jumps further. The union's `Find` stays a plain read-only walk. LBUF (Yonehara 2015) uses `FindCompress` the same way, in its final in-tile find. The `KE_2S` file comment says the two-stage (shared-memory tile) KE "performs worse then original BUF on every dataset, at least when using Nvidia Quadro 2200K".
- Reported speed-up: TPDS 2024: "the use of IC always improves BUF and BKE performance on 2D datasets"; the numbers are in Allegretti 2019 TPDS (not fetched; the `*_NoInlineCompression.cu` files exist to reproduce them).
- Accuracy / determinism impact: exact; every store writes an ancestor of i, so parent[x] <= x and root = min index. A stale read by another thread returns an older ancestor, which is still correct, just shorter-cut.
- Applicability to vkapriltag — **highest-value portable item from this category**:
  - `uf_compress.comp` currently walks with loads only and stores once at the end (`if (n != original) parent[i] = n`). Change it to store at every hop (`parent[i] = p` inside the loop, or every 2nd/4th hop to cap the store cost). No shared memory, no intrinsics; independent of connectivity and of the three-valued input.
  - It matters most for the **post-init compress**. Every pixel of a horizontal run walks the left-linked chain to the run start, so without sharing the total work is sum over runs of L^2/2 dependent loads (a 640-px background run is ~200k loads for one row). With IC, and lanes of a subgroup/wave marching in lockstep, lane t at step s+1 reads `parent[t-s]`, which lane t-s has just set to t-2s. That is Wyllie-style pointer doubling *for free*, turning O(L) steps into ~O(log L) per wave. This only holds when the stores are visible to neighbouring lanes and waves within the dispatch: likely within a wave via L1 on RDNA, uncertain on Mali, where L1 is per-core and non-coherent. Stale values are harmless.
  - It is also the pass the in-tree notes measured as "worth 1.0 ms" (the compress between init and first merge), so it is known to be expensive on Mali.
  - Distinct from the already-measured "path halving in find()" (survey A1). That touches `uf_merge`'s find, not `uf_compress`, and rewrites intermediate nodes (`parent[prev] = next`), not the thread's own slot. The two compose.
  - Bound cheaply: (1) CPU simulation of lockstep 16-wide (Mali) and 32/64-wide (RDNA) waves over a dumped `thresholded` image, counting the dependent-load depth per wave with and without IC; (2) the 3-line shader change plus an ABBA on both devices, with `labelling` span and bit-identical check. Risk: extra store traffic on Mali. The stores go to each thread's own (coalesced, cache-resident) word, so they should merge in the write path, but the 15%-bandwidth finding says not to worry much either way.
- Verification: [verified from primary source] (code); IC benefit claim [verified from primary source: TPDS 2024 text]; the lockstep pointer-doubling effect is my inference, [unverified — needs measurement]


### OpenCV CUDA `cv::cuda::connectedComponents` (BKE) — opencv_contrib `cudaimgproc`, since 4.6
- URL: https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaimgproc/src/cuda/connectedcomponents.cu (downloaded to `ccl/opencv/connectedcomponents.cu`); host wrapper `modules/cudaimgproc/src/connectedcomponents.cpp`
- Technique: YACCLAB's BKE verbatim: InitLabeling -> Compression (`FindAndCompress`, inline compression of the thread's own slot) -> Merge -> Compression -> FinalLabeling. The host wrapper asserts `connectivity == 8`, `ccltype == CCL_BKE || CCL_DEFAULT` and 8-bit input. **No 4-connectivity, no multi-valued input.** The OpenCV OpenCL (T-API) `connectedComponents` has no GPU kernel path that I found; `cv::connectedComponents` on UMat runs on the CPU (Spaghetti/SAUF/BBDT). [secondary/unverified for the T-API claim]
- Reported speed-up: see TPDS 2024 (BKE best on all 2D datasets).
- Accuracy / determinism impact: exact, root = min raster index of the block.
- Applicability to vkapriltag: not usable as-is (8-connected, binary). It confirms that the production-grade GPU CCL puts inline compression in the flattening pass (see the BKE entry). Nothing else to take.
- Verification: [verified from primary source] (source and host assertions)

### NVIDIA NPP `nppiLabelMarkersUF_*` / `nppiCompressMarkerLabelsUF_*`, and the paper it cites: "An Optimized Union-Find Algorithm for Connected Components Labeling Using GPUs" — J. Chen, Q. Yao, H. Sabirin, K. Nonaka, H. Sankoh, S. Naito (KDDI Research), arXiv 1708.08180, 2017
- URL: https://docs.nvidia.com/cuda/archive/11.4.1/npp/group__image__filter__label__markers.html ; https://arxiv.org/abs/1708.08180 (cached `ccl/chen2017uf.txt`)
- Technique: NPP docs: a region is "any pixel region where all pixels in the region have the same pixel value"; "nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity"; "The algorithm used in this implementation is based on the one described in 'An Optimized Union-Find Algorithm for Connected Components Labeling Using GPUs' by Jun Chen and others". Label IDs "are not generated in any particular order and there may be numeric gaps". A separate CompressMarkerLabelsUF makes them dense. The paper: 4-connected, three kernels (a shared-memory local merge with coarse row-column unification, boundary analysis on block borders only, and a link/flatten step) — the precursor of the C2FL entry above.
- Reported speed-up: paper, GTX 1070: vs label equivalence, conventional UF [Oliveira] and line-based UF [Yonehara] respectively "around 5x, 3x, and 1.3x"; it labels 512^2 / 1024^2 / 2048^2 / 4096^2 images in "around 0.14, 0.40, 1.10, and 3.40 ms".
- Accuracy / determinism impact: exact partition. NPP makes **no** min-index label guarantee (documented as unordered with gaps).
- Applicability to vkapriltag: NPP is the one production library whose problem statement matches vkapriltag's: multi-valued, same-value 4-connectivity. Two differences block reuse even as a reference: 127 must NOT join 127 (NPP would merge ambiguous regions; vkapriltag would need to remap 127 to unique values or mask), and NPP's labels are not root-min. The algorithm class (shared-memory tiles plus boundary merge) is the one rejected on Mali. The only use is as a desktop-NVIDIA cross-check oracle for the partition: remap 127 pixels to a distinct sentinel per pixel, which is impossible in 8/16-bit, so compare only 0/255 regions. Rejected for the product.
- Verification: [verified from primary source] (NPP doc text, paper numbers)


### FLSL on GPU: "Taming Voting Algorithms on GPUs for an Efficient Connected Component Analysis Algorithm" — F. Lemaitre, A. Hennequin, L. Lacassagne (LIP6 / CERN LHCb), ICASSP 2021 + GTC 2021 talk; and "An efficient run-based CCL algorithm for processing holes" (BW-FLSL, CPU) — Lemaitre, Maurice, Lacassagne, 2022
- URL: https://largo.lip6.fr/~lacas/Publications/ICASSP21_CCA_GPU.pdf , .../GTC21_FLSL.pdf , .../BNBW22_FLSL+holes.pdf (all cached in `ccl/`)
- Technique:
  - **FLSL (GPU)**: full-run CCL/CCA. Per image row, a warp does RLE "compress-store": `ballot` of the fg bit, `me = mc ^ funnelshift_l(mp, mc, 1)` gives the edges, and `popc(me & lanemask_le)` gives each edge's slot in `RLC[y*w + ...]`. This yields the run boundaries as semi-open intervals. Then one thread per *run* (not per pixel) unifies with overlapping runs of the previous row, using Komura/Playne `merge()` (atomicMin, no-refind retry). Relabelling is a per-run memset. It improves on HA, whose sub-runs are cut at the 32/64-px warp tile.
  - **CD (conflict detection)**: `match_any_sync` peers do an in-register tree reduction, then a single atomic per label per warp.
  - **OTF**: features ride along the union: `atomicExch(S[e2], 0)` then `atomicAdd(S[e1], s)`, with threadfences and a re-find loop until the features reach a real root.
  - **BW-FLSL (CPU, 2022)**: labels FG and BG runs in the same pass. It uses complementary connectivity (8-conn FG / 4-conn BG, or the reverse, via a `c8` flag in the run-intersection FSM), and gets the adjacency tree / hole filling from each label's "initial adjacency" (the label directly above at creation).
- Reported speed-up: GTC21 Table 1, average CCA throughput at 8192x8192, granularity 4 ("close to natural image complexity"), Gpix/s:
  | Algorithm | A100 | Jetson Nano | TX2 | AGX |
  |---|---|---|---|---|
  | naive | 1.00 | 0.140 | | |
  | HA | 13.6 | 0.463 | 1.08 | 3.39 |
  | FLSL | 19.2 | 1.04 | 2.38 | 4.95 |
  | FLSL+CD | 88.8 | 1.13 | 2.90 | 7.14 |

  - On the embedded GPUs, HA+CD and HA+OTF are *slower* than HA: "serialization is not as big an issue as for big GPUs ... those variants have an overhead".
  - Full-white image: FLSL 301 Gpix/s vs HA 16.6 on A100; Nano FLSL 2.48 vs HA 0.551.
  - Numbers are for CCA (labelling + features), not CCL alone.
- Accuracy / determinism impact: exact labels (min-index roots via atomicMin). With OTF, the feature-summation order is nondeterministic (integer sums, so still exact).
- Applicability to vkapriltag:
  - (a) **The per-pixel voting problem that CD/OTF solve does not exist in vkapriltag.** `uf_final` is already a saturating counter (the contended case is a cache-hot load), and the remaining voting (extents) runs over boundary points in `reduce_extents_hash`. It is already privatized, and subgroup reduce-by-key (≈ CD) was measured 6x slower than plain atomics on Mali. CD rejected. The embedded-GPU rows (Nano/TX2: CD/OTF give little or are negative) are consistent with the in-tree Mali result.
  - (b) **Full runs as union-find nodes.** This is the one structural idea left, and it is the direction in-tree notes already describe as "compact per-strip run arrays ... lean on exactly the warp intrinsics" (PERFORMANCE.md §8 item 1). Without ballot/popc, a row's RLE needs a per-row prefix count of run starts: a workgroup scan per row, i.e. shared memory (L2-backed on Mali) plus barriers. That is an extraction pass comparable in cost to the post-init compress it would replace. The measured evidence (tile UF lost, direct run-start scan lost 2.91 vs 2.57 ms) points against it on Mali. On RDNA it is a plausible but large rewrite; low priority.
  - (c) The **per-run feature closed form** (S, Sx, Sy per segment in one vote) matches the per-run `uf_final` idea already in Conclusions (the HA4 GTC entry).
  - (d) Side note from BW-FLSL: the standard way to label FG and BG together uses **complementary** connectivity (8/4). See the next entry: upstream AprilTag also uses 8-connectivity for white, which vkapriltag does not.
- Verification: [verified from primary source] (all three texts read; table numbers copied from GTC21 slides 26-27)

### Side finding: connectivity actually used by upstream libapriltag and by the frc971/Team766 CUDA ancestor (vs vkapriltag's 4/4) — AprilRobotics `apriltag_quad_thresh.c`; frc971 `971apriltag/labeling_allegretti_2019_BKE.cc`; Team766 `src/labeling_allegretti_2019_BKE.cu`
- URL: local clones under `fiducial/apriltag`, `fiducial/frc971_bos/third_party/971apriltag`, `fiducial/Team766_apriltags_cuda/src`
- Technique (read from code):
  - **Upstream** `do_unionfind_line2` (line 1014 ff.):
    - The macro `DO_UNIONFIND2(dx,dy)` unions only if the value is equal. It always does left, and does up unless it is redundant (`x == 1 || !(v_m1_0 == v_m1_m1 && v_m1_m1 == v_0_m1)`).
    - `if (v == 255)` it also does **up-left** (`x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1)`) and **up-right** (`!(v_0_m1 == v_1_m1)`).
    - So white is **8-connected** and black 4-connected, with a SAUF-like redundancy pruning.
    - The loop runs `x = 1 .. w-2`, so column 0 is only reached through its neighbours' left and diagonal edges: no direct vertical union at x = 0.
  - **frc971 / Team766 CUDA** (the ancestor): a modified BKE. 2x2 blocks with *one* 8-connected foreground node per block (P mask `0x777`: all 8 neighbours for 255), and *two* 4-connected background nodes per block: the left column (a,c) and the right column (b,d). A vertical 1x2 domino is always 4-connected, so this is valid (mask `0x272`: N/W/E/S only). 127 is neither class. InitLabeling -> Compression (inline) -> Merge -> Compression.
  - **vkapriltag**: 4-connected for both 0 and 255 (`uf_merge_body.glsl` header; true since the first `uf_merge.comp` in commit 7fe2e19).
- Reported speed-up: n/a.
- Accuracy / determinism impact: **a white region whose only link is diagonal is one component upstream (and in frc971), and two components in vkapriltag.** That changes which black/white blob pairs `blob_diff` forms, so it can in principle change which boundary clusters exist and how they split. In-tree "bit-identical" is defined against vkapriltag's own baseline counters plus 5/5 decoded corpus matches (OPTIMIZATION_NOTES "The bar"). It is not a label-for-label comparison with upstream, so the gate would not catch this. I did not test whether the difference changes any detection; it might never matter at tag borders in practice.
- Applicability to vkapriltag:
  1. **Correctness question for the maintainer, not a speed idea.** Check it cheaply by running upstream's `connected_components` on the CPU (same thresholded image, via `APRILTAG_VK_*` dumps) against vkapriltag's `parent[]` partition, and counting white components that differ.
  2. **Speed implication, if parity with upstream is ever enforced:** the in-tree rejection of BKE/BUF ("needs binary 8-connected input") is only half right. White *is* 8-connected upstream, and frc971's domino trick shows exactly how a 2x2-block labeller handles 8-conn white + 4-conn black + 127 singletons. That would make a BKE-style block init (fewer nodes for white, IC compression) admissible. Adding 8-conn white to the current pixel UF instead would add two diagonal unions per white pixel at a run-overlap start (pruned the upstream way).
  3. The root = min-index invariant holds either way.
- Verification: [verified from primary source] (code read). The impact on detections is [unverified — not measured].

### SparseCCL (Hennequin, Couturier, Gligorov, Lacassagne, DASIP 2019) and "A new efficient Split & Merge algorithm for embedded systems" (Maurice, Sopena, Lacassagne, ICIP 2024) — LIP6
- URL: https://largo.lip6.fr/~lacas/Publications/DASIP19_SparseCCL.pdf , .../ICIP24_EfficientSplitMerge.pdf (cached `ccl/`)
- Technique:
  - SparseCCL: CPU CCL on a coordinate-sorted list of active pixels (~0.5% density, LHCb hits). Sequential union-find over the list, with a sliding start index so that adjacency tests stay O(kn).
  - ICIP24: CPU Split & Merge segmentation using a "Three Table Array" structure and a software cache.
- Reported speed-up: SparseCCL is 1.6-2.5x faster than dense CCL on sparse images (Intel/AMD CPUs). ICIP24 Merge is 10.6x faster than the prior Split & Merge at 960x720 on a Jetson Xavier NX CPU.
- Accuracy / determinism impact: exact (SparseCCL); n/a (ICIP24).
- Applicability to vkapriltag: none. Thresholded AprilTag frames are dense: every pixel is 0/255/127. Both works are CPU-only. Rejected. Listed so the next agent does not re-open them.
- Verification: [verified from primary source]

### FastAtlas: Real-Time Compact Atlases for Texture Space Shading — N. Vining, Z. Majercik, F. Gu et al., arXiv 2502.17712, 2025 (GPU union-find in GLSL compute)
- URL: https://arxiv.org/abs/2502.17712 (cached `ccl/fastatlas2502.txt`)
- Technique: per-frame chart extraction = connected components of visible triangles over the half-edge graph. "Modified from ECL-CC":
  - init: `T[i] = min(i, j)` over connected front-facing neighbours;
  - hook: find both roots, then an `atomicCompSwap(T[root], root, other)` loop;
  - finalize.
  
  Implemented as "three separate GLSL compute shaders, synchronized with memory barriers". The high-degree kernel split of ECL-CC is dropped, because the degree is at most 3.
- Reported speed-up: the paper reports chart extraction as a column of its per-frame timing tables. I did not extract numbers, since the domain (triangles) doesn't transfer to images.
- Accuracy / determinism impact: exact partition, root = min index.
- Applicability to vkapriltag: none new. It is a second production-style GLSL port of the same find/hook/CAS union as vkapriltag's `doUnion`. vkapriltag uses atomicMin, which the LIP6 GTC21 slides state "converge[s] faster ... than atomicCAS", so no change. It confirms that no GLSL-specific trick is missing (their init is KE/ECL "link to smaller neighbour", vkapriltag's is the left-run variant).
- Verification: [verified from primary source] (algorithm text); timing numbers not extracted

### Optimized Block-Based Algorithms to Label Connected Components on GPUs (BUF/BKE, with the InlineCompression ablation) — S. Allegretti, F. Bolelli, C. Grana, IEEE TPDS 31(2), 2019/2020
- URL: https://federicobolelli.it/media/publications/pdfs/2019tpds.pdf (cached `ccl/allegretti2019tpds.txt`)
- Technique:
  - Alg. 1 defines `InlineCompress(L,a)`: `id <- a; while L[a] != a: a <- L[a]; L[id] <- a`, i.e. the thread's own slot is updated at every hop, "this way, possible concurrent threads that read a can use the updated value". IC was first introduced by Yonehara & Aizawa [LBUF].
  - IC is applied **only in the Compression kernel** of BUF/BKE. BKE runs Compression twice (after Init and after Merge).
- Reported speed-up: Table 2, Quadro K2200 (Maxwell), 2048x2048 synthetic + real datasets, ms, 8-connected binary. The row labels are misplaced in the PDF text layer; I mapped the 10 numeric rows to the 10 labels in their printed order, which is consistent with OLE being slowest as in TPDS 2024.

  | Dataset | BUF | BUF IC | BKE | BKE IC |
  |---|---|---|---|---|
  | XDOCS | 12.088 | 11.764 (-2.7%) | 11.989 | 11.253 (-6.1%) |
  | Tobacco800 | 3.268 | 3.163 (-3.2%) | 3.409 | 3.173 (-6.9%) |
  | Medical | 1.313 | 1.299 (-1.1%) | 1.221 | 1.186 (-2.9%) |
  | 3DPeS | 0.512 | 0.508 | 0.509 | 0.501 |

  - Table 3 (average memory reads in the Compression kernel): BUF XDOCS 2,021,381 -> 1,496,108 (-26%); Medical 313,871 -> 252,421 (-20%); Fingerprints 37,395 -> 30,147 (-19%).
  - In 3D, "the use of IC may slightly increase the total execution time", because trees are short and the extra writes aren't repaid.
  - Paper: the benefit "is valuable only when a convenient trade-off between saved readings and additional writings is achieved", which depends on the image and on "the order in which threads are executed", so "the definition of a break-even is very hard and cannot be done a priori".
- Accuracy / determinism impact: exact; the labels are identical with and without IC (every store writes an ancestor).
- Applicability to vkapriltag: this gives the **primary-source size of the IC effect: -20 to -26% of compression-kernel reads, -1 to -7% total labelling time on NVIDIA Maxwell**, for 2x2-block trees whose depth comes from Merge.
  - vkapriltag's post-init compress walks much deeper chains: raw left-linked runs of up to W pixels, where BUF's trees are at most a few hops. So the saved-reads fraction should be larger there, though still bounded by how much concurrent lanes actually observe each other's stores.
  - The break-even caveat ("cannot be done a priori", thread-order dependent) is exactly why the Conclusions ask for a lockstep simulation plus an ABBA rather than a prediction.
  - Mali: no data; 3D-like short trees are the losing case, so the in-chunk compress (short trees after `uf_merge`) might lose while the post-init compress wins. Consider IC **only in the post-init compress** (push-constant switch), which is what the data supports.
- Verification: [verified from primary source] (numbers read from Table 2/3 text; the Table 2 row-label mapping is my reconstruction of a garbled layout, [partially unverified])

### Giant-component handling: Afforest (Sutton, Ben-Nun, Barak, IPDPS 2018; GAPBS `cc.cc`) and ECL-CC (Jaiganesh & Burtscher, HPDC 2018) mapped onto the image case
- URL: cached `ccl/graph/gapbs_cc.cc` (Afforest, lines 69-140) and `ccl/graph/eclcc.txt`
- Technique:
  - **Afforest**: `neighbor_rounds = 2` rounds of linking each vertex to its r-th neighbour only, each followed by Compress. Then `SampleFrequentElement` draws 1024 random `comp[]` entries to estimate the largest intermediate component c. The final link phase skips every vertex already in c. This is valid because the graph is undirected, so every edge touching c's complement is still processed from the other endpoint.
  - **ECL-CC**: init to "the first neighbor ... that has a smaller ID" (KE-style), plus intermediate pointer jumping in `find` (path halving). Both are already covered in-tree (survey A1).
- Reported speed-up: not re-extracted here (graph benchmarks; the numbers belong to the graph category).
- Accuracy / determinism impact: Afforest is exact. The skipped component is chosen by sampling, but the partition is unaffected. The root identity can depend on the sample unless hooking is by min.
- Applicability to vkapriltag: **no gain expected; recorded to close the "giant background component" question.**
  - In vkapriltag the giant 0/255 components already cost almost nothing beyond the unavoidable reads. After the post-init compress every pixel is one hop from its run start. `doUnion` exits on `a == b` without an atomic once two runs share a root. `uf_final` saturates, so a contended `blob_size[root]` becomes a cache-hot read. `label_pixels` reads the same `blob_size[root]` word from every giant-component pixel (cache-hot).
  - Afforest's skip requires processing each edge from both endpoints (it relies on the undirected adjacency). vkapriltag processes each down edge once, from the top pixel, so skipping giant-component pixels would drop edges unless up-edges were added, doubling union work for everything else.
  - The only giant-component-proportional work left is chain walking in the post-init compress (long background runs), which is idea 1 / idea 1b in the Conclusions.
- Verification: [verified from primary source] (GAPBS code, ECL-CC text); the no-gain assessment is my analysis

### Own CPU simulation: bounding the Conclusions' ideas on `grayimage.pgm` (1280x800) — this agent, 2026-09-26
- URL: scripts `C:/Users/yojob/AppData/Local/Temp/claude-research/ccl/sim/ccl_sim.py` and `comp_stats.py`
- Technique: a numpy model of the pipeline front end. It is **approximate, not bit-exact** with the shaders:
  - point-sample decimation at (2x, 2y);
  - 4x4 block min/max, then a 3x3 edge-clamped filter;
  - `min_white_black_diff = 5`.

  It measures:
  - (a) the class fractions;
  - (b) chain depth after `uf_init` (left-link), i.e. the dependent loads the post-init `uf_compress` walks;
  - (c) a lockstep-wave model of that compress. For each wave (16 = Mali, 64 = RDNA wave64) it reports "wave-steps", the sum over waves of the deepest lane (a latency proxy), and total loads. It compares no IC; IC with stores visible only inside the wave ("wave"); and IC where earlier waves have already finished ("inorder", an optimistic bound);
  - (d) a *segmented* init: `parent[i] = max(run_start, i - x%K)`, and a segment-start pixel that continues a run links to i-1;
  - (e) the hybrid (link-run-start-up) init;
  - (f) components, and `uf_final` atomics split into 127 vs non-127 (non-127 = sum of min(S, M); a lower bound, since stale reads add more).
- Reported speed-up (simulated work, not time):

  | d2 (640x400) | wave16 steps | wave64 steps | loads |
  |---|---|---|---|
  | left-init, no IC (today) | 227,735 | 90,335 | 2.20 M |
  | + IC, wave-local visibility | 173,519 (-24%) | 47,463 (-47%) | 1.88 M / 1.40 M |
  | + IC, in-order (optimistic) | 75,472 (-67%) | 24,397 (-73%) | 0.76 M / 0.82 M |
  | segmented K=16, no IC | 53,196 (-77%) | 18,911 (-79%) | 0.63 M |
  | segmented K=32, no IC | 41,671 (-82%) | 14,546 (-84%) | 0.52 M |
  | hybrid up-link (all run starts) | 720,892 (+217%) | | |
  | hybrid up-link (above not a start) | 288,074 (+26%) | | |

  - d1 shows the same pattern: today 740,460 wave16 steps; IC-wave 575,389; segmented K=16 179,966; hybrid 2.30 M.
  - Mean chain depth today: 7.6 (d2), 6.2 (d1). Max: 199 (d2), 375 (d1).
  - Mean non-127 run length: 5.6 (d2), 6.3 (d1).
  - **127 fraction: 24.1% of pixels at d2, 44.5% at d1.**
  - Overlap-start unions in `uf_merge`: 31,606 at d2. 74% of them have a run-start lower pixel, i.e. the hybrid init would absorb them.
  - `uf_final` atomics at d2, M = 24: >= 22,611 non-127 vs **61,760 from 127 pixels (<= 73% of all atomics)**; at M = 5, <= 84%. At d1, M = 24: <= 87%.
  - Components at d2: 4,784 non-127, of which 390 are >= 24 px. The largest are 58,162 / 31,852 / 14,144 px.
- Accuracy / determinism impact: n/a (model). All the modelled variants keep parent[i] <= i, so root = min index.
- Applicability to vkapriltag: this turns three Conclusions items from guesses into ranked candidates.
  - **The 127 fast path is removing most of `uf_final`'s atomics.**
  - **A bounded segmented init (new; distinct from the rejected unbounded backward scan)** cuts the post-init compress's latency proxy by ~4-6x. It costs at most K-1 byte compares per pixel, as independent loads (4-8 uint loads in the u8 variant at K=16/32).
  - **IC helps 24-73%** depending on store visibility.
  - **The hybrid up-link init is ruled out** as a standalone: chains get 1.3-3x deeper.
  - Caveat: one image, approximate thresholding, and a crude GPU model (no memory-latency overlap across waves). Re-run on the corpus and on vkapriltag's own dumped `thresholded` before trusting magnitudes.
- Verification: [verified — own computation; model assumptions stated; not a GPU measurement]

### Literature sweep 2024-2026 via the arXiv API (web-search budget was exhausted) — this agent, 2026-09-26
- URL: `export.arxiv.org/api/query`. Queries: `all:"connected component labeling"` and `(abs:"union-find" OR abs:"connected components") AND abs:GPU`, newest first. Results cached as `ccl/ax*.xml`.
- Technique: I scanned the titles and abstracts of the 2024-09 to 2026 hits:
  - 2608.18970: on-chip Bragg-peak extraction that replaces CCL with a fixed-iteration cellular automaton (ASIC/FPGA).
  - 2608.22086 SweepLSD: streaming O(width) CCL on CPU/FPGA.
  - 2603.20481 RISE: multi-threaded CPU CCL.
  - 2510.01592: GPU vertex-based CCL on 3D voxel maps.
  - 2412.11809: hybrid-pixel (covered above).
  - Graph-CC papers (dynamic spanning tree, (alpha,beta)-core).
- Reported speed-up: n/a
- Accuracy / determinism impact: n/a
- Applicability to vkapriltag: **no new dense-2D GPU CCL algorithm appeared on arXiv in 2024-2026.** The newest relevant primary sources are still:
  - the Bolelli 2024 TPDS review, which says the remaining headroom is "really small, extremely so" on NVIDIA;
  - LIP6's FLSL (2021).

  MDPI *Algorithms* 18(6):344 (2025, "A Novel Connected-Components Algorithm for 2D Binarized Images") returned HTTP 403, so I did not read it. The search snippet describes it as RLE + union-by-size. Union-by-size would break root = min index in any case.
- Verification: [verified from primary source] for the arXiv abstracts. The MDPI paper is [secondary/unverified — not opened].

## Conclusions

These are ranked, concrete experiments for vkapriltag's `uf_*` stages.

Conventions used below:
- **Root-min** means the component root stays its minimum linear index, which downstream stages require.
- **Mali reference numbers** come from PERFORMANCE.md 3a (1280x800, d2, DMC 2112 MHz): labelling 0.921 ms, `uf_final` 0.091 ms, `label_pixels` 0.116 ms, GPU total 3.457 ms.
- **Sim** is the own-simulation entry above. It uses one image and an approximate threshold, so re-run it on the corpus before trusting the magnitudes.

### 1. 127 fast path in `uf_final` and `label_pixels`

A 127 pixel is always a singleton root. Its size of 1 is below `min_cluster_pixels` (default 24, never < 2), so its label is always 0.

Today, each 127 pixel still costs:
- in `uf_final`: a `parent[i]` load, a `blob_size[i]` load and a real `atomicAdd`;
- in `label_pixels`: 2 dependent loads that produce a constant 0.

Sim: 127 pixels are **24% of pixels at d2 and 45% at d1**. They account for **up to 73% (d2, M=24) and up to 87% (d1) of all `uf_final` atomics**.

Changes:
- **Change A.** `label_pixels` already reads `thresholded[i]` for the code bits. Read it first; for 127, emit `0 | code<<30` without touching `parent` or `blob_size`.
- **Change B.** `uf_final` exits early on 127. Two ways:
  - bind `thresholded` to `uf_final` (one byte read per pixel in the u8 variant); or
  - have `uf_init` write a sentinel parent for 127 (e.g. 0xFFFFFFFF), with a one-compare guard in `uf_compress`, `uf_final` and `label_pixels`. The sentinel adds no new memory stream.

Expected uplift: perhaps 30-60% of `uf_final` and ~20% of `label_pixels` at d2 on Mali, i.e. about 0.04-0.08 ms or 1-2% of GPU total. More at d1.

Risk:
- Output: none, provided the host asserts M >= 2.
- `blob_size[i]` for 127 pixels becomes 0. Nothing else reads it: I checked, and only `uf_final` and `label_pixels` bind that buffer.
- Root-min is untouched.

Device: both.

Cheap bound:
1. Count the 127 fraction on the corpus with the sim script.
2. Write Change A (about 5 lines).
3. ABBA the `uf_final` and `label_pixels` spans at d1/d2 on both devices.

### 2. Bounded, segmented run-start init

This is new. It is not the unbounded backward scan that was rejected in-tree.

The change is in `uf_init`:
- `parent[i] = max(run_start, i - x%K)`, found by scanning back at most K-1 pixels in the same row. The loads are independent: 4-8 uints in the u8 variant for K=16/32.
- A segment-start pixel that continues a run links to i-1.
- Chains shrink from O(run length) to about 2L/K.

Sim, post-init compress at d2:

| Metric | Today | K=16 | K=32 |
|---|---|---|---|
| wave16 latency proxy | baseline | -77% | -82% |
| wave64 latency proxy | baseline | -79% | -84% |
| loads | baseline | -71% | -76% |
| mean chain depth | 7.6 | 1.5 | 1.05 |
| max chain depth | 199 | 25 | 13 |

**Variant 2b: drop the post-init compress entirely** and let `uf_merge`'s `find` walk the short chains. At d2 there are only ~31.6k overlap-start unions, with mean depth ~1.5. This saves a full-image pass. That compress was worth 1.0 ms only because the raw chains were O(L) (OPTIMIZATION_NOTES item 8).

Expected uplift: a share of the 0.92 ms labelling span. My guess is -5 to -15% labelling for 2, and more for 2b if its merge doesn't regress.

Risk:
- The in-tree unbounded backward scan measured 2.91 ms vs 2.57 ms. Its per-thread cost was O(run length) with wave-max divergence, and the max run here is 199-375 px.
- The bound K removes that tail, but init cost still rises, so this may still lose on Mali.
- Output is exact. Root-min is preserved, because every link points left.

Device: both. Also try K=8, since Mali's wave is 16 wide.

Cheap bound:
1. Make K a spec constant in `uf_init`.
2. ABBA the `labelling` span at d1/d2 on both devices for {K=16, K=32} x {with, without post-init compress}.
3. Check bit-identity.

### 3. Inline compression (IC) in `uf_compress`

Store `parent[i] = n` at every hop, as BKE/BUF's `FindAndCompress` does in the OpenCV CUDA kernel.

Evidence:
- Primary data (Allegretti 2019, NVIDIA Maxwell, 2D): compression-kernel reads -20 to -26%, total time -1 to -7%. It can lose when trees are short (3D).
- Sim, post-init compress, latency proxy:

  | Store visibility assumed | wave16 | wave64 |
  |---|---|---|
  | only within a wave | -24% | -47% |
  | earlier waves' stores also visible | -67% | -73% |

This is largely **redundant with idea 2**: after a segmented init, chains are only ~1.5 deep. Do it if idea 2 loses on init cost. Apply it only to the post-init compress (push-constant switch), because the in-chunk compress has short trees.

Risk:
- Output: none. Every store writes an ancestor, so stale reads are harmless.
- Root-min is preserved.
- Extra stores on Mali.

Device: both; more likely to pay off on RDNA.

Cheap bound: a 3-line change plus an ABBA.

### 4. Path halving in `uf_merge`'s `find()`

Already measured on RDNA: -17% labelling at d1 and -2.7% at d2, bit-identical. It has not been committed.

Measure it on Mali, then commit. It composes with ideas 1-3, and root-min is preserved.

### 5. Per-run `uf_final` (HA4 GTC-style CCA)

Only run-start pixels add `min(run_len, M)`, found by a bounded forward byte scan.
- After idea 1, only the >= 22.6k non-127 atomics remain (d2).
- It also removes the `parent[]` read for non-start pixels. Mean run length is 5.6, so that is ~80% of non-127 pixels.

Expected uplift: small, about 0.01-0.03 ms on Mali.

Risk: none; the saturating proof carries over.

Device: both.

Cheap bound: an ABBA after idea 1.

### 6. Direct labelling: fold the `uf_final` count into the final compress

Komura's KE and UF are direct algorithms: one reduction plus one compress is final. The in-tree corpus always converges in one merge.

The change saves the `uf_final` dispatch and a barrier. But it gives up the 6c skip of the converged compress, which is worth -2.1 to -2.4% labelling. **It is likely a wash**, so it is low priority. If you try it, keep the changed flag as a debug assertion.

### 7. Correctness decision, not speed: white connectivity

vkapriltag labels white 4-connected. Upstream libapriltag, and frc971's CUDA ancestor, label white 8-connected. See the side-finding entry.

Recommendation: diff the partition on the CPU against upstream's `connected_components` over the corpus.

If parity with upstream is ever required, frc971's BKE variant becomes the relevant algorithm. It uses 2x2 blocks with one 8-connected white node and two 4-connected black "domino" nodes per block; 127 pixels belong to neither. That would reopen the in-tree rejection that "2x2 block methods don't apply".

### 8. Rejected / no-go

The evidence for each is in the entries above.
- **Hybrid up-link init.** Sim shows chains 1.3-3x deeper, even though it absorbs 74% of overlap-start unions.
- **Conflict detection and on-the-fly features (FLSL).** There is no per-pixel voting here, and the paper itself finds them slower on embedded GPUs.
- **Full-run RLE labelling.** It needs ballot/popc or an extra scan pass, which is the class rejected on Mali. An RDNA-only rewrite would be low priority.
- **Tile / shared-memory UF** (C2FL, NPP / Chen 2017, KE_2S). Measured loss on Mali.
- **atomicRUF (DLP).**
- **atomicCAS hooking** (FastAtlas / ECL-CC style). atomicMin converges faster, per LIP6.
- **No-refind retry.** Negligible gain.
- **Iterative label equivalence** (RADAR / OLE).
- **TMK.** Needs grid-wide sync.
- **Afforest's skip-largest-component.** It needs edges processed from both directions, and giant components are already cheap here.
- **Sparse, CPU and FPGA methods:** SparseCCL, split-merge, hybrid-pixel, SweepLSD, cellular automata.

### Could not verify

- Whether IC stores are visible across lanes or waves within a dispatch on Mali. The sim brackets this with two models.
- All Sim magnitudes. They come from one image with an approximate, non-bit-exact threshold and a crude lockstep model, not from GPU timings.
- Whether the 4- vs 8-connected white difference changes any detection on the corpus.
- Playne & Hawick 2018 TPDS full text (paywalled).
- The MDPI Algorithms 2025 paper (HTTP 403).
- The Allegretti 2019 Table 2 row-label mapping. The PDF layout is garbled; I reconstructed it.
- The Chen 2018 per-image table (garbled).
- The claim that OpenCV's T-API CCL has no GPU kernel.
- Any non-NVIDIA GPU CCL measurement in the literature. None of the surveyed papers measured Mali, Adreno or RDNA; the closest are Jetson Nano and TX2 in GTC21.
