# 04 - Graph connectivity / union-find on GPU

Entry format: Title / URL / Technique / Reported speed-up / Accuracy-determinism / Applicability to vkapriltag / Verification.
Cache root: C:/Users/yojob/AppData/Local/Temp/claude-research/

## Status

- Covered (18 entries): ArborX DBSCAN (2103.05162), ECL-MST SC23, Indigo2 SC23, Weigel 2011 lattice CC, JT 2003.01203 (JTB journal version), ConnectIt, ECL-CC, Afforest (code only; paper unreachable), cuGraph WCC, FastSV, Liu-Tarjan, GConn, IISWC24, Jayanti-Tarjan 2016 (+ Anderson-Woll second-hand), Soman (+ Groute/Gunrock/IrGL via ECL-CC), Alistarh-Fedorov-Koval 2019, Fedorov et al. SPAA23 deterministic UF, Contour (HPEC23).
- Final Conclusions written (resumed run, 2026-09-26). Web-search budget exhausted at this point; further sources would need direct URLs.
- Scanned but not entered (no new union-find content): arXiv 2409.03771 (distributed path compression for Morse-Smale; CPU/MPI), GConn Table 2 (static times, no per-find-rule breakdown), Afforest paper (unreachable again via HUJI and ETH).
- Next up if resumed: Patwary-Blair-Manne SEA 2010 and Patwary-Refsnes-Manne IPDPS 2012 (Rem + splicing, CPU), Groute PPoPP 2017 primary text, Komura 2015 CPC (lattice UF with atomicMin; probably covered by the CCL category file).

### Baseline being compared against (read from the tree, 2026-09-26)

- `uf_init.comp`: parent[i] = i-1 if same non-127 value as left neighbour, else i (runs pre-joined, root = run's leftmost pixel = min index).
- `uf_compress.comp` runs once after init (load-bearing, worth 1.0 ms on Mali) and after every merge; full walk-to-root, guarded store, early exit on `changed_flag == 0` (except after init).
- `uf_merge_body.glsl`: one thread per pixel, only DOWN edge, only at run-overlap starts; `find()` is a naive walk (no path rewriting in the committed tree; the survey's A1 path-halving experiment measured -17% labelling at d1 / -2.7% at d2 on RX 9060 XT, not committed); `doUnion` = find both, atomicMin hook larger root onto smaller, retry through `old` (this is ECL-CC/JTB-style "hook with CAS/min, retry on failure"); workgroup-aggregated changed flag.
- Invariant used below: parent[x] <= x always (init points left, hooks are atomicMin with smaller index).
- Rejected in-tree: tile-local shared-memory UF (Mali), equal-root early-out before doUnion (noise), direct run-start computation in init.

## Entries

### ConnectIt: A Framework for Static and Incremental Parallel Graph Connectivity Algorithms — Dhulipala, Hong, Shun (MIT), VLDB 2021 (arXiv 2008.03909v4)
- URL: https://arxiv.org/abs/2008.03909 (cached `ccl/graph/connectit.pdf`; re-extracted without `-layout` as `connectit_raw.txt` because the two-column text interleaves)
- Technique: A sampling phase (compute partial components on a subgraph, fully compress, find the most frequent label L_max) followed by a finish phase that skips vertices with label L_max. Every finish algorithm is "min-based" (links higher index under lower index). Full taxonomy:
  - **Union variants.** UF-Async (Jayanti-Tarjan style: find both roots, CAS the larger root to the smaller, retry; vkapriltag's `doUnion` is this family with atomicMin in place of CAS). UF-Hooks (CAS on a separate hooks[] array, then an uncontended plain write to parent[]). UF-Early (walk both paths together; hook as soon as one side is a root; optional finds afterwards to compress). UF-Rem-CAS and UF-Rem-Lock (Rem's algorithm: compare *parents* rather than roots, `while P[u] != P[v]`: WLOG P[u] > P[v]; if u is a root, CAS(P[u], u, P[v]), i.e. hook the root under v's *parent*, which need not be a root; otherwise apply a splice rule and step u up). UF-JTB (Jayanti-Tarjan-Boix-Adsera randomized linking, with FindNaive or FindTwoTrySplit).
  - **Find variants.** FindNaive (no writes), FindAtomicSplit (CAS every visited node to its grandparent), FindAtomicHalve (CAS a node to its grandparent and jump to the grandparent), FindCompress (walk to the root, then second walk that rewrites).
  - **Splice rules (Rem only, used at non-root steps).** SplitAtomicOne (one path-split step), HalveAtomicOne (one path-halve step), SpliceAtomic (Rem's original: CAS P[u] from its current value to P[v]). SpliceAtomic + FindCompress is incorrect (counter-example in App. B.2.3). SpliceAtomic variants are correct only phase-concurrently (unions and finds separated by a barrier), which is exactly vkapriltag's structure (merge pass, then separate compress pass).
  - **Sampling.** k-out (first incident edge plus k-1 random ones, union-find them, compress; k = 2 default), BFS sampling (BFS from a few random sources, direction-optimized), LDD (one round of Miller-Peng-Xu low-diameter decomposition, beta = 0.2). Finish skips L_max vertices; correctness relies on the other endpoint of each skipped edge still processing it, which holds for symmetric adjacency lists.
  - Other min-based finishes: Shiloach-Vishkin, Liu-Tarjan variants, Stergiou, label propagation.
- Reported speed-up (72-core 4x Xeon E7-8867 v4, 1 TB, graphs up to Hyperlink2012): without sampling, **UF-Rem-CAS + SplitAtomicOne + FindNaive** is fastest on every graph; HalveAtomicOne and SpliceAtomic are "almost identical". UF-Hooks is 1.20-1.49x slower (1.35x average), UF-Async 1.34-1.96x slower (1.62x average), UF-Rem-Lock 1.5-1.9x slower, UF-Early 4.8-6.6x slower, UF-JTB 2.46-5.56x slower (4.09x average). Liu-Tarjan's best is 6.74x slower, SV 5.14x slower on average. Adding extra find-compression after a union does not help UF-Rem-CAS ("FindNaive" wins). Sampling speeds up the best unsampled algorithm by 2.2x (k-out), 2.4x (BFS) and 2.3x (LDD) on average for low-diameter graphs. On very sparse high-diameter graphs such as road_usa (m/n < 3), not sampling can be better. Running time correlates with total path length (Pearson 0.738) and LLC misses (0.797), not with max path length (0.344).
- Accuracy / determinism impact: all non-JTB variants are min-based, so the final fully compressed label of every vertex is its component's minimum index. Hence **every non-randomized union/find/splice variant here changes only intermediate tree shapes, never the final labels**: bit-identical after vkapriltag's final `uf_compress`. UF-JTB links by random priority, so its roots are not minimum indices, which is incompatible with (a). Sampling is min-based too, so it is also exact.
- Applicability to vkapriltag:
  - **UF-Rem-CAS (+ SplitAtomicOne), in `uf_merge_body.glsl` `doUnion` only:** (a) compatible, because it only ever sets P[x] to a smaller index, so a root is still its tree's minimum. (b) compatible: CAS retry, lock-free, no waiting on another invocation. (c) compatible, and possibly well suited. Its early exit on `P[u] == P[v]` terminates at the first common ancestor rather than at both roots. Its hook target is P[v] rather than root(v), so a thread whose u-side root is found first never walks the v chain. Risk: hooking under non-roots makes trees deeper for later finds in the same pass, which `uf_compress` then flattens anyway. The CPU result (1.62x over UF-Async) does not transfer directly: GPU cost is latency of dependent loads, and ConnectIt's gain is attributed to lower TPL and LLC misses. That is the same resource, so it is worth one A/B.
  - **UF-Rem-Lock:** rejected under (b) (spin locks need independent forward progress, which Vulkan does not give between workgroups).
  - **UF-Hooks:** no benefit here. It exists to avoid CAS on the hot parent array, but atomicMin on parent[] is already uncontended except at big-component roots, and it adds a buffer.
  - **UF-Early:** walks both paths together. On CPU it lost to LLC misses. On a GPU the interesting part is different: the current `doUnion` runs `find(a)` to completion and only then `find(b)`, two *serialized* dependent chains. A lock-step two-chain walk issues both loads per iteration, so it is an ILP/latency experiment, not a TPL one. Stage: `uf_merge` only. Exact.
  - **Sampling + skip L_max:** vkapriltag's `uf_init` + first `uf_compress` already *is* a deterministic "first-edge" sample (the left neighbour; Sutton's scheme, with k = 1). The skip half maps onto the convergence-verification merge pass. That is the in-tree rejected "equal-root early-out", and PERFORMANCE.md says deleting the second merge entirely measured zero on the Pi. So there is nothing left for skipping there. See the Afforest entry for the only remaining angle.
- Verification: [verified from primary source] (numbers from sections 4.1, 4.1.1 and 4.2 and Appendix D pseudocode, Algorithms 8-14; some glyphs lost in extraction, pseudocode semantics reconstructed from the surrounding prose)

### ECL-CC: A High-Performance Connected Components Implementation for GPUs — Jaiganesh, Burtscher (Texas State), HPDC 2018
- URL: https://userweb.cs.txstate.edu/~burtscher/papers/hpdc18.pdf (cached `ccl/graph/eclcc.pdf`, `eclcc.txt`; re-extracted as `eclcc_raw.txt`)
- Technique: Three phases (init, compute, finalize), asynchronous and lock-free, each undirected edge processed once in one direction (`if (v > u)`), and no iteration to convergence. The paper argues that "it is sufficient to successfully hook each edge once".
  - **Init variants.** Init1 = own ID. Init2 = smallest neighbour ID. Init3 = first neighbour with a smaller ID (ECL-CC's choice).
  - **Compute.** find() with **intermediate pointer jumping** (Fig. 5: `while (par > (next = parent[par])) { parent[prev] = next; prev = par; par = next; }`, with plain, non-atomic stores). Hook = `atomicCAS(&parent[larger_rep], larger_rep, smaller_rep)`. On failure the loop takes the CAS return value as the new rep and retries *without* calling find again (Fig. 6). Load-balanced over three kernels by degree (thread, warp, block).
  - **Pointer-jumping variants.** Jump1 = multiple (full compression, two traversals). Jump2 = single (only the start vertex's pointer is set to the root). Jump3 = none. Jump4 = intermediate (halving).
  - **Finalize variants.** Fini1 = intermediate jumping. Fini2 = multiple. Fini3 = single pointer jumping (walk to the root, write only your own entry; ECL-CC's choice). Fini3 is exactly vkapriltag's `uf_compress`.
- Reported speed-up (GTX Titan X Maxwell and Tesla K40, 18 graphs, nvcc 8.0). Jump4 is fastest on average and on 17/18 graphs:
  - Jump1: 1.6x slower on average, never faster.
  - Jump2: 1.26x slower on average, **but 18% faster than Jump4 on europe_osm**, the high-diameter road graph with the longest paths (average 4.26, max 122).
  - Jump3: 3.7x slower on average (a bar cut off at 254x).
  - L2 reads correlate with runtime.
  - Init3 vs Init1: Init1 is 4% slower on average (33% faster on rmat16, >2x slower on europe_osm). Init2 is 1.4x slower on average.
  - Finalize: Fini1 0.5% slower than Fini3 on average (never more than 4%); Fini2 14% slower (1.9x on delaunay_n24).
  - Compute is 84.5% of runtime.
  - Against other codes: 1.8x Groute, 4.0x Soman, 6.4x IrGL and 8.4x Gunrock on the Titan X.
  - Observed path lengths: on the 2D grid graph `2d-2e20.sym`, average 1.35, max 9. On USA-road-d.NY, average 2.62, max 43.
- Accuracy / determinism impact: all variants are min-based, so the final labels (component minimum ID) are identical; only tree shapes change. The races on the plain halving stores are benign because every store replaces a valid ancestor pointer with another valid, smaller ancestor.
- Applicability to vkapriltag:
  - **Jump4 in `find()`** is survey item A1, measured -17% labelling at d1 and -2.7% at d2 on the RX 9060 XT. Still uncommitted, and unmeasured on Mali. Compatible with (a) (stores only a smaller ancestor), (b) (no waiting) and (c). Stage: `uf_merge` find().
  - **Jump2** ("find, then store the root into the *starting* pixel's parent entry") is a cheaper-write alternative. It is the one variant that beat Jump4 on the one high-diameter, long-path input, and images are high-diameter grids. One extra store per union endpoint, no store inside the loop. Worth A/B-ing *against* Jump4 on the Pi, where the extra writes of Jump4 are the stated risk. Exact.
  - **ECL-CC's retry without re-find:** vkapriltag's `doUnion` calls `find(a)` and `find(b)` again after a lost atomicMin. ECL-CC instead continues from the CAS return value. vkapriltag's re-`find(a)` starts from a root, so it costs about 1 load; the re-`find(old)` is a walk. The difference is small either way, and there are no numbers.
  - **Single pass is sufficient:** ECL-CC's argument ("each edge hooked successfully once is never revisited") applies to vkapriltag's `doUnion` too, including its atomicMin variant, whose lost-link case is re-unioned through `old`. The second merge pass is therefore logically redundant. PERFORMANCE.md already measured removing it at **zero** on the Pi, so this is a correctness note (it would be a *correct* build, not a "deliberately incorrect" one), not a speed item.
  - **Init3** is what `uf_init` already does (the left neighbour is the first smaller neighbour). A vertical extension is discussed under Conclusions (strip pre-join).
  - Fini3 = `uf_compress`, already optimal per this paper.
- Verification: [verified from primary source] (text sections 3-5, Figs 5-6, Table 4; per-graph bars of Fig. 8 not readable in text, only the prose numbers)

### Afforest: Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling — Sutton, Ben-Nun, Barak (HUJI), IPDPS 2018 (+ GAPBS `cc.cc`, + authors' CUDA `device/cc.cu`)
- URL: paper https://mosix.cs.huji.ac.il/pub/Parallel_Graph_Connectivity.pdf (**could not open**: both the cached `afforest.pdf` and a fresh curl returned a HUJI "Something went wrong" HTML page). Sources: `ccl/graph/gapbs_cc.cc` (cached) and https://github.com/michaelsutton/afforest `device/cc.cu` (downloaded to `ccl/graph/afforest_src/cc.cu`).
- Technique (from the code):
  - Init comp[n] = n.
  - `neighbor_rounds` = 2 sampling rounds. In round r, every vertex Links only its r-th neighbour, then Compress runs (`while comp[n] != comp[comp[n]]: comp[n] = comp[comp[n]]`).
  - Sample 1024 random vertices and take the most frequent label c (on the GPU, a 1024-thread gather kernel plus a host histogram).
  - Final link phase over the remaining neighbours (index >= neighbor_rounds), skipping every vertex with comp[n] == c. This is valid because the graph stores both directions, so an edge from c to a non-c vertex is still processed from the non-c side.
  - Final Compress.
  - `Link(u, v)` never finds roots. It is an SV/Rem-like parent-comparison loop: `p1 = comp[u], p2 = comp[v]; while p1 != p2 { high = max, low = min; prev = CAS(&comp[high], high, low); if prev == high || prev == low break; p1 = comp[comp[high]]; p2 = comp[low]; }`. On failure it jumps the high side by two levels (grandparent) without writing.
- Reported speed-up: paper not opened. ConnectIt (entry above, same 72-core machine) reports its best sampled algorithms 1.32-8.55x (3.9x average) faster than GAPBS Afforest, and its unsampled ones 0.33-4.17x (so Afforest wins on some graphs without ConnectIt's sampling). No GPU numbers verified.
- Accuracy / determinism impact: min-based (CAS high to low), so the final labels are the component minima: exact.
- Applicability to vkapriltag — "can we skip the giant background component?":
  - vkapriltag already has Afforest's round 0: `uf_init` links each pixel's left neighbour, and the compress after init is Afforest's Compress.
  - The skip does not transfer as-is. vkapriltag processes each down edge only once, from the upper pixel, so skipping pixel i because comp[i] == c is valid only if comp[i + W] == c too. That turns it into the in-tree **rejected equal-root early-out** (measured 2.550 vs 2.569 ms, noise).
  - Its only target would be the verification merge pass, which PERFORMANCE.md says costs **zero** on the Pi when deleted.
  - The first merge pass cannot be skipped this way, because the giant component does not exist until that pass builds it.
  - A second *sampling round* before the full merge (Afforest round 1 = one vertical edge per run) would be one extra dispatch (2.6-18.7 µs barrier on Mali) that builds most of the background. That only pays if the finish pass's unions are then substantially cheaper. Since those unions are *already* restricted to run-overlap starts (about one per run pair), there is little left to skip. **Verdict: skip-largest-component is not applicable; the useful transfer is the "sampling = cheap, contention-free first-neighbour linking" idea, taken into `uf_init` (see the strip pre-join in Conclusions).**
  - `Link`'s grandparent jump on the high side (`comp[comp[high]]`) is a read-only analogue of halving: an exact, cheap variant for the doUnion retry path.
- Verification: [verified from primary source] for the code (GAPBS `cc.cc`, `device/cc.cu`); paper text [could not open]; the ConnectIt vs Afforest numbers are from ConnectIt [verified from primary source].

### cuGraph weakly connected components (`weakly_connected_components_impl.cuh`) — NVIDIA RAPIDS, 2020-2025
- URL: https://github.com/rapidsai/cugraph (cached `ccl/graph/cugraph_wcc.cuh`)
- Technique: not union-find. A recursive **multi-root BFS frontier expansion**:
  - Pick a batch of new roots, high-degree first, with the degree threshold `ceil(sqrt(2 * degree_sum_threshold))` so that at least 50% compression is guaranteed; low-degree candidates are randomly shuffled.
  - Expand all frontiers at once, claiming unvisited vertices with `elementwise_atomic_cas(dst, invalid, tag)`.
  - Record the edges where two frontiers collide ("conflicts") into a sorted, unique edge buffer (lower-triangular only).
  - Build a next-level graph from the conflict edges and recurse, then map the component IDs back down the levels.
- Reported speed-up: none in the source.
- Accuracy / determinism impact: component IDs are root tags chosen by degree plus a random shuffle, **not** minimum indices, and the conflict order is scheduling-dependent. It would need a relabel pass to satisfy (a).
- Applicability to vkapriltag: rejected. BFS expansion over a high-diameter 4-connected grid needs O(diameter) frontier rounds, each a dispatch plus barrier (2.6-18.7 µs each on Mali), and the IDs violate (a). The one transferable notion is "claim with CAS into an invalid sentinel", which is irrelevant here because the init already assigns every pixel. It would change the whole labelling stage, not a variant of it.
- Verification: [verified from primary source] (code read; no paper numbers exist for this file)

### FastSV: A Distributed-Memory Connected Component Algorithm with Fast Convergence — Zhang, Azad, Hu, SIAM PP 2020 (arXiv 1910.05971v2)
- URL: https://arxiv.org/abs/1910.05971 (cached `ccl/graph/fastsv.pdf`; `fastsv_raw.txt`)
- Technique: A simplified, synchronous Shiloach-Vishkin (min-based, double-buffered f/fnext, iterated to a fixpoint) with four changes:
  - (1) **Grandparent hooking:** hook onto f[f[v]] instead of f[v].
  - (2) **Stochastic hooking:** drop the "u's parent is a root" condition, so a *non-root* f[u] may be min-assigned to f[f[v]]. This can split a tree, but it is safe with min-assignment plus iteration.
  - (3) **Aggressive hooking:** fnext[u] = min(fnext[u], f[f[v]]) for every edge.
  - (4) **Early termination:** stop when the grandparent vector f[f] is unchanged. Lemma 3.1 says f then never changes, which saves the extra verification iteration.
  - Implemented in GraphBLAS (min-select SpMV / SpMSpV).
- Reported speed-up: the combined hooking strategies (with early termination) cut iterations by 35.0% on average (min 20%, max 46.2%) against simplified SV. Most real graphs converge in 5-10 iterations. Shared memory (SuiteSparse:GraphBLAS, EC2 r5.4xlarge, 16 threads): 8.66x average (max 13.81x) over LACC. Distributed (Cori KNL, up to 262K cores): 2.21x average (max 4.27x) over LACC. Hyperlink (124.9B edges) in 30 s.
- Accuracy / determinism impact: min-based and deterministic at the fixpoint, so the final labels are component minima: exact.
- Applicability to vkapriltag:
  - The iterate-to-fixpoint family is the wrong shape for a 4-connected grid. SV-style passes propagate a label only a bounded number of tree levels per pass, so a high-diameter image needs many passes, each a full streaming dispatch plus a barrier (2.6-18.7 µs on Mali).
  - vkapriltag's asynchronous union-find completes every union within its single pass (see the ECL-CC entry).
  - Stochastic/non-root hooking with atomicMin *inside* the async `doUnion` would break the one-pass completeness argument: a lost child subtree is only re-attached by a later iteration. So it is incompatible with the current single-real-pass structure, even though it satisfies (a) and (b).
  - Early termination by grandparent stability is a cheaper convergence proof, but the verification pass measured zero on the Pi, so there is nothing to gain.
  - **Rejected** (it would change the whole labelling stage).
- Verification: [verified from primary source]

### Simple Concurrent Connected Components Algorithms — Liu, Tarjan, arXiv 1812.06177v5 (2020; TOPC 2022)
- URL: https://arxiv.org/abs/1812.06177 (cached `ccl/graph/liutarjan.pdf`; `lt_raw.txt`)
- Technique: A framework of synchronous, round-based, **minimum-labelling** algorithms (acyclicity by only ever replacing a parent with a smaller vertex). Each round is a connect step, one or more shortcuts (v.p = v.p.p), and optionally an alter step (replace edge {v,w} by {v.p,w.p}, delete it if they are equal).
  - Connect variants: direct-connect (v.p = min(v.p, w) for edge endpoints), parent-connect (v.o.p = min(v.o.p, w.o)), and root-restricted versions of each (only roots get a new parent, so no subtree ever moves).
  - Analysed algorithms: S (parent-connect, shortcut to a fixpoint), A (direct-connect + shortcut + alter), R (parent-root-connect + shortcut), RA (direct-root-connect + shortcut + alter).
  - Bounds: two of them get O(lg n) steps / O(m lg n) work; the other two get O(lg² n) steps / O(m lg² n) work.
  - ConnectIt adds variants named with letters: C/P/E connect (direct/parent/extended), U/R RootUp, S/F shortcut, A alter.
- Reported speed-up: theory paper, no timings. In ConnectIt's measurements (above), the fastest Liu-Tarjan variants (EF, PRF, PR, CRFA) are 2.64-10.2x slower than UF-Rem-CAS (6.74x average) on 72 cores.
- Accuracy / determinism impact: min-labelling, so the fixpoint labels are component minima: exact. Rounds are synchronous (read old parents, min-combine), so the results are deterministic per round.
- Applicability to vkapriltag:
  - (a) compatible: min labelling is exactly the root = minimum convention. (b) compatible: min-combine is `atomicMin`, with no waiting.
  - (c) The round-synchronous structure needs O(lg n) to O(lg² n) *rounds*, each a full-image pass separated by a barrier. vkapriltag currently converges in one real merge pass because its unions are asynchronous and walk to completion.
  - **Rejected** as a labelling replacement. The one idea worth keeping is *alter* (edge contraction), whose GPU form would be "compact the still-unresolved overlap-start edges into a worklist for the next pass". That is moot, because there is only one real pass.
- Verification: [verified from primary source] (algorithm definitions); performance only [secondary, via ConnectIt]

### GConn: Exploring the Design Space of Static and Incremental Graph Connectivity Algorithms on GPUs — Hong, Dhulipala, Shun, PACT 2020 (arXiv 2008.11839)
- URL: https://arxiv.org/abs/2008.11839 (cached `ccl/graph/gconn.pdf`; `gconn_raw.txt`)
- Technique: A GPU port of the ConnectIt design space (over 300 variants).
  - Union variants: Union-Async, Union-Early ("traverses the paths of the two inputs simultaneously and terminates once a common node is reached"), Union-Hooks, Union-Rem-CAS, Union-Rem-Lock, Union-JTB.
  - Five on-the-fly compression options: none, splitting, halving, full, splicing.
  - Sampling: k-out, BFS, and a new **HB (hook-based) sampling**. Step 1: each vertex takes min(label, label of its first neighbour), which is contention-free. Step 2: vertices still roots union their first N = 4 edges.
  - GPU optimizations: CSR coalescing, edge reorganization, vertex gathering. Merrill-style load balancing.
- Reported speed-up (Tesla V100 32 GB; also a Titan Xp):
  - No sampling: the fastest is Union-Async or Union-Rem-CAS. Rem-CAS is 1.02x slower than Async on average, the reverse of the CPU result.
  - Slowdowns against Union-Async: Union-JTB 1.26x ("due to 64-bit CAS"), Union-Hooks 1.44x ("costly memory barrier"), Union-Rem-Lock 3.81x slower than Rem-CAS ("poor performance of spin locks on GPUs"), Liu-Tarjan 2.86x, SV 4.87x. G_Afforest's finish is 12.6x slower than the fastest finish.
  - k-out sampling speeds up union-find by 6.16x on average, **except on the low-degree road networks (road_usa, europe_osm), where every sampling scheme degrades performance** because most edges get inspected in the sampling phase anyway. BFS sampling degrades high-diameter graphs by 24.59x (a kernel launch per level).
  - Overall: 2.47x over the best existing GPU code on average; 7.06x / 26.68x (unsampled / sampled) over Soman's GPU-CC; G_ECL-CC is 2.30x faster than original ECL-CC, from load balancing alone. Recommended default: Union-Async or Union-Rem-CAS plus k-out, within 20.8% of the per-graph best.
- Accuracy / determinism impact: all min-based except JTB (random rank), so final labels are exact for everything except JTB.
- Applicability to vkapriltag:
  - This is the most direct GPU evidence. **On a GPU, Union-Async (vkapriltag's family) ties Union-Rem-CAS.** The CPU 1.62x Rem advantage vanished, so expect Rem-CAS to be at best a small win here. Lock-based variants are measured at 3.81x worse, confirming (b).
  - **Sampling hurts low-degree, high-diameter graphs.** A 4-connected image (m/n = 2, with pre-joined runs leaving about one vertical union per run pair) is the extreme low-degree case. This is strong evidence against adding a sampling/skip phase.
  - HB step 1 ("contention-free first-neighbour min") is what `uf_init` does.
  - Union-Early is GConn's name for the lock-step two-path walk. Its static GPU numbers are not separately given in the text (they are in Fig. 3 bars), so the dual-chain ILP experiment remains unbounded by the literature.
  - (Resumed-run addendum) Table 4 (incremental connectivity, whole graph as one batch, V100, throughput in ops/s, columns Early / Hooks / Async / Rem-CAS / Rem-Lock / JTB / LT / SV) *is* in the text. Mapping rows to graphs by extraction order (coPapersDBLP, cit-Patents, road_usa, soc-LiveJournal1, com-ljournal, delaunay_n24, europe_osm, ...): on high-degree graphs Union-Early is 2-8x below Union-Async (e.g. row 1: 8.18e9 vs 3.17e10), but on the **low-degree, high-diameter rows the gap shrinks or vanishes**: road_usa 5.44e9 vs 7.35e9 (0.74x), delaunay_n24 4.87e9 vs 1.24e10 (0.39x), **europe_osm 6.83e9 vs 6.74e9 (Early marginally the fastest of all eight)**. Weak but real evidence that the two-path walk is not a loss on grid-like graphs on a GPU. Caveats: incremental, not static; row mapping reconstructed from pdftotext order [partially verified].
- Verification: [verified from primary source] (sections 3-4 prose; per-graph Table 2 values not transcribed)

### Performance Impact of Removing Data Races from GPU Graph Analytics Programs — Liu, VanAusdal, Burtscher (Texas State), IISWC 2024
- URL: https://userweb.cs.txstate.edu/~burtscher/papers/iiswc24a.pdf (cached `ccl/graph/iiswc24.pdf`; `iiswc_raw.txt`; suite README `ecl_suite_readme.md`)
- Technique: Replace the "benign" races in ECL-CC (and in GC, MIS, MST, SCC) with libcu++ `cuda::atomic` relaxed loads and stores. For CC this covers every read and every pointer-jumping write in `find_repres`.
- Reported speed-up (race-free / baseline, geomean over 17 graphs): **CC 0.66 on Titan V, 0.88 on RTX 2070 Super, 0.66 on A100, 0.45 on RTX 4090.** On the 2D grid `2d-2e20.sym`: 0.55 (Titan V), 0.80 (2070 Super). The cause is profiled: the baseline's plain loads and stores of the parent array get a much higher L1 hit rate, while relaxed atomics bypass or miss L1. MST loses less "due to its use of implicit path compression".
- Accuracy / determinism impact: none; both versions produce the same labels. The race-free version is portable under a formal memory model.
- Applicability to vkapriltag:
  - Direct evidence for a decision the tree already made. `uf_merge`'s `find()` uses plain, non-`coherent` loads, and any path-halving store (A1) should also be a **plain** store, not `atomicExchange` and not a `coherent`-qualified buffer.
  - Under the Vulkan memory model a stale plain load is harmless here, for the same monotonicity reasons ECL-CC gives. Every value ever written to parent[x] is an ancestor of x with a smaller index. The `atomicMin` return value is always coherent, so a hook is never wrongly considered successful.
  - Making the parent accesses "correct" (coherent or atomic) would likely cost 12-55% of the labelling kernel, going by these CUDA numbers.
  - Mali-specific: Valhall L1 behaviour for non-coherent SSBO loads was not studied. [unverified for Mali]
- Verification: [verified from primary source] (Tables IV-VII geomeans, Section V text)

### A Randomized Concurrent Algorithm for Disjoint Set Union — Jayanti, Tarjan, PODC 2016 / arXiv 1612.01514 (revised); covers Anderson-Woll 1991 second-hand
- URL: https://arxiv.org/abs/1612.01514 (downloaded to `ccl/graph/web/jt16.pdf`, `jt16.txt`)
- Technique: A wait-free APRAM union-find built on CAS.
  - Unite(x, y): loop { u = Find(u); v = Find(v); if u == v return; if u < v then CAS(u.parent, u, v) else CAS(v.parent, v, u); return on success }.
  - Linking follows a **random** node order (each node gets a random id). That randomization is what makes the bounds provable.
  - Find with **one-try splitting**: `v = u.p; w = v.p; if v == w return v; CAS(u.p, v, w); u = v`, i.e. every node on the path is pointed at its grandparent and the walk steps to the *old* parent. **Two-try splitting** repeats each update twice.
  - Anderson-Woll (1991, per this paper) used linking by *rank* plus concurrent *halving*. JT say AW's O(m α) claim was false, because it missed counting failed CAS steps.
  - Argument that matters here: in the concurrent setting "two processes doing halving in lockstep can simulate one process doing splitting", so halving is not superior to splitting.
- Reported speed-up: theory only. Expected work O(m (α(n, m/(np)) + log(np/m + 1))) with two-try splitting (a weaker bound for one-try; the conference version's proof for one-try was wrong). O(log n) steps per operation w.h.p. Practical GPU evidence comes from GConn: JTB is 1.26x slower than Union-Async on a V100.
- Accuracy / determinism impact: random-id linking makes the root the minimum *random id*, not the minimum index, so it **violates (a)**. It would need a relabel-to-min pass, which is a full extra reduction. Rank linking (AW) violates (a) the same way. With deterministic index order the algorithm stays correct but loses its bounds. That is exactly vkapriltag's current UF-Async-with-index-order.
- Applicability to vkapriltag:
  - Randomized/rank linking: **rejected** under (a).
  - Splitting vs halving: **important finding.** ECL-CC's `find_repres` (`parent[prev] = next; prev = par; par = next`) steps to the node it just skipped, so by JT's definitions it is **path splitting**, even though ECL-CC and Patwary call it "halving". The survey's A1 code is ECL-CC's, so **A1 as measured is path splitting.**
  - True path halving (`p = P[x]; g = P[p]; if (p == g) return p; P[x] = g; x = g`) issues the same dependent-load chain but **half the stores**.
  - On Mali, where the survey flags A1's extra writes (shared LPDDR bus) as the risk, halving is the natural second arm. It is exact under (a), since it stores only smaller ancestors, and fine under (b) and (c). Stage: `uf_merge` find().
  - One-try vs plain store: ECL-CC's plain store is the right GPU choice (IISWC24: atomics cost L1).
- Verification: [verified from primary source] for JT; Anderson-Woll [secondary/unverified — only via JT's description; AW paper not opened]

### A Fast GPU Algorithm for Graph Connectivity — Soman, Kothapalli, Narayanan (IIIT Hyderabad), IPDPS-W (LSPP) 2010; with Groute (Ben-Nun et al., PPoPP 2017), Gunrock and IrGL CC as described by ECL-CC
- URL: https://faculty.iiit.ac.in/~kkishore/conn_c.pdf (downloaded to `ccl/graph/web/soman.pdf`, `soman.txt`). Groute, Gunrock and IrGL are described only through ECL-CC section 2 (`eclcc_raw.txt`).
- Technique:
  - **Soman:** an iterated SV variant. Hooking with *no atomics* ("unordered algorithm"). **Alternating hook orientation** across iterations (even iterations: Parent[min] = max; odd: Parent[max] = min). Multi-level ("complete") pointer jumping in separate kernels, since global synchronization is needed between steps. Edges marked inactive once both endpoints share a parent.
  - **Groute:** splits the edge list into 2m/n segments, interleaving hooking with multiple pointer jumping per segment, and uses "atomic hooking" (locking the two representatives) to avoid re-iteration.
  - **Gunrock:** Soman plus filter operators (drop edges whose endpoints share a representative; drop vertices that are representatives).
  - **IrGL:** auto-generated Soman.
- Reported speed-up: Soman is 9-12x over sequential DFS on a Tesla C1060 (10M nodes / 60M edges in about 500 ms). "The speedup due to multilevel pointer jumping was more evident" on higher-diameter graphs. Per ECL-CC on a Titan X, ECL-CC is 1.8x faster than Groute, 4.0x faster than Soman, 6.4x than IrGL, 8.4x than Gunrock. GConn: 7.06x (unsampled) / 26.68x (sampled) over Soman's GPU-CC.
- Accuracy / determinism impact: Soman's alternating orientation makes the final root *not* the component minimum, which violates (a) and would need a relabel. Its unsynchronized hooking needs iteration until no edge is active (a data-dependent pass count). Groute's lock-based hooking violates (b).
- Applicability to vkapriltag: rejected. These are the SV-style lineage that ECL-CC and GConn beat by 1.8-26x. The one transferable observation, "multi-level jumping matters more on high-diameter graphs", is already embodied by `uf_compress` (Fini3) plus the A1 in-find compression.
- Verification: Soman [verified from primary source]; Groute, Gunrock, IrGL [secondary/unverified — via ECL-CC's description]

### In Search of the Fastest Concurrent Union-Find Algorithm — Alistarh, Fedorov, Koval (IST Austria / JetBrains), OPODIS 2019 (arXiv 1911.06347)
- URL: https://arxiv.org/abs/1911.06347 (downloaded to `ccl/graph/web/alistarh.pdf`, `alistarh.txt`)
- Technique: A multicore study of concurrent union-find (Intel 4-socket Xeon Gold 6150, 144 threads; AMD 6-socket EPYC 7571; ARM 2-socket ThunderX) over:
  - priorities: rank, pseudo-random, index;
  - compaction: none, splitting, halving, full;
  - optimizations: **plain (non-atomic, no-barrier) writes for compaction**, **early recognition** (walk both finds in step and stop at the lowest common ancestor, or link a root to the other walk's current node), **immediate parent check** (IPC: return at once if parent[u] == parent[v]), a lock-free Rem's algorithm with splitting, and HTM / lock elision.
- Reported speed-up:
  - Only 0.002% of compaction CASes fail in the worst case, so plain writes are safe and are **up to 40% faster** than CAS (volatile writes do not help).
  - IPC "significantly improves" all variants on their graphs, which have few components. Early recognition + IPC together is worse than IPC alone.
  - Rem's algorithm is better than most of the others, because it gets early recognition and IPC for free.
  - Splitting and halving perform about the same ("unable to adjudicate a clear winner"); full compression is consistently worse.
  - HTM coarse-grained locking was among the fastest (Intel only).
- Accuracy / determinism impact: compaction choices don't change the sets. Priority choice matters for (a): rank and pseudo-random priorities violate root = minimum index.
- Applicability to vkapriltag:
  - (1) **Plain stores for in-find compaction** is independent CPU confirmation of ECL-CC's plain store and of IISWC24's cost of atomics. Use plain stores for A1.
  - (2) **Splitting ≈ halving on CPU**, so on Mali the only reason to prefer halving is its half-as-many stores. That needs measuring.
  - (3) **IPC in `uf_merge`:** at a run-overlap start on the *first* merge pass, parent[i] (upper run's root) and parent[i+W] (lower run's root) are, after the post-init compress, roots of *different* runs, so IPC almost never fires. On the verification pass it is the in-tree rejected equal-root early-out (noise). Not worth re-proposing.
  - (4) **Early recognition** = UF-Early / Rem. See the ConnectIt and GConn entries.
  - (5) HTM: n/a on GPU.
- Verification: [verified from primary source] (text; figures not transcribed, so no per-graph numbers)

### Provably-Efficient and Internally-Deterministic Parallel Union-Find — Fedorov, Hashemi, Nadiradze, Alistarh (IST Austria), SPAA 2023 (arXiv 2304.09331)
- URL: https://arxiv.org/abs/2304.09331 (downloaded to `ccl/graph/web/pbbs_uf.pdf`, `detuf.txt`)
- Technique: Under a uniformly random edge order, T threads processing edges concurrently hit memory contention O(T² log|V| log|E|) times in expectation. On that basis the paper builds an *internally deterministic* parallel union-find (results match a sequential execution, including the tree shapes).
- Reported speed-up: the abstract says the "performance cost of internal determinism is limited". Numbers not extracted.
- Accuracy / determinism impact: its purpose is deterministic intermediate state.
- Applicability to vkapriltag: none needed. vkapriltag's *final* labels are already deterministic (component minimum after a full compress), and nothing downstream sees tree shapes. Its low-contention theorem assumes a random edge order. An image's overlap-start edges are spatially ordered and heavily correlated (large background component), so the premise doesn't hold. Noted for completeness.
- Verification: [verified from primary source] (abstract only)

### Contour Algorithm for Connectivity — Du, Alvarado Rodriguez, Li, Dindoost, Bader (NJIT), arXiv 2311.02811 (HPEC 2023)
- URL: https://arxiv.org/abs/2311.02811 (downloaded to `ccl/graph/web/contour.pdf`, `contour.txt`)
- Technique: Iterated **minimum-mapping**. For every edge (v, w), set L[v], L[w], L[L[v]] and L[L[w]] to the min of the endpoint labels ("order-2 operator"; order-m walks m label hops). Proven to converge in O(log d_max) iterations, O(m) work each. Optimizations: asynchronous in-place updates, **no atomics** (plain assignments; races only affect the iteration count), an early convergence check, and mixed operators (C-11mm: order 1 first, then higher order).
- Reported speed-up (CPU, Chapel/Arachne): 7.3x over FastSV and 1.4x over ConnectIt on average. On road_usa the order-1 operator needs 2369 iterations; order 2 cuts that "significantly"; order m = 1024 saves at most 3 more.
- Accuracy / determinism impact: minimum labels at the fixpoint: exact. The iteration count is nondeterministic under async races.
- Applicability to vkapriltag: (a) OK, (b) OK (no atomics at all). But (c) is its weak case: grid images are high-diameter, so even O(log d) full-image passes with a barrier each lose to one asynchronous union-find pass. One interesting detail: the "races only change the iteration count" design depends on a convergence loop, which vkapriltag has anyway (`changed_flag`). A hybrid would be "replace `atomicMin` hooking with a plain store and rely on the existing convergence loop". It is **not recommended**: a lost hook in pass 1 forces a second *real* pass, and passes cost about 1 ms on Mali. **Rejected.**
- Verification: [verified from primary source] (abstract and sections IV-V prose)


### Concurrent Disjoint Set Union — Jayanti, Tarjan, arXiv 2003.01203 (journal version merging PODC16 and JTB PODC19 "Randomized Concurrent Set Union and Generalized Wake-Up")
- URL: https://arxiv.org/abs/2003.01203 (downloaded to `ccl/graph/web/jt2003.pdf`, `jt2003.txt`); PODC19 DOI 10.1145/3293611.3331593 (not opened separately; the arXiv abstract's "second randomized algorithm ... valid even if the scheduler is adversarial" matches the JTB result, so it appears to subsume it — not confirmed)
- Technique: Three linking methods — deterministic linking by rank via DCAS, randomized linking by random index (CAS), randomized linking by rank (CAS) — each combined with one-try or two-try splitting. Correctness needs only "linking by index" (any fixed total order; CAS the smaller-ordered root under the larger) plus a weak property of compaction: parents only move up the union forest. Two-try splitting tries each pointer update twice (`CAS(u.p,v,w); v=u.p; w=v.p; CAS(u.p,v,w); u=v`).
- Reported speed-up: theory only. O(log n) steps per operation (worst-case for DCAS rank, w.h.p. for randomized) "even without path splitting"; without splitting the work is O(m log n). Two-try splitting gets the optimal O(m(α(n, m/(np)) + log(np/m + 1))) work; a matching lower bound is proved for symmetric algorithms. Key statement for GPU practice: **the analysis of sequential halving "relies on monotonicity of grandparents, which fails in the concurrent setting"; Anderson-Woll's claimed bound for concurrent halving is wrong and "we see no way to get a good work bound for their method"**, whereas splitting's analysis extends because parent monotonicity survives.
- Accuracy / determinism impact: linking by rank or random index makes the root something other than the minimum index (violates (a)). Linking by *fixed* index (vkapriltag's atomicMin on index order) is correct but carries no good work bound; its guarantee is only O(h) steps, h = union-forest height, which for index-ordered linking can be O(n) in the worst case.
- Applicability to vkapriltag:
  - Reinforces the JT16 entry: rank/random linking rejected under (a); DCAS unavailable in GLSL anyway (no 64-bit CAS guaranteed on Mali without `shaderInt64` atomics; the tree now uses int64 atomics in extents, but DCAS on two separate words is not a thing).
  - **Splitting vs halving on the GPU:** the theory favours splitting (A1 as measured) over halving in the concurrent setting; the halving arm (Conclusions item 2) is justified only by store count, not by any bound. With plain stores (not CAS) even splitting's monotonicity can regress (a late plain store may replace a newer, higher ancestor with an older one), so neither has a proven bound in vkapriltag's form; both remain correct because every stored value is an ancestor with a smaller index.
  - Two-try splitting: doubles stores, rejected for Mali on the same store-traffic grounds.
  - Worst-case index-order height: on a grid, pre-joined runs plus min-index hooking mean a hooked root always points at a smaller (earlier-row or leftward) root, and chains of hooked roots can grow along a serpentine region. This is the argument for in-find compaction (A1) rather than relying on `uf_compress` alone.
- Verification: [verified from primary source] (abstract, sections 1, 4, 5.1 text)

### Connected component identification and cluster update on GPU — Weigel (Mainz), Phys. Rev. E 84, 036709 (2011), arXiv 1105.5804
- URL: https://arxiv.org/abs/1105.5804 (downloaded to `ccl/graph/web/weigel1105.pdf`, `weigel1105.txt`)
- Technique: 2D lattice (Swendsen-Wang / percolation) cluster labelling, which is the physics twin of a 4-connected image CCL. Two phases: (1) label inside B x B shared-memory tiles by breadth-first search, tile-local union-find (balanced trees + partial path compression), or **self-labeling** (iterated min-label propagation, one thread per 2x2 sites); (2) consolidate across tiles by **label relaxation** (iterated min over boundary roots) or **hierarchical sewing** (2x2 tiles merged per level, log levels).
- Reported speed-up (GTX 480): self-labeling wins inside tiles for B up to about 128; tile-local union-find "is intrinsically serial" and slower. At L = 8192, critical q = 2 Potts: pure cluster identification 2.52 ns/site with hierarchical sewing, 6.56 ns/site with relaxation. Relaxation iterations grow as (L/B)^dmin, dmin ≈ 1.08, i.e. roughly linearly with the diameter.
- Accuracy / determinism impact: min-label based: exact labels.
- Applicability to vkapriltag: rejected. It predates the asynchronous atomic union-find approach (Komura 2015, Playne-Hawick 2018, ECL-CC) that superseded it. Its tile-local shared-memory phase is the in-tree rejected design on Mali (shared memory backed by L2). Its relaxation phase needs a diameter-proportional number of passes. The one confirming data point: iterated min-propagation pass counts scale with the shortest-path dimension (~1.08, i.e. nearly linear in image size) on lattice clusters, which is the quantitative reason why every round-synchronous method in this file loses on images.
- Verification: [verified from primary source] (text; figures not transcribed)

### A High-Performance MST Implementation for GPUs (ECL-MST) — Fallin, Gonzalez, Seo, Burtscher (Texas State), SC 2023
- URL: https://userweb.cs.txstate.edu/~burtscher/papers/sc23b.pdf (downloaded to `ccl/graph/web/sc23b.pdf`, `sc23b.txt`)
- Technique: Kruskal/Boruvka hybrid over a union-find, lock-free (atomicCAS unions, 64-bit atomicMin "deterministic reservations" of weight<<32 | edgeID). Two union-find-relevant optimizations:
  - **Implicit path compression**: no compression inside find at all; instead, whenever an edge is re-queued for the next round, the *representatives* returned by find are stored in the worklist in place of the original endpoint IDs, so the next round starts its finds at (near-)roots.
  - **Atomic guards**: a plain load checks whether an atomicMin could lower the value before issuing it.
- Reported speed-up (RTX 3080 Ti, 17 graphs, geomean, cumulative ablation): removing atomic guards +27% runtime; thread-only (no warp) +9%; no filtering +30%; **replacing implicit compression with explicit path halving (ECL-CC's GPU "intermediate pointer jumping") +58%**; processing both edge directions a further large loss. "We obtained the best performance by not including any explicit path compression." Overall 4.6x (Titan V) / 4.5x (3080 Ti) over the next-fastest code.
- Accuracy / determinism impact: none on the result; the 64-bit reservation makes MST edge choice deterministic.
- Applicability to vkapriltag:
  - **Atomic guards are already in-tree** in `reduce_extents_hash_body.glsl` (`if (x < extents[s].min_x) atomicMin(...)`). In `doUnion` a guard is pointless: the atomicMin target was just read as a root by `find()`.
  - **Implicit path compression → a stronger form of Jump2.** vkapriltag has no worklist, but it has the equivalent in the post-merge `uf_compress`: every pixel's entry becomes its root, so the verification pass's finds start at roots. That is already implicit compression between passes. *Within* the one real pass, the analogous move is Jump2 (store the found root into the starting pixel only). ECL-MST's ablation is the second independent data point (after ECL-CC's europe_osm result) that **fewer compaction stores can beat GPU path halving/splitting**, although in MST the win also came from shortening worklist-driven re-finds, so it is not a clean A/B of Jump2 vs Jump4. Strengthens Conclusions item 3 as the Mali-side arm.
- Verification: [verified from primary source] (sections 3.2 and 5.3 text; Table 5 per-graph numbers not transcribed)

### Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes: Lessons Learned from 1106 Programs (Indigo2) — Liu, Azami, VanAusdal, Burtscher (Texas State), SC 2023
- URL: https://userweb.cs.txstate.edu/~burtscher/papers/sc23a.pdf (downloaded to `ccl/graph/web/sc23a.pdf`, `sc23a.txt`)
- Technique: Hundreds of CUDA/OpenMP/C++ variants each of CC, MIS, PR, TC, BFS, SSSP, crossing styles: Atomic vs libcu++ CudaAtomic, vertex vs edge based, topology- vs data-driven, push vs pull, read-write vs read-modify-write, deterministic vs internally non-deterministic, persistent vs not, thread/warp/block granularity. (The CC codes here are label-propagation style, not ECL-CC union-find.)
- Reported speed-up (RTX 3090, Titan V): CudaAtomic with default (seq_cst, system scope) settings is a median **~10x (3090) / ~100x (Titan V) slower** than plain CUDA atomics for CC, recoverable only by specifying relaxed order and device scope. Read-write style is "slightly faster" than read-modify-write in most cases, up to 3x on GPUs. Internally non-deterministic codes are faster than deterministic ones for CC. Push beats pull for CC. Topology-driven is slower than data-driven on GPUs (median ratio < 1).
- Accuracy / determinism impact: the styles compared all produce the same CC result; "internally non-deterministic" refers to intermediate state only.
- Applicability to vkapriltag:
  - Memory-model lesson for GLSL: any atomics in `uf_merge` must stay relaxed and device-scoped. GLSL `atomicMin` on an SSBO is already relaxed, device scope, which is the fast configuration. Do not add `coherent`/`volatile` or Vulkan-memory-model acquire/release semantics to the parent buffer (see also IISWC24).
  - Read-write vs RMW: replacing the atomicMin hook with a plain store is the Contour-style hybrid already rejected above, because a lost hook forces another *real* pass (~1 ms on Mali), which outweighs a per-hook saving that is at most "slight" here.
  - Data-driven vs topology-driven: the verification merge pass is topology-driven, but it measured zero on the Pi when deleted, so there is nothing to recover.
- Verification: [verified from primary source] (sections 5.1, 5.5, 5.6 text; figure values not transcribed)

### Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs — Prokopenko, Lebrun-Grandié, Arndt (ORNL, ArborX), arXiv 2103.05162 (2021)
- URL: https://arxiv.org/abs/2103.05162 (downloaded to `ccl/graph/web/dbscan2103.pdf`, `dbscan2103.txt`)
- Technique: DBSCAN as connected components over an implicit neighbour graph found by BVH traversal, with Union called from inside the traversal callback ("fused" FDbscan). Union-find: "We chose the algorithm proposed in [Jaiganesh and Burtscher, 2018]", i.e. ECL-CC's lock-free CAS hook plus intermediate pointer jumping, and each undirected pair is unioned once.
- Reported speed-up: DBSCAN-level numbers only; no union-find variant study.
- Accuracy / determinism impact: same as ECL-CC (min-based, exact labels).
- Applicability to vkapriltag: no new variant. It is evidence that ECL-CC's find/hook is the de-facto GPU union-find in production libraries (ArborX / Kokkos) in 2021-2023; nothing newer displaced it there.
- Verification: [verified from primary source] (section 4.3 text)

## Conclusions

Constraints used for every item: (a) root = minimum index must survive; (b) no forward-progress assumption between invocations (no spin locks, no waiting on another invocation's write); (c) 4-connected grid, horizontal runs pre-joined by `uf_init`, unions only at run-overlap starts, one real merge pass plus a verification pass. "Exact" means bit-identical labels after the final `uf_compress`. Every variant ranked below only ever stores a smaller-index *ancestor* into `parent[x]` (min-based), so the final flat label is still the component minimum. Tree shapes change, labels don't.

**Cheapest way to bound items 1-5 together:** put the find variant and the union variant behind two specialization constants in `uf_merge_body.glsl`:
- `FIND_MODE`: 0 naive, 1 split (A1), 2 halve, 3 jump2.
- `UNION_MODE`: 0 current, 1 lock-step, 2 Rem.

Build one binary and run the existing ABBA harness at d1/d2/d4 on both the Pi and the RX 9060 XT, plus the 36-configuration bit-identity matrix. Specialization constants are resolved at pipeline creation, so the A/B arms have no runtime cost.

### Ranked variants

1. **A1 as-is: ECL-CC Jump4, which by Jayanti-Tarjan's definitions is path *splitting*, with plain stores. Run it on the Pi.**
   - Expected: measured −17% labelling at d1 and −2.7% at d2 on the RX 9060 XT. Unmeasured on Mali; the survey estimates 1-5% of GPU total at d2.
   - Constraints: meets (a), (b) and (c). Exact: bit-identical over 36 configurations on RDNA.
   - Device: both. Mali is the open question, because the extra stores go over the LPDDR bus it shares with the CPU.
   - Keep the stores plain and non-`coherent`. IISWC24 found race-free relaxed atomics cost 12-55% of CC time. Alistarh found plain compaction writes up to 40% faster than CAS. Indigo2 found default-order libcu++ atomics about 10-100x slower.
   - Theory also favours splitting over halving in the concurrent setting (JT 2003.01203: the analysis of halving "fails in the concurrent setting").
   - Bound: it is already written, so this is just one ABBA session.
2. **ECL-CC Jump2 (walk naively, then one plain store `parent[start] = root` per find).**
   - Store count: one store per find instead of one per visited node.
   - Evidence: it is 1.26x slower than Jump4 on ECL-CC's average, but **18% faster on europe_osm**, the only high-diameter, long-path input, which is the class a thresholded image belongs to. ECL-MST (SC23) independently found that "no explicit path compression", with representatives stored for reuse, beat GPU path halving by 58% of runtime. That is not a clean Jump2-vs-Jump4 test, but it is a second data point that fewer compaction stores can beat in-walk splitting on GPUs.
   - Expected: on Mali, between −3% and +3% of labelling relative to A1. On RDNA, probably slightly worse than A1.
   - Exact. Device: aimed at Mali. Bound: FIND_MODE 3 in the same session. About 3 lines of code.
3. **True path halving (`p=P[x]; g=P[p]; if(p==g) return p; P[x]=g; x=g;`).**
   - Same dependent-load chain as A1, with half the stores.
   - Evidence: Alistarh found splitting ≈ halving on CPUs. JT gives no concurrent bound for halving. So the only case for it is store traffic on Mali.
   - Expected: on Mali, 0 to a few percent of labelling relative to A1; about 0 on RDNA.
   - Exact. Bound: FIND_MODE 2.
4. **Lock-step two-chain walk (UF-Early / GConn Union-Early lite).** Advance both find chains in one loop and exit early on a common ancestor:
   ```glsl
   uint ra=a, rb=b, pa=parent[a], pb=parent[b];
   while ((pa!=ra || pb!=rb) && ra!=rb) {
     if (pa!=ra) { ra=pa; pa=parent[ra]; }
     if (pb!=rb) { rb=pb; pb=parent[rb]; }
   }
   // then hook exactly as today (atomicMin, retry through old)
   ```
   - Why: today `doUnion` runs two *serialized* dependent chains, and labelling is dependent-load-latency bound on Mali. This puts two independent loads in flight per iteration. It is the only item aimed at latency rather than at total path length.
   - Evidence: no static GPU number exists. GConn's incremental Table 4 (V100) shows Early 2-8x behind Async on high-degree graphs, but tied or marginally ahead on europe_osm (6.83e9 vs 6.74e9 ops/s) and 0.74x on road_usa. That row-to-graph mapping was reconstructed from the extracted text.
   - Exact. Device: aimed at Mali. Bound: UNION_MODE 1 on the Pi. Drop it if it gains less than 3% of the `uf_merge` span.
5. **UF-Rem-CAS in atomicMin form, with a split step.**
   - Evidence: ConnectIt measured it 1.62x faster than UF-Async on 72 CPU cores. GConn on a V100 measured Async and Rem-CAS tied (Rem 1.02x slower). So expect about 0 ±5% of the `uf_merge` span.
   - It gets early recognition and the immediate-parent check for free, and it interleaves the two sides (one load per step, not two in flight).
   - Sketch, with smaller index = higher in the tree:
     ```glsl
     uint pu=parent[u], pv=parent[v]; bool merged=false;
     while (pu != pv) {
       if (pu < pv) { swap(u,v); swap(pu,pv); }        // now pu > pv
       if (pu == u) {                                   // u is a root
         uint old = atomicMin(parent[u], pv);
         if (old == u) return true;
         merged = true; u = old; pu = parent[u];        // lost race: continue from old
       } else {
         uint gp = parent[pu]; parent[u] = gp;          // SplitAtomicOne, plain store
         u = pu; pu = gp;
       }
     }
     return merged;
     ```
   - Constraints: (a) holds, because only smaller indices are ever stored. (b) holds, because a lost race continues through `old`, the same pattern as today's `doUnion`. If `pv < old`, the atomicMin has re-parented u under pv, and continuing with u = old re-joins old's component. (c) holds.
   - Do **not** use Rem's SpliceAtomic rule (re-parent u under pv): ConnectIt shows it is correct only phase-concurrently, and it is incorrect when combined with a compressing find.
   - Exact. Device: both. Bound: UNION_MODE 2, and only worth pursuing if items 1-4 leave `uf_merge` dominant.
6. **ECL-CC "retry without re-find" and Afforest's read-only grandparent step on the retry path.** Tiny, because lost atomicMins are rare except near the roots of big components. Exact. Fold these in only if `doUnion` is rewritten anyway (items 4 and 5).
7. **Post-init `uf_compress` as asynchronous pointer doubling [speculative].**
   - Idea: each thread writes `parent[i] = p` with a plain store as it walks, in the style of ECL-CC Fini1 or Wyllie pointer jumping, instead of walking read-only.
   - Why: the post-init compress walks every pixel to its run start. That is O(run length) dependent loads per pixel, and O(L²) per long background run. With self-writes, lock-stepped neighbours can hop over entries that have already been shortened.
   - Evidence: ECL-CC measured Fini1 within 0.5% of Fini3, but only on short-path graphs. There is no evidence for the long-run case. The in-tree rejection of a direct run-start *scan* (2.91 vs 2.57 ms) was about O(L) *known-address* reads, not about this.
   - Exact. Device: both.
   - Bound first: timestamp the post-init compress span alone on the Pi. Skip the idea if that span is under ~0.15 ms.

### Rejected in this category

- **Sampling + skip-largest-component (Afforest, ConnectIt, GConn k-out/HB).** `uf_init` already is the first-neighbour sample. The skip would only target the verification pass, which measured zero on the Pi when deleted, and it reduces to the rejected equal-root early-out. GConn measured every sampling scheme as a slowdown on low-degree road graphs.
- **Randomized or rank linking (JT16, JTB19/JT 2003.01203, Anderson-Woll).** Violates (a). The DCAS variant is also unavailable in GLSL.
- **Lock-based hooking (UF-Rem-Lock, Groute).** Violates (b). GConn measured Rem-Lock 3.81x slower on a V100 anyway.
- **Union-Hooks.** GConn measured it 1.44x slower on GPU (memory barrier), and it needs an extra buffer.
- **Synchronous round-based min-labelling (SV, FastSV, Liu-Tarjan, Soman, Contour, label propagation, Weigel relaxation).** These need O(log n) to O(diameter) full-image passes, each with a 2.6-18.7 µs barrier on Mali, against one asynchronous pass today. Pass counts grow roughly linearly with lattice size (Weigel: dmin ≈ 1.08). They are 1.8-26x slower than ECL-CC and GConn on NVIDIA.
- **Frontier/BFS CC (cuGraph WCC).** Needs O(diameter) rounds, and its labels are not minimum indices.
- **Plain-store hooking relying on the convergence loop (Contour hybrid; Indigo2 read-write style).** A lost hook forces another *real* merge pass, about 1 ms on Mali.
- **Atomic or `coherent` parent accesses "for correctness".** 12-55% slower (IISWC24).
- **Deterministic-intermediate union-find (SPAA23).** Unnecessary: the final labels are already deterministic.
- **Vertical pre-link of run starts in `uf_init`** (`parent[runstart] = runstart - W` when the pixel above has the same value; ECL Init3 extended upward) [speculative]. It is contention-free and exact, but it turns run-start chains into column-long chains for the post-init compress: O(x + y) per pixel instead of O(x). Only reconsider it if item 7 lands.

### What could not be verified

- The Afforest paper: the HUJI link returns an error page and the ETH record has no reachable PDF. Its details come from its code only.
- Anderson-Woll: known only through JT.
- Groute, Gunrock and IrGL: known only through ECL-CC.
- The Patwary-Blair-Manne SEA 2010 union-find experiments: not reachable, and the web-search budget is exhausted.
- GConn's static per-variant (find-rule) numbers: they are in figures only. The Table 4 row mapping was reconstructed.
- Mali (Valhall) L1 behaviour for non-coherent SSBO loads and stores.
- **Any Mali measurement of items 1-7.** Every uplift above is either measured on RDNA (item 1 only) or extrapolated from NVIDIA or CPU literature.
