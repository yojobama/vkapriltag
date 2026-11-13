# Decision log

One row per idea that was built, measured or proposed: what it is, its status today, the key number as the source
states it, where it is recorded, and what superseded it. Seeded from the mining report of 2026-11 (a read of
`PERFORMANCE.md`, `OPTIMIZATION_NOTES.md` and `DESIGN.md` as they stood at 1.10.1, with section and line citations) and
extended with the decisions of the 2026-11 restructuring. How the pipeline works now: [ARCHITECTURE.md](ARCHITECTURE.md).
Why it is shaped this way: [DESIGN.md](DESIGN.md).

**Status.** *default*: on in the code today. *opt-in*: in the code, off by default (the switch is in
[PERFORMANCE.md](PERFORMANCE.md) section 2). *dropped*: tried and not kept (code deleted or never merged). *removed in
2026-11*: the code or switch existed until the 2026-11 restructuring and was deleted then. *bounded*: analysed, never built.
*n/a*: judged not applicable. A row ending in **(unverified)** gives a number or status that could not be checked against a
source or the code in this pass and comes from the mining report only.

**Sources.** `P §x` is `archive/PERFORMANCE-history.md` section x (the numbering the changelog cites; sections 8ag to 8an never
existed). `O n` is `archive/OPTIMIZATION_NOTES.md` item n (or its line range). `PR name` is `research/PROGRESS-name.md`.
`CL` is `CHANGELOG.md`. `RR` is `research/RESTRUCTURING-REPORT-2026-11.md`. The numbers are the source's: desktop is the RX 9060 XT,
Pi is the Orange Pi 5 / 5 Plus (Mali-G610).

## 1. GPU labelling and thresholding

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Tile-local union-find in shared memory | dropped | label 16x16 tiles in shared memory, then merge tiles | "16x16 tiles: 3.58->5.32 ms" on the Mali; "plausibly win on a discrete GPU" (never tried there) | O (tile section) | none |
| Block CCL (BUF, BKE, Playne) | n/a | 2x2-block labelling | judged "4-connected and three-valued"; `research/00-SUMMARY.md` finding 0.1 says upstream unions white 8-connected and this code 4-connected, which reopens it; never reconciled | O, P §8 (old) | open |
| HA4 run-level merge | default | pixels point at their run start | labelling -7 to -13%; the strip/warp half was not built | O 7 | `uf_init_scan` |
| Direct run-start scan | dropped, later superseded in effect | per-pixel run start without the merge | 2.91 vs 2.57 ms | O (run-start entry) | `uf_init_scan` (P §8t) |
| `label_pixels` | removed (before 2026-11) | labelling in one shader | added in O 7, deleted in O 18: -2.4% | O 18, P §8k | `uf_init_scan` + `uf_merge` |
| Path splitting in `find()` | default | the only find; `FIND_MODE` removed in 2026-11 | RX labelling -16.5%; Mali -1.7% / -3.2% | O 10, `PR del-switches` | none |
| One `uf_merge` pass, verification merge off | default | retrying `atomicMin`; `UF_VERIFY=1` restores the verification and the retry path | 4.71->4.32 / 2.20->2.03 ms; "the verification merge never found a change in 105 configurations" | P §8t | none |
| `uf_init_scan` | default (needs subgroup arithmetic) | each pixel starts at its run start, found with a subgroup scan | removes the compression before the merge | P §8t | none |
| Banded merge | default on discrete | 8-row bands, then across the bands | `uf_merge` 95->59 / 86->32 us on the RX; on the Mali "in-band pass alone costs nearly as much" | P §8u | none |
| `uf_merge` four pixels per thread | default on integrated | word loads, 4 pixels per thread | 578->432 us on the Mali with word loads; RX 92->124 us, so not used there; first rejected at +6% (O 9) | P §8t, O 28 | none |
| Count runs (`uf_count_runs`) | default on integrated | blob sizes counted per run piece | Pi `Detect()` 3.59->3.44 ms; RX 15-20 us slower | P §8u | none |
| Compacted (sparse) labelling | default on integrated | dispatch over the 64x4 cells that hold contrast | Pi GPU at decimation 1 about -12%; slower on the desktop, so off there. The name `SPARSE` meant predicated skipping in P §8ac and compaction in P §8ad | P §8ad, §8af | none |
| Predicated tile skipping | dropped | labelling passes return early for flat cells | "each pass gets 40 to 180 us slower" | P §8ac | compaction |
| `blob_diff` four per thread, two-pass, shuffle | dropped | wider boundary pass | +11% (O 9); the shuffle result is cited only as "section 8ag", which does not exist **(unverified)** | O 9, P §8t, §8w, §8ap | none |
| Precomputed W; tiled image with `texelFetch` | dropped | threshold variants | `threshold` +29 to +35% | O (threshold section) | none |
| Fused preprocessing (decimate + min/max, filter + threshold) | default (round 4 version) | `block_minmax_decimate_u8`, `threshold_block_u8`; `FUSED_THRESHOLD=0` turns it off | first try rejected: Mali +5.9 / +5.4 / +7.8%; round 4: Pi `Detect()` 4.09->4.05 / 1.93->1.90 ms | O 1241-1268, P §8t | none |
| Fused threshold + `uf_init_scan` | dropped | one dispatch for both | estimate -60 to -100 us; **no measurement is recorded** | P §8af, CL "Tried and not kept" | none |
| Adaptive contrast floor | default | floor = `adaptive_floor_scale` (2.0) x the median 4x4 block range | Pi decimation 1 6.89->2.29 ms; "loses 1-3 small degraded tags"; k = 2.5 loses underexposed recall (100 to 92%) | P §8ac (15), §8ae | none |
| `NOISE_GUARD` | dropped | NAPA idea | loses recall (small-tags 100 to 51 to 77%) | P §8ac (11) | adaptive floor |
| `MIN_DIFF` 20 | opt-in (`MIN_DIFF`), not the default | larger contrast floor | desktop -55 to -57%; its status was never stated | P §8ac | adaptive floor |
| Box decimation | default (decimation 2 and above) | average each block; with the integer dot product where available | d2 time -38 to -47% desktop, -45 to -49% Pi; blur error halved; tag16h5 wrong ids 779->189 | P §8ac (12), §8ao, §8aq | none |
| Hash grouping instead of the radix sort | default | (rep0, rep1) hash table | "roughly 30 dispatches ... collapse to 4"; sort + group 10.90->3.06 ms | O 4 | none |

## 2. GPU tail: boundary points, extents, selection, sort

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Sort lineage | default (last stage) | bitonic, Batcher network, comparator enumeration (RX -6 to -8%), rank-merge, rank-all up to 256 points, split sort for large blobs, any blob size up to 65,536 | rank-merge default since P §8k; the Batcher network (`SORT_CHUNK=0`) and its switch removed in 2026-11 | O 9, 15, P §8k, §8u, §8r, §8ab, `PR del-switches` | none |
| Sort split under several streams | default at decimation 1 | `SORT_SPLIT=8192` for streams spread over families | -5% / -10% | P §8r | none |
| Sort tie-break | dropped (not built) | order equal keys | O 13 found zero distinct-point ties; P §8au found real run-to-run differences it "removes"; P §8av: "not built" | O 13, P §8au, §8av | open: the front end stays non-deterministic |
| Subgroup aggregation | default at one site, discrete only | `reduce_extents_hash_subgroup`; retired at `uf_final` and `blob_diff` | `uf_final` 0.1035 vs 0.0291 ms, `blob_diff` 0.0437 vs 0.0330 ms (plain atomics win); on the Mali the 1.7-era `pipeline_total` went 11.5->18.1 ms; rule: aggregate values by key, not counters | P §6 (old) | none |
| Atomic64 / x4 extents variants | default on devices without subgroup use | 64-bit atomic add of count and moments; 4 points per thread | extents contention analysis in P §3b; the 32-bit variant has no named user device | P §3b, §8t | none |
| One-point-per-thread extents shaders, `REDUCE_X4` | removed in 2026-11 | the plain variants | `reduce_extents_hash.comp` and the atomic64 one deleted; `reduce_x4_ = !subgroup` | `PR del-switches` | x4 variants |
| `RawLineFitPoint` 48->16->8 bytes | default | packed line-fit record | 8-byte step: Mali GPU nil, memory -15%; RX -3 to -10% | O 2, P §6b | none |
| Line-fit records read in place | default | host reads the mapped device buffer | (no number) | O (second pass A2) | none |
| GPU quad fit (`GPU_QUADFIT`, `fit_quad_dp.comp`) | removed (before 2026-11) | fit on the GPU | serial +4 to +8% on the desktop; +8-9% throughput when CPU-bound, -13% when GPU-bound; needs float64, int64 and subgroup extended types, so cannot run on the Mali | P §8z, §8aa, O 32 | CPU `QuadDecode` |
| `dp_peaks_check` | removed (before 2026-11) | fall back to peaks if DP is poor | (header of P §8z says removed, body says opt-in) | P §8z | none |
| Skip points of clusters that selection rejects | bounded | needs a counting pass | 71% of 166,000 points could go; at most about 0.44 ms | P §8r, O 27 | none |
| GPU pre-filter of certain rejects | bounded | | saves 40-110 us of `quad_decode`; "worth little" | P §8y, O 31 | none |
| Persistent tail (grid barriers) | dropped (branch `perf/pf-m3-persistent-tail`) | whole tail in one dispatch | about 40 us on small frames against an 80 us gate; slower than the lean tail on every set | P §8aw, §8ay | lean tail |
| Lean tail | default on integrated | bit 1: fixed-grid kernels, no argument kernels; bit 2: `select_blobs` writes offsets | Pi small frames 0.595->0.549-0.553 ms (-45 us, 7.7%); bit 4 (omit the large sorts speculatively) reverted: 8 us against a 15 us gate | P §8ax | none |

## 3. Submit and dispatch structure

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Collapse four submits to one (first pass) | dropped, later reversed | | empty round trip 0.019 / 0.029 ms on the Mali; "not worth the refactor" | O (first pass) | the fused flow below |
| Four to three submits (A1) | dropped | commit `bcfa3dc`, reverted by `ecaeaae` | moved the residual 0.05 ms, not the predicted 0.37 ms | O (A1) | fused flow |
| Two of three readbacks removed | default | `build_indirect_args` and `count_from_buffer` | GPU total -2.0 / -4.0 / -7.9% at decimation 1 / 2 / 4 (bound -7.9 to -10.1%) | O 5, P §3c | fused flow |
| Last boundary closed by a speculative record plus retry | default (retry now only with `UF_VERIFY`) | | -0.7% (decimation 1) to -5.2% (decimation 4); exposed a destructive in-place rewrite (DEVICE_LOST) and a missing compute->transfer barrier | O 8, P §3c | none |
| Fused path on RDNA4 via `HostVisibleCached` | default | host-cached readback on the desktop | RX GPU total -19.8% / -21.3%; saves about 0.19 ms | P §3c | none |
| Unfused four-submit flow | default only where host-cached readback is missing | `DetectUnfused` | "No measured device is stuck on it"; +11.5% `Detect()` on the Mali; dead `fuse` branches and the duplicated timestamp code removed in 2026-11 (`CollectGpuTimings`) | P §8g, `PR del-experiments` | none |
| `MemoryKind::DeviceLocalReadback` | removed from use | old readback kind | replaced by `HostVisibleCached` for both readbacks; the enum value was deleted in 2026-11 | P §3c | `HostVisibleCached` |
| Pre-recorded command buffers | default | record once, replay; `PRERECORD=0` kept as a debugging switch (per-dispatch timing, capture) | desktop `Detect()` -4.7%; Pi -3.2%, host overhead 193->127 us | P §8f, §8g, `PR del-switches` | none |
| Six fewer dispatches | default | `init_extents` into `hash_group`, copies into `select_blobs`, one-workgroup scan | 4.12->4.09 / 1.99->1.93 ms; "a tiny dispatch with its barrier costs about 10 us" | P §8t, O 28 | none |
| "Dispatch diet" | bounded | | about 8 dispatches at 3.5 us, near 1% | P §8k | none |
| Fused dispatch arguments (`FUSE_ARGS`) | dropped | | Pi +0.05 to +0.12 ms; desktop net +7 us | P §8ad, §8ae | none |
| Defer the barrier after the fills | dropped | | 2.3103 vs 2.3126 ms | P §8k | none |
| What a barrier costs | measurement | | about 1-2 us each, 33 barriers = 80 us (3%) on the Mali (P §8h); no barriers 1.536->1.318 / 0.821->0.632 ms, "180 to 190 us is the serialisation itself" (P §8ap) | P §8h, §8ap | none |
| Completion by polling a flag | default with `VK_AMD_device_coherent_memory`, opt-in on the Mali | shader writes a flag the host polls | RX 0.194->0.145 ms; Pi GPU 1.522 / 0.828->1.454 / 0.740 ms; outside the spec on the Mali | P §8au | none |
| Spinning on the fence (`SPIN_WAIT`) | opt-in | | desktop 0.60 ms either way; Pi saves about 30 us (rejected in 8e and 8w, shipped opt-in in 8as) | P §8e, §8w, §8as | none |
| Lean tail, flag polling, compaction as a set | default on integrated | see sections 1 and 2 | Pi corpus median 2.549->1.058 ms (decimation 1) over 1.9.0 with the accuracy changes | P §8az | none |
| `VK_EXT_conditional_rendering` for `uf_compress` | default where available, used only in the `UF_VERIFY` path | predicated compression | measured on the four-submit path only (P §6c); never re-measured on the fused path | P §6c | none |

## 4. Ingest and memory

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Host-pointer import (`ImportHostFrame`) | opt-in (API) | GPU reads the caller's buffer | +26-29% throughput; `Detect()` -14%; `threshold` span +21 us (reads over PCIe) | P §8d | none |
| dma-buf import + `decimate_yuyv` | opt-in (API) | | `Detect()` 2.79->2.60 ms; UVC buffers need a `dc civac` flush (0.03-0.17 ms) | P §8d | none |
| Transfer-only queue upload | opt-in | `ContextOptions::use_transfer_family` | +8% / +28% at 4 / 8 streams, -10% for one stream | P §8d | none |
| Frames written into mappable device memory (discrete) | default | `GRAY_MAPPED` removed in 2026-11 | 4 streams +13 to +32%; removes a 38 us copy per frame | P §8u, `PR del-switches` | none |
| Decimation 1 reads the grey buffer in place | default | `DEC1_ALIAS` removed in 2026-11 | Pi -2.6% | P §8q, `PR del-switches` | none |
| Batching cameras into a mosaic | removed | | one stream 1.7x (RX), 1.27x (Mali); "concurrent streams already reach the same saturation" | P §8d | multi-stream lanes |
| Hardware JPEG decode into a dma-buf | bounded | | open | P §8 (old) | none |
| `FramePipeline` copies the line-fit records | open | double-buffering the read-back would avoid it | not done | P §5, §8p | none |
| Subtle: GPU memory sizing | default | dense worst case 4 points per pixel; `MAX_POINTS` caps it | 447 MiB at 1080p dense, 56 MiB at 200,000 points | P §6 (old), PERFORMANCE.md section 4 | none |

## 5. Several streams and pipelining

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| `PipelinedDetector` (CPU tail overlapped with the next GPU pass) | dropped (branch `PipelineCpuTail`) | | 1280x800 21% worse on the Mali; GPU 5.70->7.84 ms; "shared LPDDR bus or fence wait" (theory) | O (first pass) | `FramePipeline` |
| `FramePipeline` depth 1 | default API (opt-in use) | GPU pass on its own thread, decode the previous frame | Pi 2.87->2.2 ms; desktop about 0.85->0.43; neutral at 640x400 on the Pi. Contradicts the row above; resolved by shipping it | P §5, §3 | none |
| `FramePipeline` depth 2 | opt-in | second `GpuDetector` | Mali utilisation 83-93% -> 99%; latency 9.2->12.9 ms on the Pi | P §8r | none |
| Separate `VkDevice` per stream; queue per stream | measured | | RX 4 streams alternating 2.7x; Pi almost none; one queue per stream slightly worse on the Mali | P §8b, §8c | `expected_streams` |
| `expected_streams` / `StreamSlot` | opt-in | queues on both compute families | desktop +82% / +84% at 4 / 8 streams; Mali no change | P §8j | none |
| `MultiLevelDetector` | opt-in, kept as a class | per-level decimation and tag size band | Pi 2.4-3.4 ms against 1.7-2.2 ms for decimation 1; recall lower (low-contrast 98 to 84%, grey-margin 88 to 22%); "a level only pays if restricted to a region" | P §8ac, CL 1.10.0 | open |
| CPU-only or hybrid front end; CPU tail for sparse frames; ROI processing | not tried | | the documents only analyse why NAPA wins on small frames (Pi 5-6x, desktop 3.6x faster at decimation 2); "the GPU pays off only where per-pixel work dominates"; deferred in the 2026-11 plan | P §8ar, §8at, §8as, RR 6.8-6.9 | open |
| Adaptive decimation | not tried | | decimation 4 is "2.1x faster" than 2 | `survey` C2, P §3 (old) | open |

## 6. CPU: refinement, decode and the accuracy features

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| `exact` refinement | opt-in (`REFINE=exact`; also the profile fit's fallback) | upstream's arithmetic in double, `modf` replaced | `tag_decode` -25% on the Pi, -4.7% on x86 against upstream | P §4 (old) | profile fit |
| `fast`, `ultrafast`, `upstream` refinement | removed in 2026-11 | single precision; plus a decode gate; upstream's own function | `fast` +1% on x86 and nothing on ARM; `ultrafast` `tag_decode` 0.095 ms but 663 vs 687 of 768 decoded; `PERFORMANCE` 8ba used `fast`/`ultrafast` (the comparisons stay in the history) | P §4 (old), §8ba, `PR del-refine`, CL "Removed" | profile fit |
| Profile fit (`kProfile`) | default | dense edge profiles, weighted TLS line | median corner error "about 3x lower"; time -1 to -18% desktop | P §8ac (01) | none |
| Outlier drop (bit 1) | default | | halves the error with a cable across the border | P §8ac (02) | none |
| Coarse-to-fine passes (bit 2) | default (mask 31) although P §8ac's verdict was "drop" ("almost none", +/-3%); the reason for keeping it is not recorded **(unverified)** | | | P §8ac (03) | none |
| Halfway crossing (bit 4) | default | | blur error 0.146->0.122 px ("marginal") | P §8ac (04) | none |
| Clip bias (bit 8) | default | | overexposed error 0.22->0.07 px; -2 to -8% time | P §8ac (05) | none |
| Level pass for motion blur (bit 16) | default (new, 2026-11) | halfway crossing at the mid level when more than 25% of profiles touch a window end | fast-1tag-1280 d2 0.360 / 1.463 -> 0.059 / 0.114 px (NAPA 0.047 / 0.117); corpus blur p95 0.162->0.068, motion-blur 0.456->0.181; +0.07 ms on blurred frames; `PROFILE_FEATURES=15` restores | P §8bd, `PR blur-corners` | none |
| Margin gate (`probe_margin`) | opt-in | skip refinement of quads whose unrefined margin is low | Pi `pipeline_total` -12.5% (P §8l); "97% of refinement time goes to quads that never decode"; risk: a real tag under the gate | P §8l, README (old) | early reject |
| Hamming-distance probe | dropped | | "loses 9 to 20% of the tags that only decode after refinement"; accepted quads sit at distance 0 to 10, rejects at 5 to 10: "no safe cut" | P §8l, `PR early-reject` | early reject |
| Early reject with grey margin on | default (new, 2026-11) | border and dark checks on the unrefined quad, kept only if the inner W cells fail too; exemptions under 1.5 px per cell, coarse boxes, rescue candidates | CPU -22% / -20% clutter-only, -18% / -9% busy-background, -17% / -5% jpeg; +6% / +4% dense-tags; no tag lost on the corpus, tag16h5 false 3->1; `EARLY_REJECT=0` restores | P §8be, `PR early-reject` | none |
| Grey margin retry | default | decode the inner W cells when a grey quiet zone made the outline the (W+2)-cell square | grey-margin recall 48 -> 98% | P §8ac (06) | none |
| Side one out | default although P §8ac's verdict was "drop" ("never triggered on the corpus") **(unverified reason)** | refit a side one cell in | | P §8ac (07) | none |
| Third bit | default | a third corrected bit when the three wrong cells touch | obstruction recall 73 -> 86%, no new false ids; exempt from the family limit | P §8ac (08), §8ad | none |
| Family limits | default | reject detections that needed more corrected bits than the family tolerates | tag16h5 false detections 1739->14 (d1), 779->3 (d2) | P §8ac (09) | none |
| Border pre-check, dark reject | default | cheap rejects before refinement | false detections halved; -2 to -12% / -2 to -8% time | P §8ac (10, 13) | early reject extends them |
| Small-tag rescue | opt-in | labels blobs down to 8 points, refits boxes from the full-resolution image | `range` d2 351/375 -> 375/375 (12 px tags 6 -> 25); cost cut about in half but clutter-only +0.03, busy-background +0.05 ms (+11%) stay over the 5% / 0.02 ms gate, so default off | `PR range`, `PR rescue-cost`, `PR small-rescue-default` | none |
| Hull quad | opt-in | quad from the convex hull for a partly covered border | border-occlusion d2 43.0 -> 68.3%, d1 23.5 -> 68.7% (NAPA 56.5%); only that scenario changes; capped at 6 fits per frame | `PR occlusion`, `PR rescue-cost` | none |
| Polish (whole-tag photometric fit) | opt-in | | corner error p95 (d2): motion-blur 0.456->0.167, overexposed 0.294->0.053, blur 0.162->0.065, range 0.100->0.048; cost per ordinary tag 0.42->0.18 ms after the speed-up (one-tag 0.31->0.43 ms, dense-tags 1.52->6.80 ms one thread) | `PR corners`, `PR polish-speed`, `PR polish-speed2` | none |
| Own decode (`OwnDecode`) | default (new, 2026-11) | port of the quad decode, code lookup and reconcile with no per-attempt allocation | bit-identical to patched upstream on 22,560 random quads, 2,974 corpus attempts and 30,000 reconcile sets, about 330k in-process comparisons; CPU per frame 2 to 14% lower; the decode alone 1.97x faster (480k -> 945k attempts/s) | `PR own-decode` | none |
| Own types (`Detection`, `Family`, `Image`, `Quad`) | default (new, 2026-11) | replace upstream's types at every boundary | output unchanged; no CPU change beyond noise | `PR own-types`, CL "Changed (API)" | none |
| Stable id order | default (new, 2026-11) | `SortById` is a stable sort; equal ids keep reconcile's order | `qsort` ordered equal ids differently on MSVC and glibc (`own_decode_reconcile` failed on aarch64) | `PR no-upstream` (follow-up) | none |
| Decode split into candidate steps | default (new, 2026-11) | `Decode` -> `RunSteps` over named steps | cyclomatic complexity of `Decode` 152 -> 8; unrefined quad sampled once instead of twice with the margin gate | `PR decode-steps` | none |
| CPU geometry header | default (new, 2026-11) | one `Geometry.h` for the duplicated helpers | library `.cpp` -330 / +168 lines; timing within 2%; a first `TapAlong` that returned a struct made the default path 15 to 40% slower until written through pointers | `PR geometry` | none |
| Quad fit variance | measurement | the corner error is mostly estimator variance of side angles | e.g. blur tilt 3.1x position error, range 6.7x | `PR corners` | polish |
| `QuadFitMethod::kPeaks` | opt-in (`QUADFIT=peaks`) and the DP fallback | | DP falls back for about 8% of blobs | README (old) | DP default |

## 7. Tracking and composites

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Keyframe tracking | opt-in | predicted corners refined and decoded on tracked frames, full GPU on every 8th | slow-1tag-640 median 0.54 -> 0.077 ms with identical accuracy; fast motion 0 to 23% of frames tracked; `max_iterations` 8 -> 3 refuses biased results | `PR tracking`, `PR tracking2` | none |
| Async tracking | opt-in | full detector on a background thread; tentative tracks measured on the current frame | gap 16 ms: medium-2tag-640 full 0.926 -> async 0.312 ms; recall bimodal on `enter-3tag-640` (84% or 95%) | `PR tracking2` | none |
| ROI search with a crop through libapriltag | removed in 2026-11 | | "1.3-3.6 ms ... more than the full GPU path" | `PR tracking`, `PR del-experiments` | none |
| Capture starts (constant acceleration, decimated crop) | removed in 2026-11 | | a capture frame costs 0.5-0.7 ms (2x a full detect); 10-20% more fast frames; fast-2tag recall falls to 90% | `PR tracking2`, `PR del-experiments` | none |
| `max_speed_px` | removed in 2026-11 | refuse to track fast | p95 1.463 in async but tracked share 86.5% -> 66% (key) | `PR tracking2` | none |
| `CoarseStart` | default in async mode | box-decimated start of tentative tracks | without it a tentative locked 4.9 px off | `PR tracking2` | none |

## 8. Threading, timing and device wake-up

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Worker spinning (`cpu_spin_us`) | removed in 2026-11 | workers spin on a generation counter | Pi CPU phase of a dense 1280x720 frame 0.64 -> 0.85 / 1.7 / 5.5 ms at 200 / 1000 / 3000 us; desktop 200 us +15 to +60% | `PR worker-pool`, `PR del-experiments` | none |
| Cold small batches inline (`cpu_inline_max`) | removed in 2026-11 | | neutral on the Pi at gaps 0/16/33; tracking medium-4tag key 0.34 -> 0.78 ms (+130%) | `PR worker-pool` | none |
| Pin the calling thread | removed in 2026-11 | | "no better than workers only" | `PR worker-pool` | none |
| Calling-thread CPU tail; run inline | dropped (branches `perf/pf-m5-cpu-inline`) | | the documents say code only on the branch; `APRILTAG_CPU_INLINE` existed again until 2026-11 | P §8av, §8ay | none |
| Big-core pinning | default (new, 2026-11) | pin workers to the highest-capacity cores | Pi many-tags 0.199 -> 0.166 ms (-17%), dense-tags 0.645 -> 0.571 (-11%); four unpinned threads are 20-40% worse than eight; tracking gains at gap 16 "unconfirmed" | `PR worker-pool` | none |
| One shared worker pool | default (new, 2026-11) | `WorkerPool` shared by quad fit, decode and pose | three pools -> one, no measurable gain (cpu phase -1.8 to +3.6%); the serial gap between stages is 0.2 to 4 us, so overlapping the batches was not done | `PR one-pool` | none |
| Keep-alive | default on integrated (6 ms), opt-in on discrete | periodic one-thread dispatches | Pi 33 ms gaps 0.89-1.02 -> 0.66-0.69 ms; on the desktop only recovered 0.05-0.07 of 0.28 ms; fixed a Windows timer bug (15.6 ms tick) with a high-resolution waitable timer | P §8at, §8bc, CL "Fixed" | pre-warm on the desktop |
| Predictive pre-warm | default (new, 2026-11) | burst before the next frame is due when the cadence is regular | desktop 640x480 16 ms gaps 0.833 -> 0.415 ms (back to back 0.214); Pi keyframe 1.11 -> 0.87 ms | P §8bc, CL "Added" | none |
| CPU phase after a gap | open | doubles after a pause (0.042 -> 0.085 ms) | not core frequency, not the pool; spinning recovers it at more CPU than it is worth | `PR camera-rate`, P §8bc | open |
| Orange Pi governors | guidance | `performance` on the A76 clusters | `ondemand` adds 3 to 4 ms per frame at 30 and 60 fps; `schedutil` 11 to 17 ms | P §8i | none |

## 9. Build, dependencies, tests and process

| Idea | Status | What | Key number | Recorded | Superseded by |
| --- | --- | --- | --- | --- | --- |
| Patched libapriltag dependency | removed in 2026-11 | fetched at configure time, four patches | clean configure and build 28.5 -> 18.4 s; installed package and `.deb` hold nothing from upstream; upstream kept only as an optional, unpatched validation dependency (`VKAPRILTAG_UPSTREAM_VALIDATION=OFF` by default) | `PR no-upstream`, CL "Changed (build)" | own decode, own types |
| The `apriltag-decode-no-alloc` patch | removed in 2026-11 | | regenerating the test vectors without it gave a byte-identical file | `PR no-upstream` | none |
| Test net | default (new, 2026-11) | `ctest`: unit groups, golden detections, decode differential, lints, installed-package consumer | 17 tests in about 14 s; goldens reproduce through every `FORCE_NO_*` path | `PR test-net`, ARCHITECTURE.md section 10 | none |
| Options registry | default (new, 2026-11) | one table, one read per object | 84 `getenv` sites -> 1; 82 rows; fixed small-tag rescue being read by three components (a configured value now reaches `TagDecoder`) and two wrong comments | `PR options` | none |
| A/B-only switches of settled choices | removed in 2026-11 | `FIND_MODE`, `MERGE_FLAG_MODE`, `DEC1_ALIAS`, `UF_INIT_RUN`, `THRESH_RUN`, `SORT_CHUNK`, `REDUCE_X4`, `GRAY_MAPPED` | each keeps its device default; the naive find, the sorting network and the plain extents shaders deleted; `FORCE_NO_*` and `PRERECORD` kept | `PR del-switches` | none |
| Dead code and unused types | removed in 2026-11 | `QuadCorners`, the write-only `alive` vector, spinner layout, the `fuse` branches | cppcheck leftovers (false positives) left alone | `PR del-experiments` | none |
| `-ffp-contract=off` for the quad fit and decode | default (GCC and Clang) | | GCC's fusing depends on code layout, so corners would change with layout | `library/CMakeLists.txt`, `PR own-decode` | none |
| Release workflow with glslang 16.2.0 | default (1.10.1) | the `.deb` packages include the dot product shader | Ubuntu 24.04's glslang 15.1 could not compile it: about 30 us slower per frame on the Mali | CL 1.10.1 | none |
| Edge-case audit (1.9.x) | default | large blobs, odd frame sizes, pose input checks, pool, resource release | see P §8aa | P §8aa | none |
| Mesa PanVK instead of libmali | dropped | | rejected at 141 ms per frame **(unverified: from the idea files)** | `survey` | none |
| Persistent record of the research process | process | one branch per idea, one progress log per topic, A/B interleaved (ABBA) | the Pi's run-to-run variance is about +/-5%; the build type must match across both sides | P §8 (old), PERFORMANCE.md section 5 | none |

## 10. Documents (2026-11)

| Decision | What | Record |
| --- | --- | --- |
| One current-truth page | `ARCHITECTURE.md`; the options reference is generated from the registry into `PERFORMANCE.md` section 2 | RR 6.10 |
| Freeze the history | `OPTIMIZATION_NOTES.md`, the two root idea files, the chronological `PERFORMANCE.md` sections and the old `DESIGN.md` and README moved unchanged to `archive/` | RR 5.6 |
| This log | seeded from the mining report; the contradictions it found (about 20) are resolved in the rows above by naming the current default | this file |

## 11. GPU structure and the public API (2026-11)

| Decision | Status | What | Result | Record |
| --- | --- | --- | --- | --- |
| Dispatch-trace goldens | added | `DispatchTrace` records every dispatch (grid or indirect offset, a hash of its push constants), barrier, fill, copy and submit of the first frame; compared per device and profile | 17 profiles on the RX 9060 XT and the Mali-G610; 19 dispatches on the desktop and 16 on the Mali, as counted by hand before | `tests/gpu_trace_test.cpp` |
| `PipelinePlan` | added | every variant choice (shader family, decimation, labelling, merge, reduction, lean tail, sort sizes, fused submits, pre-recording) resolved once by `ResolvePlan` and, after the readback buffers exist, `FinalisePlan`; replaces about 25 members and three derivations | `unit_plan` compares both device classes under 21 variations; traces and detections unchanged | `gpu/PipelinePlan.cpp` |
| Stage table | added | one `StageDesc` per pipeline (shader chosen by the plan, buffers by `Buf` role, push size, workgroup, constants); `CreatePipelines` is a loop; 26 pipeline members became one array | `GpuDetector.cpp` 1,742 to 1,661 lines before the recorder change; traces unchanged | `gpu/StageTable.cpp` |
| One recorder for the group-to-scatter stages | added | `RecordGroupSelectScatter(grid)` serves the fused (indirect), lean (fixed grid) and unfused (host-sized) flows | traces of every profile unchanged | `GpuDetector.cpp` |
| Descriptor sets separate from pipelines | dropped | the only duplicate pipeline pair is `build_indirect_args`, and the sort pipelines differ in their constants | not worth a second binding mechanism | this entry |
| Unfused head shared with `RecordFrameHead` | dropped | with `UF_VERIFY` the head's counting compression would count blob sizes twice (the unfused flow counts in `uf_final`) | kept separate | this entry |
| `MemoryKind::DeviceLocalReadback` | removed | unused since the readbacks became `HostVisibleCached` | | `Buffer.h` |
| One public `Detector` | default (2.0.0) | options struct, `Result`, `Device`, `Error`; composes the stages by `Mode`; the stage classes left the installed headers | the five golden profiles through `vk_eval --facade` equal the stage-composed dumps; `facade_*` covers every mode; ThreadSanitizer on the Pi clean | `Detector.h`, `ARCHITECTURE.md` 9.2 |
| Second pipelined detector on its own lane | fixed | two detectors on one lane raced in `Lane::SubmitPersistentAndWait` and gave wrong results at depth 2 | found by `facade_pipelined` and ThreadSanitizer on the Mali | `Detector.cpp` |
| SWIG bindings | added | `bindings/swig/vkapriltag.i`; Python built and tested on both machines, Java and C# generated | `swig_python` passes on Windows and on the Pi | `bindings/swig/README.md` |
