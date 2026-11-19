# Progress: one worker pool per pipeline (branch research/2026-11-one-pool, base 64e45cd)

## Step 1: measurements (desktop, base build, decimation 2, 12 threads)
Trace (APRILTAG_VK_TRACE), 35 passes x 6 frames per scenario, medians in microseconds. gap = QuadDecode::Decode return to
TagDecoder::Decode start; qjw / tjw = caller's pool.join_wait summed over the quad / decode batches.

| scenario | quad.decode_all | gap | tag.decode_all | quad join_wait | decode join_wait |
|---|---|---|---|---|---|
| one-tag | 24.3 | 0.4 | 76.2 | 2.0 | 54.5 |
| many-tags | 20.3 | 0.4 | 78.2 | 0.1 | 27.6 |
| dense-tags | 39.7 | 4.4 | 199.1 | 4.3 | 26.8 |
| clutter-only | 55.2 | 0.2 | 118.8 | 26.9 | 66.6 |
| jpeg | 162.6 | 0.8 | 211.4 | 35.5 | 43.5 |
| busy-background | 45.8 | 0.5 | 117.4 | 17.5 | 65.1 |

The serial gap is 0.2 to 4 us: nothing to remove, so step 3 (overlap/fuse the batches) is NOT done. The remaining wake/join cost is the
tail wait inside each batch, which a shared pool does not change.

APRILTAG_CPU_THREADS sweep (cpu phase ms, median of 4 interleaved rounds):
| scenario | 1 | 2 | 4 | 6 | 12 |
|---|---|---|---|---|---|
| one-tag | 0.060 | 0.063 | 0.082 | 0.094 | 0.109 |
| many-tags | 0.100 | 0.087 | 0.087 | 0.091 | 0.097 |
| dense-tags | 1.214 | 0.675 | 0.394 | 0.303 | 0.287 |
| clutter-only | 0.351 | 0.235 | 0.181 | 0.171 | 0.197 |
| jpeg | 1.633 | 0.945 | 0.546 | 0.416 | 0.389 |
| busy-background | 0.357 | 0.246 | 0.178 | 0.165 | 0.185 |

On one-tag the parallel wake costs about 50 us over one thread; not addressed here (inline/spin variants were tried and deleted earlier).

## Commands
Scratch scripts (r3.py ABBA runner, an.py trace analyser) in the session scratchpad `one-pool/`. Base exes: ext/avk-compare/build-one-pool-base.

## Step 2: shared pool (commit a9b7cdb)
- `WorkerPool` shared via `std::shared_ptr`: `QuadDecode(config, pool)`, `TagDecoder(..., small_tag_rescue, pool)`, `PoseEstimator(..., tol, pool)`;
  `MakeWorkerPool(config)` builds the pool as QuadDecode did (name "cpu-worker"). Default (null) builds its own as before.
- Callers sharing: TrackingDetector (quad_decode_ + full_decoder_), MultiLevelDetector (all levels), vk_eval, apps main.cpp.
  Not changed: bench_multiqueue (uses different thread counts per stage on purpose), coarse_decoder_ (1 thread), tools using PoseEstimator alone.
- Tracking async: the background thread runs quad_decode_ while the caller tracks with full_decoder_, so EnsureBackground gives quad_decode_
  a pool of its own again (topology as before); bg_decoder_ keeps its own pool. A shared pool would make the tracked solve wait behind the quad fit.
- Contract: ParallelFor from a pool's own task throws std::logic_error (was a deadlock); calls from two threads serialise on call_mutex_
  and are counted (`contended_calls()`). Per-owner scratch is indexed by slot and owned by the stage, so serialised sharing is safe.
  Race-prone paths reasoned about: tracking background decoder (own pools), FramePipeline (GPU worker threads never touch the pools; CPU tail on the caller).
  No TSan on the desktop; run it on the Pi.
- vk_eval now passes `--threads` to QuadDecode too (default 0 unchanged).

## Gate (desktop)
- ctest 10/10 (new group unit_pool), goldens unchanged, options_lint green.
- Corpus d1/d2 vs base: differing lines are 2 per affected file, as in base-vs-base (A vs B: d1 8 lines in 4 files, d2 2 lines); new vs base d1 6 lines/3 files,
  d2 2 lines/1 file; APRILTAG_CPU_THREADS=1: d1 4, d2 0; =4: d1 6, d2 2. Same one-line GPU run-to-run noise (it appears with 1 thread too).
- track_eval full and key: byte-identical (3 runs vs base). async recall is bimodal: enter-3tag 83.5-83.8 base vs 84.0-95.2 new; medium-2tag 83.8-89.5 vs 83.0-95.8;
  noise-2tag 99.5-99.8 vs 98.2-99.5; others equal. Within the bimodal spread (3 runs each), topology in async unchanged.
- ABBA 4 rounds, 6 scenarios, d1/d2, gap 0/16 (`abba_d1.txt`, `abba_d2.txt`): cpu phase within -1.8..+3.6% except gap 0 noise (clutter d1 -9.7%, d2 +3.2%);
  total -8..+5% with GPU column moving the same amount (noise). No measurable gain or loss. Gain is structural: 2 fewer thread sets per pipeline (3 pools -> 1).
## Step 3: not done (serial gap 0.2-4 us).
