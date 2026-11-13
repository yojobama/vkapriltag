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
