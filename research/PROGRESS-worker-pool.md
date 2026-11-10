# Worker pool switches (branch research/2026-11-worker-pool, based on research/improvements-2026-11 078b649)

## Built (commit 1): `WorkerPoolOptions` in common/WorkerPool.h, all default off

| Switch | DetectorConfig | Env |
|---|---|---|
| worker spinning (us) | `cpu_spin_us` | `APRILTAG_CPU_SPIN_US` |
| cold small batch inline: max items / idle us | `cpu_inline_max`, `cpu_inline_idle_us` (1000) | `APRILTAG_CPU_INLINE`, `APRILTAG_CPU_INLINE_IDLE_US` |
| big cores only (Linux, asymmetric capacity) | `cpu_big_cores` | `APRILTAG_CPU_BIG_CORES` |
| pin calling thread too | `cpu_pin_caller` | `APRILTAG_CPU_PIN_CALLER` |

- Spinning: after a batch or `WorkerPool::Warm()` a worker spins (pause/isb) up to `spin_us` on an atomic generation counter
  before blocking. `KeepAlive::Options::on_warm` calls `WorkerPool::WarmAll()` (registry of live pools) whenever a burst starts
  (predictive burst, `GpuDetector::Prewarm()`, periodic keep-alive); `GpuDetector::NoteFrame(true)` (tracked frames in
  `TrackingDetector`) arms a CPU-only warm `lead` before the next frame.
- Inline: `ParallelFor` runs a batch of at most `inline_max` items on the caller when the previous batch ended more than
  `inline_idle_us` ago (batches of 1 were already inline).
- Big cores: workers pin to the cpus with the highest `cpu_capacity` when capacities differ; the default thread count becomes
  their number; the background tracking decoder (`TagDecoder(..., background_thread=true)`) never pins; the keep-alive thread is not a pool thread.
- The caller never waits for workers to wake: it drains the batch itself right after notifying (this was already so).

## Harness
- Desktop: ext/avk-compare/build-worker-pool (vk_eval from ours-camrate), build-worker-pool-t (track_eval). Runner
  scratchpad/worker-pool/r2.py (same script runs on the Pi; process CPU measured with GetProcessTimes / getrusage on the child;
  CPU cost = median process CPU of config minus the first config, per frame, so the init cost cancels).
  Pi: ~/w/wpool (src, build2 = vk_eval, tbuild = track_eval), ~/w/wp (lists, r2.py, piA_*/piB_* scripts).

## Results (all numbers: medians of 3 interleaved rounds; Pi runs through pi_clean.sh)
Raw outputs: scratchpad/worker-pool/{deskA,deskB,deskT}.txt (desktop), piA_*, piB_*, piC_*, piD_*, piE, piT_* (Pi). Runner r2.py
(vk_eval matrix) and tw.py (track_eval matrix). "extra CPU" = process CPU ms per frame minus the first (off) config.

### 1. Worker spinning: NO (default off, not recommended)
- Pi, `pause`/`isb` spin, window 200 / 1000 / 3000 us: CPU phase of a 1280x720 dense frame 0.64 -> 0.85 / 1.7 / 5.5 ms
  at gap 0 already; extra process CPU per frame +2.9 / +14.6 / +46 ms (gap 0), +6 / +28 / +75 ms (gap 16). Tracking
  slow-3tag-640 key frames at gap 0 0.26 -> 0.60 ms with spin 200. Making spinners read-only on their own cache line and taking a batch
  without the mutex changed nothing; `yield` in the spin loop helps only a little (3000 us yield: CPU phase 0.72 vs 0.65 ms off).
  Spinners keep every core busy so the caller, woken after the GPU wait, waits for a core. Nothing was faster than off on the Pi.
- Desktop (12 threads): 50 us 0 to +10 %, 200 us +15..+60 % on many-candidate lists, 1000 us +90..+380 %, 3000 us worse; yield-spin
  1000 us about neutral (-1.2 to +4 %) for +3..+20 ms extra CPU per frame (gap 16: +13 to +43 ms); tracking at gap 16: yield-spin 1000 us
  -3..-7 % on key mode for +23 ms CPU per frame. Pre-warm hooks (KeepAlive on_warm, NoteFrame(true)) work but only make
  spinning cost CPU.
- Left in the code (off): `cpu_spin_us`, env `APRILTAG_CPU_SPIN_US`, `APRILTAG_CPU_SPIN_YIELD`, `APRILTAG_CPU_SPIN_WORKERS`.

### 2. Big cores: YES on Linux asymmetric CPUs (default ON; no effect elsewhere)
Pi cpu phase ms (gpu column is GPU noise, +-8 %): one-tag 640 0.097 = 0.097; many-tags 1280 0.199 -> 0.166 at gap 16 (-17 %);
dense-tags 1280 0.645 -> 0.571 (-11 %); jpeg 1.26 -> 1.21 (-4 %); same at gaps 0 / 16 / 33; process CPU -1.2 ms (dense), -2.5 ms (jpeg) per frame.
4 threads unpinned (`APRILTAG_CPU_THREADS=4`) is 20 to 40 % WORSE than the 8 default: the pinning matters, not only the count.
Tracking (full / key / async medians, ms):
| sequence, gap | full off -> big | key off -> big | async off -> big |
|---|---|---|---|
| slow-3tag-640, 0 | 0.799 -> 0.728 | 0.260 -> 0.255 | 0.263 -> 0.256 |
| slow-3tag-640, 16 | 0.838 -> 0.722 | 0.525 -> 0.257 | 0.565 -> 0.276 |
| medium-4tag-1280, 0 | 1.908 -> 1.803 | 0.343 -> 0.334 | 0.768 -> 0.537 |
| medium-4tag-1280, 16 | 1.905 -> 1.792 | 0.339 -> 0.336 | 0.394 -> 0.390 |
Pinning the calling thread as well (`APRILTAG_CPU_PIN_CALLER=1`) is no better than workers only; left off.
Default-vs-`APRILTAG_CPU_BIG_CORES=0` check after switching the default on: see piE.txt below.

### 3. Small batches inline when cold: NO (default off)
- Pi vk_eval: neutral within noise at gaps 0/16/33 (cpu phase 0.64 vs 0.64 dense, 0.20 vs 0.20-0.27 many-tags at 4 or 8 items: +30 % CPU phase for
  inline 4/8 at gap 16/33). Tracking: inline 4 at gap 16: slow-3tag key 0.525 -> 0.463 (-12 %), async -15 %, but full +13 %; medium-4tag key 0.34 -> 0.78 (+130 %).
- Desktop: g1280 gap 16 +5..+9 %, d4 -10 % at gap 33 but +6 % at gap 16 (noise level), tracking gap 16 +21..+60 % (the calling thread is the slow one after a pause).
- The caller already drains the batch itself right after notifying, so the "start immediately while workers wake" variant is the existing behaviour.
- Left in code (off): `cpu_inline_max`, `cpu_inline_idle_us`, env `APRILTAG_CPU_INLINE`, `APRILTAG_CPU_INLINE_IDLE_US`.

### Correctness
- Pi: vk_eval dump of 76 mixed corpus frames (decimation 2) byte-identical for off / spin 200 / inline 8 idle 0 / big+pin / spin+inline+big;
  track_eval full, key, async dumps identical apart from the `# path ... ms` timing comment lines.
- Desktop: dense-tags, jpeg, many-tags, small-tags, blur dumps identical, off vs spin 300 + inline 8 + big + pin.
- ThreadSanitizer on the Pi (trees ~/w/wpool-tsan, wpool-tsan-t; script pi_tr2.sh): vk_eval gaps 16 / 5 / 8 and track_eval full,key,async, default and five switch combinations: no reports.

### Desktop CPU-phase doubling after a pause (640x480 0.042 -> 0.085, 1280x720 many tags 0.09 -> 0.17 ms)
`APRILTAG_VK_TRACE` spans, 1280x720 many-tags, `--threads 1`, mean per call, gap 0 -> gap 16:
quad.decode_all 21 -> 31 us, quad.fit_blob 11 -> 14, tag.decode_all 90 -> 154, tag.refine 52 -> 86, tag.quad 57 -> 96, tag.decode 3.1 -> 6.4;
readback_copy 0.1 -> 0.1 (not the cause). GPU stages slow down the same way (threshold 18 -> 47, labelling 35 -> 47, sort 21 -> 26).
So the slowdown is spread over every part, strongest in the ones that read the image and per-quad scratch memory (refine, decode: x1.7-2);
no single lock or first-touch. Together with the earlier spin results (2 ms pre-spin recovers 0.085 -> 0.060, spinning the whole gap 0.055 vs 0.042
back to back) it fits the core leaving a deep idle state (caches flushed / clock ramp) with a smaller cold-cache part. No fix found; spin-waiting
(above) is the only thing that recovers it and costs more CPU than it is worth.

### Default-on check (Pi, after the default change; `default` vs `APRILTAG_CPU_BIG_CORES=0`, 3 rounds)
dense-tags cpu phase 0.572 vs 0.641 (gap 0), 0.573 vs 0.645 (gap 16); many-tags 0.164 vs 0.195, 0.165 vs 0.200; slow-3tag-640 tracking at gap 16:
full 0.737 vs 0.790, key 0.258 vs 0.261, async 0.276 vs 0.280. CAUTION: the first tracking table above shows key 0.525 -> 0.257 at gap 16 for
"off"; that off value (0.525) did not reproduce here (off 0.261), so treat the large key/async gains at gap 16 (and medium-4tag async gap 0
0.768 -> 0.537) as unconfirmed, probably run-to-run placement luck of the unpinned pool; the reproducible gain is the CPU phase -11 to -17 % on
many-candidate frames and about -7 % on full tracking frames.
Resume: branch research/2026-11-worker-pool; Pi trees ~/w/wpool, ~/w/wpool-tsan(-t); scripts in scratchpad/worker-pool.
Left: spin only with a smarter scheme (e.g. big cores only, or lower thread count); retune default thread count with big_cores (4 now); pin_caller untested on tracking background thread paths.
