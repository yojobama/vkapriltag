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
