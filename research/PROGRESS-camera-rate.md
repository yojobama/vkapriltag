# Camera-rate timing (branch research/2026-11-camera-rate)

Harness: copy of ext/avk-compare/ours at ext/avk-compare/ours-camrate (vk_eval with VK_EVAL_GAP_MODE=sleep|spin|hires,
VK_EVAL_SPIN_TAIL_MS, VK_EVAL_HELPER=spin, and a MEAN line with upload/submit_wait/stage sums), built in
ext/avk-compare/build-camera-rate; track_eval in build-camera-rate-t. Driver: scratchpad/camera-rate/r.py
(`REPS=3 python r.py "label|ENV=a,ENV2=b" ...`, interleaved, median of reps; prints gpu cpu total per list).

## Step 1 findings (desktop, RX 9060 XT, 640x480 one tag | 1280x720 many tags, decimation 2, mean ms)

| | gpu / cpu / total 640 | gpu / cpu / total 1280 |
|---|---|---|
| back to back | 0.153 0.042 0.195 | 0.234 0.091 0.324 |
| sleep 16 | 0.748 0.085 0.833 | 0.901 0.164 1.065 |
| sleep 33 | 0.766 0.092 0.858 | 0.922 0.172 1.094 |

- Split of the GPU phase (640): upload 0.027 -> 0.104, submit_wait 0.125 -> 0.642. With APRILTAG_VK_TIMESTAMPS=1 the
  summed GPU stage time is 0.067 -> 0.115 and the stage gaps 0.006 -> 0.011, so the dispatches run only ~0.05 ms
  slower; about 0.5 ms is latency between submit and the GPU starting (wake-up), plus 0.07 ms on the BAR upload memcpy.
- Sleeping vs spinning the calling thread: NOT reproduced. Spin gap 16 ms gives gpu 0.676 (sleep 0.75), track_eval
  full 3tag-640: b2b 0.251, sleep 16 0.94, spin 16 0.87, spin 4 0.515. The penalty follows idle time of the GPU
  (spin 1 ms 0.26 gpu, 4 ms 0.34, 16 ms 0.68), not what the calling thread does. Helper thread spinning in the
  process does nothing. timeBeginPeriod(1) alone does nothing (hires16 0.69).
- Keep-alive: it does work on the desktop, but only when the Windows timer is 1 ms. APRILTAG_VK_KEEPALIVE_MS=1/2/4
  with a hires gap: gpu 0.31/0.28/0.29 (640), 0.41/0.39/0.41 (1280) against 0.69 / 0.83 without; with the default
  15.6 ms timer tick (sleep gap, no timeBeginPeriod) it recovers nothing: the wait_for rounds up to a tick.
- CPU phase after a gap: 0.042 -> 0.085. A 2 ms busy spin of the calling thread right before the frame only brings
  it to 0.060; spinning for the whole gap gives 0.055. Not core frequency; caches/predictors after the core idles.

## Step 1, Orange Pi (all runs through pi_clean.sh: service stopped, governors performance)

- pool_wake (tool in ext/avk-compare/camrate-tools, built on the Pi with g++ against WorkerPool.cpp): 8 items of 30 us,
  threads 4: first item of a worker starts 8-17 us after the call when the pool was used a moment ago, 77-85 us on the
  little cores (cpus 1-3) after a 16 ms sleep, 9 us on the big cores; threads 8, 16 ms: batch takes 170 us against 30
  ideal, 3 of the little cores never get an item. cpus of slot 0 sometimes migrate to a little core.
- Not reproduced: tracked frames at camera rate. track_eval --modes key, slow-3tag-640: tracked-frame median
  0.257 (gap 0), 0.258 (gap 16), 0.258 (spin 16), also 0.266 pinned to 4-7. medium-4tag-1280 tracked 0.336 / 0.338.
  The 0.776 in the task brief is not seen with governors pinned (it would need the board's ondemand governor).
- Full detector, 8 threads: slow-3tag-640 0.796 (gap 0), 0.829 (16), 0.845 (33) = +4 to +6 %; medium-4tag-1280
  1.88 / 1.91 / 1.90 = +1.5 %. Thread counts 1/2/4/8 at gap 16 vs 0: within +-5 %, no pool wake-up effect worth fixing
  (the 4-thread run is slow at gap 0 as well: placement on little cores).
- What does go wrong on the Pi: keyframes in tracking mode. Tracked frames never call GpuDetector::Detect, so the keep-alive
  stops after 100 ms and the next keyframe finds a cold GPU: keyframe median 0.76 (gap 0) -> 1.15-1.22 ms (gap 16), 1.05-1.10 (33).

## Fixes built (commit 5dea039, f201d0c)

KeepAlive rewritten: Windows high-resolution waitable timer (fixes the 15.6 ms rounding that made the periodic keep-alive
useless on the desktop), predictive burst (`prewarm_lead_us`, default 3000 us, `prewarm_burst` 2, `prewarm_gap_us` 1000;
envs APRILTAG_VK_PREWARM_LEAD_US / _BURST / _GAP_US), `GpuDetector::Prewarm()` for the application,
`NoteFrame()` / `ExpectGpuFrame()` used by TrackingDetector so a burst fires before each keyframe.
The period is the median of the last 6 inter-arrival times, used only if at least 4 are seen and all are within 25 % of the
median and the period is 2..200 ms; back to back frames therefore never arm it and cost one uncontended lock per Detect.

Desktop sweeps (640 one-tag, gap 16, gpu ms): keep-alive 1/2/4 ms 0.29/0.28/0.31 (was 0.75 with the old timer);
lead x burst x gap: 400x2x100 0.66, 600x4x100 0.54, 800x6x100 0.47, 1000x8x100 0.41, 1200x3x300 0.36, 1500x3x300 0.30,
2000x3x500 0.32, 2000x2x800 0.31, 3000x2x1000 0.30, 2000x1 0.34; heavier keep-alive dispatches (64 to 16384 groups) are no better
(0.30 to 0.32). The GPU needs more than 1 ms after the first submission to be awake.
Pi (key mode, gap 16, keyframe median ms; back to back 0.76): default 1.22, ka0 1.15, lead 1000 0.82, 2000 0.91, 3000 0.83,
5000 0.86, 8000 0.86, burst 3 0.82, burst 1 0.84 (about 25 samples each, differences below 0.08 are noise).
