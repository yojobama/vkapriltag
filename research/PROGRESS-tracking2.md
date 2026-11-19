# Progress: tracking2 (branch research/2026-11-tracking2, from research/improvements-2026-11 aab7041)

## Built
- `TrackingConfig::async` (APRILTAG_VK_TRACKING_ASYNC): a background thread runs GpuDetector + QuadDecode + its own TagDecoder
  (`async_threads`, default 1; APRILTAG_VK_TRACKING_ASYNC_THREADS). The tracked path keeps its own decoder. Options:
  ASYNC_INTERVAL (background every Nth frame), ASYNC_COPY=0 (borrow the caller's buffer; Detect() waits for a background
  pass still reading it), ASYNC_DROP=0 (a failed established track sends the frame to a synchronous full detect instead of
  leaving the output; mode `asyncsync` in track_eval).
- Background results become tentative tracks: each starts from a box-decimated refine (CoarseStart), then the normal full-resolution
  solve on the current frame, so every report is measured on the current frame. No live track: synchronous full detect.
  The background is not started within 2 frames of a synchronous fallback. Near-edge established tracks still force a full detect.
- Without the decimated start a tentative from stale corners locked 4.9 px off (one frame behind) with a high margin.
- `capture` (APRILTAG_VK_TRACKING_CAPTURE, _CAPTURE_SPEED): constant-acceleration / zero-velocity starts, then decimated crop (2..8).
- `max_speed_px` (APRILTAG_VK_TRACKING_MAX_SPEED, default off): refuse to track faster than N px/frame.
- `max_iterations` default 8 -> 3.
- track_eval: `--modes full,key,keycap,async,asyncsync,asynccap`, `--dump-rounds`, `--async-interval`, `--bg-threads`, `--no-copy`,
  `--spin-gap`; rejects frames whose size differs from the first (exit 1). score_seq.py, run_sequences.sh (MODES=), sequence enter-3tag-640.

## Findings
- fast-1tag-1280 error (p95 1.87 vs 1.46 on tracked frames): the refine settles up to 2 px off on blurred tags at 11-26 px/frame and
  the lag grows over consecutive tracked frames (0.4, 0.6, 1.0 px). Tighter convergence (0.002) does not fix it; one pass is far worse.
  Biased results need 6-8 passes, so max_iterations=3 refuses them: key mode p95 1.463 = full, 7.5% of frames tracked (was 23.5%).
  Async still tracks 14.5% and gets p95 1.550; max_speed_px=10 gives 1.463 in async (11% tracked) but drops medium-4tag tracked share
  from 86.5% to 66% (key) and 99% to 81.5% (async), so it stays off.
- Capture: a capture frame costs 0.5-0.7 ms (2x a full detect), tracks 10-20% more fast frames, fast-1tag p95 1.67, and in async
  fast-2tag recall falls to 90%. Verdict: not worth it for fast motion; default off. (The decimated start is used for tentatives regardless.)
- Frame copy: 12 us at 640x480, 43-47 us at 1280x720 (async_copy default on).
- gap 16 ms is not tracking-specific. All paths slow down (4 tags 1280: full 0.50 to 1.25 ms, tracked 0.19 to 0.36). GPU keep-alive
  4 ms: no change. Spinning instead of sleeping: no change (0.32-0.36). One thread: tracked 0.556 (gap 0), 0.655 (spin gap), 0.734
  (sleep gap): a cold core costs about 18%, worker wake the rest. Fewer workers (2/4/8) are no better. No cheap fix; the Pi's loss of
  gain was not reproduced here (key still gains 2.4x in mean at gap 16).
- Background thread count (4 tags 1280, asyncsync, gap 0, 6 rounds; tracked median / p95 ms / CPU ms per frame):
  all cores 0.325 / 0.453 / 1.91; 1 thread 0.229 / 0.336 / 1.21; 2 threads 0.261 / 0.379 / 1.54. Gap 16: all about 0.42 / 0.58.
  Default set to 1.
- Still-image corpus, decimation 2, tracking off: 7348 tags, 7146 found = 97.25%, 3 false; identical to the base dumps.

## Commands
    B=ext/avk-compare/build-tracking2   (configured from ours-tracking; PATH needs $B/_deps/apriltag-build)
    MODES=full,key,async,asyncsync bash tools/napa_corpus/run_sequences.sh $B/track_eval.exe napa-seq OUT --rounds 4 [--gap-ms 16]
    PYTHONPATH=tools/napa_corpus python tools/napa_corpus/score_seq.py napa-seq OUT/full OUT/key OUT/async OUT/asyncsync

## Tables
4 rounds, interleaved, desktop RX 9060 XT; times are Detect() ms. These runs predate async_threads=1 (the background decoder used all
cores), so the 4-tag async rows are about 0.1 ms pessimistic. Delay = frames from a tag's first ground-truth appearance (runs starting
after frame 0) to its first report. Wrong IDs include tags partly outside the frame, which the full-path modes report.

### gap 0
| sequence | mode | mean | median | p95 | max | full% | trk% | recall | wrongID | err med | err p95 | delay mean (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enter-3tag-640 | full | 0.248 | 0.246 | 0.298 | 0.577 | 100.0 | 0.0 | 100.0% | 13 | 0.011 | 0.057 | 0.0 (0) |
| enter-3tag-640 | key | 0.154 | 0.137 | 0.313 | 0.630 | 18.0 | 82.0 | 100.0% | 9 | 0.011 | 0.054 | 0.0 (0) |
| enter-3tag-640 | async | 0.144 | 0.149 | 0.188 | 0.433 | 0.5 | 99.5 | 95.7% | 0 | 0.011 | 0.054 | 6.5 (7) |
| enter-3tag-640 | asyncsync | 0.147 | 0.149 | 0.198 | 0.616 | 3.0 | 97.0 | 98.2% | 3 | 0.011 | 0.054 | 3.5 (7) |
| enterleave-3tag-640 | full | 0.221 | 0.233 | 0.264 | 0.288 | 100.0 | 0.0 | 100.0% | 118 | 0.011 | 0.061 | 0.0 (0) |
| enterleave-3tag-640 | key | 0.212 | 0.232 | 0.266 | 0.306 | 94.5 | 5.5 | 100.0% | 118 | 0.011 | 0.061 | 0.0 (0) |
| enterleave-3tag-640 | async | 0.164 | 0.181 | 0.259 | 0.363 | 64.0 | 36.0 | 100.0% | 60 | 0.010 | 0.059 | 0.0 (0) |
| enterleave-3tag-640 | asyncsync | 0.172 | 0.184 | 0.276 | 0.583 | 64.0 | 36.0 | 100.0% | 60 | 0.010 | 0.059 | 0.0 (0) |
| fast-1tag-1280 | full | 0.281 | 0.275 | 0.322 | 0.432 | 100.0 | 0.0 | 100.0% | 0 | 0.360 | 1.463 | - |
| fast-1tag-1280 | key | 0.306 | 0.278 | 0.508 | 0.795 | 92.5 | 7.5 | 100.0% | 0 | 0.360 | 1.463 | - |
| fast-1tag-1280 | async | 0.454 | 0.489 | 0.537 | 0.662 | 85.5 | 14.5 | 100.0% | 0 | 0.364 | 1.550 | - |
| fast-1tag-1280 | asyncsync | 0.454 | 0.486 | 0.551 | 0.746 | 85.5 | 14.5 | 100.0% | 0 | 0.364 | 1.550 | - |
| fast-2tag-640 | full | 0.256 | 0.248 | 0.290 | 0.581 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| fast-2tag-640 | key | 0.278 | 0.248 | 0.491 | 0.551 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| fast-2tag-640 | async | 0.477 | 0.482 | 0.628 | 0.823 | 98.0 | 2.0 | 99.0% | 0 | 0.545 | 1.814 | - |
| fast-2tag-640 | asyncsync | 0.480 | 0.484 | 0.605 | 0.953 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| medium-2tag-640 | full | 0.248 | 0.241 | 0.291 | 0.553 | 100.0 | 0.0 | 100.0% | 0 | 0.011 | 0.058 | - |
| medium-2tag-640 | key | 0.182 | 0.145 | 0.353 | 0.606 | 14.5 | 85.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| medium-2tag-640 | async | 0.189 | 0.168 | 0.335 | 0.494 | 1.0 | 99.0 | 97.2% | 0 | 0.011 | 0.055 | - |
| medium-2tag-640 | asyncsync | 0.188 | 0.172 | 0.250 | 0.595 | 1.5 | 98.5 | 100.0% | 0 | 0.011 | 0.058 | - |
| medium-4tag-1280 | full | 0.502 | 0.501 | 0.568 | 0.721 | 100.0 | 0.0 | 100.0% | 0 | 0.008 | 0.050 | - |
| medium-4tag-1280 | key | 0.257 | 0.197 | 0.548 | 0.644 | 13.5 | 86.5 | 100.0% | 0 | 0.008 | 0.050 | - |
| medium-4tag-1280 | async | 0.330 | 0.331 | 0.463 | 0.834 | 1.0 | 99.0 | 100.0% | 0 | 0.008 | 0.049 | - |
| medium-4tag-1280 | asyncsync | 0.320 | 0.325 | 0.457 | 0.666 | 1.0 | 99.0 | 100.0% | 0 | 0.008 | 0.049 | - |
| noise-2tag-640 | full | 0.259 | 0.252 | 0.296 | 0.575 | 100.0 | 0.0 | 100.0% | 0 | 0.034 | 0.068 | - |
| noise-2tag-640 | key | 0.201 | 0.167 | 0.373 | 0.618 | 14.5 | 85.5 | 100.0% | 0 | 0.034 | 0.070 | - |
| noise-2tag-640 | async | 0.210 | 0.197 | 0.290 | 0.501 | 1.0 | 99.0 | 99.8% | 0 | 0.033 | 0.070 | - |
| noise-2tag-640 | asyncsync | 0.213 | 0.199 | 0.292 | 0.672 | 1.5 | 98.5 | 100.0% | 0 | 0.033 | 0.070 | - |
| slow-1tag-1280 | full | 0.306 | 0.307 | 0.357 | 0.429 | 100.0 | 0.0 | 100.0% | 0 | 0.008 | 0.043 | - |
| slow-1tag-1280 | key | 0.121 | 0.080 | 0.352 | 0.434 | 12.5 | 87.5 | 100.0% | 0 | 0.008 | 0.046 | - |
| slow-1tag-1280 | async | 0.112 | 0.106 | 0.173 | 0.384 | 0.5 | 99.5 | 100.0% | 0 | 0.008 | 0.049 | - |
| slow-1tag-1280 | asyncsync | 0.109 | 0.103 | 0.172 | 0.389 | 0.5 | 99.5 | 100.0% | 0 | 0.008 | 0.049 | - |
| slow-1tag-640 | full | 0.217 | 0.219 | 0.252 | 0.304 | 100.0 | 0.0 | 100.0% | 0 | 0.011 | 0.054 | - |
| slow-1tag-640 | key | 0.075 | 0.047 | 0.238 | 0.326 | 12.5 | 87.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-1tag-640 | async | 0.062 | 0.054 | 0.103 | 0.279 | 0.5 | 99.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-1tag-640 | asyncsync | 0.063 | 0.055 | 0.104 | 0.258 | 0.5 | 99.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-3tag-640 | full | 0.257 | 0.249 | 0.291 | 0.564 | 100.0 | 0.0 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | key | 0.193 | 0.159 | 0.378 | 0.623 | 12.5 | 87.5 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | async | 0.199 | 0.194 | 0.241 | 0.622 | 0.5 | 99.5 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | asyncsync | 0.193 | 0.189 | 0.233 | 0.336 | 0.5 | 99.5 | 100.0% | 0 | 0.010 | 0.047 | - |
### gap 16 ms
| sequence | mode | mean | median | p95 | max | full% | trk% | recall | wrongID | err med | err p95 | delay mean (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enter-3tag-640 | full | 0.894 | 0.911 | 1.084 | 1.228 | 100.0 | 0.0 | 100.0% | 13 | 0.011 | 0.057 | 0.0 (0) |
| enter-3tag-640 | key | 0.370 | 0.273 | 0.991 | 1.266 | 18.0 | 82.0 | 100.0% | 9 | 0.011 | 0.054 | 0.0 (0) |
| enter-3tag-640 | async | 0.265 | 0.269 | 0.374 | 0.975 | 0.5 | 99.5 | 98.5% | 0 | 0.011 | 0.054 | 2.0 (2) |
| enter-3tag-640 | asyncsync | 0.280 | 0.268 | 0.397 | 1.165 | 3.0 | 97.0 | 99.5% | 3 | 0.011 | 0.054 | 1.0 (2) |
| enterleave-3tag-640 | full | 0.887 | 0.906 | 1.072 | 1.232 | 100.0 | 0.0 | 100.0% | 118 | 0.011 | 0.061 | 0.0 (0) |
| enterleave-3tag-640 | key | 0.845 | 0.903 | 1.070 | 1.264 | 94.5 | 5.5 | 100.0% | 118 | 0.011 | 0.061 | 0.0 (0) |
| enterleave-3tag-640 | async | 0.604 | 0.797 | 1.016 | 1.188 | 64.0 | 36.0 | 100.0% | 60 | 0.010 | 0.059 | 0.0 (0) |
| enterleave-3tag-640 | asyncsync | 0.603 | 0.788 | 1.010 | 1.246 | 64.0 | 36.0 | 100.0% | 60 | 0.010 | 0.059 | 0.0 (0) |
| fast-1tag-1280 | full | 0.981 | 0.973 | 1.171 | 1.283 | 100.0 | 0.0 | 100.0% | 0 | 0.360 | 1.463 | - |
| fast-1tag-1280 | key | 0.976 | 0.977 | 1.293 | 1.709 | 92.5 | 7.5 | 100.0% | 0 | 0.360 | 1.463 | - |
| fast-1tag-1280 | async | 1.133 | 1.246 | 1.456 | 1.695 | 85.5 | 14.5 | 100.0% | 0 | 0.364 | 1.550 | - |
| fast-1tag-1280 | asyncsync | 1.145 | 1.258 | 1.474 | 1.801 | 85.5 | 14.5 | 100.0% | 0 | 0.364 | 1.550 | - |
| fast-2tag-640 | full | 0.940 | 0.934 | 1.098 | 1.246 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| fast-2tag-640 | key | 0.969 | 0.944 | 1.276 | 1.517 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| fast-2tag-640 | async | 1.211 | 1.236 | 1.433 | 1.705 | 98.0 | 2.0 | 99.0% | 0 | 0.545 | 1.814 | - |
| fast-2tag-640 | asyncsync | 1.222 | 1.236 | 1.451 | 1.848 | 100.0 | 0.0 | 100.0% | 0 | 0.537 | 1.747 | - |
| medium-2tag-640 | full | 0.926 | 0.922 | 1.104 | 1.252 | 100.0 | 0.0 | 100.0% | 0 | 0.011 | 0.058 | - |
| medium-2tag-640 | key | 0.384 | 0.281 | 1.014 | 1.362 | 14.5 | 85.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| medium-2tag-640 | async | 0.312 | 0.293 | 0.425 | 1.326 | 1.0 | 99.0 | 99.8% | 0 | 0.011 | 0.056 | - |
| medium-2tag-640 | asyncsync | 0.312 | 0.285 | 0.417 | 1.375 | 1.5 | 98.5 | 100.0% | 0 | 0.011 | 0.058 | - |
| medium-4tag-1280 | full | 1.246 | 1.251 | 1.423 | 1.625 | 100.0 | 0.0 | 100.0% | 0 | 0.008 | 0.050 | - |
| medium-4tag-1280 | key | 0.523 | 0.372 | 1.345 | 1.668 | 13.5 | 86.5 | 100.0% | 0 | 0.008 | 0.050 | - |
| medium-4tag-1280 | async | 0.454 | 0.418 | 0.587 | 1.694 | 1.0 | 99.0 | 100.0% | 0 | 0.008 | 0.049 | - |
| medium-4tag-1280 | asyncsync | 0.452 | 0.417 | 0.577 | 1.568 | 1.0 | 99.0 | 100.0% | 0 | 0.008 | 0.049 | - |
| noise-2tag-640 | full | 0.906 | 0.939 | 1.128 | 1.432 | 100.0 | 0.0 | 100.0% | 0 | 0.034 | 0.068 | - |
| noise-2tag-640 | key | 0.409 | 0.303 | 1.056 | 1.632 | 14.5 | 85.5 | 100.0% | 0 | 0.034 | 0.070 | - |
| noise-2tag-640 | async | 0.345 | 0.325 | 0.458 | 1.487 | 1.0 | 99.0 | 99.8% | 0 | 0.033 | 0.070 | - |
| noise-2tag-640 | asyncsync | 0.360 | 0.323 | 0.491 | 3.200 | 1.5 | 98.5 | 100.0% | 0 | 0.033 | 0.070 | - |
| slow-1tag-1280 | full | 1.016 | 1.018 | 1.220 | 1.345 | 100.0 | 0.0 | 100.0% | 0 | 0.008 | 0.043 | - |
| slow-1tag-1280 | key | 0.268 | 0.143 | 1.097 | 1.373 | 12.5 | 87.5 | 100.0% | 0 | 0.008 | 0.046 | - |
| slow-1tag-1280 | async | 0.225 | 0.209 | 0.314 | 1.274 | 0.5 | 99.5 | 100.0% | 0 | 0.008 | 0.049 | - |
| slow-1tag-1280 | asyncsync | 0.224 | 0.208 | 0.317 | 1.076 | 0.5 | 99.5 | 100.0% | 0 | 0.008 | 0.049 | - |
| slow-1tag-640 | full | 0.885 | 0.891 | 1.063 | 1.318 | 100.0 | 0.0 | 100.0% | 0 | 0.011 | 0.054 | - |
| slow-1tag-640 | key | 0.213 | 0.099 | 0.981 | 1.152 | 12.5 | 87.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-1tag-640 | async | 0.137 | 0.123 | 0.197 | 0.954 | 0.5 | 99.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-1tag-640 | asyncsync | 0.139 | 0.125 | 0.196 | 0.959 | 0.5 | 99.5 | 100.0% | 0 | 0.011 | 0.056 | - |
| slow-3tag-640 | full | 0.951 | 0.957 | 1.119 | 1.276 | 100.0 | 0.0 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | key | 0.397 | 0.311 | 1.023 | 1.188 | 12.5 | 87.5 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | async | 0.330 | 0.318 | 0.409 | 1.067 | 0.5 | 99.5 | 100.0% | 0 | 0.010 | 0.047 | - |
| slow-3tag-640 | asyncsync | 0.328 | 0.319 | 0.415 | 1.117 | 0.5 | 99.5 | 100.0% | 0 | 0.010 | 0.047 | - |
