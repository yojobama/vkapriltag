# Progress: temporal tracking fast path (branch research/2026-11-tracking)

## Design (as built)
- `vkapriltag::TrackingDetector` (library/include/vkapriltag/TrackingDetector.h, src/TrackingDetector.cpp) owns
  GpuDetector + QuadDecode + a full TagDecoder, plus a single-thread TagDecoder for tracked frames.
- Frame with tracks and fewer than K-1 consecutive tracked frames: predicted corners (last corners + last
  frame-to-frame displacement) are fed to TagDecoder::Decode as quads (refine + decode, same code as the
  full path). Accepted per track if: same ID, unique, margin >= max(min_margin, ratio * last margin),
  centre within max_jump_frac * side of the prediction, side ratio within max_scale_change.
- Failed tracks: ROI search (bbox of last + predicted corners, padded), libapriltag's quad detector on the
  cropped image (refine off), corners of the detection with the same ID go through TagDecoder again.
- Anything still missing, a predicted corner near the frame edge, or a keyframe: full GPU detector.
- Detection corners from libapriltag are in reversed order to quad corners (quad p[k] = det p[3-k]).
- Switches: `TrackingConfig::enabled` (default off) / APRILTAG_VK_TRACKING, _KEYFRAME, _MIN_MARGIN, _SEARCH.
- New tags only appear on keyframes (latency up to K-1 frames) and when the full detector runs after a failure.

## Tools
- `tools/track_eval/track_eval.cpp`: runs a list as a sequence, tracking off/on in ABBA rounds, prints SUMMARY lines.
- `tools/napa_corpus/gen_sequences.py`: sequence generator (gen_corpus.py renderer + gt.json format).
- `tools/napa_corpus/run_sequences.sh`: runs all sequences, writes dumps for score_corpus.py.
- Build copy of the harness CMake (outside repo): `ext/avk-compare/ours-tracking/CMakeLists.txt` (adds track_eval
  from the worktree; needs NOMINMAX). Config: scratchpad `tracking/vsconfig.bat` (same as vsconfig.bat, ours-tracking).
- Run needs `export PATH=$B/_deps/apriltag-build:$PATH` (apriltag.dll).

## Log
Sequences: `gen_sequences.py $B/_deps/apriltag-src ext/avk-compare/napa-seq` (9 sequences x 200 frames; slow 2 px/frame peak,
medium 6-10, fast 30 with 5-sample motion blur, enter/leave, noise sigma 8). Run:
`bash run_sequences.sh $B/track_eval.exe napa-seq $OUT --rounds 8 [--gap-ms 16]`; score `score_corpus.py napa-seq $OUT/full|track`.
The "false" column of enterleave counts detections of tags whose black square is partly outside the frame (GT omits
them); identical for full and tracking.

1. First cut (predict + one refine/decode): tracked frames were accepted with up to 5 px corner error on motion-blurred
   tags: the refine creeps toward the edge, so a single pass from a bad prediction is not converged.
2. Fix: feed the refined corners back until the shift is below converge_px (0.05, up to 8 decodes), and give up at once
   if the first pass moves a corner more than 3 px. 0.25 px still left 2.5 px mean error on fast-1tag-1280; 0.05 px matches
   full. Failed attempts back off (skip 1, 2, 4.. frames) so fast motion does not pay for repeated failures.
3. ROI search (libapriltag on a decimated crop): costs 1.3-3.6 ms (decimated) to 6-90 ms (undecimated) per use, more than the
   full GPU path (0.3-0.7 ms), and rescued few frames. Left in, default off (TrackingConfig::roi_search / APRILTAG_VK_TRACKING_SEARCH=1).

## Results (desktop, no gap, 8 ABBA rounds; machine shared so absolute times are noisy; K = 8)
Accuracy (tracked dump vs full dump vs GT; recall 100% and 0 wrong IDs in every sequence for both):
median/p95/max corner error px, full | tracking:
- slow-1tag-640 0.011/0.054/0.118 | 0.011/0.056/0.120; slow-3tag-640 0.010/0.047/0.120 | 0.010/0.047/0.123
- slow-1tag-1280 0.008/0.043/0.107 | 0.008/0.046/0.107; medium-2tag-640 0.011/0.058/0.126 | 0.011/0.056/0.126
- medium-4tag-1280 0.008/0.050/0.129 | 0.008/0.050/0.123; noise-2tag-640 0.034/0.068/0.168 | 0.034/0.070/0.169
- fast-1tag-1280 0.36/1.46/2.42 | 0.41/1.87/2.45 (23% of frames tracked); fast-2tag-640 identical (0% tracked)
- enterleave-3tag-640 identical (5.5% tracked)
Frame time median ms (full | tracking) and share of frames by path (tracked/full):
slow-1tag-640 0.54 | 0.077 (87.5/12.5); slow-1tag-1280 0.44 | 0.127 (87.5/12.5); slow-3tag-640 0.42 | 0.31 (87.5/12.5);
medium-2tag-640 0.37 | 0.25 (86.5/13.5); medium-4tag-1280 0.75 | 0.40 (86.5/13.5); noise-2tag-640 0.38 | 0.30 (86.5/13.5);
fast-1tag-1280 0.32 | 0.33 (23.5/76.5); fast-2tag-640 0.36 | 0.39 (0/100); enterleave-3tag-640 0.30 | 0.30 (5.5/94.5).
Tracked-frame medians (ms): 1 tag 640: 0.07; 1 tag 1280: 0.12; 3 tags 640: 0.29; 4 tags 1280: 0.37 (refine + 2 decodes per tag).
With a 16 ms gap between frames: slow-1tag-640 0.45 | 0.14; slow-1tag-1280 0.98 | 0.26; slow-3tag-640 0.57 | 0.30;
medium-4tag-1280 1.09 | 0.50; noise 1.19 | 0.46 (desktop GPU wakes fast; the CPU path also slows a little when idle).
