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
