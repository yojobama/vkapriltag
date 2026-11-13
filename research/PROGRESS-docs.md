# Progress: documentation (Phase 5, branch research/2026-11-docs, base 9e98e17)

Documentation only; no library code changed. The only tool added is `apriltags_vulkan/tools/check_links.py`.

## Steps
1. Archive: `git mv` of OPTIMIZATION_NOTES.md, the two root idea files and DESIGN.md (as DESIGN-history.md) into
   `apriltags_vulkan/archive/`; old README copied to `archive/README-1.10-history.md`; PERFORMANCE.md split:
   sections 3, 3a to 8be go to `archive/PERFORMANCE-history.md`, sections 1, 2 (generated table byte-identical), a new 3, a new 4
   (per-device guidance) and 5 (measuring, from old section 9) stay.
2. Write ARCHITECTURE.md, DECISIONS.md, DESIGN.md, README.md.
3. Checks: ctest, link checker, 10 spot checks (below).

## Commands
- Build: `cmd //c "$(cygpath -w $S/docs/cfgbuild.bat)"` (cmake -B build-docs, tests on, apps off); tests: `$S/docs/ctest.bat --output-on-failure`.
- Links: `python apriltags_vulkan/tools/check_links.py` (default root: the repository).

## Results
- ctest: 17 of 17 pass on build-docs (the first run of the final tree failed `installed_package_consumer` only because MSVC 14.51
  was uninstalled mid-session and the cached compiler path vanished; a clean rebuild with 14.52 passes).
- Link checker (`tools/check_links.py`): 57 Markdown files, 39 relative links, 0 broken (3 in the archived README fixed).
- Line counts before -> after: PERFORMANCE.md 3424 -> 459 (history 3145 in archive); DESIGN.md 614 -> 96 (old in archive); README.md 570 -> 154;
  OPTIMIZATION_NOTES.md 2122 (moved); the two root idea files 472 and 412 (moved); new ARCHITECTURE.md 475, DECISIONS.md 194.

## Spot checks of ARCHITECTURE.md against code (all confirmed)
1. `kMaxHullAttempts` 6 and `kMaxRescueRefits` 4 (QuadDecode.cpp).
2. `PROFILE_FEATURES` default 31 (RefineEdges.cpp, ProfileFeatures).
3. Tracking defaults keyframe 8, min_margin 10, max_iterations 3 (TrackingDetector.h).
4. `kLargeSortCap` 65536 and 64-byte counter staging (GpuDetector.cpp).
5. Keep-alive 6 ms on integrated, 0 otherwise; pre-warm lead 3000 us (GpuDetector constructor).
6. Predictive pre-warm needs 4 or more intervals (KeepAlive.cpp, NextFrameLocked).
7. Third bit only for families with h >= 9 (TagDecoder.cpp, AllowsThirdBit).
8. `SortById` uses `std::stable_sort` (OwnDecode.cpp).
9. Registry has 82 rows (`vk_eval --list-option-names | wc -l`).
10. Dispatch counts 19 / 16: counted by hand from RecordFrameHead and RecordFusedTail for both device classes; not timed or run.

## Not verified
- Numbers in DECISIONS.md taken from the mining report only are marked "(unverified)"; the lavapipe CI run; gcc builds.
