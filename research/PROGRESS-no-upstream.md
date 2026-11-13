# Progress: no upstream (Phase 4b stage S3, branch research/2026-11-no-upstream, base c94dc62)

Goal of S3: build, test and package with no upstream fetch and no patches by default. Desktop only (Windows, MSVC).

## Sub-steps (one commit each; every commit builds and passes ctest)
1. Differential test from recorded vectors: `tools/upstream/gen_decode_vectors.cpp` runs the PATCHED upstream over a
   deterministic quad set and writes `tests/own_decode_vectors.bin` (format, reader and writer in
   `tests/DecodeVectors.h`); `tests/own_decode_test.cpp` now needs no upstream and compares the library's decode with the
   file, bit for bit. ctest names unchanged (`own_decode_random`, `own_decode_reconcile`, `own_decode_corpus`,
   `families_match_upstream`, `own_decode_comparison_can_fail`).
2. Fetch and patches leave the default build: `tools/upstream/CMakeLists.txt` (added only under
   `VKAPRILTAG_UPSTREAM_VALIDATION`, now default OFF) fetches stock v3.4.5, excluded from `all` and from install;
   `VKAPRILTAG_REGEN_VECTORS=ON` also applies `tools/upstream/patches/apriltag-expose-*.patch` and builds the generator;
   the mingw patch is applied only on MinGW; `apriltag-decode-no-alloc.patch` deleted; project language is C++ only.
3. `installed_package_consumer` ctest (`tests/consumer_test.py`, `tests/consumer/`), `deps_lint` extended, parity and
   release workflows updated, upstream-linked tools no longer installed.
4. NOTICE (upstream BSD-2-Clause text and the derived-file list), README, DESIGN, CHANGELOG, PERFORMANCE.md pointers.

## Vector set (`tests/own_decode_vectors.bin`, 4 837 310 bytes, generated on Windows/MSVC)
- 8 families (hash of the code and cell tables plus every field), 120 rigs (60 random-quad rigs: each family x 4
  variants {normal border 2 corrected bits, reversed 1, normal 0, reversed 3; capped by table size as before} x one
  family or beside tag16h5; 60 corpus rigs), 781 images (crops zero-padded to 256x256, constants, mt19937_64 noise by seed,
  corpus PNGs by path).
- 3120 random cases: per rig 4 rendered scenes (s = 0..3: with and without jitter, blur, noise, shading) each with
  exact, 4 perturbed (0.2, 0.6, 1.5, 3 px), rolled, other-polarity and random quads and one noise-image case, plus 16
  degenerate/extreme quads on a blank frame; 554 decoded, 360 degenerate homographies, polarity rejections counted.
- 2974 corpus cases: every attempt recorded earlier from tests/corpus (`tests/own_decode_quads.txt`, kept as the
  generator's input), 927 with a detection.
- 2500 reconcile sets (2 to 15 detections; overlaps, containment, identical shapes, quantised corners; corners stored as
  floats), 6582 detections removed; stored as the indices that survive and their order after the id sort.
- Totals: 6094 cases, 1481 with a detection, 235 with corrected bits, 6 with two families.
- Previously (live upstream): 22560 random quads, 30000 reconcile sets. The subsample is deterministic (same generator
  seeds, first scenes per rig); the equality is still bitwise on every compared field (ids, hamming, margin bits,
  homographies, centre, corners, raw sample code and margin, reconcile result and order).
- Self-test: `OWN_DECODE_TEST_SABOTAGE=1` (own sharpening 0.3) gives 105 mismatches on tag16h5, so
  `own_decode_comparison_can_fail` fails as intended.
- Regenerating without `apriltag-decode-no-alloc.patch` produced a byte-identical file (`cmp`), so that patch was dropped.

## Regenerate
    cmake -S apriltags_vulkan -B build-regen -DVKAPRILTAG_UPSTREAM_VALIDATION=ON -DVKAPRILTAG_REGEN_VECTORS=ON
    cmake --build build-regen --target gen_decode_vectors
    build-regen/tools/gen_decode_vectors apriltags_vulkan/tests/corpus apriltags_vulkan/tests/own_decode_quads.txt \
        apriltags_vulkan/tests/own_decode_vectors.bin 4 2500
(Use a separate build dir for REGEN: switching the option in a configured dir does not un-patch the fetched sources.)

## Gate results (desktop, RX 9060 XT)
- Default (`build-no-upstream`): configured with `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` in an empty directory: no `_deps`
  directory, no C compiler detection, and CMake reports that variable as unused (FetchContent is never called).
  ctest 17 of 17 (unit x8, golden, options_lint, deps_lint, the five differential tests, installed_package_consumer).
- `-DVKAPRILTAG_UPSTREAM_VALIDATION=ON` (`build-no-upstream-val`, stock upstream): all validation tools build; ctest 17
  of 17. `apriltag_vulkan_validate --data` on one-tag frame 0 and many-tags frame 0: "MATCH: decoded tag ID set agrees
  with the official libapriltag detector", corner RMS 0.012 px and 0.044 px. `apriltag_pose_validate`: 420 synthetic
  cases and 8 degenerate, worst rotation 5.2e-7 deg, 0 branch disagreements. `apriltag_synth_ground_truth`: 645 of 754
  cases valid, ALL CHECKS PASS.
- `-DVKAPRILTAG_UPSTREAM_VALIDATION=ON -DVKAPRILTAG_REGEN_VECTORS=ON` (`build-no-upstream-regen`): patched, ctest 17 of 17.
- Dependencies: `dumpbin /dependents vk_eval.exe` lists vulkan-1.dll and the CRT only; the c94dc62 build lists apriltag.dll.
- Installed package (temporary prefix): 80 files; no upstream library, header or CMake package; the installed CMake files
  name no apriltag package; `find_package(vkapriltag)` consumer builds and prints `id 316` for corpus one-tag frame 0.
  The same test passes in the validation-ON build (upstream is excluded from install).
- Corpus, 26 scenarios, default settings, `tools/napa_corpus/run_corpus.sh`, base = build of c94dc62 (twice), new (twice):
  d1: base vs base 5 frames differ (0.0022 px max), base vs new 5 and 4 frames (0.0017 and 0.0012 px max); d2: base vs base
  2 frames (0.0027 px), base vs new 1 and 0 frames (0.0027 px). No frame differs in ids or Hamming distance; all
  differences are the known run-to-run corner noise of the GPU front end. Goldens pass (also in all three builds).
- Build time, clean configure + build of everything with the default options, same machine, two runs each:
  c94dc62 28.5 and 29.1 s (configure 9.5 to 10.1 s, which clones upstream; build 19.0 s, which compiles upstream and the
  OpenCV tools); this branch 18.4 and 18.9 s (configure 3.9 s; build 14.5 to 15.1 s).

## Not verifiable on Windows
- GCC/Linux: the vectors were recorded with MSVC; the Pi/Linux ctest must show 0 mismatches (libm `sin`/`cos`/`sqrtf` and
  FMA contraction could differ in the last bit; OwnDecode.cpp keeps `-ffp-contract=off`). If it does not, regenerate the
  vectors on Linux or find the expression.
- The CMake 3.16 to 3.27 branch of `tools/upstream/CMakeLists.txt` (FetchContent_Populate) was not run (CMake 4.2.3 here).
- `ldd` path of consumer_test.py, `dpkg-deb` checks in release.yml, and the parity workflow on lavapipe (CI).
- MinGW patch path; `-DVKAPRILTAG_REGEN_VECTORS=ON` on Linux.

## Follow-up: reconcile on the Pi
`own_decode_reconcile` failed on aarch64/glibc (2082 of 2500 sets) while all 6094 decode cases passed: the recorded
"sorted" order came from the MSVC `qsort`, which orders equal ids differently from glibc's. `SortById` is now a
`std::stable_sort` on id (equal ids keep reconcile's surviving order) and the test derives the expected order from the
recorded survivors with the same stable rule (the recorded qsort order is no longer compared; survivors, their order and
each detection's fields still are). On the desktop: ctest 17 of 17, 0 reconcile mismatches, corpus d1/d2 vs base only the
usual noise frames (no id or Hamming change; 52 of 8406 dump frames hold equal ids), goldens pass. A mismatch now prints
the stage (survivor count, survivors, id order) and the ids seen. The decode comparisons use IEEE double/float arithmetic
only through the code under test and compare recorded bits; they passed on the Pi, so libm and FMA did not matter there.
