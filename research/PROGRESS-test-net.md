# Progress: in-repo test safety net (branch research/2026-11-test-net, base b74718b)

No behaviour changed. The only library edits are `library/src/TestHooks.h` and thin `testing::` wrappers at
the end of `TagDecoder.cpp` and `RefineEdges.cpp`; the helpers stay private to their anonymous namespaces.

## What is there
- `tools/vk_eval/` (vk_eval.cpp, eval_common.h, stb_image.h): the harness, built by `tools/CMakeLists.txt`.
- `tools/eval.sh` + `tools/eval_report.py`: run a corpus, score with `napa_corpus/score_corpus.py`, print
  recall / false / median / p95 / time against `tests/eval_baseline.json` (labels d1, d2).
- `tools/napa_corpus/make_subset.py`: renders the committed subset (`tests/corpus`, 10 scenarios x 5 frames,
  640x480 PNG, 6.7 MB, equal to the first frames of the full corpus; gt.json per scenario).
- `tests/profiles.json`, `tests/golden/<profile>/<scenario>.txt`, `tests/golden_test.py`: goldens and the compare.
- `tests/unit_tests.cpp` (+ `test_check.h`): CPU-only unit tests, groups decode, geometry, edges, quadfit.
- ctest: unit_decode, unit_geometry, unit_edges, unit_quadfit, golden_detections (`-DVKAPRILTAG_BUILD_TESTS=ON`, default).
- `.github/workflows/vkapriltag-parity.yml`: two steps after the parity gate.

## Commands (Git Bash; S = scratchpad with vcvars wrappers)
Configure and build (the wrappers in `$S/test-net` load vcvars64):

    cmd //c "$(cygpath -w $S/test-net/cfg.bat)"            # cmake -S <worktree>/apriltags_vulkan -B build-test-net -G Ninja
    cmd //c "$(cygpath -w $S/test-net/build.bat)"          # cmake --build build-test-net
    cmd //c "$(cygpath -w $S/test-net/ctest.bat)" --output-on-failure

Elsewhere: `cmake -S apriltags_vulkan -B build && cmake --build build -j && ctest --test-dir build --output-on-failure`.

Regenerate the goldens (after an intended behaviour change, from a build of that change):

    python apriltags_vulkan/tests/golden_test.py --vk-eval <build>/tools/vk_eval.exe --update
    python apriltags_vulkan/tests/golden_test.py --vk-eval <build>/tools/vk_eval.exe          # compare only

Regenerate the corpus subset: `python tools/napa_corpus/make_subset.py <apriltag-src-dir> tests/corpus`
(the dir holding tag36h11.c and tag16h5.c; the fetched `_deps/apriltag-src` works).

Full-corpus score against the baseline: `bash tools/eval.sh -e <vk_eval> -c <napa-corpus> -- --decimation 2`;
`-u` stores a baseline, `-s` exits non-zero on a regression. The time in the baseline is from the desktop.

## Findings and tolerances
- Three repeated golden runs on the desktop (RX 9060 XT, shared GPU) were byte-identical, for all five profiles.
- Goldens also reproduce exactly (tolerance 0.0005 px, no noise frames) with `FORCE_NO_8BIT`, `FORCE_NO_SUBGROUP`,
  `FORCE_NO_INT64_ATOMIC`, `FORCE_NO_CONDITIONAL_RENDERING` and `SPARSE=0`, i.e. through the other shader paths.
- An unmodified build of b74718b (vk_eval from `ext/avk-compare/ours`) produces dumps byte-identical to the goldens
  for all five profiles, so the wrappers change nothing. A clean rebuild of this branch passes all five ctest tests
  (about 7.4 s in total, the golden test 7 s).
- The test therefore uses: same ID multiset and Hamming per frame, corners within 0.02 px (the brief's sub-0.01 px
  cross-GPU differences with margin), and at most 2 differing frames per profile for the known one-line run-to-run
  noise of a marginal detection (the subset has 50 frames, 72 detections; border-occlusion is the marginal case).
- Lavapipe is not installed here, so the CI run on it is unverified. If it flips more than 2 frames, widen
  `--noise-frames` in `tests/CMakeLists.txt` or regenerate nothing (the goldens are the desktop's).
- Subset score (d2): 72 of 76 tags (border-occlusion 4 of 8, by design), 0 false, median 0.019 px, p95 0.111 px.

## Unit test bounds (measured, then about 1.5x)
- Edge locators (`RefineEdges` kExact / kProfile on a rendered black square, supersampled 8x8, Gaussian blur, start
  about 1 px off): worst corner error 0.02-0.56 px (kExact grows with blur, kProfile 0.02-0.36), mean offset below
  0.15 px, which pins the pixel-edge convention (pixel i covers [i, i+1)).
- Quad fit (`QuadDecode::Decode` on rasterised square rings, dp and peaks): 0.003-0.15 px, bound 0.1-0.3 px;
  peaks does not fit a 24 px square or a 45 degree 90 px diamond, so those are not in the peaks cases.

## Not verified
- Lavapipe (see above), gcc/clang builds of the new tests (only MSVC here), the Pi.
