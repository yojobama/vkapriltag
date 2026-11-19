# Progress: retire refine methods kUpstream, kUltraFast, kFast (branch research/2026-11-del-refine, base e9cd520)

Pure deletion; `kExact` and `kProfile` (default) stay.

## Removed
- `RefineEdgesMethod::kUpstream`, `kFast`, `kUltraFast`; `RefineEdgesFast` and its constants; the hand-declared
  `refine_edges` prototype in `RefineEdges.cpp`; the `ultrafast` decode gate in `TagDecoder.cpp`.
- The first hunk of `cmake/patches/apriltag-expose-decode-steps.patch` (made `refine_edges` non-static);
  the later hunk offsets were adjusted. Upstream's patched `quad_decode_task` still calls its static `refine_edges`.
- `APRILTAG_VK_REFINE=upstream|fast|ultrafast`: the value is now unrecognised, so `ResolveRefineEdgesMethod`
  returns the constructor's method (`profile` for the library and tools; `exact` for `bench_multiqueue`).
- Docs: README (TagDecoder paragraph, margin-gate sentence), CHANGELOG (Removed entry), PERFORMANCE.md section 2 table.
  History sections and measurement tables that name `ultrafast` are untouched.
- The `RefineEdgesMethod` constructor argument stays: it still selects `exact` or `profile`.
- Unit tests only used kExact and kProfile: no change.

## Gate (desktop, Windows, RX 9060 XT)
- Build dir `ext/avk-compare/build-del-refine` (patch applied on a fresh fetch, so the edited patch applies cleanly).
- `ctest`: 5 of 5 pass, goldens unchanged.
- Corpus d2: all 26 dumps byte-identical to a build of e9cd520 (`build-del-refine-base`).
- Corpus d1: 22 of 26 byte-identical; dark-background, family16h5, motion-blur, range differ in one marginal
  detection line each (sub-0.003 px). Base against base differs the same way (6 files differ in a repeat run).
- Scores d1 and d2 unchanged (found 7129 and 7146 of 7348; false 0 and 1) by `cmp.py`.
- `APRILTAG_VK_REFINE=fast` output equals the default output (blur, d2).

## Commands
    S=<scratchpad>; cfg/build/ctest bats in $S/del-refine; run_corpus.sh <vk_eval> <corpus> <out> --decimation N
    python $S/cmp.py <corpus> score_corpus.py <base_dir> <dir>
