# Progress: delete settled A/B switches (branch research/2026-11-del-switches, base b63bc57)

Pure deletion; every device class keeps its previous default.

## Removed (env override and dead alternative)
- `APRILTAG_VK_FIND_MODE`: `findNaive` and `kFindMode` gone from `uf_merge_body.glsl`; path splitting is the only find. Spec constants renumbered (flag mode 3, sparse 4).
- `APRILTAG_VK_MERGE_FLAG_MODE`: env gone; device default kept (1 on unified memory, else 0), still a spec constant.
- `APRILTAG_VK_DEC1_ALIAS`, `UF_INIT_RUN`, `THRESH_RUN`: env gone; `uf_init_run_`/`threshold_run_` are `static constexpr` 4.
- `APRILTAG_VK_SORT_CHUNK`: env, `sort_chunk_` and the Batcher network in `sort_points_local_body.glsl` gone; `kSortChunk` = 16, `s_merge` always allocated.
- `APRILTAG_VK_REDUCE_X4`: env gone; `reduce_x4_ = !subgroup`. Shaders `reduce_extents_hash.comp`, `reduce_extents_hash_atomic64.comp`, `reduce_extents_hash_body.glsl` deleted (CMake list updated; embedded shaders are generated from it).
- `APRILTAG_VK_GRAY_MAPPED`: env gone; the gray buffer is always requested as DeviceLocalMapped (staging fallback stays when it is not host visible).

## Kept
- `APRILTAG_VK_PRERECORD`: PERFORMANCE.md (sections on DISPATCH_TIMES and RenderDoc capture) documents `PRERECORD=0` as the way to record every frame for per-dispatch timing and capture, so it is a debugging switch, not only an A/B one. `prerecord_` logic unchanged.

## Changed shaders / pipelines (for the Pi check)
- `uf_merge.comp`, `uf_merge_u8.comp`, `uf_merge_x4_u8.comp` (via `uf_merge_body.glsl`): spec constants now {flag mode, sparse}.
- `sort_points_local*.comp`, `sort_points_large*.comp` (via `sort_points_local_body.glsl`): rank-merge only.
- Deleted: `reduce_extents_hash.comp`, `reduce_extents_hash_atomic64.comp`; the `_x4` pair and `_subgroup` are unchanged apart from comments.

## Gate (desktop, RX 9060 XT)
- ctest 5/5 pass, goldens unchanged.
- Golden compare passes (all 5 profiles) with each of FORCE_NO_8BIT, FORCE_NO_SUBGROUP, FORCE_NO_INT64_ATOMIC, NO_SUBGROUP+NO_INT64, FORCE_NO_CONDITIONAL_RENDERING, all four together, and SPARSE=0.
- Corpus d2: dumps byte-identical to a build of b63bc57. Corpus d1: base run twice differs in 5 scenarios (sub-pixel float noise lines); new build differs in 2 (busy-background, dark-background; one line each, 1e-3 px noise, same IDs).
- Commands: `$S/del-switches/{cfg,build,ctest}.bat <src> <build>`, `$S/del-switches/run.sh`.
