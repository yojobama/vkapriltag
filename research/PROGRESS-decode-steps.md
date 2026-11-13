# Progress: TagDecoder::Decode split into candidate steps (branch research/2026-11-decode-steps, base c3ed420)

Pure refactor of `library/src/TagDecoder.cpp`: output and behaviour unchanged, no GPU file touched. One commit per step.

## Structure now
- `DecodeEnv` (frame, switches, first family, min white cells, third-bit index), built once per `Decode()` by `MakeEnv`.
- `Candidate`: the input quad, an `OwnedQuad` (the refined quad; frees the homographies `quad_decode_index` allocates),
  `found` (the slot's detections), `probe` (the thread's trial-decode scratch), the shared `UnrefinedSample`, probe bin and margin,
  and a `StepClock` whose laps cost nothing when profiling is off.
- `RunSteps(env, c)` runs, in the old order: `NormaliseCandidate`, `MarginGateRejects`, `PreRefineRejects` (one shared
  `UnrefinedSample`), `RefineCandidate`, `DecodeCandidate`, `TryInnerSquare`, `TrySideOneOut`, `TryThirdBit`,
  `TrySmallRescue`, `ApplyFamilyLimits`, `PolishFound`. `TagDecoder::ProcessCandidate` wraps it and records the probe
  histogram, CSV rows and `SlotStats`. `Decode()` is: set up, `ParallelFor` over `ProcessCandidate`, ordered merge, reconcile.
- `OwnedQuad`, `InnerQuad`, `QuadFromHomography` replace the seven copy/null-H/sample/free sequences and ten manual `matd` frees.
- Hooks (`TestHooks.h`): `PreRefineRejects`, `RunCandidateSteps` (with `DecodeSwitches`, `DecodeSwitches.h`); tests: ctest `unit_steps`.

## Commits (base c3ed420)
| commit | step |
| --- | --- |
| f2e7ff3 | OwnedQuad, InnerQuad, QuadFromHomography |
| d61fb9e | DecodeEnv, lambda becomes `ProcessCandidate` |
| d269734 | Candidate, StepClock, `RunSteps` + statistics in `ProcessCandidate` |
| 97497a5 | NormaliseCandidate |
| 3e26f55 | MarginGateRejects, PreRefineRejects, one shared UnrefinedSample |
| f09e238 | RefineCandidate, DecodeCandidate |
| 99650d0 | TryInnerSquare |
| b37a23a | TrySideOneOut |
| 6f6021f | TryThirdBit |
| 2eb8977 | TrySmallRescue |
| 12ce841 | ApplyFamilyLimits, PolishFound |
| b248b71 | RunSteps tidy; MakeEnv, MergeSlotRecords, SumSlotStats leave Decode |
| 077f2ae | DecodeSwitches, MakeDecodeEnv, test hooks |
| 3b631cb | unit_steps tests |

## Commands (Git Bash; scratch = scratchpad `decode-steps/`)
Base build `ext/avk-compare/build-decode-steps-base` (c3ed420, repo CMake with tests), branch build `build-decode-steps`,
snapshot of the final executables `snap-decode-steps` (so a rebuild does not disturb a running gate). Scripts: `gate.sh`/`gate2.sh`
(corpus dumps, per-label dirs), `cmpd.py` (identical / jitter <= 0.05 px / real difference), `fullgate2.sh` (all switch
combinations d1/d2), `seq2.sh` (track_eval, full and key), `timing2.sh` (ABBA, `ROUNDS=4`, harness cpu median).
Configure/build: `cmake -S apriltags_vulkan -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF` through vcvars64.

## Unit tests added (`unit_steps`, 0.03 s)
Pre-refine: a rendered tag is kept (grey margin on and off); a solid black square is rejected by the dark check (and kept
with `dark_reject` off); 24 of 40 quads over a noise patch are rejected (the rest have an unreadable border and are kept, by
design); a grey-ringed outline is kept by the early reject. Recovery: plain tag decodes; grey ring decodes only with the inner
retry; three touching wrong cells decode as hamming 3 only with the third bit, three scattered ones never; a dark line one
cell above the tag decodes with hamming 2 and the top at y 41 without side-one-out and hamming 0 at y 48 with it;
tag16h5 with one wrong cell is dropped by the family limits and kept without them.

## Complexity (lizard, cyclomatic)
`TagDecoder::Decode`: 152 (this lizard) before, 8 after (file 622 -> 749 lines, 19 -> 56 functions). Largest now:
`PreRefineRejects` 22, `TrySideOneOut` 20, `~TagDecoder` 15 (unchanged), `ErrorsConnected` 14, `ProcessCandidate` 13,
`TryThirdBit` 12, `TrySmallRescue` 11, `RunSteps` 4.

## Gate (head 3b631cb against a build of c3ed420; dumps compared with cmpd.py, jitter = corner/margin difference <= 0.05 px)
- ctest: 9 of 9 pass (golden_detections unchanged, new unit_steps).
- Corpus, 12 configurations, every one with 0 real differences (only the known 1e-2 px one-frame jitter files):
  d2, d1, d2 and d1 with SMALL_RESCUE=1 HULL_QUAD=1, POLISH=1, EARLY_REJECT=0, GREY_MARGIN=0, THIRD_BIT=0, SIDE_ONE_OUT=0,
  FAMILY_LIMITS=0, PROBE_MARGIN=15 with PROBE_STATS=1, GREY_MARGIN=0 with EARLY_REJECT=0 (family16h5 run with tag16h5).
  The same 12 also passed on the commit before the test hooks (b248b71).
- track_eval, ten sequences, modes full and key: detections identical to base (base against base2 also identical).
- Timing (ABBA, 4 rounds, `vk_eval --iterations 50 --warmup 5`, cpu median ms, ratio branch / base):
  d1: one-tag 1.011, many-tags 1.000, dense-tags 0.999, clutter-only 0.993, jpeg 1.004, busy-background 1.000.
  d2: one-tag 1.007, many-tags 0.996, dense-tags 1.001, clutter-only 0.992, jpeg 1.014, busy-background 0.999.

## Odd behaviours found, not changed
1. With profiling on and polish on, `PolishFound`'s lap is taken after decode, recoveries and limits, so that time is
   counted as refine time (`refine_ms_*`) and `decode_ms` is only the remainder.
2. The probe histogram and CSV skip quads rejected by the border/dark pre-check and coarse boxes that fail to refit; quads
   stopped by the margin gate count as rejected in the histogram but get no CSV row.
3. `TrySideOneOut` runs on the outer quad's homography even when `TryInnerSquare` produced the detection (the border it reads is
   the (W + 2)-cell square's, not the tag's).
4. Side-one-out refit can land inside the true edge: a line one cell out at y 42 or 43 gives hamming 0 with the top corner at
   y 50.75 and 51.62 where the edge is at 48 (kept because the border level improves).
5. Multi-family limitation (noted, untouched): the grey-margin inner retry, side-one-out, third bit and the margin gate read
   family 0 only; the pre-refine checks run only with exactly one family.
6. The pre-refine checks keep a quad whose border cannot be read: only 24 of 40 quads over uniform noise are rejected.
7. Behaviour-neutral change: with the margin gate or probe statistics on and the pre-refine checks active, the unrefined quad
   is now sampled once instead of twice.
