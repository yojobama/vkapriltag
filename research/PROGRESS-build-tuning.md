# PROGRESS build-tuning

Branch `research/2026-11-build-tuning` (from 1879a92). Pi dir `~/w/build-tuning` (src/ = 1879a92 source, b-<name> build dirs, scripts bt_build.sh).

## Log
- Setup: branch created, source shipped to the Pi, nothing built yet. Next: base builds on both machines, vectorisation report, then LTO / flags / PGO / target measurements.

## Vectorisation report (gcc 14.2 aarch64 on the Pi, `-O3 -fopt-info-vec-missed -fopt-info-vec-optimized`; MSVC 19.5x `/O2 /Qvec-report:2`)
Commands: `~/w/build-tuning/vec.sh generic` and `vec.sh a76 -mcpu=cortex-a76` (compile line taken from `ninja -t commands`, flags appended);
outputs in `~/w/build-tuning/vec/<tag>/<File>.txt`. `-mcpu=cortex-a76` changes nothing material: same loops vectorise, only a few
SLP "basic block" decisions flip (ReadBorder side_sum init, ModelPass `S[4]` inits). Line numbers are 1879a92.
"loop" = real loop vectorisation; "SLP" = straight-line pairing of 2 doubles / 4 floats (small, no loop-level gain).

| Function (file) | Vectorised by gcc | Not vectorised, gcc's reason |
|---|---|---|
| ProfilePass (RefineEdges) | loop 281 (profile sample accumulation, 16 B) | 263 edge loop, 272 sample loop: "two or more consecutive inner loops"; 295 argmax `if (g[j] > best)`: "unsupported use"; 302/303 lobe-walk while loops: "multiple exits"; 306 `gs += g[j]` float->double: "relevant stmt not supported"; 316 iter loop, 326: "unsupported control flow" |
| LevelPass (RefineEdges) | loop 439 (16 B) | 377 edge, 434 sample: "consecutive inner loops"; 397, 437: control flow (early exit on `contrast`); 404 "unsupported outer loop form" |
| FitSide (RefineEdges) | none | 236 weighted-sum loop: "unsupported control flow" (a conditional in the body) |
| ClipBias (RefineEdges) | SLP only (545, 546) | 529/534 `&& nb < 62` early exit; 547/548 consecutive inner loops; 561, 590 control flow |
| FitQuadForBlob (QuadDecode) | SLP only (moment copies, result struct) | 426/435 cumulative-sum and peak loops: control flow; 452 peak test; 466 "relevant stmt not supported" (map `second`); 473/474/486-502 nested maxima search: control flow; 408 "unsupported use" (`mxy += wx*y2`, a reduction under `-ffp-contract=off`/no reassociation) |
| FitLine (QuadDecode) | none | straight-line code: "vectorization is not profitable" |
| FitQuadHull (QuadDecode) | none | 579, 592, 609, 627, 636, 661-665 (4-nested corner search), 687, 704, 708, 721: control flow; 620 "volatile type" (`keep[0]=1` on a vector<bool>/volatile-like proxy); 682/697 consecutive inner loops |
| SampleFamily (OwnDecode) | SLP 272 (2 B), 289 (16 B) | 274, 276, 295, 301, 302, 321, 334, 365: control flow; 347/348 consecutive inner loops; 359 "not suitable for gather load" (`sharp[i] = all_sharp[idx[i]]`) |
| ComputeHomography (OwnDecode) | loops 446, 451 (elimination row update), SLP 417, 431 | 417 `(9 >> i)` bit test: "stmt not supported"; 438 pivot max `if (val > max)`: "unsupported use"; 448 nested; 451 (in the missed list too) "complicated access pattern"; 454 control flow |
| ReadBorder (BorderSample) | loop 59, SLP 23, 24 | 26 `c += (r==0 \|\| r==w-1) ? 1 : w-1` "could not determine main exit"; 60 control flow |
| PolishFit BuildFit | loop 383 | 373 nested; 391, 406 control flow; 333 "relevant phi not supported" |
| PolishFit BuildScan / ComputeBox / Freeze | none | control flow / "not profitable" (short fixed trip counts) |
| PolishFit ModelPass | SLP only (485, 512, 546, 580, 633: `float S[4]`) | 494 nested; 582 "consecutive inner loops"; 590, 635 control flow; 595/641 `S[c] += ...`: "data ref analysis failed" (indexed accumulator) |
| PolishFit Evaluate | loop 826 (col *= gw) | 797, 833, 835 control flow (the `k += 4` stride loops); Dot4 itself is intrinsic-written (V4), nothing to report |
| PolishFit FitLevels / Optimise / MedianAbsResidual | SLP 706, 922 | 688 float->double convert "not supported"; 715 "multiple exits"; 905/912/917 struct copy "more than one data ref" |

MSVC (`/O2 /Qvec-report:2`, reason codes are Microsoft's; 1200 = loop-carried dependence, 1100/1106 = control flow or unsupported operation,
506 = loop with too-few or unsupported iteration form (short fixed counts), 1301 = non-unit stride; treat the meanings as approximate):
vectorised loops per function: ProfilePass 1 (the sample accumulation, same as gcc), BuildFit 1, Evaluate 1, Optimise 1, Dot4 2; everything
else 0: LevelPass, FitSide, ClipBias, FitQuadForBlob (about 90 loops, mostly 506/1301/502), FitQuadHull, SampleFamily, ComputeHomography
(gcc vectorises the elimination row update, MSVC does not: code 1200/1203), ReadBorder, FitLevels, BuildScan, Freeze, ComputeBox.

Reading for the other agents: the CPU tail is branchy scalar code over short, data-dependent loops (early exits, argmax, nested searches);
the compilers find no loop worth vectorising beyond the single sample accumulation loops (ProfilePass 281, LevelPass 439, BuildFit 383,
Evaluate 826). Gains there need restructuring by hand (fixed-trip loops without early exit, struct-of-arrays, intrinsics as Dot4 does), not flags.

## Experiments
Method: options implemented first (`cmake/BuildTuning.cmake`: VKAPRILTAG_LTO, VKAPRILTAG_TUNE, VKAPRILTAG_CODEGEN, VKAPRILTAG_PGO[_DIR], all off by default),
variants built as `$A/build-build-tuning-<name>` (desktop, `bt/dbuild.sh NAME args`) and `~/w/build-tuning/b-<name>` (Pi, `bt_build.sh`).
Timing: `bt/abn.sh DEC OUT ROUNDS name=exe ...` (whole corpus, 5 iterations per frame, interleaved with rotating order, 5 rounds), summary `bt/an.py`
(median over rounds of corpus median/p95 per frame, gpu, cpu, four scenarios). Scripts live in the scratchpad `bt/` dir; copies of the essentials are in `research/build-tuning-scripts/`.

### E1 desktop (MSVC 19.5, Ryzen 5600X), 5 rounds, % against `base` (unmodified flags)
Decimation 2: base 0.3530 / p95 0.6640 ms, gpu 0.2390, cpu 0.1036.
- lto (/GL /LTCG): med +2.4%, cpu +5.0% (slower), dumps equal.
- avx2 (/arch:AVX2): med -4.2%, p95 -3.6%, cpu -12.6%; dumps equal in the quick check (ctest pending).
- gw (/Gw): med +3.2% (noise or worse; no gain).
- lto+avx2: med -3.0%, cpu -10.5%.
Decimation 1: base 0.4090 / 0.8615, gpu 0.2984, cpu 0.1079: every variant came out 1 to 6% SLOWER including `gw`, gpu part +1% everywhere,
so the base binary has a systematic advantage (binary layout / build provenance). Need a second copy of base built the same way to measure the noise floor (E2).

### E2 desktop with a second base (base2 = same flags as base, built from this branch with the new module): noise floor and variants, 7 rounds
Cross-session drift of the unmodified flags is large (corpus median 0.353 in E1, 0.3685 in E2 at decimation 2; the GPU is shared), so only the
interleaved columns inside one run mean anything. Dump equality: lines that differ against `base2_1` (the GPU front end gives 0 to 7 lines of noise per run).
Decimation 2 (base2 0.3685 / p95 0.6924, gpu 0.2433, cpu 0.1117): lto med -4.7% p95 -3.5% cpu -1.8%; avx2 med -7.8% p95 -8.5% cpu -16.9%; lto+avx2 med -3.5% p95 -3.8% cpu -8.0%.
Decimation 1 (base2 0.4136 / 0.8590, gpu 0.3046, cpu 0.1098): lto +1.0% / -0.9% / cpu -1.5%; avx2 -0.1% / -0.5% / cpu -4.6%; lto+avx2 -1.4% / -2.9% / cpu -6.8%.
Per scenario (d2, total ms change): avx2 one-tag -3.0%, dense-tags -5.8%, jpeg -12.6%, busy-background -9.2%.
Dumps equal within noise for all (0 to 5 differing lines against the base2 run; base2 itself 7 at d1). So: /arch:AVX2 is the only win; LTO is within noise.

### E3 Pi (RK3588, gcc 14.2), 5 rounds, 4 variants per run against `base`
Decimation 2, group A (base med 0.7714, p95 1.2690, gpu 0.5877, cpu 0.1546): lto med -3.2% cpu +0.9%; a76 (-mcpu=cortex-a76) med -3.3% cpu -3.0%; lse (-march=armv8.2-a) med 0.0% cpu +2.8%.
Decimation 2, group B (base 0.7863 / 1.2714, cpu 0.1627): cg (-fno-math-errno -fno-trapping-math -fno-semantic-interposition -fno-plt) -5.8% cpu -6.9%; o2 (-O2 instead of -O3) cpu +19.5%, one-tag +3.4%, dense +11.1%, jpeg +8.9% (O3 is clearly better); lto+a76 -3.9% cpu -7.1%.
Decimation 1: group A (base 1.0649 / 2.0588, cpu 0.1438): lto +0.6% cpu +1.0%; a76 -0.2% cpu -0.6%; lse +4.3% cpu +4.7%. Group B (base 1.0622 / 2.0635, cpu 0.1447): cg +3.8% cpu +14.5% (contradicts d2: noise); o2 +13.1% cpu +42.7%; lto+a76 +1.7% cpu -0.6%.
The Pi cpu part is 0.14 to 0.16 ms of 0.77 to 1.06, and it swings by +-10% between runs (threads land on A55 or A76 cores), so only -O2 (clearly worse) is distinguishable from noise.
Dumps: 4 to 11 differing lines per run against the base run, the same as base against base (5 to 9): GPU noise.
Next: pin to the A76 cores (`taskset -c 4-7`) to take the scheduler out of the cpu medians, break cg into its four flags, combine, then PGO (gcc on the Pi, MSVC on the desktop), ctest per config, SWIG and consumer checks.

### E4 desktop (ctest and more variants; 7 rounds, decimation 2, base2 and a second `base` binary in the same interleave)
ctest: 44/44 green (own_decode_*, goldens, installed_package_consumer) for base, lto, avx2, avx2+/fp:contract (avx2fc), avx2+/Ob3 /GS- (x2) and lto+avx2.
MSVC does not fuse under /arch:AVX2 with the default /fp:precise, and the tests also pass with /fp:contract on the whole library.
`base` and `base2` (identical flags, separate builds) differ by 3.8% in the corpus median and 7% in cpu inside one interleaved run: binary layout noise.
Corpus median / cpu, against base2 (0.3778 / 0.1208): base -3.8% / -7.2%; avx2 -4.0% / -11.5%; x2 -5.3% / -16.2%; avx2fc -8.6% / -19.0%.
Against `base` the same variants are +0.2..-5% / -4.6..-13%: AVX2 helps the cpu part by 5 to 15%, the pipeline by 0 to 5%; /Ob3 /GS- adds nothing detectable;
/fp:contract is the best but is bit-identical only by test, not by guarantee (it lets MSVC fuse in QuadDecode and OwnDecode, which GCC's `-ffp-contract=off` forbids).

### E5 Pi pinned to the A76 cores (`PIN="taskset -c 4-7"`), decimation 2, 5 rounds: the run-to-run spread drops to under 1%
Group a (base 0.7143, cpu 0.1496): -fno-math-errno +0.3% / cpu +0.6%; -fno-trapping-math +0.2% / +0.6%; -fno-semantic-interposition -fno-plt +0.4% / -0.2%. Nothing.
Group b (base 0.7175, cpu 0.1499): all four codegen flags -0.1% / +0.8%; + -mcpu=cortex-a76 -0.5% / -1.1%; + lto -0.7% / -1.9%. Dumps identical (0 to 1 lines).
The unpinned -5.8% / -6.9% of E3 was scheduler noise. Conclusion: on the Pi no compiler flag moves the pipeline by more than 1% (cpu part is 0.15 of 0.72 ms).

### E6 desktop MSVC PGO (`VKAPRILTAG_PGO=GENERATE`, `tools/pgo/train.sh`, `pgomgr /merge`, `USE`; /GL /LTCG implied), 7 rounds
Train: 4 scenarios at decimation 1 and 2 plus the 10 sequences through track_eval (54 s). MSVC writes the .pgc files beside the executable and the
link does not merge them: `pgomgr /merge "<exe>!*.pgc" <dir>\<exe>.pgd` is needed before the USE link (train.sh does not do it yet: TODO).
Result against base2: d2 med +1.7%, p95 +2.9%, cpu +1.5% (base itself +1.4%); d1 +4.0%, cpu -1.8%. No gain; the link printed nothing about the profile, so
whether it applied is unverified. MSVC PGO also only tunes the executable that is linked, not an installed static `.lib` used by a consumer.
Dumps equal within noise.

### Pi status at the end
Unreachable (ping and plink time out) from the E5 / PGO stage on. Done on the Pi before that: E3, E5 (`p1c`: a76, lto, lse pinned, first rounds only seen), vectorisation report,
gcc PGO generate build (`b-pgo`) and its training run (gcda written under `~/w/build-tuning/b-pgo/pgo`). NOT run: the gcc PGO USE build and its timing, `ctest` on the Pi for any
tuned build, SWIG, track_eval on tuned builds, the 4-flag combinations at decimation 1 pinned.

### Gate (defaults are unchanged, nothing is on by default)
Default configure: flags identical to 1879a92 (base2 build.ninja has the same compile lines; dumps equal; deps_lint and options_lint pass).
Desktop `ctest` 44/44 for base, lto, avx2, avx2+/fp:contract, avx2+/Ob3 /GS-, lto+avx2 (own_decode_*, goldens and installed_package_consumer included; LTO consumer works because the
MSVC linker handles /GL objects of the installed .lib with the same compiler). Not run: SWIG with LTO, track_eval of tuned builds, anything on the Pi.

### Recommendation
- Default: all off (portable, bit-identical). Keep as is.
- x86-64 Windows, desktop only (not the release artefact): `-DVKAPRILTAG_TUNE=/arch:AVX2` (cpu part -9% to -17%, bit-equality tests pass). gcc: `-march=x86-64-v3` untested here; it would contract to FMA in files without `-ffp-contract=off`, so check dumps first.
- Pi / aarch64: no flag helps (under 1%). The `.deb` and release.yml stay on the default flags; do not use LSE or `-mcpu` (breaks Cortex-A53/A72 era boards for ~0%). `-O2` is worse (+19% cpu): keep `-O3`.
- LTO and PGO: no measurable gain on either machine; leave off. If PGO is revisited, offer it only as the opt-in `VKAPRILTAG_PGO` with `tools/pgo/train.sh`, never in the default or release build.
- Allocator and threading (item 5): not measured (budget). Outline atomics helpers stay (1% on the Pi) because LSE-only builds are not portable.
