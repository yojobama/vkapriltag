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
