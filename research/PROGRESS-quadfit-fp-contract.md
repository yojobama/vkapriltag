# Quad fit fp-contract fix

- CI (ubuntu-26.04, gcc targeting x86-64-v3 with FMA) failed unit_quadfit32: scalar qf::FitLine and SSE2 qf::FitLines disagreed in 2 of 37 sets, because gcc contracted the scalar a*b+c*d.
- Fix: library/CMakeLists.txt gives src/gpu/QuadFitF32.cpp -ffp-contract=off beside -fno-math-errno (gcc/clang branch; MSVC is not in it and the build passes no /fp: option).
- WSL gcc 13.3 repro (-O3 -fno-math-errno): v3 with -ffp-contract=off 0 mismatches; v3 alone 2; v2 0. WSL has no Vulkan/glslang, so the library was not configured there.
- MinGW g++ 16.1 on Windows, same driver: v3 with -ffp-contract=off 0; v3 alone 2.
- MSVC Release, Ninja: ctest 47/47 passed.
- Corpus (decimation 2): main and this branch score identically; the noise and underexposed dumps differ slightly (known run-to-run variation).
