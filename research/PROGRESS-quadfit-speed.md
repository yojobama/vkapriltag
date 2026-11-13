# Progress: quadfit-speed

Branch research/2026-11-quadfit-speed (from 1879a92). Base build: $A/build-quadfit-speed-base (built, untouched source).

## State log
- Read QuadDecode.cpp, OwnDecode.cpp, BorderSample.cpp, TagDecoder steps. Recompute audit (TagDecoder): the unrefined
  sample is shared (SharedSample) between margin gate and pre-refine check; the inner-square and final decodes use
  different quads (after refine), so the homography is not recomputed for the same quad except when refine leaves it unchanged.
  TryThirdBit reuses c.raw (no resample). ReadBorder runs on the unrefined quad (precheck) and in TrySideOneOut twice.
- Plan: (1) DP path on unpacked int32 arrays (exact in int32: coords < 2^14); (2) peaks path; (3) ReadBorder median.
