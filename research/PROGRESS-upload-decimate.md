# PROGRESS upload-decimate (branch research/2026-11-upload-decimate, base 1879a92)

Idea: at decimation above 1, decimate on the CPU while uploading (read the caller's frame once, write the decimated
image into a host-visible decimated buffer), skipping the full-frame upload and the GPU decimate dispatch.

## Log
- 1. Implemented (commit 4902f2e): `gpu/CpuDecimate.cpp` (NEON / SSE2 for decimation 2 and 4, scalar otherwise, box and
  top-left, 8-bit and 32-bit layouts), `PipelinePlan::cpu_decimate` (printed by Describe; needs a mapped coherent decimated
  buffer and no transfer lane; otherwise the GPU path stays), option `APRILTAG_VK_CPU_DECIMATE`. Imported frames keep
  their GPU decimate. Not yet built or measured.
