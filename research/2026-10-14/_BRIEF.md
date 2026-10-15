# Research round 2026-10-14: brief shared by every agent

## Goal
Find options that would make vkapriltag faster, both device-agnostic and device-specific.
Targets: **Orange Pi 5 (RK3588, Mali-G610 MP4, proprietary libmali g24p0, Vulkan 1.3.276)**, the
deployment target, and **AMD RX 9060 XT (RDNA4, Windows)**, the desktop. Each agent covers one area
(see the file name you were given). Unusual hardware (WMMA / "tensor" units, ray-tracing units,
work graphs, graph compilers) is in scope **only if there is a tangible benefit**; a clear "no, and
here is why" is a valid and useful result.

## Write as you go (the user may run out of tokens at any moment)
- Your output file is `research/2026-10-14/<NN>-<topic>.md`. Create it **in your first few tool
  calls** with a header and a "Status: in progress" line, then append each finding as soon as you
  have it. Never hold findings back for a final write-up. Update the Status line and a short
  "Top conclusions" section at the top as your view changes. When done, set "Status: complete".
- Each finding uses this shape:
  - **Idea** (one line) and which device(s) it applies to
  - **Mechanism**: why it would be faster, tied to the actual code (file:line) and to measured
    numbers from the tree where they exist
  - **Expected gain**, with the basis (measured in-tree / published number / estimate; say which)
  - **Cost and risk** (effort, accuracy impact vs libapriltag, maintenance)
  - **Prior art in the tree**: was it already tried? cite OPTIMIZATION_NOTES.md / PERFORMANCE.md /
    research/*.md. If it was rejected, only re-propose it with new evidence.
  - **Verdict**: pursue / measure first / reject, and **how to measure it** with the project's tools
  - Sources (URLs) for external claims.

## Rules
- **Read-only on the code.** Do not edit any file outside your own output file. No git commands that
  change state (no commit, switch, stash, checkout). Others are working in the same tree.
- **No builds and no timing runs** on this PC: several agents run at once and would distort each
  other and the GPU. Reading code, docs, git log/show, web search and `vulkaninfo` are fine.
- Be concrete and skeptical. Many obvious ideas were already measured and lost. Do not repeat a
  rejected idea without new evidence. Do not invent numbers.

## Required reading before proposing anything (skim, then go deep where relevant to your topic)
- `research/00-SUMMARY.md` and `research/_BRIEF.md` (previous round, 2026-09; 8 category files
  `research/01..08-*.md` with 219 sources: implementations, algorithmic speedups, GPU CCL, union-find,
  scan/compaction/sort, atomics/hashing/threshold, Mali compute, Vulkan extensions).
- `vkapriltag-speedup-survey.md`, `vkapriltag-optimization-ideas.md` (repo root; already reviewed
  against the tree, verdicts inline).
- `apriltags_vulkan/OPTIMIZATION_NOTES.md` (what was tried, with numbers) and
  `apriltags_vulkan/PERFORMANCE.md` (current numbers; sections 8d-8g are from the latest session).
- `apriltags_vulkan/DESIGN.md`.
- Code: `apriltags_vulkan/library/shaders/*.comp|*.glsl` (GPU stages: decimate, threshold /
  block_minmax / block_filter, uf_init / uf_merge / uf_compress / uf_final (union-find labelling),
  blob_diff, hash_group, reduce_extents_hash*, select_blobs, scan_*, scatter_index_points,
  sort_points_local, build_indirect_args), `library/src/gpu/GpuDetector.cpp` (host orchestration,
  fused single-submit frame, pre-recorded command buffers), `library/src/gpu/QuadDecode.cpp`,
  `library/src/TagDecoder.cpp`, `library/src/gpu/FramePipeline.cpp`, `library/src/vk/*`.

## What the latest session established (2026-10, branch perf/full-prerecord, not yet in research/)
- A built-in tracer (`library/include/vkapriltag/common/Profiler.h`, `--profile` / `--trace` on
  `apriltag_vulkan_validate`) with GPU timestamp spans on a virtual track.
- CPU tail: **96.8% of the CPU refine time is spent on quads that never decode** (refine runs
  before decode rejects them); see PERFORMANCE.md 8e for the candidate list.
- Vulkan host overhead per Detect(): desktop ~58 us with pre-recorded command buffers (83 us
  without); Orange Pi 126 us (197 us without). Recording itself was ~22 us desktop / ~66 us Pi
  and is now gone on the fused path. Host-pointer import (zero-copy ingest) measured in 8d.
- Orange Pi at 1280x800 (decimate 2, Release, governors pinned): Detect() ~2.70 ms,
  pipeline_total ~4.20 ms, 1 stream ~381 fps, 2-4 streams ~445 fps (GPU-bound).
  Desktop: 1 stream ~3,000 fps, 4 streams ~4,600 fps.
- The fused path (one submission per frame) is taken on both GPUs; union-find labelling always
  converges in 2 iterations on the corpus and stress images; retries never occur in practice.
- GPU clock state varies up to ~20% between processes: GPU-side claims need many interleaved runs;
  host-side claims should use host overhead (gpu.detect minus gpu.submit_wait).
