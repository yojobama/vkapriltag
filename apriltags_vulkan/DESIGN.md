# Design rationale

Why the library is shaped the way it is. This page is short on purpose: the pipeline as built is in
[ARCHITECTURE.md](ARCHITECTURE.md), every idea tried (kept or dropped, with numbers) is in [DECISIONS.md](DECISIONS.md), the
numbers and switches are in [PERFORMANCE.md](PERFORMANCE.md). The per-file notes that used to live here (many of them stale:
they describe `label_pixels` and the Batcher sort, both removed) are frozen in
[archive/DESIGN-history.md](archive/DESIGN-history.md).

## 1. The problem and the target

An AprilTag detector spends its time in two different kinds of work. The front end touches every pixel (decimate,
threshold, find connected regions, find the boundary between black and white regions). The back end touches a few
thousand boundary points and a handful of candidate quads (fit four lines, sample a code, check Hamming distance). The
library is built for embedded and mobile GPUs, primarily an ARM Mali-G610 on a Rockchip RK3588 (Orange Pi 5), where the CPU
is slow and the GPU is idle, and it is verified on a discrete AMD GPU as well. It ports frc971's CUDA `GpuDetector`
(itself built on AprilRobotics' `apriltag`) to Vulkan compute so that it runs without CUDA.

## 2. Why a GPU front end with a CPU tail

- **The pixel work is embarrassingly parallel and bandwidth-shaped; the tail is scalar and branchy.** The GPU removes the
  O(pixels) cost. What reaches the CPU is a few thousand boundary points, sorted by angle per blob, as 8-byte records. Quad
  fitting, edge refinement and code sampling are small, irregular computations that are simpler to write and verify as C++
  than as compute shaders, and the quad fit on the GPU was built once and removed: it cost more than it saved on the desktop
  and needed features (64-bit floats, extended subgroup types) the Mali does not have (`DECISIONS.md` section 2).
- **Pose stays on the CPU.** It is per detected tag, tiny, and a GPU round trip would cost more than the solve.
- **The honest limit.** A CPU detector (NAPA) is about 3.6 times faster on small frames on the desktop and 2 to 6 times on the
  Pi: the GPU path has a floor of 0.3 ms (desktop) to 0.5 to 0.75 ms (Mali) from the dispatch chain. The GPU wins where per-pixel
  work dominates (large frames, decimation 1, noisy frames). A CPU or hybrid front end was never built.

## 3. Why these stages, in this order

The stage list is the shortest chain that turns pixels into per-blob sorted point lists without a host round trip in the middle:

1. **Decimate, block min/max, threshold.** AprilTag's adaptive threshold needs the min and max of each 4x4 block and its
   neighbours; a contrast floor (set from the frame's noise) keeps sensor noise from being labelled. Decimation is the largest
   speed lever (decimation 4 is about 2x faster than 2) and box averaging removes the aliasing of point sampling.
2. **Union-find labelling** (one merge pass; a verification merge and a retry path exist behind a switch). Union-find maps
   well to atomics and needs no ordered scan; run starts found with a subgroup scan remove a pass.
3. **Boundary points between differently coloured regions**, compacted with one atomic counter, each tagged with the pair of
   region roots it separates.
4. **Group by pair (hash), reduce per-pair extents, select** by size, aspect, fill and border polarity: this replaced a radix sort
   and cut roughly 30 dispatches to 4.
5. **Scatter and sort by angle** per blob, writing the final line-fit record directly, so the CPU needs no second pass over the
   points.

Everything is sized from device-side counts through indirect dispatch (or fixed grids on integrated GPUs), so one frame is one
queue submission. That mattered more than any shader change on the Mali: a submit boundary was 0.44 to 0.61 ms of a 12 ms frame,
and a trivial dependent dispatch costs about 25 us there against 4 us on the desktop. The design therefore prefers fewer, wider
dispatches over fewer instructions, and records the frame once and replays it.

## 4. Why the device-class forks

Two devices behave differently enough that one pipeline does not fit both; each fork is a measured result on one device, not
a guess:

- **Discrete (RDNA4).** Cheap atomics, cheap dispatches, wide workgroups. Subgroup aggregation pays at one site (reducing values by
  key), banded merging helps, a 1024-wide sort workgroup helps, and the extra work of compaction costs more than it saves.
- **Integrated (Valhall).** Shared memory is backed by L2 and subgroup ballot/shuffle sequences are slower than plain atomics
  (the subgroup extents shader took the 1.7-era frame from 11.5 to 18.1 ms), but a dispatch is expensive, so the tail is made
  leaner (no argument kernels), labelling is compacted to cells with contrast, the merge covers four pixels per thread, and a keep-alive
  stops the GPU dropping to a low-power state between frames.
- **8-bit versus 32-bit storage.** The 8-bit shader family is the fast one and holds the adaptive floor, fused threshold and compaction;
  the 32-bit family exists only so that a device without `storageBuffer8BitAccess` still runs. Both families, the unfused four-submit
  flow (for devices without host-cached readback memory) and the non-subgroup variants are kept as fallbacks and are exercised through
  `APRILTAG_VK_FORCE_NO_*` in the golden test, not on separate hardware.

The requirement is core Vulkan 1.1 only; every optional extension has a fallback.

## 5. Why the CPU tail looks the way it does

- **Candidates are cheap to propose, expensive to refine.** Most quads in a cluttered frame are not tags and edge refinement
  dominated the tail, so the decoder is a chain of named steps (reject early, refine, decode, recover, limit) rather than one
  function, and the early-reject checks run on the unrefined quad.
- **Accuracy lives in the refinement.** The default is a dense-profile edge fit with weighted line fitting and a blur level pass; the
  whole-tag polish and the small-tag and hull rescues are opt-in because their cost on cluttered frames is above the gate.
- **The decode is in-house** so that detections do not depend on a patched, fetched third-party library, allocate nothing per attempt and
  can be checked bit for bit against vectors recorded from upstream. Upstream remains only as an optional validation dependency.
- **One worker pool, stable merge order.** Candidates are decoded in parallel and merged in quad order, so thread count never changes the
  output; one pool serves the whole pipeline; on asymmetric CPUs the workers are pinned to the big cores.
- **Temporal reuse is opt-in.** Tracking re-detects known tags on the CPU from predicted corners and runs the GPU only on keyframes or
  in a background thread; it helps slow scenes (0.54 to 0.077 ms) and does nothing for fast ones.

## 6. Principles that survived the measurements

- Measure on both device classes, interleaved (ABBA), and state the cost: most plausible ideas lost on one of them.
- A behaviour change goes behind a configuration field and an environment switch; with it off the output must equal the base.
- Delete settled A/B switches: they hid dead code and made documentation lie. Keep only switches that force a fallback or help debugging.
- Prefer a rule over a constant: the subgroup lesson (aggregate values by key, not counters) transferred; a tuned workgroup size did not.
- Determinism is a property of the CPU tail. The GPU front end is not bit-stable (atomic append order), so tests allow corner jitter of
  about 1e-3 px and one marginal detection per profile.

## 7. Deliberately out of scope

No lens-distortion model (undistort corners before pose), one border polarity and one decimation per detector, no pose inside `Detect`,
no multi-frame fusion other than the optional tracker. Restructuring ideas that were deferred (a resolved pipeline plan and stage table,
a front-end interface, a CPU front end for sparse frames) are listed in `research/RESTRUCTURING-REPORT-2026-11.md` section 6.
