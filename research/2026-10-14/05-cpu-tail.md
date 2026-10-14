# 05 - CPU tail (quad fit / refine / decode / threading)

Status: complete

## Top conclusions (ranked by expected gain on the Orange Pi)

Every gain below is an estimate unless it says "measured". No builds or timing runs were made for
this survey. The first action for all of them is one `--profile --trace` run on the Pi, because
the tree has no current Pi split of the 1.5 ms tail into `quad_decode` and `tag_decode`.

1. **F1: gate refinement on an unrefined probe with a relaxed Hamming bound. Pursue.** 96.8% of
   refine time goes to quads that never decode. `ultrafast` already proves the gate pays (x86
   single-thread `tag_decode` 1.79 -> 0.23 ms, measured), but its Hamming-2 bound costs recall
   (663 vs 687/768). A probe that accepts unrefined Hamming <= 4-5 still rejects about 98% or more
   of clutter (by Hamming-ball volume: 0.23% / 1.5% of random 36-bit words), while letting
   marginal real tags through to the unchanged `exact` refine and decode.
   - **Accuracy.** Not provably exact (refinement can move corners by up to about 7x the search
     range). Every kept detection is bit-identical to `exact`, and the only possible loss is a
     missed tag, which the synthetic sweep measures.
   - **Pi gain.** About -0.55 to -0.65 ms of the 4.20 ms serial `pipeline_total` (-13 to -15%).
     At full multi-stream throughput it frees roughly 1.3-1.8 cores of CPU.
   - **Effort.** About 200-300 lines.
   - **Measure it first, with existing code.** Histogram the unrefined Hamming distance of every
     tag `exact` decodes on the sweep.
2. **F2: RK3588 cpuidle `cpu-sleep` (exit 120 us, min-residency 1 ms) is enabled during every Pi
   measurement. Measure first; zero code.** It can add up to 120 us to each pool wake and, more
   importantly, to the fence wake on the GPU-bound pipelined path. That is up to about 0.2 ms
   serial and 0-4% of 1-stream fps. A/B with
   `/sys/devices/system/cpu/cpu*/cpuidle/state1/disable`. The fix, if it shows, is a
   `/dev/cpu_dma_latency` PM-QoS request, or spin-then-park. The earlier "fence spinning does
   nothing" result was measured on the desktop only. Bit-identical.
3. **F3: in `FramePipeline` the GPU waits on the caller between passes.** The caller wakes,
   copies the points and notifies before the next pass can start. Double-buffered readback would
   let the `gpu-driver` thread chain frames itself. Estimate +2 to +5% 1-stream pipelined fps on
   the Pi **if** the measured gap between `pipeline.gpu_pass` spans is 50-150 us; nothing at 2 or
   more streams. Measure first. Bit-identical.
4. **F4: worker pools.** Use one pool instead of three per stream (`QuadDecode`, `TagDecoder`,
   `PoseEstimator`; 56 tail workers with 4 streams), claim work largest-first, and pin the
   `gpu-driver` thread to an A76. Tens of us up to about 0.1 ms. Cheap and bit-identical. The
   "load is balanced" result in 8e came from 12 identical Zen 3 threads and does not transfer to
   4xA76 + 4xA55.
5. **F6a: the same probe at Hamming 2 in front of the decode when `refine_edges` is off.** That is
   the sample app's setting. Exact if the probe is an operation-for-operation port of
   `quad_decode`. About -0.2 ms on the Pi in that mode (estimate). Shares F1's code.
6. **F5: a bit-identical faster `exact` refine.** Far-tap reuse is exact in double (34% fewer
   bilinear samples), and branch-free accumulation can keep the sequential order. About -20% of
   refine before F1 (proxy: `fast` on x86, measured), and only 10-20 us of single-tag latency
   after F1. Side finding: on aarch64, GCC contracts FMAs by default in both the C++ refine and
   the gnu17 upstream build. So "`exact` == upstream by construction" should be verified on the Pi
   with a full-precision corner diff; x86 cannot reveal a mismatch.
7. **F8: `QuadDecode`. Measure first.** 0.44 ms of wall time for a few hundred light O(n) blob
   fits looks like waiting (pool wake, a late large blob on an A55), not arithmetic. The known
   exact micro-wins are small. Inline `hypotf` as glibc's own double expression: bit-identical on
   glibc and overflow-safe, which answers the old rejection, but it is cold under `kDp`. Also a
   top-10 selection instead of `stable_sort`, and one fused pass.
8. **F9: per-edge parallel refine.** About 1% serial latency after F1. Low priority.
- **Rejected:**
  - **F7, batching refine/sample/decode (or the quad fit) on the GPU.** It needs a second
    submission (quads are a CPU product), Mali has no fp64 so exactness is impossible, the Pi's GPU
    is the throughput limit, and F1 leaves almost nothing to move.
  - **F6b, removing upstream's per-call allocations in the final decode.** A few us after F1; not
    worth patching upstream.
  - **F10, pose.** Not on the default path, and already 0.016 ms per tag.

Context for the ranking: with `FramePipeline` the Pi is already GPU-bound (381 / 445 fps), so tail
work shows up as **serial latency, CPU headroom and LPDDR contention**, not as pipelined fps. The
items that can move pipelined fps directly are F2 and F3, because they sit between GPU passes.

## Baseline facts used below (from the tree)

- Code path: `QuadDecode::Decode` (CPU quad fit, `library/src/gpu/QuadDecode.cpp:456`), then
  `TagDecoder::Decode` (`library/src/TagDecoder.cpp:75`), which runs for every candidate quad:
  `RefineEdges` (`TagDecoder.cpp:154-158`, default `kExact`, `RefineEdges.cpp:47-149`) and then
  upstream `quad_decode_index` (`TagDecoder.cpp:163`, patched out of `quad_decode_task`,
  `cmake/patches/apriltag-expose-decode-steps.patch:27`).
- This is upstream's own order: `quad_decode_task` refines first ("refine edges is not dependent
  upon the tag family, thus apply this optimization BEFORE the other work"), then
  `quad_update_homographies`, then `quad_decode` for each family (patch lines 43-47, 154-177).
  Upstream has no pre-decode gate. In upstream the quad search (`apriltag_quad_thresh`) dominates
  CPU time, so refining rejects does not stand out there. Here the quad search runs on the GPU, so
  the refinement of rejects is most of what the CPU still does.
- RX 9060 XT, `grayimage.pgm` 1280x800, decimation 2, 75 candidates and 1 tag (PERFORMANCE.md 8e):
  tail CPU 2.75 ms summed over threads, of which refine of never-decoding quads is 2.32 ms
  (**96.8% of refine**), refine of the decoded quad 0.08 ms, final `quad_decode_index` 0.35 ms
  (4.7 us/quad). Single-thread `tag_decode`: upstream 2.29, exact 1.79, fast 1.43,
  **ultrafast 0.23 ms**.
- Orange Pi (PERFORMANCE.md section 3, older tree, same image and decimation): `quad_decode`
  0.44 ms, `tag_decode` 0.79 ms (wall, 8 threads). Latest (8g): `Detect()` 2.69 ms,
  `pipeline_total` 4.20 ms (serial: `Detect` + `quad_decode` + `tag_decode`, see
  `validate_against_libapriltag.cpp:144-162`), so the tail is about 1.5 ms. **No current Pi
  split between `quad_decode` and `tag_decode` exists in the tree**, so the first thing to measure
  is `--profile` on the Pi (it prints `TagDecoder::Stats` and the per-span table).
- What "matches libapriltag" means in this tree: same decoded ID set and corner RMS against the
  unmodified upstream detector (`validate_against_libapriltag.cpp:246-262`). The candidate quads
  already differ from upstream's (GPU fit plus `kDp` corner seeding, `GpuDetector.h:58`). So the
  strict bar for a tail change is "bit-identical to the current `exact` output", and the project's
  acceptance bar is "same ID set, same corner RMS, same synthetic-sweep recall".

## Findings

### F1. Gate refinement on an unrefined probe with a relaxed Hamming bound (all devices; Pi first)

- **Idea.** Before refining a quad, sample it unrefined (border models, polarity test, 36 data
  bits, exactly what `quad_decode` does) and compute the **minimum Hamming distance** of the
  observed code (all 4 rotations) to the family. Refine and run the real decode only when that
  distance is <= H_probe, with H_probe larger than the decode's own `maxhamming` (2 for 36h11):
  4 to 6 is the range to try. Optionally, always refine quads below a size threshold, since
  unrefined corner error is largest relative to cell size there.
- **Mechanism.** `ultrafast` (`TagDecoder.cpp:131-150`) already gates refinement, but on a *full*
  unrefined decode (Hamming <= 2). That gate costs recall (663 vs 687 of 768 on the synthetic
  sweep, PERFORMANCE.md section 4) because a marginal real tag can have 3 or more bit errors from
  unrefined corners and only 0 to 2 after refinement. Non-tags almost never decode at any
  plausible bound: for 36h11 (587 codes x 4 rotations = 2348 codewords, 36 bits) the fraction of
  uniformly random 36-bit words within distance h of some codeword is about 2348 x V(h) / 2^36,
  where V(h) = sum C(36,k) for k <= h. That gives 0.002% at h=2, 0.23% at h=4, 1.5% at h=5 and
  8% at h=6. Real clutter is less random than that: uniform blobs give near-all-0 or all-1 codes,
  which are far from every 36h11 codeword. So a probe at h=4 or 5 should still reject about 98% or
  more of non-tags, while admitting real tags whose unrefined read is a few bits off. For quads
  that pass, the refine and the final decode are the unchanged `exact` path, so **every surviving
  detection is bit-identical** to today's output. The only possible difference is a lost
  detection: a tag with more than H_probe unrefined bit errors that would have decoded after
  refinement.
- **Why it cannot be made provably exact.** Refinement moves each edge sample by at most
  `range = decimate + 1` px along the normal (`RefineEdges.cpp:69-90`; n0 is a weighted mean of
  n in [-range, range]). A least-squares line through samples in a band of half-width r can sit
  about 1.5r off the original line at the segment ends. Corners are line intersections, so with
  the 11.5 degree minimum corner angle that `cos_critical_rad = 0.98` allows, a corner can move by
  up to about 1.5r / sin(11.5 deg), roughly 7r: around 20 px at decimation 2. No useful bound on
  bit flips follows from that. So "exact" is only reachable empirically, through the synthetic
  sweep and the corpus. A gate based on border contrast alone (PERFORMANCE.md 8e's first
  candidate, survey C4) is weaker: GPU blobs already pass the border-polarity test
  (`reversed_border` in the fit), so a plain dark rectangle on a light background passes any
  border test. It is rejected only by the code lookup, and only the bit read sees that.
- **Probe implementation.** Do not reuse `quad_decode_index` for the probe. It allocates per call
  (`quad_copy`, `matd_copy` x2, `matd_inverse`, a `calloc` for `values`, and `sharpen`'s scratch)
  and stops at Hamming 2. A probe needs a homography from 4 points (closed form, 3x3, double or
  float), 64 border samples plus 36 bit samples, two 3-parameter gray-model solves, the polarity
  test, the sharpen pass, and a min-distance search. For the search, either brute force (2348 x
  XOR+popcount, about 1-2 us on an A76) or a pigeonhole index: split 36 bits into h+1 chunks, and
  any codeword within distance h must match one chunk exactly. Upstream's own post-v3.4.5
  `quick_decode` uses this with 4 chunks for h <= 3 (research/01-fiducial-implementations.md:208).
  The probe does not need to match upstream bit for bit, because it only gates. Expected cost is
  1-3 us/quad on an A76, against roughly 40-60 us for an `exact` refine (estimate: x86 is about
  20 us/quad single-thread from 8e's 1.79 ms x 87% / 75; A76 at about 2-3x).
- **Expected gain.** Measured in-tree upper bound: `ultrafast` takes single-thread `tag_decode`
  from 1.79 to 0.23 ms on x86 (-87%), and the relaxed gate costs only a few percent more refines
  than that. On the Pi, `tag_decode` (0.79 ms wall on the older tree) would fall to roughly
  0.15-0.25 ms, about **-0.55 to -0.65 ms of the 4.20 ms serial `pipeline_total` (-13 to -15%)**
  (estimate scaled from the x86 ratio). With `FramePipeline` the Pi is GPU-bound (381 fps for 1
  stream, 445 for 2-4 streams, PERFORMANCE.md 8g), so fps would not move directly. What moves is
  CPU headroom: at 445 fps aggregate, `exact` `tag_decode` is about 3-4 ms of CPU per frame
  (0.79 ms wall x about 4-5x parallelism, estimate), which is **1.3-1.8 cores busy** on an 8-core
  part that also has to run the camera stack. The gate returns most of that. It also takes
  traffic off the shared LPDDR bus, the suspected cause of the original pipelining regression
  (OPTIMIZATION_NOTES.md "Pipelining the CPU tail ..."); that effect is unquantified.
  Desktop: `tag_decode` wall 0.28 ms falls to roughly 0.05 ms, out of about 1.0 ms
  `pipeline_total`.
- **Cost and risk.** About 200-300 lines (probe, codebook index, a new `RefineEdgesMethod` or a
  separate `gate_hamming` config with an env override). The accuracy risk is recall only, and it
  is measurable. The corner RMS of found tags is unchanged by construction. Maintenance: the probe
  duplicates upstream's sampling geometry (`patterns[]`, `bit_x`/`bit_y`, `width_at_border`) and
  has to follow it if upstream changes. It needs a reversed-border variant (that path is untested
  in the tree, DESIGN.md:456). Families with small minimum distance (16h5, d=5) need H_probe <= 2,
  so the gain there is the same as `ultrafast`'s.
- **Prior art.** `ultrafast` (shipped as an option, recall cost recorded, PERFORMANCE.md section 4).
  PERFORMANCE.md 8e candidate 1 (border-gradient gate, not tried). Survey C4 ("recall risk unless
  proven conservative"). Nothing rejected it. The new evidence is the Hamming-volume argument
  above: the recall loss of `ultrafast` comes from its Hamming-2 bound, not from gating as such.
- **Verdict: pursue (top item).** How to measure:
  1. Before writing the probe, bound the recall cost with the existing code. Instrument the
     `kUltraFast` probe (`TagDecoder.cpp:137`) to record the unrefined Hamming distance per quad,
     using a brute-force popcount scan in a debug build. Run `apriltag_synth_ground_truth` with
     `exact` and record, for each tag `exact` decodes, its unrefined distance. The histogram gives
     recall as a function of H_probe directly (target: 687/768, i.e. zero loss). On `grayimage` and
     the corpus, the same histogram over rejected quads gives the pass rate.
  2. Then A/B `tag_decode` and `pipeline_total` on the Pi with `pi_bench.py` (ABBA), plus
     `--dump-quads` and the detections diff (expected: identical sets) and
     `bench_multiqueue --streams 4` CPU utilisation (`/proc/stat` or `top`) for the headroom claim.
- **Scope note.** This applies only with `td->refine_edges = true`. That is upstream's default,
  and the validate tool and `bench_multiqueue` use it (`validate_against_libapriltag.cpp:111`,
  `bench_multiqueue.cpp:189`). The sample app turns refinement **off**
  (`apps/apriltag_vulkan/main.cpp:80`). With refinement off, the tail is `quad_decode` plus 75
  plain decodes, and the matching cheap trick is F6. The polarity claim above is in the code:
  `select_blobs.comp:76-78` drops blobs of the wrong border polarity before the CPU sees them.

### F2. Deep CPU idle on RK3588 adds up to 120 us to every wake-up: pool wakes and the fence wake (Pi only)

- **Idea.** The RK3588 device tree gives every core one PSCI idle state, `cpu-sleep`, with
  **entry 100 us, exit 120 us, min-residency 1000 us**. This holds in mainline and in the Rockchip
  6.1 BSP the Pi runs (sources below). `pi_gov.sh perf` pins cpufreq governors and the DMC, but not
  cpuidle, so every timing in the tree was taken with `cpu-sleep` enabled. Any thread that blocks
  for longer than about 1 ms is a candidate for that state, and pays the 120 us exit on wake.
- **Mechanism, tied to the code.** Three wake-ups per frame sit on a critical path:
  1. **Tail pools.** `WorkerPool` workers block on a condition variable between batches
     (`WorkerPool.cpp:58`) and are woken by `notify_all` (`WorkerPool.cpp:89`). In the serial path
     the `quad-worker` pool sleeps through the whole `Detect()` (2.7 ms), and the `tag-worker`
     pool sleeps for about 4 ms. Both are long enough for the menu/TEO governor to pick
     `cpu-sleep`. The caller starts draining immediately as slot 0, so the batch still progresses,
     but for up to about 120 us only one core works. The tail is about 1.5 ms of wall time, and two
     batches each lose up to 120 us x 7/8 of the machine's parallelism. Upper bound: about
     0.2 ms per frame (estimate).
  2. **The fence wait.** The `gpu-driver` thread (or the caller in serial mode) blocks in
     `vkWaitForFences` for roughly 2.3 ms of GPU work, then wakes. If its core entered
     `cpu-sleep`, the readback and the next submission start 120 us late. **That 120 us is GPU
     idle time** in the GPU-bound pipelined mode: up to 4.5% of a 2.62 ms frame (381 fps).
     PERFORMANCE.md 8e rejected "spin on the fence", but measured it **only on the RX 9060 XT
     under Windows**, which has no comparable 120 us idle exit. It is untested on the Pi.
  3. **Pipeline hand-off.** In `FramePipeline::HarvestInFlight` (`FramePipeline.cpp:66-69`) the
     caller sleeps on `cv_` while it waits for the GPU (`pipeline.wait_gpu`). In GPU-bound mode
     that is a long sleep, so the same exit cost lands between the end of one GPU pass and the
     launch of the next (see F3).
- **Expected gain.** Estimate, not measured: 0 to 0.2 ms of serial `pipeline_total` (0-5%), and
  0 to 4% of 1-stream pipelined fps. Whether the governor actually picks `cpu-sleep` at these
  idle lengths depends on its prediction, which is why this is "measure first".
- **Cost and risk.** None for the experiment. Production fixes, cheapest first:
  - Hold a PM-QoS request while the detector runs: open `/dev/cpu_dma_latency` and write an
    int32 below 120 (e.g. 50). That vetoes `cpu-sleep` system-wide while the fd is open. Standard
    Linux API, about 10 lines, costs idle power. Make it a `DetectorConfig` option, off by
    default.
  - Or spin briefly before parking. Workers spin on `generation_` for about 50-200 us before
    calling `wait`, and the fence wait spins on `vkGetFenceStatus` for the last stretch. The
    expected end is known from the previous frame's `submit_wait`.
  - Merging the `quad-worker` and `tag-worker` pools into one (see F4) means the workers that
    just finished `QuadDecode` are hot for `TagDecoder`, which removes one of the two pool wakes
    with no spinning.
  Accuracy: none, bit-identical (scheduling only).
- **Prior art.** 8e "spinning on the fence" was rejected on the desktop only. Nothing in the tree
  mentions cpuidle; `OPTIMIZATION_NOTES.md` "Deployment note: the DRAM controller governor" is
  the same class of finding (a platform power knob dominating measurements).
- **Verdict: measure first. It costs nothing to measure and needs no code.** On the Pi, ABBA with
  `pi_bench.py`, toggling
  `echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state1/disable` (and `0` to restore)
  between runs. Compare `Detect()`, `pipeline_total`, the pipelined `--pipelined` frame time and
  `bench_multiqueue --streams 1`. `cat /sys/devices/system/cpu/cpu*/cpuidle/state1/usage` before
  and after a run shows whether the state is entered at all. In `--trace` output, the start of
  each worker's `pool.drain` relative to the caller's, and how often `pool.drain` has
  `arg == 0` (a worker that woke after the batch was gone), show the pool-wake share directly.
- Sources: mainline `rk3588-base.dtsi`
  (https://github.com/torvalds/linux/blob/master/arch/arm64/boot/dts/rockchip/rk3588-base.dtsi),
  Armbian Rockchip 6.1 BSP `rk3588s.dtsi`
  (https://github.com/armbian/linux-rockchip/blob/rk-6.1-rkr5.1/arch/arm64/boot/dts/rockchip/rk3588s.dtsi):
  both have `CPU_SLEEP` entry 100 / exit 120 / min-residency 1000 us on all 8 cores. PM QoS:
  https://docs.kernel.org/power/pm_qos_interface.html

### F3. FramePipeline: the GPU waits for the caller between passes (all devices; matters on the GPU-bound Pi)

- **Idea.** The next GPU pass starts only after the caller has woken from `wait_gpu`, copied the
  line-fit points out, and notified the `gpu-driver` thread (`FramePipeline::Push`,
  `FramePipeline.cpp:101-118`; the copy is `done_points_.assign(...)` at `:83`). Remove the
  caller from that path: double-buffer the readback (two `last_line_fit_points` and extents
  targets that alternate per frame) so the `gpu-driver` thread can start frame N+1 itself as soon
  as frame N finishes. The caller then only swaps spans: no copy, and no wake on the GPU's
  critical path.
- **Mechanism.** In the pipelined mode the Pi is GPU-bound (381 fps for 1 stream against a
  2.69 ms `Detect()`, PERFORMANCE.md 8g), so every microsecond between `pipeline.gpu_pass` spans
  is lost throughput. The gap is made of: the caller's wake from `cv_` (possibly from `cpu-sleep`,
  F2), the copy of `points x 8 bytes` (on the order of 10^4 points, so about 100-150 KB, roughly
  20-40 us on an A76; estimate), the `notify_one`, and the `gpu-driver` thread's own wake.
  PERFORMANCE.md 5 already records that zero-copy readback and pipelining "are in tension and do
  not compound". Double-buffered readback resolves that tension.
- **Expected gain.** The desktop gap measured "only 20 us between frames" (PERFORMANCE.md 8f).
  The Pi gap has not been measured. If it is 50-150 us (copy plus one or two wakes), removing it
  is +2 to +5% 1-stream pipelined fps on the Pi (estimate). There is nothing to gain with 2 or
  more streams, because other streams fill the GPU's gaps (445 fps either way).
- **Cost and risk.** Moderate: a second set of readback buffers (the line-fit buffer is sized for
  the worst case, so this is device memory on a unified-memory part), cached command buffers per
  readback target (pre-recording keys on buffers, 8g), and an API change to `last_line_fit_points`
  ownership. Bit-identical output.
- **Prior art.** `FramePipeline` (PERFORMANCE.md 5); the 8f gate measured the desktop gap only.
- **Verdict: measure first.** `--pipelined --trace` on the Pi: the gap between consecutive
  `pipeline.gpu_pass` spans on the `gpu-driver` row is the ceiling. Pursue if it exceeds about
  60 us (2%).

### F4. Worker pools: one shared pool, LPT order, and big.LITTLE placement (Pi; bit-identical)

- **Idea.** Three cheap, independent changes:
  (a) `QuadDecode`, `TagDecoder` and `PoseEstimator` each create their own 8-thread pool
  (`QuadDecode.cpp:451`, `TagDecoder.cpp:48`, `PoseEstimator.cpp:671`), and so does every stream
  in `bench_multiqueue`. Use one pool per process (or one per stream). The README already claims
  "Both CPU tails and pose all run on the same `WorkerPool`" (README.md:361), but the code does
  not do that.
  (b) Claim work largest-first: blobs by point count in `QuadDecode`, quads by perimeter in
  `TagDecoder`, since refine cost scales with `nsamples = max(16, edge/8)` per edge
  (`RefineEdges.cpp:65`).
  (c) On RK3588, pin the `gpu-driver` thread and the caller to A76 cores (cpu4-7), and let the tail
  pool use all 8.
- **Mechanism.**
  (a) With 4 streams there are 4 x 2 x 7 = 56 tail workers, plus 4 `gpu-driver` threads and 4
  callers, on 8 cores. Separate pools also mean the `tag-worker` pool is always cold when
  `TagDecoder` starts (F2).
  (b) `WorkerPool` hands out one index at a time from an atomic (`WorkerPool.cpp:44`), so
  heterogeneous cores self-balance, **except at the end**: the last expensive item claimed by an
  A55 (capacity 530 vs 1024 DMIPS/MHz in the DT, and in-order) finishes about 2.5-3x later than
  on an A76. PERFORMANCE.md 8e found "load already balanced" **on 12 identical Zen 3 threads**.
  That does not carry over to big.LITTLE. After F1, the remaining refines are the 1-5 real tags,
  and whether such a refine lands on an A55 sets the tail's wall time. LPT ordering makes those
  items go first, to whichever cores are already awake. Results are merged by index
  (`TagDecoder.cpp:175`, `QuadDecode.cpp:498`), so the output is unchanged.
  (c) Under EAS, a thread that mostly sleeps in a fence wait has low utilisation and is placed on
  a little core. The `gpu-driver` thread's 126 us of host work per `Detect()` (8g) is serial with
  the GPU in the GPU-bound mode. On an A55 it would be roughly 2x longer (estimate).
- **Expected gain.** (a) Removes one cold pool wake per frame (see F2) and the oversubscription in
  multi-stream. (b) Up to one A55 item duration off the tail's wall time: tens of us before F1,
  and a larger share after it (estimate). (c) Up to about 100 us per frame of GPU-critical host
  time, if EAS is in fact placing `gpu-driver` on an A55. Check that first (`ps -eLo
  tid,psr,comm` while it runs, or `pool.*` / `gpu.*` spans in a trace).
- **Cost and risk.** (a) Small refactor: inject a `WorkerPool&`. (b) An index permutation per
  frame (sort 75 or a few hundred keys). (c) `pthread_setaffinity_np` behind a config option,
  Linux-only. All bit-identical. Risk: pinning interacts with the user's own threads, so keep it
  opt-in.
- **Prior art.** The first-pass pipelining attempt swept `APRILTAG_CPU_THREADS` 1-8 (minimum at 6)
  without placement control (OPTIMIZATION_NOTES.md "Pipelining the CPU tail ..."). research
  00-SUMMARY C7 ("pin CPU-tail workers to the A76s") was listed, not measured.
- **Verdict: measure first, then pursue (b) and (a); they are cheap and exact.** Measure with
  `--trace` on the Pi: per-thread `pool.drain` duration and `pool.join_wait` (the caller idling
  on the last item) give the imbalance directly. A/B `taskset -c 4-7` for the whole process
  against the default, and `APRILTAG_CPU_THREADS=4` against 8.

### F5. A bit-identical faster `exact` refine: reuse the far tap, no branches (all devices; secondary once F1 lands)

- **Idea.** Give `RefineEdgesT<double>` the two structural tricks of `kFast` and leave out the
  precision change. That keeps `exact`'s output bit-for-bit.
  1. **Far-tap reuse is exact in double.** At step s the near tap is at offset `n + 1` and the far
     tap at `n - 1`, with `n = -range + 0.25*s` (`RefineEdges.cpp:90-102`). At step s+8,
     `n' - 1 = n + 1`, and every one of these values is a small multiple of 0.25, so both are exact
     in double. The far tap at s+8 therefore evaluates the identical expression
     `sx0 + v*snx - delta` with the identical `v`: same bilinear sample, same bounds test. A
     profile of `max_steps + 8` samples (33 at decimation 2) replaces `2*max_steps` (50), which is
     **34% fewer bilinear evaluations** (eight byte loads each).
  2. **Branch-free accumulation, sequential order kept.** Replace both `continue`s and
     `if (g1 < g2) continue` with a selected weight of +0.0. `Mn + (+-0.0) == Mn` and
     `Mcount + 0.0 == Mcount` for every finite value, including Mn = +0 (+0 + -0 = +0 in
     round-to-nearest), so the sums are unchanged provided the accumulation stays one sequential
     chain per sample. (`kFast`'s 4-lane split would break that.) The removed branch is
     data-dependent, roughly 50/50 on textured rejects, so it mispredicts on A76 and Zen alike.
  3. **Optional SIMD across samples, not across steps.** 2 double lanes (NEON) or 4 (AVX2), each
     lane one sample with its own sequential sum, and the per-sample results added to Mx..Myy in
     sample order afterwards. Bilinear taps stay scalar loads (no gathers on NEON), so expect
     little from this on the Pi (PERFORMANCE.md 8e makes the same point).
- **A precondition worth checking first (correctness, Pi).** "`exact` is bit-identical to upstream
  by construction" (PERFORMANCE.md 4) assumes both sides round the same way. On aarch64, FMA is
  baseline and GCC's default is `-ffp-contract=fast` for C++ and for GNU C. The upstream library is
  built with `-std=gnu17` (`library/CMakeLists.txt:33`), `RefineEdges.cpp` with `-std=c++20`, so
  both contract. They agree only if GCC fuses the same pairs in both translation units. That is
  likely for identical expressions but not guaranteed, and **x86 cannot reveal it** (no FMA in the
  baseline ISA, and MSVC does not contract under `/fp:precise`). Check it on the Pi:
  `APRILTAG_VK_REFINE=upstream` against `exact` with corners printed at `%.17g` (or `--dump-quads`
  after refinement), on the corpus and the synthetic sweep. Any hand-written NEON variant must
  either reproduce GCC's fusion decisions or build both sides with `-ffp-contract=off`, which would
  itself be a (tiny) output change against today's binaries.
- **Expected gain.** Upper-bound proxy, measured in-tree: `fast` (which does items 1-2 *and* drops
  to float) is 1.43 vs 1.79 ms single-thread on x86, -20% of `tag_decode`. On an A76, scalar double
  and float add/mul have the same throughput, so most of that should carry over to a double
  version (estimate). On the Pi, `fast` has not been re-measured since its current form; the
  "nothing on ARM" line in PERFORMANCE.md 4 predates the profile reuse. **After F1** only the
  1-5 real tags are refined, so this falls to about 10-20 us of the per-tag critical path (an
  estimate for one A76 refine at about 50-60 us).
- **Cost and risk.** About 80 lines replacing the body of `RefineEdgesT<double>`. It is exact
  under the precondition above; gate it with the full-precision corner diff, not only RMS.
- **Prior art.** PERFORMANCE.md 8e "Bit-identical refine kernel" (proposed, only lane-wise SIMD
  considered, not tried). `kFast` uses the same tricks non-exactly.
- **Verdict: measure first.** One Pi run of `APRILTAG_VK_REFINE=fast` against `exact` bounds the
  gain before any code is written. If F1 ships, pursue this only for latency.

### F6. Final decode: drop the per-call allocations, and an exact gate for refine-off mode (all devices)

- **Idea.** (a) With `refine_edges` off (the sample app's setting, `main.cpp:80`) the tail is 75
  full decodes at about 4.7 us each (x86, 8e). Running F1's probe at H_probe = 2 in front of them
  rejects about 98% for about 1 us each. It is **exact** if the probe reproduces `quad_decode`'s
  arithmetic in double: `homography_compute2` is public in upstream's `common/homography.c`, so
  the sampling, gray models and sharpen can be ported line for line. A float probe with
  H_probe = 3-4 is near-exact by margin instead. (b) For the quads that do reach
  `quad_decode_index`, the allocations (`quad_copy` and two `matd_copy`, `matd_inverse`, a
  `calloc` of `values`, `sharpen`'s buffer, plus `matd_op` and a detection on success) are about
  6-8 malloc/free pairs per call (patch lines 57, 67; upstream `quad_decode`).
- **Expected gain.** (a) Refine-off `tag_decode`: the Pi figure in the README's older table is
  0.33 ms. Most of it would go: about -0.2 ms per frame on the Pi (estimate). (b) After F1 only
  the few survivors run the real decode, so (b) is worth a few us and **not worth patching
  upstream**. That answers 8e's third candidate.
- **Cost and risk.** (a) The same code as F1's probe. Exactness depends on the port being
  operation-identical, including the FMA-contraction caveat in F5. (b) Upstream patch
  maintenance for negligible gain: reject.
- **Prior art.** PERFORMANCE.md 8e candidate 3 (unprofiled). research/01:208, the pigeonhole
  `quick_decode` in upstream after v3.4.5, which matters for detector creation time and memory,
  not per frame.
- **Verdict:** (a) pursue together with F1 (shared code); (b) reject.

### F7. Moving refine / sampling / decode to the GPU as one batched dispatch: reject (Pi), not worth it (desktop)

- **Idea.** Upload the candidate quads, then one dispatch refines every edge and samples every
  bit, and the results are read back.
- **Why not.**
  1. **The quads are a CPU product.** `QuadDecode` fits them after the readback, so a GPU decode
     needs a second submission per frame (GPU, CPU fit, GPU, CPU). That is a serial round trip
     (19 us empty on Mali, PERFORMANCE.md 3) on the latency path, plus a second queue use per
     frame.
  2. **Mali-G610 has no fp64** (`shaderFloat64 = false`, DESIGN.md:383). `exact` refine and
     upstream's double homography and sampling cannot be reproduced. A float GPU version is at
     best `kFast`-equivalent, which breaks the bit-identity bar against `exact`.
  3. **The Pi's GPU is the throughput limit** (445 fps at 2-4 streams, 8g). Every microsecond
     moved onto it costs multi-stream fps, while the A76/A55 cores are idle enough to absorb the
     tail in pipelined mode.
  4. **F1 shrinks the work to almost nothing.** After gating, the tail's refine and decode work
     is about 75 probes plus 1-5 refines, about 0.1-0.2 ms of CPU. A batched GPU dispatch cannot
     beat that by enough to pay for the extra submission.
  On the desktop (fp64 available, GPU idle 2/3 of the frame) a float or double port could save
  about 0.2 ms of `tag_decode` wall today, but F1 takes most of that for far less work.
- **Same verdict for moving `QuadDecode` (corner fit) to the GPU** on the Pi: it trades CPU time
  that pipelining already hides for GPU time that sets the fps ceiling, and DP seeding is double.
- **Verdict: reject.** Revisit only if a deployment is latency-bound, single-stream and serial,
  on a GPU with fp64 and headroom (the desktop), and F1 has not been done.

### F8. `QuadDecode` (0.44 ms on the Pi, older figure): measure before optimising; known small exact wins

- **What it does per frame.** A serial span scan over all points (`QuadDecode.cpp:467-476`),
  then one parallel task per selected blob (several hundred). Each task does a prefix-sum pass
  and, under the default `kDp`, about 4 more O(n) double passes (`FindDpCornerIndices`, `:166`)
  plus 4 `FitLine`s. The roughly 3% of blobs that fall back (research/02 cites 2.94% at
  decimation 2) run the per-point windowed `FitLineError` (one `hypotf` per point), the 7-tap
  filter, `std::stable_sort` of all peaks and up to 210 combinations. The final serial
  `UpdateFitQuads` handles about 75 quads.
- **Known exact micro-wins (small):**
  - `hypotf` calls are a PLT call each. glibc's `hypotf` is defined as
    `(float)sqrt((double)x*x + (double)y*y)` for finite inputs (`sysdeps/ieee754/flt-32/e_hypotf.c`).
    Inlining exactly that expression is **bit-identical on glibc and overflow-safe** (the float
    squares cannot overflow a double). That removes the objection on record in OPTIMIZATION_NOTES
    "`sqrtf(a*a+b*b)` instead of `hypotf` - rejected" (overflow cliff), which is new evidence. But
    under `kDp` `hypotf` is only hot in fallback blobs, so the old 0.29 ms (8%, measured when
    `kPeaks` was the default) no longer applies. Expect at most a few percent of `quad_decode`.
    MSVC's `hypotf` is a different implementation, so the bit-identity argument covers
    Linux/glibc only.
  - In the fallback, `std::stable_sort(peaks)` (`:351`) allocates a merge buffer per call and sorts
    every local maximum, only to keep the top 10. A fixed 10-slot insertion keyed on
    (value, index) selects exactly the same 10 in the same order: bit-identical, no allocation.
  - Folding the farthest-from-p0 search into the prefix-sum loop saves one O(n) pass:
    bit-identical, tiny.
- **The real unknown is wall time against CPU time.** 0.44 ms of wall time across 8 threads for a
  few hundred light O(n) tasks is high. Candidates are F2 (cold pool wake) and F4b (one large
  blob claimed late by an A55), not arithmetic.
- Source for the glibc claim: https://github.com/bminor/glibc/blob/master/sysdeps/ieee754/flt-32/e_hypotf.c
  (`float r = math_narrow_eval ((float) sqrt ((double) x * (double) x + (double) y * (double) y));`).
  Both squares are exact in double, so FMA contraction of the sum cannot change the result.
- **Verdict: measure first.** `--profile`/`--trace` on the Pi: the `quad.fit_blob` spans carry
  the blob's point count as `arg`, so the trace shows the largest blob's duration, which core ran
  it, worker start skew, and `pool.join_wait`. Do the micro-wins only if the trace shows compute,
  not waiting.

### F9. Per-edge parallel refine for single-tag latency (after F1; bit-identical)

- **Idea.** `RefineEdgesT` fits the four edges independently: each edge reads only the two
  original corners, and corners move only in the final `IntersectLines` (`RefineEdges.cpp:51-148`).
  Make (quad, edge) the work item for quads that pass the F1 gate, then intersect.
- **Expected gain.** Once F1 has removed the rejects, a 1-tag frame's tail critical path is
  roughly one refine (about 50-60 us on an A76, estimate) plus one decode. Splitting by edge
  divides the refine by up to 4 across cores that are otherwise idle: about -30 to -40 us of
  serial latency on the Pi, about 1% of `pipeline_total`. Nothing in pipelined mode.
- **Cost and risk.** Moderate (a second `ParallelFor` phase, or a two-level work list). It is
  exact, since each edge's arithmetic is unchanged. Pool dispatch overhead (and F2) can eat the
  gain.
- **Verdict: low priority. Measure after F1 and F2.**

### F10. Pose estimation is not on the default path

`TagDecoder` stops at 2D detections (`TagDecoder.h:21`), the sample app does not wire pose
(`main.cpp:148`), and `PoseEstimator` is already 0.016 ms per tag on the Pi (README.md:348). It
does create its own 8-thread pool (`PoseEstimator.cpp:671`), which belongs in F4a's shared pool.
Nothing else to gain here.
