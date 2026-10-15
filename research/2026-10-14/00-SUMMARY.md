# Research round 2026-10-14: combined summary

Six parallel surveys (brief: `_BRIEF.md`). Nothing in this round was built or timed; every gain is an
estimate from the tree's own numbers unless marked *measured*. Per-area detail, sources and the
reasoning behind each verdict are in the category files:

| File | Area | Headline |
| --- | --- | --- |
| `01-gpu-kernels-agnostic.md` | GPU stages, any device | ~-10% of Mali GPU time from 6 exact changes, if the bounds confirm |
| `02-rdna4-desktop.md` | RX 9060 XT | multi-stream: use both queue families (+70% measured in an older table) |
| `03-mali-rk3588.md` | Mali-G610 / RK3588 board | thread placement + governors, ingest, GPU power-down at paced rates |
| `04-graph-optimization.md` | Per-frame dispatch graph, CCL, compilers | labelling-tail re-plan; barrier cost never actually measured |
| `05-cpu-tail.md` | Refine / decode / threading | probe-before-refine: -13 to -15% of Pi `pipeline_total` |
| `06-tensor-and-rt-cores.md` | WMMA, int dot, RT units | no tangible benefit on either device; arithmetic per stage |

## Step zero (blocks most rankings)

1. **Record a current per-span `--profile --trace` table on the Orange Pi.** The tree has none for
   the current code; `hash_group`, `scatter` and blob-scan were never recorded on Mali, and the CPU
   tail (~1.5 ms) has no `quad_decode` / `tag_decode` split there. (01, 05)
2. **Clean the Pi's measurement state.** `lumenvision.service` (`/opt/lumenvision/Server`) runs at
   boot and uses ~2 CPU cores (checked 2026-10-14: 205% CPU, no GPU mapping, GPU load 0). It was
   running during the PERFORMANCE.md 8g measurements, so those GPU-side numbers stand but the
   CPU-tail and `pipeline_total` numbers carry CPU contention (A/B pairs were interleaved, so the
   deltas hold). The A55 policy is also left at min = max = 1.8 GHz (not set by `pi_gov.sh`, which
   only touches governors). Stop the service and record the clocks before any A/B. (03)
3. **Barrier-cost split test.** The existing inter-span gap metric cannot see barriers (they sit
   before the span-end timestamp), so "intra-submit fusion has ~0.04 ms left" is unproven. Record one
   large dispatch as two halves with and without a barrier, ABBA, timestamps off. Decides the value of
   every fusion item below. (04)

## Ranked options

### A. Orange Pi (deployment target)

| # | Option | Est. gain (Pi) | Exact? | Verdict | Source |
| --- | --- | --- | --- | --- | --- |
| A1 | Probe the unrefined quad's bits; refine only if within Hamming 4-5 of a code | -0.55 to -0.65 ms serial (-13 to -15% `pipeline_total`); 1.3-1.8 cores freed | kept detections bit-identical; recall loss possible, measure on the synth sweep | pursue | 05 F1 |
| A2 | Thread placement: driver thread + tail on A76, governors/`ondemand` behaviour at camera rates, one shared worker pool instead of three per stream, largest-first work order | 1-3 ms latency at paced rates (governor), tens of us to 0.1 ms (pools) | yes | pursue (no-code test first: `performance` + `taskset -c 4-7`, `--fps 60`) | 03 #1, 05 F4 |
| A3 | Wake-up latency: `cpu-sleep` idle state (120-220 us exit), GPU IRQs on an A55, fence wake | tens of us to ~0.3 ms; 0-4% 1-stream fps | yes | measure first (toggle idle state, IRQ affinity, then wait-then-spin) | 03 #4, 05 F2 |
| A4 | GPU power-down between paced frames (`coarse_demand`, cores off after 800 us) | unknown; benchmarks never see it | yes | measure first (`power_policy=always_on` vs default at 30/60 fps) | 03 #3 |
| A5 | Labelling tail re-plan: counting in the last real `uf_compress`, drop guarded compress passes, per-merge flag word | -0.06 to -0.09 ms (2.5-3.5% GPU) | yes | pursue | 04 F2, 01 #4 |
| A6 | `blob_diff` append: one atomic per pixel/thread instead of per point | up to -4 to -5% GPU if the counter is the cost | yes | measure first (no-atomic bound) | 01 #1, 03 |
| A7 | Delete `label_pixels` (blob_diff reads threshold codes first) | -2 to -3% GPU | yes | pursue | 01 #2, 04 F6 |
| A8 | Rank-by-counting sort for small blobs (no barriers, no shared memory) | 0-5% GPU | yes (unique keys) | measure first (blob-size histogram) | 01 #3, 03 |
| A9 | Clears overlap the chain head (defer one barrier) | up to 1.5-2% GPU | yes | measure first | 04 F3 |
| A10 | Dispatch diet: fold `init_extents`, `merge_extents`, the count scan, `build_*_args` | Mali 1-2%, desktop 4-6% | yes | measure first, desktop first | 04 F4, 01 #7 |
| A11 | Run-merge before 64-bit atomics in `reduce_extents_hash` / `scatter`; extents copy keyed by `gl_CoreIDARM` (popcount rank, mask 0x50005) | -2 to -3% GPU | yes | measure first | 01 #5, 03 |
| A12 | Ingest: MPP hardware MJPEG -> dma-buf import, or gray-only libjpeg-turbo straight into the mapped frame | several ms of A76 decode (unmeasured) + 0.19 ms copy (*measured*) | yes | pursue in the calling app; library needs stride/offset on `ImportDmaBufFrame` | 03 #2 |
| A13 | Double-buffer the `FramePipeline` readback so the GPU thread chains frames | +2 to +5% 1-stream fps | yes | measure first (gap between `pipeline.gpu_pass` spans) | 05 F3 |

Not additive: A5/A7/A10 and A6/A11 touch the same shaders; a second concurrent stream already
recovers ~0.25 ms per frame of GPU idle on Mali, which caps every intra-frame scheduling gain (04).

**Tooling:** the kernel has `CONFIG_MALI_BIFROST_GATOR_SUPPORT=y`, so Streamline / gatord and
libGPUCounters should work on this board (PERFORMANCE.md's "no external profiler" is wrong here);
g24p0 lacks `VK_KHR_pipeline_executable_properties`, so register counts come from `malioc`. (03 #5)

**Correctness check to run on the Pi:** diff `APRILTAG_VK_REFINE=upstream` against `exact` corners
at full precision; aarch64 GCC contracts multiply-adds by default and x86 cannot reveal a mismatch. (05)

### B. RX 9060 XT (desktop)

| # | Option | Est. gain | Verdict | Source |
| --- | --- | --- | --- | --- |
| B1 | Spread streams across both queue families instead of all on family 0 | 4,600 -> 5,200 (compute family) / 7,800 fps (alternating, with import), *measured* in older tables | pursue | 02 F2 |
| B2 | Prefer the compute-only family for a single stream (`Context.cpp:468-470`) | +4 to +24% (older tables) | pursue after one ABBA | 02 F1 |
| B3 | Split the ~115 us submit-to-wake round trip with `VK_AMD_buffer_marker` + calibrated timestamps | ceiling 20-28% of 1-stream frame | measure first | 02 F3 |
| B4 | Device-local copy of `selected[]` for its three GPU readers (now read across PCIe) | ~40 us GPU (attribution suspect) | measure first at HEAD | 02 F6 |
| B5 | Bounded union-find chains (labelling is 52% of GPU here vs 27% on Mali) | 15-40 us | measure first (`shader_clock` depth histogram) | 02 F4 |

Single-stream frame (~333 us): ~115 us launch/completion latency, ~217 us of command buffer on
one pipe of which ~90 us is idle GPU, a few us host. 27 dispatches, 10 fills, 7 copies, ~31-33 barriers.

## Rejected this round (with the evidence)

- **Matrix units / WMMA** (RDNA4: 18 shapes, all 16x16x16 f16/bf16/fp8/int8; no 1/4-bit): no stage
  is a dense MAC product short of ALU; matrix forms that exist are slower (labelling closure ~1.3 ms
  vs 172 us; rank sort ~40x comparisons). Mali-G610 has no matrix unit, no RT, no ARM tensors. (06)
- **Integer dot product:** no 8-bit dot-product operand anywhere in the shaders or tail. (06)
- **Ray-tracing units:** published wins need large static sets and spatial pruning, all on NVIDIA;
  any BVH here is rebuilt per frame and built by compute shaders on AMD. (06)
  *Would change:* large families (52h13 / 49h12) make GPU brute-force decode worthwhile with plain
  popcount; soft-decision decoding over such a family is the one WMMA win (~8x), giving up parity.
- **Work graphs** (`VK_AMDX_shader_enqueue` not exposed on driver 26.8.1), **DGC** (cannot switch
  compute pipelines), **megakernels** (need forward progress Vulkan does not guarantee). (02, 04)
- **Smart Access Memory** (off; VRAM-mapped types are uncached for CPU reads anyway). (02)
- **ML-compiler auto-tuning** (TVM / IREE / XLA cannot express this code); per-pipeline workgroup
  sweeps via specialization constants are cheap but ~0-2%. (04)
- **RK3588 NPU** (no stage fits; 4-8 ms per-call floor vs 2.7 ms `Detect()`), **RGA** (averaging
  scale-down is not upstream's point-sampled decimation; adds an ioctl round trip). (03)
- **GPU refine/decode batch** (needs a second submission; no fp64 on Mali; Pi GPU is the limit). (05)
- **CCL literature:** nothing new since research/03-04; labelling already converges in the minimum
  two passes, so only per-pass cost remains. (04)

## Suggested order

1. Step zero (Pi profile with the service stopped and clocks recorded; barrier split test).
2. No-code Pi tests: A2 governor/affinity, A3 idle states and IRQ affinity, A4 power policy.
3. A1 probe-before-refine (largest single CPU item), with the synth-sweep recall check.
4. B1 + B2 on the desktop (small code, measured upside).
5. GPU shader bundle on the Pi in order of confidence: A5, A7, then the measure-first items.
