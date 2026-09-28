# Research brief (shared by every category agent)

Date: 2026-09-26. Goal: cover as much ground as possible on how to speed up
vkapriltag, one category per agent, writing findings to disk as you go.

## Persistence rules (the reason this file exists)

Earlier research runs were killed by API rate limits and lost everything
they hadn't written down.

1. **Write your category file after every source you process**, not at the
   end. Append one finished entry right after reading a source.
2. **Keep a `## Status` section at the top of your category file**: sources
   covered so far, and a "next up" queue. Update it after each entry, so a
   fresh agent can resume from the file alone.
3. **On start, read your category file.** If entries already exist, skip
   those sources and continue from its "next up" queue.
4. **Breadth before depth.** Get 8–10 sources in first, then deepen.
5. **Touch only your own category file** (plus downloads under the cache
   directory). Other agents are writing the other files at the same time.
6. **Never invent citations or numbers.** Tag every entry
   `[verified from primary source]` or `[secondary/unverified]`. If you
   couldn't open a source, say so.
7. **Finish with a `## Conclusions` section** at the end of your file:
   - the ranked concrete ideas for vkapriltag from this category, each with
     expected uplift, accuracy/determinism risk, device (Mali / RDNA /
     both), and how to bound it cheaply before building;
   - what you couldn't verify.

## Entry format

```
### <Title> — <authors/org>, <date>
- URL:
- Technique: (2-5 sentences)
- Reported speed-up: (exact baseline + hardware)
- Accuracy / determinism impact:
- Applicability to vkapriltag: (concrete; already done / rejected in-tree?; Mali vs RDNA)
- Verification: [verified from primary source] / [secondary/unverified]
```

## Cached material from earlier runs — reuse it, don't re-download

Location: `C:\Users\yojob\AppData\Local\Temp\claude-research\` (about 1 GB).
Put new downloads under the same tree.

- **PDFs:** `curl`, then `pdftotext -layout` (Git Bash: `/mingw64/bin/pdftotext`).
- **gpuinfo reports:** `curl -s -A "Mozilla/5.0" "https://vulkan.gpuinfo.org/displayreport.php?id=<ID>"`, then `grep -o 'VK_[A-Z0-9]*_[a-z0-9_]*'`.

### `ccl\`
- Paper texts:
  - Bolelli 2024 TPDS GPU-CCL review (`bolelli2024tpds.txt`)
  - Chen 2018 coarse-to-fine
  - Windisch 2023
  - HA4 GTC2019
  - multi-value CCL (`multivalue1402`)
  - hybrid-pixel 2024
  - `generalized1603`
- `yacclab\`: CUDA sources — KE4, RADAR, TUF, UF_InlineCompression, Allegretti 2018, Cabaret 2017, HA4.
- `graph\`:
  - paper texts: ConnectIt, ECL-CC, FastSV, GConn, IISWC24, Liu-Tarjan
  - sources: cuGraph WCC (`cugraph_wcc.cuh`), GAPBS `cc.cc` (Afforest)
- `hash\`:
  - hash tables: SlabHash, WarpCore, WarpSpeed, BGHT, Karnagel, Lessley, McKee, plos25
  - thresholding: `binarize1905`
  - `arm_bp` (Arm best practices)
  - AprilTag 2 and ArUco3 texts
- `scansort\`:
  - Merrill decoupled look-back, Onesweep, Hou segmented sort
  - forward-progress paper (`progress2109`)
  - "decoupled fallback" paper `.tex`
  - GPUSorting mobile notes (`gs_mobile`)

### `fiducial\`
- Git clones: `RealtimeRoboticsGroup_aos`, `frc971_bos`, `Team766_apriltags_cuda`, `FRC-Team-4143_GpuDetectorJNI`, AprilRobotics `apriltag`.
- AprilTag 2/3 paper texts.
- NVIDIA VPI AprilTag performance data: `perf_*_apriltag.json`, `vpi_apriltags.html`.
- Subfolders: `cv\`, `fpga\`, `learned\`.

### `mali\`
- libmali g24p0 and g29p1 extension lists, extracted from the driver binaries: `g24p0.ext.txt`, `g29p1.ext.txt`.
- gpuinfo report extension lists: `ext_*.txt`, `e*.txt`.
- Local RX 9060 XT list: `e_rdna4.txt`, `vi_local.txt`.
- Arm GPU Best Practices 3.4 text: `bp34.txt`.
- GitHub issue comment dumps: JeffyCN mirrors #21, ubuntu-rockchip #617/#888, photon-image-modifier #161.
- vulkaninfo pastebins: `pb_*.txt`.
- mali-vulkan-icd-wrapper docs: `wr_*.md`.

## Project context

vkapriltag (repo `C:\Users\yojob\Projects\vkapriltag`) is a Vulkan-compute
port of the AprilTag 3 detector. It is derived from FRC 971's CUDA detector
via Team766/apriltags_cuda, and targets PhotonVision.

For detail, read:
- `apriltags_vulkan/OPTIMIZATION_NOTES.md`
- `apriltags_vulkan/PERFORMANCE.md`
- `vkapriltag-speedup-survey.md` at the repo root
- shaders in `apriltags_vulkan/library/shaders/`

**Targets:**
- Orange Pi 5 / RK3588 / Mali-G610 MP4 (libmali, unified memory): GPU phase 3.76 ms at 1280x800 decimation 2, CPU tail about 1.2 ms.
- Desktop RX 9060 XT (RDNA4): 0.66 ms.

**Correctness gate:** bit-identical to upstream libapriltag.

**GPU pipeline:**
1. decimate (point sample)
2. 4x4 block min/max, then 3x3 filter
3. three-valued threshold: 0 / 255 / 127 (ambiguous)
4. union-find CCL, 4-connected:
   - `uf_init` pre-joins horizontal runs.
   - Then about 2 iterations of `uf_merge` + `uf_compress`. `uf_merge` unions only at run-overlap starts, walks `find()`, and hooks with atomicMin, so root = min index. That property is required downstream.
5. `uf_final` / `label_pixels`
6. `blob_diff`: boundary points, atomic append
7. `hash_group`: open addressing by blob pair
8. `reduce_extents_hash`: 8-way privatized, int64 atomics
9. `select_blobs`, then scan + scatter
10. `sort_points_local`: per-blob odd-even mergesort + line-fit moments

**CPU:** quad fit, upstream tag decode, optional `refine_edges`, pose.
Frame pipelining overlaps the CPU tail with the next frame's GPU work.

**Mali facts:**
- No dedicated shared memory (it is backed by L2).
- Subgroup size fixed at 16. Subgroup reduce-by-key is 6x slower than plain atomics.
- Only about 15% of GPU time is DRAM-bandwidth bound.
- Labelling (about 27% of GPU time) is limited by dependent loads.
- Extents and sort are limited by atomics and latency.
- Barrier cost is 2.6–18.7 µs, scaling with in-flight work.

**Already rejected in-tree** (note them, but don't re-propose without new
evidence):
- tile-local shared-memory union-find
- BUF / BKE / Playne 2x2-block CCL (needs binary 8-connected input)
- subgroup aggregation on Mali
- fusing the preprocessing passes
- 4 px/thread vectorization of `uf_merge` / `blob_diff`
- fp16 coordinates
- tiled image / `texelFetch`
- precomputed gradient W
- submit collapsing on dGPU
- 16-bit storage, sync2, BDA, push descriptors (for speed)

**Already found:**
- Path halving in `find()` (ECL-CC intermediate pointer jumping) measured labelling −17% at decimation 1 and −2.7% at decimation 2 on the RX 9060 XT, bit-identical.
- `VK_EXT_conditional_rendering` for the converged `uf_compress` measured −3% labelling on RDNA. Not available on Mali.
- SpectrumJetson: frc971's newer detector is 30–40% faster than FRC-4143's Aug-2024 copy. There the GPU is about 12% busy and JPEG decode is the real bottleneck.
- rapidtag: big.LITTLE core pinning.
- ArUco3: multi-scale + temporal, up to 40x.
- YoloTag: 55 vs 24 fps.
