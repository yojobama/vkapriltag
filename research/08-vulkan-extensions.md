# 08 - Vulkan extensions availability and use

Entry format: Title / URL / Technique / Reported speed-up / Accuracy-determinism / Applicability to vkapriltag / Verification.
Cache root: C:/Users/yojob/AppData/Local/Temp/claude-research/

## Status

- Covered: cached lists (libmali g24p0/g29p1 strings, gpuinfo 37373/41734/46182/50479/51129/51727, pastebins g6p0, RDNA4 vulkaninfo) -> master table below.
- Entries done: external_memory_host, dma_buf family, g29p1 wrapper evidence, subgroup_size_control, pipeline_executable_properties, AMD_shader_info, shader_clock, global_priority, conditional_rendering, DGC, descriptor_buffer/push_descriptor/maintenance5-9, cooperative_matrix + integer_dot_product, 8/16-bit + fp16, robustness, pipeline_binary, memory_priority, copy_memory_indirect (+ RDNA4 fused-submit gate), timeline/sync2/host_query_reset/calibrated_timestamps, ARM shader_core_builtins, ARM scheduling_controls, ARM tensors/data_graph.
- Feature-bit source for later entries: `python mali/props.py rep_<id>.html '<regex>'` (gpuinfo) and `grep` on `mali/vi_local.txt` (RDNA4).
- Also done: ARM shader_instrumentation/counters_by_region, atomic_int64 bits + atomic_float, workgroup_memory_explicit_layout, reconvergence/rotate/quad/partitioned.
- Also done: AMD_* set + AMDX_shader_enqueue, shader_object, video decode/MJPEG.
- Conclusions finalized (28 entries). Optional follow-up: NVIDIA/Intel availability spot-check; `vulkaninfo` on the Pi under g29p1.

## 0. Master availability table

### How each column was obtained (read this before trusting a "Y")

| Column | Device / driver / OS | Source | Kind |
| --- | --- | --- | --- |
| g6p0 Lnx | Mali-G610 ("Mali-LODX"), libmali `v1.g6p0-01eac0`, API 1.2.165, Linux Wayland (Rock 5C, ginkage) | `mali/pb_35H2dvsR.txt` (67 dev ext), `pb_NYD1z027.txt` (64; same driver, lacks 3 WSI exts) | real `vulkaninfo` |
| g18p0 And | Mali-G610, `v1.g18p0-01eac0`, API 1.1.231, **Android**, Radxa rock5b_gen2 | gpuinfo **46182** (`rep_46182.html`, `ext_46182.txt`, 100) | real report |
| g24p0 str | libmali g24p0 binary (`g24p0-00eac0`, rk_so_ver 10) | `mali/g24p0.ext.txt` (149) | **`strings`** |
| g24p0 Lnx rpt | Mali-G610, `v1.g24p0-00eac0`, API 1.3.276, Linux (Armbian/Ubuntu) | gpuinfo **41734** (`rep_41734.html`, 132) | real report - **this is the Orange Pi deployment driver** |
| g29p1 str | libmali g29p1 binary (`g29p1-11eac1`, rk_so_ver 19, from `libmali-valhall-g610-g29p1_1.10-1_arm64.deb`) | `mali/g29p1.ext.txt` (170) | **`strings`**, no real report exists |
| r38p1 And | Mali-G610 MC4, `v1.r38p1-01eac0`, API 1.3.219, Android | gpuinfo **37373** (downloaded this run: `rep_37373.html`; parse.py finds 97, the survey said 110) | real report |
| r54p2 And | Mali-G610, `v1.r54p2-00eac0`, API 1.4.305, Android (HUAWEI CHZ-AL00) | gpuinfo **50479** (156) | real report |
| panvk 25.0 | Mali-G610, Mesa panvk 25.0.7 (Debian 13) | gpuinfo **51129** (71) | real report |
| panvk 26.3 | Mali-G610 MC4, Mesa 26.3.0-devel (Ubuntu) | gpuinfo **51727** (181) | real report |
| RDNA4 | RX 9060 XT, AMD proprietary 26.8.1 (LLPC), API 1.4.349, Windows 11 | `mali/vi_local.txt` (local `vulkaninfo`), `e_rdna4.txt` | real, **221** device extensions (the survey's "239" is wrong) |

(`r15490.html` is a Mali-**G52** r25p1 report - not a G610, ignored.)
Set files and generator: `claude-research/mali/sets/*.ext`, `sets/table.py`, `sets/list.txt`.

**Caveat on the libmali "str" columns.** I reproduced both lists exactly with
`strings -n 8 <lib>.so | grep -E '^VK_[A-Z0-9]+_[A-Za-z0-9_]+$' | sort -u`
(149 and 170 lines, zero diff). Comparing g24p0-strings with the real g24p0
report 41734 shows what that method gets wrong:
- **False positives (in binary, not exposed on G610):** instance extensions
  (`VK_KHR_surface`, `display`, `get_physical_device_properties2`,
  `external_*_capabilities`, `device_group_creation`,
  `get_surface_capabilities2`, `EXT_debug_report`, `EXT_debug_utils`,
  `EXT_headless_surface`) and features of *other* Valhall/5th-gen parts that one
  binary serves: `KHR_acceleration_structure`, `ray_query`,
  `ray_tracing_pipeline`, `pipeline_library`, `deferred_host_operations`,
  **`KHR_cooperative_matrix`**, `KHR_fragment_shading_rate`,
  `EXT_conservative_rasterization`.
- **False negatives:** `EXT_swapchain_maintenance1`, `KHR_present_id` are exposed
  but not present as standalone strings (WSI layer).
- So a g29p1-strings "Y" means *the driver knows the name*; exposure on the G610
  is **unconfirmed**. The best proxy is r54p2 Android (a newer DDK for the same
  GPU): of the 21 names g29p1 adds over g24p0, r54p2 exposes 17 on G610
  (all but `conditional_rendering`, `memory_type_properties` and the two
  `ray_tracing_*` names). **`VK_EXT_conditional_rendering`
  is therefore likely NOT exposed on G610 even under g29p1** (it is also absent
  from r54p2 on the same GPU) - must be checked with `vulkaninfo` on the Pi.
- Feature bits matter as much as names: g24p0 exposes
  `VK_ARM_scheduling_controls` but reports `schedulingControls = false`;
  r38p1 Android reports `shaderInt64 = false`, `shaderBufferInt64Atomics = false`
  (so the adopted int64-atomic extents path falls back there).

### Table (Y = listed; for "str" columns see the caveat)

| Extension | g6p0 Lnx | g18p0 And | g24p0 str | g24p0 Lnx rpt | g29p1 str | r38p1 And | r54p2 And | panvk 25.0 | panvk 26.3 | RDNA4 26.8.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `VK_EXT_external_memory_host` | - | - | - | - | Y | - | Y | - | - | Y |
| `VK_EXT_external_memory_dma_buf` | Y | Y | Y | Y | Y | Y | Y | Y | Y | - |
| `VK_KHR_external_memory_fd` | Y | Y | Y | Y | Y | Y | Y | Y | Y | - |
| `VK_EXT_image_drm_format_modifier` | Y | Y | Y | Y | Y | Y | Y | Y | Y | - |
| `VK_EXT_queue_family_foreign` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_ANDROID_external_memory_android_hardware_buffer` | - | Y | - | - | - | Y | Y | - | - | - |
| `VK_EXT_subgroup_size_control` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_pipeline_executable_properties` | - | - | - | - | Y | - | Y | Y | Y | Y |
| `VK_AMD_shader_info` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_shader_clock` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_KHR_global_priority` | - | - | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_global_priority` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_global_priority_query` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_conditional_rendering` | - | - | - | - | Y | - | - | - | Y | Y |
| `VK_EXT_device_generated_commands` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_NV_device_generated_commands_compute` | - | - | - | - | - | - | - | - | - | - |
| `VK_EXT_descriptor_buffer` | - | - | - | - | Y | - | Y | - | - | Y |
| `VK_KHR_push_descriptor` | - | - | Y | Y | Y | - | Y | Y | Y | Y |
| `VK_KHR_buffer_device_address` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_inline_uniform_block` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_maintenance4` | - | - | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_maintenance5` | - | - | Y | Y | Y | - | Y | - | Y | Y |
| `VK_KHR_maintenance6` | - | - | Y | Y | Y | - | Y | - | Y | Y |
| `VK_KHR_maintenance7` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_KHR_maintenance8` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_KHR_maintenance9` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_KHR_maintenance10` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_cooperative_matrix` | - | - | Y | - | Y | - | - | - | - | Y |
| `VK_KHR_shader_integer_dot_product` | - | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_8bit_storage` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_16bit_storage` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_shader_float16_int8` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_robustness2` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_KHR_robustness2` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_EXT_pipeline_robustness` | - | Y | Y | Y | Y | - | Y | Y | Y | - |
| `VK_EXT_image_robustness` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_memory_priority` | - | - | - | - | - | - | - | - | - | Y |
| `VK_EXT_pageable_device_local_memory` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_pipeline_binary` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_EXT_pipeline_creation_cache_control` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_pipeline_creation_feedback` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_EXT_calibrated_timestamps` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_calibrated_timestamps` | - | - | Y | Y | Y | - | Y | - | Y | Y |
| `VK_EXT_host_query_reset` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_timeline_semaphore` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_synchronization2` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_copy_memory_indirect` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_NV_copy_memory_indirect` | - | - | - | - | - | - | - | - | - | - |
| `VK_KHR_shader_atomic_int64` | - | Y | Y | Y | Y | - | Y | - | Y | Y |
| `VK_EXT_shader_atomic_float` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_EXT_shader_atomic_float2` | - | - | - | - | - | - | - | - | - | Y |
| `VK_EXT_shader_image_atomic_int64` | - | Y | Y | Y | Y | - | Y | - | Y | Y |
| `VK_KHR_workgroup_memory_explicit_layout` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_KHR_zero_initialize_workgroup_memory` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_shader_maximal_reconvergence` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_KHR_shader_subgroup_uniform_control_flow` | - | - | Y | Y | Y | - | Y | - | Y | Y |
| `VK_KHR_shader_subgroup_rotate` | - | - | - | - | Y | - | Y | Y | Y | Y |
| `VK_KHR_shader_quad_control` | - | - | Y | Y | Y | - | Y | - | Y | Y |
| `VK_KHR_shader_subgroup_extended_types` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_EXT_shader_subgroup_ballot` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_shader_expect_assume` | - | - | Y | Y | Y | - | Y | Y | Y | Y |
| `VK_KHR_shader_float_controls2` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_EXT_scalar_block_layout` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `VK_KHR_vulkan_memory_model` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_compute_shader_derivatives` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_EXT_host_image_copy` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_EXT_map_memory_placed` | - | - | - | - | - | - | - | - | Y | - |
| `VK_KHR_map_memory2` | - | - | Y | Y | Y | - | Y | Y | Y | Y |
| `VK_EXT_memory_budget` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_EXT_memory_type_properties` | - | - | - | - | Y | - | - | - | - | - |
| `VK_EXT_external_memory_acquire_unmodified` | - | - | Y | Y | Y | - | Y | - | Y | - |
| `VK_EXT_device_fault` | - | - | Y | Y | Y | - | Y | - | - | Y |
| `VK_EXT_device_memory_report` | - | Y | Y | Y | Y | Y | Y | - | Y | - |
| `VK_EXT_frame_boundary` | - | - | Y | Y | Y | - | Y | - | - | Y |
| `VK_EXT_shader_object` | - | - | - | - | - | - | - | - | - | Y |
| `VK_EXT_mutable_descriptor_type` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_KHR_video_queue` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_video_decode_queue` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_video_decode_h264` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_video_decode_av1` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_video_encode_queue` | - | - | - | - | - | - | - | - | - | Y |
| `VK_KHR_sampler_ycbcr_conversion` | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| `VK_KHR_shader_untyped_pointers` | - | - | - | - | - | - | - | - | Y | Y |
| `VK_EXT_shader_replicated_composites` | - | - | - | - | Y | - | Y | - | Y | Y |
| `VK_KHR_shader_relaxed_extended_instruction` | - | - | - | - | - | - | - | Y | Y | - |
| `VK_KHR_shader_bfloat16` | - | - | - | - | - | - | - | - | - | Y |
| `VK_EXT_shader_float8` | - | - | - | - | - | - | - | - | - | Y |
| `VK_ARM_scheduling_controls` | - | - | Y | Y | Y | - | Y | - | Y | - |
| `VK_ARM_shader_core_builtins` | - | - | Y | Y | Y | - | Y | - | Y | - |
| `VK_ARM_shader_core_properties` | - | - | Y | Y | Y | - | Y | - | Y | - |
| `VK_ARM_render_pass_striped` | - | - | - | - | - | - | - | - | - | - |
| `VK_ARM_tensors` | - | - | - | - | - | - | - | - | - | - |
| `VK_ARM_data_graph` | - | - | - | - | - | - | - | - | - | - |
| `VK_ARM_pipeline_opacity_micromap` | - | - | - | - | - | - | - | - | - | - |
| `VK_ARM_format_pack` | - | - | - | - | - | - | - | - | - | - |
| `VK_AMD_anti_lag` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_shader_core_properties` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_shader_core_properties2` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_buffer_marker` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_memory_overallocation_behavior` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_device_coherent_memory` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_shader_ballot` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMD_pipeline_compiler_control` | - | - | - | - | - | - | - | - | - | - |
| `VK_AMD_gcn_shader` | - | - | - | - | - | - | - | - | - | Y |
| `VK_AMDX_shader_enqueue` | - | - | - | - | - | - | - | - | - | - |
| `VK_EXT_shader_module_identifier` | - | - | - | - | - | - | - | Y | Y | Y |
| `VK_EXT_debug_utils` | - | - | Y | - | Y | - | - | - | - | - |
| `VK_EXT_tooling_info` | - | Y | Y | Y | Y | Y | Y | Y | Y | Y |

## 1. Ingest: zero-copy frame import

### VK_EXT_external_memory_host — Khronos (AMD/others), 2017; refpage read 2026-09-26
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_host.html
- Technique: `VkImportMemoryHostPointerInfoEXT` wraps an existing host allocation (or host-mapped foreign device memory) as a `VkDeviceMemory`; bind a `VkBuffer` (storage usage) to it and `decimate.comp` reads the caller's frame in place. Pointer **and** size must be multiples of `minImportedHostPointerAlignment`; `vkGetMemoryHostPointerPropertiesEXT` returns the memory types usable for that pointer (implementation-dependent). The host memory must not be freed while the `VkDeviceMemory` lives; normal host/device synchronization applies; importing the same pages twice is not guaranteed to work.
- Reported speed-up: none published for this workload. In-tree number to bound it: `upload` = **0.22 ms** on the G610 at 1280x800 dec 1 (OPTIMIZATION_NOTES.md summary table) - the 1 MB `gray_buf_.Write()` memcpy into `DeviceLocalMapped` memory (Mali) or into `upload_staging_` + a `vkCmdCopyBuffer` (RX 9060 XT).
- Accuracy / determinism impact: exact (same bytes).
- Applicability to vkapriltag:
  - **Availability:** RX 9060 XT yes (`minImportedHostPointerAlignment = 0x1000`). Mali-G610: **not** exposed by g24p0 (report 41734 - the deployment driver); present in the g29p1 binary strings; exposed by Android r54p2 on the same GPU with alignment **4096**. So on the Pi it needs the g29p1 upgrade, then a `vulkaninfo` check.
  - **API shape:** a per-frame import is not free (page pinning / GPU MMU map in kbase); import only a *ring* of caller buffers once and reuse, or require the caller to allocate frames from a pool the library hands out (then plain `DeviceLocalMapped` memory already gives zero copy and no extension is needed - the cheaper design). Frame pointers from `cv::Mat`/JNI `ByteBuffer` are rarely 4 KiB aligned, and the size must be rounded up - the import has to cover whole pages, so the library needs `W*H` rounded to 4 KiB of *readable* memory.
  - **RDNA:** the imported memory is host (system) memory read over PCIe by the GPU; decimate at dec 2 reads 1/4 of the bytes, but the 3x3/min-max passes then read the decimated copy from VRAM, so only one PCIe read of the frame. This replaces staging memcpy + DMA copy; saves CPU time (~0.1-0.2 ms serial) rather than GPU time.
  - **Mali:** memory type must be checked for HOST_CACHED/coherency; a non-coherent cached import would need `vkFlushMappedMemoryRanges`-equivalent cache maintenance by the driver (kbase does this on import/sync) - cost unknown.
- Verification: [verified from primary source] for the mechanism and the two alignment values (gpuinfo 50479, local vulkaninfo); uplift is an in-tree bound, not a measurement.

### VK_EXT_external_memory_dma_buf + VK_KHR_external_memory_fd (+ VK_EXT_image_drm_format_modifier, VK_EXT_queue_family_foreign) — Khronos, 2017-2018
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_dma_buf.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryFdInfoKHR.html
- Technique: adds handle type `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` to `KHR_external_memory_fd`. "Enables applications to import a dma_buf as VkDeviceMemory ... and to create VkBuffer objects that can be bound to that memory." Importing "transfers ownership of the file descriptor ... The application must not perform any operations on the file descriptor after a successful import" (so `dup()` the V4L2/MPP fd first). *Buffers* need nothing else; *images* need `image_drm_format_modifier` to describe pitch/tiling. `queue_family_foreign` gives a proper acquire/release barrier against the ISP/decoder (`VK_QUEUE_FAMILY_FOREIGN_EXT`).
- Reported speed-up: none published for this use. Bound in-tree: the 0.22 ms upload memcpy (dec 1) plus, in the demo capture app, the YUYV->gray CPU extraction pass (survey B2).
- Accuracy / determinism impact: exact (same luma bytes), provided the shader honours the source stride.
- Applicability to vkapriltag:
  - **Availability:** every Mali-G610 driver in the table (g6p0, g18p0, g24p0, r38p1, r54p2, panvk) exposes all four; RX 9060 XT on Windows does not (Linux RADV/amdgpu does, not checked here).
  - **Path on the Pi:** V4L2 `VIDIOC_EXPBUF` (UVC / rkisp) or MPP's DRM-PRIME output -> `dup(fd)` -> `vkAllocateMemory` with `VkImportMemoryFdInfoKHR` + `VkExternalMemoryBufferCreateInfo` -> bind a storage `VkBuffer` -> `decimate.comp` reads the Y plane (NV12 / GREY) or every other byte (YUYV) with a push-constant stride. **Keep it a buffer**: the image path (and hence `drm_format_modifier`) is unnecessary because the tree already rejected `texelFetch`/tiled images (+29-35% threshold span).
  - Import cost per frame is avoidable: V4L2 and MPP recycle a small fixed ring of buffers, so import each fd once, cache `VkDeviceMemory` by (inode/fd), and only barrier per frame.
  - **Coherency risk:** CMA/dma-heap camera buffers are often *uncached* on the CPU side (fine for the GPU) - but `TagDecoder` samples the full-resolution frame on the CPU, and CPU reads from write-combined/uncached memory are slow on A76. Measure the CPU decode tail before and after; may need a CPU-side copy for the decoder anyway (which would still remove the GPU-upload copy).
  - Fence: V4L2 has no sync_file for capture completion (DQBUF means done), so a plain queue submit after DQBUF is correct; MPP likewise returns completed frames.
- Verification: [verified from primary source] for the mechanism quotes and availability (reports listed in section 0). No throughput numbers exist; uplift is an in-tree bound.

### g29p1 on RK3588 Linux actually runs Vulkan compute, and exposes the newer names — mali-vulkan-icd-wrapper docs (GinKage et al.), 2026
- URL: cached `claude-research/mali/wrapper_readme.md`, `wr_compatibility.md`, `wr_diagnostics.md`, `wr_installation.md` (project by GinKage, https://github.com/ginkage)
- Technique: an ICD wrapper that loads libmali **g29p1** for Vulkan apps on Armbian Noble / kernel `6.1.115-vendor-rk35xx` (same kernel as the vkapriltag Pi), adding the WSI the blob lacks. Headless compute does not need the wrapper's WSI.
- Reported speed-up: n/a.
- Accuracy / determinism impact: n/a (driver swap; bit-identity must be rechecked, esp. atomics/ordering - should be unaffected).
- Applicability to vkapriltag: this is the only secondary evidence that g29p1-only names are *advertised on a G610 under Linux*: the wrapper has a switch to hide `VK_EXT_external_memory_host` from device enumeration (so the blob advertises it), and its descriptor-buffer GPU test runs on g29p1 ("fixed-count descriptor-buffer layouts pass"; variable-count layouts read zero - a driver bug). So g29p1 is the route to `external_memory_host`, `pipeline_executable_properties`, `robustness2`, `maximal_reconvergence`, `subgroup_rotate`, `pipeline_binary` on the Pi. `conditional_rendering` remains doubtful (absent from r54p2 on the same GPU).
- Verification: [secondary/unverified] - third-party docs; no `vulkaninfo` dump of g29p1 on G610 was found.

## 2. Shader-execution controls

### VK_EXT_subgroup_size_control (core 1.3) and wave32 on RDNA — AMD, "RDNA 3: Beyond the current gen" (Lou Kramer, Reboot Develop Blue, Apr 2023) + GPUOpen RDNA Performance Guide
- URL: https://gpuopen.com/download/RDNA3_Beyond-the-current-gen-v4.pdf (cached `claude-research/ext08/rdna3_beyond.txt`), https://gpuopen.com/learn/rdna-performance-guide/
- Technique: `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{requiredSubgroupSize=32}` on the compute stage (allowed when `requiredSubgroupSizeStages` contains COMPUTE) pins a pipeline to wave32 or wave64; `REQUIRE_FULL_SUBGROUPS` / `ALLOW_VARYING_SUBGROUP_SIZE` flags control partial subgroups. AMD's slide: "Wave32 for highly divergent code or long running threads; Wave64 for ALU heavy and better data access patterns ... Driver will try to pick the optimal wave size for the given GPU ... If you do set it manually [measure]". Same deck: for 2D stores, "Best bet is wave64 with 8x8 thread-group" (write-compression: a wave32 cannot fully overwrite a 256-byte 8x8x4B block). Performance guide: "Make the workgroup size a multiple of 64".
- Reported speed-up: none for this workload (AMD gives only direction).
- Accuracy / determinism impact: exact for shaders without subgroup ops. `reduce_extents_hash_subgroup` is exact at any size (per survey A3); anything using `gl_SubgroupSize`-dependent layouts must be re-checked.
- Applicability to vkapriltag:
  - **RDNA4 (RX 9060 XT):** `minSubgroupSize=32, maxSubgroupSize=64`, `requiredSubgroupSizeStages` includes COMPUTE, and the default reported `subgroupSize = 64` - the proprietary driver's heuristic choice per shader is not observable without `pipeline_executable_properties` (see next entry, which reports the chosen size). Candidates for wave32: `uf_merge` / `uf_compress` / `uf_final` (`find()` pointer chasing - "long running, divergent"), `hash_group` (probe loops of varying length), `sort_points_local` (per-blob variable-length networks). Candidates to *keep* wave64: `decimate`, `block_minmax`, `block_filter`, `threshold` (streaming, 2D stores - AMD's write-compression note). Sweep = one create-info per pipeline, env-var gated; measure with the existing ABBA harness.
  - **Mali:** `min == max == 16` in every G610 report (g24p0, r38p1, r54p2) - nothing to control (confirms OPTIMIZATION_NOTES item 4).
  - **Also relevant:** `computeFullSubgroups = true` on both; not needed.
- Verification: [verified from primary source] for AMD quotes and the device limits; the RDNA4 per-shader uplift is unmeasured.

## 3. Diagnostics that steer tuning (no per-frame gain by themselves)

### VK_KHR_pipeline_executable_properties — Khronos (Intel-authored), 2019
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineExecutablePropertiesKHR.html
- Technique: create the pipeline with `VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR` (and optionally `..._INTERNAL_REPRESENTATIONS_BIT_KHR`), then `vkGetPipelineExecutablePropertiesKHR` / `...StatisticsKHR` / `...InternalRepresentationsKHR`. `VkPipelineExecutablePropertiesKHR.subgroupSize` is "the subgroup size with which this pipeline executable is dispatched" - i.e. it reveals the wave size the AMD driver picked. Statistics are driver-defined name/value pairs (register counts, spills, LDS, code size on AMD/RADV/Intel; unknown for libmali). "Intended to be used by debugging and performance tools".
- Reported speed-up: n/a (diagnostic).
- Accuracy / determinism impact: none (capture flag may change compile caching; do not ship it on).
- Applicability to vkapriltag:
  - **RDNA4:** available (`pipelineExecutableInfo = true`). First step of the wave32 sweep: dump `subgroupSize` + VGPR count per pipeline (the ~20 pipelines) to know which ones the driver already runs at wave32; only the wave64 ones are sweep candidates. Also shows whether `find()` loops spill.
  - **Mali:** not on g24p0; in g29p1 strings and exposed by r54p2 Android and panvk - with g29p1 on the Pi this would give register/spill counts per shader for the first time (Mali Offline Compiler `malioc` is the existing alternative - covered by category 07 presumably). Arm BP 3.4 s9.2: shader cores "split those registers across a variable number of threads depending on the register usage" - so register count decides how many threads hide the dependent-load latency that bounds labelling (exact Valhall thresholds not verified here).
- Verification: [verified from primary source] (refpages, availability table). Contents of libmali's statistics unverified.

### VK_AMD_shader_info — AMD, 2017
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderStatisticsInfoAMD.html
- Technique: `vkGetShaderInfoAMD(STATISTICS)` returns `VkShaderStatisticsInfoAMD` (resource usage incl. VGPR/SGPR/LDS/scratch, `numPhysicalVgprs`, `numAvailableVgprs`, `computeWorkGroupSize`); `DISASSEMBLY` returns ISA text. AMD-only.
- Reported speed-up: n/a.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: RDNA4 only; overlaps `pipeline_executable_properties`. Disassembly is useful to confirm that `atomicMin` in `uf_merge` compiles to a single `buffer_atomic_umin` without a return (no-return atomics are cheaper) and whether `find()` loads are `s_load` vs `buffer_load`. Prefer the KHR extension for portability (same code works on g29p1).
- Verification: [verified from primary source] (refpage).

### VK_KHR_shader_clock — Khronos, 2019
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_clock.html
- Technique: SPIR-V `OpReadClockKHR` with `Subgroup` scope (`clockARB()`, feature `shaderSubgroupClock`) or `Device` scope (`clockRealtimeEXT()`, `shaderDeviceClock`). Units/monotonicity are not specified by the extension.
- Reported speed-up: n/a (instrumentation).
- Accuracy / determinism impact: none when compiled out.
- Applicability to vkapriltag: RDNA4 has both; **no proprietary Mali driver has it** (only panvk 26.3). Use: per-invocation cycle histograms of `find()` walks and hash probes, to decide whether the labelling tail is a few long chains (argues for wave32 / path compression changes) or uniform. On Mali, iteration counters via atomics are the substitute.
- Verification: [verified from primary source].

## 4. Scheduling and command-stream control

### VK_KHR_global_priority (was EXT_global_priority + EXT_global_priority_query) — Khronos, promoted 2023
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_global_priority.html
- Technique: `VkDeviceQueueGlobalPriorityCreateInfoKHR` at device creation requests LOW/MEDIUM/HIGH/REALTIME for a queue; `VkQueueFamilyGlobalPriorityPropertiesKHR` lists levels the driver treats distinctly. The driver "will attempt to skew hardware resource allocation in favor of the higher-priority task"; above-default requests "may" fail with `VK_ERROR_NOT_PERMITTED_KHR` if the caller lacks privileges, and privileges cannot be queried in advance. The extension says nothing about preemption.
- Reported speed-up: none; it is a contention/latency tool, not a throughput one.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag:
  - **Availability:** KHR on g24p0 (41734), g29p1, r38p1, r54p2, panvk, RDNA4; EXT-only on g6p0/g18p0. `globalPriorityQuery = true` everywhere it is listed.
  - **When it matters:** only if something else uses the GPU concurrently - on the Orange Pi under PhotonVision the GPU is otherwise idle unless a desktop compositor or hardware-accelerated stream overlay runs; on desktop, the compositor/games. Implement as: try HIGH, on `VK_ERROR_NOT_PERMITTED_KHR` retry at MEDIUM (must be a retry of `vkCreateDevice`, since the error is at device creation). Linux kernel permission rules (amdgpu: HIGH needs `CAP_SYS_NICE` or DRM master; Mali kbase CSF queue-group priority checks) are [secondary/unverified] here.
  - Expected gain in an idle-GPU deployment: 0. In a shared-GPU deployment it bounds tail latency; measure p99 of GPU span with a synthetic competing load (e.g., `vkcube` or a busy-loop compute app) before/after.
- Verification: [verified from primary source] (refpage + table); OS permission details unverified.

### VK_EXT_conditional_rendering — Khronos (AMD/others), 2018 — already built in-tree
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_conditional_rendering.html ; in-tree PERFORMANCE.md 6c
- Technique: `vkCmdBeginConditionalRenderingEXT(buffer, offset)` - subsequent draws/**dispatches** are discarded if the 32-bit value is zero (or non-zero with `INVERTED`). "Conditional rendering should not affect copies and blits" - so it cannot skip `vkCmdFillBuffer`/`vkCmdCopyBuffer`. Buffer needs `CONDITIONAL_RENDERING_BIT_EXT` usage; the predicate read is ordered with `VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT` / `ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT`.
- Reported speed-up: in-tree, RX 9060 XT: labelling -3.0% at dec 1, GPU total <=1%. Mali ceiling from removing the dispatch outright: labelling -4.4%, GPU total -1.4%, of which the in-shader guard already captures ~half.
- Accuracy / determinism impact: exact (bit-identical over 60 configs).
- Applicability to vkapriltag:
  - **Correction to the survey's availability column:** not exposed on any *proprietary* G610 report (g6p0, g18p0, g24p0, r38p1, r54p2). It **is** in the g29p1 binary strings, and panvk 26.3 exposes it - but since r54p2 (a newer DDK) does not expose it on G610, the g29p1 string is most likely for other GPUs (e.g. 5th-gen Mali). Best case on G610 if it were exposed: the remaining ~0.7% GPU total.
  - Portable substitute already analysed in-tree (PERFORMANCE 6c): device-computed zero-group `vkCmdDispatchIndirect` - costs one extra dispatch + barrier (2.6-18.7 us on Mali) vs ~30 us saved; "not obviously positive". An alternative that costs **no extra barrier**: have the *merge* shader itself write the next compress's `VkDispatchIndirectCommand` (x = changed ? N : 0) - the merge->compress barrier already exists; it only needs `INDIRECT_COMMAND_READ` added to its dst stage/access. The write race (many invocations setting x=N) is benign because every writer writes the same N; use one args slot per iteration, zero-filled by the frame clear that already exists (y=z=1 preset), so no new fill and no new barrier. Unknown: whether an indirect dispatch costs more to launch than a direct one on Mali CSF. Worth one ABBA run on Mali: ceiling is the other ~0.7%. Also worth checking the survey's A6 per-iteration gating with the same trick.
- Verification: [verified from primary source] for mechanism and in-tree numbers; the merge-writes-args variant is my proposal, unmeasured.

### VK_EXT_device_generated_commands — Khronos (NVIDIA/AMD/Valve), 2024; plus NVIDIA forum report "Extremely poor VK_EXT_device_generated_commands performance", 2025
- URL: https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_device_generated_commands.html ; https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdExecuteGeneratedCommandsEXT.html ; https://forums.developer.nvidia.com/t/extremely-poor-vk-ext-device-generated-commands-performance/324189
- Technique: an indirect-commands layout of tokens (`EXECUTION_SET` = bind a pipeline/shader from an execution set, `PUSH_CONSTANT`, `DISPATCH` = a `VkDispatchIndirectCommand`) is replayed `min(maxSequenceCount, *sequenceCountAddress)` times from a device buffer; optional explicit preprocessing (`vkCmdPreprocessGeneratedCommandsEXT`, `COMMAND_PREPROCESS` stage). Sequences cannot carry pipeline barriers, and with `UNORDERED_SEQUENCES` their order is implementation-defined - so one execute is a set of *independent* dispatches.
- Reported speed-up: the forum report (RTX 3080) is the opposite: 2,500+ compute dispatches took 0.59-0.63 ms as a `vkCmdDispatch` loop but **14-15 ms via EXT DGC** (3 ms with an empty shader); NVIDIA's engineer called the gap "in line with what is expected" for a stress test and said benefits come from GPU-driven culling. EXT draws were 12x slower than the NV extension on the same card.
- Accuracy / determinism impact: none inherently.
- Applicability to vkapriltag: **reject.** The only device-decided control flow in the frame is the labelling loop (merge -> barrier -> compress, repeated) and the tail's sizes; the loop needs barriers between its dispatches, which DGC cannot encode, and the tail's sizes are already device-decided via `vkCmdDispatchIndirect` + `build_indirect_args.comp`. Availability: RDNA4, panvk 26.3 only; no proprietary Mali. Conditional rendering / indirect zero-dispatch cover the "skip" case more cheaply.
- Verification: [verified from primary source] (proposal, refpage) + [secondary/unverified] forum numbers (user report).

## 5. Host-side / API-overhead extensions

### VK_EXT_descriptor_buffer, VK_KHR_push_descriptor, VK_KHR_maintenance5 / 6 / 7 / 8 / 9 — Khronos refpages
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance5.html (and `..._maintenance6/7/8/9.html`); cached wrapper finding `claude-research/mali/wr_diagnostics.md`
- Technique:
  - `descriptor_buffer`: descriptors become bytes in a buffer the app writes, bound by offset (`vkCmdSetDescriptorBufferOffsetsEXT`); `push_descriptor`: descriptors recorded into the command buffer (`vkCmdPushDescriptorSetKHR`), no pool/set.
  - maintenance5: `VkPipelineCreateFlags2`/`VkBufferUsageFlags2`, `VkShaderModuleCreateInfo` chained straight into pipeline creation (shader modules deprecated), stronger DEVICE_LOST propagation. maintenance6: pNext-extensible `vkCmdBindDescriptorSets2`/`vkCmdPushConstants2`/`vkCmdPushDescriptorSet2`, `VkBindMemoryStatus`. maintenance7: dynamic-buffer limits, layered-driver info, u32/u64 query consistency. maintenance8: `OpSRem`/`OpSMod` defined for negatives, 64 more access flags, pipeline-cache merge sync. maintenance9: device creation without queues, `VK_QUERY_POOL_CREATE_RESET_BIT_KHR` (pools born reset), any integer width for bit ops, **no queue-family ownership transfer needed for buffers**.
- Reported speed-up: none of these is a GPU-time feature.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: the tree allocates each pipeline's descriptor set once (`ComputePipeline.cpp:114-130`) and per dispatch only calls `vkCmdBindDescriptorSets` + `vkCmdPushConstants` - so descriptor buffers/push descriptors save at most a few hundred ns per dispatch x ~60 dispatches, which only matters if survey A4 (pre-recorded command buffers) is *not* done; A4 removes recording entirely and dominates. Availability: `push_descriptor` on g24p0 (`maxPushDescriptors = 32`), not on r38p1; `descriptor_buffer` g29p1/r54p2 only - and the wrapper project found a **g29p1 bug**: descriptor-buffer layouts with `VARIABLE_DESCRIPTOR_COUNT` "cause shader descriptor reads to resolve to zero" (fixed-count layouts pass). maintenance5/6 on g24p0; 7 on g29p1/r54p2; 8/9 RDNA4 + panvk 26.3 only. maintenance9's `QUERY_POOL_CREATE_RESET` would drop the per-frame `vkCmdResetQueryPool` of the profiling build only. Keep the tree's "no optional features" posture: **no action**, consistent with OPTIMIZATION_NOTES item 4 and survey section 6.
- Verification: [verified from primary source] for maintenance5-9 contents; descriptor-buffer/push-descriptor mechanisms from spec knowledge [secondary/unverified] (standard, but not re-read this run); the g29p1 bug is [secondary/unverified] (wrapper docs).


## 6. Math-unit extensions

### VK_KHR_cooperative_matrix and VK_KHR_shader_integer_dot_product — Khronos, 2023 / 2021
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_integer_dot_product.html
- Technique: cooperative matrix = SPIR-V matrix types whose storage "is spread across all invocations in some scope (usually a subgroup)", sizes/types enumerated by `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (WMMA on RDNA). Integer dot product = SPIR-V `OpSDot/OpUDot/OpSUDot` (+ accumulate-saturate) over 4x8-bit packed or vector operands; the `integerDotProduct*Accelerated` properties say whether a form beats the equivalent multiply-add sequence. The refpage: "particularly useful for neural network inference ... but find uses in other general-purpose compute applications as well."
- Reported speed-up: none for image-processing pipelines of this kind.
- Accuracy / determinism impact: exact for integer forms (dot product is exact integer arithmetic; accumulate-sat saturates).
- Applicability to vkapriltag:
  - **Availability:** integer dot product is on every proprietary G610 driver since g18p0 with `integerDotProduct8BitUnsignedAccelerated`, `...8BitSignedAccelerated`, `...4x8BitPackedUnsignedAccelerated = true` (37373, 41734, 50479); panvk 26.3 reports the 8-bit forms as not accelerated. RDNA4: accelerated (vi_local). Cooperative matrix: RDNA4 yes (`cooperativeMatrix = true`); **no G610 report exposes it** - the g24p0/g29p1 strings are for other (5th-gen) Mali parts (see section 0 caveat).
  - **Where it could apply:** the pipeline has no GEMM. The only dot-shaped arithmetic is (a) the 3x3 box/filter stages - but those are min/max, not sums; (b) the per-point line-fit moments in `sort_points_local` (`W*x`, `W*y`, `W*x*x`, ...) - these are 32/64-bit integer products of coordinates up to 1280 and weights up to 361 (PERFORMANCE.md table), which do not fit 8-bit operands; (c) gradient `gx/gy` = differences of neighbouring 8-bit pixels - two subtractions, nothing for a dot unit to save. The GPU spans are bound by dependent loads (labelling) and atomics/latency (extents, sort), not ALU (brief), so even a valid mapping would not move the frame.
  - **Verdict: reject** for the current pipeline on both devices. Would matter only for a learned front end (a YoloTag-style detector - category of learned detectors), where `KHR_cooperative_matrix` on RDNA4 and int8 dot on G610 are the relevant paths.
- Verification: [verified from primary source] (refpages, availability/feature bits from the cached reports); the "no mapping" judgement is mine.

### VK_KHR_8bit_storage, VK_KHR_16bit_storage, VK_KHR_shader_float16_int8 — Khronos (core 1.2 / 1.1) — in-tree status
- URL: in-tree `OPTIMIZATION_NOTES.md` item 4 table and line ~443; `library/src/vk/Context.cpp` (`supports_8bit_storage_`); refpages not re-read (core features).
- Technique: 8/16-bit types in storage buffers / push constants; `shaderFloat16`/`shaderInt8` arithmetic.
- Reported speed-up: in-tree, 8-bit storage is **adopted** (`_u8` variants of every image consumer, selected by `DeviceCaps::has_8bit_storage`, A/B with `APRILTAG_VK_FORCE_NO_8BIT`). 16-bit storage and fp16 **rejected**: traffic-only (capped at ~15% of GPU time on Mali), the line-fit record halving measured nil, fp16 lossy (11-bit mantissa lands in corner positions).
- Accuracy / determinism impact: 8-bit exact; fp16 not bit-identical (rejected for that reason).
- Applicability to vkapriltag:
  - **Availability:** `storageBuffer8BitAccess`, `storageBuffer16BitAccess`, `storagePushConstant8/16`, `shaderFloat16`, `shaderInt8`, `shaderInt16` are true on every G610 report checked (37373, 41734, 50479) and on RDNA4. So nothing blocks either device.
  - **One remaining 16-bit angle, also a no:** the UF parent array cannot be 16-bit (640x400 = 256 000 labels at dec 2 > 65 535). A 16-bit *tile-local* parent is what the rejected shared-memory UF would need, and that was rejected for other reasons.
  - **No action.** Consistent with the tree.
- Verification: [verified from primary source] for the feature bits (reports) and the in-tree verdicts (notes); nothing new measured.

## 7. Robustness

### VK_EXT_pipeline_robustness, VK_EXT/KHR_robustness2 — Khronos, 2022 / 2020 (KHR 2025)
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_pipeline_robustness.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_robustness2.html
- Technique: `pipeline_robustness` lets an app "request robustness on a per-pipeline stage basis" because robustness "may have an adverse effect on performance". `robustness2` defines stricter behaviour (`robustBufferAccess2`: out-of-bounds "writes must be discarded, out of bound reads must return zero") and `nullDescriptor` (`VK_NULL_HANDLE` allowed in descriptors); the spec warns these "may be expensive on some implementations".
- Reported speed-up: none published; the motivation is avoiding the cost of robustness.
- Accuracy / determinism impact: none for in-bounds code.
- Applicability to vkapriltag:
  - **The tree already runs non-robust:** `Context.cpp` zeroes `features2.features` (only `shaderInt64` is ever set), so core `robustBufferAccess` is off and there is no robustness cost to remove. `pipeline_robustness` would only matter if a device enabled robustness by default - its `defaultRobustnessStorageBuffers` property is how to check (gpuinfo 41734 shows it as "unknown", i.e. not captured).
  - **Availability:** `pipeline_robustness` on g18p0/g24p0/r54p2/panvk (not RDNA4 26.8.1 - it is core 1.4 there); `robustness2` on g29p1-str/r54p2/panvk 26.3/RDNA4, but on G610 r54p2 and panvk 26.3 `robustBufferAccess2 = false` and only `nullDescriptor = true`. RDNA4: `robustBufferAccess2 = true`.
  - `nullDescriptor` could let unused optional bindings (e.g., profiling buffers) be left null instead of bound to a dummy - host convenience only.
  - **Verdict: no action.** One cheap sanity check worth doing on the Pi when bumping drivers: query `VkPhysicalDevicePipelineRobustnessPropertiesEXT::defaultRobustnessStorageBuffers`; if a driver reports robust-by-default, set `VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT` on the hot pipelines (`uf_merge`, `hash_group`) and A/B.
- Verification: [verified from primary source] (refpages, feature bits in reports 41734/50479/51727, vi_local, Context.cpp).

## 8. Startup and memory residency

### VK_KHR_pipeline_binary — Khronos, 2024
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_binary.html
- Technique: "provides a method to obtain binary data associated with individual pipelines such that applications can manage caching themselves instead of using VkPipelineCache objects" - `vkGetPipelineKeyKHR` (key to look up), `vkCreatePipelineBinariesKHR` (from a pipeline or from stored data), `vkGetPipelineBinaryDataKHR`, `vkReleaseCapturedPipelineDataKHR`.
- Reported speed-up: in-tree cold pipeline creation is **240.7 ms** once per machine; with only the driver cache warm ~18 ms (OPTIMIZATION_NOTES "Pipeline caching"). The app-level `VkPipelineCache` already exists (`PipelineCache.cpp`, `APRILTAG_VK_PIPELINE_CACHE=0` to disable).
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: startup-only; nothing per frame. The existing VkPipelineCache already covers libmali g24p0 (which lacks pipeline_binary). Availability: RDNA4 (`pipelineBinaries = true`, `pipelineBinaryInternalCache = true`, `...PrefersInternalCache = false`), r54p2 Android, panvk 26.3, g29p1 strings. Only benefit over VkPipelineCache: per-pipeline keys make it possible to ship precompiled binaries for a known driver build (e.g. a PhotonVision image with a pinned libmali) so even the first boot skips the 240 ms; the notes' unmeasured "first-dispatch" deferred-codegen cost is not addressed by either. **Verdict: no action** unless first-boot latency becomes a product requirement.
- Verification: [verified from primary source] (refpage, vi_local, in-tree notes).

### VK_EXT_memory_priority / VK_EXT_pageable_device_local_memory — Khronos (AMD/NVIDIA), 2019 / 2021
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_priority.html
- Technique: a per-allocation priority (0..1) that guides the driver when "the implementation may transparently move memory from one heap to another when a heap becomes full" - i.e. what stays in VRAM under oversubscription. `pageable_device_local_memory` lets the driver page device-local memory and change priority after allocation.
- Reported speed-up: none; effect only exists under VRAM pressure.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: RDNA4 only (no Mali driver exposes either; on unified-memory Mali there is one heap and nothing to migrate). The working set is a few tens of MB against 8/16 GB VRAM, so eviction happens only if a co-running game fills VRAM. Setting priority 1.0 on the UF/label/hash buffers is a two-line defensive change with zero expected gain on an idle desktop. **Verdict: optional, low priority.**
- Verification: [verified from primary source] (refpage, table).

## 9. Device-sized copies and the RX 9060 XT's four-submit frame

### VK_KHR_copy_memory_indirect (and a no-extension alternative) — Khronos, 2025; applied to the in-tree fused-submit gate
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_copy_memory_indirect.html ; in-tree `library/src/vk/Buffer.cpp` (`MemoryKind::DeviceLocalReadback`), `library/src/gpu/GpuDetector.cpp:351-397` (`fused_submits_`), PERFORMANCE.md 3c; RDNA4 memory types in `mali/vi_local.txt` lines ~1288-1410.
- Technique: "copies between memory and image regions using indirect parameters that are read by the device from a buffer during execution" - `vkCmdCopyMemoryIndirectKHR` reads {src address, dst address, size} records from device memory (buffer device addresses mandatory; feature `indirectMemoryCopy`).
- Reported speed-up: none published. In-tree bound: fusing the tail into one submission measured **-2.0% / -4.0..4.4% / -7.9% GPU total** (dec 1/2/4) on Mali, and closing the last boundary another -0.7..-5.2%. On the RX 9060 XT that fused path **never engages**; an empty submit+fence round trip there is **42 us** (vs 19 us on Mali), and PERFORMANCE.md 6c says the RDNA GPU total "is dominated by its submit round trips". Three mid-frame boundaries x >=42 us = >=0.13 ms against a 0.66 ms GPU phase.
- Accuracy / determinism impact: exact (same bytes, same order).
- Applicability to vkapriltag:
  - **Why the fused path is off on RDNA4:** `fused_submits_` needs `selected_extents_buf_` and `line_fit_points_buf_` to be `DeviceLocalReadback` = DEVICE_LOCAL | HOST_VISIBLE **and** HOST_CACHED. The RDNA4 Windows driver has no such type: its DEVICE_LOCAL|HOST_VISIBLE types (2, 6) are uncached and sit on a **256 MiB** BAR heap (heap 2 - Resizable BAR is apparently off on this box), while HOST_VISIBLE|COHERENT|**CACHED** exists only as system memory (type 3, heap 1). So the gate is a memory-type accident, not a real hazard.
  - **Option A (no extension, cheapest bound):** on discrete parts allocate those two buffers as `HostVisibleCached` system memory (type 3) and let `select_blobs` / `sort_points_local` write them directly over PCIe - GPU writes to host memory are posted; ~1.6 MB/frame line-fit at 1080p, far less at 1280x800 dec 2. Cost: later shaders that *read* `selected[]` (`scan`, `sort_points_local`, bindings at GpuDetector.cpp:615/642/689) would read it across PCIe - a few KB, but latency-bound. Bound in one line: force `MemoryKind::HostVisibleCached` for the two buffers, set `fused_submits_` true, ABBA against the current path (`APRILTAG_VK_FUSE_SUBMITS=0`), check bit-identity.
  - **Option B (extension):** keep the buffers device-local, and at the end of the single submission `vkCmdCopyMemoryIndirectKHR` into a HOST_CACHED sysmem staging buffer with the size written by `build_indirect_args.comp` (the count is device-side, which is exactly why the fused path could not record a sized `vkCmdCopyBuffer`). Available: RDNA4 (`indirectMemoryCopy = true`, KHR) and panvk 26.3; **no proprietary Mali** - irrelevant there since Mali already takes the fused path. Option B keeps `selected[]` reads in VRAM; Option A needs no extension. A third, portable variant: a trivial copy compute shader with a `vkCmdDispatchIndirect` sized by the same counters, writing into type-3 memory.
  - **Relation to the in-tree rejection "submit collapsing on dGPU":** that verdict (A1, commits `bcfa3dc`/`ecaeaae`/`f436eb3`) was about building the device-side-sizing machinery (measured 0.05 ms for one removed submission, "0.1-0.15 ms for a lot of machinery"). The machinery now exists and ships on Mali; what remains on RDNA4 is only the readback memory type. That is new evidence, so the idea is re-proposed.
  - **Driver note:** the third-pass RDNA4 numbers are from the Windows proprietary driver 1.4.349 (OPTIMIZATION_NOTES third pass) - the same driver family as `vi_local.txt`; PERFORMANCE.md's startup sample shows RADV 1.4.354 on the same card, whose memory types were not checked here.
  - **Also check:** with Resizable BAR enabled in firmware the BAR heap grows but its types stay uncached (AMD exposes no DEVICE_LOCAL|HOST_CACHED type on Windows in this dump), so ReBAR alone would not open the gate.
- Verification: [verified from primary source] for the refpage, memory types, gate code and in-tree numbers; the RDNA4 uplift is my estimate (unmeasured) - bound it with Option A first.

## 10. Synchronization and timing plumbing

### VK_KHR_timeline_semaphore, VK_KHR_synchronization2, VK_EXT_host_query_reset, VK_KHR/EXT_calibrated_timestamps — Khronos (core 1.2 / 1.3 / 1.2 / ext)
- URL: in-tree OPTIMIZATION_NOTES item 4 table ("Structural/diagnostic convenience", sync2 "intra-submit gaps 0.067 ms total"); https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_calibrated_timestamps.html (refpage not re-read this run)
- Technique: timeline semaphores = one 64-bit counter semaphore that the host can wait/signal (replaces fence+binary-semaphore bookkeeping, allows wait-before-signal submission); sync2 = `VkDependencyInfo`/`vkCmdPipelineBarrier2` with 64-bit stage/access masks; host_query_reset = `vkResetQueryPool` from the host (no `vkCmdResetQueryPool` in the command buffer); calibrated_timestamps = `vkGetCalibratedTimestampsKHR` returns correlated GPU and host (`CLOCK_MONOTONIC_RAW` / QPC) timestamps plus max deviation.
- Reported speed-up: none per frame. In-tree: sync2's target (intra-submit barrier gaps) is **0.067 ms/frame** on Mali; timestamps themselves cost **~9%** of GPU time on Mali (3.16 vs 2.90 ms) - none of these extensions changes that cost.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag:
  - **Availability:** timeline, host_query_reset: every driver in the table. sync2: every driver except g6p0. calibrated_timestamps: EXT everywhere except panvk 25; KHR on g24p0+. Timestamp resolution: `timestampPeriod` = 41.67 ns on g24p0 (41734), 76.9 ns on r38p1, 520.8 ns on r54p2, **10 ns on RDNA4** - so on Mali the per-span timestamps have ~42 ns granularity against spans of tens of us (fine), but the 18.7 us barrier costs are measured with only ~450 ticks.
  - **Timeline semaphores** make the frame-pipelining (`PipelinedDetector`, 1-deep) cleaner if it ever goes 2-deep: CPU decode of frame N, GPU of N+1 and upload of N+2 each wait on a counter value rather than per-slot fences. No GPU time; only useful if the pipelining depth is increased (category on scheduling).
  - **Calibrated timestamps** would let the profiler place GPU spans on the host timeline, making the host-side gaps (the "submit boundary" accounting that `DetectProfile::gap_crosses_submit` currently infers) directly measurable - relevant to bounding section 9's RDNA4 idea. Diagnostic only.
  - **Verdict:** no per-frame gain; adopt calibrated timestamps in the profiling build only if the RDNA4 submit-gap work (section 9) needs host/GPU correlation.
- Verification: [verified from primary source] for availability/timestampPeriod (reports, vi_local) and in-tree numbers; extension mechanics from spec knowledge [secondary/unverified] (not re-read this run).

## 11. Arm vendor extensions

### VK_ARM_shader_core_builtins (+ VK_ARM_shader_core_properties) — Arm, 2022-2023; with Arm GPU Best Practices 3.4 s10.13 "Atomics"
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_shader_core_builtins.html ; cached `claude-research/mali/bp34.txt` lines 5267-5311
- Technique: SPIR-V builtins `CoreIDARM` (`gl_CoreIDARM`), `CoreCountARM`, `CoreMaxIDARM`, `WarpIDARM`, `WarpMaxIDARM`; properties `shaderCoreCount`, `shaderCoreMask`, `shaderWarpsPerCore`. The refpage names the use case: "using core identifiers to reduce atomic contention". Arm BP 3.4: "Atomic operations from different shader cores. Hitting the same cache line requires data coherency snooping through L2 cache, which is computationally expensive ... aim to spread out the contention by keeping the atomic operations local to a single shader core. Atomics are efficient when a shader core controls the necessary cache line in its L1" and, for OpenCL, "consider the use of the cl_arm_get_core_id extension to allow explicit management of per-shader-core atomic variables" - `gl_CoreIDARM` is the Vulkan equivalent. Also: "Consider spacing atomics 64 bytes apart".
- Reported speed-up: Arm gives none. In-tree bounds: `extents` is 0.221 ms with int64 atomics; a deliberately-incorrect plain-store build measured 0.062 ms (common.glsl comment, older state: 0.434 -> 0.062) - so ~0.16 ms (~5% of Mali GPU total) is still atomic cost. The workgroup-indexed privatization sweep **plateaued** at K=8/16/32 (-20.3/-20.0/-20.6%), which is what you'd expect if the residual is *cross-core line migration* (copy `gl_WorkGroupID.x & 7` is still touched by workgroups on all four cores over the dispatch) rather than raw same-address serialization.
- Accuracy / determinism impact: exact - integer min/max/add are order-independent and `merge_extents.comp` folds all copies.
- Applicability to vkapriltag:
  - **Availability:** exposed on the **deployment driver** g24p0 (41734: `shaderCoreBuiltins = true`, `shaderCoreCount = 4`, `shaderCoreMask = 327685` = `0x50005` = core IDs 0, 2, 16, 18, `shaderWarpsPerCore = 64`), g29p1, r54p2 (same values), panvk 26.3. Not r38p1/g6p0/g18p0. Not on RDNA.
  - **Idea (Mali, extents):** in `reduce_extents_hash*.comp` replace the copy index `gl_WorkGroupID.x & (kExtentsCopies-1)` with a per-core index: `bitCount(coreMask & ((1u << gl_CoreIDARM) - 1u))` (mask passed as a push constant/spec constant; IDs are sparse, so compact them), optionally times 2 plus `gl_WarpIDARM & 1` to keep 8 copies. Then each copy's cache lines stay owned by one core's L1 - the exact pattern Arm recommends. Same trick applies to the other contended global counters: `uf_final`'s `blob_size[r]` histogram (the background blob), `hash_group`'s `raw_blob_counter`/`drop_counter`, and `select_blobs`/`blob_diff` append counters (these need per-core counters + an offset scan, so they are not drop-in; extents and `blob_size` are).
  - **Second, extension-free half of the same guidance:** `MinMaxExtentsGpu` is 48 bytes, so neighbouring blobs share 64-byte lines; pad to 64 B (or stride copies by 64 B) and measure separately.
  - **Bound first:** one shader-variant flip with `APRILTAG_VK_...` env gate; ABBA on the Pi at dec 1/2; ceiling is the 0.16 ms atomic residual. BP 3.4 ("How to debug") says the counters include L1 snoops from other cores, but the cached Mali-G610 counter guide (`mali/g610_counters.txt`) exposes no named snoop counter in its Streamline expressions - the usable one is `$MaliCoreLoadStoreCyclesAtomicAccessCycles` (s1.8.1.6); raw hwcnt snoop counters via libGPUCounters are unverified. So the bound is the ABBA timing itself, with the atomic-cycles counter as a secondary signal.
  - Gate on the extension + feature; the tree's "no vendor extensions" posture (OPTIMIZATION_NOTES item 4 table, "no identified bottleneck") now has an identified bottleneck, so this is new evidence.
- Verification: [verified from primary source] for the refpage text, BP quotes, feature values (41734/50479) and in-tree numbers; the migration diagnosis and the uplift are my hypothesis, unmeasured.

### VK_ARM_scheduling_controls — Arm, 2023 (rev 2 adds dispatch parameters)
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_scheduling_controls.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkDispatchParametersARM.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSchedulingControlsFlagBitsARM.html
- Technique: "a collection of controls to modify the scheduling behavior of Arm Mali devices". Flag `SHADER_CORE_COUNT` = `VkDeviceQueueShaderCoreControlCreateInfoARM` limits how many shader cores a queue/device uses. Flag `DISPATCH_PARAMETERS` = `vkCmdSetDispatchParametersARM` with `workGroupBatchSize` ("the number of workgroups in each batch distributed to shader cores"), `maxQueuedWorkGroupBatches`, `maxWarpsPerShaderCore`.
- Reported speed-up: none published.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag:
  - **Dispatch parameters would be the interesting part** for Mali: `workGroupBatchSize` controls which consecutive workgroups land on the same core (L1 locality of `find()` chains in labelling, which is dependent-load bound), and `maxWarpsPerShaderCore` would let atomics-bound passes trade occupancy for contention. **But no G610 driver exposes it:** `schedulingControlsFlags = 1` (core count only) on g24p0 (where `schedulingControls = false` anyway), r54p2 (`true`) and panvk 26.3; and `vkCmdSetDispatchParametersARM` does not appear among the `vkCmd*` entry-point strings of either the g24p0 or g29p1 binary (checked this run with `strings | grep '^vkCmd'`; the same method does find `vkCmdBeginConditionalRenderingEXT` in g29p1).
  - Core-count control only helps when sharing the GPU (reserve cores for a compositor/encoder) - a throughput *loss* for vkapriltag alone.
  - **Verdict: not usable on the G610 today;** recheck on newer DDKs (r5x Android shows only flag 1 too).
- Verification: [verified from primary source] (refpages, reports, binary strings).

### VK_ARM_tensors, VK_ARM_data_graph — Arm, 2025
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_tensors.html (not opened this run)
- Technique: tensor resources and a data-graph pipeline (graph of ML ops, e.g. from TOSA SPIR-V) dispatched through Vulkan, aimed at Arm's neural-accelerated GPUs.
- Reported speed-up: n/a.
- Accuracy / determinism impact: n/a.
- Applicability to vkapriltag: **absent from every G610 driver and from both libmali binaries' strings**, and from RDNA4. Relevant only to a learned front end on future Arm GPUs. Reject.
- Verification: availability [verified from primary source] (table); mechanism [secondary/unverified].

### VK_ARM_shader_instrumentation, VK_ARM_performance_counters_by_region — Arm, 2025-2026 (in vk.xml header 364)
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_shader_instrumentation.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_performance_counters_by_region.html ; `claude-research/ext08/vk.xml`
- Technique: shader_instrumentation can "instrument shaders and capture performance metrics per shader type from commands executed by a queue" (begin/end sessions in a command buffer, enumerate metrics). performance_counters_by_region captures shader-core counters "per region ('tile')" per render-pass instance - render passes only.
- Reported speed-up: n/a (tools).
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: counters-by-region is graphics-only - reject. shader_instrumentation would be the in-API way to get per-shader Mali metrics (e.g. to test section 11's L1-snoop hypothesis), but it is **absent from every G610 driver and from the g24p0/g29p1 binary strings**; use Arm Performance Studio / `libGPUCounters` (category 07) instead.
- Verification: [verified from primary source] (refpages, vk.xml, table).

## 12. Atomics and shared memory

### VK_KHR_shader_atomic_int64 (feature bits), VK_EXT_shader_atomic_float / float2 — Khronos, 2018 / 2020-2021
- URL: in-tree OPTIMIZATION_NOTES item 4 table; feature bits from the cached reports; `claude-research/ext08/vk.xml`
- Technique: 64-bit integer atomics on buffers/shared memory; float atomic add / min/max on buffers and shared memory.
- Reported speed-up: int64 is **adopted** in-tree: `extents` 0.221 vs 0.252 ms (-12..-13%) on Mali. Float atomics: no in-tree use.
- Accuracy / determinism impact: int64 exact (count and `pxgx_plus_pygy_sum` packed in one word). **Float atomic add is order-dependent** - summation order changes across runs, so it breaks the bit-identical gate; min/max float would be exact but every extents field is already integer.
- Applicability to vkapriltag:
  - int64 feature bits: `shaderBufferInt64Atomics` and `shaderSharedInt64Atomics` true on g24p0, r54p2, panvk 26.3, RDNA4; **false on r38p1 Android** (with `shaderInt64 = false`), and g6p0 (the first-pass driver) lacks the extension - so the fallback `reduce_extents_hash.comp` must stay.
  - One more int64 packing is plausible: min_x/min_y (and max_x/max_y) could be packed as two 32-bit halves in one 64-bit `atomicMin` only if the ordering of the pair were lexicographic-compatible - it is not (min of a pair != pair of mins), so **no**. The only exactly-packable fields are sums whose low half cannot carry, which the tree already exploited.
  - Float atomics: RDNA4 has `shaderBufferFloat32AtomicAdd`/`MinMax`; panvk 26.3 has only load/store/exchange (`shaderBufferFloat32AtomicAdd = false`); no proprietary Mali exposes the extension. **Reject** (determinism).
- Verification: [verified from primary source] (reports, vi_local, in-tree notes).

### VK_KHR_workgroup_memory_explicit_layout — Khronos, 2021
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_workgroup_memory_explicit_layout.html (not re-opened this run)
- Technique: workgroup (shared) memory declared as explicitly laid-out blocks that can alias each other (e.g. view the same bytes as `uint` and `uint8_t`/`uint16_t`), with 8/16-bit and scalar layouts.
- Reported speed-up: none for this workload.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: the tree has only three shared arrays: `scan_block.comp` `uint temp[1024]`, `sort_points_local` `uint s_packed[kLocalCap=1024]` (4 KiB), and a single `wg_changed` flag in `uf_merge`. None needs type-punned aliasing or would shrink with it. Availability: RDNA4 and panvk 26.3 only (no proprietary Mali). On Mali shared memory is backed by L2 anyway (brief). **Reject.**
- Verification: availability/in-tree usage [verified from primary source]; mechanism [secondary/unverified] (spec knowledge).

## 13. Subgroup extras

### VK_KHR_shader_maximal_reconvergence, VK_KHR_shader_subgroup_rotate, VK_KHR_shader_quad_control, VK_EXT_shader_subgroup_partitioned — Khronos, 2023-2025
- URL: `claude-research/ext08/vk.xml` (names/support); gpuinfo 41734 raw HTML (`partitioned_bit_nv` shown with class `na`); vi_local.txt lines 741-752
- Technique: maximal reconvergence = SPIR-V guarantee that invocations that diverged reconverge at the earliest structured point (makes subgroup ops after divergent loops well-defined); rotate = `subgroupRotate`/clustered rotate; quad_control = quad-scope votes/derivatives (fragment-oriented); subgroup_partitioned = native reduce-by-key (`subgroupPartition` + partitioned reductions).
- Reported speed-up: none for this workload.
- Accuracy / determinism impact: none (exact integer ops).
- Applicability to vkapriltag:
  - **Mali:** subgroup aggregation is already measured and rejected (57% regression; reduce-by-key 6x slower than plain atomics), so none of these revives it. Availability for the record: maximal_reconvergence and rotate only in g29p1 strings / r54p2; quad_control on g24p0. **Partitioned is not supported on any G610** - the `partitioned_bit_nv` in gpuinfo's `subgroupSupportedOperations` is rendered with class `na`, i.e. absent (my feature parser initially misread it), and no driver lists `NV_`/`EXT_shader_subgroup_partitioned`.
  - **RDNA4:** rotate (`SUBGROUP_FEATURE_ROTATE_BIT`), maximal reconvergence and quad control are exposed; partitioned is not. Maximal reconvergence is a *correctness* aid for the existing `reduce_extents_hash_subgroup.comp` leader-election loop (ballot inside a loop in divergent code) - worth enabling if that variant is ever made the RDNA default, not a speed-up. The in-tree result that `blob_diff`/`uf_final` subgroup variants *lose* on the discrete part stands.
  - **Verdict: no action.**
- Verification: [verified from primary source] (vk.xml, report HTML, vi_local).

## 14. AMD vendor extensions (RX 9060 XT, driver 26.8.1)

### VK_AMD_* compute-relevant set, and VK_AMDX_shader_enqueue (work graphs) — AMD
- URL: `claude-research/ext08/vk.xml` (registered AMD names), `mali/e_rdna4.txt` / `vi_local.txt` (exposed set); https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMD_buffer_marker.html (not re-opened this run)
- Technique / applicability, one line each (exposed on the RX 9060 XT unless marked):
  - `AMD_shader_trinary_minmax` (`min3/max3/mid3`): matches `block_minmax` (4x4) and `block_filter` (3x3) exactly - but LLVM/LLPC already forms `v_min3`/`v_max3` from nested `min(min(a,b),c)`, so the explicit builtins should compile identically. Check once in `AMD_shader_info` disassembly; expected gain 0. Those stages are not the bottleneck anyway.
  - `AMD_buffer_marker` (`vkCmdWriteBufferMarker(2)AMD`: write a 32-bit value when a stage completes): a finer-grained "this point was reached" than a fence - usable as GPU crash breadcrumbs, or for the host to poll the four-submit path's counts slightly earlier than `vkWaitForFences` returns. The four-submit path still has to *record and submit* after the readback, so section 9's fusion dominates this. Diagnostic only.
  - `AMD_device_coherent_memory` (DEVICE_COHERENT / DEVICE_UNCACHED types 4-7): bypasses GPU caches; for breadcrumbs, slower for real data. No.
  - `AMD_shader_core_properties/2`: CU/SIMD/wave-slot counts (vi_local did not print the values) - only useful to size persistent-thread dispatches, which the tree does not use.
  - `AMD_shader_ballot`, `gcn_shader`, `gpu_shader_int16/half_float`: legacy, superseded by core subgroup/16-bit features.
  - `AMD_memory_overallocation_behavior`, `anti_lag`, `gpa_interface`: allocation policy / game input latency / profiler hook - no.
  - `AMD_pipeline_compiler_control`: registered but **not exposed** by 26.8.1.
  - `AMDX_shader_enqueue` (work graphs): the one AMD feature that could express the labelling loop plus the device-sized tail as a GPU-driven graph without host round trips. **Not exposed** by 26.8.1 (provisional AMDX, typically behind a beta driver), not on Mali. Reject for now; section 9 closes the same round trips portably.
- Reported speed-up: none for this workload.
- Accuracy / determinism impact: none.
- Verification: registry/availability [verified from primary source]; per-extension mechanics from spec knowledge [secondary/unverified] (not re-read); the min3 codegen claim is unverified until checked in disassembly.

## 15. Other

### VK_EXT_shader_object — Khronos (Nintendo/Valve/others), 2023
- URL: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_object.html (not re-opened this run)
- Technique: `VkShaderEXT` objects bound with `vkCmdBindShadersEXT` instead of pipelines; state is all dynamic.
- Reported speed-up: none for compute - a compute pipeline has no state to make dynamic.
- Accuracy / determinism impact: none.
- Applicability to vkapriltag: RDNA4 only (no Mali, no panvk in the table). The tree builds its ~20 compute pipelines once and caches them; shader objects offer nothing per frame and lose the pipeline cache path. **Reject.**
- Verification: availability [verified from primary source]; mechanism [secondary/unverified].

### Vulkan Video decode (VK_KHR_video_decode_*) and the MJPEG question — Khronos registry, header 364
- URL: `claude-research/ext08/vk.xml` (downloaded this run from KhronosGroup/Vulkan-Docs `main`)
- Technique: Vulkan Video decode queues for H.264, H.265, AV1 and VP9 (`VK_KHR_video_decode_h264/h265/av1/vp9`), encode for H.264/H.265/AV1.
- Reported speed-up: n/a.
- Accuracy / determinism impact: n/a (decode is exact for these codecs).
- Applicability to vkapriltag:
  - **There is no JPEG/MJPEG decode extension in the registry** (vk.xml header 364: zero occurrences of "jpeg"). USB cameras used with PhotonVision deliver MJPEG or YUYV, and the survey's SpectrumJetson finding is that JPEG decode, not the detector, is the real bottleneck there. So Vulkan Video cannot take the JPEG decode off the CPU on either device.
  - **Mali-G610:** no Vulkan Video at all in any report. The RK3588's own JPEG decoder is reached through Rockchip MPP (`MPP_VIDEO_CodingMJPEG`), whose DRM-PRIME output is a dma_buf - which is exactly section 1's import path. So the concrete pipeline is: MPP MJPEG decode -> dma_buf fd -> `VkImportMemoryFdInfoKHR` -> `decimate.comp` reads the Y plane. Bound: time `libjpeg-turbo` decode of a 1280x800 frame on an A76 vs MPP decode + import (category on ingest/CPU owns the measurement).
  - **RDNA4:** H.264/AV1 decode queues exist (`VK_KHR_video_decode_h264/av1` in the table); relevant only for network H.264 streams. The decode output is an NV12 image in VRAM - binding its luma plane as the detector input would skip a PCIe upload entirely, but it needs the image path the tree rejected (`texelFetch`/tiled, +29-35% threshold span) or a copy to a buffer. Low priority.
- Verification: [verified from primary source] (vk.xml grep, table). MPP details [secondary/unverified].

## Conclusions

28 entries plus the master availability table (section 0). Ranked by expected value for vkapriltag. "Bound" = the cheap measurement to run before building. Every idea below is bit-exact by construction (integer arithmetic, same bytes, or order-independent atomics) unless noted.

| # | Idea | Device | Stage | Expected benefit | Accuracy / determinism risk | Bound first by |
|---|---|---|---|---|---|---|
| 1 | **Turn on the fused single-submission tail on the discrete RDNA4.** Put `selected_extents_buf_` and `line_fit_points_buf_` in HOST_CACHED system memory (memory type 3), or keep them in VRAM and copy them out with `KHR_copy_memory_indirect` (or with an indirect-dispatched copy shader). See section 9. | RDNA4 | frame tail / submit boundaries | The fused path is off on RDNA4 only because no memory type is both device-local and host-cached. My estimate: 3 boundaries x >=42 us round trip = >=0.13 ms of the 0.66 ms GPU phase. On Mali the same fusion gave -2..-8%, then another -0.7..-5%. | Exact. The one risk is performance: in option A, later shaders read `selected[]` across PCIe (a few KB, latency-bound). | Force `HostVisibleCached` for the two buffers and set `fused_submits_ = true`; ABBA against `APRILTAG_VK_FUSE_SUBMITS=0` and check bit-identity. This re-proposes an in-tree rejection ("submit collapsing on dGPU"), with new evidence: the sizing machinery that rejection priced now ships. |
| 2 | **Give each shader core its own copy of the extents accumulator** using `gl_CoreIDARM` (`VK_ARM_shader_core_builtins`, exposed on the g24p0 deployment driver, 4 cores, IDs 0/2/16/18). This follows Arm BP 3.4 s10.13: "keep the atomic operations local to a single shader core". Separately, pad `MinMaxExtentsGpu` from 48 to 64 bytes. See section 11. | Mali | `reduce_extents_hash*`; possibly the `uf_final` `blob_size` histogram | Ceiling ~0.16 ms (~5% of GPU time): the gap between the current 0.221 ms and the 0.062 ms plain-store bound. The K=8/16/32 plateau fits cross-core cache-line migration, which workgroup-indexed copies cannot fix. | Exact (`merge_extents` folds the copies). Needs a vendor-extension gate and compaction of the sparse core IDs. | One env-gated shader variant; ABBA on the Pi at dec 1/2, with `$MaliCoreLoadStoreCyclesAtomicAccessCycles` as a secondary signal. |
| 3 | **Import frames by dma_buf** (`EXT_external_memory_dma_buf` + `KHR_external_memory_fd`) as a storage buffer with a stride push constant, importing each ring fd once. With MPP MJPEG decode this is the only way to take JPEG decode off the CPU, since Vulkan has no JPEG decode. | Mali (every driver) | ingest -> `decimate` | Removes the 0.22 ms upload memcpy (dec 1) and the capture app's YUYV->gray pass. With MPP it also removes CPU JPEG decode. | Exact. Risk: CPU `TagDecoder` reads from uncached CMA memory may be slower. | Time the upload, plus a CPU microbench of decoder-style reads from a V4L2/MPP-exported buffer. |
| 4 | **Sweep wave32/wave64 per pipeline** with `requiredSubgroupSize`, after dumping the driver's choice and VGPR counts with `KHR_pipeline_executable_properties`. | RDNA4 (Mali is fixed at 16) | `uf_*`, `hash_group`, `sort_points_local` | Unknown. AMD's guidance is wave32 for divergent, long-running code. | Exact for non-subgroup shaders; recheck the subgroup extents variant. | Env-gated create-info per pipeline, ABBA. |
| 5 | **Have the merge shader write the next `uf_compress`'s indirect args**, as a portable stand-in for conditional rendering (which no proprietary G610 driver exposes). | Mali | labelling loop | Ceiling ~0.7% of GPU time. | Exact. Unknown indirect-launch cost on CSF. | One ABBA run. |
| 6 | **`EXT_external_memory_host` import of a caller-provided ring** (4 KiB aligned, size rounded up). | RDNA4 now; Mali needs g29p1 | ingest | ~0.1-0.2 ms of serial CPU time. | Exact. Caller alignment; per-frame import cost. | Time the staging memcpy + copy on the desktop. |
| 7 | **Diagnostics:** `pipeline_executable_properties` (g29p1, RDNA4) for register and spill counts, `shader_clock` (RDNA4) for `find()`-walk histograms, calibrated timestamps for host/GPU gap accounting (supports #1). | both | tuning | Indirect. | None. | n/a |
| 8 | **Defensive:** `KHR_global_priority` HIGH with a MEDIUM fallback; `EXT_memory_priority` 1.0 on the hot buffers (RDNA4); query `defaultRobustnessStorageBuffers` whenever the driver is bumped. | both / RDNA4 | queue / allocation | 0 on an idle GPU; protects against contention. | None. | p99 GPU span with a competing load. |

Rejected, no action:
- **DGC:** cannot encode the labelling barriers, and an NVIDIA forum report shows a 20x slowdown for dispatch storms.
- **descriptor_buffer, push_descriptor, maintenance5-9:** API overhead only. g29p1 also has a descriptor-buffer bug with variable counts.
- **cooperative_matrix, integer_dot_product:** the pipeline has no GEMM or 8-bit dot shape, and the G610 has no cooperative matrix.
- **16-bit storage, fp16:** already rejected in-tree; fp16 is lossy.
- **Robustness extensions:** the tree already runs non-robust.
- **pipeline_binary:** startup only, and VkPipelineCache already exists.
- **timeline semaphores, sync2, host_query_reset:** plumbing only.
- **Float atomics:** order-dependent sums break bit-identity.
- **workgroup_memory_explicit_layout:** nothing to alias.
- **maximal_reconvergence, rotate, quad_control, partitioned:** Mali subgroup work is already rejected, and no G610 supports partitioned.
- **ARM:**
  - `scheduling_controls`: dispatch parameters are not exposed on any G610 driver and absent from the g24p0/g29p1 entry points.
  - `tensors`, `data_graph`, `shader_instrumentation`: absent.
  - `performance_counters_by_region`: render passes only.
- **AMD:** trinary_minmax (LLPC already forms min3/max3; unverified), buffer_marker, device_coherent_memory, AMDX_shader_enqueue (not exposed).
- **shader_object:** nothing for compute.
- **Vulkan Video:** no JPEG codec; H.264/AV1 decode on RDNA4 only matters for network streams.

What I couldn't verify:
- **g29p1 exposure on a G610:** every "g29p1 str" Y is only a string in the binary. No g29p1 `vulkaninfo` on a G610 exists (the wrapper docs are the only secondary evidence). Run `vulkaninfo` on the Pi after installing g29p1. It matters for #6, #7, `robustness2`, `pipeline_binary` and `maximal_reconvergence`.
- **`EXT_conditional_rendering` on G610 + g29p1:** probably not exposed, since r54p2 lacks it on the same GPU, although its entry points are in the g29p1 binary.
- **RDNA4 memory types:** read from the Windows 26.8.1 driver only. RADV's types, and whether the RX 9060 XT's 0.66 ms figure includes the submit gaps, were not checked. dma_buf on RDNA4 needs Linux (not checked).
- **Mechanism of #2:** the cross-core-migration diagnosis is my hypothesis. The cached G610 counter guide has no named L1-snoop counter.
- **Specs not re-read this run:** sync2, timeline, calibrated timestamps, workgroup_memory_explicit_layout, shader_object, the AMD one-liners and ARM_tensors; tagged [secondary/unverified] in their entries.
- **NVIDIA / Intel availability:** not spot-checked this run.
