#include "vkapriltag/gpu/PipelinePlan.h"

#include <algorithm>
#include <sstream>

namespace apriltag_vulkan {

uint32_t NextPow2(uint32_t v) {
  if (v < 1) return 1;
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

// Largest power of two <= v (v is raised to 1).
uint32_t PrevPow2(uint32_t v) {
  v = std::max(v, 1u);
  return NextPow2(v / 2 + 1);
}

PipelinePlan ResolvePlan(const vk::DeviceCaps &caps, const DetectorConfig &config, const Options &env) {
  PipelinePlan p;
  const bool integrated = caps.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  const uint32_t decimated_width = config.width / config.decimation;

  // Launch geometry comes from the device, never from a literal.
  p.wg1d = vk::WorkgroupSize{caps.wg1d, 1, 1};
  p.wg2d = vk::WorkgroupSize{caps.wg2d_x, caps.wg2d_y, 1};
  p.scan_wg = caps.scan_wg;

  // sort_points_local.comp workgroup size: the device's maximum invocation count rounded down to a
  // power of two, limited to 256 threads on integrated GPUs and 1,024 on others.
  p.local_sort_cap = PrevPow2(std::max(caps.max_workgroup_invocations, 1u));
  p.local_sort_cap = std::min(p.local_sort_cap, integrated ? 256u : 1024u);
  if (const auto v = env.ULong("APRILTAG_VK_SORT_WG")) {
    const uint32_t wanted = static_cast<uint32_t>(*v);
    if (wanted >= 1u) {
      p.local_sort_cap = std::min(PrevPow2(wanted), PrevPow2(std::max(caps.max_workgroup_invocations, 1u)));
    }
  }

  // Blobs of up to local_sort_split points sort whole in shared memory, larger ones in runs of that
  // length; at most 4096 (the 12-bit packed index) and bounded by shared memory.
  const uint32_t shared_cap = std::min(4096u, PrevPow2(std::max(caps.max_shared_memory_bytes / 4u, 1u)));
  p.local_sort_split = std::min(1024u, shared_cap);
  if (const auto v = env.ULong("APRILTAG_VK_SORT_SPLIT")) {
    const uint32_t wanted = static_cast<uint32_t>(*v);
    if (wanted >= 1u) p.local_sort_split = PrevPow2(wanted);
  }
  p.local_sort_split = std::min(p.local_sort_split, shared_cap);
  if (const auto v = env.ULong("APRILTAG_VK_SORT_RANK_ALL")) {
    p.sort_rank_all = static_cast<uint32_t>(*v) & ~3u;
  }
  // The shared memory of sort_points_local (two arrays of local_sort_split words and the ranking array) must fit.
  p.sort_rank_all = std::min(p.sort_rank_all, p.local_sort_split);
  const uint32_t shared_words = caps.max_shared_memory_bytes / 4u;
  while (p.local_sort_split > 1u && p.local_sort_split * 2u + p.sort_rank_all > shared_words) {
    p.local_sort_split /= 2u;
    p.sort_rank_all = std::min(p.sort_rank_all, p.local_sort_split) & ~3u;
  }

  p.uf_verify = env.Bool("APRILTAG_VK_UF_VERIFY", false);
  p.predicated_compress = caps.has_conditional_rendering;

  // The 8-bit-storage shader family; the decimated and thresholded buffers switch together.
  p.u8 = caps.has_8bit_storage;
  // At decimation 1 with one byte per pixel the decimated image is the grey buffer itself.
  p.alias_decimated = p.u8 && config.decimation == 1;

  // Subgroup aggregation is used only for the extents reduction, and not on integrated GPUs.
  p.subgroup = caps.has_subgroup_ballot && caps.has_subgroup_arithmetic && caps.has_subgroup_shuffle && !integrated;

  p.box_decimate = env.Bool("APRILTAG_VK_BOX_DECIMATE", config.box_decimate) && config.decimation > 1;
  const bool dot_off = env.Has("APRILTAG_VK_DOT") && !env.Bool("APRILTAG_VK_DOT", true);
#ifdef VKAPRILTAG_HAVE_DOT_SHADER
  p.box_dot = p.box_decimate && p.u8 && caps.has_integer_dot_product &&
              (config.decimation == 2 || config.decimation == 4) && !dot_off;
#else
  (void)dot_off;
  p.box_dot = false;
#endif

  // One thread per 4x4 block; the adaptive contrast floor is set per frame on the device and needs it.
  p.threshold_block =
      p.u8 && !(env.Has("APRILTAG_VK_FUSED_THRESHOLD") && !env.Bool("APRILTAG_VK_FUSED_THRESHOLD", true));
  // The host decimates into a mapped decimated buffer; FinalisePlan confirms it once the buffer exists.
  p.want_cpu_decimate = (config.decimation == 2 || config.decimation == 4) && env.Bool("APRILTAG_VK_CPU_DECIMATE", true);
  p.minmax_decimate = p.threshold_block && !p.alias_decimated && !p.box_decimate && !p.want_cpu_decimate;
  const double floor_scale = env.Float("APRILTAG_VK_ADAPTIVE_FLOOR", config.adaptive_floor_scale);
  p.floor_scale_tenths = static_cast<uint32_t>(std::max(0.0, floor_scale * 10.0 + 0.5));
  p.adaptive_floor = p.threshold_block && p.floor_scale_tenths > 0;

  // Needs subgroup arithmetic and at most 64 subgroups per workgroup.
  p.uf_init_scan = !p.uf_verify && caps.has_subgroup_arithmetic && caps.subgroup_size > 0 &&
                   p.wg1d.x / caps.subgroup_size <= 64u;
  if (env.Has("APRILTAG_VK_UF_INIT_SCAN")) {
    p.uf_init_scan = p.uf_init_scan && env.Bool("APRILTAG_VK_UF_INIT_SCAN", true);
  }
  // Blob sizes counted per run piece on integrated GPUs with uf_init_scan.
  p.count_runs = p.uf_init_scan && integrated;
  if (env.Has("APRILTAG_VK_COUNT_RUNS")) {
    p.count_runs = p.uf_init_scan && env.Bool("APRILTAG_VK_COUNT_RUNS", true);
  }
  const bool want_sparse = env.Bool("APRILTAG_VK_SPARSE", config.sparse_labelling);
  p.sparse = p.threshold_block && p.uf_init_scan && p.count_runs && want_sparse;

  // Convergence-flag write mode: a read-guarded global atomic on unified memory, else shared-memory aggregation.
  p.merge_flag_mode = caps.unified_memory ? 1u : 0u;
  // Four pixels per thread on integrated GPUs, with 8-bit storage and a width that is a multiple of 4.
  const bool merge_x4_possible = p.u8 && decimated_width % 4u == 0u;
  p.merge_x4 = merge_x4_possible && integrated;
  if (env.Has("APRILTAG_VK_MERGE_X4")) {
    p.merge_x4 = merge_x4_possible && env.Bool("APRILTAG_VK_MERGE_X4", true);
  }
  // Rows merged in bands of this many (a power of two) before the band boundaries, on discrete GPUs.
  p.merge_band = integrated || p.sparse ? 0u : 8u;
  if (const auto v = env.ULong("APRILTAG_VK_MERGE_BAND")) {
    const uint32_t wanted = static_cast<uint32_t>(*v);
    p.merge_band = wanted >= 2u ? PrevPow2(wanted) : 0u;
  }

  // Reduction variant priority: subgroup, then 64-bit atomic, then plain 32-bit; the atomic variants take 4 points per thread.
  p.extents_atomic64 = !p.subgroup && caps.has_int64_atomics;
  p.reduce_x4 = !p.subgroup;
  return p;
}

void FinalisePlan(PipelinePlan &plan, const vk::DeviceCaps &caps, const Options &env, bool has_transfer_lane,
                  const BufferOutcome &outcome) {
  plan.gray_direct_write = outcome.gray_direct_write;
  plan.linefit_direct_read = outcome.linefit_direct_read;
  plan.extents_direct_read = outcome.extents_direct_read;

  // Not with the transfer-queue upload, which keeps the staged full frame.
  plan.cpu_decimate = plan.want_cpu_decimate && outcome.decimated_direct_write && !has_transfer_lane;
  plan.minmax_decimate = plan.threshold_block && !plan.alias_decimated && !plan.box_decimate && !plan.cpu_decimate;

  // With both readbacks in place, the last three submissions record as one.
  plan.fused_submits = outcome.linefit_direct_read && outcome.extents_direct_read;
  if (env.Has("APRILTAG_VK_FUSE_SUBMITS")) {
    plan.fused_submits = env.Bool("APRILTAG_VK_FUSE_SUBMITS", true) && plan.fused_submits;
  }

  // Replays pre-recorded command buffers on the fused path; not used with the transfer-queue upload.
  plan.prerecord = plan.fused_submits && !has_transfer_lane;
  if (env.Has("APRILTAG_VK_PRERECORD")) {
    plan.prerecord = plan.prerecord && env.Bool("APRILTAG_VK_PRERECORD", true);
  }

  // A leaner tail where a dispatch is costly (bit 1: fixed-grid kernels, bit 2: offsets from select).
  plan.lean_tail = caps.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3u : 0u;
  if (const auto v = env.ULong("APRILTAG_VK_LEAN_TAIL")) plan.lean_tail = static_cast<uint32_t>(*v);
  if (!plan.fused_submits) plan.lean_tail &= ~2u;
  // The fixed-grid kernels need the 4-point reduction and the fused path, not the subgroup reduction.
  if (!(plan.fused_submits && plan.reduce_x4 && !plan.subgroup)) plan.lean_tail &= ~1u;
}

std::string PipelinePlan::Describe() const {
  std::ostringstream os;
  auto line = [&os](const char *name, const auto &value) { os << name << ' ' << value << '\n'; };
  line("wg1d", wg1d.x);
  line("wg2d", std::to_string(wg2d.x) + "x" + std::to_string(wg2d.y));
  line("scan_wg", scan_wg);
  line("local_sort_cap", local_sort_cap);
  line("local_sort_split", local_sort_split);
  line("sort_rank_all", sort_rank_all);
  line("u8", u8);
  line("alias_decimated", alias_decimated);
  line("box_decimate", box_decimate);
  line("box_dot", box_dot);
  line("threshold_block", threshold_block);
  line("minmax_decimate", minmax_decimate);
  line("cpu_decimate", cpu_decimate);
  line("adaptive_floor", adaptive_floor);
  line("floor_scale_tenths", floor_scale_tenths);
  line("uf_verify", uf_verify);
  line("predicated_compress", predicated_compress);
  line("uf_init_scan", uf_init_scan);
  line("count_runs", count_runs);
  line("sparse", sparse);
  line("merge_flag_mode", merge_flag_mode);
  line("merge_x4", merge_x4);
  line("merge_band", merge_band);
  line("subgroup", subgroup);
  line("extents_atomic64", extents_atomic64);
  line("reduce_x4", reduce_x4);
  line("gray_direct_write", gray_direct_write);
  line("linefit_direct_read", linefit_direct_read);
  line("extents_direct_read", extents_direct_read);
  line("fused_submits", fused_submits);
  line("prerecord", prerecord);
  line("lean_tail", lean_tail);
  return os.str();
}

}  // namespace apriltag_vulkan
