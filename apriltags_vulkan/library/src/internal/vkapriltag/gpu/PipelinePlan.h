#pragma once

#include <cstdint>
#include <string>

#include "vkapriltag/common/Options.h"
#include "vkapriltag/gpu/DetectorConfig.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan {

// Every variant choice of the GPU frame, resolved once from the device, the configuration and the options.
struct PipelinePlan {
  // Launch geometry from the device.
  vk::WorkgroupSize wg1d;
  vk::WorkgroupSize wg2d;
  uint32_t scan_wg = 128;

  // Sort workgroup size, whole-blob limit and single-pass ranking limit.
  uint32_t local_sort_cap = 0;
  uint32_t local_sort_split = 0;
  uint32_t sort_rank_all = 256u;

  // Shader family and decimation.
  bool u8 = false;
  bool alias_decimated = false;
  bool box_decimate = false;
  bool box_dot = false;
  bool threshold_block = false;
  bool minmax_decimate = false;
  bool adaptive_floor = false;
  uint32_t floor_scale_tenths = 0;

  // Labelling.
  bool uf_verify = false;
  bool predicated_compress = false;
  bool uf_init_scan = false;
  bool count_runs = false;
  bool sparse = false;
  uint32_t merge_flag_mode = 0;
  bool merge_x4 = false;
  uint32_t merge_band = 0;

  // Boundary reduction.
  bool subgroup = false;
  bool extents_atomic64 = false;
  bool reduce_x4 = false;

  // Resolved after the buffers exist (FinalisePlan).
  bool gray_direct_write = false;
  bool linefit_direct_read = false;
  bool extents_direct_read = false;
  bool fused_submits = false;
  bool prerecord = false;
  uint32_t lean_tail = 0;

  // One "name value" line per choice, in declaration order.
  std::string Describe() const;
};

// The choices that depend only on the device, the (normalised) configuration and the options.
PipelinePlan ResolvePlan(const vk::DeviceCaps &caps, const DetectorConfig &config, const Options &env);

// What the allocation of the frame and readback buffers actually returned.
struct BufferOutcome {
  bool gray_direct_write = false;
  bool linefit_direct_read = false;
  bool extents_direct_read = false;
};

// The choices that depend on the buffers: fused submits, pre-recording and the lean tail.
void FinalisePlan(PipelinePlan &plan, const vk::DeviceCaps &caps, const Options &env, bool has_transfer_lane,
                  const BufferOutcome &outcome);

uint32_t NextPow2(uint32_t v);
uint32_t PrevPow2(uint32_t v);

}  // namespace apriltag_vulkan
