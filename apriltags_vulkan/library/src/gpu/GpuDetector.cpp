#include "vkapriltag/gpu/GpuDetector.h"

#include "vkapriltag/common/Profiler.h"
#include "vkapriltag/vk/EmbeddedShaders.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace apriltag_vulkan {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Same timeline as prof::NowNs().
prof::Ns ToNs(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

// Prefers SPIR-V embedded in the binary, falling back to SHADER_DIR when the build has none
// (VKAPRILTAG_EMBED_SHADERS=OFF).
vk::ShaderSource ShaderPath(const char *name) {
  if (const vk::EmbeddedShader *embedded = vk::FindEmbeddedShader(name)) {
    return vk::ShaderSource(embedded->code, embedded->bytes, name);
  }
  return vk::ShaderSource(std::string(SHADER_DIR) + "/" + name + ".comp.spv");
}

uint32_t NextPow2(uint32_t v) {
  if (v < 1) return 1;
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

// Largest power of two <= v (v must be >= 1).
uint32_t PrevPow2(uint32_t v) {
  v = std::max(v, 1u);
  return NextPow2(v / 2 + 1);
}

const VkBufferUsageFlags kSsboUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

// Offsets (in 4-byte slots) into the shared counter readback staging buffer.
constexpr uint32_t kSlotUfChanged = 0;
constexpr uint32_t kSlotQbpCount = 1;
constexpr uint32_t kSlotSelectedCount = 2;
constexpr uint32_t kSlotPointCount = 3;
constexpr uint32_t kSlotRawBlobs = 4;
constexpr uint32_t kSlotHashDrops = 5;
constexpr uint32_t kSlotOversizedSortBlobs = 6;
constexpr VkDeviceSize kCounterStagingBytes = 64;

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : GpuDetector(ctx, ctx.default_lane(), config) {}

GpuDetector::GpuDetector(vk::Context &ctx, vk::Lane &lane, const DetectorConfig &config)
    : GpuDetector(ctx, lane, nullptr, config) {}

GpuDetector::~GpuDetector() {
  // Detect() returns only after its submission completes, so nothing recorded is in flight.
  for (const RecordedFrame &r : recorded_frames_) lane_.FreePersistentCommands(r.cmd);
  if (upload_semaphore_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx_.device());
    vkDestroySemaphore(ctx_.device(), upload_semaphore_, nullptr);
  }
}

GpuDetector::GpuDetector(vk::Context &ctx, vk::Lane &lane, vk::Lane *transfer_lane,
                         const DetectorConfig &config)
    : ctx_(ctx), lane_(lane), transfer_lane_(transfer_lane), config_(config) {
  if (transfer_lane_ != nullptr) {
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vk::CheckVk(vkCreateSemaphore(ctx_.device(), &sem_info, nullptr, &upload_semaphore_),
            "vkCreateSemaphore");
  }
  if (config_.decimation == 0) {
    throw std::runtime_error("decimation must be >= 1");
  }
  if (config_.width % config_.decimation != 0 || config_.height % config_.decimation != 0) {
    throw std::runtime_error("width and height must each be evenly divisible by decimation");
  }
  // QBPoint/IPoint pack the doubled decimated coordinate (at most 2*width/decimation) into 14 bits
  // per axis; rejects configurations that would truncate it.
  const uint32_t max_packed_x = 2u * (config_.width / config_.decimation);
  const uint32_t max_packed_y = 2u * (config_.height / config_.decimation);
  if (max_packed_x > 16383 || max_packed_y > 16383) {
    throw std::runtime_error(
        "2*(width/decimation) and 2*(height/decimation) must each be <= 16383");
  }
  // RawLineFitPoint packs blob_index into 22 bits (see common.glsl), bounding max_raw_blobs.
  static_assert(sizeof(RawLineFitPoint) == 8, "RawLineFitPoint packing changed");
  if (config_.max_raw_blobs > (1u << 22)) {
    throw std::runtime_error(
        "max_raw_blobs must be <= 2^22 to fit RawLineFitPoint's packed blob_index");
  }

  // --- Environment overrides. These must all be applied before any capacity
  // or launch geometry is derived from the config. ---

  // Caps the boundary-point count, and with it device memory.
  if (const char *cap = std::getenv("APRILTAG_VK_MAX_POINTS")) {
    const long parsed = std::strtol(cap, nullptr, 10);
    if (parsed > 0) config_.max_boundary_points = static_cast<uint32_t>(parsed);
  }
  // Labelling iterations per chunk.
  if (const char *chunk = std::getenv("APRILTAG_VK_UF_CHUNK")) {
    const long parsed = std::strtol(chunk, nullptr, 10);
    if (parsed > 0) config_.uf_iterations_per_chunk = static_cast<uint32_t>(parsed);
  }
  if (config_.uf_iterations_per_chunk == 0) config_.uf_iterations_per_chunk = 1;

  // Derives a minimum cluster size from the smallest tag side (full-resolution pixels) and raises
  // min_cluster_pixels to it if lower.
  if (const char *min_tag_px = std::getenv("APRILTAG_VK_MIN_TAG_PX")) {
    const long parsed = std::strtol(min_tag_px, nullptr, 10);
    if (parsed > 0) config_.min_tag_pixels = static_cast<uint32_t>(parsed);
  }
  if (config_.min_tag_pixels > 0) {
    // Perimeter, in decimated pixels, of a square tag whose side is min_tag_pixels.
    const uint32_t derived_min_cluster = 4u * (config_.min_tag_pixels / config_.decimation);
    config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, derived_min_cluster);
  }
  // Floors min_cluster_pixels at 2, as the ambiguous-pixel paths in uf_final.comp and
  // blob_diff.comp require.
  config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, 2u);

  // Launch geometry comes from the device, never from a literal.
  const vk::DeviceCaps &caps = ctx_.caps();
  wg1d_ = vk::WorkgroupSize{caps.wg1d, 1, 1};
  wg2d_ = vk::WorkgroupSize{caps.wg2d_x, caps.wg2d_y, 1};
  scan_wg_ = caps.scan_wg;


  decimated_width_ = config_.width / config_.decimation;
  decimated_height_ = config_.height / config_.decimation;

  // Resolves max_blobs == 0 (auto) in proportion to the decimated pixel count, anchored at 2048 for
  // 1080p at decimation 2 and clamped to max_raw_blobs.
  if (config_.max_blobs == 0) {
    constexpr uint64_t kAnchorDecimatedPx = 1920ull * 1080ull / 4ull;  // 1080p at decimation 2
    constexpr uint64_t kAnchorMaxBlobs = 2048ull;
    const uint64_t decimated_px =
        uint64_t(decimated_width_) * uint64_t(decimated_height_);
    const uint64_t scaled = kAnchorMaxBlobs * decimated_px / kAnchorDecimatedPx;
    config_.max_blobs = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(scaled, kAnchorMaxBlobs), config_.max_raw_blobs));
  }
  // Rounded up; block_minmax.comp and threshold.comp handle the ragged trailing block.
  block_width_ = (decimated_width_ + 3) / 4;
  block_height_ = (decimated_height_ + 3) / 4;
  interior_width_ = decimated_width_ - 2;
  interior_height_ = decimated_height_ - 2;
  dense_qbp_count_ = 4u * interior_width_ * interior_height_;

  qbp_capacity_ = dense_qbp_count_;
  if (config_.max_boundary_points > 0) {
    qbp_capacity_ = std::min(qbp_capacity_, config_.max_boundary_points);
  }
  ipoint_capacity_ = qbp_capacity_;

  // The grayscale frame is uploaded packed, four 8-bit pixels per uint32 (width and height are
  // even, so the product divides by four).
  gray_words_ = (config_.width * config_.height) / 4u;

  // sort_points_local.comp workgroup size: the device's maximum invocation count rounded down to a
  // power of two.
  local_sort_cap_ = PrevPow2(std::max(caps.max_workgroup_invocations, 1u));
  // Limited to 256 threads; APRILTAG_VK_SORT_WG overrides it.
  local_sort_cap_ = std::min(local_sort_cap_, 256u);
  if (const char *v = std::getenv("APRILTAG_VK_SORT_WG")) {
    const uint32_t wanted = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
    if (wanted >= 1u) {
      local_sort_cap_ = std::min(PrevPow2(wanted), PrevPow2(std::max(caps.max_workgroup_invocations, 1u)));
    }
  }

  // Sort slots per blob: 2048 at decimation 2, scaling with 1/decimation and never below 2048.
  const uint32_t kCapAtDecimation2 = 2048u;
  const uint32_t scaled_cap = (kCapAtDecimation2 * 2u) / config_.decimation;
  // At most 4096, the limit of sort_points_local_body.glsl's 12-bit local index (kLocalIndexBits),
  // and bounded by shared memory.
  local_sort_virtual_cap_ =
      std::min(std::max(scaled_cap, kCapAtDecimation2), 4096u);
  local_sort_virtual_cap_ = std::min(
      local_sort_virtual_cap_, PrevPow2(std::max(caps.max_shared_memory_bytes / 4u, 1u)));

  CreateBuffers();
  CreatePipelines();

  // Persists newly compiled pipelines immediately rather than only at ~Context().
  ctx_.FlushPipelineCache();

  // Opt-in per-shader GPU timing; tracing also enables it for the GPU track.
  bool want_timestamps = prof::Enabled();
  if (const char *ts = std::getenv("APRILTAG_VK_TIMESTAMPS")) {
    want_timestamps = want_timestamps || std::strtol(ts, nullptr, 10) != 0;
  }
  if (want_timestamps && caps.timestamps_supported) {
    timestamp_pool_ = vk::QueryPool(ctx_.device(), kNumGpuStageSpans * 2);
    timestamps_enabled_ = timestamp_pool_.valid();
  }
  if (VKAPRILTAG_PROFILING) gpu_track_ = prof::Track("GPU");
}

void GpuDetector::CreateBuffers() {
  auto ssbo = [this](VkDeviceSize bytes) {
    vk::Buffer b(ctx_, std::max<VkDeviceSize>(bytes, 4), kSsboUsage, vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    return b;
  };

  // Host-visible device memory written directly on unified-memory parts; otherwise device-local,
  // filled through a persistently mapped staging buffer.
  gray_buf_ = vk::Buffer(
      ctx_, VkDeviceSize(gray_words_) * 4,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      ctx_.caps().unified_memory ? vk::MemoryKind::DeviceLocalMapped
                                 : vk::MemoryKind::DeviceLocal,
      transfer_lane_ != nullptr ? ctx_.queue_families() : std::vector<uint32_t>{});
  device_bytes_ += gray_buf_.size();
  gray_direct_write_ = gray_buf_.host_visible();

  if (!gray_direct_write_) {
    upload_staging_ = vk::Buffer(ctx_, VkDeviceSize(gray_words_) * 4,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::MemoryKind::HostVisible);
  }
  counter_staging_ = vk::Buffer(ctx_, kCounterStagingBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                vk::MemoryKind::HostVisibleCached);

  // 1 byte per pixel with storageBuffer8BitAccess, else 4. decimated_buf_ and thresholded_buf_ share
  // the setting.
  const VkDeviceSize decimated_bytes_per_pixel = ctx_.caps().has_8bit_storage ? 1 : 4;
  decimated_buf_ =
      ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel);
  minmax_unfiltered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  thresholded_buf_ =
      ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel);

  parent_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  // Also the conditional-rendering predicate for uf_compress (see predicated_compress_).
  predicated_compress_ = ctx_.caps().has_conditional_rendering;
  {
    const VkBufferUsageFlags usage =
        kSsboUsage | (predicated_compress_ ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0);
    vk::Buffer b(ctx_, 4, usage, vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    uf_changed_buf_ = std::move(b);
  }

  // QBPoint is packed into one uint32 on the GPU (see common.glsl); there is no host mirror.
  qbp_compacted_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * sizeof(uint32_t));
  qbp_counter_buf_ = ssbo(4);
  // Sized to the point capacity; one interleaved (rep0, rep1) uvec2 key per point.
  qbp_keys_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 8);

  extents_buf_ = ssbo(VkDeviceSize(ExtentsSlotCount(config_.max_raw_blobs)) *
                      sizeof(MinMaxExtentsGpu));
  // Host-visible and cached where available, so QuadDecode reads it in place (see fused_submits_);
  // otherwise staged.
  {
    vk::Buffer b(ctx_, VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu), kSsboUsage,
                 vk::MemoryKind::HostVisibleCached);
    device_bytes_ += b.size();
    selected_extents_buf_ = std::move(b);
  }
  extents_direct_read_ =
      selected_extents_buf_.host_visible() && selected_extents_buf_.host_cached();
  selected_counter_buf_ = ssbo(4);
  remap_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * 4);

  index_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  blob_point_offsets_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * 4);

  // Host-visible and cached where available, so QuadDecode reads it in place; otherwise staged.
  {
    vk::Buffer b(ctx_, VkDeviceSize(ipoint_capacity_) * sizeof(RawLineFitPoint), kSsboUsage,
                 vk::MemoryKind::HostVisibleCached);
    device_bytes_ += b.size();
    line_fit_points_buf_ = std::move(b);
  }
  linefit_direct_read_ = line_fit_points_buf_.host_visible() && line_fit_points_buf_.host_cached();

  // With both readbacks in place, the last three submissions record as one (see fused_submits_ and
  // build_indirect_args.comp).
  fused_submits_ = linefit_direct_read_ && extents_direct_read_;
  if (const char *v = std::getenv("APRILTAG_VK_FUSE_SUBMITS")) {
    fused_submits_ = (v[0] != '0') && linefit_direct_read_ && extents_direct_read_;
  }

  // Replays pre-recorded command buffers on the fused path (APRILTAG_VK_PRERECORD=0 records every
  // frame); not used with the transfer-queue upload.
  prerecord_ = fused_submits_ && transfer_lane_ == nullptr;
  if (const char *v = std::getenv("APRILTAG_VK_PRERECORD")) {
    prerecord_ = prerecord_ && v[0] != '0';
  }

  // Open-addressing table for the (rep0, rep1) grouping, sized to max_raw_blobs; hash_group.comp
  // drops points beyond its probe cap.
  hash_table_size_ = NextPow2(std::max(config_.max_raw_blobs, 1u));
  hash_owner_buf_ = ssbo(VkDeviceSize(hash_table_size_) * 4);
  slot_dense_buf_ = ssbo(VkDeviceSize(hash_table_size_) * 4);
  point_slot_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);
  blob_cursor_buf_ = ssbo(VkDeviceSize(std::max(config_.max_blobs, 1u)) * 4);
  raw_blob_counter_buf_ = ssbo(4);
  hash_drop_counter_buf_ = ssbo(4);
  oversized_sort_counter_buf_ = ssbo(4);

  // VkDispatchIndirectCommands built on the device by build_indirect_args_pl_.
  {
    // Raw blobs, boundary points, selected blobs.
    vk::Buffer b(ctx_, 3 * 12, kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    indirect_args_buf_ = std::move(b);
  }

  blob_scan_chain_ = BuildScanChain(std::max(config_.max_blobs, 1u));

  // Readback staging starts at a fixed size and grows on demand.
  const VkDeviceSize initial_readback =
      VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu) + 16 +
      VkDeviceSize(std::min<uint32_t>(ipoint_capacity_, 1u << 16)) * sizeof(RawLineFitPoint);
  EnsureReadbackCapacity(std::max<VkDeviceSize>(initial_readback, 4096));
}

void GpuDetector::EnsureReadbackCapacity(VkDeviceSize bytes) {
  if (bytes <= readback_capacity_) return;
  // Grows geometrically.
  VkDeviceSize new_capacity = std::max<VkDeviceSize>(readback_capacity_ * 2, bytes);
  readback_staging_ = vk::Buffer(ctx_, new_capacity, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 vk::MemoryKind::HostVisibleCached);
  readback_capacity_ = readback_staging_.size();
}

GpuDetector::ScanChain GpuDetector::BuildScanChain(uint32_t capacity) {
  ScanChain chain;
  uint32_t size = capacity;
  // Fan-out per level is the scan block size, matching scan_block.comp's local_size_x.
  while (size > scan_wg_) {
    uint32_t next = (size + scan_wg_ - 1) / scan_wg_;
    chain.level_capacities.push_back(next);
    vk::Buffer b(ctx_, VkDeviceSize(std::max(next, 1u)) * 4, kSsboUsage,
                 vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    chain.level_buffers.push_back(std::move(b));
    size = next;
  }
  return chain;
}

void GpuDetector::CreatePipelines() {
  // Picks the _u8 shader variant with storageBuffer8BitAccess; decimated_buf_ and thresholded_buf_
  // switch together.
  const bool u8 = ctx_.caps().has_8bit_storage;
  auto pick = [u8](const char *base_name, const char *u8_name) {
    return u8 ? u8_name : base_name;
  };

  // Subgroup aggregation is used only for reduce_extents_hash, and not on integrated GPUs. Needs
  // ballot, arithmetic and shuffle.
  const bool subgroup = ctx_.caps().has_subgroup_ballot && ctx_.caps().has_subgroup_arithmetic &&
                        ctx_.caps().has_subgroup_shuffle &&
                        ctx_.caps().type != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

  // 2D dispatches recover (x, y) from gl_GlobalInvocationID.xy without a divide. decimation is
  // specialization constant 3, after the workgroup size's 0-2.
  decimate_pl_ = vk::ComputePipeline(ctx_, ShaderPath(pick("decimate", "decimate_u8")),
                                     {gray_buf_.get(), decimated_buf_.get()}, 8, wg2d_,
                                     {config_.decimation});
  block_minmax_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("block_minmax", "block_minmax_u8")),
      {decimated_buf_.get(), minmax_unfiltered_buf_.get()}, 16, wg2d_);
  block_filter_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_filter"), {minmax_unfiltered_buf_.get(), minmax_filtered_buf_.get()},
      8, wg2d_);
  threshold_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("threshold", "threshold_u8")),
      {decimated_buf_.get(), minmax_filtered_buf_.get(), thresholded_buf_.get()}, 16, wg2d_);

  uf_init_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_init", "uf_init_u8")), {parent_buf_.get(), thresholded_buf_.get()},
      8, wg2d_);
  // Dispatched 2D with a one-row-tall workgroup. find_mode 1 is path splitting, 0 is naive;
  // APRILTAG_VK_FIND_MODE overrides.
  uint32_t find_mode = 1u;
  if (const char *v = std::getenv("APRILTAG_VK_FIND_MODE")) {
    find_mode = (v[0] != '0') ? 1u : 0u;
  }
  // Convergence-flag write mode (see uf_merge_body.glsl): 1 (read-guarded global atomicOr) on
  // unified memory, otherwise 0 (shared-memory aggregation).
  uint32_t merge_flag_mode = ctx_.caps().unified_memory ? 1u : 0u;
  if (const char *v = std::getenv("APRILTAG_VK_MERGE_FLAG_MODE")) {
    merge_flag_mode = (v[0] != '0') ? 1u : 0u;
  }
  uf_merge_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_merge", "uf_merge_u8")),
      {parent_buf_.get(), thresholded_buf_.get(), uf_changed_buf_.get()}, 8,
      vk::WorkgroupSize{wg1d_.x, 1, 1}, {find_mode, merge_flag_mode});
  // Binding 1 / the third push constant are the convergence flag and the
  // opt-in to honouring it; see uf_compress.comp.
  uf_compress_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_compress"), {parent_buf_.get(), uf_changed_buf_.get()}, 12, wg1d_);
  // As uf_compress plus uf_final's count: bindings 2 and 3 are the thresholded codes and blob_size.
  uf_compress_count_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_compress_count", "uf_compress_count_u8")),
      {parent_buf_.get(), uf_changed_buf_.get(), thresholded_buf_.get(), blob_size_buf_.get()}, 16,
      wg1d_);
  // Binding 2 lets the shader skip ambiguous (127) pixels.
  uf_final_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_final", "uf_final_u8")),
      {parent_buf_.get(), blob_size_buf_.get(), thresholded_buf_.get()}, 12, wg1d_);

  // Binding 4 is uf_changed_buf_, read only by the fused path's speculative attempt. Bindings 5 and 6
  // let the shader test a pixel's blob size and read the threshold codes itself.
  blob_diff_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("blob_diff", "blob_diff_u8")),
      {parent_buf_.get(), qbp_compacted_buf_.get(), qbp_counter_buf_.get(),
       qbp_keys_buf_.get(), uf_changed_buf_.get(), blob_size_buf_.get(), thresholded_buf_.get()},
      20, wg2d_);

  init_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("init_extents"), {extents_buf_.get()}, 4, wg1d_);
  merge_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("merge_extents"), {extents_buf_.get()}, 4, wg1d_);

  select_blobs_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("select_blobs"),
      {extents_buf_.get(), selected_extents_buf_.get(), selected_counter_buf_.get(),
       remap_buf_.get()},
      44, wg1d_);

  // Per-blob point base offsets: extract_blob_counts.comp copies each selected blob's count, then a
  // chain scan makes the inclusive prefix sum.
  extract_blob_counts_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("extract_blob_counts"),
      {selected_extents_buf_.get(), selected_counter_buf_.get(), blob_point_offsets_buf_.get()}, 4,
      wg1d_);
  {
    const vk::WorkgroupSize scan_wg{scan_wg_, 1, 1};
    VkBuffer prev_values = blob_point_offsets_buf_.get();
    VkBuffer prev_output = blob_point_offsets_buf_.get();
    for (size_t i = 0; i < blob_scan_chain_.level_buffers.size(); ++i) {
      VkBuffer block_sums = blob_scan_chain_.level_buffers[i].get();
      blob_scan_block_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_block"), {prev_values, prev_output, block_sums}, 4, scan_wg));
      prev_values = block_sums;
      prev_output = block_sums;
    }
    blob_scan_block_pls_.push_back(vk::ComputePipeline(
        ctx_, ShaderPath("scan_block"), {prev_values, prev_output, prev_output}, 4, scan_wg));

    std::vector<VkBuffer> level_values = {blob_point_offsets_buf_.get()};
    for (auto &buf : blob_scan_chain_.level_buffers) level_values.push_back(buf.get());
    for (size_t i = 0; i + 1 < level_values.size(); ++i) {
      blob_scan_add_offsets_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_add_offsets"), {level_values[i], level_values[i + 1]}, 4,
          scan_wg));
    }
  }

  // Rank-merge sort run length (a power of two, 0 for the sorting network);
  // APRILTAG_VK_SORT_CHUNK overrides it.
  uint32_t sort_chunk = 16u;
  if (const char *v = std::getenv("APRILTAG_VK_SORT_CHUNK")) {
    sort_chunk = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
    if (sort_chunk != 0u && (sort_chunk & (sort_chunk - 1u)) != 0u) sort_chunk = 0u;
  }
  sort_points_local_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("sort_points_local", "sort_points_local_u8")),
      {selected_extents_buf_.get(), blob_point_offsets_buf_.get(), index_points_buf_.get(),
       decimated_buf_.get(), line_fit_points_buf_.get(), oversized_sort_counter_buf_.get(),
       // binding 6: the device-side selected-blob count.
       selected_counter_buf_.get()},
      20, vk::WorkgroupSize{local_sort_cap_, 1, 1}, {local_sort_virtual_cap_, sort_chunk});

  hash_group_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("hash_group"),
      {qbp_keys_buf_.get(), hash_owner_buf_.get(),
       point_slot_buf_.get(), slot_dense_buf_.get(), raw_blob_counter_buf_.get(),
       hash_drop_counter_buf_.get(), qbp_counter_buf_.get()},
      16, wg1d_);
  // One pipeline per source counter: the counter is binding 0, and
  // ComputePipeline binds its buffers at construction.
  build_indirect_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {raw_blob_counter_buf_.get(), indirect_args_buf_.get()}, 16,
      vk::WorkgroupSize{1, 1, 1});
  build_qbp_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {qbp_counter_buf_.get(), indirect_args_buf_.get()}, 16, vk::WorkgroupSize{1, 1, 1});
  build_sort_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {selected_counter_buf_.get(), indirect_args_buf_.get()}, 16, vk::WorkgroupSize{1, 1, 1});
  // Reduction variant priority: subgroup, then 64-bit atomic, then plain 32-bit. The atomic64
  // variant views extents_buf_ again at binding 4 as uint64_t words.
  const bool extents_atomic64 = !subgroup && ctx_.caps().has_int64_atomics;
  extents_atomic64_ = extents_atomic64;
  // Binding 4 is bound for all three variants so binding 5 (boundary-point count) keeps its index.
  std::vector<VkBuffer> reduce_extents_buffers{qbp_compacted_buf_.get(), point_slot_buf_.get(),
                                               slot_dense_buf_.get(),   extents_buf_.get(),
                                               extents_buf_.get(),      qbp_counter_buf_.get()};
  reduce_extents_hash_pl_ = vk::ComputePipeline(
      ctx_,
      ShaderPath(subgroup ? "reduce_extents_hash_subgroup"
                          : (extents_atomic64 ? "reduce_extents_hash_atomic64"
                                              : "reduce_extents_hash")),
      reduce_extents_buffers, 12, wg1d_);
  scatter_index_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("scatter_index_points"),
      {qbp_compacted_buf_.get(), point_slot_buf_.get(), slot_dense_buf_.get(), remap_buf_.get(),
       selected_extents_buf_.get(), blob_point_offsets_buf_.get(), blob_cursor_buf_.get(),
       index_points_buf_.get(), qbp_counter_buf_.get()},
      16, wg1d_);
}

void GpuDetector::RunInclusiveScan(
    VkCommandBuffer cmd, uint32_t count, const ScanChain &chain,
    const std::vector<vk::ComputePipeline> &scan_block_pipelines,
    const std::vector<vk::ComputePipeline> &scan_add_offsets_pipelines) {
  if (count == 0) return;
  (void)chain;

  // Element count at each level; stop once a level fits in a single block.
  std::vector<uint32_t> ns;
  ns.push_back(count);
  while (ns.back() > scan_wg_) {
    ns.push_back((ns.back() + scan_wg_ - 1) / scan_wg_);
  }
  // The chain was built for the worst-case capacity, so a smaller count can
  // only need fewer levels than there are pipelines.
  const size_t levels = std::min(ns.size(), scan_block_pipelines.size());

  // Down pass: scan each level.
  for (size_t i = 0; i < levels; ++i) {
    struct { uint32_t count; } pc{ns[i]};
    scan_block_pipelines[i].Dispatch1D(cmd, ns[i], &pc);
  }
  // Up pass: fold each level's block offsets back into the level below.
  for (size_t i = levels - 1; i-- > 0;) {
    if (i >= scan_add_offsets_pipelines.size()) continue;
    struct { uint32_t count; } pc{ns[i]};
    scan_add_offsets_pipelines[i].Dispatch1D(cmd, ns[i], &pc);
  }
}

void GpuDetector::RecordCounterCopy(VkCommandBuffer cmd, const vk::Buffer &counter,
                                    uint32_t slot) {
  counter.RecordCopyTo(cmd, counter_staging_, 4, 0, VkDeviceSize(slot) * 4);
}

uint32_t GpuDetector::ReadCounterSlot(uint32_t slot) {
  VKA_PROF_SCOPE("gpu.counter_read");
  const auto t0 = Clock::now();
  uint32_t value = 0;
  counter_staging_.Read(&value, 4, VkDeviceSize(slot) * 4);
  last_profile_.cpu_counter_read_ms += MsSince(t0, Clock::now());
  return value;
}

VkCommandBuffer GpuDetector::BeginTimedCommands() {
  VKA_PROF_SCOPE("gpu.begin_commands");
  const auto t0 = Clock::now();
  VkCommandBuffer cmd = lane_.BeginCommands();
  last_profile_.cpu_begin_ms += MsSince(t0, Clock::now());
  return cmd;
}

void GpuDetector::SubmitTimedAndWait(VkCommandBuffer cmd) {
  VKA_PROF_SCOPE("gpu.submit_wait");
  const auto t0 = Clock::now();
  if (prof::Enabled() && frame_first_submit_ns_ == 0) frame_first_submit_ns_ = ToNs(t0);
  lane_.SubmitAndWait(cmd, pending_wait_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  pending_wait_ = VK_NULL_HANDLE;
  last_profile_.cpu_submit_wait_ms += MsSince(t0, Clock::now());
}

void GpuDetector::AddImportedFrame(const uint8_t *key, vk::Buffer buffer, bool yuyv) {
  const bool u8 = ctx_.caps().has_8bit_storage;
  ImportedFrame frame;
  frame.key = key;
  frame.decimate = vk::ComputePipeline(
      ctx_,
      ShaderPath(yuyv ? (u8 ? "decimate_yuyv_u8" : "decimate_yuyv") : (u8 ? "decimate_u8" : "decimate")),
      {buffer.get(), decimated_buf_.get()}, 8, wg2d_, {config_.decimation});
  frame.buffer = std::move(buffer);
  imported_frames_.push_back(std::move(frame));
}

void GpuDetector::ImportHostFrame(const uint8_t *ptr, size_t bytes) {
  AddImportedFrame(ptr,
                   vk::Buffer::ImportHostPointer(ctx_, const_cast<uint8_t *>(ptr), bytes,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                   false);
}

void GpuDetector::ImportDmaBufFrame(const uint8_t *key, int dmabuf_fd, size_t bytes, bool yuyv) {
  AddImportedFrame(key,
                   vk::Buffer::ImportDmaBuf(ctx_, dmabuf_fd, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                   yuyv);
}

namespace {

struct UfCompressPc {
  uint32_t dw, dh, honour_changed_flag;
};
struct UfCountPc {
  uint32_t dw, dh, honour_changed_flag, min_blob;
};

}  // namespace

// What Detect() learns before choosing a flow: the host timestamps and the frame's input buffer.
struct GpuDetector::FrameStart {
  Clock::time_point t_begin, t_upload0, t_upload1;
  uint64_t submits_at_start = 0;
  const ImportedFrame *imported = nullptr;
  VkDeviceSize gray_bytes = 0;
};

void GpuDetector::RecordUfChunk(VkCommandBuffer c, uint32_t iterations, bool trailing_compress) {
  using vk::BarrierKind;
  const uint32_t pixels = decimated_width_ * decimated_height_;
  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};

  // Records `iterations` labelling passes, clearing the flag before the last merge.
  // Without `trailing_compress` the last merge is a verification merge with nothing after it.
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const bool last = iter + 1 == iterations;
    const bool predicate = last && predicated_compress_ && trailing_compress;
    if (last) {
      // Also orders an earlier predicate read of the flag before the fill.
      vk::ComputePipeline::Barrier(c, predicate ? BarrierKind::ComputeTransferAndPredicate
                                                : BarrierKind::ComputeAndTransfer);
      uf_changed_buf_.FillZero(c);
      vk::ComputePipeline::Barrier(c, BarrierKind::ComputeAndTransfer);
    }
    if (last && !trailing_compress) {
      uf_merge_pl_.Dispatch2D(c, decimated_width_, decimated_height_, &dwdh_pc, BarrierKind::None);
      continue;
    }
    // The barrier makes the merge's flag write visible to the compression's guard or predicate read.
    uf_merge_pl_.Dispatch2D(c, decimated_width_, decimated_height_, &dwdh_pc,
                            predicate ? BarrierKind::ComputeAndPredicate : BarrierKind::Compute);
    if (!trailing_compress && iter + 2 == iterations) {
      // The last real compression, just before the verification merge: it also counts blob sizes.
      const UfCountPc count_pc{decimated_width_, decimated_height_, 1u, config_.min_cluster_pixels};
      uf_compress_count_pl_.Dispatch1D(c, pixels, &count_pc);
    } else if (predicate) {
      // Runs only when the flag is nonzero, so the guard is skipped.
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, 0u};
      ctx_.CmdBeginConditionalRendering(c, uf_changed_buf_.get(), 0);
      uf_compress_pl_.Dispatch1D(c, pixels, &compress_pc, BarrierKind::None);
      ctx_.CmdEndConditionalRendering(c);
      vk::ComputePipeline::Barrier(c, BarrierKind::Compute);
    } else {
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, 1u};
      uf_compress_pl_.Dispatch1D(c, pixels, &compress_pc);
    }
  }
}

void GpuDetector::RecordFrameHead(VkCommandBuffer cmd, const ImportedFrame *imported,
                                  uint32_t chunk) {
  using vk::BarrierKind;
  const uint32_t pixels = decimated_width_ * decimated_height_;
  const VkDeviceSize gray_bytes = VkDeviceSize(config_.width) * config_.height;

  // Push constants shared across several stages.
  // decimate_pl_ takes the decimated dimensions directly; decimation is a specialisation constant.
  struct { uint32_t dw, dh; } dims_pc{decimated_width_, decimated_height_};
  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};
  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};

  // One reset covers every span's timestamp pair for the frame (see vk::QueryPool).
  timestamp_pool_.Reset(cmd);
  if (!gray_direct_write_ && imported == nullptr && transfer_lane_ == nullptr) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanClear));
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanClear));
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  struct { uint32_t dw, dh, bw, bh; } minmax_pc{decimated_width_, decimated_height_, block_width_,
                                                block_height_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  (imported != nullptr ? imported->decimate : decimate_pl_)
      .Dispatch2D(cmd, decimated_width_, decimated_height_, &dims_pc);
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff, bw; } thresh_pc{
      decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_};
  threshold_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &thresh_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  uf_init_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &dwdh_pc);
  // Flattens the run chains uf_init.comp built before the first merge. honour_changed_flag is 0
  // because the flag is still zero from the clear.
  if (chunk == 1) {
    // The only merge is the verification merge, so this is the last real compression: it also
    // counts blob sizes (see RecordUfChunk).
    const UfCountPc count_pc{decimated_width_, decimated_height_, 0u, config_.min_cluster_pixels};
    uf_compress_count_pl_.Dispatch1D(cmd, pixels, &count_pc);
  } else {
    const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
    uf_compress_pl_.Dispatch1D(cmd, pixels, &compress_init_pc);
  }

  RecordUfChunk(cmd, chunk, /*trailing_compress=*/false);
  // The labelling span ends here; any extra convergence chunks are not attributed to it.
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelling));
  // Orders the compute writes before the transfer read below; the default Compute barrier does not
  // cover the transfer stage.
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
  // Copies the convergence flag, read back after the frame's submission returns.
  RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
}

void GpuDetector::RecordFusedTail(VkCommandBuffer cmd, bool speculative) {
  using vk::BarrierKind;
  const uint32_t pixels = decimated_width_ * decimated_height_;

  // uf_final counts to the size floor, which must equal blob_diff's.
  struct { uint32_t dw, dh, min_blob; } uf_final_pc{decimated_width_, decimated_height_,
                                                    config_.min_cluster_pixels};
  // The speculative first attempt counted blob sizes inside its last real compression; the tail
  // rerun counts here.
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanUfFinal));
  if (!speculative) uf_final_pl_.Dispatch1D(cmd, pixels, &uf_final_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanUfFinal));

  // blob_diff tests each pixel's blob size itself (against the same floor as uf_final) and appends
  // valid boundary points directly to the compacted buffer.
  struct { uint32_t w, h, capacity, honour_changed_flag, min_blob; } blobdiff_pc{
      decimated_width_, decimated_height_, qbp_capacity_, speculative ? 1u : 0u,
      config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  blob_diff_pl_.Dispatch2D(cmd, interior_width_, interior_height_, &blobdiff_pc,
                           BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);

  // Slot 1 of the indirect args: ceil(boundary points / wg1d), for hash_group, reduce_extents_hash
  // and scatter_index_points. The counts stay on the device.
  struct { uint32_t wg, slot, divide, clamp; } qbp_args_pc{wg1d_.x, 1u, 1u, qbp_capacity_};
  build_qbp_args_pl_.DispatchRaw(cmd, 1, 1, 1, &qbp_args_pc, BarrierKind::None);
  vk::ComputePipeline::IndirectDispatchBarrier(cmd);

  // Groups the boundary points by (rep0, rep1) with a hash table; see hash_group.comp.
  struct { uint32_t count, table_mask, max_probes, count_from_buffer; } hash_pc{
      0u, hash_table_size_ - 1u, 128u, 1u};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanHashGroup));
  hash_group_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &hash_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanHashGroup));

  // Builds the dispatch arguments on the device from raw_blob_counter_buf_; both consumers
  // bounds-check against max_raw_blobs.
  struct { uint32_t wg, slot, divide, clamp; } build_indirect_pc{wg1d_.x, 0u, 1u,
                                                                config_.max_raw_blobs};
  build_indirect_args_pl_.DispatchRaw(cmd, 1, 1, 1, &build_indirect_pc, BarrierKind::None);
  vk::ComputePipeline::IndirectDispatchBarrier(cmd);

  struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanExtents));
  init_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);

  struct { uint32_t count, max_raw_blobs, count_from_buffer; } reduce_pc{0u, config_.max_raw_blobs,
                                                                         1u};
  reduce_extents_hash_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &reduce_pc);

  // Folds the per-workgroup copies into the canonical entries, indirect over the raw blob count.
  merge_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanExtents));

  struct {
    uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal,
        gx_gy_biased;
    float aspect_max, fill_min, fill_max;
  } select_pc{config_.max_raw_blobs,      config_.max_blobs,
              config_.tag_width,          config_.min_cluster_pixels,
              config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
              config_.normal_border ? 1u : 0u, extents_atomic64_ ? 1u : 0u,
              config_.aspect_max,         config_.fill_min,
              config_.fill_max};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSelect));
  select_blobs_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &select_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSelect));

  struct { uint32_t capacity; } extract_pc{config_.max_blobs};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBlobScan));
  extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, &extract_pc);
  RunInclusiveScan(cmd, config_.max_blobs, blob_scan_chain_, blob_scan_block_pls_,
                   blob_scan_add_offsets_pls_);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBlobScan));

  struct { uint32_t count, capacity, max_raw_blobs, count_from_buffer; } scatter_pc{
      0u, ipoint_capacity_, config_.max_raw_blobs, 1u};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanScatter));
  scatter_index_points_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &scatter_pc,
                                            BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanScatter));

  // The number of distinct (rep0, rep1) pairs this frame; profiling only.
  RecordCounterCopy(cmd, raw_blob_counter_buf_, kSlotRawBlobs);
  RecordCounterCopy(cmd, hash_drop_counter_buf_, kSlotHashDrops);

  // blob_point_offsets_buf_[max_blobs - 1] is the inclusive scan total, the sum of every selected
  // blob's point count.
  blob_point_offsets_buf_.RecordCopyTo(
      cmd, counter_staging_, 4, VkDeviceSize(std::max(config_.max_blobs, 1u) - 1u) * 4,
      VkDeviceSize(kSlotPointCount) * 4);
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);

  // Slot 2: one workgroup per selected blob, clamped to max_blobs.
  struct { uint32_t wg, slot, divide, clamp; } sort_args_pc{wg1d_.x, 2u, 0u, config_.max_blobs};
  build_sort_args_pl_.DispatchRaw(cmd, 1, 1, 1, &sort_args_pc, BarrierKind::Compute);
  vk::ComputePipeline::IndirectDispatchBarrier(cmd);

  // One workgroup per blob sorts its points by angle in shared memory and writes the
  // RawLineFitPoints (see sort_points_local.comp).
  struct { uint32_t num_selected_blobs, count_from_buffer, max_blobs; int dw, dh; } sort_pc{
      0u, 1u, config_.max_blobs, static_cast<int>(decimated_width_),
      static_cast<int>(decimated_height_)};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
  sort_points_local_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 24, &sort_pc,
                                         BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));

  // Both readbacks are read in place on the fused path, so there is nothing to copy; the span
  // still marks the point where the results are final.
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanReadbackCopy));
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanReadbackCopy));
  // Counted by sort_points_local, so this is the first submit that can carry it back.
  RecordCounterCopy(cmd, oversized_sort_counter_buf_, kSlotOversizedSortBlobs);
  vk::ComputePipeline::HostReadBarrier(cmd);
}

// Flattens parent[] unconditionally; the retry path needs a flat forest after a discarded attempt.
void GpuDetector::RecordFlatten(VkCommandBuffer cmd) {
  const UfCompressPc compress_pc{decimated_width_, decimated_height_, 0u};
  uf_compress_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, &compress_pc);
}

void GpuDetector::RecordUfContinuation(VkCommandBuffer cmd, uint32_t iterations) {
  RecordFlatten(cmd);
  RecordUfChunk(cmd, iterations);
  // The default compute barrier does not cover the transfer read of this copy.
  vk::ComputePipeline::Barrier(cmd, vk::BarrierKind::ComputeAndTransfer);
  RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
  vk::ComputePipeline::HostReadBarrier(cmd);
}

void GpuDetector::RecordTailRerun(VkCommandBuffer cmd) {
  // Re-zeroes the buffers the discarded tail dirtied and reruns the tail; the head has no per-span
  // timestamps on this frame.
  timestamp_pool_.Reset(cmd);
  // Also needed when no continuation ran (max_uf_iterations reached by the first attempt).
  RecordFlatten(cmd);
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd, vk::BarrierKind::ComputeAndTransfer);
  RecordFusedTail(cmd, /*speculative=*/false);
}

VkCommandBuffer GpuDetector::FindRecorded(RecordingKind kind, const uint8_t *key,
                                          uint32_t n) const {
  for (const RecordedFrame &r : recorded_frames_) {
    if (r.kind == kind && r.key == key && r.n == n) return r.cmd;
  }
  return VK_NULL_HANDLE;
}

GpuDetector::Submission GpuDetector::FirstAttempt(const ImportedFrame *imported, uint32_t chunk) {
  const uint8_t *key = imported != nullptr ? imported->key : nullptr;
  if (prerecord_) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kFirstAttempt, key, chunk)) {
      return {cached, true};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd = lane_.BeginPersistentCommands();
    RecordFrameHead(cmd, imported, chunk);
    RecordFusedTail(cmd, /*speculative=*/true);
    lane_.EndCommands(cmd);
    recorded_frames_.push_back({RecordingKind::kFirstAttempt, key, chunk, cmd});
    return {cmd, true};
  }
  recorded_this_frame_ = true;
  VkCommandBuffer cmd = BeginTimedCommands();
  RecordFrameHead(cmd, imported, chunk);
  RecordFusedTail(cmd, /*speculative=*/true);
  return {cmd, false};
}

GpuDetector::Submission GpuDetector::Continuation(uint32_t iterations) {
  if (prerecord_) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kContinuation, nullptr, iterations)) {
      return {cached, true};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd = lane_.BeginPersistentCommands();
    RecordUfContinuation(cmd, iterations);
    lane_.EndCommands(cmd);
    recorded_frames_.push_back({RecordingKind::kContinuation, nullptr, iterations, cmd});
    return {cmd, true};
  }
  recorded_this_frame_ = true;
  VkCommandBuffer cmd = BeginTimedCommands();
  RecordUfContinuation(cmd, iterations);
  return {cmd, false};
}

GpuDetector::Submission GpuDetector::TailRerun() {
  if (prerecord_) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kTailRerun, nullptr, 0)) {
      return {cached, true};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd = lane_.BeginPersistentCommands();
    RecordTailRerun(cmd);
    lane_.EndCommands(cmd);
    recorded_frames_.push_back({RecordingKind::kTailRerun, nullptr, 0, cmd});
    return {cmd, true};
  }
  recorded_this_frame_ = true;
  VkCommandBuffer cmd = BeginTimedCommands();
  RecordTailRerun(cmd);
  return {cmd, false};
}

void GpuDetector::SubmitTimed(const Submission &s) {
  VKA_PROF_SCOPE("gpu.submit_wait");
  const auto t0 = Clock::now();
  if (prof::Enabled() && frame_first_submit_ns_ == 0) frame_first_submit_ns_ = ToNs(t0);
  if (s.persistent) {
    lane_.SubmitPersistentAndWait(s.cmd, pending_wait_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  } else {
    lane_.SubmitAndWait(s.cmd, pending_wait_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }
  pending_wait_ = VK_NULL_HANDLE;
  last_profile_.cpu_submit_wait_ms += MsSince(t0, Clock::now());
}

void GpuDetector::Detect(const uint8_t *gray_frame) {
  if (!fused_submits_) {
    DetectUnfused(gray_frame);
    return;
  }

  FrameStart frame;
  frame.t_begin = Clock::now();
  frame_first_submit_ns_ = 0;
  last_profile_ = DetectProfile{};
  frame.submits_at_start = lane_.submit_count;
  frame.gray_bytes = VkDeviceSize(config_.width) * config_.height;

  // ------------------------------------------------------------------
  // Upload: one memcpy of the raw 8-bit frame.
  // ------------------------------------------------------------------
  frame.t_upload0 = Clock::now();
  for (const ImportedFrame &f : imported_frames_) {
    if (f.key == gray_frame) frame.imported = &f;
  }
  if (frame.imported != nullptr) {
    // Imported: the GPU reads the caller's buffer directly.
  } else if (gray_direct_write_) {
    gray_buf_.Write(gray_frame, frame.gray_bytes);
  } else {
    upload_staging_.Write(gray_frame, frame.gray_bytes);
  }
  frame.t_upload1 = Clock::now();
  if (transfer_lane_ != nullptr && !gray_direct_write_ && frame.imported == nullptr) {
    // The staged frame is copied on the transfer queue; the first compute submission waits for it.
    VkCommandBuffer tcmd = transfer_lane_->BeginCommands();
    gray_buf_.RecordCopyFrom(tcmd, upload_staging_, frame.gray_bytes);
    transfer_lane_->SubmitSignal(tcmd, upload_semaphore_);
    pending_wait_ = upload_semaphore_;
  }

  // Seeds the first labelling chunk with last frame's converged iteration count.
  uint32_t chunk = std::max(config_.uf_iterations_per_chunk, last_uf_iterations_);
  chunk = std::min(chunk, config_.max_uf_iterations);
  DetectFused(frame, chunk);
}

void GpuDetector::DetectFused(const FrameStart &frame, uint32_t chunk) {
  // The tail is a single submission; only the labelling readback can split the frame.
  last_profile_.gap_crosses_submit[kSpanBoundary] = false;   // boundary -> hash_group
  last_profile_.gap_crosses_submit[kSpanScatter] = false;    // scatter -> sort
  recorded_this_frame_ = false;

  // Time spent obtaining command buffers: recording on a miss, a lookup on a hit.
  double record_ms = 0.0;
  auto timed = [&](auto &&obtain) {
    VKA_PROF_SCOPE("gpu.record");
    const auto t0 = Clock::now();
    const Submission s = obtain();
    record_ms += MsSince(t0, Clock::now());
    return s;
  };

  // First attempt: assumes `chunk` labelling iterations converged and runs the whole frame as one
  // submission. Only if the convergence flag says otherwise is there more to do.
  uint32_t uf_iterations = chunk;
  SubmitTimed(timed([&] { return FirstAttempt(frame.imported, chunk); }));
  bool converged = ReadCounterSlot(kSlotUfChanged) == 0;

  if (!converged) {
    // The chunk undershot: discards the tail's results and resumes convergence from the current
    // parent[], which uf_merge and uf_compress can continue from.
    while (!converged && uf_iterations < config_.max_uf_iterations) {
      const uint32_t next =
          std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
      SubmitTimed(timed([&] { return Continuation(next); }));
      uf_iterations += next;
      converged = ReadCounterSlot(kSlotUfChanged) == 0;
    }
    // parent[] is converged (or max_uf_iterations was hit): rerun the tail on it.
    SubmitTimed(timed([&] { return TailRerun(); }));
  }
  const auto t_submitted = Clock::now();

  last_profile_.record_ms = record_ms;
  last_profile_.replayed = !recorded_this_frame_;
  FinishFused(frame, t_submitted);

  last_uf_iterations_ = uf_iterations;
  last_profile_.uf_iterations = uf_iterations;
  last_profile_.uf_converged = converged;
}

void GpuDetector::FinishFused(const FrameStart &frame, Clock::time_point t_submitted) {
  const uint32_t oversized_sort_blobs = ReadCounterSlot(kSlotOversizedSortBlobs);

  // The counts were consumed on the device; read them now for the profile and the host views.
  const uint32_t qbp_count = std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const uint32_t raw_blobs = (qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0;
  const uint32_t hash_drops = (qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0;
  const uint32_t qualifying = ReadCounterSlot(kSlotSelectedCount);
  const uint32_t selected_blobs = std::min(qualifying, config_.max_blobs);
  const uint32_t blob_drops = qualifying - selected_blobs;
  const uint32_t num_points =
      (qbp_count > 0) ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_) : 0;
  const VkDeviceSize extents_bytes = VkDeviceSize(selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize linefit_bytes = VkDeviceSize(num_points) * sizeof(RawLineFitPoint);

  // ------------------------------------------------------------------
  // Host views of the persistently mapped readback buffers (the fused path requires both to be
  // host-visible and cached).
  // ------------------------------------------------------------------
  last_selected_extents.resize(selected_blobs);
  if (extents_bytes > 0) {
    if (!selected_extents_buf_.coherent()) selected_extents_buf_.InvalidateRange(0, extents_bytes);
    std::memcpy(last_selected_extents.data(), selected_extents_buf_.mapped(),
                static_cast<size_t>(extents_bytes));
  }
  // Zero-copy view of the mapping, valid until the next Detect(); invalidates first when the
  // memory is not coherent.
  if (!line_fit_points_buf_.coherent()) line_fit_points_buf_.InvalidateRange(0, linefit_bytes);
  last_line_fit_points = std::span<const RawLineFitPoint>(
      static_cast<const RawLineFitPoint *>(line_fit_points_buf_.mapped()), num_points);
  const auto t_end = Clock::now();

  // Host phases: recording time is reported as threshold_label_ms and the submission as
  // linefit_ms, so gpu_ms (their sum) is the host cost of the GPU phase.
  if (prof::Enabled()) {
    prof::CompleteHere("gpu.detect", ToNs(frame.t_begin), ToNs(t_end),
                       static_cast<int64_t>(lane_.submit_count - frame.submits_at_start));
    prof::CompleteHere("gpu.host.upload", ToNs(frame.t_upload0), ToNs(frame.t_upload1));
    prof::CompleteHere("gpu.host.readback", ToNs(t_submitted), ToNs(t_end));
  }
  last_profile_.upload_ms = MsSince(frame.t_upload0, frame.t_upload1);
  last_profile_.threshold_label_ms = last_profile_.record_ms;
  last_profile_.boundary_ms = 0.0;
  last_profile_.sort_group_ms = 0.0;
  last_profile_.linefit_ms = MsSince(frame.t_upload1, t_submitted) - last_profile_.record_ms;
  last_profile_.readback_ms = MsSince(t_submitted, t_end);
  last_profile_.gpu_ms = last_profile_.threshold_label_ms + last_profile_.boundary_ms +
                         last_profile_.sort_group_ms + last_profile_.linefit_ms;
  last_profile_.total_ms = MsSince(frame.t_begin, t_end);
  last_profile_.upload_bytes = frame.gray_bytes;
  last_profile_.readback_bytes = 16;  // the counter slots; both payloads are read in place
  last_profile_.selected_blobs = selected_blobs;
  last_profile_.points = num_points;
  last_profile_.boundary_points = qbp_count;
  last_profile_.raw_blobs = raw_blobs;
  last_profile_.hash_probe_drops = hash_drops;
  last_profile_.selected_blob_drops = blob_drops;
  last_profile_.oversized_sort_blobs = oversized_sort_blobs;
  // uf_iterations and uf_converged are set by DetectFused, once any retry is resolved.
  last_profile_.submits = static_cast<uint32_t>(lane_.submit_count - frame.submits_at_start);

  if (timestamps_enabled_) {
    const std::vector<vk::QueryPool::Result> results = timestamp_pool_.ReadResults();
    const double ns_per_tick = ctx_.caps().timestamp_period_ns;
    for (uint32_t s = 0; s < kNumGpuStageSpans; ++s) {
      const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
      const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(s))];
      // Both ends of a span are written together, so both are available or neither.
      if (start.available && end.available && end.ticks >= start.ticks) {
        last_profile_.gpu_stage_ms[s] = double(end.ticks - start.ticks) * ns_per_tick / 1.0e6;
      }
    }
    // Gap g is between span g's end and span g+1's start; see DetectProfile::gpu_gap_ms.
    for (uint32_t g = 0; g + 1 < kNumGpuStageSpans; ++g) {
      const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(g))];
      const vk::QueryPool::Result &next_start =
          results[SpanStart(static_cast<GpuStageSpan>(g + 1))];
      if (end.available && next_start.available && next_start.ticks >= end.ticks) {
        last_profile_.gpu_gap_ms[g] =
            double(next_start.ticks - end.ticks) * ns_per_tick / 1.0e6;
      }
    }
    last_profile_.has_gpu_stage_breakdown = true;

    // Pins the first available GPU span start to the first submission's host time; later spans keep
    // their GPU-clock spacing.
    if (prof::Enabled() && frame_first_submit_ns_ != 0) {
      uint64_t base_ticks = 0;
      bool have_base = false;
      for (uint32_t s = 0; s < kNumGpuStageSpans && !have_base; ++s) {
        const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
        if (start.available) {
          base_ticks = start.ticks;
          have_base = true;
        }
      }
      auto to_host_ns = [&](uint64_t ticks) {
        return frame_first_submit_ns_ +
               static_cast<prof::Ns>(double(ticks - base_ticks) * ns_per_tick);
      };
      for (uint32_t s = 0; s < kNumGpuStageSpans && have_base; ++s) {
        const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
        const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(s))];
        if (start.available && end.available && end.ticks >= start.ticks &&
            start.ticks >= base_ticks) {
          prof::Complete(gpu_track_, kGpuStageNames[s], to_host_ns(start.ticks),
                         to_host_ns(end.ticks));
        }
      }
    }
  }
}

// The flow for devices without host-visible, cached readbacks: dispatches are sized from counts
// read between submissions, so it records every frame.
void GpuDetector::DetectUnfused(const uint8_t *gray_frame) {
  using vk::BarrierKind;
  const auto t_begin = Clock::now();
  frame_first_submit_ns_ = 0;
  last_profile_ = DetectProfile{};
  const uint64_t submits_at_start = lane_.submit_count;

  const uint32_t pixels = decimated_width_ * decimated_height_;
  const VkDeviceSize gray_bytes = VkDeviceSize(config_.width) * config_.height;

  // ------------------------------------------------------------------
  // Upload: one memcpy of the raw 8-bit frame.
  // ------------------------------------------------------------------
  const auto t_upload0 = Clock::now();
  const ImportedFrame *imported = nullptr;
  for (const ImportedFrame &f : imported_frames_) {
    if (f.key == gray_frame) imported = &f;
  }
  if (imported != nullptr) {
  } else if (gray_direct_write_) {
    gray_buf_.Write(gray_frame, gray_bytes);
  } else {
    upload_staging_.Write(gray_frame, gray_bytes);
  }
  const auto t_upload1 = Clock::now();
  if (transfer_lane_ != nullptr && !gray_direct_write_ && imported == nullptr) {
    // The staged frame is copied on the transfer queue; the first compute submission waits for it.
    VkCommandBuffer tcmd = transfer_lane_->BeginCommands();
    gray_buf_.RecordCopyFrom(tcmd, upload_staging_, gray_bytes);
    transfer_lane_->SubmitSignal(tcmd, upload_semaphore_);
    pending_wait_ = upload_semaphore_;
  }

  // Push constants shared across several stages.
  // decimate_pl_ takes the decimated dimensions directly; decimation is a specialisation constant.
  struct { uint32_t dw, dh; } dims_pc{decimated_width_, decimated_height_};
  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};
  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};

  // ------------------------------------------------------------------
  // Submit 1: decimate, adaptive threshold, and the first chunk of
  // connected-component labelling.
  // ------------------------------------------------------------------
  // Record the frame's last three submissions as one when every dispatch in
  // them can be sized on the device. See fused_submits_ in the header.
  const bool fuse = false;  // this flow serves only the unfused configuration
  if (fuse) {
    // The tail is a single submission; only the labelling readback splits the frame.
    last_profile_.gap_crosses_submit[kSpanBoundary] = false;   // boundary -> hash_group
    last_profile_.gap_crosses_submit[kSpanScatter] = false;    // scatter -> sort
  }
  VkCommandBuffer cmd = BeginTimedCommands();
  // One reset covers every span's timestamp pair for the frame (see vk::QueryPool).
  timestamp_pool_.Reset(cmd);
  if (!gray_direct_write_ && imported == nullptr && transfer_lane_ == nullptr) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanClear));
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanClear));
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  struct { uint32_t dw, dh, bw, bh; } minmax_pc{decimated_width_, decimated_height_, block_width_,
                                                block_height_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  (imported != nullptr ? imported->decimate : decimate_pl_)
      .Dispatch2D(cmd, decimated_width_, decimated_height_, &dims_pc);
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff, bw; } thresh_pc{
      decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_};
  threshold_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &thresh_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  uf_init_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &dwdh_pc);
  // Flattens the run chains uf_init.comp built before the first merge. honour_changed_flag is 0
  // because the flag is still zero from the clear.
  struct UfCompressPc { uint32_t dw, dh, honour_changed_flag; };
  const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
  uf_compress_pl_.Dispatch1D(cmd, pixels, &compress_init_pc);

  // Records `iterations` labelling passes. The flag is cleared before the last merge, so zero
  // afterwards means converged. The last uf_compress is predicated on it when available.
  auto record_uf_chunk = [&](VkCommandBuffer c, uint32_t iterations) {
    for (uint32_t iter = 0; iter < iterations; ++iter) {
      const bool last = iter + 1 == iterations;
      const bool predicate = last && predicated_compress_;
      if (last) {
        // Also orders an earlier predicate read of the flag before the fill.
        vk::ComputePipeline::Barrier(c, predicated_compress_
                                            ? BarrierKind::ComputeTransferAndPredicate
                                            : BarrierKind::ComputeAndTransfer);
        uf_changed_buf_.FillZero(c);
        vk::ComputePipeline::Barrier(c, BarrierKind::ComputeAndTransfer);
      }
      // The barrier makes the merge's flag write visible to the compression's guard or predicate read.
      uf_merge_pl_.Dispatch2D(c, decimated_width_, decimated_height_, &dwdh_pc,
                              predicate ? BarrierKind::ComputeAndPredicate
                                        : BarrierKind::Compute);
      if (predicate) {
        // Runs only when the flag is nonzero, so the guard is skipped.
        const UfCompressPc compress_pc{decimated_width_, decimated_height_, 0u};
        ctx_.CmdBeginConditionalRendering(c, uf_changed_buf_.get(), 0);
        uf_compress_pl_.Dispatch1D(c, pixels, &compress_pc, BarrierKind::None);
        ctx_.CmdEndConditionalRendering(c);
        vk::ComputePipeline::Barrier(c, BarrierKind::Compute);
      } else {
        const UfCompressPc compress_pc{decimated_width_, decimated_height_, 1u};
        uf_compress_pl_.Dispatch1D(c, pixels, &compress_pc);
      }
    }
  };

  // Seeds the first chunk with last frame's converged iteration count.
  uint32_t chunk = std::max(config_.uf_iterations_per_chunk, last_uf_iterations_);
  chunk = std::min(chunk, config_.max_uf_iterations);
  record_uf_chunk(cmd, chunk);
  // The labelling span ends here; any extra convergence chunks are not attributed to it.
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelling));
  // Orders the compute writes before the transfer read below; the default Compute barrier does not
  // cover the transfer stage.
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
  // Copies the convergence flag, read back at once (unfused) or after the fused tail returns.
  RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);

  uint32_t uf_iterations = chunk;
  bool converged = false;

  // ------------------------------------------------------------------
  // Submit 2 (unfused) / rest of the frame (fused): blob sizes, boundary extraction, grouping,
  // selection, sort, line-fit and every readback. A lambda so the fused path can run it twice.
  //
  // Returns whether the uf_changed copy recorded before the call showed convergence. `speculative`
  // (fused first attempt only) stops blob_diff_pl_ acting on unconverged labels.
  auto finish_frame = [&](VkCommandBuffer cmd, bool speculative) -> bool {
  const auto t_label = Clock::now();

  // uf_final counts to the size floor, which must equal blob_diff's.
  struct { uint32_t dw, dh, min_blob; } uf_final_pc{decimated_width_, decimated_height_,
                                                    config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanUfFinal));
  uf_final_pl_.Dispatch1D(cmd, pixels, &uf_final_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanUfFinal));

  // blob_diff tests each pixel's blob size itself (against the same floor as uf_final) and appends
  // valid boundary points directly to the compacted buffer.
  struct { uint32_t w, h, capacity, honour_changed_flag, min_blob; } blobdiff_pc{
      decimated_width_, decimated_height_, qbp_capacity_, speculative ? 1u : 0u,
      config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  blob_diff_pl_.Dispatch2D(cmd, interior_width_, interior_height_, &blobdiff_pc,
                           BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);
  if (!fuse) {
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
  }

  // When fusing, the count stays on the device; the host value is used only on the unfused path.
  const uint32_t qbp_count =
      fuse ? 0u : std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const auto t_boundary = Clock::now();

  // ------------------------------------------------------------------
  // Submit 3: group boundary points by (rep0, rep1), select plausible quads, and scatter the
  // survivors into per-blob runs.
  // ------------------------------------------------------------------
  if (!fuse) cmd = BeginTimedCommands();
  // Slot 1 of the indirect args: ceil(boundary points / wg1d), for
  // hash_group, reduce_extents_hash and scatter_index_points.
  if (fuse) {
    struct { uint32_t wg, slot, divide, clamp; } qbp_args_pc{wg1d_.x, 1u, 1u, qbp_capacity_};
    build_qbp_args_pl_.DispatchRaw(cmd, 1, 1, 1, &qbp_args_pc, BarrierKind::None);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }
  if (fuse || qbp_count > 0) {
    // Groups the boundary points by (rep0, rep1) with a hash table; see hash_group.comp.
    struct { uint32_t count, table_mask, max_probes, count_from_buffer; } hash_pc{
        qbp_count, hash_table_size_ - 1u, 128u, fuse ? 1u : 0u};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanHashGroup));
    if (fuse) {
      hash_group_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &hash_pc);
    } else {
      hash_group_pl_.Dispatch1D(cmd, qbp_count, &hash_pc);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanHashGroup));

    // Builds the dispatch arguments on the device from raw_blob_counter_buf_; both consumers
    // bounds-check against max_raw_blobs.
    struct { uint32_t wg, slot, divide, clamp; } build_indirect_pc{wg1d_.x, 0u, 1u,
                                                                  config_.max_raw_blobs};
    build_indirect_args_pl_.DispatchRaw(cmd, 1, 1, 1, &build_indirect_pc, BarrierKind::None);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);

    struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanExtents));
    init_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);

    struct { uint32_t count, max_raw_blobs, count_from_buffer; } reduce_pc{
        qbp_count, config_.max_raw_blobs, fuse ? 1u : 0u};
    if (fuse) {
      reduce_extents_hash_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &reduce_pc);
    } else {
      reduce_extents_hash_pl_.Dispatch1D(cmd, qbp_count, &reduce_pc);
    }

    // Folds the per-workgroup copies into the canonical entries, indirect over the raw blob count.
    merge_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanExtents));

    struct {
      uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal,
          gx_gy_biased;
      float aspect_max, fill_min, fill_max;
    } select_pc{config_.max_raw_blobs,      config_.max_blobs,
                config_.tag_width,          config_.min_cluster_pixels,
                config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
                config_.normal_border ? 1u : 0u, extents_atomic64_ ? 1u : 0u,
                config_.aspect_max,         config_.fill_min,
                config_.fill_max};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSelect));
    select_blobs_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &select_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSelect));

    struct { uint32_t capacity; } extract_pc{config_.max_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBlobScan));
    extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, &extract_pc);
    RunInclusiveScan(cmd, config_.max_blobs, blob_scan_chain_, blob_scan_block_pls_,
                     blob_scan_add_offsets_pls_);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBlobScan));

    struct { uint32_t count, capacity, max_raw_blobs, count_from_buffer; } scatter_pc{
        qbp_count, ipoint_capacity_, config_.max_raw_blobs, fuse ? 1u : 0u};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanScatter));
    if (fuse) {
      scatter_index_points_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &scatter_pc,
                                                BarrierKind::ComputeAndTransfer);
    } else {
      scatter_index_points_pl_.Dispatch1D(cmd, qbp_count, &scatter_pc,
                                          BarrierKind::ComputeAndTransfer);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanScatter));

    // The number of distinct (rep0, rep1) pairs this frame; profiling only.
    RecordCounterCopy(cmd, raw_blob_counter_buf_, kSlotRawBlobs);
    RecordCounterCopy(cmd, hash_drop_counter_buf_, kSlotHashDrops);

    // blob_point_offsets_buf_[max_blobs - 1] is the inclusive scan total, the sum of every selected
    // blob's point count.
    blob_point_offsets_buf_.RecordCopyTo(
        cmd, counter_staging_, 4, VkDeviceSize(std::max(config_.max_blobs, 1u) - 1u) * 4,
        VkDeviceSize(kSlotPointCount) * 4);
  }
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);
  if (!fuse) {
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
  }

  const uint32_t num_raw_blobs =
      fuse ? 0u : ((qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0);
  const uint32_t hash_probe_drops =
      fuse ? 0u : ((qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0);
  // Blobs that passed the filters, which can exceed max_blobs; kept unclamped to report
  // selected_blob_drops.
  const uint32_t qualifying_blobs = fuse ? 0u : ReadCounterSlot(kSlotSelectedCount);
  const uint32_t num_selected_blobs = fuse ? 0u : std::min(qualifying_blobs, config_.max_blobs);
  const uint32_t selected_blob_drops = fuse ? 0u : (qualifying_blobs - num_selected_blobs);
  const uint32_t num_points =
      fuse ? 0u
           : ((qbp_count > 0) ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_) : 0);
  const auto t_sort_group = Clock::now();

  // ------------------------------------------------------------------
  // Submit 4: sort each blob's points around its perimeter, compute line-fit moments, and stage
  // both payloads for readback.
  // ------------------------------------------------------------------
  // Staging sizes are needed only on the unfused path; the fused path reads both in place.
  const VkDeviceSize extents_bytes =
      fuse ? 0 : VkDeviceSize(num_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize linefit_offset = (extents_bytes + 15) & ~VkDeviceSize(15);
  const VkDeviceSize linefit_bytes =
      fuse ? 0 : VkDeviceSize(num_points) * sizeof(RawLineFitPoint);
  if (!fuse) EnsureReadbackCapacity(linefit_offset + linefit_bytes);

  if (!fuse) cmd = BeginTimedCommands();
  // Slot 2: one workgroup per selected blob, clamped to max_blobs.
  if (fuse) {
    struct { uint32_t wg, slot, divide, clamp; } sort_args_pc{wg1d_.x, 2u, 0u, config_.max_blobs};
    build_sort_args_pl_.DispatchRaw(cmd, 1, 1, 1, &sort_args_pc, BarrierKind::Compute);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }
  if (fuse || num_points > 0) {
    // One workgroup per blob sorts its points by angle in shared memory and writes the
    // RawLineFitPoints (see sort_points_local.comp).
    struct { uint32_t num_selected_blobs, count_from_buffer, max_blobs; int dw, dh; } sort_pc{
        num_selected_blobs, fuse ? 1u : 0u, config_.max_blobs,
        static_cast<int>(decimated_width_), static_cast<int>(decimated_height_)};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
    if (fuse) {
      sort_points_local_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 24, &sort_pc,
                                             BarrierKind::ComputeAndTransfer);
    } else {
      sort_points_local_pl_.DispatchRaw(cmd, num_selected_blobs, 1, 1, &sort_pc,
                                        BarrierKind::ComputeAndTransfer);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanReadbackCopy));
  if (extents_bytes > 0 && !extents_direct_read_) {
    selected_extents_buf_.RecordCopyTo(cmd, readback_staging_, extents_bytes, 0, 0);
  }
  // Skipped when the line-fit buffer is host-visible and cached; QuadDecode reads it in place.
  if (linefit_bytes > 0 && !linefit_direct_read_) {
    line_fit_points_buf_.RecordCopyTo(cmd, readback_staging_, linefit_bytes, 0, linefit_offset);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanReadbackCopy));
  // Counted by sort_points_local, so this is the first submit that can carry it back.
  RecordCounterCopy(cmd, oversized_sort_counter_buf_, kSlotOversizedSortBlobs);
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);
  const auto t_linefit = Clock::now();
  const uint32_t oversized_sort_blobs = ReadCounterSlot(kSlotOversizedSortBlobs);

  // On the fused path the counts were consumed on the device; read them now for the readback spans
  // and the profile.
  const uint32_t final_qbp_count =
      fuse ? std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_) : qbp_count;
  const uint32_t final_raw_blobs =
      fuse ? ((final_qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0) : num_raw_blobs;
  const uint32_t final_hash_drops =
      fuse ? ((final_qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0) : hash_probe_drops;
  const uint32_t final_qualifying =
      fuse ? ReadCounterSlot(kSlotSelectedCount) : qualifying_blobs;
  const uint32_t final_selected_blobs =
      fuse ? std::min(final_qualifying, config_.max_blobs) : num_selected_blobs;
  const uint32_t final_blob_drops =
      fuse ? (final_qualifying - final_selected_blobs) : selected_blob_drops;
  const uint32_t final_num_points =
      fuse ? ((final_qbp_count > 0)
                  ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_)
                  : 0)
           : num_points;
  const VkDeviceSize final_extents_bytes =
      VkDeviceSize(final_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize final_linefit_bytes =
      VkDeviceSize(final_num_points) * sizeof(RawLineFitPoint);

  // ------------------------------------------------------------------
  // Host-side copies out of the persistently mapped readback buffer.
  // ------------------------------------------------------------------
  last_selected_extents.resize(final_selected_blobs);
  if (final_extents_bytes > 0) {
    if (extents_direct_read_) {
      if (!selected_extents_buf_.coherent()) {
        selected_extents_buf_.InvalidateRange(0, final_extents_bytes);
      }
      std::memcpy(last_selected_extents.data(), selected_extents_buf_.mapped(),
                  static_cast<size_t>(final_extents_bytes));
    } else {
      readback_staging_.Read(last_selected_extents.data(), final_extents_bytes, 0);
    }
  }
  if (linefit_direct_read_) {
    // Zero-copy view of the mapping, valid until the next Detect(); invalidates first when the
    // memory is not coherent.
    if (!line_fit_points_buf_.coherent()) {
      line_fit_points_buf_.InvalidateRange(0, final_linefit_bytes);
    }
    last_line_fit_points = std::span<const RawLineFitPoint>(
        static_cast<const RawLineFitPoint *>(line_fit_points_buf_.mapped()), final_num_points);
  } else {
    linefit_scratch_.resize(final_num_points);
    if (final_linefit_bytes > 0) {
      readback_staging_.Read(linefit_scratch_.data(), final_linefit_bytes, linefit_offset);
    }
    last_line_fit_points = std::span<const RawLineFitPoint>(linefit_scratch_);
  }
  const auto t_end = Clock::now();

  if (prof::Enabled()) {
    // Host-side phases of this frame, from the timestamps already taken above.
    prof::CompleteHere("gpu.detect", ToNs(t_begin), ToNs(t_end),
                       static_cast<int64_t>(lane_.submit_count - submits_at_start));
    prof::CompleteHere("gpu.host.upload", ToNs(t_upload0), ToNs(t_upload1));
    prof::CompleteHere("gpu.host.threshold_label", ToNs(t_upload1), ToNs(t_label));
    prof::CompleteHere("gpu.host.boundary", ToNs(t_label), ToNs(t_boundary));
    prof::CompleteHere("gpu.host.sort_group", ToNs(t_boundary), ToNs(t_sort_group));
    prof::CompleteHere("gpu.host.linefit", ToNs(t_sort_group), ToNs(t_linefit));
    prof::CompleteHere("gpu.host.readback", ToNs(t_linefit), ToNs(t_end));
  }

  last_profile_.upload_ms = MsSince(t_upload0, t_upload1);
  last_profile_.threshold_label_ms = MsSince(t_upload1, t_label);
  last_profile_.boundary_ms = MsSince(t_label, t_boundary);
  last_profile_.sort_group_ms = MsSince(t_boundary, t_sort_group);
  last_profile_.linefit_ms = MsSince(t_sort_group, t_linefit);
  last_profile_.readback_ms = MsSince(t_linefit, t_end);
  last_profile_.gpu_ms = last_profile_.threshold_label_ms + last_profile_.boundary_ms +
                         last_profile_.sort_group_ms + last_profile_.linefit_ms;
  last_profile_.total_ms = MsSince(t_begin, t_end);
  last_profile_.upload_bytes = gray_bytes;
  last_profile_.readback_bytes = extents_bytes + linefit_bytes + 16;
  last_profile_.selected_blobs = final_selected_blobs;
  last_profile_.points = final_num_points;
  last_profile_.boundary_points = final_qbp_count;
  last_profile_.raw_blobs = final_raw_blobs;
  last_profile_.hash_probe_drops = final_hash_drops;
  last_profile_.selected_blob_drops = final_blob_drops;
  last_profile_.oversized_sort_blobs = oversized_sort_blobs;
  // uf_iterations and uf_converged are set at the end of Detect(), once the fused retry is resolved.
  last_profile_.submits = static_cast<uint32_t>(lane_.submit_count - submits_at_start);

  if (timestamps_enabled_) {
    const std::vector<vk::QueryPool::Result> results = timestamp_pool_.ReadResults();
    const double ns_per_tick = ctx_.caps().timestamp_period_ns;
    for (uint32_t s = 0; s < kNumGpuStageSpans; ++s) {
      const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
      const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(s))];
      // Both ends of a span are written together, so both are available or neither.
      if (start.available && end.available && end.ticks >= start.ticks) {
        last_profile_.gpu_stage_ms[s] = double(end.ticks - start.ticks) * ns_per_tick / 1.0e6;
      }
    }
    // Gap g is between span g's end and span g+1's start; see DetectProfile::gpu_gap_ms.
    for (uint32_t g = 0; g + 1 < kNumGpuStageSpans; ++g) {
      const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(g))];
      const vk::QueryPool::Result &next_start =
          results[SpanStart(static_cast<GpuStageSpan>(g + 1))];
      if (end.available && next_start.available && next_start.ticks >= end.ticks) {
        last_profile_.gpu_gap_ms[g] =
            double(next_start.ticks - end.ticks) * ns_per_tick / 1.0e6;
      }
    }
    last_profile_.has_gpu_stage_breakdown = true;

    // Pins the first available GPU span start to the first submission's host time; later spans keep
    // their GPU-clock spacing.
    if (prof::Enabled() && frame_first_submit_ns_ != 0) {
      uint64_t base_ticks = 0;
      bool have_base = false;
      for (uint32_t s = 0; s < kNumGpuStageSpans && !have_base; ++s) {
        const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
        if (start.available) {
          base_ticks = start.ticks;
          have_base = true;
        }
      }
      auto to_host_ns = [&](uint64_t ticks) {
        return frame_first_submit_ns_ + static_cast<prof::Ns>(double(ticks - base_ticks) * ns_per_tick);
      };
      for (uint32_t s = 0; s < kNumGpuStageSpans && have_base; ++s) {
        const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
        const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(s))];
        if (start.available && end.available && end.ticks >= start.ticks &&
            start.ticks >= base_ticks) {
          prof::Complete(gpu_track_, kGpuStageNames[s], to_host_ns(start.ticks),
                         to_host_ns(end.ticks));
        }
      }
    }
  }

  // Quad fitting itself happens in QuadDecode (CPU tail); Detect() only runs
  // the GPU pipeline and exposes its outputs via last_selected_extents /
  // last_line_fit_points.
  return ReadCounterSlot(kSlotUfChanged) == 0;
  };  // finish_frame

  if (fuse) {
    // Fast path: assumes `chunk` iterations converged (seeded from last frame's count) and records
    // the whole tail into the labelling command buffer with no submit between.
    converged = finish_frame(cmd, /*speculative=*/true);
    if (!converged) {
      // The chunk undershot: discards the tail's results and resumes convergence from the current
      // parent[], which uf_merge and uf_compress can continue from.
      while (!converged && uf_iterations < config_.max_uf_iterations) {
        const uint32_t next =
            std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
        cmd = BeginTimedCommands();
        record_uf_chunk(cmd, next);
        // The default compute barrier does not cover the transfer read of this copy.
        vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
        RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
        vk::ComputePipeline::HostReadBarrier(cmd);
        SubmitTimedAndWait(cmd);
        uf_iterations += next;
        converged = ReadCounterSlot(kSlotUfChanged) == 0;
      }

      // parent[] is converged (or max_uf_iterations was hit). Re-zeroes the buffers the discarded tail
      // dirtied (record_uf_chunk re-zeroes uf_changed_buf_) and reruns the tail. The timestamp pool
      // is reset, so this frame has no per-span breakdown.
      cmd = BeginTimedCommands();
      timestamp_pool_.Reset(cmd);
      qbp_counter_buf_.FillZero(cmd);
      selected_counter_buf_.FillZero(cmd);
      blob_size_buf_.FillZero(cmd);
      hash_owner_buf_.FillZero(cmd);
      blob_cursor_buf_.FillZero(cmd);
      raw_blob_counter_buf_.FillZero(cmd);
      hash_drop_counter_buf_.FillZero(cmd);
      oversized_sort_counter_buf_.FillZero(cmd);
      vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
      finish_frame(cmd, /*speculative=*/false);
    }
  } else {
    // Unfused: submits the labelling chunk, reads back convergence, runs any extra chunks, then
    // submits the tail separately.
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
    converged = ReadCounterSlot(kSlotUfChanged) == 0;
    while (!converged && uf_iterations < config_.max_uf_iterations) {
      const uint32_t next =
          std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
      cmd = BeginTimedCommands();
      record_uf_chunk(cmd, next);
      // The default compute barrier does not cover the transfer read of this copy.
      vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
      RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
      vk::ComputePipeline::HostReadBarrier(cmd);
      SubmitTimedAndWait(cmd);
      uf_iterations += next;
      converged = ReadCounterSlot(kSlotUfChanged) == 0;
    }
    cmd = BeginTimedCommands();
    finish_frame(cmd, /*speculative=*/false);
  }
  last_uf_iterations_ = uf_iterations;
  last_profile_.uf_iterations = uf_iterations;
  last_profile_.uf_converged = converged;
}

std::string GpuDetector::DescribeSizing() const {
  std::ostringstream os;
  os << "apriltag_vulkan: " << config_.width << "x" << config_.height << " -> decimated "
     << decimated_width_ << "x" << decimated_height_ << ", boundary point capacity "
     << qbp_capacity_;
  if (config_.max_boundary_points > 0 && qbp_capacity_ < dense_qbp_count_) {
    os << " (capped from dense " << dense_qbp_count_ << ")";
  }
  os << ", blob capacity " << config_.max_blobs;
  os << ", device memory " << (device_bytes_ / (1024 * 1024)) << " MiB";
  os << ", gray upload " << (gray_direct_write_ ? "direct (unified memory)" : "staged");
  // Whether the line-fit readback avoided its copies depends on a host-cached memory type.
  os << ", linefit readback " << (linefit_direct_read_ ? "direct (host-cached)" : "staged");
  os << ", uf_compress skip "
     << (predicated_compress_ ? "predicated (conditional rendering)" : "in-shader");
  return os.str();
}

}  // namespace apriltag_vulkan
