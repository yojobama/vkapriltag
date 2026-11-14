#include "vkapriltag/gpu/GpuDetector.h"

#include "vkapriltag/common/Profiler.h"
#include "vkapriltag/gpu/CpuDecimate.h"
#include "vkapriltag/vk/EmbeddedShaders.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

// Largest blob the sort orders (larger ones pass through unsorted); it also bounds mode 2's cost.
// A tag border at the largest frame (8191 decimated pixels a side) is about 50,000 points.

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : GpuDetector(ctx, ctx.default_lane(), config) {}

GpuDetector::GpuDetector(vk::Context &ctx, vk::Lane &lane, const DetectorConfig &config)
    : GpuDetector(ctx, lane, nullptr, config) {}

GpuDetector::~GpuDetector() {
  keepalive_.reset();
  vk::DispatchProbe::Report(ctx_.device(), this);
  // Detect() returns only after its submission completes, so nothing recorded is in flight.
  for (const RecordedFrame &r : recorded_frames_) lane_.FreePersistentCommands(r.cmd);
  if (upload_semaphore_ != VK_NULL_HANDLE) {
    // Only this detector's lanes use the semaphore; vkDeviceWaitIdle would need every queue's lock.
    try {
      transfer_lane_->WaitIdle();
      lane_.WaitIdle();
    } catch (...) {
      // The device is lost; nothing is executing any more.
    }
    vkDestroySemaphore(ctx_.device(), upload_semaphore_, nullptr);
  }
}

GpuDetector::GpuDetector(vk::Context &ctx, vk::Lane &lane, vk::Lane *transfer_lane,
                         const DetectorConfig &config)
    : ctx_(ctx), lane_(lane), transfer_lane_(transfer_lane), config_(config),
      env_(Options::FromEnvironment()) {
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
  // The boundary passes work on the decimated interior, which needs at least one pixel.
  if (config_.width / config_.decimation < 3 || config_.height / config_.decimation < 3) {
    throw std::runtime_error("width/decimation and height/decimation must each be >= 3");
  }
  if (config_.max_uf_iterations == 0) {
    throw std::runtime_error("max_uf_iterations must be >= 1");
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
  if (env_.Has("APRILTAG_VK_MIN_DIFF")) {
    config_.min_white_black_diff = static_cast<uint32_t>(std::max(0, env_.Int("APRILTAG_VK_MIN_DIFF", 0)));
  }
  if (const auto parsed = env_.Long("APRILTAG_VK_MAX_POINTS"); parsed && *parsed > 0) {
    config_.max_boundary_points = static_cast<uint32_t>(*parsed);
  }
  // Labelling iterations per chunk.
  if (const auto parsed = env_.Long("APRILTAG_VK_UF_CHUNK"); parsed && *parsed > 0) {
    config_.uf_iterations_per_chunk = static_cast<uint32_t>(*parsed);
  }
  if (config_.uf_iterations_per_chunk == 0) config_.uf_iterations_per_chunk = 1;

  // Derives a minimum cluster size from the smallest tag side (full-resolution pixels) and raises
  // min_cluster_pixels to it if lower.
  if (const auto parsed = env_.Long("APRILTAG_VK_MIN_TAG_PX"); parsed && *parsed > 0) {
    config_.min_tag_pixels = static_cast<uint32_t>(*parsed);
  }
  config_.small_tag_rescue = ResolveSmallTagRescue(env_, config_.small_tag_rescue);
  if (config_.small_tag_rescue && config_.decimation > 1) {
    config_.min_cluster_pixels = std::min<uint32_t>(config_.min_cluster_pixels, 8u);
    // Raises the contrast floor so noise specks of 8 to 24 pixels are not labelled.
    config_.min_white_black_diff = std::max<uint32_t>(config_.min_white_black_diff, 8u);
  }
  if (config_.min_tag_pixels > 0) {
    // Perimeter, in decimated pixels, of a square tag whose side is min_tag_pixels.
    const uint32_t derived_min_cluster = 4u * (config_.min_tag_pixels / config_.decimation);
    config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, derived_min_cluster);
  }
  if (config_.max_tag_pixels > 0) {
    const uint32_t derived_max_cluster = 8u * std::max(1u, config_.max_tag_pixels / config_.decimation);
    config_.max_cluster_pixels = std::min(config_.max_cluster_pixels, derived_max_cluster);
  }
  // Floors min_cluster_pixels at 2, as the ambiguous-pixel paths in uf_final.comp and
  // blob_diff.comp require.
  config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, 2u);

  // Launch geometry comes from the device, never from a literal.
  const vk::DeviceCaps &caps = ctx_.caps();
  plan_ = ResolvePlan(caps, config_, env_);


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

  // The grayscale frame is uploaded packed, four 8-bit pixels per uint32, the last word padded.
  gray_words_ = (config_.width * config_.height + 3u) / 4u;


  // Every dispatch must fit maxComputeWorkGroupCount[0]. The per-pixel ones are checked here; the
  // capacities bound the device-built indirect counts (boundary points, blobs, large-sort runs).
  const uint64_t max_groups = caps.max_workgroup_count[0];
  const uint64_t decimated_pixels = uint64_t(decimated_width_) * decimated_height_;
  if ((decimated_pixels + plan_.wg1d.x - 1) / plan_.wg1d.x > max_groups) {
    throw std::runtime_error("the decimated frame exceeds this device's workgroup-count limit; "
                             "raise the decimation");
  }
  qbp_capacity_ = static_cast<uint32_t>(std::min<uint64_t>(qbp_capacity_, max_groups * plan_.wg1d.x));
  ipoint_capacity_ = qbp_capacity_;
  config_.max_raw_blobs =
      static_cast<uint32_t>(std::min<uint64_t>(config_.max_raw_blobs, max_groups * plan_.wg1d.x));
  // The large-blob sort launches up to one workgroup per run or per workgroup of points, plus one
  // per listed blob for the remainders; half the limit each.
  const uint64_t sort_points_per_group = std::min(plan_.local_sort_split, plan_.local_sort_cap);
  qbp_capacity_ = static_cast<uint32_t>(
      std::min<uint64_t>(qbp_capacity_, (max_groups / 2) * sort_points_per_group));
  ipoint_capacity_ = qbp_capacity_;
  config_.max_blobs =
      static_cast<uint32_t>(std::min<uint64_t>(config_.max_blobs, max_groups / 2));

  CreateBuffers();
  CreatePipelines();

  // Completion by polling a flag: on by default only with device-coherent host memory, where the
  // write is visible to the host while the submission runs; APRILTAG_VK_FLAG_POLL=1 forces it.
  const bool want_flag = env_.Bool("APRILTAG_VK_FLAG_POLL", ctx_.caps().has_device_coherent_memory);
  if (want_flag && plan_.prerecord && plan_.fused_submits) {
    completion_flag_ = std::make_unique<vk::CompletionFlag>(ctx_);
  }

  int keepalive_ms = env_.Int("APRILTAG_VK_KEEPALIVE_MS", config_.keepalive_ms);
  if (keepalive_ms < 0) {
    keepalive_ms = ctx_.caps().type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 6 : 0;
  }
  int prewarm_lead_us = env_.Int("APRILTAG_VK_PREWARM_LEAD_US", config_.prewarm_lead_us);
  if (prewarm_lead_us < 0) prewarm_lead_us = 3000;
  uint32_t prewarm_burst = config_.prewarm_burst;
  if (env_.Has("APRILTAG_VK_PREWARM_BURST")) prewarm_burst = static_cast<uint32_t>(std::max(1, env_.Int("APRILTAG_VK_PREWARM_BURST", 1)));
  uint32_t prewarm_gap_us = config_.prewarm_gap_us;
  if (env_.Has("APRILTAG_VK_PREWARM_GAP_US")) prewarm_gap_us = static_cast<uint32_t>(std::max(0, env_.Int("APRILTAG_VK_PREWARM_GAP_US", 0)));
  if (keepalive_ms > 0 || prewarm_lead_us > 0) {
    if (std::unique_ptr<vk::Lane> keepalive_lane = ctx_.CreateSiblingLane(lane_)) {
      keepalive_buf_ = vk::Buffer(ctx_, 4, kSsboUsage, vk::MemoryKind::DeviceLocal);
      keepalive_pl_ = vk::ComputePipeline(ctx_, ShaderPath("keepalive"), {keepalive_buf_.get()}, 0,
                                          vk::WorkgroupSize{1, 1, 1});
      KeepAlive::Options options;
      options.interval = std::chrono::milliseconds(std::max(keepalive_ms, 0));
      options.lead = std::chrono::microseconds(prewarm_lead_us);
      options.burst = prewarm_burst;
      if (env_.Has("APRILTAG_VK_PREWARM_GROUPS")) options.groups = static_cast<uint32_t>(std::max(1, env_.Int("APRILTAG_VK_PREWARM_GROUPS", 1)));
      options.burst_gap = std::chrono::microseconds(prewarm_gap_us);
      keepalive_ = std::make_unique<KeepAlive>(std::move(keepalive_lane), keepalive_pl_, options);
    }
  }

  // Persists newly compiled pipelines immediately rather than only at ~Context().
  ctx_.FlushPipelineCache();

  // Opt-in per-shader GPU timing; tracing also enables it for the GPU track.
  bool want_timestamps = prof::Enabled();
  if (const auto ts = env_.Long("APRILTAG_VK_TIMESTAMPS")) want_timestamps = want_timestamps || *ts != 0;
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

  // Host-visible device memory written directly where the device has it (unified memory, or a
  // discrete GPU's mappable window); otherwise device-local, filled through a persistently mapped
  // staging buffer.
  gray_buf_ = vk::Buffer(
      ctx_, VkDeviceSize(gray_words_) * 4,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      vk::MemoryKind::DeviceLocalMapped,
      transfer_lane_ != nullptr ? ctx_.queue_families() : std::vector<uint32_t>{});
  device_bytes_ += gray_buf_.size();
  plan_.gray_direct_write = gray_buf_.host_visible();

  if (!plan_.gray_direct_write) {
    upload_staging_ = vk::Buffer(ctx_, VkDeviceSize(gray_words_) * 4,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::MemoryKind::HostVisible);
  }
  counter_staging_ = vk::Buffer(ctx_, kCounterStagingBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                vk::MemoryKind::HostVisibleCached);

  // 1 byte per pixel with storageBuffer8BitAccess, else 4. decimated_buf_ and thresholded_buf_ share
  // the setting.
  const VkDeviceSize decimated_bytes_per_pixel = ctx_.caps().has_8bit_storage ? 1 : 4;
  const VkDeviceSize decimated_bytes = VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel;
  if (plan_.want_cpu_decimate) {
    decimated_buf_ = vk::Buffer(ctx_, std::max<VkDeviceSize>(decimated_bytes, 4), kSsboUsage,
                                vk::MemoryKind::DeviceLocalMapped);
    device_bytes_ += decimated_buf_.size();
  } else {
    decimated_buf_ = ssbo(decimated_bytes);
  }
  minmax_unfiltered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  contrast_floor_buf_ = ssbo(16);
  active_cells_buf_ = ssbo(VkDeviceSize((block_width_ + 15u) / 16u) * block_height_ * 4);
  cell_list_buf_ = ssbo(VkDeviceSize(1 + (block_width_ + 15u) / 16u * block_height_) * 4);
  thresholded_buf_ =
      ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel);

  parent_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  // Also the conditional-rendering predicate for uf_compress (see plan_.predicated_compress).
  {
    const VkBufferUsageFlags usage =
        kSsboUsage | (plan_.predicated_compress ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0);
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
  // Host-visible and cached where available, so QuadDecode reads it in place (see plan_.fused_submits);
  // otherwise staged.
  {
    vk::Buffer b(ctx_, VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu), kSsboUsage,
                 vk::MemoryKind::HostVisibleCached);
    device_bytes_ += b.size();
    selected_extents_buf_ = std::move(b);
  }
  plan_.extents_direct_read =
      selected_extents_buf_.host_visible() && selected_extents_buf_.host_cached();
  selected_counter_buf_ = ssbo(4);
  points_total_buf_ = ssbo(4);
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
  plan_.linefit_direct_read = line_fit_points_buf_.host_visible() && line_fit_points_buf_.host_cached();

  FinalisePlan(plan_, ctx_.caps(), env_, transfer_lane_ != nullptr,
               BufferOutcome{plan_.gray_direct_write, plan_.linefit_direct_read, plan_.extents_direct_read,
                             decimated_buf_.host_visible() && decimated_buf_.coherent()});

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
  {
    // Eight words of dispatch args and count, then three words (LargeBlob) per selected blob.
    vk::Buffer b(ctx_, VkDeviceSize(8 + 3 * std::max(config_.max_blobs, 1u)) * 4,
                 kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    large_sort_buf_ = std::move(b);
  }
  sort_scratch_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * 4);

  // VkDispatchIndirectCommands built on the device by Pl(Stage::kBuildBlobArgs).
  {
    // Slot 1: the boundary points, one per thread; slot 2: 4 per thread.
    vk::Buffer b(ctx_, 3 * 12, kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    indirect_args_buf_ = std::move(b);
  }
  {
    vk::Buffer b(ctx_, 2 * 12, kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                 vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    blob_args_buf_ = std::move(b);
  }
  {
    // The labelling passes' arguments over the listed cells (cell_args.comp).
    vk::Buffer b(ctx_, 12, kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    cell_args_buf_ = std::move(b);
  }

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

VkBuffer GpuDetector::Handle(Buf role) const {
  switch (role) {
    case Buf::kGray: return gray_buf_.get();
    case Buf::kDecimated: return decimated_buf_.get();
    case Buf::kDecimatedView: return plan_.alias_decimated ? gray_buf_.get() : decimated_buf_.get();
    case Buf::kMinMaxUnfiltered: return minmax_unfiltered_buf_.get();
    case Buf::kMinMaxFiltered: return minmax_filtered_buf_.get();
    case Buf::kContrastFloor: return contrast_floor_buf_.get();
    case Buf::kActiveCells: return active_cells_buf_.get();
    case Buf::kCellList: return cell_list_buf_.get();
    case Buf::kCellArgs: return cell_args_buf_.get();
    case Buf::kThresholded: return thresholded_buf_.get();
    case Buf::kParent: return parent_buf_.get();
    case Buf::kBlobSize: return blob_size_buf_.get();
    case Buf::kUfChanged: return uf_changed_buf_.get();
    case Buf::kQbpCompacted: return qbp_compacted_buf_.get();
    case Buf::kQbpCounter: return qbp_counter_buf_.get();
    case Buf::kQbpKeys: return qbp_keys_buf_.get();
    case Buf::kExtents: return extents_buf_.get();
    case Buf::kSelectedExtents: return selected_extents_buf_.get();
    case Buf::kSelectedCounter: return selected_counter_buf_.get();
    case Buf::kRemap: return remap_buf_.get();
    case Buf::kPointsTotal: return points_total_buf_.get();
    case Buf::kIndexPoints: return index_points_buf_.get();
    case Buf::kBlobPointOffsets: return blob_point_offsets_buf_.get();
    case Buf::kLineFitPoints: return line_fit_points_buf_.get();
    case Buf::kHashOwner: return hash_owner_buf_.get();
    case Buf::kPointSlot: return point_slot_buf_.get();
    case Buf::kSlotDense: return slot_dense_buf_.get();
    case Buf::kBlobCursor: return blob_cursor_buf_.get();
    case Buf::kRawBlobCounter: return raw_blob_counter_buf_.get();
    case Buf::kHashDropCounter: return hash_drop_counter_buf_.get();
    case Buf::kOversizedSortCounter: return oversized_sort_counter_buf_.get();
    case Buf::kLargeSort: return large_sort_buf_.get();
    case Buf::kSortScratch: return sort_scratch_buf_.get();
    case Buf::kIndirectArgs: return indirect_args_buf_.get();
    case Buf::kBlobArgs: return blob_args_buf_.get();
    case Buf::kCount: break;
  }
  return VK_NULL_HANDLE;
}


void GpuDetector::CreatePipelines() {
  const StageContext stage_context{plan_, config_};
  for (size_t i = 0; i < kNumStages; ++i) {
    const StageDesc &desc = StageInfo(static_cast<Stage>(i));
    if (!desc.enabled(stage_context)) continue;
    std::vector<VkBuffer> buffers;
    for (Buf role : desc.bindings) buffers.push_back(Handle(role));
    const std::string shader = desc.shader(stage_context);
    pl_[i] = vk::ComputePipeline(ctx_, ShaderPath(shader.c_str()), buffers, desc.push_bytes,
                                 desc.workgroup(stage_context), desc.spec(stage_context));
  }

  lean_point_groups_ = std::min((qbp_capacity_ + plan_.wg1d.x - 1u) / plan_.wg1d.x, 128u);
  lean_select_groups_ = std::min((config_.max_raw_blobs + plan_.wg1d.x - 1u) / plan_.wg1d.x, 32u);
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
  lane_.SubmitAndWait(cmd, std::exchange(pending_wait_, VK_NULL_HANDLE),
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  last_profile_.cpu_submit_wait_ms += MsSince(t0, Clock::now());
}

void GpuDetector::AddImportedFrame(const uint8_t *key, vk::Buffer buffer, bool yuyv) {
  const bool u8 = ctx_.caps().has_8bit_storage;
  ImportedFrame frame;
  frame.key = key;
  frame.decimate = vk::ComputePipeline(
      ctx_,
      ShaderPath(yuyv ? (u8 ? "decimate_yuyv_u8" : "decimate_yuyv")
                      : plan_.box_decimate ? (u8 ? "decimate_box_u8" : "decimate_box")
                                      : (u8 ? "decimate_u8" : "decimate")),
      {buffer.get(), plan_.alias_decimated ? gray_buf_.get() : decimated_buf_.get()}, 8, plan_.wg2d,
      {config_.decimation});
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

void GpuDetector::RecordInitScan(VkCommandBuffer cmd) {
  struct { uint32_t dw, dh, min_blob; } pc{decimated_width_, decimated_height_,
                                          config_.min_cluster_pixels};
  if (plan_.sparse) {
    Pl(Stage::kUfInitScan).DispatchIndirect(cmd, cell_args_buf_.get(), 0, &pc);
    return;
  }
  Pl(Stage::kUfInitScan).Dispatch2D(cmd, (decimated_width_ + 3u) / 4u, decimated_height_, &pc);
}

void GpuDetector::RecordCountRuns(VkCommandBuffer cmd) {
  // The segment is uf_init_scan's: 4 pixels per thread of its workgroup.
  struct { uint32_t dw, dh, min_blob, segment; } pc{decimated_width_, decimated_height_,
                                                    config_.min_cluster_pixels,
                                                    plan_.sparse ? 64u : 4u * plan_.wg1d.x};
  if (plan_.sparse) {
    Pl(Stage::kUfCountRuns).DispatchIndirect(cmd, cell_args_buf_.get(), 0, &pc);
    return;
  }
  Pl(Stage::kUfCountRuns).Dispatch2D(cmd, (decimated_width_ + 3u) / 4u, decimated_height_, &pc);
}

void GpuDetector::RecordMerge(VkCommandBuffer cmd, vk::BarrierKind barrier) {
  struct { uint32_t dw, dh, band_mask, band_phase; } pc{decimated_width_, decimated_height_, 0u, 0u};
  const uint32_t columns = plan_.merge_x4 ? decimated_width_ / 4u : decimated_width_;
  if (plan_.sparse) {
    Pl(Stage::kUfMerge).DispatchIndirect(cmd, cell_args_buf_.get(), 0, &pc, barrier);
    return;
  }
  if (plan_.merge_band == 0u) {
    Pl(Stage::kUfMerge).Dispatch2D(cmd, columns, decimated_height_, &pc, barrier);
    return;
  }
  pc.band_mask = plan_.merge_band - 1u;
  Pl(Stage::kUfMerge).Dispatch2D(cmd, columns, decimated_height_, &pc);
  pc.band_phase = 1u;
  Pl(Stage::kUfMerge).Dispatch2D(cmd, columns, decimated_height_ / plan_.merge_band, &pc, barrier);
}

void GpuDetector::RecordUfChunk(VkCommandBuffer c, uint32_t iterations, bool trailing_compress) {
  using vk::BarrierKind;
  const uint32_t pixels = decimated_width_ * decimated_height_;

  // Records `iterations` labelling passes, clearing the flag before the last merge.
  // Without `trailing_compress` the last merge is a verification merge with nothing after it.
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const bool last = iter + 1 == iterations;
    const bool predicate = last && plan_.predicated_compress && trailing_compress;
    if (last) {
      // Also orders an earlier predicate read of the flag before the fill.
      vk::ComputePipeline::Barrier(c, predicate ? BarrierKind::ComputeTransferAndPredicate
                                                : BarrierKind::ComputeAndTransfer);
      uf_changed_buf_.FillZero(c);
      vk::ComputePipeline::Barrier(c, BarrierKind::ComputeAndTransfer);
    }
    if (last && !trailing_compress) {
      RecordMerge(c, BarrierKind::None);
      continue;
    }
    // The barrier makes the merge's flag write visible to the compression's guard or predicate read.
    RecordMerge(c, predicate ? BarrierKind::ComputeAndPredicate : BarrierKind::Compute);
    if (!trailing_compress && iter + 2 == iterations) {
      // The last real compression, just before the verification merge: it also counts blob sizes.
      const UfCountPc count_pc{decimated_width_, decimated_height_, 1u, config_.min_cluster_pixels};
      Pl(Stage::kUfCompressCount).Dispatch1D(c, pixels, &count_pc);
    } else if (predicate) {
      // Runs only when the flag is nonzero, so the guard is skipped.
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, 0u};
      ctx_.CmdBeginConditionalRendering(c, uf_changed_buf_.get(), 0);
      Pl(Stage::kUfCompress).Dispatch1D(c, pixels, &compress_pc, BarrierKind::None);
      ctx_.CmdEndConditionalRendering(c);
      vk::ComputePipeline::Barrier(c, BarrierKind::Compute);
    } else {
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, 1u};
      Pl(Stage::kUfCompress).Dispatch1D(c, pixels, &compress_pc);
    }
  }
}

void GpuDetector::RecordThreshold(VkCommandBuffer cmd, const ImportedFrame *imported) {
  // decimate takes the decimated dimensions directly; decimation is a specialisation constant.
  struct { uint32_t dw, dh; } dims_pc{decimated_width_, decimated_height_};
  struct { uint32_t dw, dh, bw, bh; } minmax_pc{decimated_width_, decimated_height_, block_width_,
                                                block_height_};
  if (imported != nullptr) {
    imported->decimate.Dispatch2D(cmd, decimated_width_, decimated_height_, &dims_pc);
    Pl(Stage::kBlockMinmax).Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  } else if (plan_.minmax_decimate) {
    Pl(Stage::kBlockMinmaxDecimate).Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  } else {
    if (!plan_.alias_decimated && !plan_.cpu_decimate) {
      Pl(Stage::kDecimate).Dispatch2D(cmd, decimated_width_, decimated_height_, &dims_pc);
    }
    Pl(Stage::kBlockMinmax).Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  }

  if (plan_.threshold_block) {
    struct { uint32_t dw, dh, min_diff, bw, bh; } thresh_pc{
        decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_,
        block_height_};
    if (plan_.adaptive_floor) {
      struct { uint32_t blocks, min_diff, scale_tenths; } floor_pc{
          block_width_ * block_height_, config_.min_white_black_diff, plan_.floor_scale_tenths};
      Pl(Stage::kNoiseFloor).DispatchRaw(cmd, 1, 1, 1, &floor_pc);
    }
    Pl(Stage::kThresholdBlock).Dispatch2D(cmd, block_width_, block_height_, &thresh_pc);
    if (plan_.sparse) {
      Pl(Stage::kCellArgs).DispatchRaw(cmd, 1, 1, 1, nullptr);
      vk::ComputePipeline::IndirectDispatchBarrier(cmd);
    }
    return;
  }
  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};
  Pl(Stage::kBlockFilter).Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);
  struct { uint32_t dw, dh, min_diff, bw; } thresh_pc{
      decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_};
  Pl(Stage::kThreshold).Dispatch2D(cmd, (decimated_width_ + kThresholdRun - 1) / kThresholdRun,
                          decimated_height_, &thresh_pc);
}

void GpuDetector::RecordFrameHead(VkCommandBuffer cmd, const ImportedFrame *imported,
                                  uint32_t chunk) {
  using vk::BarrierKind;
  const uint32_t pixels = decimated_width_ * decimated_height_;
  const VkDeviceSize gray_bytes = VkDeviceSize(config_.width) * config_.height;

  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};

  // One reset covers every span's timestamp pair for the frame (see vk::QueryPool).
  timestamp_pool_.Reset(cmd);
  if (ctx_.caps().timestamps_supported) vk::DispatchProbe::Begin(cmd, ctx_.device(), this);
  if (!plan_.gray_direct_write && !plan_.cpu_decimate && imported == nullptr && transfer_lane_ == nullptr) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanClear));
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  RecordLeanOffsetsClear(cmd);
  uf_changed_buf_.FillZero(cmd);
  if (plan_.sparse) {
    active_cells_buf_.FillZero(cmd);
    cell_list_buf_.FillZeroRange(cmd, 0, 4);
  }
  // uf_init_scan writes every run piece's entry for uf_count_runs, so the fill is needed only without
  // it.
  if (!plan_.count_runs) blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  large_sort_buf_.FillZeroRange(cmd, 0, 32);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanClear));
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
  vk::DispatchProbe::Mark(cmd, "<clears>");

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  RecordThreshold(cmd, imported);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  if (plan_.uf_init_scan) {
    RecordInitScan(cmd);
  } else {
    Pl(Stage::kUfInit).Dispatch2D(cmd, (decimated_width_ + kUfInitRun - 1) / kUfInitRun,
                          decimated_height_, &dwdh_pc);
  }
  if (!plan_.uf_verify) {
    // One merge joins every component. With plan_.count_runs the run pieces are counted and pointed at
    // their roots; otherwise a compression after the merge also counts blob sizes, skipping its
    // walk when the merge joined nothing only if parent[] was flat before the merge.
    if (!plan_.uf_init_scan) {
      const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
      Pl(Stage::kUfCompress).Dispatch1D(cmd, pixels, &compress_init_pc);
    }
    RecordMerge(cmd);
    if (plan_.count_runs) {
      RecordCountRuns(cmd);
    } else {
      const UfCountPc count_pc{decimated_width_, decimated_height_, plan_.uf_init_scan ? 0u : 1u,
                               config_.min_cluster_pixels};
      Pl(Stage::kUfCompressCount).Dispatch1D(cmd, pixels, &count_pc);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelling));
    return;
  }
  // Flattens the run chains uf_init.comp built before the first merge. honour_changed_flag is 0
  // because the flag is still zero from the clear.
  if (chunk == 1) {
    // The only merge is the verification merge, so this is the last real compression: it also
    // counts blob sizes (see RecordUfChunk).
    const UfCountPc count_pc{decimated_width_, decimated_height_, 0u, config_.min_cluster_pixels};
    Pl(Stage::kUfCompressCount).Dispatch1D(cmd, pixels, &count_pc);
  } else {
    const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
    Pl(Stage::kUfCompress).Dispatch1D(cmd, pixels, &compress_init_pc);
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

// Groups the boundary points by (rep0, rep1), reduces the extents, selects the blobs, assigns the
// point offsets and scatters the points. The grids are fixed (lean), indirect from device counts, or
// sized by the host from qbp_count (the unfused flow).
void GpuDetector::RecordGroupSelectScatter(VkCommandBuffer cmd, TailGrid grid, uint32_t qbp_count) {
  using vk::BarrierKind;
  const bool host = grid == TailGrid::kHost;
  const bool lean_grid = grid == TailGrid::kLean;
  const uint32_t count = host ? qbp_count : qbp_capacity_;
  const uint32_t count_from_buffer = host ? 0u : 1u;

  // Slot 1 of the indirect args: ceil(boundary points / wg1d), for hash_group, reduce_extents_hash
  // and scatter_index_points. The counts stay on the device.
  struct { uint32_t wg, slot, divide, clamp, quarter_slot; } qbp_args_pc{plan_.wg1d.x, 1u, 1u,
                                                                       qbp_capacity_, 2u};
  if (grid == TailGrid::kIndirect) {
    Pl(Stage::kBuildQbpArgs).DispatchRaw(cmd, 1, 1, 1, &qbp_args_pc, BarrierKind::None);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }

  // Groups the boundary points by (rep0, rep1) with a hash table; see hash_group.comp.
  // With count_from_buffer, `count` caps the device's counter at the buffer capacity.
  struct { uint32_t count, table_mask, max_probes, count_from_buffer, max_raw_blobs; } hash_pc{
      count, hash_table_size_ - 1u, 128u, count_from_buffer, config_.max_raw_blobs};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanHashGroup));
  if (lean_grid) {
    Pl(Stage::kHashGroup).DispatchRaw(cmd, lean_point_groups_, 1, 1, &hash_pc);
  } else if (host) {
    Pl(Stage::kHashGroup).Dispatch1D(cmd, qbp_count, &hash_pc);
  } else {
    Pl(Stage::kHashGroup).DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &hash_pc);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanHashGroup));

  // Builds select_blobs' dispatch arguments on the device from raw_blob_counter_buf_; it
  // bounds-checks against max_raw_blobs. reduce_extents_hash does not read them.
  struct { uint32_t wg, slot, divide, clamp, quarter_slot; } build_indirect_pc{
      plan_.wg1d.x, 0u, 1u, config_.max_raw_blobs, 0xFFFFFFFFu};
  if (!lean_grid) {
    Pl(Stage::kBuildBlobArgs).DispatchRaw(cmd, 1, 1, 1, &build_indirect_pc, BarrierKind::None);
  }

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanExtents));
  struct { uint32_t count, max_raw_blobs, count_from_buffer; } reduce_pc{count, config_.max_raw_blobs,
                                                                         count_from_buffer};
  if (lean_grid) {
    Pl(Stage::kReduceExtents).DispatchRaw(cmd, lean_point_groups_, 1, 1, &reduce_pc);
  } else {
    if (host) {
      Pl(Stage::kReduceExtents).Dispatch1D(cmd, plan_.reduce_x4 ? (qbp_count + 3u) / 4u : qbp_count, &reduce_pc);
    } else {
      Pl(Stage::kReduceExtents).DispatchIndirect(cmd, indirect_args_buf_.get(), plan_.reduce_x4 ? 24 : 12,
                                               &reduce_pc);
    }
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanExtents));

  struct {
    uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal,
        gx_gy_biased;
    float aspect_max, fill_min, fill_max;
  } select_pc{config_.max_raw_blobs,      config_.max_blobs,
              config_.tag_width,          config_.min_cluster_pixels,
              config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
              config_.normal_border ? 1u : 0u, plan_.extents_atomic64 ? 1u : 0u,
              config_.aspect_max,         config_.fill_min,
              config_.fill_max};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSelect));
  if (lean_grid) {
    Pl(Stage::kSelectBlobs).DispatchRaw(cmd, lean_select_groups_, 1, 1, &select_pc);
  } else {
    Pl(Stage::kSelectBlobs).DispatchIndirect(cmd, blob_args_buf_.get(), 0, &select_pc);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSelect));

  struct { uint32_t capacity, sort_args_slot; } scan_pc{config_.max_blobs, 1u};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBlobScan));
  if ((plan_.lean_tail & 2u) == 0) Pl(Stage::kScanBlobCounts).DispatchRaw(cmd, 1, 1, 1, &scan_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBlobScan));

  struct { uint32_t count, capacity, max_raw_blobs, count_from_buffer; } scatter_pc{
      count, ipoint_capacity_, config_.max_raw_blobs, count_from_buffer};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanScatter));
  if (lean_grid) {
    Pl(Stage::kScatterIndexPoints).DispatchRaw(cmd, lean_point_groups_, 1, 1, &scatter_pc,
                                         BarrierKind::ComputeAndTransfer);
  } else if (host) {
    Pl(Stage::kScatterIndexPoints).Dispatch1D(cmd, qbp_count, &scatter_pc, BarrierKind::ComputeAndTransfer);
  } else {
    Pl(Stage::kScatterIndexPoints).DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &scatter_pc,
                                              BarrierKind::ComputeAndTransfer);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanScatter));
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
  if (!speculative) Pl(Stage::kUfFinal).Dispatch1D(cmd, pixels, &uf_final_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanUfFinal));

  // blob_diff tests each pixel's blob size itself (against the same floor as uf_final) and appends
  // valid boundary points directly to the compacted buffer.
  struct { uint32_t w, h, capacity, honour_changed_flag, min_blob; } blobdiff_pc{
      decimated_width_, decimated_height_, qbp_capacity_, speculative && plan_.uf_verify ? 1u : 0u,
      config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  if (plan_.sparse) {
    Pl(Stage::kBlobDiff).DispatchIndirect(cmd, cell_args_buf_.get(), 0, &blobdiff_pc,
                                   BarrierKind::ComputeAndTransfer);
  } else {
    Pl(Stage::kBlobDiff).Dispatch2D(cmd, interior_width_, interior_height_, &blobdiff_pc,
                             BarrierKind::ComputeAndTransfer);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);

  RecordGroupSelectScatter(cmd, (plan_.lean_tail & 1u) != 0 ? TailGrid::kLean : TailGrid::kIndirect, 0);

  // The number of distinct (rep0, rep1) pairs this frame; profiling only.
  RecordCounterCopy(cmd, raw_blob_counter_buf_, kSlotRawBlobs);
  RecordCounterCopy(cmd, hash_drop_counter_buf_, kSlotHashDrops);

  // The sum of every selected blob's point count: blob_point_offsets_buf_[max_blobs - 1], the
  // inclusive scan total, or select_blobs' running total.
  if ((plan_.lean_tail & 2u) != 0) {
    RecordCounterCopy(cmd, points_total_buf_, kSlotPointCount);
  } else {
    blob_point_offsets_buf_.RecordCopyTo(
        cmd, counter_staging_, 4, VkDeviceSize(std::max(config_.max_blobs, 1u) - 1u) * 4,
        VkDeviceSize(kSlotPointCount) * 4);
  }
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);

  // The sort's arguments (one workgroup per selected blob) were written by scan_blob_counts or, with
  // lean offsets, by select_blobs.
  vk::ComputePipeline::IndirectDispatchBarrier(cmd);

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
  RecordSort(cmd, 0u, true);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));

  // Both readbacks are read in place on the fused path, so there is nothing to copy; the span
  // still marks the point where the results are final.
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanReadbackCopy));
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanReadbackCopy));
  // Counted by sort_points_local, so this is the first submit that can carry it back.
  RecordCounterCopy(cmd, oversized_sort_counter_buf_, kSlotOversizedSortBlobs);
  vk::ComputePipeline::HostReadBarrier(cmd);
  if (completion_flag_) completion_flag_->RecordSet(cmd);
}

namespace {

struct SortPc {
  uint32_t num_selected_blobs, count_from_buffer, max_blobs;
  int32_t dw, dh;
};

}  // namespace

// Sorts each blob's points by angle (sort_points_local_body.glsl); blobs above plan_.local_sort_split
// points are sorted in runs by several workgroups, then merged in rounds of one thread per point.
void GpuDetector::RecordSort(VkCommandBuffer cmd, uint32_t num_selected_blobs,
                             bool count_from_buffer) {
  using vk::BarrierKind;
  const SortPc pc{num_selected_blobs, count_from_buffer ? 1u : 0u, config_.max_blobs,
                  static_cast<int32_t>(decimated_width_), static_cast<int32_t>(decimated_height_)};
  if (count_from_buffer) {
    Pl(Stage::kSortLocal).DispatchIndirect(cmd, blob_args_buf_.get(), 12, &pc,
                                           BarrierKind::Compute);
  } else {
    Pl(Stage::kSortLocal).DispatchRaw(cmd, num_selected_blobs, 1, 1, &pc, BarrierKind::Compute);
  }
  // Blobs above plan_.local_sort_split points: their runs, then one pass placing every point.
  vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  Pl(Stage::kSortLargeRuns).DispatchIndirect(cmd, large_sort_buf_.get(), 0, &pc, BarrierKind::Compute);
  Pl(Stage::kSortLargeMerge).DispatchIndirect(cmd, large_sort_buf_.get(), 12, &pc,
                                        BarrierKind::ComputeAndTransfer);
}

// Flattens parent[] unconditionally; the retry path needs a flat forest after a discarded attempt.
void GpuDetector::RecordFlatten(VkCommandBuffer cmd) {
  const UfCompressPc compress_pc{decimated_width_, decimated_height_, 0u};
  Pl(Stage::kUfCompress).Dispatch1D(cmd, decimated_width_ * decimated_height_, &compress_pc);
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
  RecordLeanOffsetsClear(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd, vk::BarrierKind::ComputeAndTransfer);
  RecordFusedTail(cmd, /*speculative=*/false);
}

// Zeroes the running point total and the sort's group count and sets its y and z to 1, for
// select_blobs' offsets (lean tail bit 2).
void GpuDetector::RecordLeanOffsetsClear(VkCommandBuffer cmd) {
  if ((plan_.lean_tail & 2u) == 0) return;
  points_total_buf_.FillZero(cmd);
  blob_args_buf_.FillZeroRange(cmd, 12, 4);
  vkCmdFillBuffer(cmd, blob_args_buf_.get(), 16, 8, 1);
}

VkCommandBuffer GpuDetector::FindRecorded(RecordingKind kind, const uint8_t *key,
                                          uint32_t n) const {
  for (const RecordedFrame &r : recorded_frames_) {
    if (r.kind == kind && r.key == key && r.n == n) return r.cmd;
  }
  return VK_NULL_HANDLE;
}

VkCommandBuffer GpuDetector::RecordPersistent(
    const std::function<void(VkCommandBuffer)> &record) {
  VkCommandBuffer cmd = lane_.BeginPersistentCommands();
  try {
    record(cmd);
    lane_.EndCommands(cmd);
  } catch (...) {
    lane_.FreePersistentCommands(cmd);
    throw;
  }
  return cmd;
}

GpuDetector::Submission GpuDetector::FirstAttempt(const ImportedFrame *imported, uint32_t chunk) {
  const uint8_t *key = imported != nullptr ? imported->key : nullptr;
  if (plan_.prerecord) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kFirstAttempt, key, chunk)) {
      return {cached, true, completion_flag_ != nullptr};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd = RecordPersistent([&](VkCommandBuffer c) {
      RecordFrameHead(c, imported, chunk);
      RecordFusedTail(c, /*speculative=*/true);
    });
    recorded_frames_.push_back({RecordingKind::kFirstAttempt, key, chunk, cmd});
    return {cmd, true, completion_flag_ != nullptr};
  }
  recorded_this_frame_ = true;
  VkCommandBuffer cmd = BeginTimedCommands();
  RecordFrameHead(cmd, imported, chunk);
  RecordFusedTail(cmd, /*speculative=*/true);
  return {cmd, false};
}

GpuDetector::Submission GpuDetector::Continuation(uint32_t iterations) {
  if (plan_.prerecord) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kContinuation, nullptr, iterations)) {
      return {cached, true};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd =
        RecordPersistent([&](VkCommandBuffer c) { RecordUfContinuation(c, iterations); });
    recorded_frames_.push_back({RecordingKind::kContinuation, nullptr, iterations, cmd});
    return {cmd, true};
  }
  recorded_this_frame_ = true;
  VkCommandBuffer cmd = BeginTimedCommands();
  RecordUfContinuation(cmd, iterations);
  return {cmd, false};
}

GpuDetector::Submission GpuDetector::TailRerun() {
  if (plan_.prerecord) {
    if (VkCommandBuffer cached = FindRecorded(RecordingKind::kTailRerun, nullptr, 0)) {
      return {cached, true, completion_flag_ != nullptr};
    }
    recorded_this_frame_ = true;
    VkCommandBuffer cmd = RecordPersistent([&](VkCommandBuffer c) { RecordTailRerun(c); });
    recorded_frames_.push_back({RecordingKind::kTailRerun, nullptr, 0, cmd});
    return {cmd, true, completion_flag_ != nullptr};
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
  const VkSemaphore wait = std::exchange(pending_wait_, VK_NULL_HANDLE);
  if (s.persistent) {
    lane_.SubmitPersistentAndWait(s.cmd, wait, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  s.flagged ? completion_flag_.get() : nullptr);
  } else {
    lane_.SubmitAndWait(s.cmd, wait, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }
  last_profile_.cpu_submit_wait_ms += MsSince(t0, Clock::now());
}

void GpuDetector::Detect(const uint8_t *gray_frame) {
  if (keepalive_) keepalive_->Touch(auto_prewarm_.load(std::memory_order_relaxed));
  try {
    DetectFrame(gray_frame);
  } catch (...) {
    DropPendingWait();
    throw;
  }
}

void GpuDetector::Prewarm() {
  if (keepalive_) keepalive_->Prewarm();
}

void GpuDetector::NoteFrame() {
  if (keepalive_) keepalive_->NoteFrame();
}

void GpuDetector::ExpectGpuFrame() {
  if (keepalive_) keepalive_->ExpectGpuFrame();
}

void GpuDetector::DropPendingWait() noexcept {
  const VkSemaphore wait = std::exchange(pending_wait_, VK_NULL_HANDLE);
  if (wait == VK_NULL_HANDLE) return;
  try {
    lane_.SubmitAndWait(lane_.BeginCommands(), wait, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  } catch (...) {
    // The device is failing; the next frame reports it.
  }
}

void GpuDetector::UploadDecimated(const uint8_t *gray_frame) {
  CpuDecimate(gray_frame, config_.width, config_.height, config_.decimation, plan_.box_decimate, !plan_.u8,
              decimated_buf_.mapped());
}

void GpuDetector::DetectFrame(const uint8_t *gray_frame) {
  if (!plan_.fused_submits) {
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
  } else if (plan_.cpu_decimate) {
    UploadDecimated(gray_frame);
  } else if (plan_.gray_direct_write) {
    gray_buf_.Write(gray_frame, frame.gray_bytes);
  } else {
    upload_staging_.Write(gray_frame, frame.gray_bytes);
  }
  frame.t_upload1 = Clock::now();
  if (transfer_lane_ != nullptr && !plan_.gray_direct_write && frame.imported == nullptr) {
    // The staged frame is copied on the transfer queue; the first compute submission waits for it.
    VkCommandBuffer tcmd = transfer_lane_->BeginCommands();
    // The previous frame's copy into gray_buf_ was the last write on this queue.
    vk::ComputePipeline::TransferWriteBarrier(tcmd);
    gray_buf_.RecordCopyFrom(tcmd, upload_staging_, frame.gray_bytes);
    transfer_lane_->SubmitSignal(tcmd, upload_semaphore_);
    pending_wait_ = upload_semaphore_;
  }

  // Seeds the first labelling chunk with last frame's converged iteration count.
  uint32_t chunk = std::max(config_.uf_iterations_per_chunk, last_uf_iterations_);
  chunk = std::min(chunk, config_.max_uf_iterations);
  DetectFused(frame, plan_.uf_verify ? chunk : 1u);
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
  bool converged = !plan_.uf_verify || ReadCounterSlot(kSlotUfChanged) == 0;

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

// Reads the GPU timestamps of the frame into last_profile_ and the trace.
void GpuDetector::CollectGpuTimings() {
  vk::DispatchProbe::Collect(ctx_.device(), ctx_.caps().timestamp_period_ns, this);
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

  CollectGpuTimings();
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
  } else if (plan_.cpu_decimate) {
    UploadDecimated(gray_frame);
  } else if (plan_.gray_direct_write) {
    gray_buf_.Write(gray_frame, gray_bytes);
  } else {
    upload_staging_.Write(gray_frame, gray_bytes);
  }
  const auto t_upload1 = Clock::now();
  if (transfer_lane_ != nullptr && !plan_.gray_direct_write && imported == nullptr) {
    // The staged frame is copied on the transfer queue; the first compute submission waits for it.
    VkCommandBuffer tcmd = transfer_lane_->BeginCommands();
    // The previous frame's copy into gray_buf_ was the last write on this queue.
    vk::ComputePipeline::TransferWriteBarrier(tcmd);
    gray_buf_.RecordCopyFrom(tcmd, upload_staging_, gray_bytes);
    transfer_lane_->SubmitSignal(tcmd, upload_semaphore_);
    pending_wait_ = upload_semaphore_;
  }

  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};

  // ------------------------------------------------------------------
  // Submit 1: decimate, adaptive threshold, and the first chunk of
  // connected-component labelling.
  // ------------------------------------------------------------------
  VkCommandBuffer cmd = BeginTimedCommands();
  // One reset covers every span's timestamp pair for the frame (see vk::QueryPool).
  timestamp_pool_.Reset(cmd);
  if (ctx_.caps().timestamps_supported) vk::DispatchProbe::Begin(cmd, ctx_.device(), this);
  if (!plan_.gray_direct_write && !plan_.cpu_decimate && imported == nullptr && transfer_lane_ == nullptr) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanClear));
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  if (plan_.sparse) {
    active_cells_buf_.FillZero(cmd);
    cell_list_buf_.FillZeroRange(cmd, 0, 4);
  }
  if (!plan_.count_runs) blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  large_sort_buf_.FillZeroRange(cmd, 0, 32);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanClear));
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  RecordThreshold(cmd, imported);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  if (plan_.uf_init_scan) {
    RecordInitScan(cmd);
  } else {
    Pl(Stage::kUfInit).Dispatch2D(cmd, (decimated_width_ + kUfInitRun - 1) / kUfInitRun,
                          decimated_height_, &dwdh_pc);
  }
  // Flattens the run chains uf_init.comp built before the first merge. honour_changed_flag is 0
  // because the flag is still zero from the clear.
  const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
  if (!plan_.uf_init_scan) Pl(Stage::kUfCompress).Dispatch1D(cmd, pixels, &compress_init_pc);

  // Seeds the first chunk with last frame's converged iteration count.
  uint32_t chunk = std::max(config_.uf_iterations_per_chunk, last_uf_iterations_);
  chunk = std::min(chunk, config_.max_uf_iterations);
  if (plan_.uf_verify) {
    RecordUfChunk(cmd, chunk);
  } else {
    // One merge joins every component; the compression skips its walk when it joined nothing and
    // parent[] was flat before the merge.
    chunk = 1;
    RecordMerge(cmd);
    if (plan_.count_runs) {
      RecordCountRuns(cmd);
    } else {
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, plan_.uf_init_scan ? 0u : 1u};
      Pl(Stage::kUfCompress).Dispatch1D(cmd, pixels, &compress_pc);
    }
  }
  // The labelling span ends here; any extra convergence chunks are not attributed to it.
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelling));
  // Orders the compute writes before the transfer read below; the default Compute barrier does not
  // cover the transfer stage.
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
  // Copies the convergence flag, read back at once.
  RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);

  // ------------------------------------------------------------------
  // Submit 1 ends here: reads back convergence and, with UF_VERIFY, runs any extra labelling chunks.
  // ------------------------------------------------------------------
  uint32_t uf_iterations = chunk;
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);
  bool converged = !plan_.uf_verify || ReadCounterSlot(kSlotUfChanged) == 0;
  while (!converged && uf_iterations < config_.max_uf_iterations) {
    const uint32_t next =
        std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
    cmd = BeginTimedCommands();
    RecordUfChunk(cmd, next);
    // The default compute barrier does not cover the transfer read of this copy.
    vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);
    RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
    uf_iterations += next;
    converged = ReadCounterSlot(kSlotUfChanged) == 0;
  }

  // ------------------------------------------------------------------
  // Submit 2: blob sizes and boundary extraction.
  // ------------------------------------------------------------------
  cmd = BeginTimedCommands();
  const auto t_label = Clock::now();

  // uf_final counts to the size floor, which must equal blob_diff's.
  struct { uint32_t dw, dh, min_blob; } uf_final_pc{decimated_width_, decimated_height_,
                                                    config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanUfFinal));
  // uf_count_runs has already counted with plan_.count_runs.
  if (!plan_.count_runs) Pl(Stage::kUfFinal).Dispatch1D(cmd, pixels, &uf_final_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanUfFinal));

  // blob_diff tests each pixel's blob size itself (against the same floor as uf_final) and appends
  // valid boundary points directly to the compacted buffer.
  struct { uint32_t w, h, capacity, honour_changed_flag, min_blob; } blobdiff_pc{
      decimated_width_, decimated_height_, qbp_capacity_, 0u, config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  if (plan_.sparse) {
    Pl(Stage::kBlobDiff).DispatchIndirect(cmd, cell_args_buf_.get(), 0, &blobdiff_pc,
                                   BarrierKind::ComputeAndTransfer);
  } else {
    Pl(Stage::kBlobDiff).Dispatch2D(cmd, interior_width_, interior_height_, &blobdiff_pc,
                             BarrierKind::ComputeAndTransfer);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);

  const uint32_t qbp_count = std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const auto t_boundary = Clock::now();

  // ------------------------------------------------------------------
  // Submit 3: group boundary points by (rep0, rep1), select plausible quads, and scatter the
  // survivors into per-blob runs.
  // ------------------------------------------------------------------
  cmd = BeginTimedCommands();
  if (qbp_count > 0) {
    RecordGroupSelectScatter(cmd, TailGrid::kHost, qbp_count);

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
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);

  const uint32_t num_raw_blobs = (qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0;
  const uint32_t hash_probe_drops = (qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0;
  // Blobs that passed the filters, which can exceed max_blobs; kept unclamped to report
  // selected_blob_drops.
  const uint32_t qualifying_blobs = ReadCounterSlot(kSlotSelectedCount);
  const uint32_t num_selected_blobs = std::min(qualifying_blobs, config_.max_blobs);
  const uint32_t selected_blob_drops = qualifying_blobs - num_selected_blobs;
  const uint32_t num_points =
      (qbp_count > 0) ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_) : 0;
  const auto t_sort_group = Clock::now();

  // ------------------------------------------------------------------
  // Submit 4: sort each blob's points around its perimeter, compute line-fit moments, and stage
  // both payloads for readback.
  // ------------------------------------------------------------------
  const VkDeviceSize extents_bytes = VkDeviceSize(num_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize linefit_offset = (extents_bytes + 15) & ~VkDeviceSize(15);
  const VkDeviceSize linefit_bytes = VkDeviceSize(num_points) * sizeof(RawLineFitPoint);
  EnsureReadbackCapacity(linefit_offset + linefit_bytes);

  cmd = BeginTimedCommands();
  if (num_points > 0) {
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
    RecordSort(cmd, num_selected_blobs, false);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanReadbackCopy));
  if (extents_bytes > 0 && !plan_.extents_direct_read) {
    selected_extents_buf_.RecordCopyTo(cmd, readback_staging_, extents_bytes, 0, 0);
  }
  // Skipped when the line-fit buffer is host-visible and cached; QuadDecode reads it in place.
  if (linefit_bytes > 0 && !plan_.linefit_direct_read) {
    line_fit_points_buf_.RecordCopyTo(cmd, readback_staging_, linefit_bytes, 0, linefit_offset);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanReadbackCopy));
  // Counted by sort_points_local, so this is the first submit that can carry it back.
  RecordCounterCopy(cmd, oversized_sort_counter_buf_, kSlotOversizedSortBlobs);
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);
  const auto t_linefit = Clock::now();
  const uint32_t oversized_sort_blobs = ReadCounterSlot(kSlotOversizedSortBlobs);

  // ------------------------------------------------------------------
  // Host-side copies out of the persistently mapped readback buffer.
  // ------------------------------------------------------------------
  last_selected_extents.resize(num_selected_blobs);
  if (extents_bytes > 0) {
    if (plan_.extents_direct_read) {
      if (!selected_extents_buf_.coherent()) {
        selected_extents_buf_.InvalidateRange(0, extents_bytes);
      }
      std::memcpy(last_selected_extents.data(), selected_extents_buf_.mapped(),
                  static_cast<size_t>(extents_bytes));
    } else {
      readback_staging_.Read(last_selected_extents.data(), extents_bytes, 0);
    }
  }
  if (plan_.linefit_direct_read) {
    // Zero-copy view of the mapping, valid until the next Detect(); invalidates first when the
    // memory is not coherent.
    if (!line_fit_points_buf_.coherent()) {
      line_fit_points_buf_.InvalidateRange(0, linefit_bytes);
    }
    last_line_fit_points = std::span<const RawLineFitPoint>(
        static_cast<const RawLineFitPoint *>(line_fit_points_buf_.mapped()), num_points);
  } else {
    linefit_scratch_.resize(num_points);
    if (linefit_bytes > 0) {
      readback_staging_.Read(linefit_scratch_.data(), linefit_bytes, linefit_offset);
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
  last_profile_.selected_blobs = num_selected_blobs;
  last_profile_.points = num_points;
  last_profile_.boundary_points = qbp_count;
  last_profile_.raw_blobs = num_raw_blobs;
  last_profile_.hash_probe_drops = hash_probe_drops;
  last_profile_.selected_blob_drops = selected_blob_drops;
  last_profile_.oversized_sort_blobs = oversized_sort_blobs;
  // uf_iterations and uf_converged are set at the end of this function.
  last_profile_.submits = static_cast<uint32_t>(lane_.submit_count - submits_at_start);

  CollectGpuTimings();

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
  os << ", gray upload "
     << (plan_.cpu_decimate ? "decimated on the CPU into mapped device memory"
         : plan_.gray_direct_write ? "direct (mapped device memory)" : "staged");
  // Whether the line-fit readback avoided its copies depends on a host-cached memory type.
  os << ", linefit readback " << (plan_.linefit_direct_read ? "direct (host-cached)" : "staged");
  os << ", uf_compress skip "
     << (plan_.predicated_compress ? "predicated (conditional rendering)" : "in-shader");
  return os.str();
}

}  // namespace apriltag_vulkan
