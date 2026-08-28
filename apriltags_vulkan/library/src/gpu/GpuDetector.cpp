#include "vkapriltag/gpu/GpuDetector.h"

#include "vkapriltag/vk/EmbeddedShaders.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace apriltag_vulkan {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Prefers SPIR-V compiled into the binary, falling back to SHADER_DIR when
// this build has none (VKAPRILTAG_EMBED_SHADERS=OFF). Embedding is what lets
// the library run somewhere with no install prefix - see EmbeddedShaders.h.
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
constexpr VkDeviceSize kCounterStagingBytes = 64;

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : ctx_(ctx), config_(config) {
  if (config_.width % 2 != 0 || config_.height % 2 != 0) {
    throw std::runtime_error("width and height must both be even");
  }

  // --- Environment overrides. These must all be applied before any capacity
  // or launch geometry is derived from the config. ---

  // The boundary-point cap is the main device-memory lever on unified-memory
  // parts, where the dense worst-case sizing is expensive.
  if (const char *cap = std::getenv("APRILTAG_VK_MAX_POINTS")) {
    const long parsed = std::strtol(cap, nullptr, 10);
    if (parsed > 0) config_.max_boundary_points = static_cast<uint32_t>(parsed);
  }
  // Labelling chunk size. Smaller chunks detect convergence more precisely at
  // the cost of an extra round trip when they guess low.
  if (const char *chunk = std::getenv("APRILTAG_VK_UF_CHUNK")) {
    const long parsed = std::strtol(chunk, nullptr, 10);
    if (parsed > 0) config_.uf_iterations_per_chunk = static_cast<uint32_t>(parsed);
  }
  if (config_.uf_iterations_per_chunk == 0) config_.uf_iterations_per_chunk = 1;

  // Scale-relative minimum cluster size: derives a floor for
  // min_cluster_pixels from the smallest tag side (full-resolution pixels)
  // the caller declares, then raises min_cluster_pixels to that floor (never
  // lowers it, so an explicit min_cluster_pixels always still applies).
  if (const char *min_tag_px = std::getenv("APRILTAG_VK_MIN_TAG_PX")) {
    const long parsed = std::strtol(min_tag_px, nullptr, 10);
    if (parsed > 0) config_.min_tag_pixels = static_cast<uint32_t>(parsed);
  }
  if (config_.min_tag_pixels > 0) {
    // Perimeter, in decimated-grid boundary points, of a square tag whose
    // full-resolution side is min_tag_pixels, at this pipeline's fixed 2x
    // decimation: 4 sides x (min_tag_pixels / 2) decimated pixels each.
    const uint32_t derived_min_cluster = 2u * config_.min_tag_pixels;
    config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, derived_min_cluster);
  }

  // Launch geometry comes from the device, never from a literal.
  const vk::DeviceCaps &caps = ctx_.caps();
  wg1d_ = vk::WorkgroupSize{caps.wg1d, 1, 1};
  wg2d_ = vk::WorkgroupSize{caps.wg2d_x, caps.wg2d_y, 1};
  scan_wg_ = caps.scan_wg;

  decimated_width_ = config_.width / 2;
  decimated_height_ = config_.height / 2;
  // Rounded up rather than floored: the input is only guaranteed even, not a
  // multiple of 8, so decimated_width_/height_ need not be a multiple of 4.
  // block_minmax.comp/threshold.comp handle the resulting ragged trailing
  // block explicitly (edge-clamped sampling / an explicit block_width push
  // constant) rather than assuming block_width_ * 4 == decimated_width_.
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

  // The grayscale frame is uploaded packed, four 8-bit pixels per uint32.
  // width and height are each guaranteed even (checked above), so their
  // product is guaranteed divisible by four.
  gray_words_ = (config_.width * config_.height) / 4u;

  // sort_points_local.comp handles one selected blob per workgroup, sorting
  // its points entirely in shared memory (2 words/point: key + local index).
  // local_sort_cap_ is the workgroup's thread count, bounded by the device's
  // max workgroup invocation count (a real dispatch limit); it no longer
  // needs to also bound the shared array size (see local_sort_virtual_cap_
  // below), but 1024 remains a sane ceiling matching radix_wg_/wg1d_'s
  // derivation from caps below.
  local_sort_cap_ = PrevPow2(std::max(caps.max_workgroup_invocations, 1u));
  // 256 rather than the device maximum. Once the bitonic network is sized to
  // the blob (see sort_points_local.comp) the typical network is ~256 slots
  // wide, so a 1024-thread workgroup leaves three quarters of its lanes idle
  // in every stage while still reserving the full shared-memory allocation.
  // Measured on Mali-G610 at 1080p: 1024 threads 4.03 ms, 512 threads 3.08 ms,
  // 256 threads 3.08 ms, 128 threads 3.83 ms, 64 threads 5.71 ms.
  local_sort_cap_ = std::min(local_sort_cap_, 256u);

  // The shader's virtual per-blob capacity is decoupled from the workgroup's
  // thread count (each thread handles multiple elements in a strided
  // pattern), so it only needs to fit the shared-memory budget - not the
  // device's max workgroup invocation count. This lets blobs bigger than the
  // device can run as a single workgroup (e.g. a large tag's own border)
  // still get properly angle-sorted instead of falling back to an unsorted
  // identity copy. Capped at 4096 (32 KiB on an 8-byte/element budget) rather
  // than using the full shared-memory budget: every workgroup allocates this
  // much shared memory regardless of the blob it draws, so sizing it to the
  // device's absolute max here would collapse occupancy for the vast
  // majority of (much smaller) blobs. 4096 comfortably covers real tag
  // borders (a 1920x1080 frame's largest observed tag border is ~1200
  // points) while leaving most of the shared-memory budget free.
  local_sort_virtual_cap_ =
      std::min(PrevPow2(std::max(caps.max_shared_memory_bytes / 8u, 1u)), 2048u);

  CreateBuffers();
  CreatePipelines();

  // Persist any newly-compiled pipelines now rather than relying solely on
  // ~Context(): a process that gets killed (common for a camera-loop
  // binary) rather than shut down cleanly would otherwise lose the cache
  // every time. Cheap when nothing changed - see PipelineCache::Save().
  ctx_.FlushPipelineCache();

  // Opt-in per-shader GPU timing (see DetectProfile::gpu_stage_ms /
  // kGpuStageNames). Off by default: a query pool costs nothing idle, but
  // constructing one and threading the reset/write/read calls through every
  // frame is pure overhead a normal run shouldn't pay for.
  if (const char *ts = std::getenv("APRILTAG_VK_TIMESTAMPS")) {
    if (std::strtol(ts, nullptr, 10) != 0 && caps.timestamps_supported) {
      timestamp_pool_ = vk::QueryPool(ctx_.device(), kNumGpuStageSpans * 2);
      timestamps_enabled_ = timestamp_pool_.valid();
    }
  }
}

void GpuDetector::CreateBuffers() {
  auto ssbo = [this](VkDeviceSize bytes) {
    vk::Buffer b(ctx_, std::max<VkDeviceSize>(bytes, 4), kSsboUsage, vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    return b;
  };

  // On unified-memory parts (Mali and other integrated GPUs) the grayscale
  // buffer is host-visible device memory written directly by the CPU, which
  // removes the staging copy entirely. On discrete cards there is no such
  // memory type (absent resizable BAR), so Buffer falls back to device-local
  // and we DMA through a persistently mapped staging buffer instead.
  gray_buf_ = vk::Buffer(
      ctx_, VkDeviceSize(gray_words_) * 4,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      ctx_.caps().unified_memory ? vk::MemoryKind::DeviceLocalMapped
                                 : vk::MemoryKind::DeviceLocal);
  device_bytes_ += gray_buf_.size();
  gray_direct_write_ = gray_buf_.host_visible();

  if (!gray_direct_write_) {
    upload_staging_ = vk::Buffer(ctx_, VkDeviceSize(gray_words_) * 4,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::MemoryKind::HostVisible);
  }
  counter_staging_ = vk::Buffer(ctx_, kCounterStagingBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                vk::MemoryKind::HostVisibleCached);

  decimated_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  minmax_unfiltered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  thresholded_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);

  parent_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  uf_changed_buf_ = ssbo(4);
  pixel_label_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);

  qbp_compacted_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * sizeof(QBPoint));
  qbp_counter_buf_ = ssbo(4);
  // Only the grouping hash reads these now, so they are sized to the actual
  // point capacity rather than the power of two a bitonic network needed.
  qbp_keys_hi_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);
  qbp_keys_lo_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);

  extents_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * sizeof(MinMaxExtentsGpu));
  selected_extents_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu));
  selected_counter_buf_ = ssbo(4);
  remap_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * 4);

  index_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  index_points_counter_buf_ = ssbo(4);
  index_points_sorted_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  blob_point_offsets_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * 4);

  line_fit_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(RawLineFitPoint));

  // Open-addressing table for the (rep0, rep1) grouping, sized to
  // max_raw_blobs itself (previously 4x, for a 25% worst-case load factor -
  // but that worst case is the same defensive margin every other capacity
  // clamp in this pipeline already provides: a frame that actually reaches
  // max_raw_blobs distinct pairs degrades to dropping the excess via the
  // probe-cap fallback in hash_group.comp, exactly like select_blobs.comp's
  // counter clamp or qbp_compacted_buf_'s capacity does elsewhere). A real
  // 1080p frame's raw blob count is a couple orders of magnitude below
  // max_raw_blobs (734 of 65536, measured), so this table is sparse at
  // realistic load either way - the 4x only mattered for a pathological
  // frame this table's own probe-cap fallback already covers.
  hash_table_size_ = NextPow2(std::max(config_.max_raw_blobs, 1u));
  hash_owner_buf_ = ssbo(VkDeviceSize(hash_table_size_) * 4);
  slot_dense_buf_ = ssbo(VkDeviceSize(hash_table_size_) * 4);
  point_slot_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);
  blob_cursor_buf_ = ssbo(VkDeviceSize(std::max(config_.max_blobs, 1u)) * 4);
  raw_blob_counter_buf_ = ssbo(4);
  hash_drop_counter_buf_ = ssbo(4);

  blob_scan_chain_ = BuildScanChain(std::max(config_.max_blobs, 1u));

  // Readback staging starts at a size that covers the extents plus a healthy
  // number of line-fit points, and grows on demand. Steady state therefore
  // performs no allocation at all.
  const VkDeviceSize initial_readback =
      VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu) + 16 +
      VkDeviceSize(std::min<uint32_t>(ipoint_capacity_, 1u << 16)) * sizeof(RawLineFitPoint);
  EnsureReadbackCapacity(std::max<VkDeviceSize>(initial_readback, 4096));
}

void GpuDetector::EnsureReadbackCapacity(VkDeviceSize bytes) {
  if (bytes <= readback_capacity_) return;
  // Grow geometrically so a slowly rising point count cannot cause a
  // reallocation every frame.
  VkDeviceSize new_capacity = std::max<VkDeviceSize>(readback_capacity_ * 2, bytes);
  readback_staging_ = vk::Buffer(ctx_, new_capacity, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 vk::MemoryKind::HostVisibleCached);
  readback_capacity_ = readback_staging_.size();
}

GpuDetector::ScanChain GpuDetector::BuildScanChain(uint32_t capacity) {
  ScanChain chain;
  uint32_t size = capacity;
  // Fan-out per level is the scan block size, which is device dependent - it
  // must match scan_block.comp's specialized local_size_x exactly.
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
  decimate_pl_ = vk::ComputePipeline(ctx_, ShaderPath("decimate"),
                                     {gray_buf_.get(), decimated_buf_.get()}, 8, wg1d_);
  block_minmax_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_minmax"), {decimated_buf_.get(), minmax_unfiltered_buf_.get()}, 16,
      wg2d_);
  block_filter_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_filter"), {minmax_unfiltered_buf_.get(), minmax_filtered_buf_.get()},
      8, wg2d_);
  threshold_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("threshold"),
      {decimated_buf_.get(), minmax_filtered_buf_.get(), thresholded_buf_.get()}, 16, wg1d_);

  uf_init_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_init"), {parent_buf_.get(), thresholded_buf_.get()}, 8, wg1d_);
  uf_merge_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_merge"),
      {parent_buf_.get(), thresholded_buf_.get(), uf_changed_buf_.get()}, 8, wg1d_);
  uf_compress_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("uf_compress"), {parent_buf_.get()}, 8, wg1d_);
  uf_final_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_final"),
                                     {parent_buf_.get(), blob_size_buf_.get()}, 8, wg1d_);

  blob_diff_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("blob_diff"),
      {thresholded_buf_.get(), pixel_label_buf_.get(), qbp_compacted_buf_.get(),
       qbp_counter_buf_.get(), qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get()},
      12, wg1d_);

  label_pixels_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("label_pixels"),
      {parent_buf_.get(), blob_size_buf_.get(), pixel_label_buf_.get()}, 8, wg1d_);

  init_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("init_extents"), {extents_buf_.get()}, 4, wg1d_);

  select_blobs_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("select_blobs"),
      {extents_buf_.get(), selected_extents_buf_.get(), selected_counter_buf_.get(),
       remap_buf_.get()},
      40, wg1d_);

  // Per-blob point base-offset assignment: extract_blob_counts.comp copies
  // each selected blob's point count into blob_point_offsets_buf_, then the
  // same chain-scan pattern turns it into an inclusive prefix sum so
  // rewrite_index_points.comp / sort_points_local.comp can place/find each
  // blob's points at a deterministic, contiguous range.
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

  sort_points_local_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("sort_points_local"),
      {selected_extents_buf_.get(), blob_point_offsets_buf_.get(), index_points_buf_.get(),
       index_points_sorted_buf_.get()},
      4, vk::WorkgroupSize{local_sort_cap_, 1, 1}, {local_sort_virtual_cap_});

  hash_group_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("hash_group"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), hash_owner_buf_.get(),
       point_slot_buf_.get(), slot_dense_buf_.get(), raw_blob_counter_buf_.get(),
       hash_drop_counter_buf_.get()},
      12, wg1d_);
  reduce_extents_hash_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("reduce_extents_hash"),
      {qbp_compacted_buf_.get(), point_slot_buf_.get(), slot_dense_buf_.get(),
       hash_owner_buf_.get(), extents_buf_.get()},
      8, wg1d_);
  scatter_index_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("scatter_index_points"),
      {qbp_compacted_buf_.get(), point_slot_buf_.get(), slot_dense_buf_.get(), remap_buf_.get(),
       selected_extents_buf_.get(), blob_point_offsets_buf_.get(), blob_cursor_buf_.get(),
       index_points_buf_.get(), index_points_counter_buf_.get()},
      12, wg1d_);

  compute_line_fit_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("compute_line_fit_points"),
      {index_points_sorted_buf_.get(), decimated_buf_.get(), line_fit_points_buf_.get()}, 12,
      wg1d_);

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

uint32_t GpuDetector::ReadCounterSlot(uint32_t slot) const {
  uint32_t value = 0;
  counter_staging_.Read(&value, 4, VkDeviceSize(slot) * 4);
  return value;
}

void GpuDetector::Detect(const uint8_t *gray_frame) {
  using vk::BarrierKind;
  const auto t_begin = Clock::now();
  last_profile_ = DetectProfile{};
  const uint64_t submits_at_start = ctx_.submit_count;

  const uint32_t pixels = decimated_width_ * decimated_height_;
  const VkDeviceSize gray_bytes = VkDeviceSize(config_.width) * config_.height;

  // ------------------------------------------------------------------
  // Upload. One memcpy of the raw 8-bit frame - no widening loop, and a
  // quarter of the bytes the uint32-per-pixel layout used to move.
  // ------------------------------------------------------------------
  const auto t_upload0 = Clock::now();
  if (gray_direct_write_) {
    gray_buf_.Write(gray_frame, gray_bytes);
  } else {
    upload_staging_.Write(gray_frame, gray_bytes);
  }
  const auto t_upload1 = Clock::now();

  // Push constants shared across several stages.
  struct { uint32_t w, h; } dims_pc{config_.width, config_.height};
  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};
  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};

  // ------------------------------------------------------------------
  // Submit 1: decimate, adaptive threshold, and the first chunk of
  // connected-component labelling.
  // ------------------------------------------------------------------
  VkCommandBuffer cmd = ctx_.BeginCommands();
  // One reset covers every span's timestamp pair for the whole frame: every
  // later submission this frame runs strictly after this one's fence has
  // signaled (SubmitAndWait blocks), so nothing after this point can race a
  // query this reset just cleared. See vk::QueryPool's own comment.
  timestamp_pool_.Reset(cmd);
  if (!gray_direct_write_) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  index_points_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  struct { uint32_t dw, dh, bw, bh; } minmax_pc{decimated_width_, decimated_height_, block_width_,
                                                block_height_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  decimate_pl_.Dispatch1D(cmd, pixels, &dims_pc);
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff, bw; } thresh_pc{
      decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_};
  threshold_pl_.Dispatch1D(cmd, pixels, &thresh_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  uf_init_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);
  // Flatten the run chains uf_init.comp just built before the first merge
  // pass walks them. Without this the vertical unions pay an O(run length)
  // find() each, which cancels out exactly what the run-based init saved.
  uf_compress_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);

  // Records `iterations` labelling passes. The convergence flag is cleared
  // immediately before the LAST merge of the chunk, so a zero readback means
  // "the final pass changed nothing", i.e. genuinely converged - clearing it
  // once at the start of the chunk instead would conflate "converged" with
  // "changed something earlier in this chunk" and never terminate tightly.
  auto record_uf_chunk = [&](VkCommandBuffer c, uint32_t iterations) {
    for (uint32_t iter = 0; iter < iterations; ++iter) {
      if (iter + 1 == iterations) {
        vk::ComputePipeline::Barrier(c, BarrierKind::ComputeAndTransfer);
        uf_changed_buf_.FillZero(c);
        vk::ComputePipeline::Barrier(c, BarrierKind::ComputeAndTransfer);
      }
      uf_merge_pl_.Dispatch1D(c, pixels, &dwdh_pc);
      uf_compress_pl_.Dispatch1D(c, pixels, &dwdh_pc);
    }
  };

  // Seed the first chunk with however many iterations converged last frame, so
  // steady-state video settles in one chunk plus its verification rather than
  // rediscovering the count in 8-iteration steps every frame.
  uint32_t chunk = std::max(config_.uf_iterations_per_chunk, last_uf_iterations_);
  chunk = std::min(chunk, config_.max_uf_iterations);
  record_uf_chunk(cmd, chunk);
  // "labelling" ends here even if a rare extra convergence chunk follows
  // below (see the while loop) - a query can't be rewritten without an
  // intervening reset, and the common case (this corpus, default config)
  // always converges within the first chunk. The wall-clock
  // threshold_label_ms figure still includes any extra chunks; only this
  // per-shader breakdown misses them.
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelling));
  RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);

  uint32_t uf_iterations = chunk;
  bool converged = ReadCounterSlot(kSlotUfChanged) == 0;
  while (!converged && uf_iterations < config_.max_uf_iterations) {
    const uint32_t next =
        std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
    cmd = ctx_.BeginCommands();
    record_uf_chunk(cmd, next);
    RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
    vk::ComputePipeline::HostReadBarrier(cmd);
    ctx_.SubmitAndWait(cmd);
    uf_iterations += next;
    converged = ReadCounterSlot(kSlotUfChanged) == 0;
  }
  last_uf_iterations_ = uf_iterations;
  const auto t_label = Clock::now();

  // ------------------------------------------------------------------
  // Submit 2: blob sizes, boundary point extraction, and compaction. The
  // compacted count is the number every later stage is sized by.
  // ------------------------------------------------------------------
  cmd = ctx_.BeginCommands();
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelFinalize));
  uf_final_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);

  // Fold blob identity and the min-size test into one spatially-local value
  // per pixel, so blob_diff.comp does no random gathers at all. See
  // label_pixels.comp.
  struct { uint32_t count, min_blob; } label_pc{pixels, config_.min_cluster_pixels};
  label_pixels_pl_.Dispatch1D(cmd, pixels, &label_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelFinalize));

  // blob_diff appends valid boundary points (with their sort keys) directly
  // into the compacted buffer, so there is no dense intermediate array and no
  // separate full-capacity compaction pass.
  struct { uint32_t w, h, capacity; } blobdiff_pc{decimated_width_, decimated_height_,
                                                   qbp_capacity_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  blob_diff_pl_.Dispatch1D(cmd, interior_width_ * interior_height_, &blobdiff_pc,
                           BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);

  const uint32_t qbp_count = std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const auto t_boundary = Clock::now();

  // ------------------------------------------------------------------
  // Submit 3: group the boundary points by (rep0, rep1), select plausible tag
  // quads, and scatter the survivors into per-blob contiguous runs.
  //
  // Everything here is dispatched over qbp_count, not qbp_capacity_ - at 1080p
  // a couple of hundred thousand real points rather than the ~2M dense bound.
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  cmd = ctx_.BeginCommands();
  if (qbp_count > 0) {
    // Group the boundary points by (rep0, rep1) with a hash table instead of
    // sorting them. See hash_group.comp: the pipeline only ever needed the
    // grouping, never the order.
    struct { uint32_t count, table_mask, max_probes; } hash_pc{
        qbp_count, hash_table_size_ - 1u, 128u};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanHashGroup));
    hash_group_pl_.Dispatch1D(cmd, qbp_count, &hash_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanHashGroup));

    // init_extents and select_blobs are dispatched over max_raw_blobs rather
    // than the frame's actual raw blob count, which only exists on the device
    // at this point. That is about 6 MB of wasted traffic, worth 0.25 ms here.
    // Buying the exact count with an extra submit + fence was measured to be a
    // wash (the round trip costs what the traffic does), so the right fix is
    // vkCmdDispatchIndirect - see OPTIMIZATION_NOTES.md.
    struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanExtents));
    init_extents_pl_.Dispatch1D(cmd, config_.max_raw_blobs, &extentscap_pc);

    struct { uint32_t count, max_raw_blobs; } reduce_pc{qbp_count, config_.max_raw_blobs};
    reduce_extents_hash_pl_.Dispatch1D(cmd, qbp_count, &reduce_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanExtents));

    struct {
      uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal;
      float aspect_max, fill_min, fill_max;
    } select_pc{config_.max_raw_blobs,      config_.max_blobs,
                config_.tag_width,          config_.min_cluster_pixels,
                config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
                config_.normal_border ? 1u : 0u,
                config_.aspect_max,         config_.fill_min,
                config_.fill_max};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSelect));
    select_blobs_pl_.Dispatch1D(cmd, config_.max_raw_blobs, &select_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSelect));

    struct { uint32_t capacity; } extract_pc{config_.max_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBlobScan));
    extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, &extract_pc);
    RunInclusiveScan(cmd, config_.max_blobs, blob_scan_chain_, blob_scan_block_pls_,
                     blob_scan_add_offsets_pls_);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBlobScan));

    struct { uint32_t count, capacity, max_raw_blobs; } scatter_pc{qbp_count, ipoint_capacity_,
                                                                   config_.max_raw_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanScatter));
    scatter_index_points_pl_.Dispatch1D(cmd, qbp_count, &scatter_pc,
                                        BarrierKind::ComputeAndTransfer);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanScatter));

    // raw_blob_counter_buf_ is now the number of distinct (rep0, rep1) pairs
    // this frame - assigned directly by hash_group.comp's winning CAS thread
    // (see its comment), so this is the frame's raw blob count with no scan
    // needed. Profiling only.
    RecordCounterCopy(cmd, raw_blob_counter_buf_, kSlotRawBlobs);
    RecordCounterCopy(cmd, hash_drop_counter_buf_, kSlotHashDrops);
  }
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);
  RecordCounterCopy(cmd, index_points_counter_buf_, kSlotPointCount);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);

  const uint32_t num_raw_blobs = (qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0;
  const uint32_t hash_probe_drops = (qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0;
  const uint32_t num_selected_blobs =
      std::min(ReadCounterSlot(kSlotSelectedCount), config_.max_blobs);
  const uint32_t num_points = std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_);
  const auto t_sort_group = Clock::now();

  // ------------------------------------------------------------------
  // Submit 4: sort the surviving points around each blob's perimeter, compute
  // per-point line-fit moments, and stage both payloads for readback.
  // ------------------------------------------------------------------
  const VkDeviceSize extents_bytes = VkDeviceSize(num_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize linefit_offset = (extents_bytes + 15) & ~VkDeviceSize(15);
  const VkDeviceSize linefit_bytes = VkDeviceSize(num_points) * sizeof(RawLineFitPoint);
  EnsureReadbackCapacity(linefit_offset + linefit_bytes);

  cmd = ctx_.BeginCommands();
  if (num_points > 0) {
    // rewrite_index_points.comp already packed every selected blob's points
    // into one contiguous range, so sorting each blob's own points into
    // angular order is a one-workgroup-per-blob shared-memory sort - no
    // composite (blob_index, theta) key, no separate gather pass.
    struct { uint32_t num_selected_blobs; } sort_pc{num_selected_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
    sort_points_local_pl_.DispatchRaw(cmd, num_selected_blobs, 1, 1, &sort_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));

    struct { uint32_t count; int dw, dh; } linefit_pc{
        num_points, static_cast<int>(decimated_width_), static_cast<int>(decimated_height_)};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLinefitCompute));
    compute_line_fit_points_pl_.Dispatch1D(cmd, num_points, &linefit_pc,
                                           BarrierKind::ComputeAndTransfer);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLinefitCompute));
  }
  if (extents_bytes > 0) {
    selected_extents_buf_.RecordCopyTo(cmd, readback_staging_, extents_bytes, 0, 0);
  }
  if (linefit_bytes > 0) {
    line_fit_points_buf_.RecordCopyTo(cmd, readback_staging_, linefit_bytes, 0, linefit_offset);
  }
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);
  const auto t_linefit = Clock::now();

  // ------------------------------------------------------------------
  // Host-side copies out of the persistently mapped readback buffer.
  // ------------------------------------------------------------------
  last_selected_extents.resize(num_selected_blobs);
  last_line_fit_points.resize(num_points);
  if (extents_bytes > 0) {
    readback_staging_.Read(last_selected_extents.data(), extents_bytes, 0);
  }
  if (linefit_bytes > 0) {
    readback_staging_.Read(last_line_fit_points.data(), linefit_bytes, linefit_offset);
  }
  const auto t_end = Clock::now();

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
  last_profile_.uf_iterations = uf_iterations;
  last_profile_.uf_converged = converged;
  last_profile_.submits = static_cast<uint32_t>(ctx_.submit_count - submits_at_start);

  if (timestamps_enabled_) {
    const std::vector<vk::QueryPool::Result> results = timestamp_pool_.ReadResults();
    const double ns_per_tick = ctx_.caps().timestamp_period_ns;
    for (uint32_t s = 0; s < kNumGpuStageSpans; ++s) {
      const vk::QueryPool::Result &start = results[SpanStart(static_cast<GpuStageSpan>(s))];
      const vk::QueryPool::Result &end = results[SpanEnd(static_cast<GpuStageSpan>(s))];
      // Both ends of a span are always written together (see the WriteTimestamp
      // call sites above), so either both are available or neither is - this
      // is a defensive check, not an expected partial-write case.
      if (start.available && end.available && end.ticks >= start.ticks) {
        last_profile_.gpu_stage_ms[s] = double(end.ticks - start.ticks) * ns_per_tick / 1.0e6;
      }
    }
    last_profile_.has_gpu_stage_breakdown = true;
  }

  // Quad fitting itself happens in QuadDecode (CPU tail); Detect() only runs
  // the GPU pipeline and exposes its outputs via last_selected_extents /
  // last_line_fit_points.
}

std::string GpuDetector::DescribeSizing() const {
  std::ostringstream os;
  os << "apriltag_vulkan: " << config_.width << "x" << config_.height << " -> decimated "
     << decimated_width_ << "x" << decimated_height_ << ", boundary point capacity "
     << qbp_capacity_;
  if (config_.max_boundary_points > 0 && qbp_capacity_ < dense_qbp_count_) {
    os << " (capped from dense " << dense_qbp_count_ << ")";
  }
  os << ", device memory " << (device_bytes_ / (1024 * 1024)) << " MiB";
  os << ", gray upload " << (gray_direct_write_ ? "direct (unified memory)" : "staged");
  return os.str();
}

}  // namespace apriltag_vulkan
