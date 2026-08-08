#include "vkapriltag/gpu/GpuDetector.h"

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

std::string ShaderPath(const char *name) {
  return std::string(SHADER_DIR) + "/" + name + ".comp.spv";
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

// Number of bits needed to represent every value in [0, max_value].
uint32_t BitsFor(uint32_t max_value) {
  uint32_t bits = 0;
  while (max_value > 0) {
    ++bits;
    max_value >>= 1;
  }
  return std::max(bits, 1u);
}

// 4-bit digits needed to cover `bits` bits.
uint32_t DigitsFor(uint32_t bits) { return (bits + 3u) / 4u; }

const VkBufferUsageFlags kSsboUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

// Offsets (in 4-byte slots) into the shared counter readback staging buffer.
constexpr uint32_t kSlotUfChanged = 0;
constexpr uint32_t kSlotQbpCount = 1;
constexpr uint32_t kSlotSelectedCount = 2;
constexpr uint32_t kSlotPointCount = 3;
constexpr uint32_t kSlotNumRoots = 4;
constexpr VkDeviceSize kCounterStagingBytes = 64;

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : ctx_(ctx), config_(config) {
  if (config_.width % 8 != 0 || config_.height % 8 != 0) {
    throw std::runtime_error("width and height must both be multiples of 8");
  }

  // --- Environment overrides. These must all be applied before any capacity
  // or launch geometry is derived from the config. ---

  // The boundary-point cap is the main device-memory lever on unified-memory
  // parts, where the dense worst-case sizing is expensive.
  if (const char *cap = std::getenv("APRILTAG_VK_MAX_POINTS")) {
    const long parsed = std::strtol(cap, nullptr, 10);
    if (parsed > 0) config_.max_boundary_points = static_cast<uint32_t>(parsed);
  }
  // A/B the sort strategy per device without a rebuild.
  if (const char *sort_mode = std::getenv("APRILTAG_VK_SORT")) {
    if (std::strcmp(sort_mode, "global") == 0) {
      config_.sort_algorithm = DetectorConfig::SortAlgorithm::GlobalBitonic;
    } else if (std::strcmp(sort_mode, "hybrid") == 0) {
      config_.sort_algorithm = DetectorConfig::SortAlgorithm::SharedMemoryBitonic;
    } else if (std::strcmp(sort_mode, "radix") == 0) {
      config_.sort_algorithm = DetectorConfig::SortAlgorithm::Radix;
    }
  }
  // Labelling chunk size. Smaller chunks detect convergence more precisely at
  // the cost of an extra round trip when they guess low.
  if (const char *chunk = std::getenv("APRILTAG_VK_UF_CHUNK")) {
    const long parsed = std::strtol(chunk, nullptr, 10);
    if (parsed > 0) config_.uf_iterations_per_chunk = static_cast<uint32_t>(parsed);
  }
  if (config_.uf_iterations_per_chunk == 0) config_.uf_iterations_per_chunk = 1;

  // Launch geometry comes from the device, never from a literal.
  const vk::DeviceCaps &caps = ctx_.caps();
  wg1d_ = vk::WorkgroupSize{caps.wg1d, 1, 1};
  wg2d_ = vk::WorkgroupSize{caps.wg2d_x, caps.wg2d_y, 1};
  scan_wg_ = caps.scan_wg;
  // Hard cap of 128: see radix_scatter.comp's byte-packed rank counters.
  radix_wg_ = vk::WorkgroupSize{std::min(caps.wg1d, 128u), 1, 1};

  decimated_width_ = config_.width / 2;
  decimated_height_ = config_.height / 2;
  block_width_ = decimated_width_ / 4;
  block_height_ = decimated_height_ / 4;
  interior_width_ = decimated_width_ - 2;
  interior_height_ = decimated_height_ - 2;
  dense_qbp_count_ = 4u * interior_width_ * interior_height_;

  qbp_capacity_ = dense_qbp_count_;
  if (config_.max_boundary_points > 0) {
    qbp_capacity_ = std::min(qbp_capacity_, config_.max_boundary_points);
  }
  qbp_padded_n_ = NextPow2(std::max(qbp_capacity_, 1u));
  ipoint_capacity_ = qbp_capacity_;

  // The grayscale frame is uploaded packed, four 8-bit pixels per uint32.
  // width % 8 == 0 guarantees the pixel count divides evenly by four.
  gray_words_ = (config_.width * config_.height) / 4u;

  // sort_points_local.comp handles one selected blob per workgroup, sorting
  // its points entirely in shared memory (2 words/point: key + local index).
  // local_sort_cap_ is the workgroup's thread count, bounded by the device's
  // max workgroup invocation count (a real dispatch limit); it no longer
  // needs to also bound the shared array size (see local_sort_virtual_cap_
  // below), but 1024 remains a sane ceiling matching radix_wg_/wg1d_'s
  // derivation from caps below.
  local_sort_cap_ = PrevPow2(std::max(caps.max_workgroup_invocations, 1u));
  local_sort_cap_ = std::min(local_sort_cap_, 1024u);

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
                                vk::MemoryKind::HostVisible);

  decimated_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  minmax_unfiltered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  thresholded_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);

  parent_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  uf_changed_buf_ = ssbo(4);
  root_dense_id_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);

  qbp_compacted_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * sizeof(QBPoint));
  qbp_counter_buf_ = ssbo(4);
  qbp_keys_hi_buf_ = ssbo(VkDeviceSize(qbp_padded_n_) * 4);
  qbp_keys_lo_buf_ = ssbo(VkDeviceSize(qbp_padded_n_) * 4);
  qbp_payload_buf_ = ssbo(VkDeviceSize(qbp_padded_n_) * 4);
  qbp_sorted_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * sizeof(QBPoint));

  heads_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);
  raw_blob_index_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 4);

  extents_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * sizeof(MinMaxExtentsGpu));
  selected_extents_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu));
  selected_counter_buf_ = ssbo(4);
  remap_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * 4);

  index_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  index_points_counter_buf_ = ssbo(4);
  index_points_sorted_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  blob_point_offsets_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * 4);

  line_fit_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(RawLineFitPoint));

  // Radix sort ping-pong scratch and histogram. Only the boundary-point sort
  // uses this any more (index points are sorted per-blob in shared memory by
  // sort_points_local.comp), so it is sized to qbp_capacity_ alone.
  const uint32_t sort_capacity = qbp_capacity_;
  radix_scratch_hi_ = ssbo(VkDeviceSize(sort_capacity) * 4);
  radix_scratch_lo_ = ssbo(VkDeviceSize(sort_capacity) * 4);
  radix_scratch_pay_ = ssbo(VkDeviceSize(sort_capacity) * 4);
  radix_max_blocks_ = (sort_capacity + radix_wg_.x - 1) / radix_wg_.x;
  radix_hist_ = ssbo(VkDeviceSize(16) * std::max(radix_max_blocks_, 1u) * 4);

  qbp_scan_chain_ = BuildScanChain(qbp_capacity_);
  root_scan_chain_ = BuildScanChain(decimated_width_ * decimated_height_);
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
                                 vk::MemoryKind::HostVisible);
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
      ctx_, ShaderPath("block_minmax"), {decimated_buf_.get(), minmax_unfiltered_buf_.get()}, 8,
      wg2d_);
  block_filter_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_filter"), {minmax_unfiltered_buf_.get(), minmax_filtered_buf_.get()},
      8, wg2d_);
  threshold_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("threshold"),
      {decimated_buf_.get(), minmax_filtered_buf_.get(), thresholded_buf_.get()}, 12, wg1d_);

  uf_init_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_init"), {parent_buf_.get()}, 8, wg1d_);
  uf_merge_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_merge"),
      {parent_buf_.get(), thresholded_buf_.get(), uf_changed_buf_.get()}, 8, wg1d_);
  uf_compress_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("uf_compress"), {parent_buf_.get()}, 8, wg1d_);
  uf_final_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_final"),
                                     {parent_buf_.get(), blob_size_buf_.get()}, 8, wg1d_);

  blob_diff_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("blob_diff"),
      {thresholded_buf_.get(), parent_buf_.get(), blob_size_buf_.get(), qbp_compacted_buf_.get(),
       qbp_counter_buf_.get(), qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(),
       qbp_payload_buf_.get(), root_dense_id_buf_.get()},
      16, wg1d_);

  fill_max_key_qbp_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("fill_max_key"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()}, 8, wg1d_);
  qbp_bitonic_sort_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("bitonic_sort"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()}, 12, wg1d_);
  qbp_bitonic_local_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("bitonic_local"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()}, 12, wg1d_);
  qbp_gather_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("gather_generic"),
      {qbp_compacted_buf_.get(), qbp_sorted_buf_.get(), qbp_payload_buf_.get()}, 12, wg1d_);

  mark_heads_pl_ = vk::ComputePipeline(ctx_, ShaderPath("mark_heads"),
                                       {qbp_sorted_buf_.get(), heads_buf_.get()}, 4, wg1d_);

  // Multi-level scan pipeline chain for heads -> raw_blob_index. These must
  // use the same workgroup size as the scan chain's fan-out divisor, since
  // scan_add_offsets.comp indexes block_offsets by gl_WorkGroupID.
  {
    const vk::WorkgroupSize scan_wg{scan_wg_, 1, 1};
    VkBuffer prev_values = heads_buf_.get();
    VkBuffer prev_output = raw_blob_index_buf_.get();
    for (size_t i = 0; i < qbp_scan_chain_.level_buffers.size(); ++i) {
      VkBuffer block_sums = qbp_scan_chain_.level_buffers[i].get();
      qbp_scan_block_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_block"), {prev_values, prev_output, block_sums}, 4, scan_wg));
      prev_values = block_sums;
      prev_output = block_sums;
    }
    // Final (top) level: scan in place. scan_block still requires a block_sums
    // binding, so reuse the same buffer as a scratch dump (unused downstream).
    qbp_scan_block_pls_.push_back(vk::ComputePipeline(
        ctx_, ShaderPath("scan_block"), {prev_values, prev_output, prev_output}, 4, scan_wg));

    // scan_add_offsets pipelines, one per level from the top back down to
    // level 0 (raw_blob_index_buf_ is level 0's output).
    std::vector<VkBuffer> level_values = {raw_blob_index_buf_.get()};
    for (auto &buf : qbp_scan_chain_.level_buffers) level_values.push_back(buf.get());
    for (size_t i = 0; i + 1 < level_values.size(); ++i) {
      qbp_scan_add_offsets_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_add_offsets"), {level_values[i], level_values[i + 1]}, 4,
          scan_wg));
    }
  }

  init_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("init_extents"), {extents_buf_.get()}, 4, wg1d_);
  reduce_extents_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("reduce_extents"),
      {qbp_sorted_buf_.get(), heads_buf_.get(), raw_blob_index_buf_.get(), extents_buf_.get()}, 8,
      wg1d_);

  select_blobs_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("select_blobs"),
      {extents_buf_.get(), selected_extents_buf_.get(), selected_counter_buf_.get(),
       remap_buf_.get()},
      28, wg1d_);

  // Dense blob id assignment: mark_roots.comp flags each union-find root,
  // then an inclusive scan (same chain machinery as qbp_scan_chain_ above,
  // just over `pixels` elements) turns the flags into a 1-based rank used by
  // blob_diff.comp to build a narrow boundary-point sort key.
  mark_roots_pl_ = vk::ComputePipeline(ctx_, ShaderPath("mark_roots"),
                                       {parent_buf_.get(), root_dense_id_buf_.get()}, 8, wg1d_);
  {
    const vk::WorkgroupSize scan_wg{scan_wg_, 1, 1};
    VkBuffer prev_values = root_dense_id_buf_.get();
    VkBuffer prev_output = root_dense_id_buf_.get();
    for (size_t i = 0; i < root_scan_chain_.level_buffers.size(); ++i) {
      VkBuffer block_sums = root_scan_chain_.level_buffers[i].get();
      root_scan_block_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_block"), {prev_values, prev_output, block_sums}, 4, scan_wg));
      prev_values = block_sums;
      prev_output = block_sums;
    }
    root_scan_block_pls_.push_back(vk::ComputePipeline(
        ctx_, ShaderPath("scan_block"), {prev_values, prev_output, prev_output}, 4, scan_wg));

    std::vector<VkBuffer> level_values = {root_dense_id_buf_.get()};
    for (auto &buf : root_scan_chain_.level_buffers) level_values.push_back(buf.get());
    for (size_t i = 0; i + 1 < level_values.size(); ++i) {
      root_scan_add_offsets_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_add_offsets"), {level_values[i], level_values[i + 1]}, 4,
          scan_wg));
    }
  }

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

  rewrite_index_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("rewrite_index_points"),
      {qbp_sorted_buf_.get(), raw_blob_index_buf_.get(), remap_buf_.get(),
       selected_extents_buf_.get(), index_points_buf_.get(), index_points_counter_buf_.get(),
       blob_point_offsets_buf_.get()},
      12, wg1d_);
  sort_points_local_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("sort_points_local"),
      {selected_extents_buf_.get(), blob_point_offsets_buf_.get(), index_points_buf_.get(),
       index_points_sorted_buf_.get()},
      4, vk::WorkgroupSize{local_sort_cap_, 1, 1}, {local_sort_virtual_cap_});

  compute_line_fit_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("compute_line_fit_points"),
      {index_points_sorted_buf_.get(), decimated_buf_.get(), line_fit_points_buf_.get()}, 12,
      wg1d_);

  // ---------------- Radix sort ----------------
  // Gather variants that read the payload out of the scratch set, for when an
  // odd number of digit passes leaves the result there.
  qbp_gather_scratch_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("gather_generic"),
      {qbp_compacted_buf_.get(), qbp_sorted_buf_.get(), radix_scratch_pay_.get()}, 12, wg1d_);

  radix_scan_hist_pl_ = vk::ComputePipeline(ctx_, ShaderPath("radix_scan_hist"),
                                            {radix_hist_.get()}, 4, wg1d_);

  const uint32_t kRadixPush = 16;  // n, shift, use_hi, num_blocks
  qbp_radix_hist_a_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("radix_histogram"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), radix_hist_.get()}, kRadixPush, radix_wg_);
  qbp_radix_hist_b_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("radix_histogram"),
      {radix_scratch_hi_.get(), radix_scratch_lo_.get(), radix_hist_.get()}, kRadixPush, radix_wg_);
  qbp_radix_scatter_ab_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("radix_scatter"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get(),
       radix_scratch_hi_.get(), radix_scratch_lo_.get(), radix_scratch_pay_.get(),
       radix_hist_.get()},
      kRadixPush, radix_wg_);
  qbp_radix_scatter_ba_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("radix_scatter"),
      {radix_scratch_hi_.get(), radix_scratch_lo_.get(), radix_scratch_pay_.get(),
       qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get(),
       radix_hist_.get()},
      kRadixPush, radix_wg_);

  // Digit counts, derived from a provable bound on each key word so no pass
  // is spent on bits that are always zero.
  //
  //   boundary points: keys are (rep0, rep1), both DENSE blob ids in
  //                    [0, num_roots) - see mark_roots.comp. num_roots is
  //                    data-dependent, so these are placeholders here;
  //                    Detect() overwrites both every frame once num_roots
  //                    for that frame is known.
  const uint32_t pixels = decimated_width_ * decimated_height_;
  const uint32_t rep_digits = DigitsFor(BitsFor(pixels > 0 ? pixels - 1 : 0));
  qbp_radix_.hist_from_a = &qbp_radix_hist_a_pl_;
  qbp_radix_.hist_from_b = &qbp_radix_hist_b_pl_;
  qbp_radix_.scatter_a_to_b = &qbp_radix_scatter_ab_pl_;
  qbp_radix_.scatter_b_to_a = &qbp_radix_scatter_ba_pl_;
  qbp_radix_.lo_digits = rep_digits;
  qbp_radix_.hi_digits = rep_digits;
}

bool GpuDetector::RunRadixSort(VkCommandBuffer cmd, const RadixPipelines &pipelines,
                               uint32_t count) {
  if (count == 0) return false;

  // Must match the histogram/scatter pipelines' own workgroup size, since
  // Dispatch1D derives its group count from that.
  const uint32_t num_blocks = (count + radix_wg_.x - 1) / radix_wg_.x;
  const uint32_t hist_len = 16u * num_blocks;

  // Least-significant digit first, low key word before high: that is exactly
  // ascending order on the 64-bit (hi, lo) composite, and every pass is stable
  // so the ordering established by earlier digits survives.
  bool in_scratch = false;
  const uint32_t passes = pipelines.passes();
  for (uint32_t pass = 0; pass < passes; ++pass) {
    const bool use_hi = pass >= pipelines.lo_digits;
    const uint32_t shift = (use_hi ? (pass - pipelines.lo_digits) : pass) * 4u;

    struct {
      uint32_t n, shift, use_hi, num_blocks;
    } pc{count, shift, use_hi ? 1u : 0u, num_blocks};

    // Histogram: every (bin, block) slot is overwritten, so no clear needed.
    (in_scratch ? *pipelines.hist_from_b : *pipelines.hist_from_a).Dispatch1D(cmd, count, &pc);

    struct {
      uint32_t total;
    } scan_pc{hist_len};
    radix_scan_hist_pl_.DispatchRaw(cmd, 1, 1, 1, &scan_pc);

    (in_scratch ? *pipelines.scatter_b_to_a : *pipelines.scatter_a_to_b)
        .Dispatch1D(cmd, count, &pc);
    in_scratch = !in_scratch;
  }
  return in_scratch;
}

void GpuDetector::RunBitonicSort(VkCommandBuffer cmd,
                                 const vk::ComputePipeline &local_pipeline,
                                 const vk::ComputePipeline &global_pipeline,
                                 uint32_t padded_n) {
  // A stage (k, j) pairs index i with i^j, so it stays inside one workgroup's
  // L contiguous elements whenever j < L and can run in shared memory.
  // L == 1 disables the shared-memory path entirely, degenerating to one
  // global dispatch per stage (the original scheme).
  const uint32_t L =
      (config_.sort_algorithm == DetectorConfig::SortAlgorithm::SharedMemoryBitonic)
          ? local_pipeline.workgroup_size().x
          : 1u;
  struct LocalPc { uint32_t n, k_start, k_end; };

  // Phase 1: every k <= L has all of its stages local (j <= k/2 < L), so this
  // single dispatch covers log2(L)*(log2(L)+1)/2 stages - 36 of them at L=256.
  if (L > 1) {
    LocalPc initial{padded_n, 2u, L};
    local_pipeline.Dispatch1D(cmd, padded_n, &initial);
  }

  // Phase 2: wider k. Stages down to j == L need global barriers; the
  // remaining j < L tail collapses into one more local dispatch.
  for (uint32_t k = (L > 1 ? L : 1u) << 1; k != 0 && k <= padded_n; k <<= 1) {
    for (uint32_t j = k >> 1; j >= L && j > 0; j >>= 1) {
      struct { uint32_t n, j, k; } pc{padded_n, j, k};
      global_pipeline.Dispatch1D(cmd, padded_n, &pc);
    }
    if (L > 1) {
      LocalPc tail{padded_n, k, k};
      local_pipeline.Dispatch1D(cmd, padded_n, &tail);
    }
  }
}

void GpuDetector::RunSizedSort(VkCommandBuffer cmd, const vk::ComputePipeline &fill_pipeline,
                               const vk::ComputePipeline &local_pipeline,
                               const vk::ComputePipeline &global_pipeline, uint32_t count,
                               uint32_t padded_n) {
  // Only the [count, padded_n) tail needs sentinel keys; blob_diff /
  // rewrite_index_points wrote real keys into [0, count).
  if (padded_n > count) {
    struct { uint32_t begin, end; } pc{count, padded_n};
    fill_pipeline.Dispatch1D(cmd, padded_n - count, &pc);
  }
  RunBitonicSort(cmd, local_pipeline, global_pipeline, padded_n);
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
  const bool use_radix = config_.sort_algorithm == DetectorConfig::SortAlgorithm::Radix;

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
  if (!gray_direct_write_) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  index_points_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  decimate_pl_.Dispatch1D(cmd, pixels, &dims_pc);
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, &dwdh_pc);
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff; } thresh_pc{decimated_width_, decimated_height_,
                                                  config_.min_white_black_diff};
  threshold_pl_.Dispatch1D(cmd, pixels, &thresh_pc);

  uf_init_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);

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
  uf_final_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);

  // Dense blob id assignment: mark_roots.comp flags each union-find root,
  // then an inclusive scan turns the 0/1 flags into a 1-based rank among all
  // roots in ascending pixel-index order. blob_diff.comp uses this to key the
  // boundary-point sort on the actual number of raw blobs instead of a full
  // pixel index.
  mark_roots_pl_.Dispatch1D(cmd, pixels, &dwdh_pc);
  RunInclusiveScan(cmd, pixels, root_scan_chain_, root_scan_block_pls_,
                   root_scan_add_offsets_pls_);

  // blob_diff appends valid boundary points (with their sort keys) directly
  // into the compacted buffer, so there is no dense intermediate array and no
  // separate full-capacity compaction pass.
  struct { uint32_t w, h, min_blob, capacity; } blobdiff_pc{
      decimated_width_, decimated_height_, config_.min_cluster_pixels, qbp_capacity_};
  blob_diff_pl_.Dispatch1D(cmd, interior_width_ * interior_height_, &blobdiff_pc,
                           BarrierKind::ComputeAndTransfer);

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);
  // The scan's last element is the total number of union-find roots this
  // frame - read back alongside qbp_count so the sort's key width can be
  // sized to the actual root count rather than a pixels-wide worst case.
  root_dense_id_buf_.RecordCopyTo(cmd, counter_staging_, 4,
                                  VkDeviceSize(pixels - 1) * 4, VkDeviceSize(kSlotNumRoots) * 4);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);

  const uint32_t qbp_count = std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const uint32_t qbp_padded = NextPow2(std::max(qbp_count, 1u));
  const uint32_t num_roots = ReadCounterSlot(kSlotNumRoots);
  const uint32_t rep_digits = DigitsFor(BitsFor(num_roots > 0 ? num_roots - 1 : 0));
  qbp_radix_.lo_digits = rep_digits;
  qbp_radix_.hi_digits = rep_digits;
  const auto t_boundary = Clock::now();

  // ------------------------------------------------------------------
  // Submit 3: sort the boundary points by (rep0, rep1), group them into raw
  // blobs, select plausible tag quads, and rewrite the survivors.
  //
  // Everything here is dispatched over qbp_count, not qbp_capacity_. At 1080p
  // that is the difference between a ~2M-element bitonic network (231 stages)
  // and one sized to the few hundred thousand points a real frame produces.
  // ------------------------------------------------------------------
  cmd = ctx_.BeginCommands();
  if (qbp_count > 0) {
    bool result_in_scratch = false;
    if (use_radix) {
      result_in_scratch = RunRadixSort(cmd, qbp_radix_, qbp_count);
    } else {
      RunSizedSort(cmd, fill_max_key_qbp_pl_, qbp_bitonic_local_pl_, qbp_bitonic_sort_pl_,
                   qbp_count, qbp_padded);
    }

    // No pre-clear of qbp_sorted_buf_ is needed any more: the gather covers
    // exactly [0, qbp_count), every payload there is a real index, and nothing
    // downstream reads past that point. This used to be a 66 MB fill per frame.
    struct { uint32_t count, stride, src_count; } gather_pc{qbp_count, sizeof(QBPoint) / 4,
                                                            qbp_count};
    (result_in_scratch ? qbp_gather_scratch_pl_ : qbp_gather_pl_)
        .Dispatch1D(cmd, qbp_count, &gather_pc);

    struct { uint32_t count; } count_pc{qbp_count};
    mark_heads_pl_.Dispatch1D(cmd, qbp_count, &count_pc);
    RunInclusiveScan(cmd, qbp_count, qbp_scan_chain_, qbp_scan_block_pls_,
                     qbp_scan_add_offsets_pls_);

    struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
    init_extents_pl_.Dispatch1D(cmd, config_.max_raw_blobs, &extentscap_pc);

    struct { uint32_t count, max_raw_blobs; } reduce_pc{qbp_count, config_.max_raw_blobs};
    reduce_extents_pl_.Dispatch1D(cmd, qbp_count, &reduce_pc);

    struct {
      uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal;
    } select_pc{config_.max_raw_blobs,      config_.max_blobs,
                config_.tag_width,          config_.min_cluster_pixels,
                config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
                config_.normal_border ? 1u : 0u};
    select_blobs_pl_.Dispatch1D(cmd, config_.max_raw_blobs, &select_pc);

    // Per-blob point base offsets: copy each selected blob's count into
    // blob_point_offsets_buf_ (reading num_selected straight off
    // selected_counter_buf_, since the host hasn't read it back yet this
    // submission) and inclusive-scan it, so rewrite_index_points.comp can
    // place every blob's points at a deterministic, contiguous offset.
    struct { uint32_t capacity; } extract_pc{config_.max_blobs};
    extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, &extract_pc);
    RunInclusiveScan(cmd, config_.max_blobs, blob_scan_chain_, blob_scan_block_pls_,
                     blob_scan_add_offsets_pls_);

    struct { uint32_t count, capacity, max_raw_blobs; } rewrite_pc{qbp_count, ipoint_capacity_,
                                                                   config_.max_raw_blobs};
    rewrite_index_points_pl_.Dispatch1D(cmd, qbp_count, &rewrite_pc);
  }
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);
  RecordCounterCopy(cmd, index_points_counter_buf_, kSlotPointCount);
  vk::ComputePipeline::HostReadBarrier(cmd);
  ctx_.SubmitAndWait(cmd);

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
    sort_points_local_pl_.DispatchRaw(cmd, num_selected_blobs, 1, 1, &sort_pc);

    struct { uint32_t count; int dw, dh; } linefit_pc{
        num_points, static_cast<int>(decimated_width_), static_cast<int>(decimated_height_)};
    compute_line_fit_points_pl_.Dispatch1D(cmd, num_points, &linefit_pc,
                                           BarrierKind::ComputeAndTransfer);
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
  last_profile_.qbp_sort_n = (qbp_count > 0) ? (use_radix ? qbp_count : qbp_padded) : 0;
  last_profile_.ipoint_sort_n = num_points;
  last_profile_.sort_passes = use_radix ? qbp_radix_.passes() : 0;
  last_profile_.uf_iterations = uf_iterations;
  last_profile_.uf_converged = converged;
  last_profile_.submits = static_cast<uint32_t>(ctx_.submit_count - submits_at_start);

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
