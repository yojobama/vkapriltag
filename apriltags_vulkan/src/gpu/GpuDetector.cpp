#include "gpu/GpuDetector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace apriltag_vulkan {

namespace {

std::string ShaderPath(const char *name) {
  return std::string(SHADER_DIR) + "/" + name + ".comp.spv";
}

uint32_t NextPow2(uint32_t v) {
  if (v < 1) return 1;
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

VkBufferUsageFlags kSsboUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

vk::Buffer MakeSsbo(vk::Context &ctx, VkDeviceSize bytes) {
  return vk::Buffer(ctx, std::max<VkDeviceSize>(bytes, 4), kSsboUsage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : ctx_(ctx), config_(config) {
  if (config_.width % 8 != 0 || config_.height % 8 != 0) {
    throw std::runtime_error("width and height must both be multiples of 8");
  }

  decimated_width_ = config_.width / 2;
  decimated_height_ = config_.height / 2;
  block_width_ = decimated_width_ / 4;
  block_height_ = decimated_height_ / 4;
  interior_width_ = decimated_width_ - 2;
  interior_height_ = decimated_height_ - 2;
  dense_qbp_count_ = 4u * interior_width_ * interior_height_;
  qbp_capacity_ = dense_qbp_count_;
  qbp_padded_n_ = NextPow2(std::max(qbp_capacity_, 1u));
  ipoint_capacity_ = qbp_capacity_;
  ipoint_padded_n_ = NextPow2(std::max(ipoint_capacity_, 1u));

  CreateBuffers();
  CreatePipelines();
}

void GpuDetector::CreateBuffers() {
  gray_buf_ = MakeSsbo(ctx_, VkDeviceSize(config_.width) * config_.height * 4);
  decimated_buf_ = MakeSsbo(ctx_, VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  minmax_unfiltered_buf_ = MakeSsbo(ctx_, VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = MakeSsbo(ctx_, VkDeviceSize(block_width_) * block_height_ * 4);
  thresholded_buf_ = MakeSsbo(ctx_, VkDeviceSize(decimated_width_) * decimated_height_ * 4);

  parent_buf_ = MakeSsbo(ctx_, VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = MakeSsbo(ctx_, VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  uf_changed_buf_ = MakeSsbo(ctx_, 4);

  qbp_dense_buf_ = MakeSsbo(ctx_, VkDeviceSize(dense_qbp_count_) * sizeof(QBPoint));
  qbp_compacted_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_capacity_) * sizeof(QBPoint));
  qbp_counter_buf_ = MakeSsbo(ctx_, 4);
  qbp_keys_hi_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_padded_n_) * 4);
  qbp_keys_lo_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_padded_n_) * 4);
  qbp_payload_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_padded_n_) * 4);
  qbp_sorted_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_capacity_) * sizeof(QBPoint));

  heads_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_capacity_) * 4);
  raw_blob_index_buf_ = MakeSsbo(ctx_, VkDeviceSize(qbp_capacity_) * 4);

  extents_buf_ = MakeSsbo(ctx_, VkDeviceSize(config_.max_raw_blobs) * sizeof(MinMaxExtentsGpu));
  selected_extents_buf_ =
      MakeSsbo(ctx_, VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu));
  selected_counter_buf_ = MakeSsbo(ctx_, 4);
  remap_buf_ = MakeSsbo(ctx_, VkDeviceSize(config_.max_raw_blobs) * 4);

  index_points_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  index_points_counter_buf_ = MakeSsbo(ctx_, 4);
  ipoint_keys_hi_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_padded_n_) * 4);
  ipoint_keys_lo_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_padded_n_) * 4);
  ipoint_payload_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_padded_n_) * 4);
  index_points_sorted_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));

  line_fit_points_buf_ = MakeSsbo(ctx_, VkDeviceSize(ipoint_capacity_) * sizeof(RawLineFitPoint));

  qbp_scan_chain_ = BuildScanChain(qbp_capacity_);
}

GpuDetector::ScanChain GpuDetector::BuildScanChain(uint32_t capacity) {
  ScanChain chain;
  uint32_t size = capacity;
  while (size > 1024) {
    uint32_t next = (size + 1023) / 1024;
    chain.level_capacities.push_back(next);
    chain.level_buffers.push_back(MakeSsbo(ctx_, VkDeviceSize(std::max(next, 1u)) * 4));
    size = next;
  }
  return chain;
}

void GpuDetector::CreatePipelines() {
  decimate_pl_ = vk::ComputePipeline(ctx_, ShaderPath("decimate"),
                                     {gray_buf_.get(), decimated_buf_.get()}, 8);
  block_minmax_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_minmax"), {decimated_buf_.get(), minmax_unfiltered_buf_.get()}, 8);
  block_filter_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_filter"), {minmax_unfiltered_buf_.get(), minmax_filtered_buf_.get()},
      8);
  threshold_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("threshold"),
      {decimated_buf_.get(), minmax_filtered_buf_.get(), thresholded_buf_.get()}, 12);

  uf_init_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_init"), {parent_buf_.get()}, 8);
  uf_merge_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_merge"),
      {parent_buf_.get(), thresholded_buf_.get(), uf_changed_buf_.get()}, 8);
  uf_compress_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_compress"), {parent_buf_.get()}, 8);
  uf_final_pl_ = vk::ComputePipeline(ctx_, ShaderPath("uf_final"),
                                     {parent_buf_.get(), blob_size_buf_.get()}, 8);

  blob_diff_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("blob_diff"),
      {thresholded_buf_.get(), parent_buf_.get(), blob_size_buf_.get(), qbp_dense_buf_.get()}, 12);

  fill_max_key_qbp_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("fill_max_key"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()}, 4);
  compact_qbp_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("compact_qbp"),
      {qbp_dense_buf_.get(), qbp_compacted_buf_.get(), qbp_counter_buf_.get(),
       qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()},
      8);
  qbp_bitonic_sort_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("bitonic_sort"),
      {qbp_keys_hi_buf_.get(), qbp_keys_lo_buf_.get(), qbp_payload_buf_.get()}, 12);
  qbp_gather_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("gather_generic"),
      {qbp_compacted_buf_.get(), qbp_sorted_buf_.get(), qbp_payload_buf_.get()}, 12);

  mark_heads_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("mark_heads"), {qbp_sorted_buf_.get(), heads_buf_.get()}, 4);

  // Multi-level scan pipeline chain for heads -> raw_blob_index.
  {
    VkBuffer prev_values = heads_buf_.get();
    VkBuffer prev_output = raw_blob_index_buf_.get();
    for (size_t i = 0; i < qbp_scan_chain_.level_buffers.size(); ++i) {
      VkBuffer block_sums = qbp_scan_chain_.level_buffers[i].get();
      qbp_scan_block_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_block"), {prev_values, prev_output, block_sums}, 4));
      prev_values = block_sums;
      prev_output = block_sums;
    }
    // Final (top) level: scan in place with no further block_sums consumer
    // needed downstream, but scan_block still requires a block_sums binding;
    // reuse the same buffer as a scratch dump (unused beyond this point).
    qbp_scan_block_pls_.push_back(
        vk::ComputePipeline(ctx_, ShaderPath("scan_block"), {prev_values, prev_output, prev_output}, 4));

    // scan_add_offsets pipelines, one per level from the top back down to
    // level 0 (raw_blob_index_buf_ is level 0's output).
    std::vector<VkBuffer> level_values = {raw_blob_index_buf_.get()};
    for (auto &buf : qbp_scan_chain_.level_buffers) level_values.push_back(buf.get());
    for (size_t i = 0; i + 1 < level_values.size(); ++i) {
      qbp_scan_add_offsets_pls_.push_back(vk::ComputePipeline(
          ctx_, ShaderPath("scan_add_offsets"), {level_values[i], level_values[i + 1]}, 4));
    }
  }

  init_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("init_extents"), {extents_buf_.get()}, 4);
  reduce_extents_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("reduce_extents"),
      {qbp_sorted_buf_.get(), heads_buf_.get(), raw_blob_index_buf_.get(), extents_buf_.get()}, 8);

  select_blobs_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("select_blobs"),
      {extents_buf_.get(), selected_extents_buf_.get(), selected_counter_buf_.get(),
       remap_buf_.get()},
      28);

  fill_max_key_ipoint_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("fill_max_key"),
      {ipoint_keys_hi_buf_.get(), ipoint_keys_lo_buf_.get(), ipoint_payload_buf_.get()}, 4);
  rewrite_index_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("rewrite_index_points"),
      {qbp_sorted_buf_.get(), raw_blob_index_buf_.get(), remap_buf_.get(),
       selected_extents_buf_.get(), index_points_buf_.get(), index_points_counter_buf_.get(),
       ipoint_keys_hi_buf_.get(), ipoint_keys_lo_buf_.get(), ipoint_payload_buf_.get()},
      12);
  ipoint_bitonic_sort_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("bitonic_sort"),
      {ipoint_keys_hi_buf_.get(), ipoint_keys_lo_buf_.get(), ipoint_payload_buf_.get()}, 12);
  ipoint_gather_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("gather_generic"),
      {index_points_buf_.get(), index_points_sorted_buf_.get(), ipoint_payload_buf_.get()}, 12);

  compute_line_fit_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("compute_line_fit_points"),
      {index_points_sorted_buf_.get(), decimated_buf_.get(), line_fit_points_buf_.get()}, 12);
}

void GpuDetector::RunBitonicSort(VkCommandBuffer cmd, const vk::ComputePipeline &sort_pipeline,
                                 uint32_t padded_n) {
  for (uint32_t k = 2; k <= padded_n; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      struct { uint32_t n, j, k; } pc{padded_n, j, k};
      sort_pipeline.Dispatch1D(cmd, padded_n, 256, &pc);
    }
  }
}

void GpuDetector::RunInclusiveScan(VkCommandBuffer cmd, VkBuffer /*values*/, VkBuffer /*output*/,
                                   uint32_t count, const ScanChain &chain,
                                   const std::vector<vk::ComputePipeline> &scan_block_pipelines,
                                   const std::vector<vk::ComputePipeline> &scan_add_offsets_pipelines) {
  // Down pass: scan level 0 (the real values), then each block-sums level.
  std::vector<uint32_t> counts = {count};
  for (uint32_t cap : chain.level_capacities) {
    counts.push_back((counts.back() + 1023) / 1024);
    (void)cap;
  }

  for (size_t i = 0; i < scan_block_pipelines.size(); ++i) {
    uint32_t c = (i < counts.size()) ? counts[i] : 1;
    struct { uint32_t count; } pc{c};
    scan_block_pipelines[i].Dispatch1D(cmd, std::max(c, 1u), 1024, &pc);
  }

  // Up pass: propagate offsets from the topmost level back down to level 0.
  for (size_t i = scan_add_offsets_pipelines.size(); i-- > 0;) {
    uint32_t c = counts[i];
    struct { uint32_t count; } pc{c};
    scan_add_offsets_pipelines[i].Dispatch1D(cmd, std::max(c, 1u), 1024, &pc);
  }
}

void GpuDetector::Detect(const uint8_t *gray_frame) {
  // Upload the grayscale frame, expanding to uint32 per pixel (avoids
  // needing the optional 8-bit-storage-buffer device feature).
  {
    std::vector<uint32_t> expanded(VkDeviceSize(config_.width) * config_.height);
    for (size_t i = 0; i < expanded.size(); ++i) expanded[i] = gray_frame[i];
    gray_buf_.Upload(ctx_, expanded.data(), expanded.size() * 4);
  }

  VkCommandBuffer cmd = ctx_.BeginOneShotCommands();

  // Zero the small atomic counters and per-frame accumulators.
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  index_points_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd);

  // 1. Grayscale decimation + adaptive threshold.
  struct { uint32_t w, h; } dims_pc{config_.width, config_.height};
  decimate_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &dims_pc);

  struct { uint32_t dw, dh; } bwbh_pc{decimated_width_, decimated_height_};
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, 16, 16, &bwbh_pc);

  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, 16, 16, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff; } thresh_pc{decimated_width_, decimated_height_,
                                                  config_.min_white_black_diff};
  threshold_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &thresh_pc);

  // 2. Connected component labeling (union-find), iterated to convergence.
  uf_init_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &bwbh_pc);
  for (int iter = 0; iter < 64; ++iter) {
    uf_changed_buf_.FillZero(cmd);
    vk::ComputePipeline::Barrier(cmd);
    uf_merge_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &bwbh_pc);
    uf_compress_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &bwbh_pc);
  }
  uf_final_pl_.Dispatch1D(cmd, decimated_width_ * decimated_height_, 256, &bwbh_pc);

  // 3. Boundary point extraction.
  struct { uint32_t w, h, min_blob; } blobdiff_pc{decimated_width_, decimated_height_,
                                                  config_.min_cluster_pixels};
  blob_diff_pl_.Dispatch1D(cmd, interior_width_ * interior_height_, 256, &blobdiff_pc);

  // 4. Compact valid boundary points + sort by (rep0, rep1).
  struct { uint32_t capacity; } fillcap_pc{qbp_padded_n_};
  fill_max_key_qbp_pl_.Dispatch1D(cmd, qbp_padded_n_, 256, &fillcap_pc);

  struct { uint32_t total, capacity; } compact_pc{dense_qbp_count_, qbp_capacity_};
  compact_qbp_pl_.Dispatch1D(cmd, dense_qbp_count_, 256, &compact_pc);

  RunBitonicSort(cmd, qbp_bitonic_sort_pl_, qbp_padded_n_);

  // Zero the destination first: gather_generic.comp intentionally skips
  // writing for sorted positions beyond the real compacted point count
  // (they hold sentinel payload values), so without this the tail would
  // otherwise retain a *previous frame's* stale QBPoint data, which could
  // corrupt blob extents downstream. A zeroed QBPoint has valid == 0, which
  // mark_heads.comp / reduce_extents.comp explicitly skip.
  qbp_sorted_buf_.FillZero(cmd);
  vk::ComputePipeline::Barrier(cmd);

  struct { uint32_t count, stride, src_count; } gather_pc{qbp_capacity_, sizeof(QBPoint) / 4,
                                                          qbp_capacity_};
  qbp_gather_pl_.Dispatch1D(cmd, qbp_capacity_, 256, &gather_pc);

  // 5. Mark blob boundaries + scan to assign compact raw blob indices.
  struct { uint32_t count; } count_pc{qbp_capacity_};
  mark_heads_pl_.Dispatch1D(cmd, qbp_capacity_, 256, &count_pc);
  RunInclusiveScan(cmd, heads_buf_.get(), raw_blob_index_buf_.get(), qbp_capacity_,
                   qbp_scan_chain_, qbp_scan_block_pls_, qbp_scan_add_offsets_pls_);

  // 6. Reduce into per-raw-blob extents, then select plausible tag quads.
  struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
  init_extents_pl_.Dispatch1D(cmd, config_.max_raw_blobs, 256, &extentscap_pc);

  struct { uint32_t count, max_raw_blobs; } reduce_pc{qbp_capacity_, config_.max_raw_blobs};
  reduce_extents_pl_.Dispatch1D(cmd, qbp_capacity_, 256, &reduce_pc);

  struct {
    uint32_t max_raw_blobs, max_blobs, tag_width, min_cluster, max_cluster, reversed, normal;
  } select_pc{config_.max_raw_blobs,      config_.max_blobs,
             config_.tag_width,          config_.min_cluster_pixels,
             config_.max_cluster_pixels, config_.reversed_border ? 1u : 0u,
             config_.normal_border ? 1u : 0u};
  select_blobs_pl_.Dispatch1D(cmd, config_.max_raw_blobs, 256, &select_pc);

  // 7. Rewrite surviving points to (compact blob index, angle) form, sort by
  // (blob_index, angle) so each blob's points are ordered around its
  // perimeter (needed for windowed line fitting on the CPU side).
  fill_max_key_ipoint_pl_.Dispatch1D(cmd, ipoint_padded_n_, 256, &fillcap_pc);

  struct { uint32_t count, capacity, max_raw_blobs; } rewrite_pc{qbp_capacity_, ipoint_capacity_,
                                                                 config_.max_raw_blobs};
  rewrite_index_points_pl_.Dispatch1D(cmd, qbp_capacity_, 256, &rewrite_pc);

  RunBitonicSort(cmd, ipoint_bitonic_sort_pl_, ipoint_padded_n_);

  struct { uint32_t count, stride, src_count; } igather_pc{ipoint_capacity_, sizeof(IPoint) / 4,
                                                           ipoint_capacity_};
  ipoint_gather_pl_.Dispatch1D(cmd, ipoint_capacity_, 256, &igather_pc);

  // 8. Per-point raw line-fit moments (cumulative summation happens on the
  // CPU after readback - see QuadDecode).
  struct { uint32_t count; int dw, dh; } linefit_pc{
      ipoint_capacity_, static_cast<int>(decimated_width_), static_cast<int>(decimated_height_)};
  compute_line_fit_points_pl_.Dispatch1D(cmd, ipoint_capacity_, 256, &linefit_pc);

  ctx_.EndOneShotCommands(cmd);

  // --- Read back the small, CPU-tail-relevant results. ---
  uint32_t num_selected_blobs = 0;
  selected_counter_buf_.Download(ctx_, &num_selected_blobs, 4);
  num_selected_blobs = std::min(num_selected_blobs, config_.max_blobs);

  last_selected_extents.assign(num_selected_blobs, MinMaxExtentsGpu{});
  if (num_selected_blobs > 0) {
    selected_extents_buf_.Download(ctx_, last_selected_extents.data(),
                                   VkDeviceSize(num_selected_blobs) * sizeof(MinMaxExtentsGpu));
  }

  uint32_t num_points = 0;
  index_points_counter_buf_.Download(ctx_, &num_points, 4);
  num_points = std::min(num_points, ipoint_capacity_);

  last_line_fit_points.assign(num_points, RawLineFitPoint{});
  if (num_points > 0) {
    line_fit_points_buf_.Download(ctx_, last_line_fit_points.data(),
                                  VkDeviceSize(num_points) * sizeof(RawLineFitPoint));
  }

  // Quad fitting itself happens in QuadDecode (CPU tail); Detect() only runs
  // the GPU pipeline and exposes its outputs via last_selected_extents /
  // last_line_fit_points.
}

}  // namespace apriltag_vulkan
