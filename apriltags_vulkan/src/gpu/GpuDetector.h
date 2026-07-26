#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/Types.h"
#include "vk/Buffer.h"
#include "vk/ComputePipeline.h"
#include "vk/Context.h"

namespace apriltag_vulkan {

// Configuration mirroring the relevant subset of apriltag_detector_t /
// tag_detector_ fields the CUDA GpuDetector reads (qtp.min_white_black_diff,
// min/max cluster pixels, border polarity, tag family width, line-fit
// thresholds).
struct DetectorConfig {
  uint32_t width = 0;
  uint32_t height = 0;

  uint32_t min_white_black_diff = 5;
  uint32_t min_cluster_pixels = 24;
  uint32_t max_cluster_pixels = 100000;
  uint32_t tag_width = 8;  // in "bit squares"; matches quick_decode tag_width usage
  bool reversed_border = false;
  bool normal_border = true;

  float max_line_fit_mse = 10.0f;
  double cos_critical_rad = 0.98;  // ~cos(11 degrees), matches typical apriltag default

  // Defensive caps (deliberately smaller than the CUDA implementation's
  // dense worst-case sizing, which would otherwise waste tens of MB with no
  // real-world benefit; both are generous for realistic scenes).
  uint32_t max_raw_blobs = 65536;
  uint32_t max_blobs = 2048;
};

// A single detected quad's corners, in full-resolution pixel coordinates,
// ready for CPU-side apriltag decoding.
struct DetectedQuad {
  double p[4][2];
};

// GPU-accelerated (Vulkan compute) re-implementation of
// frc971::apriltag::GpuDetector's image-processing pipeline: grayscale
// decimation, adaptive threshold, connected-component labeling, boundary
// point extraction, blob selection, and per-point line-fit moment
// computation. The remaining small-N combinatorial stages (peak finding,
// quad corner fitting, and final tag decode) run on the CPU - see
// QuadDecode.h - since they only ever touch a few thousand candidate values
// per frame and are much simpler to implement correctly as scalar C++ than
// as GPU compute shaders.
class GpuDetector {
 public:
  GpuDetector(vk::Context &ctx, const DetectorConfig &config);

  // Runs the full GPU pipeline on one grayscale frame (already extracted
  // from the camera's native format on the CPU). Results are exposed via
  // last_selected_extents / last_line_fit_points for QuadDecode to consume.
  void Detect(const uint8_t *gray_frame);

  const DetectorConfig &config() const { return config_; }

 private:
  void CreateBuffers();
  void CreatePipelines();

  // A chain of scratch buffers implementing a multi-level block scan over
  // an array of up to `capacity` uint32 values.
  struct ScanChain {
    std::vector<vk::Buffer> level_buffers;   // block-sum scratch, one per extra level
    std::vector<uint32_t> level_capacities;  // capacity of each extra level
  };
  ScanChain BuildScanChain(uint32_t capacity);
  // Runs an inclusive scan of `values` (length `count`, capacity must be <=
  // the capacity the ScanChain was built for) in place, using `output` as
  // the destination for the first-level scan result (may alias `values`).
  void RunInclusiveScan(VkCommandBuffer cmd, VkBuffer values, VkBuffer output, uint32_t count,
                       const ScanChain &chain,
                       const std::vector<vk::ComputePipeline> &scan_block_pipelines,
                       const std::vector<vk::ComputePipeline> &scan_add_offsets_pipelines);

  void RunBitonicSort(VkCommandBuffer cmd, const vk::ComputePipeline &sort_pipeline,
                     uint32_t padded_n);

  vk::Context &ctx_;
  DetectorConfig config_;

  uint32_t decimated_width_ = 0;
  uint32_t decimated_height_ = 0;
  uint32_t block_width_ = 0;
  uint32_t block_height_ = 0;
  uint32_t interior_width_ = 0;   // decimated_width - 2
  uint32_t interior_height_ = 0;  // decimated_height - 2
  uint32_t dense_qbp_count_ = 0;  // 4 * interior_width * interior_height
  uint32_t qbp_capacity_ = 0;     // == dense_qbp_count_, upper bound after compaction
  uint32_t qbp_padded_n_ = 0;     // next pow2 >= qbp_capacity_, for bitonic sort
  uint32_t ipoint_capacity_ = 0;  // == qbp_capacity_, upper bound on selected points
  uint32_t ipoint_padded_n_ = 0;  // next pow2 >= ipoint_capacity_

  // --- Buffers (device-local, long-lived, sized once at construction) ---
  vk::Buffer gray_buf_, decimated_buf_;
  vk::Buffer minmax_unfiltered_buf_, minmax_filtered_buf_;
  vk::Buffer thresholded_buf_;
  vk::Buffer parent_buf_, blob_size_buf_, uf_changed_buf_;
  vk::Buffer qbp_dense_buf_;
  vk::Buffer qbp_compacted_buf_, qbp_counter_buf_;
  vk::Buffer qbp_keys_hi_buf_, qbp_keys_lo_buf_, qbp_payload_buf_;
  vk::Buffer qbp_sorted_buf_;
  vk::Buffer heads_buf_, raw_blob_index_buf_;
  vk::Buffer extents_buf_;
  vk::Buffer selected_extents_buf_, selected_counter_buf_, remap_buf_;
  vk::Buffer index_points_buf_, index_points_counter_buf_;
  vk::Buffer ipoint_keys_hi_buf_, ipoint_keys_lo_buf_, ipoint_payload_buf_;
  vk::Buffer index_points_sorted_buf_;
  vk::Buffer line_fit_points_buf_;

  ScanChain qbp_scan_chain_;

  // --- Pipelines ---
  vk::ComputePipeline decimate_pl_;
  vk::ComputePipeline block_minmax_pl_;
  vk::ComputePipeline block_filter_pl_;
  vk::ComputePipeline threshold_pl_;
  vk::ComputePipeline uf_init_pl_, uf_merge_pl_, uf_compress_pl_, uf_final_pl_;
  vk::ComputePipeline blob_diff_pl_;
  vk::ComputePipeline fill_max_key_qbp_pl_, compact_qbp_pl_;
  vk::ComputePipeline qbp_bitonic_sort_pl_;
  vk::ComputePipeline qbp_gather_pl_;
  vk::ComputePipeline mark_heads_pl_;
  std::vector<vk::ComputePipeline> qbp_scan_block_pls_;
  std::vector<vk::ComputePipeline> qbp_scan_add_offsets_pls_;
  vk::ComputePipeline init_extents_pl_, reduce_extents_pl_;
  vk::ComputePipeline select_blobs_pl_;
  vk::ComputePipeline fill_max_key_ipoint_pl_, rewrite_index_points_pl_;
  vk::ComputePipeline ipoint_bitonic_sort_pl_;
  vk::ComputePipeline ipoint_gather_pl_;
  vk::ComputePipeline compute_line_fit_points_pl_;

 public:
  // Readback results exposed for QuadDecode after Detect() runs the GPU
  // pipeline; sized to the actual (not capacity) counts for the frame.
  std::vector<MinMaxExtentsGpu> last_selected_extents;
  std::vector<RawLineFitPoint> last_line_fit_points;
};

}  // namespace apriltag_vulkan
