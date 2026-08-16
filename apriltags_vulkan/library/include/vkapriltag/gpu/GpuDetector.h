#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vkapriltag/gpu/Types.h"
#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"

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

  // Upper bound on boundary/index points kept per frame.
  //
  // 0 means "size for the dense worst case", i.e. 4 points per interior
  // decimated pixel, which is what the CUDA implementation allocates and
  // therefore what reproduces its behaviour exactly. That is also ~400 MB of
  // device memory at 1080p, which is fine on a discrete card but painful on a
  // unified-memory mobile part such as Mali, where a few hundred thousand is
  // both plenty for real scenes and far cheaper. Points beyond the cap are
  // dropped by the compaction shaders, so lowering it trades worst-case
  // fidelity for memory.
  uint32_t max_boundary_points = 0;

  // Connected-component labelling is iterated to convergence rather than a
  // fixed count. These bound and batch that loop: iterations are issued
  // `uf_iterations_per_chunk` at a time, and the convergence flag is only read
  // back between chunks, so a typical frame costs one or two extra round
  // trips instead of one per iteration.
  uint32_t max_uf_iterations = 64;
  // 2 rather than a larger guess because that is what real frames measure:
  // uf_compress does FULL path compression every iteration, so a single merge
  // pass plus its compression already resolves these blob shapes, and the
  // second pass exists only to observe that nothing changed. Verified at
  // 1080p: boundary points, blob count, point count and candidate quads are
  // all bit-identical for 2 through 64 iterations. Scenes needing more simply
  // cost an extra chunk on their first frame, after which the adaptive seed
  // remembers the higher count.
  // Env override: APRILTAG_VK_UF_CHUNK=<n>
  uint32_t uf_iterations_per_chunk = 2;

  // Degree of parallelism for the CPU tail (QuadDecode), counting the calling
  // thread. 0 = std::thread::hardware_concurrency(), 1 = fully serial.
  // Env override: APRILTAG_CPU_THREADS=<n>
  uint32_t cpu_threads = 0;
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
//
// Work sizing: every stage after boundary-point compaction is dispatched over
// the number of points the frame actually produced, not the worst-case
// capacity. The counts only exist on the device, so they are read back at two
// points in the frame; that costs a couple of extra queue submissions and
// saves one to two orders of magnitude of dispatched work, exactly as the CUDA
// original does when it copies its DeviceSelect::If count to the host before
// calling DeviceRadixSort.
class GpuDetector {
 public:
  struct DetectProfile {
    // Wall-clock, host side.
    double upload_ms = 0.0;
    double threshold_label_ms = 0.0;  // submits up to and including labelling
    double boundary_ms = 0.0;         // blob_diff + compaction
    double sort_group_ms = 0.0;       // qbp sort, grouping, blob selection
    double linefit_ms = 0.0;          // ipoint sort + line fit moments
    double readback_ms = 0.0;         // payload readback + host copies
    double gpu_ms = 0.0;              // sum of the GPU-work phases above
    double total_ms = 0.0;

    uint64_t upload_bytes = 0;
    uint64_t readback_bytes = 0;

    uint32_t selected_blobs = 0;
    uint32_t points = 0;

    // Work actually dispatched, which is the interesting part.
    uint32_t boundary_points = 0;     // compacted QBPoints this frame
    uint32_t raw_blobs = 0;           // distinct (rep0, rep1) pairs this frame
    uint32_t uf_iterations = 0;       // labelling passes until convergence
    uint32_t submits = 0;             // queue submissions this frame
    bool uf_converged = true;         // false if max_uf_iterations was hit
  };

  GpuDetector(vk::Context &ctx, const DetectorConfig &config);

  // Runs the full GPU pipeline on one grayscale frame (already extracted
  // from the camera's native format on the CPU). Results are exposed via
  // last_selected_extents / last_line_fit_points for QuadDecode to consume.
  void Detect(const uint8_t *gray_frame);

  const DetectorConfig &config() const { return config_; }
  const DetectProfile &last_profile() const { return last_profile_; }

  // Total device memory allocated for the pipeline's buffers, and a one-line
  // human-readable summary of the sizing. Worth logging on memory-constrained
  // parts.
  uint64_t device_bytes() const { return device_bytes_; }
  std::string DescribeSizing() const;

 private:
  void CreateBuffers();
  void CreatePipelines();

  // A chain of scratch buffers implementing a multi-level block scan over
  // an array of up to `capacity` uint32 values, fanning out by scan_wg_ per
  // level.
  struct ScanChain {
    std::vector<vk::Buffer> level_buffers;   // block-sum scratch, one per extra level
    std::vector<uint32_t> level_capacities;  // capacity of each extra level
  };
  ScanChain BuildScanChain(uint32_t capacity);

  // Runs an inclusive scan of `count` values in place, touching only the
  // levels that `count` actually requires.
  void RunInclusiveScan(VkCommandBuffer cmd, uint32_t count, const ScanChain &chain,
                        const std::vector<vk::ComputePipeline> &scan_block_pipelines,
                        const std::vector<vk::ComputePipeline> &scan_add_offsets_pipelines);

  // Copies a device counter into the shared counter staging buffer and reads
  // it back after the submission retires.
  void RecordCounterCopy(VkCommandBuffer cmd, const vk::Buffer &counter, uint32_t slot);
  uint32_t ReadCounterSlot(uint32_t slot) const;

  // Grows the readback staging buffer if needed (never shrinks, so steady
  // state performs no allocation).
  void EnsureReadbackCapacity(VkDeviceSize bytes);

  vk::Context &ctx_;
  DetectorConfig config_;

  // Launch geometry, taken from the device's limits.
  vk::WorkgroupSize wg1d_;
  vk::WorkgroupSize wg2d_;
  uint32_t scan_wg_ = 128;
  uint32_t decimated_width_ = 0;
  uint32_t decimated_height_ = 0;
  uint32_t block_width_ = 0;
  uint32_t block_height_ = 0;
  uint32_t interior_width_ = 0;   // decimated_width - 2
  uint32_t interior_height_ = 0;  // decimated_height - 2
  uint32_t dense_qbp_count_ = 0;  // 4 * interior_width * interior_height
  uint32_t qbp_capacity_ = 0;     // upper bound after compaction
  uint32_t ipoint_capacity_ = 0;  // upper bound on selected points

  uint32_t gray_words_ = 0;  // packed 4 pixels per uint32

  // --- Buffers (device-local, long-lived, sized once at construction) ---
  vk::Buffer gray_buf_, decimated_buf_;
  vk::Buffer minmax_unfiltered_buf_, minmax_filtered_buf_;
  vk::Buffer thresholded_buf_;
  vk::Buffer parent_buf_, blob_size_buf_, uf_changed_buf_;
  vk::Buffer qbp_compacted_buf_, qbp_counter_buf_;
  vk::Buffer qbp_keys_hi_buf_, qbp_keys_lo_buf_;
  vk::Buffer extents_buf_;
  vk::Buffer selected_extents_buf_, selected_counter_buf_, remap_buf_;
  // Per-pixel "1 + union-find root, or 0 if the blob is too small" (see
  // label_pixels.comp). Sized to `pixels`.
  vk::Buffer pixel_label_buf_;
  vk::Buffer index_points_buf_, index_points_counter_buf_;
  vk::Buffer index_points_sorted_buf_;
  // Inclusive scan of each selected blob's point count (see
  // extract_blob_counts.comp), sized to config_.max_blobs. Gives
  // rewrite_index_points.comp / sort_points_local.comp each blob's base
  // offset into index_points_buf_/index_points_sorted_buf_.
  vk::Buffer blob_point_offsets_buf_;
  vk::Buffer line_fit_points_buf_;

  // --- Hash grouping (replaces the global (rep0, rep1) sort) ---
  // hash_owner_buf_[slot] is 0 when free, else 1 + the index of the boundary
  // point that claimed the slot. point_slot_buf_[i] is point i's slot.
  // slot_dense_buf_ is mark_slots.comp's 0/1 flags, scanned in place into a
  // 1-based raw blob id. blob_cursor_buf_ is one output cursor per selected
  // blob for scatter_index_points.comp.
  vk::Buffer hash_owner_buf_, point_slot_buf_, slot_dense_buf_, blob_cursor_buf_;
  uint32_t hash_table_size_ = 0;

  // --- Host-visible staging, allocated once and permanently mapped ---
  // upload_staging_ is unused on unified-memory devices, where gray_buf_ is
  // itself host-visible and written directly.
  vk::Buffer upload_staging_;
  vk::Buffer counter_staging_;
  vk::Buffer readback_staging_;
  VkDeviceSize readback_capacity_ = 0;
  bool gray_direct_write_ = false;

  // Scan chains for the root-dense-id assignment (sized to `pixels`) and the
  // per-blob point-offset assignment (sized to config_.max_blobs).
  ScanChain slot_scan_chain_;
  ScanChain blob_scan_chain_;
  // Largest point count sort_points_local.comp can sort in shared memory for
  // one blob (a power of two, derived from device limits at construction).
  // The workgroup itself still only has local_sort_cap_ threads (bounded by
  // the device's max workgroup invocations); local_sort_virtual_cap_ can
  // exceed that because each thread handles multiple elements in a strided
  // pattern, so it's bounded only by the shared-memory budget (2 words per
  // element: theta key + local index).
  uint32_t local_sort_cap_ = 0;
  uint32_t local_sort_virtual_cap_ = 0;

  // --- Pipelines ---
  vk::ComputePipeline decimate_pl_;
  vk::ComputePipeline block_minmax_pl_;
  vk::ComputePipeline block_filter_pl_;
  vk::ComputePipeline threshold_pl_;
  vk::ComputePipeline uf_init_pl_, uf_merge_pl_, uf_compress_pl_, uf_final_pl_;
  // blob_diff now also performs the atomic compaction that compact_qbp.comp
  // used to do as a separate full-capacity pass.
  vk::ComputePipeline blob_diff_pl_;
  vk::ComputePipeline init_extents_pl_;
  vk::ComputePipeline label_pixels_pl_;
  vk::ComputePipeline select_blobs_pl_;

  vk::ComputePipeline compute_line_fit_points_pl_;
  vk::ComputePipeline hash_group_pl_, mark_slots_pl_, reduce_extents_hash_pl_,
      scatter_index_points_pl_;
  std::vector<vk::ComputePipeline> slot_scan_block_pls_;
  std::vector<vk::ComputePipeline> slot_scan_add_offsets_pls_;


  // Per-blob point base-offset assignment (extract_blob_counts.comp + an
  // inclusive scan) and the segmented local sort that replaces a flat
  // radix/bitonic sort of index points.
  vk::ComputePipeline extract_blob_counts_pl_;
  std::vector<vk::ComputePipeline> blob_scan_block_pls_;
  std::vector<vk::ComputePipeline> blob_scan_add_offsets_pls_;
  vk::ComputePipeline sort_points_local_pl_;

  DetectProfile last_profile_;
  uint64_t device_bytes_ = 0;

  // Seeds the first labelling chunk, so steady-state video converges in one
  // chunk plus one verification rather than rediscovering the count each frame.
  uint32_t last_uf_iterations_ = 0;

 public:
  // Readback results exposed for QuadDecode after Detect() runs the GPU
  // pipeline; sized to the actual (not capacity) counts for the frame.
  std::vector<MinMaxExtentsGpu> last_selected_extents;
  std::vector<RawLineFitPoint> last_line_fit_points;
};

}  // namespace apriltag_vulkan
