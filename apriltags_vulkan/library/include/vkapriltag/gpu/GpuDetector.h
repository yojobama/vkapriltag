#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <span>
#include <vector>

#include "vkapriltag/gpu/Types.h"
#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"
#include "vkapriltag/vk/QueryPool.h"

namespace apriltag_vulkan {

// Detector configuration.
struct DetectorConfig {
  uint32_t width = 0;
  uint32_t height = 0;

  // Integer downsampling factor (1 disables); samples one pixel per NxN block. width and height
  // must be divisible by it.
  uint32_t decimation = 2;

  uint32_t min_white_black_diff = 5;
  uint32_t min_cluster_pixels = 24;
  uint32_t max_cluster_pixels = 100000;
  uint32_t tag_width = 8;  // in "bit squares"; matches quick_decode tag_width usage
  bool reversed_border = false;
  bool normal_border = true;

  // Smallest tag side in full-resolution pixels; 0 disables. Otherwise min_cluster_pixels is
  // raised to the boundary-point count of a square tag of that size.
  // Env override: APRILTAG_VK_MIN_TAG_PX=<n>
  uint32_t min_tag_pixels = 0;

  // Bounding-box aspect ratio cap (max(w,h)/min(w,h)) applied in select_blobs.comp; 0 disables.
  float aspect_max = 8.0f;

  // Bounds on boundary-point count / bounding-box perimeter 2*(w+h); 0/0 disables.
  float fill_min = 0.5f;
  float fill_max = 2.5f;

  float max_line_fit_mse = 10.0f;
  double cos_critical_rad = 0.98;  // ~cos(11 degrees), matches typical apriltag default

  // Corner seeding for the per-blob quad fit. kPeaks: windowed error, peak detection and
  // combinatorial search. kDp: seeds 4 corners geometrically, falling back to kPeaks per blob
  // when the segments fail the max_line_fit_mse gate.
  // Env override: APRILTAG_VK_QUADFIT=peaks|dp
  enum class QuadFitMethod { kPeaks, kDp };
  QuadFitMethod quad_fit_method = QuadFitMethod::kDp;

  // Caps on the raw (pre-selection) blob count.
  uint32_t max_raw_blobs = 65536;
  // Ceiling on blobs surviving selection. 0 sizes it from the frame (scaled with
  // width*height/decimation^2, clamped to max_raw_blobs). Too small a value makes detections
  // depend on GPU scheduling (DetectProfile::selected_blob_drops).
  uint32_t max_blobs = 0;

  // Upper bound on boundary points kept per frame. 0 sizes for the dense worst case (4 points
  // per interior decimated pixel); points beyond the cap are dropped by the compaction shaders.
  uint32_t max_boundary_points = 0;

  // Labelling iterates to convergence; the convergence flag is read back once per chunk.
  uint32_t max_uf_iterations = 64;
  // Iterations issued per chunk (a merge pass plus one to observe no change).
  // Env override: APRILTAG_VK_UF_CHUNK=<n>
  uint32_t uf_iterations_per_chunk = 2;

  // Degree of parallelism for the CPU tail (QuadDecode), counting the calling
  // thread. 0 = std::thread::hardware_concurrency(), 1 = fully serial.
  // Env override: APRILTAG_CPU_THREADS=<n>
  uint32_t cpu_threads = 0;
};

// A detected quad's corners in full-resolution pixel coordinates.
struct DetectedQuad {
  double p[4][2];
};

// Vulkan compute pipeline for decimation, thresholding, connected-component labelling,
// boundary extraction, blob selection and line-fit moments. Peak finding and quad fitting
// run on the CPU (see QuadDecode.h). Stages after boundary compaction are sized from
// device-side counts read back at two points in the frame.
class GpuDetector {
 public:
  // Number of named GPU timestamp spans; sizes gpu_stage_ms, gpu_gap_ms (one fewer),
  // kGpuStageNames and kGpuGapCrossesSubmit.
  static constexpr size_t kNumGpuStages = 13;
  static constexpr size_t kNumGpuGaps = kNumGpuStages - 1;

  struct DetectProfile {
    // Wall-clock, host side.
    // Which inter-span gaps crossed a queue submission this frame.
    std::array<bool, kNumGpuGaps> gap_crosses_submit = kGpuGapCrossesSubmit;
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

    // Work dispatched this frame.
    uint32_t boundary_points = 0;     // compacted QBPoints this frame
    uint32_t raw_blobs = 0;           // distinct (rep0, rep1) pairs this frame
    // Boundary points hash_group.comp could not place within max_probes; dropped from grouping.
    uint32_t hash_probe_drops = 0;
    // Blobs that passed select_blobs.comp but found no slot because more than max_blobs qualified.
    // Which are dropped depends on GPU scheduling, so detections are then not reproducible.
    uint32_t selected_blob_drops = 0;
    // Selected blobs with more points than local_sort_virtual_cap_; they stay unsorted, so their
    // quads are meaningless. Nonzero is normal for large background blobs; suspect it when an
    // expected tag is missing.
    uint32_t oversized_sort_blobs = 0;
    uint32_t uf_iterations = 0;       // labelling passes until convergence
    uint32_t submits = 0;             // queue submissions this frame
    bool uf_converged = true;         // false if max_uf_iterations was hit

    // Per-span GPU timing from vkCmdWriteTimestamp pairs; populated only when timestamps are
    // supported and enabled. gpu_stage_ms[i] is 0 for a span skipped this frame.
    bool has_gpu_stage_breakdown = false;
    std::array<double, kNumGpuStages> gpu_stage_ms = {};

    // GPU-clock gap between the end of span i and the start of span i+1. Gaps that cross a
    // submit boundary include submit/fence overhead.
    std::array<double, kNumGpuGaps> gpu_gap_ms = {};

    // Host-side cost of driving the GPU; cpu_submit_wait_ms includes the GPU execution it waits on.
    double cpu_begin_ms = 0.0;        // BeginCommands: ring fence wait + resets
    double cpu_submit_wait_ms = 0.0;  // EndCommandBuffer + QueueSubmit + WaitForFences
    double cpu_counter_read_ms = 0.0; // ReadCounterSlot invalidate + read
  };

  // Names for DetectProfile::gpu_stage_ms, in index order (one timestamp pair per span).
  static constexpr std::array<const char *, kNumGpuStages> kGpuStageNames = {
      "clear",          // per-frame buffer fills + gray upload copy
      "threshold",      // decimate + block_minmax + block_filter + threshold
      "labelling",      // uf_init + uf_compress + the uf_merge/uf_compress loop
      "uf_final",       // uf_final.comp - per-blob pixel-count histogram
      "label_pixels",   // label_pixels.comp - per-pixel blob identity and min-size test
      "boundary",       // blob_diff (append + compaction)
      "hash_group",     // hash_group.comp (also assigns dense raw blob ids)
      "extents",        // init_extents.comp + reduce_extents_hash.comp
      "select",         // select_blobs.comp
      "blob_scan",      // extract_blob_counts.comp + its scan chain
      "scatter",        // scatter_index_points.comp
      "sort",           // sort_points_local.comp (fused with line-fit moments)
      "readback_copy",  // device->staging copies of the extents and line-fit payloads
  };

  // Whether the gap after kGpuStageNames[g] crosses a queue submit in the four-submit layout.
  // DetectProfile::gap_crosses_submit holds the layout that actually ran.
  static constexpr std::array<bool, kNumGpuGaps> kGpuGapCrossesSubmit = {
      false,  // clear -> threshold (submit 1)
      false,  // threshold -> labelling (submit 1)
      true,   // labelling -> uf_final (submit 1 -> 2)
      false,  // uf_final -> label_pixels (submit 2)
      false,  // label_pixels -> boundary (submit 2)
      true,   // boundary -> hash_group (submit 2 -> 3)
      false,  // hash_group -> extents (submit 3)
      false,  // extents -> select (submit 3)
      false,  // select -> blob_scan (submit 3)
      false,  // blob_scan -> scatter (submit 3)
      true,   // scatter -> sort (submit 3 -> 4)
      false,  // sort -> readback_copy (submit 4)
  };

  GpuDetector(vk::Context &ctx, const DetectorConfig &config);
  // As above, submitting through `lane`, which must outlive the detector.
  GpuDetector(vk::Context &ctx, vk::Lane &lane, const DetectorConfig &config);

  // Runs the GPU pipeline on one grayscale frame; results land in last_selected_extents and
  // last_line_fit_points.
  void Detect(const uint8_t *gray_frame);

  const DetectorConfig &config() const { return config_; }
  const DetectProfile &last_profile() const { return last_profile_; }

  // Total device memory allocated for the pipeline's buffers, and a one-line sizing summary.
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
  // Non-const: adds its cost to last_profile_.
  uint32_t ReadCounterSlot(uint32_t slot);

  // Wrappers for lane_.BeginCommands()/SubmitAndWait() that add host-side cost to last_profile_.
  VkCommandBuffer BeginTimedCommands();
  void SubmitTimedAndWait(VkCommandBuffer cmd);

  // Grows the readback staging buffer if needed (never shrinks, so steady
  // state performs no allocation).
  void EnsureReadbackCapacity(VkDeviceSize bytes);

  vk::Context &ctx_;
  vk::Lane &lane_;
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
  // Reused as the per-pixel label (1 + root, or 0 if too small) once labelling converges.
  vk::Buffer parent_buf_, blob_size_buf_, uf_changed_buf_;
  vk::Buffer qbp_compacted_buf_, qbp_counter_buf_;
  vk::Buffer qbp_keys_buf_;
  // Holds kExtentsCopies-1 extra copies of the first kPrivateExtentsBlobs entries behind the
  // canonical array (see common.glsl's ExtentsSlot).
  static constexpr uint32_t kExtentsCopies = 8;
  static constexpr uint32_t kPrivateExtentsBlobs = 4096;
  static uint32_t ExtentsSlotCount(uint32_t max_raw_blobs) {
    return max_raw_blobs +
           (kExtentsCopies - 1) * std::min(kPrivateExtentsBlobs, max_raw_blobs);
  }
  vk::Buffer extents_buf_;
  vk::Buffer selected_extents_buf_, selected_counter_buf_, remap_buf_;
  vk::Buffer index_points_buf_;
  // Inclusive scan of per-blob point counts (max_blobs entries): each blob's base offset into index_points_buf_.
  vk::Buffer blob_point_offsets_buf_;
  // Written by sort_points_local.comp (sort fused with line-fit moments).
  vk::Buffer line_fit_points_buf_;

  // --- Hash grouping (replaces the global (rep0, rep1) sort) ---
  // hash_owner_buf_[slot]: 0 if free, else 1 + index of the claiming point; point_slot_buf_[i]: point i's slot.
  // slot_dense_buf_[slot]: 1-based raw blob id; raw_blob_counter_buf_: raw blob count after hash_group.comp.
  // blob_cursor_buf_: per-selected-blob output cursor for scatter_index_points.comp.
  vk::Buffer hash_owner_buf_, point_slot_buf_, slot_dense_buf_, blob_cursor_buf_;
  vk::Buffer raw_blob_counter_buf_;
  // Points hash_group.comp couldn't place within max_probes - see
  // DetectProfile::hash_probe_drops.
  vk::Buffer hash_drop_counter_buf_;
  // Blobs sort_points_local.comp had to pass through unsorted - see
  // DetectProfile::oversized_sort_blobs.
  vk::Buffer oversized_sort_counter_buf_;
  uint32_t hash_table_size_ = 0;

  // Indirect dispatch args built on-device from raw_blob_counter_buf_.
  vk::Buffer indirect_args_buf_;

  // --- Host-visible staging, allocated once and permanently mapped ---
  // upload_staging_ is unused when gray_buf_ is host-visible and written directly.
  vk::Buffer upload_staging_;
  vk::Buffer counter_staging_;
  vk::Buffer readback_staging_;
  VkDeviceSize readback_capacity_ = 0;
  bool gray_direct_write_ = false;
  // True when line_fit_points_buf_ is host-visible and cached, so it is read in place.
  bool linefit_direct_read_ = false;
  bool extents_direct_read_ = false;
  // True when the last three submissions are recorded as one (requires both direct-read paths).
  bool fused_submits_ = false;
  // True when each labelling chunk's last uf_compress is predicated on uf_changed_buf_ with
  // VK_EXT_conditional_rendering; otherwise uf_compress.comp's honour_changed_flag guard applies.
  bool predicated_compress_ = false;
  // True when reduce_extents_hash_atomic64.comp is the reduction shader; select_blobs.comp then
  // unbiases gx_sum/gy_sum.
  bool extents_atomic64_ = false;
  // Backs last_line_fit_points only on the staging path.
  std::vector<RawLineFitPoint> linefit_scratch_;

  // Scan chain for the per-blob point-offset assignment (sized to config_.max_blobs).
  ScanChain blob_scan_chain_;
  // Largest per-blob point count sort_points_local.comp sorts in shared memory (power of two).
  // local_sort_cap_ is the workgroup thread count; the virtual cap may exceed it (threads stride).
  uint32_t local_sort_cap_ = 0;
  uint32_t local_sort_virtual_cap_ = 0;

  // --- Pipelines ---
  vk::ComputePipeline decimate_pl_;
  vk::ComputePipeline block_minmax_pl_;
  vk::ComputePipeline block_filter_pl_;
  vk::ComputePipeline threshold_pl_;
  vk::ComputePipeline uf_init_pl_, uf_merge_pl_, uf_compress_pl_, uf_final_pl_;
  // Also performs the atomic compaction of boundary points.
  vk::ComputePipeline blob_diff_pl_;
  vk::ComputePipeline init_extents_pl_;
  vk::ComputePipeline merge_extents_pl_;
  vk::ComputePipeline build_qbp_args_pl_, build_sort_args_pl_;
  vk::ComputePipeline label_pixels_pl_;
  vk::ComputePipeline select_blobs_pl_;
  // Builds indirect_args_buf_ from raw_blob_counter_buf_.
  vk::ComputePipeline build_indirect_args_pl_;

  vk::ComputePipeline hash_group_pl_, reduce_extents_hash_pl_, scatter_index_points_pl_;


  // Per-blob point base-offset assignment and the segmented local sort.
  vk::ComputePipeline extract_blob_counts_pl_;
  std::vector<vk::ComputePipeline> blob_scan_block_pls_;
  std::vector<vk::ComputePipeline> blob_scan_add_offsets_pls_;
  vk::ComputePipeline sort_points_local_pl_;

  DetectProfile last_profile_;
  uint64_t device_bytes_ = 0;

  // Seeds the first labelling chunk.
  uint32_t last_uf_iterations_ = 0;

  // --- GPU timestamp profiling (see kGpuStageNames) ---
  // Constructed only when timestamps are supported and APRILTAG_VK_TIMESTAMPS=1; otherwise
  // WriteTimestamp() is a no-op.
  vk::QueryPool timestamp_pool_;
  bool timestamps_enabled_ = false;
  enum GpuStageSpan {
    kSpanClear = 0,
    kSpanThreshold,
    kSpanLabelling,
    kSpanUfFinal,
    kSpanLabelPixels,
    kSpanBoundary,
    kSpanHashGroup,
    kSpanExtents,
    kSpanSelect,
    kSpanBlobScan,
    kSpanScatter,
    kSpanSort,
    kSpanReadbackCopy,
    kNumGpuStageSpans,
  };
  // kNumGpuStageSpans must equal kNumGpuStages.
  static_assert(static_cast<size_t>(kNumGpuStageSpans) == kNumGpuStages,
                "GpuStageSpan and kNumGpuStages disagree about the span count");

  // WriteTimestamp() index for a span's start/end - 2 slots per span.
  static constexpr uint32_t SpanStart(GpuStageSpan s) { return static_cast<uint32_t>(s) * 2; }
  static constexpr uint32_t SpanEnd(GpuStageSpan s) { return static_cast<uint32_t>(s) * 2 + 1; }

 public:
  // Readback results after Detect(), sized to the frame's actual counts.
  std::vector<MinMaxExtentsGpu> last_selected_extents;
  // Non-owning view, valid until the next Detect(). Points into device memory when it is
  // host-visible and cached, otherwise into an internal scratch vector.
  std::span<const RawLineFitPoint> last_line_fit_points;
};

}  // namespace apriltag_vulkan
