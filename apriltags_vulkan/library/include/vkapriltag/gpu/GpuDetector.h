#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <algorithm>
#include <chrono>
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
  static constexpr size_t kNumGpuStages = 12;
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

    // Fused path only: host time spent recording, and whether the frame only replayed buffers.
    double record_ms = 0.0;
    bool replayed = false;
  };

  // Names for DetectProfile::gpu_stage_ms, in index order (one timestamp pair per span).
  static constexpr std::array<const char *, kNumGpuStages> kGpuStageNames = {
      "clear",          // per-frame buffer fills + gray upload copy
      "threshold",      // decimate + block_minmax + block_filter + threshold
      "labelling",      // uf_init + uf_compress + the uf_merge/uf_compress loop
      "uf_final",       // uf_final.comp - per-blob pixel-count histogram
      "boundary",       // blob_diff (blob-size test, append + compaction)
      "hash_group",     // hash_group.comp (also assigns dense raw blob ids)
      "extents",        // reduce_extents_hash.comp
      "select",         // select_blobs.comp
      "blob_scan",      // scan_blob_counts.comp
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
      false,  // uf_final -> boundary (submit 2)
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
  // As above, and when the frame is staged (discrete GPUs) the upload copy runs on `transfer_lane`
  // (a transfer-only queue) and the first compute submission waits for it.
  GpuDetector(vk::Context &ctx, vk::Lane &lane, vk::Lane *transfer_lane,
              const DetectorConfig &config);
  ~GpuDetector();

  // Runs the GPU pipeline on one grayscale frame; results land in last_selected_extents and
  // last_line_fit_points.
  void Detect(const uint8_t *gray_frame);

  // Registers an external frame buffer so Detect(key) reads it in place, without the upload copy.
  // The buffer holds the packed 8-bit frame, `bytes` >= width * height, and must outlive the
  // detector. Host pointer: `ptr` and `bytes` aligned to caps().min_host_pointer_alignment (the
  // pointer is also the key). dma-buf: `key` is any unique pointer the caller later passes to
  // Detect; the descriptor is duplicated.
  void ImportHostFrame(const uint8_t *ptr, size_t bytes);
  // With `yuyv`, the buffer is a packed YUYV frame (2 bytes per pixel, no row padding) and only its
  // luma is read.
  void ImportDmaBufFrame(const uint8_t *key, int dmabuf_fd, size_t bytes, bool yuyv = false);

  const DetectorConfig &config() const { return config_; }
  const DetectProfile &last_profile() const { return last_profile_; }

  // Total device memory allocated for the pipeline's buffers, and a one-line sizing summary.
  uint64_t device_bytes() const { return device_bytes_; }
  std::string DescribeSizing() const;

 private:
  void CreateBuffers();
  void CreatePipelines();


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
  vk::Lane *transfer_lane_ = nullptr;
  VkSemaphore upload_semaphore_ = VK_NULL_HANDLE;
  VkSemaphore pending_wait_ = VK_NULL_HANDLE;
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
  // True when the threshold and sort stages read gray_buf_ as the decimated image.
  bool alias_decimated_ = false;
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
  // Blobs of more than local_sort_split_ points and their indirect dispatch args (see
  // sort_points_local_body.glsl), and the packed keys between merge rounds.
  vk::Buffer large_sort_buf_, sort_scratch_buf_;
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
  // True when a second uf_merge checks that the labelling converged (APRILTAG_VK_UF_VERIFY=1); one
  // merge pass joins every component, so by default the frame trusts it.
  bool uf_verify_ = false;
  // True when reduce_extents_hash_atomic64.comp is the reduction shader; select_blobs.comp then
  // unbiases gx_sum/gy_sum.
  bool extents_atomic64_ = false;
  // Backs last_line_fit_points only on the staging path.
  std::vector<RawLineFitPoint> linefit_scratch_;

  // Scan chain for the per-blob point-offset assignment (sized to config_.max_blobs).
  // Largest per-blob point count sort_points_local.comp sorts in shared memory (power of two).
  // local_sort_cap_ is the workgroup thread count; the virtual cap may exceed it (threads stride).
  uint32_t local_sort_cap_ = 0;
  uint32_t local_sort_virtual_cap_ = 0;
  // Blobs above this many points are sorted in runs of this length by several workgroups and
  // merged by whole-dispatch rounds; equal to local_sort_virtual_cap_ when that is not needed.
  uint32_t local_sort_split_ = 0;

  // --- Pipelines ---
  vk::ComputePipeline decimate_pl_;
  struct ImportedFrame {
    const uint8_t *key = nullptr;
    vk::Buffer buffer;
    vk::ComputePipeline decimate;
  };
  std::vector<ImportedFrame> imported_frames_;
  void AddImportedFrame(const uint8_t *key, vk::Buffer buffer, bool yuyv);
  vk::ComputePipeline block_minmax_pl_;
  vk::ComputePipeline block_filter_pl_;
  vk::ComputePipeline threshold_pl_;
  vk::ComputePipeline uf_init_pl_, uf_merge_pl_, uf_compress_pl_, uf_final_pl_;
  // Run starts from a workgroup max-scan, so no compression precedes the merge (single-merge only).
  vk::ComputePipeline uf_init_scan_pl_;
  bool uf_init_scan_ = false;
  // Pixels of a row each uf_init thread initialises.
  uint32_t uf_init_run_ = 4;
  // Pixels of a row each threshold thread covers.
  uint32_t threshold_run_ = 4;
  // Also performs the atomic compaction of boundary points.
  // uf_compress plus uf_final's count, for the last real compression of the first attempt.
  vk::ComputePipeline uf_compress_count_pl_;
  vk::ComputePipeline blob_diff_pl_;
  vk::ComputePipeline build_qbp_args_pl_;
  vk::ComputePipeline select_blobs_pl_;
  // Builds indirect_args_buf_ from raw_blob_counter_buf_.
  vk::ComputePipeline build_indirect_args_pl_;

  vk::ComputePipeline hash_group_pl_, reduce_extents_hash_pl_, scatter_index_points_pl_;


  // Per-blob point base-offset assignment and the segmented local sort.
  vk::ComputePipeline scan_blob_counts_pl_;
  vk::ComputePipeline sort_points_local_pl_;
  vk::ComputePipeline sort_large_runs_pl_, sort_large_merge_pl_;

  DetectProfile last_profile_;
  uint64_t device_bytes_ = 0;

  // Seeds the first labelling chunk.
  uint32_t last_uf_iterations_ = 0;

  // --- GPU timestamp profiling (see kGpuStageNames) ---
  // Constructed only when timestamps are supported and APRILTAG_VK_TIMESTAMPS=1; otherwise
  // WriteTimestamp() is a no-op.
  vk::QueryPool timestamp_pool_;
  bool timestamps_enabled_ = false;
  // Tracing: the "GPU" track id, and the first submission time that anchors GPU timestamps.
  uint32_t gpu_track_ = 0;
  int64_t frame_first_submit_ns_ = 0;

  // --- Fused-path recording and pre-recording (on by default; APRILTAG_VK_PRERECORD=0 disables) ---
  // The Record* functions only record, so their buffers can be cached and replayed; the unfused
  // path always records.
  enum class RecordingKind : uint8_t {
    kFirstAttempt,   // head + speculative tail; keyed by (input buffer, labelling chunk)
    kContinuation,   // `n` more labelling iterations plus the convergence-flag copy
    kTailRerun,      // zero the tail's counters and rerun the tail non-speculatively
  };
  struct RecordedFrame {
    RecordingKind kind = RecordingKind::kFirstAttempt;
    const uint8_t *key = nullptr;  // imported frame's key, or null for the staged copy path
    uint32_t n = 0;                // chunk size (kFirstAttempt) or iteration count (kContinuation)
    VkCommandBuffer cmd = VK_NULL_HANDLE;
  };
  std::vector<RecordedFrame> recorded_frames_;
  bool prerecord_ = false;
  // Whether this frame recorded anything (false when every submission was a cached replay).
  bool recorded_this_frame_ = false;

  // A command buffer ready to submit, and whether it is a persistent (pre-recorded) one.
  struct Submission {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool persistent = false;
  };
  // Detect() is the shared prologue (upload) followed by one of the two flows below.
  struct FrameStart;
  void DetectFused(const FrameStart &frame, uint32_t chunk);
  void DetectUnfused(const uint8_t *gray_frame);

  // Pure recording. The head is shared by every flow: timestamp reset, upload copy, counter clears,
  // decimate and threshold, the first labelling chunk, and the convergence-flag copy.
  void RecordFrameHead(VkCommandBuffer cmd, const ImportedFrame *imported, uint32_t chunk);
  void RecordUfChunk(VkCommandBuffer cmd, uint32_t iterations, bool trailing_compress = true);
  // The fused tail: blob sizes through line-fit moments; `speculative` stops blob_diff acting on
  // unconverged labels.
  void RecordFusedTail(VkCommandBuffer cmd, bool speculative);
  void RecordFlatten(VkCommandBuffer cmd);
  // Records the per-blob sort and, when split, the run sort and merge rounds for large blobs.
  void RecordSort(VkCommandBuffer cmd, uint32_t num_selected_blobs, bool count_from_buffer);
  void RecordUfContinuation(VkCommandBuffer cmd, uint32_t iterations);
  void RecordTailRerun(VkCommandBuffer cmd);

  // Return the command buffer for each kind: replayed from the cache, or recorded now.
  Submission FirstAttempt(const ImportedFrame *imported, uint32_t chunk);
  Submission Continuation(uint32_t iterations);
  Submission TailRerun();
  VkCommandBuffer FindRecorded(RecordingKind kind, const uint8_t *key, uint32_t n) const;
  // Submits and waits; adds the host-side cost to last_profile_.
  void SubmitTimed(const Submission &s);

  // Everything after the last submission of a fused frame: counter reads, host views of the
  // readback buffers, the profile and the GPU-timestamp spans.
  void FinishFused(const FrameStart &frame, std::chrono::steady_clock::time_point t_submitted);
  enum GpuStageSpan {
    kSpanClear = 0,
    kSpanThreshold,
    kSpanLabelling,
    kSpanUfFinal,
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
