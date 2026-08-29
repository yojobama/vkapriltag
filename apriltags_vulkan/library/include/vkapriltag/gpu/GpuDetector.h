#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vkapriltag/gpu/Types.h"
#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/ComputePipeline.h"
#include "vkapriltag/vk/Context.h"
#include "vkapriltag/vk/QueryPool.h"

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

  // Smallest tag side the caller cares about, in FULL-RESOLUTION pixels. 0
  // (default) disables this and leaves min_cluster_pixels as the only floor.
  // When set, GpuDetector raises min_cluster_pixels to whatever a
  // min_tag_pixels-wide square tag's boundary-point count would be at this
  // pipeline's fixed 2x decimation (4 sides x min_tag_pixels/2 decimated
  // pixels each), so undersized noise/text/foliage blobs never reach the
  // expensive per-blob CPU quad fit. Declaring this is strictly a recall/
  // throughput trade: tags smaller than it are guaranteed to be dropped.
  // Env override: APRILTAG_VK_MIN_TAG_PX=<n>
  uint32_t min_tag_pixels = 0;

  // Bounding-box aspect ratio cap (max(w,h)/min(w,h)) applied in
  // select_blobs.comp. 0 disables the test. 8 clears views up to ~82 degrees
  // off-normal - beyond what libapriltag can reliably decode anyway - while
  // rejecting stringy non-quad blobs (text edges, foliage) that pass the raw
  // point-count test. Measured against the validation corpus: the real tag's
  // aspect ratio stayed within 1.0-1.02 at every scale tested, comfortably
  // inside this bound; background blobs from the same scene ranged 1-28.
  float aspect_max = 8.0f;

  // Fill-ratio bounds: a selected blob's boundary-point count divided by its
  // bounding box's own perimeter (2*(w+h)). A clean quadrilateral traces out
  // close to one boundary point per perimeter pixel, so this ratio sits near
  // 1.0; 0/0 disables the test. Measured against the validation corpus, the
  // real tag's fill ratio was 1.00-1.01 at every scale tested; background
  // blobs from the same scene spanned roughly 0.5-1.7 (1st-99th percentile).
  // The bounds here are deliberately wider than that observed spread - the
  // corpus is a single photographed scene, not a stress test for this
  // filter, so they're set to comfortably clear the real tag rather than to
  // aggressively trim this specific background.
  float fill_min = 0.5f;
  float fill_max = 2.5f;

  float max_line_fit_mse = 10.0f;
  double cos_critical_rad = 0.98;  // ~cos(11 degrees), matches typical apriltag default

  // Corner-seeding algorithm for QuadDecode's per-blob quad fit (CPU tail).
  // kPeaks (default) is the exact port of the original windowed-error +
  // 7-tap-filter + peak-detection + C(10,4) combinatorial search. kDp seeds
  // 4 corners geometrically instead - the two mutually-farthest boundary
  // points, plus the point of maximum perpendicular deviation on each
  // resulting arc - skipping the combinatorial search entirely, and falls
  // back to kPeaks per-blob whenever DP doesn't cleanly yield 4 points whose
  // segments pass the same max_line_fit_mse gate the combinatorial search
  // uses. See QuadDecode.cpp's FitQuadForBlob/TryDpQuad.
  // Env override: APRILTAG_VK_QUADFIT=peaks|dp
  enum class QuadFitMethod { kPeaks, kDp };
  QuadFitMethod quad_fit_method = QuadFitMethod::kPeaks;

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
    // Boundary points hash_group.comp couldn't place within max_probes probes
    // (see its comment) - i.e. dropped, not grouped into any blob. 0 for
    // every real scene at the current table sizing; watch this if
    // max_raw_blobs / the hash table sizing is ever tightened further.
    uint32_t hash_probe_drops = 0;
    uint32_t uf_iterations = 0;       // labelling passes until convergence
    uint32_t submits = 0;             // queue submissions this frame
    bool uf_converged = true;         // false if max_uf_iterations was hit

    // Per-shader-group GPU timing, from vkCmdWriteTimestamp pairs bracketing
    // each named span - finer than the four submission-level phases above.
    // Populated only when vk::DeviceCaps::timestamps_supported is true;
    // gpu_stage_ms[i] is 0 for a span this frame's control flow skipped
    // entirely (e.g. the boundary-point stages when qbp_count == 0). See
    // GpuDetector::kGpuStageNames for what each index means. This is
    // diagnostic only - printed by the validate tool behind
    // APRILTAG_VK_TIMESTAMPS=1 - and costs nothing when timestamps aren't
    // supported or the pool wasn't constructed.
    bool has_gpu_stage_breakdown = false;
    std::array<double, 12> gpu_stage_ms = {};

    // --- Host-side cost of driving the GPU, split out from the phase timers
    // above. The phase timers (threshold_label_ms etc.) are wall-clock and so
    // bundle four distinct things together: command recording, the queue
    // submit, the blocking fence wait (which spans the GPU's actual
    // execution), and the counter readbacks between submits. gpu_stage_ms
    // measures only the third of those, and only the parts inside a named
    // span - so `sum(gpu_stage_ms)` being well under `gpu_ms` says the
    // difference is host-side, but not which part. These three say which.
    //
    // cpu_submit_wait_ms in particular INCLUDES the GPU execution it waits
    // on, so the quantity that matters for "is the round trip itself
    // expensive?" is (cpu_submit_wait_ms - sum(gpu_stage_ms)): submit ioctl
    // plus fence-signal wakeup latency, paid once per submission.
    double cpu_begin_ms = 0.0;        // BeginCommands: ring fence wait + resets
    double cpu_submit_wait_ms = 0.0;  // EndCommandBuffer + QueueSubmit + WaitForFences
    double cpu_counter_read_ms = 0.0; // ReadCounterSlot invalidate + read
  };

  // Names for DetectProfile::gpu_stage_ms, in index order. Each entry is one
  // vkCmdWriteTimestamp pair (start, end) recorded around the named group of
  // dispatches - see the kSpan* constants and their use in Detect().
  static constexpr std::array<const char *, 12> kGpuStageNames = {
      "clear",          // the per-frame vkCmdFillBuffer set + the gray upload
                        // copy. ~2.3 MB of fills at 1080p, and outside every
                        // other span, so it was landing in the unattributed
                        // gap that the submit-count reduction failed to move.
      "threshold",      // decimate + block_minmax + block_filter + threshold
      "labelling",      // uf_init + uf_compress + the uf_merge/uf_compress loop
      "label_finalize", // uf_final + label_pixels
      "boundary",       // blob_diff (append + compaction)
      "hash_group",     // hash_group.comp (also assigns dense raw blob ids)
      "extents",        // init_extents.comp + reduce_extents_hash.comp
      "select",         // select_blobs.comp
      "blob_scan",      // extract_blob_counts.comp + its scan chain
      "scatter",        // scatter_index_points.comp
      "sort",           // sort_points_local.comp - fused with the line-fit
                        // moment computation, see its own comment
      "readback_copy",  // the device->staging vkCmdCopyBuffer pair for the
                        // extents + line-fit payloads. Inside submit 4 but
                        // not compute, so it would otherwise land in the
                        // unattributed gap alongside genuine round-trip cost.
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
  // Non-const because it accumulates its own cost into last_profile_ (the
  // counter readback is a real per-submission cost on a device where the
  // staging buffer is non-coherent and needs an invalidate - see
  // MemoryKind::HostVisibleCached).
  uint32_t ReadCounterSlot(uint32_t slot);

  // ctx_.BeginCommands() / ctx_.SubmitAndWait() with the host-side cost
  // accumulated into last_profile_ (see DetectProfile::cpu_*_ms). Detect()
  // uses these rather than calling the Context methods directly, so the
  // per-submission overhead is attributed rather than silently folded into
  // whichever wall-clock phase happened to contain it.
  VkCommandBuffer BeginTimedCommands();
  void SubmitTimedAndWait(VkCommandBuffer cmd);

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
  // parent_buf_ is repurposed after labelling converges: label_pixels.comp
  // overwrites each entry in place with "1 + union-find root, or 0 if the
  // blob is too small" (see that shader's comment), so blob_diff.comp reads
  // it as a per-pixel label rather than a raw union-find parent.
  vk::Buffer parent_buf_, blob_size_buf_, uf_changed_buf_;
  vk::Buffer qbp_compacted_buf_, qbp_counter_buf_;
  vk::Buffer qbp_keys_hi_buf_, qbp_keys_lo_buf_;
  vk::Buffer extents_buf_;
  vk::Buffer selected_extents_buf_, selected_counter_buf_, remap_buf_;
  vk::Buffer index_points_buf_;
  // Inclusive scan of each selected blob's point count (see
  // extract_blob_counts.comp), sized to config_.max_blobs. Gives
  // rewrite_index_points.comp / sort_points_local.comp each blob's base
  // offset into index_points_buf_.
  vk::Buffer blob_point_offsets_buf_;
  // Written directly by sort_points_local.comp, fused with the angular
  // sort - see that shader's comment. No intermediate sorted-IPoint buffer.
  vk::Buffer line_fit_points_buf_;

  // --- Hash grouping (replaces the global (rep0, rep1) sort) ---
  // hash_owner_buf_[slot] is 0 when free, else 1 + the index of the boundary
  // point that claimed the slot. point_slot_buf_[i] is point i's slot.
  // slot_dense_buf_[slot] is a 1-based raw blob id, assigned directly by
  // hash_group.comp's winning atomicCompSwap thread (see its comment) - no
  // separate mark+scan pass. raw_blob_counter_buf_ is that assignment's
  // shared atomic counter; its value after hash_group.comp runs is the
  // frame's raw blob count. blob_cursor_buf_ is one output cursor per
  // selected blob for scatter_index_points.comp.
  vk::Buffer hash_owner_buf_, point_slot_buf_, slot_dense_buf_, blob_cursor_buf_;
  vk::Buffer raw_blob_counter_buf_;
  // Points hash_group.comp couldn't place within max_probes - see
  // DetectProfile::hash_probe_drops.
  vk::Buffer hash_drop_counter_buf_;
  uint32_t hash_table_size_ = 0;

  // VkDispatchIndirectCommand built on-device from raw_blob_counter_buf_ by
  // build_indirect_args_pl_, so init_extents_pl_ / select_blobs_pl_ dispatch
  // over the frame's actual raw blob count instead of max_raw_blobs.
  vk::Buffer indirect_args_buf_;

  // --- Host-visible staging, allocated once and permanently mapped ---
  // upload_staging_ is unused on unified-memory devices, where gray_buf_ is
  // itself host-visible and written directly.
  vk::Buffer upload_staging_;
  vk::Buffer counter_staging_;
  vk::Buffer readback_staging_;
  VkDeviceSize readback_capacity_ = 0;
  bool gray_direct_write_ = false;

  // Scan chain for the per-blob point-offset assignment (sized to
  // config_.max_blobs). The hash table's raw-blob numbering no longer needs
  // one - see raw_blob_counter_buf_.
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
  // Builds indirect_args_buf_ from raw_blob_counter_buf_ - see that buffer's
  // comment.
  vk::ComputePipeline build_indirect_args_pl_;

  vk::ComputePipeline hash_group_pl_, reduce_extents_hash_pl_, scatter_index_points_pl_;


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

  // --- GPU timestamp profiling (see kGpuStageNames) ---
  // Two timestamps (start, end) per named span; kGpuStageNames.size() spans.
  // Constructed only when caps().timestamps_supported and
  // APRILTAG_VK_TIMESTAMPS=1 are both set, so a normal run allocates nothing
  // and every WriteTimestamp() call below is a no-op (QueryPool::valid() ==
  // false).
  vk::QueryPool timestamp_pool_;
  bool timestamps_enabled_ = false;
  enum GpuStageSpan {
    kSpanClear = 0,
    kSpanThreshold,
    kSpanLabelling,
    kSpanLabelFinalize,
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
  // WriteTimestamp() index for a span's start/end - 2 slots per span.
  static constexpr uint32_t SpanStart(GpuStageSpan s) { return static_cast<uint32_t>(s) * 2; }
  static constexpr uint32_t SpanEnd(GpuStageSpan s) { return static_cast<uint32_t>(s) * 2 + 1; }

 public:
  // Readback results exposed for QuadDecode after Detect() runs the GPU
  // pipeline; sized to the actual (not capacity) counts for the frame.
  std::vector<MinMaxExtentsGpu> last_selected_extents;
  std::vector<RawLineFitPoint> last_line_fit_points;
};

}  // namespace apriltag_vulkan
