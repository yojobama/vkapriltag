#pragma once

#include <cstdint>

#include "vkapriltag/common/WorkerPool.h"

namespace apriltag_vulkan {

// Detector configuration.
struct DetectorConfig {
  uint32_t width = 0;
  uint32_t height = 0;

  // Integer downsampling factor (1 disables); samples one pixel per NxN block. width and height
  // must be divisible by it and at least 3 times it.
  uint32_t decimation = 2;

  uint32_t min_white_black_diff = 5;

  // Raises the block contrast floor above min_white_black_diff to this many times the frame's median
  // 4x4 block range (its sensor noise), so noise blocks are not labelled; 0 keeps the fixed floor.
  // Needs 8-bit storage (the fused threshold). Blocks with a range under twice the noise are dropped,
  // so a tag whose edge contrast is that low is lost.
  // Env override: APRILTAG_VK_ADAPTIVE_FLOOR=<scale>
  float adaptive_floor_scale = 2.0f;

  // Dispatches the labelling and boundary passes over only the 64 x 4 pixel cells that hold contrast,
  // so flat regions launch nothing. Used where a launch is costly (integrated GPUs, which also use
  // uf_count_runs) and with the fused threshold and subgroup arithmetic; elsewhere the dense passes run.
  // Same detections either way. Env override: APRILTAG_VK_SPARSE=0|1
  bool sparse_labelling = true;
  uint32_t min_cluster_pixels = 24;
  uint32_t max_cluster_pixels = 100000;
  uint32_t tag_width = 8;  // in "bit squares"; matches quick_decode tag_width usage
  bool reversed_border = false;
  bool normal_border = true;

  // Smallest tag side in full-resolution pixels; 0 disables. Otherwise min_cluster_pixels is
  // raised to the boundary-point count of a square tag of that size.
  // Env override: APRILTAG_VK_MIN_TAG_PX=<n>
  uint32_t min_tag_pixels = 0;

  // Small-tag rescue (decimation above 1): labels blobs down to 8 boundary points, hands those that
  // fit no quad to TagDecoder as boxes, and refits them from the full-resolution image.
  // Env override: APRILTAG_VK_SMALL_RESCUE=0|1
  bool small_tag_rescue = false;

  // Largest tag side in full-resolution pixels; 0 disables. Otherwise max_cluster_pixels is lowered
  // to twice the boundary-point count of a square tag of that size.
  uint32_t max_tag_pixels = 0;

  // Averages each decimation block instead of copying its top-left pixel (decimation above 1).
  // Env override: APRILTAG_VK_BOX_DECIMATE=0|1
  bool box_decimate = true;

  // Milliseconds between the one-thread dispatches that keep the GPU awake between frames while
  // frames are arriving; 0 disables, a negative value picks 6 on integrated GPUs and 0 elsewhere.
  // Env override: APRILTAG_VK_KEEPALIVE_MS=<ms>
  int keepalive_ms = -1;

  // Predictive pre-warm: learns the frame period from Detect() calls and submits a short burst of
  // trivial dispatches this many microseconds before the next frame is due, only while the cadence
  // is regular. 0 disables; a negative value picks 3000. Prewarm()
  // works whenever this is non-zero or keepalive_ms is positive.
  // Env override: APRILTAG_VK_PREWARM_LEAD_US=<us>
  int prewarm_lead_us = -1;
  // Dispatches per pre-warm burst and the pause between them in microseconds.
  // Env override: APRILTAG_VK_PREWARM_BURST=<n>, APRILTAG_VK_PREWARM_GAP_US=<us>
  uint32_t prewarm_burst = 2;
  uint32_t prewarm_gap_us = 1000;

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

  // Retries a blob whose quad fit failed on the convex hull of its boundary points, for a tag whose
  // border is partly covered. Env override: APRILTAG_VK_HULL_QUAD=0|1
  bool hull_quad = false;

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

  // CPU tail worker pool options (see WorkerPoolOptions). Env override: APRILTAG_CPU_BIG_CORES
  bool cpu_big_cores = true;

  WorkerPoolOptions PoolOptions() const {
    WorkerPoolOptions o;
    o.big_cores = cpu_big_cores;
    return o;
  }
};

}  // namespace apriltag_vulkan
