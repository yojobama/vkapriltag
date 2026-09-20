#include "vkapriltag/gpu/GpuDetector.h"

#include "vkapriltag/vk/EmbeddedShaders.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

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
constexpr uint32_t kSlotOversizedSortBlobs = 6;
constexpr VkDeviceSize kCounterStagingBytes = 64;

}  // namespace

GpuDetector::GpuDetector(vk::Context &ctx, const DetectorConfig &config)
    : ctx_(ctx), config_(config) {
  if (config_.decimation == 0) {
    throw std::runtime_error("decimation must be >= 1");
  }
  if (config_.width % config_.decimation != 0 || config_.height % config_.decimation != 0) {
    throw std::runtime_error("width and height must each be evenly divisible by decimation");
  }
  // QBPoint/IPoint pack a boundary coordinate into 14 bits per axis (see
  // common.glsl's PackXY). That coordinate is the *doubled* decimated index
  // (x2 = 2*decimated_x +/- 1, so boundaries land on half-integers), whose
  // maximum is 2*width/decimation - not width. The two coincide only at
  // decimation 2, which is why a bare `width <= 16383` test was right while
  // decimation was hardcoded and is off by 2x at decimation 1. Checked
  // against the real packed range so a lossy configuration is rejected up
  // front instead of silently truncating coordinates.
  const uint32_t max_packed_x = 2u * (config_.width / config_.decimation);
  const uint32_t max_packed_y = 2u * (config_.height / config_.decimation);
  if (max_packed_x > 16383 || max_packed_y > 16383) {
    throw std::runtime_error(
        "2*(width/decimation) and 2*(height/decimation) must each be <= 16383");
  }
  // label_pixels.comp packs `1 + root` into the low 30 bits of parent[] and
  // the pixel's threshold code into the top 2 (see common.glsl). The check
  // above already implies this - it bounds the decimated grid at 8191x8191 =
  // 67,092,481 pixels, under 2^26 - so this is executable documentation of
  // the coupling rather than a reachable failure, and it will fire first if
  // the packing above is ever widened.
  if (VkDeviceSize(config_.width / config_.decimation) *
          (config_.height / config_.decimation) >=
      (1u << 30)) {
    throw std::runtime_error(
        "decimated pixel count must be < 2^30 to fit label_pixels.comp's packed label");
  }
  // RawLineFitPoint packs blob_index into 22 bits and W into 10 (see
  // common.glsl). blob_index is bounded by max_blobs, itself clamped to
  // max_raw_blobs below, and W is bounded at 361 by construction - so
  // neither can overflow unless max_raw_blobs is widened past 2^22. Checked
  // here rather than left implicit, because the failure mode would be a
  // silently mis-grouped point rather than a crash.
  static_assert(sizeof(RawLineFitPoint) == 8, "RawLineFitPoint packing changed");
  if (config_.max_raw_blobs > (1u << 22)) {
    throw std::runtime_error(
        "max_raw_blobs must be <= 2^22 to fit RawLineFitPoint's packed blob_index");
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
    // full-resolution side is min_tag_pixels, at this detector's decimation:
    // 4 sides x (min_tag_pixels / decimation) decimated pixels each.
    const uint32_t derived_min_cluster = 4u * (config_.min_tag_pixels / config_.decimation);
    config_.min_cluster_pixels = std::max(config_.min_cluster_pixels, derived_min_cluster);
  }

  // Launch geometry comes from the device, never from a literal.
  const vk::DeviceCaps &caps = ctx_.caps();
  wg1d_ = vk::WorkgroupSize{caps.wg1d, 1, 1};
  wg2d_ = vk::WorkgroupSize{caps.wg2d_x, caps.wg2d_y, 1};
  scan_wg_ = caps.scan_wg;


  decimated_width_ = config_.width / config_.decimation;
  decimated_height_ = config_.height / config_.decimation;

  // Resolve max_blobs == 0 ("auto") to a ceiling that tracks the frame,
  // because the number of blobs a frame produces does. It goes as the
  // DECIMATED pixel count - area/decimation^2 - so a flat ceiling silently
  // loses its margin as either the sensor grows or decimation is lowered,
  // and overflowing it is not a graceful degradation: select_blobs.comp
  // discards the excess in atomicAdd order, so the frame's detections stop
  // being reproducible (see DetectProfile::selected_blob_drops).
  //
  // Anchored on 1080p at decimation 2 - the configuration the old flat 2048
  // was chosen and validated against - so every frame at or below that size
  // resolves to exactly 2048 and keeps today's measured cost bit for bit.
  // Only larger sensors or finer decimation ask for more.
  //
  // The headroom this leaves is deliberate and cheap. Blob density is
  // scene-dependent (texture produces blobs; a flat wall does not), so the
  // linear-in-pixels estimate has to sit well above the typical case for the
  // atypical one to still fit - measured, a 12 MP frame at decimation 2
  // qualifies 1878 blobs against the 12042 this resolves to. The cost of
  // that slack is one dispatch and one scan over the capacity
  // (extract_blob_counts + RunInclusiveScan; the per-frame READBACK is sized
  // by the actual count, not this), measured at +2.2 us for a 24x capacity
  // increase - 0.06% of a 3.6 ms frame. Silent irreproducibility is not
  // worth trading for microseconds.
  //
  // Clamped to max_raw_blobs: a selected blob is always a subset of the raw
  // blobs, so capacity beyond that can never be reached.
  if (config_.max_blobs == 0) {
    constexpr uint64_t kAnchorDecimatedPx = 1920ull * 1080ull / 4ull;  // 1080p at decimation 2
    constexpr uint64_t kAnchorMaxBlobs = 2048ull;
    const uint64_t decimated_px =
        uint64_t(decimated_width_) * uint64_t(decimated_height_);
    const uint64_t scaled = kAnchorMaxBlobs * decimated_px / kAnchorDecimatedPx;
    config_.max_blobs = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(scaled, kAnchorMaxBlobs), config_.max_raw_blobs));
  }
  // Rounded up rather than floored: width/height are only guaranteed
  // divisible by decimation, not by a further factor of 4, so
  // decimated_width_/height_ need not be a multiple of 4.
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
  // its points entirely in shared memory (1 word/point: key and local index
  // share a word, packed key-high/index-low - see that shader's comment).
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
  // still get properly angle-sorted instead of falling back to the unsorted
  // identity copy in sort_points_local_body.glsl - a fallback that yields
  // geometrically meaningless quads, since FitQuadForBlob consumes these
  // points as an ordered walk around the perimeter.
  //
  // The ceiling has to scale with decimation. A blob's perimeter in
  // decimated points goes as 1/decimation, so the ~1200-point border of a
  // 1080p tag at decimation 2 is ~2400 points at decimation 1. 2048 slots
  // (the old fixed ceiling) covered the former and silently mis-sorted the
  // latter. Anchor on the value validated at decimation 2 and scale it, but
  // never request less than that anchor: coarser decimation keeps exactly
  // today's geometry - and therefore today's measured cost - while only
  // decimation 1 pays for the larger shared array.
  const uint32_t kCapAtDecimation2 = 2048u;
  const uint32_t scaled_cap = (kCapAtDecimation2 * 2u) / config_.decimation;
  // Hard-capped at 4096 regardless of the shared-memory budget:
  // sort_points_local.comp packs the local sort index into the low 12 bits
  // of its shared word (kLocalIndexBits), so 4096 slots is the most the
  // packing itself can address, not merely a budget choice - this must stay
  // in sync with that shader's kLocalIndexBits. On every real device
  // (Vulkan guarantees >= 16 KiB of shared memory, this device has 32 KiB)
  // the /4u budget bound is already looser than the 4096 index cap.
  local_sort_virtual_cap_ =
      std::min(std::max(scaled_cap, kCapAtDecimation2), 4096u);
  local_sort_virtual_cap_ = std::min(
      local_sort_virtual_cap_, PrevPow2(std::max(caps.max_shared_memory_bytes / 4u, 1u)));

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

  // 1 byte/pixel on devices with storageBuffer8BitAccess (see
  // decimate_u8.comp's comment), else the 4-byte-per-pixel fallback layout
  // every shader in the base corpus assumes. Shared by decimated_buf_ and
  // thresholded_buf_: threshold.comp/threshold_u8.comp reads the former and
  // writes the latter in the same dispatch, so the two switch together, not
  // independently.
  const VkDeviceSize decimated_bytes_per_pixel = ctx_.caps().has_8bit_storage ? 1 : 4;
  decimated_buf_ =
      ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel);
  minmax_unfiltered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  minmax_filtered_buf_ = ssbo(VkDeviceSize(block_width_) * block_height_ * 4);
  thresholded_buf_ =
      ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * decimated_bytes_per_pixel);

  parent_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  blob_size_buf_ = ssbo(VkDeviceSize(decimated_width_) * decimated_height_ * 4);
  uf_changed_buf_ = ssbo(4);

  // QBPoint is packed into one uint32 on the GPU side (see common.glsl); no
  // C++ mirror struct exists since nothing on the host ever reads one back.
  qbp_compacted_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * sizeof(uint32_t));
  qbp_counter_buf_ = ssbo(4);
  // Only the grouping hash reads this now, so it is sized to the actual point
  // capacity rather than the power of two a bitonic network needed. One
  // interleaved uvec2 per point rather than two parallel uint arrays: the
  // hash's probe loop compares a whole (rep0, rep1) key, so split arrays cost
  // two random gathers into two separate buffers per probe. Same total bytes.
  qbp_keys_buf_ = ssbo(VkDeviceSize(qbp_capacity_) * 8);

  extents_buf_ = ssbo(VkDeviceSize(ExtentsSlotCount(config_.max_raw_blobs)) *
                      sizeof(MinMaxExtentsGpu));
  // Allocated readback-capable for the same reason as line_fit_points_buf_:
  // where the memory type comes back host-visible AND cached, QuadDecode
  // reads it in place. That is not about the copy's size - it is a few KB -
  // but about its SIZE BEING KNOWN AT RECORD TIME: a staged copy needs
  // num_selected_blobs on the host, which is exactly the readback that used
  // to force a mid-frame SubmitAndWait. Reading in place removes the last
  // host dependency in the frame's tail; see fused_submits_.
  {
    vk::Buffer b(ctx_, VkDeviceSize(config_.max_blobs) * sizeof(MinMaxExtentsGpu), kSsboUsage,
                 vk::MemoryKind::DeviceLocalReadback);
    device_bytes_ += b.size();
    selected_extents_buf_ = std::move(b);
  }
  extents_direct_read_ =
      selected_extents_buf_.host_visible() && selected_extents_buf_.host_cached();
  selected_counter_buf_ = ssbo(4);
  remap_buf_ = ssbo(VkDeviceSize(config_.max_raw_blobs) * 4);

  index_points_buf_ = ssbo(VkDeviceSize(ipoint_capacity_) * sizeof(IPoint));
  blob_point_offsets_buf_ = ssbo(VkDeviceSize(config_.max_blobs) * 4);

  // The one buffer big enough for its readback path to matter: ~1.6 MB per
  // frame at 1080p. It used to be plain DeviceLocal, copied on-device into
  // readback_staging_ and then memcpy'd again into a std::vector - two full
  // copies of the same bytes on a unified-memory part. Asking for
  // device-local + host-visible + cached lets QuadDecode read it in place.
  //
  // Conditional on the memory type that actually came back being CACHED, not
  // merely host-visible: reading 1.6 MB through an uncached mapping is much
  // slower than the copy it would replace (Buffer.h records a measured 4x on
  // Mali for exactly that mistake). Discrete parts without resizable BAR
  // fall back to DeviceLocal and keep the staging path.
  {
    vk::Buffer b(ctx_, VkDeviceSize(ipoint_capacity_) * sizeof(RawLineFitPoint), kSsboUsage,
                 vk::MemoryKind::DeviceLocalReadback);
    device_bytes_ += b.size();
    line_fit_points_buf_ = std::move(b);
  }
  linefit_direct_read_ = line_fit_points_buf_.host_visible() && line_fit_points_buf_.host_cached();

  // Both readbacks in place => nothing in the frame's tail needs a host-side
  // count at record time, so the last three submissions collapse into one.
  // See fused_submits_ in the header, and build_indirect_args.comp for the
  // device-side dispatch sizing that replaces the readbacks.
  fused_submits_ = linefit_direct_read_ && extents_direct_read_;
  if (const char *v = std::getenv("APRILTAG_VK_FUSE_SUBMITS")) {
    fused_submits_ = (v[0] != '0') && linefit_direct_read_ && extents_direct_read_;
  }

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
  oversized_sort_counter_buf_ = ssbo(4);

  // VkDispatchIndirectCommand (groupCountX/Y/Z), built on-device from
  // raw_blob_counter_buf_ by build_indirect_args_pl_ so init_extents_pl_ /
  // select_blobs_pl_ dispatch over the frame's actual raw blob count instead
  // of the worst-case max_raw_blobs - see build_indirect_args.comp.
  {
    // Three VkDispatchIndirectCommands: raw blobs, boundary points,
    // selected blobs. See build_indirect_args.comp for what each feeds.
    vk::Buffer b(ctx_, 3 * 12, kSsboUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                vk::MemoryKind::DeviceLocal);
    device_bytes_ += b.size();
    indirect_args_buf_ = std::move(b);
  }

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
  // Picks the "_u8" shader variant when the device supports
  // storageBuffer8BitAccess, else the 32-bit-per-pixel fallback every other
  // device uses - see decimate_u8.comp's comment. Covers every consumer of
  // decimated_buf_ (decimate/block_minmax/sort_points_local) and
  // thresholded_buf_ (threshold/uf_init/uf_merge/label_pixels) - the two
  // switch together (threshold(_u8).comp reads the former and writes the
  // latter in one dispatch, so they can't vary independently).
  // blob_diff is no longer on that list: it reads the threshold out of
  // parent[]'s spare bits instead (see common.glsl).
  const bool u8 = ctx_.caps().has_8bit_storage;
  auto pick = [u8](const char *base_name, const char *u8_name) {
    return u8 ? u8_name : base_name;
  };

  // Picks the subgroup-aggregated variant (see uf_final_subgroup.comp /
  // reduce_extents_hash_subgroup.comp / blob_diff_body.glsl's
  // AGGREGATE_APPEND_COUNTER) when the device's subgroup exposes BALLOT,
  // ARITHMETIC and SHUFFLE in COMPUTE - gated on all three together for all
  // three sites for simplicity, even though not every site needs every bit:
  // blob_diff's variant only needs BALLOT; uf_final's and reduce_extents_
  // hash's reduce-by-key loop also need SHUFFLE, to look a runtime-computed
  // leader lane's value up (subgroupBroadcast's id must be a compile-time
  // constant - a real SPIR-V restriction - so a dynamic lane index needs
  // subgroupShuffle instead); reduce_extents_hash's needs ARITHMETIC too, to
  // reduce per-point values, not just counts.
  //
  // Excludes integrated GPUs outright, regardless of what they report
  // supporting: measured on the Orange Pi 5's Mali-G610 (which reports all
  // three capabilities and produces bit-correct results with them), this is
  // a catastrophic regression, not a wash - pipeline_total went from
  // 11.5 ms to 18.1 ms, with reduce_extents_hash_subgroup.comp's "extents"
  // span alone going from 0.91 ms to 6.04 ms, reproduced consistently
  // across repeated runs. The same class of result OPTIMIZATION_NOTES.md
  // already recorded for tile-local shared-memory union-find: Mali
  // (Valhall) has no dedicated hardware for these GPU-compute patterns the
  // way discrete parts do, so software-costly ballot/shuffle/arithmetic
  // sequences can cost far more than the plain atomics they replace. On the
  // Windows/RX 9060 XT test machine (a discrete GPU, subgroupSize=64) the
  // same code measures a small but real and repeatable win instead
  // (pipeline_total best 1.40 -> 1.33 ms), so this is excluded specifically
  // for integrated GPUs, not disabled outright.
  const bool subgroup = ctx_.caps().has_subgroup_ballot && ctx_.caps().has_subgroup_arithmetic &&
                        ctx_.caps().has_subgroup_shuffle &&
                        ctx_.caps().type != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  auto pick_sg = [subgroup](const char *base_name, const char *subgroup_name) {
    return subgroup ? subgroup_name : base_name;
  };

  // 2D so the shader recovers its (x, y) from gl_GlobalInvocationID.xy rather
  // than a runtime `%`/`/` by a push-constant width - see each shader's own
  // comment. Mali (Valhall) has no integer divide instruction.
  // decimation is baked in as a specialization constant (id 3, right after
  // the workgroup size's 0/1/2) rather than read from the push constant
  // block: it is fixed for this detector's whole lifetime, so resolving it
  // at pipeline-creation time lets the compiler fold `dx * decimation` into
  // a shift when decimation is a power of two, exactly as the literal `2`
  // it replaces already did - a runtime (push-constant) value would not get
  // that treatment, and Mali (Valhall) has no integer divide instruction to
  // fall back on.
  decimate_pl_ = vk::ComputePipeline(ctx_, ShaderPath(pick("decimate", "decimate_u8")),
                                     {gray_buf_.get(), decimated_buf_.get()}, 8, wg2d_,
                                     {config_.decimation});
  block_minmax_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("block_minmax", "block_minmax_u8")),
      {decimated_buf_.get(), minmax_unfiltered_buf_.get()}, 16, wg2d_);
  block_filter_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("block_filter"), {minmax_unfiltered_buf_.get(), minmax_filtered_buf_.get()},
      8, wg2d_);
  threshold_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("threshold", "threshold_u8")),
      {decimated_buf_.get(), minmax_filtered_buf_.get(), thresholded_buf_.get()}, 16, wg2d_);

  uf_init_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_init", "uf_init_u8")), {parent_buf_.get(), thresholded_buf_.get()},
      8, wg2d_);
  uf_merge_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("uf_merge", "uf_merge_u8")),
      {parent_buf_.get(), thresholded_buf_.get(), uf_changed_buf_.get()}, 8, wg1d_);
  // Binding 1 / the third push constant are the convergence flag and the
  // opt-in to honouring it; see uf_compress.comp.
  uf_compress_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("uf_compress"), {parent_buf_.get(), uf_changed_buf_.get()}, 12, wg1d_);
  uf_final_pl_ = vk::ComputePipeline(ctx_, ShaderPath(pick_sg("uf_final", "uf_final_subgroup")),
                                     {parent_buf_.get(), blob_size_buf_.get()}, 12, wg1d_);

  // Two-way now, not four: blob_diff used to be parametrized on the u8 axis
  // as well, because it read thresholded_buf_ directly. label_pixels.comp
  // folds the threshold into the parent[] word instead (see common.glsl), so
  // blob_diff has no thresholded binding at all and the u8 axis moved to
  // label_pixels - where it costs 2 variants instead of doubling 2 into 4.
  blob_diff_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick_sg("blob_diff", "blob_diff_subgroup")),
      {parent_buf_.get(), qbp_compacted_buf_.get(), qbp_counter_buf_.get(),
       qbp_keys_buf_.get()},
      12, wg2d_);

  label_pixels_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath(pick("label_pixels", "label_pixels_u8")),
      {parent_buf_.get(), blob_size_buf_.get(), thresholded_buf_.get()}, 8, wg1d_);

  init_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("init_extents"), {extents_buf_.get()}, 4, wg1d_);
  merge_extents_pl_ =
      vk::ComputePipeline(ctx_, ShaderPath("merge_extents"), {extents_buf_.get()}, 4, wg1d_);

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
      ctx_, ShaderPath(pick("sort_points_local", "sort_points_local_u8")),
      {selected_extents_buf_.get(), blob_point_offsets_buf_.get(), index_points_buf_.get(),
       decimated_buf_.get(), line_fit_points_buf_.get(), oversized_sort_counter_buf_.get(),
       // binding 6: the device-side selected-blob count, so this dispatch can
       // be issued indirectly in the same submission that produced it.
       selected_counter_buf_.get()},
      20, vk::WorkgroupSize{local_sort_cap_, 1, 1}, {local_sort_virtual_cap_});

  hash_group_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("hash_group"),
      {qbp_keys_buf_.get(), hash_owner_buf_.get(),
       point_slot_buf_.get(), slot_dense_buf_.get(), raw_blob_counter_buf_.get(),
       hash_drop_counter_buf_.get(), qbp_counter_buf_.get()},
      16, wg1d_);
  // One pipeline per source counter: the counter is binding 0, and
  // ComputePipeline binds its buffers at construction.
  build_indirect_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {raw_blob_counter_buf_.get(), indirect_args_buf_.get()}, 16,
      vk::WorkgroupSize{1, 1, 1});
  build_qbp_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {qbp_counter_buf_.get(), indirect_args_buf_.get()}, 16, vk::WorkgroupSize{1, 1, 1});
  build_sort_args_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("build_indirect_args"),
      {selected_counter_buf_.get(), indirect_args_buf_.get()}, 16, vk::WorkgroupSize{1, 1, 1});
  // Three-way choice, in priority order: the subgroup-aggregated variant
  // where that is enabled (discrete parts only - see the `subgroup` comment
  // above), else the 64-bit-atomic variant where the device supports it,
  // else the plain 32-bit one. The int64 variant binds extents_buf_ a
  // SECOND time at binding 4 as a uint64_t view; the two views address
  // disjoint fields of the struct, so this is aliasing in name only.
  const bool extents_atomic64 = !subgroup && ctx_.caps().has_int64_atomics;
  // Binding 4 is the uint64_t view of extents_buf_, declared only by the
  // atomic64 variant; it is bound unconditionally so that binding 5 (the
  // device-side boundary-point count) lands at the same index in all three.
  std::vector<VkBuffer> reduce_extents_buffers{qbp_compacted_buf_.get(), point_slot_buf_.get(),
                                               slot_dense_buf_.get(),   extents_buf_.get(),
                                               extents_buf_.get(),      qbp_counter_buf_.get()};
  reduce_extents_hash_pl_ = vk::ComputePipeline(
      ctx_,
      ShaderPath(subgroup ? "reduce_extents_hash_subgroup"
                          : (extents_atomic64 ? "reduce_extents_hash_atomic64"
                                              : "reduce_extents_hash")),
      reduce_extents_buffers, 12, wg1d_);
  scatter_index_points_pl_ = vk::ComputePipeline(
      ctx_, ShaderPath("scatter_index_points"),
      {qbp_compacted_buf_.get(), point_slot_buf_.get(), slot_dense_buf_.get(), remap_buf_.get(),
       selected_extents_buf_.get(), blob_point_offsets_buf_.get(), blob_cursor_buf_.get(),
       index_points_buf_.get(), qbp_counter_buf_.get()},
      16, wg1d_);
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

uint32_t GpuDetector::ReadCounterSlot(uint32_t slot) {
  const auto t0 = Clock::now();
  uint32_t value = 0;
  counter_staging_.Read(&value, 4, VkDeviceSize(slot) * 4);
  last_profile_.cpu_counter_read_ms += MsSince(t0, Clock::now());
  return value;
}

VkCommandBuffer GpuDetector::BeginTimedCommands() {
  const auto t0 = Clock::now();
  VkCommandBuffer cmd = ctx_.BeginCommands();
  last_profile_.cpu_begin_ms += MsSince(t0, Clock::now());
  return cmd;
}

void GpuDetector::SubmitTimedAndWait(VkCommandBuffer cmd) {
  const auto t0 = Clock::now();
  ctx_.SubmitAndWait(cmd);
  last_profile_.cpu_submit_wait_ms += MsSince(t0, Clock::now());
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
  // decimate_pl_'s only consumer of this: the decimated dimensions are
  // already known host-side (decimated_width_/height_ are also this
  // dispatch's own launch geometry, just below), so passing them directly
  // avoids the shader recomputing width/decimation on every invocation -
  // decimation is a specialization constant now, not a push constant, so
  // that recomputation would otherwise be a genuine runtime divide instead
  // of the free (compile-time-constant) one the literal `2` used to be.
  struct { uint32_t dw, dh; } dims_pc{decimated_width_, decimated_height_};
  struct { uint32_t dw, dh; } dwdh_pc{decimated_width_, decimated_height_};
  struct { uint32_t bw, bh; } blockdims_pc{block_width_, block_height_};

  // ------------------------------------------------------------------
  // Submit 1: decimate, adaptive threshold, and the first chunk of
  // connected-component labelling.
  // ------------------------------------------------------------------
  // Record the frame's last three submissions as one when every dispatch in
  // them can be sized on the device. See fused_submits_ in the header.
  const bool fuse = fused_submits_;
  if (fuse) {
    // The tail is one submission now; only the labelling readback still
    // splits the frame.
    last_profile_.gap_crosses_submit[kSpanBoundary] = false;   // boundary -> hash_group
    last_profile_.gap_crosses_submit[kSpanScatter] = false;    // scatter -> sort
  }
  VkCommandBuffer cmd = BeginTimedCommands();
  // One reset covers every span's timestamp pair for the whole frame: every
  // later submission this frame runs strictly after this one's fence has
  // signaled (SubmitAndWait blocks), so nothing after this point can race a
  // query this reset just cleared. See vk::QueryPool's own comment.
  timestamp_pool_.Reset(cmd);
  if (!gray_direct_write_) {
    gray_buf_.RecordCopyFrom(cmd, upload_staging_, gray_bytes);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanClear));
  qbp_counter_buf_.FillZero(cmd);
  selected_counter_buf_.FillZero(cmd);
  uf_changed_buf_.FillZero(cmd);
  blob_size_buf_.FillZero(cmd);
  hash_owner_buf_.FillZero(cmd);
  blob_cursor_buf_.FillZero(cmd);
  raw_blob_counter_buf_.FillZero(cmd);
  hash_drop_counter_buf_.FillZero(cmd);
  oversized_sort_counter_buf_.FillZero(cmd);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanClear));
  vk::ComputePipeline::Barrier(cmd, BarrierKind::ComputeAndTransfer);

  struct { uint32_t dw, dh, bw, bh; } minmax_pc{decimated_width_, decimated_height_, block_width_,
                                                block_height_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanThreshold));
  decimate_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &dims_pc);
  block_minmax_pl_.Dispatch2D(cmd, block_width_, block_height_, &minmax_pc);
  block_filter_pl_.Dispatch2D(cmd, block_width_, block_height_, &blockdims_pc);

  struct { uint32_t dw, dh, min_diff, bw; } thresh_pc{
      decimated_width_, decimated_height_, config_.min_white_black_diff, block_width_};
  threshold_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &thresh_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanThreshold));

  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelling));
  uf_init_pl_.Dispatch2D(cmd, decimated_width_, decimated_height_, &dwdh_pc);
  // Flatten the run chains uf_init.comp just built before the first merge
  // pass walks them. Without this the vertical unions pay an O(run length)
  // find() each, which cancels out exactly what the run-based init saved.
  //
  // honour_changed_flag = 0: no merge has run yet, so the flag is still zero
  // from the frame's clear, and skipping on it would skip exactly the pass
  // OPTIMIZATION_NOTES.md item 8 measured at 1.0 ms.
  struct UfCompressPc { uint32_t dw, dh, honour_changed_flag; };
  const UfCompressPc compress_init_pc{decimated_width_, decimated_height_, 0u};
  uf_compress_pl_.Dispatch1D(cmd, pixels, &compress_init_pc);

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
      // honour_changed_flag = 1: if that merge joined nothing, parent[] is
      // untouched and was already flat (every chunk ends with a
      // compression), so this pass has provably nothing to do and can skip
      // its full-array read. The dispatch's own default compute barrier is
      // what makes the merge's flag write visible here.
      const UfCompressPc compress_pc{decimated_width_, decimated_height_, 1u};
      uf_compress_pl_.Dispatch1D(c, pixels, &compress_pc);
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
  SubmitTimedAndWait(cmd);

  uint32_t uf_iterations = chunk;
  bool converged = ReadCounterSlot(kSlotUfChanged) == 0;
  while (!converged && uf_iterations < config_.max_uf_iterations) {
    const uint32_t next =
        std::min(config_.uf_iterations_per_chunk, config_.max_uf_iterations - uf_iterations);
    cmd = BeginTimedCommands();
    record_uf_chunk(cmd, next);
    RecordCounterCopy(cmd, uf_changed_buf_, kSlotUfChanged);
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
    uf_iterations += next;
    converged = ReadCounterSlot(kSlotUfChanged) == 0;
  }
  last_uf_iterations_ = uf_iterations;
  const auto t_label = Clock::now();

  // ------------------------------------------------------------------
  // Submit 2: blob sizes, boundary point extraction, and compaction. The
  // compacted count is the number every later stage is sized by.
  // ------------------------------------------------------------------
  cmd = BeginTimedCommands();
  // uf_final needs the min-size floor too, to saturate its counter at the
  // same threshold label_pixels tests against (see uf_final.comp), so it
  // can't share dwdh_pc with uf_init/uf_merge/uf_compress. The floor is
  // deliberately the same value label_pc carries below - they must stay
  // equal or the saturation stops matching the predicate.
  struct { uint32_t dw, dh, min_blob; } uf_final_pc{decimated_width_, decimated_height_,
                                                    config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanUfFinal));
  uf_final_pl_.Dispatch1D(cmd, pixels, &uf_final_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanUfFinal));

  // Fold blob identity and the min-size test into one spatially-local value
  // per pixel, so blob_diff.comp does no random gathers at all. See
  // label_pixels.comp.
  struct { uint32_t count, min_blob; } label_pc{pixels, config_.min_cluster_pixels};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanLabelPixels));
  label_pixels_pl_.Dispatch1D(cmd, pixels, &label_pc);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanLabelPixels));

  // blob_diff appends valid boundary points (with their sort keys) directly
  // into the compacted buffer, so there is no dense intermediate array and no
  // separate full-capacity compaction pass.
  struct { uint32_t w, h, capacity; } blobdiff_pc{decimated_width_, decimated_height_,
                                                   qbp_capacity_};
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBoundary));
  blob_diff_pl_.Dispatch2D(cmd, interior_width_, interior_height_, &blobdiff_pc,
                           BarrierKind::ComputeAndTransfer);
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBoundary));

  RecordCounterCopy(cmd, qbp_counter_buf_, kSlotQbpCount);
  if (!fuse) {
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
  }

  // When fusing, the count stays on the device: build_indirect_args.comp
  // turns qbp_counter_buf_ into a dispatch size and the consuming shaders
  // take their bound from the same buffer (count_from_buffer = 1). The host
  // value below is only meaningful on the unfused path; every use of it is
  // guarded by `fuse`.
  const uint32_t qbp_count =
      fuse ? 0u : std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_);
  const auto t_boundary = Clock::now();

  // ------------------------------------------------------------------
  // Submit 3: group the boundary points by (rep0, rep1), select plausible tag
  // quads, and scatter the survivors into per-blob contiguous runs.
  //
  // Everything here is dispatched over qbp_count, not qbp_capacity_ - at 1080p
  // a couple of hundred thousand real points rather than the ~2M dense bound.
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  if (!fuse) cmd = BeginTimedCommands();
  // Slot 1 of the indirect args: ceil(boundary points / wg1d), for
  // hash_group, reduce_extents_hash and scatter_index_points.
  if (fuse) {
    struct { uint32_t wg, slot, divide, clamp; } qbp_args_pc{wg1d_.x, 1u, 1u, qbp_capacity_};
    build_qbp_args_pl_.DispatchRaw(cmd, 1, 1, 1, &qbp_args_pc, BarrierKind::None);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }
  if (fuse || qbp_count > 0) {
    // Group the boundary points by (rep0, rep1) with a hash table instead of
    // sorting them. See hash_group.comp: the pipeline only ever needed the
    // grouping, never the order.
    struct { uint32_t count, table_mask, max_probes, count_from_buffer; } hash_pc{
        qbp_count, hash_table_size_ - 1u, 128u, fuse ? 1u : 0u};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanHashGroup));
    if (fuse) {
      hash_group_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &hash_pc);
    } else {
      hash_group_pl_.Dispatch1D(cmd, qbp_count, &hash_pc);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanHashGroup));

    // init_extents and select_blobs used to be dispatched over max_raw_blobs
    // rather than the frame's actual raw blob count, which only exists on
    // the device at this point - about 6 MB of wasted traffic. Building the
    // dispatch arguments on the device (from raw_blob_counter_buf_, which
    // hash_group_pl_'s default Compute-kind barrier already made visible)
    // avoids the host round-trip that made buying the exact count a wash
    // before. Both consumers still bounds-check every invocation against
    // max_raw_blobs (unchanged, passed as before), so a rounded-up overshoot
    // from build_indirect_args.comp's ceiling division is harmless.
    struct { uint32_t wg, slot, divide, clamp; } build_indirect_pc{wg1d_.x, 0u, 1u,
                                                                  config_.max_raw_blobs};
    build_indirect_args_pl_.DispatchRaw(cmd, 1, 1, 1, &build_indirect_pc, BarrierKind::None);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);

    struct { uint32_t capacity; } extentscap_pc{config_.max_raw_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanExtents));
    init_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);

    struct { uint32_t count, max_raw_blobs, count_from_buffer; } reduce_pc{
        qbp_count, config_.max_raw_blobs, fuse ? 1u : 0u};
    if (fuse) {
      reduce_extents_hash_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &reduce_pc);
    } else {
      reduce_extents_hash_pl_.Dispatch1D(cmd, qbp_count, &reduce_pc);
    }

    // Fold the per-workgroup copies back into the canonical entries before
    // select_blobs.comp reads them. Indirect over the same raw blob count as
    // init_extents, so this is a few hundred invocations.
    merge_extents_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &extentscap_pc);
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
    select_blobs_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 0, &select_pc);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSelect));

    struct { uint32_t capacity; } extract_pc{config_.max_blobs};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanBlobScan));
    extract_blob_counts_pl_.Dispatch1D(cmd, config_.max_blobs, &extract_pc);
    RunInclusiveScan(cmd, config_.max_blobs, blob_scan_chain_, blob_scan_block_pls_,
                     blob_scan_add_offsets_pls_);
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanBlobScan));

    struct { uint32_t count, capacity, max_raw_blobs, count_from_buffer; } scatter_pc{
        qbp_count, ipoint_capacity_, config_.max_raw_blobs, fuse ? 1u : 0u};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanScatter));
    if (fuse) {
      scatter_index_points_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 12, &scatter_pc,
                                                BarrierKind::ComputeAndTransfer);
    } else {
      scatter_index_points_pl_.Dispatch1D(cmd, qbp_count, &scatter_pc,
                                          BarrierKind::ComputeAndTransfer);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanScatter));

    // raw_blob_counter_buf_ is now the number of distinct (rep0, rep1) pairs
    // this frame - assigned directly by hash_group.comp's winning CAS thread
    // (see its comment), so this is the frame's raw blob count with no scan
    // needed. Profiling only.
    RecordCounterCopy(cmd, raw_blob_counter_buf_, kSlotRawBlobs);
    RecordCounterCopy(cmd, hash_drop_counter_buf_, kSlotHashDrops);

    // No separate output counter for scatter_index_points.comp's write count
    // (see its comment): blob_point_offsets_buf_[max_blobs - 1] is the
    // inclusive scan's final running total, which already equals the sum of
    // every selected blob's point count - entries beyond num_selected_blobs
    // are zero-padded by extract_blob_counts.comp, so this static offset
    // (known at record time, unlike num_selected_blobs) gives the exact same
    // total regardless of how many blobs were actually selected this frame.
    blob_point_offsets_buf_.RecordCopyTo(
        cmd, counter_staging_, 4, VkDeviceSize(std::max(config_.max_blobs, 1u) - 1u) * 4,
        VkDeviceSize(kSlotPointCount) * 4);
  }
  RecordCounterCopy(cmd, selected_counter_buf_, kSlotSelectedCount);
  if (!fuse) {
    vk::ComputePipeline::HostReadBarrier(cmd);
    SubmitTimedAndWait(cmd);
  }

  const uint32_t num_raw_blobs =
      fuse ? 0u : ((qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0);
  const uint32_t hash_probe_drops =
      fuse ? 0u : ((qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0);
  // The counter select_blobs.comp incremented is the number of blobs that
  // PASSED the filters, which can exceed max_blobs; the shader drops the
  // overflow (`if (pos >= pc.max_blobs) return;`) to stay inside the output
  // buffer. Keep the unclamped value so the drop is reported rather than
  // silently absorbed by the std::min below - see selected_blob_drops.
  const uint32_t qualifying_blobs = fuse ? 0u : ReadCounterSlot(kSlotSelectedCount);
  const uint32_t num_selected_blobs = fuse ? 0u : std::min(qualifying_blobs, config_.max_blobs);
  const uint32_t selected_blob_drops = fuse ? 0u : (qualifying_blobs - num_selected_blobs);
  const uint32_t num_points =
      fuse ? 0u
           : ((qbp_count > 0) ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_) : 0);
  const auto t_sort_group = Clock::now();

  // ------------------------------------------------------------------
  // Submit 4: sort the surviving points around each blob's perimeter, compute
  // per-point line-fit moments, and stage both payloads for readback.
  // ------------------------------------------------------------------
  // Staging sizes, and therefore this whole block, only exist on the unfused
  // path: fusing requires both readbacks to be read in place (see
  // fused_submits_), so there is nothing to stage and no size to know yet.
  const VkDeviceSize extents_bytes =
      fuse ? 0 : VkDeviceSize(num_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize linefit_offset = (extents_bytes + 15) & ~VkDeviceSize(15);
  const VkDeviceSize linefit_bytes =
      fuse ? 0 : VkDeviceSize(num_points) * sizeof(RawLineFitPoint);
  if (!fuse) EnsureReadbackCapacity(linefit_offset + linefit_bytes);

  if (!fuse) cmd = BeginTimedCommands();
  // Slot 2: one workgroup per selected blob, clamped exactly as the host's
  // std::min(qualifying_blobs, max_blobs) did.
  if (fuse) {
    struct { uint32_t wg, slot, divide, clamp; } sort_args_pc{wg1d_.x, 2u, 0u, config_.max_blobs};
    build_sort_args_pl_.DispatchRaw(cmd, 1, 1, 1, &sort_args_pc, BarrierKind::Compute);
    vk::ComputePipeline::IndirectDispatchBarrier(cmd);
  }
  if (fuse || num_points > 0) {
    // rewrite_index_points.comp already packed every selected blob's points
    // into one contiguous range, so sorting each blob's own points into
    // angular order is a one-workgroup-per-blob shared-memory sort - no
    // composite (blob_index, theta) key, no separate gather pass. Fused with
    // the line-fit moment computation (see sort_points_local.comp's own
    // comment): once a blob's points are in final order each thread already
    // knows which source point lands at its output position, so it samples
    // the decimated image and writes the RawLineFitPoint directly instead of
    // a separate dispatch re-reading a sorted intermediate.
    struct { uint32_t num_selected_blobs, count_from_buffer, max_blobs; int dw, dh; } sort_pc{
        num_selected_blobs, fuse ? 1u : 0u, config_.max_blobs,
        static_cast<int>(decimated_width_), static_cast<int>(decimated_height_)};
    timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanSort));
    if (fuse) {
      sort_points_local_pl_.DispatchIndirect(cmd, indirect_args_buf_.get(), 24, &sort_pc,
                                             BarrierKind::ComputeAndTransfer);
    } else {
      sort_points_local_pl_.DispatchRaw(cmd, num_selected_blobs, 1, 1, &sort_pc,
                                        BarrierKind::ComputeAndTransfer);
    }
    timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanSort));
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanStart(kSpanReadbackCopy));
  if (extents_bytes > 0 && !extents_direct_read_) {
    selected_extents_buf_.RecordCopyTo(cmd, readback_staging_, extents_bytes, 0, 0);
  }
  // Skipped entirely on parts where the line-fit buffer is itself
  // host-visible and cached: QuadDecode reads it in place instead. See
  // CreateBuffers.
  if (linefit_bytes > 0 && !linefit_direct_read_) {
    line_fit_points_buf_.RecordCopyTo(cmd, readback_staging_, linefit_bytes, 0, linefit_offset);
  }
  timestamp_pool_.WriteTimestamp(cmd, SpanEnd(kSpanReadbackCopy));
  // Counted by sort_points_local, so this is the first submit that can carry
  // it back.
  RecordCounterCopy(cmd, oversized_sort_counter_buf_, kSlotOversizedSortBlobs);
  vk::ComputePipeline::HostReadBarrier(cmd);
  SubmitTimedAndWait(cmd);
  const auto t_linefit = Clock::now();
  const uint32_t oversized_sort_blobs = ReadCounterSlot(kSlotOversizedSortBlobs);

  // On the fused path every count above was deferred: the GPU consumed them
  // directly and the host only needs them now, for the readback spans and
  // the profile. All four counters rode home in the same submission.
  const uint32_t final_qbp_count =
      fuse ? std::min(ReadCounterSlot(kSlotQbpCount), qbp_capacity_) : qbp_count;
  const uint32_t final_raw_blobs =
      fuse ? ((final_qbp_count > 0) ? ReadCounterSlot(kSlotRawBlobs) : 0) : num_raw_blobs;
  const uint32_t final_hash_drops =
      fuse ? ((final_qbp_count > 0) ? ReadCounterSlot(kSlotHashDrops) : 0) : hash_probe_drops;
  const uint32_t final_qualifying =
      fuse ? ReadCounterSlot(kSlotSelectedCount) : qualifying_blobs;
  const uint32_t final_selected_blobs =
      fuse ? std::min(final_qualifying, config_.max_blobs) : num_selected_blobs;
  const uint32_t final_blob_drops =
      fuse ? (final_qualifying - final_selected_blobs) : selected_blob_drops;
  const uint32_t final_num_points =
      fuse ? ((final_qbp_count > 0)
                  ? std::min(ReadCounterSlot(kSlotPointCount), ipoint_capacity_)
                  : 0)
           : num_points;
  const VkDeviceSize final_extents_bytes =
      VkDeviceSize(final_selected_blobs) * sizeof(MinMaxExtentsGpu);
  const VkDeviceSize final_linefit_bytes =
      VkDeviceSize(final_num_points) * sizeof(RawLineFitPoint);

  // ------------------------------------------------------------------
  // Host-side copies out of the persistently mapped readback buffer.
  // ------------------------------------------------------------------
  last_selected_extents.resize(final_selected_blobs);
  if (final_extents_bytes > 0) {
    if (extents_direct_read_) {
      if (!selected_extents_buf_.coherent()) {
        selected_extents_buf_.InvalidateRange(0, final_extents_bytes);
      }
      std::memcpy(last_selected_extents.data(), selected_extents_buf_.mapped(),
                  static_cast<size_t>(final_extents_bytes));
    } else {
      readback_staging_.Read(last_selected_extents.data(), final_extents_bytes, 0);
    }
  }
  if (linefit_direct_read_) {
    // Zero copies: the shader wrote these bytes straight into host-visible,
    // host-cached device memory, so hand QuadDecode a view of them. Valid
    // until the next Detect() overwrites the buffer.
    //
    // The mapping is coherent on every type seen so far, but invalidate when
    // it isn't - Buffer::Read() would have done this, and skipping the copy
    // must not also skip the invalidate.
    if (!line_fit_points_buf_.coherent()) {
      line_fit_points_buf_.InvalidateRange(0, final_linefit_bytes);
    }
    last_line_fit_points = std::span<const RawLineFitPoint>(
        static_cast<const RawLineFitPoint *>(line_fit_points_buf_.mapped()), final_num_points);
  } else {
    linefit_scratch_.resize(final_num_points);
    if (final_linefit_bytes > 0) {
      readback_staging_.Read(linefit_scratch_.data(), final_linefit_bytes, linefit_offset);
    }
    last_line_fit_points = std::span<const RawLineFitPoint>(linefit_scratch_);
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
  last_profile_.selected_blobs = final_selected_blobs;
  last_profile_.points = final_num_points;
  last_profile_.boundary_points = final_qbp_count;
  last_profile_.raw_blobs = final_raw_blobs;
  last_profile_.hash_probe_drops = final_hash_drops;
  last_profile_.selected_blob_drops = final_blob_drops;
  last_profile_.oversized_sort_blobs = oversized_sort_blobs;
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
    // Gap g is between span g's end and span (g+1)'s start - see
    // DetectProfile::gpu_gap_ms's comment for why this is worth computing
    // separately from gpu_stage_ms itself.
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
  os << ", blob capacity " << config_.max_blobs;
  os << ", device memory " << (device_bytes_ / (1024 * 1024)) << " MiB";
  os << ", gray upload " << (gray_direct_write_ ? "direct (unified memory)" : "staged");
  // Worth logging next to the upload path: whether the line-fit readback
  // avoided its copies depends on a HOST_CACHED memory type existing, which
  // varies by driver even among unified-memory parts, so a run that silently
  // took the staging path would otherwise be indistinguishable.
  os << ", linefit readback " << (linefit_direct_read_ ? "direct (host-cached)" : "staged");
  return os.str();
}

}  // namespace apriltag_vulkan
