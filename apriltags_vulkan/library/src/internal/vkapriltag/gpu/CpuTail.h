#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "vkapriltag/gpu/Types.h"

namespace apriltag_vulkan {

// The settings of the GPU tail's group, select, scatter and sort shaders that the CPU tail reproduces.
struct CpuTailConfig {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t decimation = 2;
  bool box_decimate = true;
  uint32_t max_raw_blobs = 65536;
  uint32_t max_blobs = 0;
  uint32_t point_capacity = 0;
  uint32_t min_cluster_pixels = 24;
  uint32_t max_cluster_pixels = 100000;
  bool reversed_border = false;
  bool normal_border = true;
  float aspect_max = 8.0f;
  float fill_min = 0.5f;
  float fill_max = 2.5f;
  uint32_t large_sort_cap = 65536;
  // The device computes sqrt as the reciprocal of the reciprocal square root (Mali).
  bool sqrt_via_rsqrt = false;
};

// hash_group, reduce_extents_hash, select_blobs, scatter_index_points and sort_points_local on the
// host. Blobs are numbered in order of their first boundary point; scratch memory only grows.
class CpuTail {
 public:
  explicit CpuTail(const CpuTailConfig &config);

  // compacted: packed QBPoints; keys: (rep0, rep1) word pairs; gray: the frame the GPU decimated.
  void Run(const uint32_t *compacted, const uint32_t *keys, uint32_t count, const uint8_t *gray);

  const std::vector<MinMaxExtentsGpu> &extents() const { return selected_; }
  std::span<const RawLineFitPoint> points() const { return {out_.data(), num_points_}; }
  // Blobs that passed every filter, including those beyond max_blobs.
  uint32_t qualifying_blobs() const { return qualifying_; }
  uint32_t raw_blobs() const { return raw_blobs_; }
  uint32_t oversized_blobs() const { return oversized_; }

 private:
  struct Slot {
    uint32_t k0 = 0, k1 = 0, id1 = 0;
  };
  void GrowTable();
  int32_t DecimatedAt(int32_t x, int32_t y) const;
  RawLineFitPoint LineFitPoint(uint32_t packed, uint32_t blob) const;

  CpuTailConfig config_;
  const uint8_t *gray_ = nullptr;
  std::vector<Slot> table_;
  std::vector<MinMaxExtentsGpu> raw_;
  std::vector<uint32_t> point_blob_;
  std::vector<uint32_t> remap_, offset_, cursor_;
  std::vector<uint64_t> sort_keys_, sort_tmp_;
  std::vector<uint32_t> histogram_;
  std::vector<MinMaxExtentsGpu> selected_;
  std::vector<RawLineFitPoint> out_;
  uint32_t num_points_ = 0;
  uint32_t qualifying_ = 0;
  uint32_t raw_blobs_ = 0;
  uint32_t oversized_ = 0;
};

}  // namespace apriltag_vulkan
