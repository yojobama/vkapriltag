#include "vkapriltag/gpu/StageTable.h"

#include <array>

namespace apriltag_vulkan {

namespace {

using Ctx = StageContext;

bool Always(const Ctx &) { return true; }

// The 8-bit-storage variant when the plan has it.
std::string Pick(const Ctx &c, const char *base, const char *u8) { return c.plan.u8 ? u8 : base; }

vk::WorkgroupSize Wg1d(const Ctx &c) { return c.plan.wg1d; }
vk::WorkgroupSize Wg2d(const Ctx &c) { return c.plan.wg2d; }
vk::WorkgroupSize Single(const Ctx &) { return vk::WorkgroupSize{1, 1, 1}; }
std::vector<uint32_t> NoSpec(const Ctx &) { return {}; }

// The three sort pipelines differ only in their mode constant (0 whole blobs, 1 runs of large blobs, 2 merge).
std::string SortShader(const Ctx &c, uint32_t mode) {
  return mode == 0 ? Pick(c, "sort_points_local", "sort_points_local_u8") : Pick(c, "sort_points_large", "sort_points_large_u8");
}
std::vector<uint32_t> SortSpec(const Ctx &c, uint32_t mode) {
  return {c.plan.local_sort_split, kSortChunk, mode, kLargeSortCap, mode == 0u ? c.plan.sort_rank_all : 0u};
}

const std::vector<Buf> kSortBindings = {Buf::kSelectedExtents,   Buf::kBlobPointOffsets,     Buf::kIndexPoints,
                                        Buf::kDecimatedView,     Buf::kLineFitPoints,        Buf::kOversizedSortCounter,
                                        Buf::kSelectedCounter,   Buf::kLargeSort,            Buf::kSortScratch};

const std::array<StageDesc, kNumStages> &Table() {
  static const std::array<StageDesc, kNumStages> table = {{
      {Stage::kDecimate, "decimate", Always,
       [](const Ctx &c) -> std::string {
         return c.plan.box_dot ? "decimate_box_dot_u8"
                : c.plan.box_decimate ? Pick(c, "decimate_box", "decimate_box_u8")
                                      : Pick(c, "decimate", "decimate_u8");
       },
       {Buf::kGray, Buf::kDecimated}, 8, Wg2d, [](const Ctx &c) { return std::vector<uint32_t>{c.config.decimation}; }},
      {Stage::kBlockMinmax, "block_minmax", Always, [](const Ctx &c) { return Pick(c, "block_minmax", "block_minmax_u8"); },
       {Buf::kDecimatedView, Buf::kMinMaxUnfiltered}, 16, Wg2d, NoSpec},
      {Stage::kBlockFilter, "block_filter", Always, [](const Ctx &) -> std::string { return "block_filter"; },
       {Buf::kMinMaxUnfiltered, Buf::kMinMaxFiltered}, 8, Wg2d, NoSpec},
      {Stage::kThreshold, "threshold", Always, [](const Ctx &c) { return Pick(c, "threshold", "threshold_u8"); },
       {Buf::kDecimatedView, Buf::kMinMaxFiltered, Buf::kThresholded}, 16, Wg2d,
       [](const Ctx &) { return std::vector<uint32_t>{kThresholdRun}; }},
      {Stage::kBlockMinmaxDecimate, "block_minmax_decimate", [](const Ctx &c) { return c.plan.minmax_decimate; },
       [](const Ctx &) -> std::string { return "block_minmax_decimate_u8"; },
       {Buf::kGray, Buf::kDecimated, Buf::kMinMaxUnfiltered}, 16, Wg2d,
       [](const Ctx &c) { return std::vector<uint32_t>{c.config.decimation}; }},
      {Stage::kThresholdBlock, "threshold_block", [](const Ctx &c) { return c.plan.threshold_block; },
       [](const Ctx &) -> std::string { return "threshold_block_u8"; },
       {Buf::kDecimatedView, Buf::kMinMaxUnfiltered, Buf::kThresholded, Buf::kContrastFloor, Buf::kActiveCells, Buf::kCellList},
       20, Wg2d,
       [](const Ctx &c) { return std::vector<uint32_t>{c.plan.adaptive_floor ? 1u : 0u, c.plan.sparse ? 1u : 0u}; }},
      {Stage::kNoiseFloor, "noise_floor", [](const Ctx &c) { return c.plan.threshold_block && c.plan.adaptive_floor; },
       [](const Ctx &) -> std::string { return "noise_floor"; }, {Buf::kMinMaxUnfiltered, Buf::kContrastFloor}, 12, Wg1d,
       NoSpec},
      {Stage::kCellArgs, "cell_args", [](const Ctx &c) { return c.plan.threshold_block && c.plan.sparse; },
       [](const Ctx &) -> std::string { return "cell_args"; }, {Buf::kCellList, Buf::kCellArgs}, 0, Single, NoSpec},
      {Stage::kUfInit, "uf_init", Always, [](const Ctx &c) { return Pick(c, "uf_init", "uf_init_u8"); },
       {Buf::kParent, Buf::kThresholded}, 8, Wg2d, [](const Ctx &) { return std::vector<uint32_t>{kUfInitRun}; }},
      {Stage::kUfInitScan, "uf_init_scan", [](const Ctx &c) { return c.plan.uf_init_scan; },
       [](const Ctx &c) { return Pick(c, "uf_init_scan", "uf_init_scan_u8"); },
       {Buf::kParent, Buf::kThresholded, Buf::kBlobSize, Buf::kCellList}, 12,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.sparse ? 16u : c.plan.wg1d.x, 1, 1}; },
       [](const Ctx &c) { return std::vector<uint32_t>{c.plan.count_runs ? 1u : 0u, c.plan.sparse ? 1u : 0u}; }},
      {Stage::kUfCountRuns, "uf_count_runs", [](const Ctx &c) { return c.plan.count_runs; },
       [](const Ctx &c) { return Pick(c, "uf_count_runs", "uf_count_runs_u8"); },
       {Buf::kParent, Buf::kThresholded, Buf::kBlobSize, Buf::kCellList}, 16,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.sparse ? 16u : c.plan.wg1d.x, 1, 1}; },
       [](const Ctx &c) { return std::vector<uint32_t>{c.plan.sparse ? 1u : 0u}; }},
      {Stage::kUfMerge, "uf_merge", Always,
       [](const Ctx &c) { return c.plan.merge_x4 ? std::string("uf_merge_x4_u8") : Pick(c, "uf_merge", "uf_merge_u8"); },
       {Buf::kParent, Buf::kThresholded, Buf::kUfChanged, Buf::kCellList}, 16,
       [](const Ctx &c) {
         return vk::WorkgroupSize{c.plan.sparse ? (c.plan.merge_x4 ? 16u : 64u) : c.plan.wg1d.x, 1, 1};
       },
       [](const Ctx &c) { return std::vector<uint32_t>{c.plan.merge_flag_mode, c.plan.sparse ? 1u : 0u}; }},
      {Stage::kUfCompress, "uf_compress", Always, [](const Ctx &) -> std::string { return "uf_compress"; },
       {Buf::kParent, Buf::kUfChanged}, 12, Wg1d, NoSpec},
      {Stage::kUfCompressCount, "uf_compress_count", Always,
       [](const Ctx &c) { return Pick(c, "uf_compress_count", "uf_compress_count_u8"); },
       {Buf::kParent, Buf::kUfChanged, Buf::kThresholded, Buf::kBlobSize}, 16, Wg1d, NoSpec},
      {Stage::kUfFinal, "uf_final", Always, [](const Ctx &c) { return Pick(c, "uf_final", "uf_final_u8"); },
       {Buf::kParent, Buf::kBlobSize, Buf::kThresholded}, 12, Wg1d, NoSpec},
      {Stage::kBlobDiff, "blob_diff", Always, [](const Ctx &c) { return Pick(c, "blob_diff", "blob_diff_u8"); },
       {Buf::kParent, Buf::kQbpCompacted, Buf::kQbpCounter, Buf::kQbpKeys, Buf::kUfChanged, Buf::kBlobSize,
        Buf::kThresholded, Buf::kCellList},
       20, [](const Ctx &c) { return c.plan.sparse ? vk::WorkgroupSize{16, 4, 1} : c.plan.wg2d; },
       [](const Ctx &c) { return std::vector<uint32_t>{c.plan.count_runs ? 1u : 0u, c.plan.sparse ? 1u : 0u}; }},
      {Stage::kBuildQbpArgs, "build_indirect_args", Always, [](const Ctx &) -> std::string { return "build_indirect_args"; },
       {Buf::kQbpCounter, Buf::kIndirectArgs}, 20, Single, NoSpec},
      {Stage::kHashGroup, "hash_group", Always, [](const Ctx &) -> std::string { return "hash_group"; },
       {Buf::kQbpKeys, Buf::kHashOwner, Buf::kPointSlot, Buf::kSlotDense, Buf::kRawBlobCounter, Buf::kHashDropCounter,
        Buf::kQbpCounter, Buf::kExtents},
       20, Wg1d, NoSpec},
      {Stage::kBuildBlobArgs, "build_indirect_args", Always, [](const Ctx &) -> std::string { return "build_indirect_args"; },
       {Buf::kRawBlobCounter, Buf::kBlobArgs}, 20, Single, NoSpec},
      {Stage::kReduceExtents, "reduce_extents_hash", Always,
       [](const Ctx &c) -> std::string {
         return c.plan.subgroup           ? "reduce_extents_hash_subgroup"
                : c.plan.extents_atomic64 ? "reduce_extents_hash_atomic64_x4"
                                          : "reduce_extents_hash_x4";
       },
       {Buf::kQbpCompacted, Buf::kPointSlot, Buf::kSlotDense, Buf::kExtents, Buf::kExtents, Buf::kQbpCounter}, 12, Wg1d, NoSpec},
      {Stage::kSelectBlobs, "select_blobs", Always, [](const Ctx &) -> std::string { return "select_blobs"; },
       {Buf::kExtents, Buf::kSelectedExtents, Buf::kSelectedCounter, Buf::kRemap, Buf::kRawBlobCounter,
        Buf::kBlobPointOffsets, Buf::kPointsTotal, Buf::kBlobArgs},
       44, Wg1d, [](const Ctx &c) { return std::vector<uint32_t>{(c.plan.lean_tail & 2u) != 0 ? 1u : 0u}; }},
      {Stage::kScanBlobCounts, "scan_blob_counts", Always, [](const Ctx &) -> std::string { return "scan_blob_counts"; },
       {Buf::kSelectedExtents, Buf::kSelectedCounter, Buf::kBlobPointOffsets, Buf::kBlobArgs}, 8,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.scan_wg, 1, 1}; }, NoSpec},
      {Stage::kScatterIndexPoints, "scatter_index_points", Always, [](const Ctx &) -> std::string { return "scatter_index_points"; },
       {Buf::kQbpCompacted, Buf::kPointSlot, Buf::kSlotDense, Buf::kRemap, Buf::kSelectedExtents, Buf::kBlobPointOffsets,
        Buf::kBlobCursor, Buf::kIndexPoints, Buf::kQbpCounter},
       16, Wg1d, NoSpec},
      {Stage::kSortLocal, "sort_points_local", Always, [](const Ctx &c) { return SortShader(c, 0); }, kSortBindings, 20,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.local_sort_cap, 1, 1}; },
       [](const Ctx &c) { return SortSpec(c, 0); }},
      {Stage::kSortLargeRuns, "sort_points_large_runs", Always, [](const Ctx &c) { return SortShader(c, 1); }, kSortBindings, 20,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.local_sort_cap, 1, 1}; },
       [](const Ctx &c) { return SortSpec(c, 1); }},
      {Stage::kSortLargeMerge, "sort_points_large_merge", Always, [](const Ctx &c) { return SortShader(c, 2); }, kSortBindings, 20,
       [](const Ctx &c) { return vk::WorkgroupSize{c.plan.local_sort_cap, 1, 1}; },
       [](const Ctx &c) { return SortSpec(c, 2); }},
  }};
  return table;
}

}  // namespace

const StageDesc &StageInfo(Stage stage) {
  const StageDesc &d = Table()[static_cast<size_t>(stage)];
  return d;
}

}  // namespace apriltag_vulkan
