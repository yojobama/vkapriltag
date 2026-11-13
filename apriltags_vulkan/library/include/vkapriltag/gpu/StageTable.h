#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vkapriltag/gpu/DetectorConfig.h"
#include "vkapriltag/gpu/PipelinePlan.h"
#include "vkapriltag/vk/ComputePipeline.h"

namespace apriltag_vulkan {

// Every compute pipeline of the frame, one entry per shader instance.
enum class Stage : uint8_t {
  kDecimate,
  kBlockMinmax,
  kBlockFilter,
  kThreshold,
  kBlockMinmaxDecimate,
  kThresholdBlock,
  kNoiseFloor,
  kCellArgs,
  kUfInit,
  kUfInitScan,
  kUfCountRuns,
  kUfMerge,
  kUfCompress,
  kUfCompressCount,
  kUfFinal,
  kBlobDiff,
  kBuildQbpArgs,
  kHashGroup,
  kBuildBlobArgs,
  kReduceExtents,
  kSelectBlobs,
  kScanBlobCounts,
  kScatterIndexPoints,
  kSortLocal,
  kSortLargeRuns,
  kSortLargeMerge,
  kCount,
};
constexpr size_t kNumStages = static_cast<size_t>(Stage::kCount);

// The buffers a stage binds, by role; the detector maps a role to its buffer.
enum class Buf : uint8_t {
  kGray,
  kDecimated,
  kDecimatedView,
  kMinMaxUnfiltered,
  kMinMaxFiltered,
  kContrastFloor,
  kActiveCells,
  kCellList,
  kCellArgs,
  kThresholded,
  kParent,
  kBlobSize,
  kUfChanged,
  kQbpCompacted,
  kQbpCounter,
  kQbpKeys,
  kExtents,
  kSelectedExtents,
  kSelectedCounter,
  kRemap,
  kPointsTotal,
  kIndexPoints,
  kBlobPointOffsets,
  kLineFitPoints,
  kHashOwner,
  kPointSlot,
  kSlotDense,
  kBlobCursor,
  kRawBlobCounter,
  kHashDropCounter,
  kOversizedSortCounter,
  kLargeSort,
  kSortScratch,
  kIndirectArgs,
  kBlobArgs,
  kCount,
};

// Pixels of a row covered per thread by the threshold and uf_init shaders, and the sort's constants.
constexpr uint32_t kThresholdRun = 4;
constexpr uint32_t kUfInitRun = 4;
constexpr uint32_t kSortChunk = 16;
constexpr uint32_t kLargeSortCap = 65536;

struct StageContext {
  const PipelinePlan &plan;
  const DetectorConfig &config;
};

// What a pipeline is built from: which shader the plan selects, its buffers, push-constant size, workgroup
// size and specialisation constants.
struct StageDesc {
  Stage id;
  const char *label;
  bool (*enabled)(const StageContext &);
  std::string (*shader)(const StageContext &);
  std::vector<Buf> bindings;
  uint32_t push_bytes;
  vk::WorkgroupSize (*workgroup)(const StageContext &);
  std::vector<uint32_t> (*spec)(const StageContext &);
};

const StageDesc &StageInfo(Stage stage);

}  // namespace apriltag_vulkan
