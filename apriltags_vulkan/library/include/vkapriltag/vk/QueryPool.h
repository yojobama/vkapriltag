#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace apriltag_vulkan::vk {

// A timestamp query pool for per-dispatch GPU profiling. Only useful when
// DeviceCaps::timestamps_supported is true; construct with count == 0 (the
// default) to get an inert, always-empty pool so callers need no #ifdef at
// the call site on devices/builds that don't want this instrumentation.
//
// Usage per frame: Reset() once at the very start of the frame's first
// command buffer (a single reset covers every index for the whole frame,
// since every submission this frame runs strictly after the previous one's
// fence has signaled - see Context::SubmitAndWait), then WriteTimestamp()
// at each point of interest across however many command buffers/submissions
// the frame uses, then ReadResults() once after the last submission
// completes. Some indices may go unwritten on a given frame (e.g. a stage
// skipped because no points survived compaction) - ReadResults() reports
// those as unavailable rather than blocking or returning garbage.
class QueryPool {
 public:
  QueryPool() = default;
  QueryPool(VkDevice device, uint32_t count);
  ~QueryPool();

  QueryPool(const QueryPool &) = delete;
  QueryPool &operator=(const QueryPool &) = delete;
  QueryPool(QueryPool &&other) noexcept;
  QueryPool &operator=(QueryPool &&other) noexcept;

  bool valid() const { return pool_ != VK_NULL_HANDLE; }
  uint32_t count() const { return count_; }

  // Resets every query in the pool to "unavailable". Must be recorded before
  // any WriteTimestamp() in the same frame; no-op if !valid().
  void Reset(VkCommandBuffer cmd) const;

  // Records a timestamp at `index` (< count()), captured after every prior
  // command in this command buffer has completed the given pipeline stage.
  // No-op if !valid() or index >= count().
  void WriteTimestamp(VkCommandBuffer cmd, uint32_t index,
                      VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) const;

  // One entry per query index. `available` is false for a query that was
  // reset but never written this frame (e.g. a skipped stage) - its
  // `ticks` is 0 in that case. Returns an empty vector if !valid().
  struct Result {
    uint64_t ticks = 0;
    bool available = false;
  };
  std::vector<Result> ReadResults() const;

 private:
  void Destroy();

  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  uint32_t count_ = 0;
};

}  // namespace apriltag_vulkan::vk
