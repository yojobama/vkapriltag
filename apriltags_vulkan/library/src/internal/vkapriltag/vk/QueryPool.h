#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace apriltag_vulkan::vk {

// Timestamp query pool for per-dispatch GPU profiling; a pool with count == 0 is inert. Reset()
// once per frame, WriteTimestamp() at each point of interest, ReadResults() after the last
// submission completes.
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

  // Resets every query to "unavailable"; no-op if !valid().
  void Reset(VkCommandBuffer cmd) const;

  // Records a timestamp at `index` once prior commands complete `stage`; no-op if !valid() or
  // index >= count().
  void WriteTimestamp(VkCommandBuffer cmd, uint32_t index,
                      VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) const;

  // One entry per query; `available` is false (and `ticks` 0) for queries not written this frame.
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
