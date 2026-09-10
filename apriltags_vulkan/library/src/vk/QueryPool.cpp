#include "vkapriltag/vk/QueryPool.h"

#include <utility>

namespace apriltag_vulkan::vk {

QueryPool::QueryPool(VkDevice device, uint32_t count) : device_(device), count_(count) {
  if (count_ == 0) return;

  VkQueryPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = count_;
  vkCreateQueryPool(device_, &info, nullptr, &pool_);
}

QueryPool::~QueryPool() { Destroy(); }

void QueryPool::Destroy() {
  if (pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, pool_, nullptr);
  pool_ = VK_NULL_HANDLE;
  count_ = 0;
}

QueryPool::QueryPool(QueryPool &&other) noexcept
    : device_(other.device_), pool_(other.pool_), count_(other.count_) {
  other.pool_ = VK_NULL_HANDLE;
  other.count_ = 0;
}

QueryPool &QueryPool::operator=(QueryPool &&other) noexcept {
  if (this != &other) {
    Destroy();
    device_ = other.device_;
    pool_ = other.pool_;
    count_ = other.count_;
    other.pool_ = VK_NULL_HANDLE;
    other.count_ = 0;
  }
  return *this;
}

void QueryPool::Reset(VkCommandBuffer cmd) const {
  if (pool_ == VK_NULL_HANDLE) return;
  vkCmdResetQueryPool(cmd, pool_, 0, count_);
}

void QueryPool::WriteTimestamp(VkCommandBuffer cmd, uint32_t index,
                               VkPipelineStageFlagBits stage) const {
  if (pool_ == VK_NULL_HANDLE || index >= count_) return;
  vkCmdWriteTimestamp(cmd, stage, pool_, index);
}

std::vector<QueryPool::Result> QueryPool::ReadResults() const {
  if (pool_ == VK_NULL_HANDLE) return {};

  // Each query yields two 64-bit words (value, availability) rather than one,
  // so a query a given frame never wrote (a stage this frame skipped, e.g.
  // no boundary points survived compaction) reports as unavailable instead
  // of blocking forever or racing WAIT_BIT against a query that will never
  // complete.
  std::vector<uint64_t> raw(static_cast<size_t>(count_) * 2, 0);
  vkGetQueryPoolResults(device_, pool_, 0, count_, raw.size() * sizeof(uint64_t), raw.data(),
                        2 * sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

  std::vector<Result> out(count_);
  for (uint32_t i = 0; i < count_; ++i) {
    out[i].ticks = raw[2 * i];
    out[i].available = raw[2 * i + 1] != 0;
  }
  return out;
}

}  // namespace apriltag_vulkan::vk
