#include "vkapriltag/vk/CompletionFlag.h"

#include <atomic>

namespace apriltag_vulkan::vk {

CompletionFlag::CompletionFlag(const Context &ctx)
    : spec_defined_(ctx.caps().has_device_coherent_memory) {
  buffer_ = Buffer(ctx, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   spec_defined_ ? MemoryKind::DeviceCoherentHost : MemoryKind::HostVisible);
  Arm();
}

void CompletionFlag::Arm() {
  reinterpret_cast<std::atomic<uint32_t> *>(buffer_.mapped())->store(0, std::memory_order_release);
}

void CompletionFlag::RecordSet(VkCommandBuffer cmd) const {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  vkCmdFillBuffer(cmd, buffer_.get(), 0, 4, 1);
}

bool CompletionFlag::Wait(std::chrono::microseconds limit) const {
  const auto *word = reinterpret_cast<const std::atomic<uint32_t> *>(buffer_.mapped());
  const auto deadline = std::chrono::steady_clock::now() + limit;
  do {
    if (word->load(std::memory_order_acquire) != 0) return true;
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

}  // namespace apriltag_vulkan::vk
