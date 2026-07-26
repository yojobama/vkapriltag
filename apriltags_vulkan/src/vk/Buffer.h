#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vk/Context.h"

namespace apriltag_vulkan::vk {

// A device-local storage buffer usable as an SSBO from compute shaders, with
// helpers to upload/download its contents via a temporary host-visible
// staging buffer. No VMA / third party allocator dependency - just raw
// vkAllocateMemory, which is perfectly fine for the modest number of
// long-lived buffers this detector needs.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  VkBuffer get() const { return buffer_; }
  VkDeviceSize size() const { return size_; }

  // Uploads `data` (size bytes) into this buffer via a staging buffer and a
  // one-shot command buffer copy. Only valid for device-local buffers.
  void Upload(const Context &ctx, const void *data, VkDeviceSize bytes);
  // Downloads the first `bytes` of this buffer into `out` via staging.
  void Download(const Context &ctx, void *out, VkDeviceSize bytes) const;

  // Fills the whole buffer with repeated 32-bit `value` using vkCmdFillBuffer,
  // recorded on `cmd` (caller manages submission).
  void FillZero(VkCommandBuffer cmd) const;

 private:
  void Destroy();

  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
};

}  // namespace apriltag_vulkan::vk
