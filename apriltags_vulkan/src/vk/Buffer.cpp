#include "vk/Buffer.h"

#include <cstring>
#include <utility>

namespace apriltag_vulkan::vk {

namespace {

void CreateRawBuffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkBuffer *out_buffer,
                     VkDeviceMemory *out_memory) {
  VkDevice device = ctx.device();
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CheckVk(vkCreateBuffer(device, &buffer_info, nullptr, out_buffer), "vkCreateBuffer");

  VkMemoryRequirements mem_reqs{};
  vkGetBufferMemoryRequirements(device, *out_buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = ctx.FindMemoryType(mem_reqs.memoryTypeBits, properties);
  CheckVk(vkAllocateMemory(device, &alloc_info, nullptr, out_memory), "vkAllocateMemory");
  CheckVk(vkBindBufferMemory(device, *out_buffer, *out_memory, 0), "vkBindBufferMemory");
}

}  // namespace

Buffer::Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags properties)
    : device_(ctx.device()), size_(size) {
  CreateRawBuffer(ctx, size, usage, properties, &buffer_, &memory_);
}

Buffer::~Buffer() { Destroy(); }

void Buffer::Destroy() {
  if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
  if (memory_) vkFreeMemory(device_, memory_, nullptr);
  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
}

Buffer::Buffer(Buffer &&other) noexcept
    : device_(other.device_), buffer_(other.buffer_), memory_(other.memory_), size_(other.size_) {
  other.buffer_ = VK_NULL_HANDLE;
  other.memory_ = VK_NULL_HANDLE;
  other.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    Destroy();
    device_ = other.device_;
    buffer_ = other.buffer_;
    memory_ = other.memory_;
    size_ = other.size_;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
  }
  return *this;
}

void Buffer::Upload(const Context &ctx, const void *data, VkDeviceSize bytes) {
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VkDeviceMemory staging_memory = VK_NULL_HANDLE;
  CreateRawBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &staging_buffer, &staging_memory);

  void *mapped = nullptr;
  CheckVk(vkMapMemory(device_, staging_memory, 0, bytes, 0, &mapped), "vkMapMemory");
  std::memcpy(mapped, data, static_cast<size_t>(bytes));
  vkUnmapMemory(device_, staging_memory);

  VkCommandBuffer cmd = ctx.BeginOneShotCommands();
  VkBufferCopy region{};
  region.size = bytes;
  vkCmdCopyBuffer(cmd, staging_buffer, buffer_, 1, &region);
  ctx.EndOneShotCommands(cmd);

  vkDestroyBuffer(device_, staging_buffer, nullptr);
  vkFreeMemory(device_, staging_memory, nullptr);
}

void Buffer::Download(const Context &ctx, void *out, VkDeviceSize bytes) const {
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VkDeviceMemory staging_memory = VK_NULL_HANDLE;
  CreateRawBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &staging_buffer, &staging_memory);

  VkCommandBuffer cmd = ctx.BeginOneShotCommands();
  VkBufferCopy region{};
  region.size = bytes;
  vkCmdCopyBuffer(cmd, buffer_, staging_buffer, 1, &region);
  ctx.EndOneShotCommands(cmd);

  void *mapped = nullptr;
  CheckVk(vkMapMemory(device_, staging_memory, 0, bytes, 0, &mapped), "vkMapMemory");
  std::memcpy(out, mapped, static_cast<size_t>(bytes));
  vkUnmapMemory(device_, staging_memory);

  vkDestroyBuffer(device_, staging_buffer, nullptr);
  vkFreeMemory(device_, staging_memory, nullptr);
}

void Buffer::FillZero(VkCommandBuffer cmd) const {
  vkCmdFillBuffer(cmd, buffer_, 0, size_, 0);
}

}  // namespace apriltag_vulkan::vk
