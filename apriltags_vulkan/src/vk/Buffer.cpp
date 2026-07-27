#include "vk/Buffer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace apriltag_vulkan::vk {

namespace {

struct MemoryRequest {
  VkMemoryPropertyFlags required;
  VkMemoryPropertyFlags preferred;
  bool map;
};

MemoryRequest RequestFor(const Context &ctx, MemoryKind kind) {
  switch (kind) {
    case MemoryKind::DeviceLocal:
      return {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false};
    case MemoryKind::HostVisible:
      // HOST_CACHED is only a preference: without it, host reads of readback
      // staging go through uncached memory and cost several times more.
      return {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              ctx.caps().has_host_cached
                  ? static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                  : static_cast<VkMemoryPropertyFlags>(0),
              true};
    case MemoryKind::DeviceLocalMapped:
      return {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              0, true};
  }
  return {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false};
}

}  // namespace

Buffer::Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind)
    : device_(ctx.device()), size_(std::max<VkDeviceSize>(size, 4)) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size_;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CheckVk(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_), "vkCreateBuffer");

  VkMemoryRequirements mem_reqs{};
  vkGetBufferMemoryRequirements(device_, buffer_, &mem_reqs);

  MemoryRequest request = RequestFor(ctx, kind);
  uint32_t type_index =
      ctx.FindMemoryType(mem_reqs.memoryTypeBits, request.required, request.preferred);

  // DeviceLocalMapped is a best-effort request: discrete GPUs without a
  // resizable BAR have no such memory type, so fall back to plain
  // device-local and let the caller stage through a separate host buffer.
  if (type_index == UINT32_MAX && kind == MemoryKind::DeviceLocalMapped) {
    request = RequestFor(ctx, MemoryKind::DeviceLocal);
    type_index = ctx.FindMemoryType(mem_reqs.memoryTypeBits, request.required, request.preferred);
  }
  if (type_index == UINT32_MAX) {
    vkDestroyBuffer(device_, buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    throw std::runtime_error("No Vulkan memory type satisfies this buffer's requirements");
  }

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = type_index;
  CheckVk(vkAllocateMemory(device_, &alloc_info, nullptr, &memory_), "vkAllocateMemory");
  CheckVk(vkBindBufferMemory(device_, buffer_, memory_, 0), "vkBindBufferMemory");

  // Map once, for the buffer's whole lifetime.
  if (request.map) {
    CheckVk(vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
  }
}

Buffer::~Buffer() { Destroy(); }

void Buffer::Destroy() {
  if (mapped_ != nullptr && memory_ != VK_NULL_HANDLE) {
    vkUnmapMemory(device_, memory_);
  }
  if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
  if (memory_) vkFreeMemory(device_, memory_, nullptr);
  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
  size_ = 0;
}

Buffer::Buffer(Buffer &&other) noexcept
    : device_(other.device_),
      buffer_(other.buffer_),
      memory_(other.memory_),
      size_(other.size_),
      mapped_(other.mapped_) {
  other.buffer_ = VK_NULL_HANDLE;
  other.memory_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    Destroy();
    device_ = other.device_;
    buffer_ = other.buffer_;
    memory_ = other.memory_;
    size_ = other.size_;
    mapped_ = other.mapped_;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void Buffer::Write(const void *src, VkDeviceSize bytes, VkDeviceSize offset) {
  if (mapped_ == nullptr) {
    throw std::runtime_error("Buffer::Write on a buffer that is not host visible");
  }
  if (offset + bytes > size_) {
    throw std::runtime_error("Buffer::Write out of range");
  }
  std::memcpy(static_cast<uint8_t *>(mapped_) + offset, src, static_cast<size_t>(bytes));
}

void Buffer::Read(void *dst, VkDeviceSize bytes, VkDeviceSize offset) const {
  if (mapped_ == nullptr) {
    throw std::runtime_error("Buffer::Read on a buffer that is not host visible");
  }
  if (offset + bytes > size_) {
    throw std::runtime_error("Buffer::Read out of range");
  }
  std::memcpy(dst, static_cast<const uint8_t *>(mapped_) + offset, static_cast<size_t>(bytes));
}

void Buffer::RecordCopyFrom(VkCommandBuffer cmd, const Buffer &src, VkDeviceSize bytes,
                            VkDeviceSize src_offset, VkDeviceSize dst_offset) const {
  if (bytes == 0) return;
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = bytes;
  vkCmdCopyBuffer(cmd, src.buffer_, buffer_, 1, &region);
}

void Buffer::RecordCopyTo(VkCommandBuffer cmd, const Buffer &dst, VkDeviceSize bytes,
                          VkDeviceSize src_offset, VkDeviceSize dst_offset) const {
  if (bytes == 0) return;
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = bytes;
  vkCmdCopyBuffer(cmd, buffer_, dst.buffer_, 1, &region);
}

void Buffer::FillZero(VkCommandBuffer cmd) const {
  vkCmdFillBuffer(cmd, buffer_, 0, size_, 0);
}

void Buffer::FillZeroRange(VkCommandBuffer cmd, VkDeviceSize offset, VkDeviceSize bytes) const {
  if (bytes == 0) return;
  // vkCmdFillBuffer requires 4-byte aligned offset and multiple-of-4 size.
  const VkDeviceSize aligned_offset = offset & ~VkDeviceSize(3);
  VkDeviceSize aligned_bytes = (bytes + (offset - aligned_offset) + 3) & ~VkDeviceSize(3);
  if (aligned_offset >= size_) return;
  aligned_bytes = std::min(aligned_bytes, size_ - aligned_offset);
  if (aligned_bytes == 0) return;
  vkCmdFillBuffer(cmd, buffer_, aligned_offset, aligned_bytes, 0);
}

}  // namespace apriltag_vulkan::vk
