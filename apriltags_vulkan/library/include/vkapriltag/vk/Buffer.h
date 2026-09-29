#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan::vk {

// Where a buffer's memory should live.
enum class MemoryKind {
  // Fastest for shader access. Not host visible on discrete GPUs.
  DeviceLocal,

  // Host staging for writes: HOST_VISIBLE | HOST_COHERENT, plus HOST_CACHED when available.
  HostVisible,

  // Host staging for reads: HOST_VISIBLE and preferably HOST_CACHED; not necessarily coherent
  // (Buffer::Read() invalidates when needed).
  HostVisibleCached,

  // DEVICE_LOCAL and host-visible (unified memory or resizable BAR); falls back to DeviceLocal, so
  // callers must check host_visible().
  DeviceLocalMapped,

  // As above, preferring HOST_CACHED, for buffers a shader writes and the host reads; callers
  // must check host_visible() and host_cached().
  DeviceLocalReadback,
};

// A Vulkan buffer plus its memory allocation; host-visible allocations stay mapped for their
// lifetime.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind);

  // A buffer backed by caller memory (VK_EXT_external_memory_host). `ptr` and `size` must be
  // multiples of caps().min_host_pointer_alignment and the memory must outlive the buffer.
  static Buffer ImportHostPointer(const Context &ctx, void *ptr, VkDeviceSize size,
                                  VkBufferUsageFlags usage);

  // A buffer backed by a dma-buf. `fd` is duplicated, so the caller keeps its own descriptor.
  static Buffer ImportDmaBuf(const Context &ctx, int fd, VkDeviceSize size,
                             VkBufferUsageFlags usage);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  VkBuffer get() const { return buffer_; }
  VkDeviceSize size() const { return size_; }

  // Non-null only for host-visible allocations.
  void *mapped() const { return mapped_; }
  bool host_visible() const { return mapped_ != nullptr; }

  // Host access to mapped memory: a memcpy, wrapped in flush (Write) / invalidate (Read) when the
  // memory is not HOST_COHERENT.
  void Write(const void *src, VkDeviceSize bytes, VkDeviceSize offset = 0);
  void Read(void *dst, VkDeviceSize bytes, VkDeviceSize offset = 0) const;

  // Makes a range of the mapping host-visible for in-place reads; no-op on coherent memory.
  void InvalidateRange(VkDeviceSize offset, VkDeviceSize bytes) const;

  // True unless the memory type is HOST_VISIBLE without HOST_COHERENT.
  bool coherent() const { return coherent_; }

  // True when the memory type is HOST_CACHED; zero-copy readback through the mapping needs this.
  bool host_cached() const { return host_cached_; }

  // --- Record-only helpers: no submission, no allocation. ---

  // Records a copy of `bytes` from `src` into this buffer.
  void RecordCopyFrom(VkCommandBuffer cmd, const Buffer &src, VkDeviceSize bytes,
                      VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0) const;

  // Records a copy of `bytes` from this buffer into `dst`.
  void RecordCopyTo(VkCommandBuffer cmd, const Buffer &dst, VkDeviceSize bytes,
                    VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0) const;

  // Zero-fills the whole buffer, or a 4-byte-aligned sub-range.
  void FillZero(VkCommandBuffer cmd) const;
  void FillZeroRange(VkCommandBuffer cmd, VkDeviceSize offset, VkDeviceSize bytes) const;

 private:
  void Destroy();

  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  void *mapped_ = nullptr;
  bool coherent_ = true;
  bool host_cached_ = false;
  VkDeviceSize non_coherent_atom_size_ = 1;
};

}  // namespace apriltag_vulkan::vk
