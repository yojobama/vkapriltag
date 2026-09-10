#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan::vk {

// Where a buffer's memory should live. Picking this per buffer (rather than
// passing raw VkMemoryPropertyFlags everywhere) keeps the unified-memory
// fallbacks in one place.
enum class MemoryKind {
  // Fastest for shader access. Not host visible on discrete GPUs.
  DeviceLocal,

  // Host staging memory: HOST_VISIBLE | HOST_COHERENT, additionally
  // HOST_CACHED when the driver offers it as well. Use this for host WRITES
  // (upload staging): a plain memcpy needs no flush precisely because it's
  // coherent, and upload is not read-sensitive to cache state.
  HostVisible,

  // Host staging memory for host READS (readback/counter staging):
  // HOST_VISIBLE required, HOST_CACHED preferred, COHERENT NOT required.
  // Some devices (e.g. Mali-G610 on the Orange Pi 5) expose no memory type
  // that is both COHERENT and CACHED - asking HostVisible's required set for
  // both silently falls through to an uncached type, which measured 4x
  // slower for a 1.5 MB per-frame readback on that device. Reading through
  // this kind requires an explicit invalidate, which Buffer::Read() does
  // automatically when the buffer's selected memory type isn't coherent -
  // see Buffer::coherent_.
  HostVisibleCached,

  // DEVICE_LOCAL and host-visible at once. Exists on unified-memory parts
  // (Mali and other integrated GPUs) and on discrete cards with resizable
  // BAR. Falls back to plain DeviceLocal when unavailable, so callers must
  // check host_visible() and keep a staging path for the fallback.
  DeviceLocalMapped,
};

// A Vulkan buffer plus its memory allocation. Host-visible allocations are
// mapped once at construction and stay mapped for their whole lifetime -
// mapping is not free, and nothing here benefits from unmapping.
//
// No VMA dependency: this detector allocates a fixed set of long-lived
// buffers at startup, so raw vkAllocateMemory is entirely adequate. What
// matters is that no allocation happens per frame.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  VkBuffer get() const { return buffer_; }
  VkDeviceSize size() const { return size_; }

  // Non-null only for host-visible allocations (HostVisible always, or
  // DeviceLocalMapped when the device actually supports it).
  void *mapped() const { return mapped_; }
  bool host_visible() const { return mapped_ != nullptr; }

  // Host-side access to mapped memory. Both are a plain memcpy, transparently
  // wrapped in vkFlushMappedMemoryRanges (Write) / vkInvalidateMappedMemoryRanges
  // (Read) when the buffer's actual memory type isn't HOST_COHERENT - see
  // MemoryKind::HostVisibleCached. Coherent memory (every other kind) pays
  // nothing extra: coherent() short-circuits both calls.
  void Write(const void *src, VkDeviceSize bytes, VkDeviceSize offset = 0);
  void Read(void *dst, VkDeviceSize bytes, VkDeviceSize offset = 0) const;

  // True unless this buffer's memory type is HOST_VISIBLE without
  // HOST_COHERENT (only possible for MemoryKind::HostVisibleCached, and only
  // when the device has no memory type that is both).
  bool coherent() const { return coherent_; }

  // --- Record-only helpers. No submission, no allocation. ---

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
  VkDeviceSize non_coherent_atom_size_ = 1;
};

}  // namespace apriltag_vulkan::vk
