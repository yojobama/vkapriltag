#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace apriltag_vulkan::vk {

// Tunables for device selection. Everything here can also be driven from the
// environment, so one binary can be pointed at a different GPU (or told to
// tolerate a software implementation) without a rebuild - handy when the same
// code has to run on a discrete NVIDIA/AMD card and on an embedded Mali part.
struct ContextOptions {
  // Permit a software rasterizer (Mesa lavapipe/llvmpipe, or any
  // VK_PHYSICAL_DEVICE_TYPE_CPU implementation) to be selected.
  //
  // OFF by default, and that default is deliberate: a missing GPU ICD makes
  // the loader hand back a CPU implementation that is functionally perfect
  // and ~100x slower, which is indistinguishable from "the GPU port is slow"
  // unless something says so out loud.
  //
  // Env override: APRILTAG_VK_ALLOW_CPU=1
  bool allow_cpu_device = false;

  // -1 selects by score (discrete > integrated > virtual > cpu). Otherwise a
  // raw index into vkEnumeratePhysicalDevices order.
  // Env override: APRILTAG_VK_DEVICE=<n>
  int device_index = -1;

  // Enable VK_LAYER_KHRONOS_validation when it is installed.
  // Env override: APRILTAG_VK_VALIDATION=1
  bool enable_validation = false;

  // Override the 1D compute workgroup size (rounded down to a power of two
  // and clamped to device limits). 0 = pick automatically.
  // Env override: APRILTAG_VK_WG=<n>
  uint32_t workgroup_size_override = 0;

  // Pretend the device reports at most this many invocations per workgroup.
  // Purely a testing aid: it lets a desktop GPU exercise the exact launch
  // geometry a constrained part would get (Mali-G610 reports 512, and Vulkan
  // guarantees only 128), so the portability paths can be validated without
  // the hardware in hand. 0 = use the real limit.
  // Env override: APRILTAG_VK_MAX_INVOCATIONS=<n>
  uint32_t max_invocations_override = 0;

  // Print the selected device and derived launch geometry to stderr.
  bool verbose = true;
};

// The subset of device limits/features this pipeline's launch geometry and
// memory strategy actually depend on. Collected once at startup so no hot
// path ever calls a vkGetPhysicalDevice* query.
struct DeviceCaps {
  std::string name = "<unknown>";
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  uint32_t api_version = 0;
  uint32_t vendor_id = 0;

  // --- Compute limits ---
  uint32_t max_workgroup_invocations = 128;
  uint32_t max_workgroup_size[3] = {128, 128, 64};
  uint32_t max_workgroup_count[3] = {65535, 65535, 65535};
  uint32_t max_shared_memory_bytes = 16384;

  // --- Optional features we must not assume ---
  // Mali (and plenty of other mobile parts) expose neither of these, so every
  // shader here is written to need neither. Reported for diagnostics only.
  bool has_shader_float64 = false;
  bool has_shader_int64 = false;

  // --- Memory topology ---
  // True when device-local memory is also host-visible (integrated/unified
  // parts such as Mali). Lets us skip staging copies entirely.
  bool unified_memory = false;
  bool has_host_cached = false;

  // --- Timestamp queries (for honest GPU-side timings) ---
  bool timestamps_supported = false;
  float timestamp_period_ns = 0.0f;

  // --- Derived, portable launch geometry ---
  // Vulkan only guarantees maxComputeWorkGroupInvocations >= 128, so nothing
  // may hardcode 256 - let alone the 1024 the scan shaders used to assume
  // (Mali-G610 caps out at 512, so that combination simply failed there).
  uint32_t wg1d = 128;
  uint32_t wg2d_x = 8;
  uint32_t wg2d_y = 8;
  uint32_t scan_wg = 128;  // power of two; == scan_block.comp's local_size_x

  bool is_cpu_device() const { return type == VK_PHYSICAL_DEVICE_TYPE_CPU; }
};

// Owns the Vulkan instance, physical/logical device, one compute+transfer
// queue, a command pool, and a small ring of reusable command buffers and
// fences.
//
// Requires only Vulkan 1.1 and zero optional device features or extensions,
// so it runs unmodified on desktop NVIDIA/AMD/Intel, on Mali/Adreno mobile
// drivers, and on Mesa's software implementations.
class Context {
 public:
  explicit Context(const ContextOptions &options = ContextOptions{});
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return queue_family_; }
  VkCommandPool command_pool() const { return command_pool_; }
  const DeviceCaps &caps() const { return caps_; }

  std::vector<DeviceCaps> EnumerateDevices();

  // Finds a memory type satisfying `required`, preferring one that also has
  // every bit of `preferred`. Returns UINT32_MAX when nothing satisfies
  // `required`, so callers with a fallback can test rather than catch.
  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) const;

  // Same, but throws when no memory type satisfies `required`.
  uint32_t FindMemoryTypeOrThrow(uint32_t type_bits, VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred = 0) const;

  // Begins recording into a pooled, reused command buffer, first waiting for
  // that slot's previous submission to retire. No per-frame allocation.
  VkCommandBuffer BeginCommands() const;

  // Ends, submits, and waits on a fence - not vkQueueWaitIdle, which
  // serializes every queue on the device. The command buffer is recycled.
  void SubmitAndWait(VkCommandBuffer cmd) const;

  // A human readable summary of the selected device and the launch geometry
  // derived from it.
  std::string DescribeDevice() const;
  std::string DescribeDevice(const DeviceCaps &caps) const;
 private:
  void CreateInstance(const ContextOptions &options);
  void SelectPhysicalDevice(const ContextOptions &options);
  void CreateLogicalDevice();
  void QueryCaps(const ContextOptions &options);
  void CreateCommandResources();

  static constexpr size_t kCommandRing = 4;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  VkPhysicalDeviceMemoryProperties mem_props_{};
  DeviceCaps caps_;

  // Reusable command buffers plus the fence tracking each one's submission.
  mutable VkCommandBuffer cmd_ring_[kCommandRing] = {};
  mutable VkFence fence_ring_[kCommandRing] = {};
  mutable size_t ring_next_ = 0;
  mutable size_t ring_active_ = 0;

 public:
  // Number of queue submissions made since construction. Exposed because
  // submission count is one of the things worth watching per frame.
  mutable uint64_t submit_count = 0;
};

// Throws std::runtime_error with a descriptive message if `result` is not
// VK_SUCCESS.
void CheckVk(VkResult result, const char *what);

}  // namespace apriltag_vulkan::vk
