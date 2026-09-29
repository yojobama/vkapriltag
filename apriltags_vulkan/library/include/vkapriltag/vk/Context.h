#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "vkapriltag/vk/PipelineCache.h"

namespace apriltag_vulkan::vk {

// Device selection tunables; each can also be set from the environment.
struct ContextOptions {
  // Permit a software rasterizer (any VK_PHYSICAL_DEVICE_TYPE_CPU device) to be selected.
  // Env override: APRILTAG_VK_ALLOW_CPU=1
  bool allow_cpu_device = false;

  // Force DeviceCaps::has_8bit_storage / has_int64_atomics / has_subgroup_* to false.
  // Env override: APRILTAG_VK_FORCE_NO_8BIT=1
  bool force_no_8bit_storage = false;
  // Env override: APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1
  bool force_no_int64_atomics = false;
  // Env override: APRILTAG_VK_FORCE_NO_SUBGROUP=1
  bool force_no_subgroup = false;
  // Env override: APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING=1
  bool force_no_conditional_rendering = false;

  // -1 selects by score (discrete > integrated > virtual > cpu); otherwise an index into
  // vkEnumeratePhysicalDevices order.
  // Env override: APRILTAG_VK_DEVICE=<n>
  int device_index = -1;

  // Queue family to create the compute queue on; -1 picks the first compute-capable family.
  int queue_family = -1;

  // Compute queues created per family (clamped to what the family offers); each can back a Lane.
  uint32_t queues_per_family = 1;
  // Also creates queues on the first other compute-capable family.
  bool use_secondary_compute_family = false;

  // Enable VK_LAYER_KHRONOS_validation when it is installed.
  // Env override: APRILTAG_VK_VALIDATION=1
  bool enable_validation = false;

  // Override the 1D workgroup size (rounded down to a power of two, clamped to device limits);
  // 0 = automatic.
  // Env override: APRILTAG_VK_WG=<n>
  uint32_t workgroup_size_override = 0;

  // Override the 2D workgroup size; 0 = automatic (16x16 if 256 invocations are supported, else 8x8).
  // Env override: APRILTAG_VK_WG2D=<w>x<h>
  uint32_t workgroup_size_2d_x = 0;
  uint32_t workgroup_size_2d_y = 0;

  // Cap the reported invocations per workgroup; 0 = use the real limit.
  // Env override: APRILTAG_VK_MAX_INVOCATIONS=<n>
  uint32_t max_invocations_override = 0;

  // Persist compiled pipelines to disk (see vk::PipelineCache).
  // Env override: APRILTAG_VK_PIPELINE_CACHE=0
  bool use_pipeline_cache = true;

  // Print the selected device and derived launch geometry to stderr.
  bool verbose = true;
};

// The device limits and features the launch geometry and memory strategy depend on.
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
  // Reported for diagnostics only; no shader requires them.
  bool has_shader_float64 = false;
  bool has_shader_int64 = false;
  // VK_KHR_8bit_storage (or core 1.2) with storageBuffer8BitAccess; enabled at device creation and
  // selects the 8-bit-storage shader variants.
  bool has_8bit_storage = false;
  // VK_KHR_shader_atomic_int64 + shaderBufferInt64Atomics + shaderInt64.
  bool has_int64_atomics = false;
  // VK_EXT_conditional_rendering + conditionalRendering; enabled at device creation.
  bool has_conditional_rendering = false;
  // Subgroup capabilities (VkPhysicalDeviceSubgroupProperties); the subgroup shader variants are
  // used when both ballot and arithmetic are set.
  bool has_subgroup_ballot = false;
  bool has_subgroup_arithmetic = false;
  // Runtime-variable lane index (subgroupShuffle), needed by the reduce-by-key subgroup shaders.
  bool has_subgroup_shuffle = false;

  // --- Memory topology ---
  // True when device-local memory is also host-visible, so staging copies can be skipped.
  bool unified_memory = false;
  bool has_host_cached = false;
  // nonCoherentAtomSize: flush/invalidate ranges of non-coherent memory align to this.
  VkDeviceSize non_coherent_atom_size = 1;

  // --- Timestamp queries ---
  bool timestamps_supported = false;
  float timestamp_period_ns = 0.0f;

  // --- Derived, portable launch geometry ---
  // Launch geometry derived from the device limits.
  uint32_t wg1d = 128;
  uint32_t wg2d_x = 8;
  uint32_t wg2d_y = 8;
  uint32_t scan_wg = 128;  // power of two; == scan_block.comp's local_size_x

  bool is_cpu_device() const { return type == VK_PHYSICAL_DEVICE_TYPE_CPU; }
};

class Lane;

// Owns the Vulkan instance, device, compute+transfer queues, and a default Lane. Requires only
// Vulkan 1.1.
class Context {
public:
  explicit Context(const ContextOptions &options = ContextOptions{});
  explicit Context(const std::string& deviceName, const ContextOptions& options = ContextOptions{});
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return queue_family_; }
  const DeviceCaps &caps() const { return caps_; }

  // VK_NULL_HANDLE when pipeline caching is disabled.
  VkPipelineCache pipeline_cache() const { return pipeline_cache_.handle(); }

  // Writes the pipeline cache to disk now (skipped if unchanged).
  void FlushPipelineCache() const { pipeline_cache_.Save(); }

  // Finds a memory type with all `required` bits, preferring one with `preferred` too; returns
  // UINT32_MAX if none.
  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) const;

  // As above, but throws if none satisfies `required`.
  uint32_t FindMemoryTypeOrThrow(uint32_t type_bits, VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred = 0) const;

  // The property flags of a memory type index returned by FindMemoryType.
  VkMemoryPropertyFlags MemoryTypeFlags(uint32_t type_index) const {
    return mem_props_.memoryTypes[type_index].propertyFlags;
  }

  // The queue table, interleaved across families (family A queue 0, family B queue 0, A 1, ...).
  size_t queue_count() const { return queues_.size(); }
  uint32_t queue_slot_family(size_t slot) const { return queues_[slot].family; }

  // A submission lane on queue `slot`. Must be destroyed before the Context.
  std::unique_ptr<Lane> CreateLane(size_t slot) const;

  // The lane on queue 0, used by the forwarding functions below.
  Lane &default_lane() const { return *default_lane_; }

  // Forward to the default lane.
  VkCommandBuffer BeginCommands() const;
  void SubmitAndWait(VkCommandBuffer cmd) const;

  // Brackets commands that are skipped when the 32-bit value at `offset` in `buffer` is zero
  // (nonzero with `inverted`); only dispatches are predicated. The value's producer needs a
  // BarrierKind::*AndPredicate barrier, and `buffer` needs
  // VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT. Requires caps().has_conditional_rendering.
  void CmdBeginConditionalRendering(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset,
                                    bool inverted = false) const;
  void CmdEndConditionalRendering(VkCommandBuffer cmd) const;

  // A summary of the selected device and derived launch geometry.
  std::string DescribeDevice() const;

  static std::string DescribeDevice(const DeviceCaps &caps);
  static std::vector<DeviceCaps> EnumerateDevices();
 private:
  void CreateInstance(const ContextOptions &options);
  void SelectPhysicalDevice(const ContextOptions &options);
  void SelectPhysicalDevice(const std::string& deviceName, const ContextOptions& options);
  void CreateLogicalDevice(const ContextOptions &options);
  void QueryCaps(const ContextOptions &options);
  void CreateCommandResources();

  struct QueueSlot {
    uint32_t family = 0;
    uint32_t index = 0;
    VkQueue queue = VK_NULL_HANDLE;
    std::unique_ptr<std::mutex> submit_mutex;  // vkQueueSubmit needs external synchronisation
  };

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  std::vector<QueueSlot> queues_;
  std::unique_ptr<Lane> default_lane_;

  VkPhysicalDeviceMemoryProperties mem_props_{};
  DeviceCaps caps_;
  // Set by CreateLogicalDevice, read by QueryCaps to populate caps_.has_8bit_storage.
  bool supports_8bit_storage_ = false;
  bool supports_int64_atomics_ = false;
  bool supports_conditional_rendering_ = false;
  // Extension entry points, loaded only when supports_conditional_rendering_.
  PFN_vkCmdBeginConditionalRenderingEXT begin_conditional_rendering_ = nullptr;
  PFN_vkCmdEndConditionalRenderingEXT end_conditional_rendering_ = nullptr;
  PipelineCache pipeline_cache_;

};

// One queue's submission state: a command pool, a ring of reusable command buffers and the fence
// tracking each submission. Used by one thread at a time.
class Lane {
 public:
  ~Lane();
  Lane(const Lane &) = delete;
  Lane &operator=(const Lane &) = delete;

  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return family_; }

  // Begins recording into a pooled command buffer, first waiting for its previous submission.
  VkCommandBuffer BeginCommands();

  // Ends, submits and waits on a fence; the command buffer is recycled.
  void SubmitAndWait(VkCommandBuffer cmd);

  // Queue submissions since construction.
  uint64_t submit_count = 0;

 private:
  friend class Context;
  Lane(VkDevice device, VkQueue queue, uint32_t family, std::mutex *submit_mutex);

  static constexpr size_t kCommandRing = 4;

  VkDevice device_;
  VkQueue queue_;
  uint32_t family_;
  std::mutex *submit_mutex_;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ring_[kCommandRing] = {};
  VkFence fence_ring_[kCommandRing] = {};
  size_t ring_next_ = 0;
  size_t ring_active_ = 0;
};

// Throws std::runtime_error if `result` is not VK_SUCCESS.
void CheckVk(VkResult result, const char *what);

}  // namespace apriltag_vulkan::vk
