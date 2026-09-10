#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vkapriltag/vk/PipelineCache.h"

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

  // Diagnostic aids: force DeviceCaps::has_8bit_storage / has_subgroup_* to
  // false even when the device actually supports them, so the two feature
  // axes can be isolated and A/B'd independently on hardware that supports
  // both (bisecting a regression, or measuring one optimization's effect
  // without the other's).
  // Env override: APRILTAG_VK_FORCE_NO_8BIT=1
  bool force_no_8bit_storage = false;
  // Env override: APRILTAG_VK_FORCE_NO_SUBGROUP=1
  bool force_no_subgroup = false;

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

  // Override the 2D compute workgroup size (used by decimate/threshold/
  // uf_init/blob_diff/block_minmax/block_filter). 0 = pick automatically
  // (16x16 where the device supports 256 invocations per group, 8x8
  // otherwise). A testing aid for sweeping launch geometry on a specific
  // device - see OPTIMIZATION_NOTES.md's workgroup-size item.
  // Env override: APRILTAG_VK_WG2D=<w>x<h>
  uint32_t workgroup_size_2d_x = 0;
  uint32_t workgroup_size_2d_y = 0;

  // Pretend the device reports at most this many invocations per workgroup.
  // Purely a testing aid: it lets a desktop GPU exercise the exact launch
  // geometry a constrained part would get (Mali-G610 reports 512, and Vulkan
  // guarantees only 128), so the portability paths can be validated without
  // the hardware in hand. 0 = use the real limit.
  // Env override: APRILTAG_VK_MAX_INVOCATIONS=<n>
  uint32_t max_invocations_override = 0;

  // Persist compiled pipelines to disk (see vk::PipelineCache) so the ~30
  // vkCreateComputePipelines calls GpuDetector makes only pay their full
  // SPIR-V -> ISA compile cost once per device per shader build, not on
  // every process startup. The cache file is keyed to the physical device
  // and the compiled shader corpus, so it never serves stale data after a
  // driver update or shader rebuild.
  // Env override: APRILTAG_VK_PIPELINE_CACHE=0
  bool use_pipeline_cache = true;

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
  // True when VK_KHR_8bit_storage (or its Vulkan 1.2 core promotion) is
  // present AND storageBuffer8BitAccess is actually supported, in which case
  // Context has already requested and enabled it at device creation. Unlike
  // the two above, this one IS used - GpuDetector picks 8-bit-storage shader
  // variants for decimated_buf_/thresholded_buf_ when this is true (see
  // GpuDetector::ShaderPath), with the plain 32-bit-per-pixel shaders as the
  // fallback on parts that lack it (the codebase's default assumption).
  bool has_8bit_storage = false;
  // Subgroup capability, queried via VkPhysicalDeviceSubgroupProperties
  // (core Vulkan 1.1, no extension/device-feature enablement needed - unlike
  // 8-bit storage, subgroup operations are gated purely by what the SPIR-V
  // is allowed to use, which the driver permits directly from these bits).
  // GpuDetector picks subgroup-aggregated shader variants (uf_final,
  // reduce_extents_hash, blob_diff's counter) when both are true; ballot
  // alone (without arithmetic) still lets blob_diff's variant work, since
  // it only needs ballot + broadcast + elect, but the codebase gates all
  // three sites on the same combined flag for simplicity, since Vulkan 1.1
  // guarantees ARITHMETIC and BALLOT are reported together far more often
  // than apart in practice.
  bool has_subgroup_ballot = false;
  bool has_subgroup_arithmetic = false;
  // Needed for a RUNTIME-variable lane index (subgroupShuffle) - the reduce-
  // by-key loop in uf_final_subgroup.comp / reduce_extents_hash_subgroup.
  // comp elects a leader lane computed at runtime (findLSB of a ballot), and
  // subgroupBroadcast's id operand must be a compile-time constant (a real
  // SPIR-V restriction, not a portability guess - OpGroupNonUniformBroadcast
  // requires a constant id; only OpGroupNonUniformShuffle takes a dynamic
  // one), so a fixed-lane broadcast cannot substitute here.
  bool has_subgroup_shuffle = false;

  // --- Memory topology ---
  // True when device-local memory is also host-visible (integrated/unified
  // parts such as Mali). Lets us skip staging copies entirely.
  bool unified_memory = false;
  bool has_host_cached = false;
  // VkPhysicalDeviceLimits::nonCoherentAtomSize. Non-coherent host-visible
  // reads/writes (see MemoryKind::HostVisibleCached) must be invalidated/
  // flushed on a range aligned to this size.
  VkDeviceSize non_coherent_atom_size = 1;

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
  explicit Context(const std::string& deviceName, const ContextOptions& options = ContextOptions{});
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

  // VK_NULL_HANDLE when pipeline caching is disabled (ContextOptions::
  // use_pipeline_cache = false or APRILTAG_VK_PIPELINE_CACHE=0), which every
  // pipeline-creation call already treats as "no cache".
  VkPipelineCache pipeline_cache() const { return pipeline_cache_.handle(); }

  // Writes the current pipeline cache to disk now, skipping the write if it
  // is unchanged since it was last loaded/saved. Also called automatically
  // from the destructor, but GpuDetector calls this explicitly right after
  // building its pipelines so a process that is killed rather than shut
  // down cleanly still keeps the cache.
  void FlushPipelineCache() const { pipeline_cache_.Save(); }

  // Finds a memory type satisfying `required`, preferring one that also has
  // every bit of `preferred`. Returns UINT32_MAX when nothing satisfies
  // `required`, so callers with a fallback can test rather than catch.
  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) const;

  // Same, but throws when no memory type satisfies `required`.
  uint32_t FindMemoryTypeOrThrow(uint32_t type_bits, VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred = 0) const;

  // The property flags of a memory type index returned by FindMemoryType, so
  // a caller (Buffer) can tell which optional bits (e.g. HOST_COHERENT)
  // actually landed on the type it was handed.
  VkMemoryPropertyFlags MemoryTypeFlags(uint32_t type_index) const {
    return mem_props_.memoryTypes[type_index].propertyFlags;
  }

  // Begins recording into a pooled, reused command buffer, first waiting for
  // that slot's previous submission to retire. No per-frame allocation.
  VkCommandBuffer BeginCommands() const;

  // Ends, submits, and waits on a fence - not vkQueueWaitIdle, which
  // serializes every queue on the device. The command buffer is recycled.
  void SubmitAndWait(VkCommandBuffer cmd) const;

  // A human readable summary of the selected device and the launch geometry
  // derived from it.
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

  static constexpr size_t kCommandRing = 4;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  VkPhysicalDeviceMemoryProperties mem_props_{};
  DeviceCaps caps_;
  // Set by CreateLogicalDevice (which runs before QueryCaps and is the only
  // place that can actually request+enable the extension), read by
  // QueryCaps to populate caps_.has_8bit_storage.
  bool supports_8bit_storage_ = false;
  PipelineCache pipeline_cache_;

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
