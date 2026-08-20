#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace apriltag_vulkan::vk {

// Persists a VkPipelineCache to disk across process runs, so the ~30
// vkCreateComputePipelines calls GpuDetector::CreatePipelines() makes only
// pay their full SPIR-V -> ISA compile cost once per device per shader
// build, not on every startup.
//
// The on-disk file is keyed to the physical device (vendor/device ID plus
// the driver's own pipelineCacheUUID, which the Vulkan spec defines to
// change whenever the driver's compiled pipeline data would no longer be
// compatible) and to a hash of the compiled .spv corpus, so a driver update
// or a shader rebuild lands on a fresh file instead of feeding stale data
// back to the driver. None of this is a correctness requirement - the
// driver itself validates the blob's header and silently ignores anything
// it doesn't recognize - it just keeps the cache directory from
// accumulating dead entries.
//
// A missing, unwritable, or corrupt cache is never fatal: it just means
// this run pays the full compile cost, exactly as if caching were disabled.
class PipelineCache {
 public:
  PipelineCache() = default;

  // `shader_dir` is hashed (the compiled .spv files) to derive part of the
  // cache's identity. `enabled` is a construction-time kill switch: when
  // false, this is a cheap no-op wrapper and handle() returns
  // VK_NULL_HANDLE.
  PipelineCache(VkDevice device, VkPhysicalDevice physical_device,
                const std::string &shader_dir, bool enabled, bool verbose);
  ~PipelineCache();

  PipelineCache(const PipelineCache &) = delete;
  PipelineCache &operator=(const PipelineCache &) = delete;
  PipelineCache(PipelineCache &&other) noexcept;
  PipelineCache &operator=(PipelineCache &&other) noexcept;

  // Pass to vkCreateComputePipelines/vkCreateGraphicsPipelines. VK_NULL_HANDLE
  // when disabled, which every pipeline-creation call already treats as
  // "no cache".
  VkPipelineCache handle() const { return cache_; }

  // Writes the current cache contents to disk, skipping the write if they
  // are unchanged since the last load/save (the common case: a warm run
  // that hits every entry already on disk touches the filesystem zero
  // times). Safe to call multiple times; also called from the destructor.
  void Save() const;

  // Flushes and destroys the underlying VkPipelineCache immediately, rather
  // than waiting for this object's own destructor. Context must call this
  // before vkDestroyDevice() - member destruction order would otherwise run
  // after the device is already gone.
  void ReleaseBeforeDeviceDestruction() {
    Save();
    Destroy();
  }

 private:
  void Destroy();
  std::string CacheFilePath(VkPhysicalDevice physical_device, const std::string &shader_dir,
                            bool verbose) const;

  VkDevice device_ = VK_NULL_HANDLE;
  VkPipelineCache cache_ = VK_NULL_HANDLE;
  std::string file_path_;
  bool verbose_ = false;
  // FNV-1a of the payload last loaded from (or saved to) disk, so Save() can
  // skip the write when nothing new landed in the cache.
  mutable uint64_t last_synced_hash_ = 0;
  mutable bool has_synced_hash_ = false;
};

}  // namespace apriltag_vulkan::vk
