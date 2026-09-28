#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace apriltag_vulkan::vk {

// Persists a VkPipelineCache to disk, keyed by physical device, driver pipelineCacheUUID and
// shader corpus hash. A missing, unwritable or corrupt cache is never fatal.
class PipelineCache {
 public:
  PipelineCache() = default;

  // Hashes the .spv files in `shader_dir` for the cache key; `enabled` false makes this a no-op
  // wrapper with a null handle().
  PipelineCache(VkDevice device, VkPhysicalDevice physical_device,
                const std::string &shader_dir, bool enabled, bool verbose);

  // As above, for embedded shaders: the caller supplies the corpus digest.
  PipelineCache(VkDevice device, VkPhysicalDevice physical_device, uint64_t shader_corpus_hash,
                bool enabled, bool verbose);
  ~PipelineCache();

  PipelineCache(const PipelineCache &) = delete;
  PipelineCache &operator=(const PipelineCache &) = delete;
  PipelineCache(PipelineCache &&other) noexcept;
  PipelineCache &operator=(PipelineCache &&other) noexcept;

  // Pass to pipeline creation; VK_NULL_HANDLE when disabled.
  VkPipelineCache handle() const { return cache_; }

  // Writes the cache to disk unless unchanged since the last load/save. Also called by the
  // destructor.
  void Save() const;

  // Flushes and destroys the VkPipelineCache; Context calls this before vkDestroyDevice().
  void ReleaseBeforeDeviceDestruction() {
    Save();
    Destroy();
  }

 private:
  void Destroy();
  std::string CacheFilePath(VkPhysicalDevice physical_device, uint64_t shader_corpus_hash,
                            bool verbose) const;

  VkDevice device_ = VK_NULL_HANDLE;
  VkPipelineCache cache_ = VK_NULL_HANDLE;
  std::string file_path_;
  bool verbose_ = false;
  // FNV-1a of the payload last loaded or saved, used to skip redundant writes.
  mutable uint64_t last_synced_hash_ = 0;
  mutable bool has_synced_hash_ = false;
};

}  // namespace apriltag_vulkan::vk
