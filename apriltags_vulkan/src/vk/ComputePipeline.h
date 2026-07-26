#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Shader.h"

namespace apriltag_vulkan::vk {

// A compute pipeline bound to a fixed list of SSBO bindings (in order,
// starting at binding 0) plus an optional push constant block. Since this
// detector allocates all of its GPU buffers once up front and reuses them
// every frame, the descriptor set is written exactly once at construction
// time.
class ComputePipeline {
 public:
  ComputePipeline() = default;
  ComputePipeline(const Context &ctx, const std::string &spv_path,
                  const std::vector<VkBuffer> &buffers, uint32_t push_constant_bytes);
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;
  ComputePipeline(ComputePipeline &&other) noexcept;
  ComputePipeline &operator=(ComputePipeline &&other) noexcept;

  // Records a dispatch of ceil(elements / local_size_x) workgroups in X
  // (with Y = Z = 1), pushing `push_constants` (must match the size passed
  // at construction) and inserting a full buffer memory barrier afterwards
  // so subsequent dispatches/copies see this dispatch's writes.
  void Dispatch1D(VkCommandBuffer cmd, uint32_t elements, uint32_t local_size_x,
                 const void *push_constants) const;
  void Dispatch2D(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                  uint32_t local_size_x, uint32_t local_size_y,
                  const void *push_constants) const;
  void DispatchRaw(VkCommandBuffer cmd, uint32_t gx, uint32_t gy, uint32_t gz,
                   const void *push_constants) const;

  // Inserts a full read/write buffer memory barrier (safe, if pessimistic,
  // synchronization between any two compute dispatches).
  static void Barrier(VkCommandBuffer cmd);

 private:
  void Destroy();

  VkDevice device_ = VK_NULL_HANDLE;
  Shader shader_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  uint32_t push_constant_bytes_ = 0;
};

}  // namespace apriltag_vulkan::vk
