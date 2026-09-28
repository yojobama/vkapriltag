#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/Context.h"
#include "vkapriltag/vk/Shader.h"

namespace apriltag_vulkan::vk {

// How strongly to order a dispatch against what follows it.
enum class BarrierKind {
  // No barrier.
  None,
  // Shader writes become visible to subsequent shader reads/writes.
  Compute,
  // Also orders against transfer operations on either side.
  ComputeAndTransfer,
};

// Workgroup dimensions, supplied to shaders as specialization constants 0/1/2 and chosen from the
// device limits at startup.
struct WorkgroupSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;

  uint32_t invocations() const { return x * y * z; }
};

// A compute pipeline bound to a fixed list of SSBOs (in order from binding 0) plus an optional push
// constant block; the descriptor set is written once at construction.
class ComputePipeline {
 public:
  ComputePipeline() = default;
  // `extra_specialization_constants` are bound to consecutive constant IDs starting at 3.
  ComputePipeline(const Context &ctx, const ShaderSource &shader_source,
                  const std::vector<VkBuffer> &buffers, uint32_t push_constant_bytes,
                  WorkgroupSize workgroup_size,
                  std::vector<uint32_t> extra_specialization_constants = {});
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;
  ComputePipeline(ComputePipeline &&other) noexcept;
  ComputePipeline &operator=(ComputePipeline &&other) noexcept;

  const WorkgroupSize &workgroup_size() const { return workgroup_size_; }

  // Records a dispatch of `elements` invocations in X using the pipeline's workgroup size; a zero
  // count records nothing.
  void Dispatch1D(VkCommandBuffer cmd, uint32_t elements, const void *push_constants,
                  BarrierKind barrier = BarrierKind::Compute) const;

  void Dispatch2D(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                  const void *push_constants,
                  BarrierKind barrier = BarrierKind::Compute) const;

  void DispatchRaw(VkCommandBuffer cmd, uint32_t gx, uint32_t gy, uint32_t gz,
                   const void *push_constants,
                   BarrierKind barrier = BarrierKind::Compute) const;

  // As DispatchRaw, but the group counts (x, y, z uint32s) are read from `indirect_buffer` at
  // `offset`. The buffer needs VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT and must be ordered with
  // IndirectDispatchBarrier.
  void DispatchIndirect(VkCommandBuffer cmd, VkBuffer indirect_buffer, VkDeviceSize offset,
                        const void *push_constants,
                        BarrierKind barrier = BarrierKind::Compute) const;

  // Inserts a standalone buffer memory barrier.
  static void Barrier(VkCommandBuffer cmd, BarrierKind kind = BarrierKind::Compute);

  // Makes preceding transfer writes visible to host reads of mapped memory (required before
  // reading a readback buffer after a fence wait).
  static void HostReadBarrier(VkCommandBuffer cmd);

  // Makes a compute shader's write to an indirect-argument buffer visible to DispatchIndirect
  // (the DRAW_INDIRECT stage is not covered by BarrierKind::Compute).
  static void IndirectDispatchBarrier(VkCommandBuffer cmd);

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
  WorkgroupSize workgroup_size_;
  uint32_t max_workgroup_count_[3] = {65535, 65535, 65535};
};

}  // namespace apriltag_vulkan::vk
