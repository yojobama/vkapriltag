#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Shader.h"

namespace apriltag_vulkan::vk {

// How strongly to order a dispatch against what follows it.
//
// The original port emitted the widest possible barrier after every single
// dispatch (COMPUTE|TRANSFER on both sides, every access bit set). With ~570
// dispatches per frame that is ~570 full pipeline drains, most of which only
// ever needed shader-write -> shader-read ordering.
enum class BarrierKind {
  // No barrier. For back-to-back dispatches that touch disjoint buffers, or
  // when the caller inserts its own.
  None,
  // Shader writes become visible to subsequent shader reads/writes.
  Compute,
  // Also orders against transfer operations (vkCmdCopyBuffer/vkCmdFillBuffer)
  // on either side.
  ComputeAndTransfer,
};

// Workgroup dimensions, supplied to the shader through specialization
// constants (ids 0/1/2) rather than baked into the GLSL. Vulkan only
// guarantees maxComputeWorkGroupInvocations >= 128, and real parts differ a
// lot (Mali-G610 tops out at 512 where desktop GPUs allow 1024), so the host
// picks these from the device's reported limits at startup.
struct WorkgroupSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;

  uint32_t invocations() const { return x * y * z; }
};

// A compute pipeline bound to a fixed list of SSBO bindings (in order,
// starting at binding 0) plus an optional push constant block. Since this
// detector allocates all of its GPU buffers once up front and reuses them
// every frame, the descriptor set is written exactly once at construction.
class ComputePipeline {
 public:
  ComputePipeline() = default;
  ComputePipeline(const Context &ctx, const std::string &spv_path,
                  const std::vector<VkBuffer> &buffers, uint32_t push_constant_bytes,
                  WorkgroupSize workgroup_size);
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;
  ComputePipeline(ComputePipeline &&other) noexcept;
  ComputePipeline &operator=(ComputePipeline &&other) noexcept;

  const WorkgroupSize &workgroup_size() const { return workgroup_size_; }

  // Records a dispatch covering `elements` invocations in X, using this
  // pipeline's own workgroup size (so the host can never disagree with the
  // shader about it). A zero element count records nothing at all - with
  // count-driven dispatch sizes, empty frames are normal.
  void Dispatch1D(VkCommandBuffer cmd, uint32_t elements, const void *push_constants,
                  BarrierKind barrier = BarrierKind::Compute) const;

  void Dispatch2D(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                  const void *push_constants,
                  BarrierKind barrier = BarrierKind::Compute) const;

  void DispatchRaw(VkCommandBuffer cmd, uint32_t gx, uint32_t gy, uint32_t gz,
                   const void *push_constants,
                   BarrierKind barrier = BarrierKind::Compute) const;

  // Inserts a standalone buffer memory barrier.
  static void Barrier(VkCommandBuffer cmd, BarrierKind kind = BarrierKind::Compute);

  // Makes preceding transfer writes visible to host reads of mapped memory.
  // Required before waiting on a fence and reading a readback buffer: a fence
  // alone does not make device writes host-visible in the Vulkan memory model.
  static void HostReadBarrier(VkCommandBuffer cmd);

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
