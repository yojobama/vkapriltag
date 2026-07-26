#include "vk/ComputePipeline.h"

#include <algorithm>

namespace apriltag_vulkan::vk {

ComputePipeline::ComputePipeline(const Context &ctx, const std::string &spv_path,
                                 const std::vector<VkBuffer> &buffers,
                                 uint32_t push_constant_bytes)
    : device_(ctx.device()),
      shader_(ctx.device(), spv_path),
      push_constant_bytes_(push_constant_bytes) {
  // Descriptor set layout: one storage buffer binding per entry in `buffers`.
  std::vector<VkDescriptorSetLayoutBinding> bindings(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    bindings[i] = VkDescriptorSetLayoutBinding{};
    bindings[i].binding = static_cast<uint32_t>(i);
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  CheckVk(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_),
          "vkCreateDescriptorSetLayout");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = push_constant_bytes;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &set_layout_;
  if (push_constant_bytes > 0) {
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
  }
  CheckVk(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_),
          "vkCreatePipelineLayout");

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = shader_.module();
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = pipeline_layout_;
  CheckVk(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                   &pipeline_),
          "vkCreateComputePipelines");

  if (!buffers.empty()) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<uint32_t>(buffers.size());

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    CheckVk(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
            "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo set_alloc_info{};
    set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc_info.descriptorPool = descriptor_pool_;
    set_alloc_info.descriptorSetCount = 1;
    set_alloc_info.pSetLayouts = &set_layout_;
    CheckVk(vkAllocateDescriptorSets(device_, &set_alloc_info, &descriptor_set_),
            "vkAllocateDescriptorSets");

    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffer_infos[i] = VkDescriptorBufferInfo{buffers[i], 0, VK_WHOLE_SIZE};

      writes[i] = VkWriteDescriptorSet{};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                          nullptr);
  }
}

ComputePipeline::~ComputePipeline() { Destroy(); }

void ComputePipeline::Destroy() {
  if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
  if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
  if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  if (set_layout_) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
  descriptor_pool_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  pipeline_layout_ = VK_NULL_HANDLE;
  set_layout_ = VK_NULL_HANDLE;
}

ComputePipeline::ComputePipeline(ComputePipeline &&other) noexcept
    : device_(other.device_),
      shader_(std::move(other.shader_)),
      set_layout_(other.set_layout_),
      pipeline_layout_(other.pipeline_layout_),
      pipeline_(other.pipeline_),
      descriptor_pool_(other.descriptor_pool_),
      descriptor_set_(other.descriptor_set_),
      push_constant_bytes_(other.push_constant_bytes_) {
  other.set_layout_ = VK_NULL_HANDLE;
  other.pipeline_layout_ = VK_NULL_HANDLE;
  other.pipeline_ = VK_NULL_HANDLE;
  other.descriptor_pool_ = VK_NULL_HANDLE;
  other.descriptor_set_ = VK_NULL_HANDLE;
}

ComputePipeline &ComputePipeline::operator=(ComputePipeline &&other) noexcept {
  if (this != &other) {
    Destroy();
    device_ = other.device_;
    shader_ = std::move(other.shader_);
    set_layout_ = other.set_layout_;
    pipeline_layout_ = other.pipeline_layout_;
    pipeline_ = other.pipeline_;
    descriptor_pool_ = other.descriptor_pool_;
    descriptor_set_ = other.descriptor_set_;
    push_constant_bytes_ = other.push_constant_bytes_;
    other.set_layout_ = VK_NULL_HANDLE;
    other.pipeline_layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.descriptor_pool_ = VK_NULL_HANDLE;
    other.descriptor_set_ = VK_NULL_HANDLE;
  }
  return *this;
}

void ComputePipeline::DispatchRaw(VkCommandBuffer cmd, uint32_t gx, uint32_t gy, uint32_t gz,
                                  const void *push_constants) const {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  if (descriptor_set_ != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                           &descriptor_set_, 0, nullptr);
  }
  if (push_constant_bytes_ > 0) {
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       push_constant_bytes_, push_constants);
  }
  vkCmdDispatch(cmd, gx, gy, gz);
  Barrier(cmd);
}

void ComputePipeline::Dispatch1D(VkCommandBuffer cmd, uint32_t elements, uint32_t local_size_x,
                                 const void *push_constants) const {
  uint32_t groups = (elements + local_size_x - 1) / local_size_x;
  DispatchRaw(cmd, std::max(groups, 1u), 1, 1, push_constants);
}

void ComputePipeline::Dispatch2D(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                 uint32_t local_size_x, uint32_t local_size_y,
                                 const void *push_constants) const {
  uint32_t gx = (width + local_size_x - 1) / local_size_x;
  uint32_t gy = (height + local_size_y - 1) / local_size_y;
  DispatchRaw(cmd, std::max(gx, 1u), std::max(gy, 1u), 1, push_constants);
}

void ComputePipeline::Barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                      &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace apriltag_vulkan::vk
