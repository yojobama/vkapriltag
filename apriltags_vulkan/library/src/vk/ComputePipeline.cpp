#include "vkapriltag/vk/ComputePipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace apriltag_vulkan::vk {

namespace {

struct ProbeState {
  std::mutex mutex;
  const void *owner = nullptr;
  std::thread::id recording;  // the thread recording the owner's frame
  bool noted_other = false;
  VkQueryPool pool = VK_NULL_HANDLE;
  uint32_t idx = 0;
  std::vector<std::string> names;
  uint32_t frames = 0;
  std::vector<std::pair<std::string, std::vector<double>>> rows;
};
ProbeState &Probe() {
  static ProbeState s;
  return s;
}
constexpr uint32_t kProbeSlots = 256;

}  // namespace

bool DispatchProbe::Enabled() {
  static const bool on = std::getenv("APRILTAG_VK_DISPATCH_TIMES") != nullptr;
  return on;
}

void DispatchProbe::Begin(VkCommandBuffer cmd, VkDevice device, const void *owner) {
  if (!Enabled()) return;
  ProbeState &p = Probe();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (p.owner == nullptr) {
    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = kProbeSlots;
    if (vkCreateQueryPool(device, &qi, nullptr, &p.pool) != VK_SUCCESS) {
      p.pool = VK_NULL_HANDLE;
      return;
    }
    p.owner = owner;
  }
  if (owner != p.owner) {
    // Another detector's frame: its Marks on this thread must not land in the owner's pool.
    if (p.recording == std::this_thread::get_id()) p.recording = std::thread::id();
    if (!p.noted_other) {
      std::fprintf(stderr, "APRILTAG_VK_DISPATCH_TIMES: timing the first detector only\n");
      p.noted_other = true;
    }
    return;
  }
  p.recording = std::this_thread::get_id();
  vkCmdResetQueryPool(cmd, p.pool, 0, kProbeSlots);
  p.idx = 0;
  p.names.clear();
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, p.pool, p.idx++);
  p.names.push_back("<start>");
}

void DispatchProbe::Mark(VkCommandBuffer cmd, const char *name) {
  if (!Enabled()) return;
  ProbeState &p = Probe();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (p.pool == VK_NULL_HANDLE || p.recording != std::this_thread::get_id() ||
      p.idx >= kProbeSlots) {
    return;
  }
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, p.pool, p.idx++);
  p.names.push_back(name);
}

void DispatchProbe::Collect(VkDevice device, double ns_per_tick, const void *owner) {
  if (!Enabled()) return;
  ProbeState &p = Probe();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (owner != p.owner || p.pool == VK_NULL_HANDLE || p.idx < 2) return;
  std::vector<uint64_t> t(p.idx);
  if (vkGetQueryPoolResults(device, p.pool, 0, p.idx, t.size() * 8, t.data(), 8,
                            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
    return;
  }
  if (p.rows.size() != p.idx - 1) {
    p.rows.clear();
    for (uint32_t i = 1; i < p.idx; ++i) p.rows.push_back({p.names[i], {}});
  }
  for (uint32_t i = 1; i < p.idx; ++i) {
    p.rows[i - 1].second.push_back(double(t[i] - t[i - 1]) * ns_per_tick / 1000.0);
  }
  ++p.frames;
}

void DispatchProbe::Report(VkDevice device, const void *owner) {
  if (!Enabled()) return;
  ProbeState &p = Probe();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (owner != p.owner) return;
  if (p.pool != VK_NULL_HANDLE) vkDestroyQueryPool(device, p.pool, nullptr);
  p.pool = VK_NULL_HANDLE;
  p.owner = nullptr;
  p.recording = std::thread::id();
  std::fprintf(stderr, "dispatch times over %u frames (median us, previous timestamp to this one):\n",
               p.frames);
  double total = 0;
  for (auto &r : p.rows) {
    std::vector<double> v = r.second;
    if (v.empty()) continue;
    std::sort(v.begin(), v.end());
    const double m = v[v.size() / 2];
    total += m;
    // Labels are shader paths when the SPIR-V is loaded from files.
    std::string n = r.first;
    const size_t slash = n.find_last_of("/\\");
    if (slash != std::string::npos) n = n.substr(slash + 1);
    std::fprintf(stderr, "  %-40s %9.1f\n", n.c_str(), m);
  }
  std::fprintf(stderr, "  %-40s %9.1f\n", "sum of medians", total);
  p.rows.clear();
  p.frames = 0;
}

ComputePipeline::ComputePipeline(const Context &ctx, const ShaderSource &shader_source,
                                 const std::vector<VkBuffer> &buffers,
                                 uint32_t push_constant_bytes, WorkgroupSize workgroup_size,
                                 std::vector<uint32_t> extra_specialization_constants)
    : device_(ctx.device()),
      shader_(ctx.device(), shader_source),
      push_constant_bytes_(push_constant_bytes),
      workgroup_size_(workgroup_size),
      label_(shader_source.label) {
  // Releases whatever was created so far if a later step throws.
  struct Cleanup {
    ComputePipeline *self;
    ~Cleanup() {
      if (self != nullptr) self->Destroy();
    }
  } cleanup{this};
  for (int i = 0; i < 3; ++i) max_workgroup_count_[i] = ctx.caps().max_workgroup_count[i];

  // Fail at build time rather than with an obscure driver error later.
  if (workgroup_size_.invocations() > ctx.caps().max_workgroup_invocations ||
      workgroup_size_.x > ctx.caps().max_workgroup_size[0] ||
      workgroup_size_.y > ctx.caps().max_workgroup_size[1] ||
      workgroup_size_.z > ctx.caps().max_workgroup_size[2]) {
    throw std::runtime_error(shader_source.label + ": requested workgroup size " +
                             std::to_string(workgroup_size_.x) + "x" +
                             std::to_string(workgroup_size_.y) + "x" +
                             std::to_string(workgroup_size_.z) +
                             " exceeds this device's compute limits");
  }

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

  // Workgroup dimensions are specialization constants 0/1/2; caller constants follow from ID 3.
  std::vector<uint32_t> spec_data = {workgroup_size_.x, workgroup_size_.y, workgroup_size_.z};
  spec_data.insert(spec_data.end(), extra_specialization_constants.begin(),
                   extra_specialization_constants.end());
  std::vector<VkSpecializationMapEntry> spec_entries(spec_data.size());
  for (uint32_t i = 0; i < spec_data.size(); ++i) {
    spec_entries[i].constantID = i;
    spec_entries[i].offset = i * sizeof(uint32_t);
    spec_entries[i].size = sizeof(uint32_t);
  }
  VkSpecializationInfo spec_info{};
  spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
  spec_info.pMapEntries = spec_entries.data();
  spec_info.dataSize = spec_data.size() * sizeof(uint32_t);
  spec_info.pData = spec_data.data();

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = shader_.module();
  stage_info.pName = "main";
  stage_info.pSpecializationInfo = &spec_info;

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = pipeline_layout_;
  CheckVk(vkCreateComputePipelines(device_, ctx.pipeline_cache(), 1, &pipeline_info, nullptr,
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
  cleanup.self = nullptr;
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
      push_constant_bytes_(other.push_constant_bytes_),
      workgroup_size_(other.workgroup_size_),
      label_(std::move(other.label_)) {
  for (int i = 0; i < 3; ++i) max_workgroup_count_[i] = other.max_workgroup_count_[i];
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
    workgroup_size_ = other.workgroup_size_;
    label_ = std::move(other.label_);
    for (int i = 0; i < 3; ++i) max_workgroup_count_[i] = other.max_workgroup_count_[i];
    other.set_layout_ = VK_NULL_HANDLE;
    other.pipeline_layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.descriptor_pool_ = VK_NULL_HANDLE;
    other.descriptor_set_ = VK_NULL_HANDLE;
  }
  return *this;
}

void ComputePipeline::DispatchRaw(VkCommandBuffer cmd, uint32_t gx, uint32_t gy, uint32_t gz,
                                  const void *push_constants, BarrierKind barrier) const {
  if (gx == 0 || gy == 0 || gz == 0) return;
  if (gx > max_workgroup_count_[0] || gy > max_workgroup_count_[1] ||
      gz > max_workgroup_count_[2]) {
    throw std::runtime_error("Compute dispatch exceeds maxComputeWorkGroupCount");
  }

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
  Barrier(cmd, barrier);
  DispatchProbe::Mark(cmd, label_.c_str());
}

void ComputePipeline::DispatchIndirect(VkCommandBuffer cmd, VkBuffer indirect_buffer,
                                       VkDeviceSize offset, const void *push_constants,
                                       BarrierKind barrier) const {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  if (descriptor_set_ != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                           &descriptor_set_, 0, nullptr);
  }
  if (push_constant_bytes_ > 0) {
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       push_constant_bytes_, push_constants);
  }
  vkCmdDispatchIndirect(cmd, indirect_buffer, offset);
  Barrier(cmd, barrier);
  DispatchProbe::Mark(cmd, label_.c_str());
}

void ComputePipeline::Dispatch1D(VkCommandBuffer cmd, uint32_t elements,
                                 const void *push_constants, BarrierKind barrier) const {
  if (elements == 0) return;
  const uint32_t groups = (elements + workgroup_size_.x - 1) / workgroup_size_.x;
  DispatchRaw(cmd, groups, 1, 1, push_constants, barrier);
}

void ComputePipeline::Dispatch2D(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                 const void *push_constants, BarrierKind barrier) const {
  if (width == 0 || height == 0) return;
  const uint32_t gx = (width + workgroup_size_.x - 1) / workgroup_size_.x;
  const uint32_t gy = (height + workgroup_size_.y - 1) / workgroup_size_.y;
  DispatchRaw(cmd, gx, gy, 1, push_constants, barrier);
}

void ComputePipeline::Barrier(VkCommandBuffer cmd, BarrierKind kind) {
  if (kind == BarrierKind::None) return;

  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

  VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (kind == BarrierKind::ComputeAndTransfer ||
      kind == BarrierKind::ComputeTransferAndPredicate) {
    stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  } else {
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  }
  if (kind == BarrierKind::ComputeAndPredicate ||
      kind == BarrierKind::ComputeTransferAndPredicate) {
    // The predicate read writes nothing, so the source side is an execution dependency only.
    stages |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT;
    barrier.dstAccessMask |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
  }

  vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void ComputePipeline::HostReadBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

void ComputePipeline::TransferWriteBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

void ComputePipeline::IndirectDispatchBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, nullptr, 0,
                       nullptr);
}

}  // namespace apriltag_vulkan::vk
