#pragma once

#include <vulkan/vulkan.h>

#include <string>

namespace apriltag_vulkan::vk {

// Loads a SPIR-V binary from disk and wraps a VkShaderModule.
class Shader {
 public:
  Shader() = default;
  Shader(VkDevice device, const std::string &spv_path);
  ~Shader();

  Shader(const Shader &) = delete;
  Shader &operator=(const Shader &) = delete;
  Shader(Shader &&other) noexcept;
  Shader &operator=(Shader &&other) noexcept;

  VkShaderModule module() const { return module_; }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkShaderModule module_ = VK_NULL_HANDLE;
};

}  // namespace apriltag_vulkan::vk
