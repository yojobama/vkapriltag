#include "vkapriltag/vk/Shader.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan::vk {

namespace {
std::vector<char> ReadFile(const std::string &path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + path);
  }
  size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}
}  // namespace

Shader::Shader(VkDevice device, const std::string &spv_path) : device_(device) {
  std::vector<char> code = ReadFile(spv_path);

  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  CheckVk(vkCreateShaderModule(device_, &create_info, nullptr, &module_),
          "vkCreateShaderModule");
}

Shader::~Shader() {
  if (module_) vkDestroyShaderModule(device_, module_, nullptr);
}

Shader::Shader(Shader &&other) noexcept : device_(other.device_), module_(other.module_) {
  other.module_ = VK_NULL_HANDLE;
}

Shader &Shader::operator=(Shader &&other) noexcept {
  if (this != &other) {
    if (module_) vkDestroyShaderModule(device_, module_, nullptr);
    device_ = other.device_;
    module_ = other.module_;
    other.module_ = VK_NULL_HANDLE;
  }
  return *this;
}

}  // namespace apriltag_vulkan::vk
