#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace apriltag_vulkan::vk {

// Where a shader module's SPIR-V comes from: a file on disk, or a blob
// compiled into the binary (see EmbeddedShaders.h). Implicitly constructible
// from a path so existing path-based callers are unaffected.
struct ShaderSource {
  ShaderSource(std::string spv_path) : path(spv_path), label(std::move(spv_path)) {}
  ShaderSource(const uint32_t *spv_code, size_t spv_bytes, std::string spv_label)
      : code(spv_code), bytes(spv_bytes), label(std::move(spv_label)) {}

  bool embedded() const { return code != nullptr; }

  std::string path;              // used when code == nullptr
  const uint32_t *code = nullptr;
  size_t bytes = 0;
  std::string label;             // for diagnostics; the path, or the shader name
};

// Wraps a VkShaderModule built from SPIR-V, read either from disk or from
// memory.
class Shader {
 public:
  Shader() = default;
  Shader(VkDevice device, const ShaderSource &source);
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
