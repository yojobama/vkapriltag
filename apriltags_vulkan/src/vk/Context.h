#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace apriltag_vulkan::vk {

// Owns the Vulkan instance, physical/logical device, a single compute+
// transfer capable queue, and a command pool. Deliberately avoids any
// vendor-specific extensions so it works unmodified on any Vulkan 1.2
// conformant driver (tested against Mesa RADV on a Radeon GPU, but will
// equally pick up NVIDIA/Intel/software implementations).
class Context {
 public:
  Context();
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return queue_family_; }
  VkCommandPool command_pool() const { return command_pool_; }

  // Finds a memory type index satisfying `type_bits` and `properties`.
  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags properties) const;

  // Allocates and begins a single-use primary command buffer.
  VkCommandBuffer BeginOneShotCommands() const;
  // Ends, submits, and waits for a single-use command buffer created by
  // BeginOneShotCommands(), then frees it.
  void EndOneShotCommands(VkCommandBuffer cmd) const;

 private:
  void CreateInstance();
  void SelectPhysicalDevice();
  void CreateLogicalDevice();

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
};

// Throws std::runtime_error with a descriptive message if `result` is not
// VK_SUCCESS.
void CheckVk(VkResult result, const char *what);

}  // namespace apriltag_vulkan::vk
