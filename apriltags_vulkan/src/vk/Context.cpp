#include "vk/Context.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace apriltag_vulkan::vk {

void CheckVk(VkResult result, const char *what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with VkResult " +
                             std::to_string(static_cast<int>(result)));
  }
}

Context::Context() {
  CreateInstance();
  SelectPhysicalDevice();
  CreateLogicalDevice();

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_;
  CheckVk(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
          "vkCreateCommandPool");
}

Context::~Context() {
  if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
  if (device_) vkDestroyDevice(device_, nullptr);
  if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Context::CreateInstance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "apriltag_vulkan";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  CheckVk(vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");
}

void Context::SelectPhysicalDevice() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (count == 0) {
    throw std::runtime_error("No Vulkan capable physical devices found");
  }
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance_, &count, devices.data());

  // Score devices: prefer any device that exposes a compute-capable queue
  // family, preferring discrete GPUs but falling back to whatever is
  // available (integrated, virtual, or CPU) to stay GPU-agnostic.
  int best_score = -1;
  for (VkPhysicalDevice candidate : devices) {
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, qfs.data());

    bool has_compute = false;
    for (const auto &qf : qfs) {
      if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        has_compute = true;
        break;
      }
    }
    if (!has_compute) continue;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(candidate, &props);

    int score = 1;
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 2;

    if (score > best_score) {
      best_score = score;
      physical_device_ = candidate;
    }
  }

  if (physical_device_ == VK_NULL_HANDLE) {
    throw std::runtime_error("No Vulkan 1.2 device with a compute queue found");
  }

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  std::cerr << "Using Vulkan device: " << props.deviceName << std::endl;
}

void Context::CreateLogicalDevice() {
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qfs.data());

  queue_family_ = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; ++i) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queue_family_ = i;
      break;
    }
  }
  if (queue_family_ == UINT32_MAX) {
    throw std::runtime_error("No compute queue family found");
  }

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = queue_family_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;

  CheckVk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
          "vkCreateDevice");
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

uint32_t Context::FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

VkCommandBuffer Context::BeginOneShotCommands() const {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer cmd;
  CheckVk(vkAllocateCommandBuffers(device_, &alloc_info, &cmd), "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CheckVk(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");
  return cmd;
}

void Context::EndOneShotCommands(VkCommandBuffer cmd) const {
  CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  CheckVk(vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
  CheckVk(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

}  // namespace apriltag_vulkan::vk
