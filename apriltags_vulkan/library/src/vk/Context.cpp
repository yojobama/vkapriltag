#include "vkapriltag/vk/Context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace apriltag_vulkan::vk {

namespace {

const char *VkResultName(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    default: return "VK_ERROR_<other>";
  }
}

const char *DeviceTypeName(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU (software)";
    default: return "other";
  }
}

bool EnvFlag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool EnvInt(const char *name, int *out) {
  const char *v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return false;
  *out = std::atoi(v);
  return true;
}

// Largest power of two <= v (v >= 1).
uint32_t FloorPow2(uint32_t v) {
  uint32_t r = 1;
  while ((r << 1) != 0 && (r << 1) <= v) r <<= 1;
  return r;
}

// Devices are ranked so that a software implementation can never silently
// outrank real hardware.
int ScoreDeviceType(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
    default: return 1;
  }
}

bool HasComputeQueue(VkPhysicalDevice dev) {
  uint32_t n = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(n);
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, qfs.data());
  for (const auto &qf : qfs) {
    if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) return true;
  }
  return false;
}

}  // namespace

void CheckVk(VkResult result, const char *what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed: " + VkResultName(result) + " (" +
                             std::to_string(static_cast<int>(result)) + ")");
  }
}

Context::Context(const ContextOptions &options_in) {
  ContextOptions options = options_in;
  // Environment overrides, so the same binary can be retargeted at runtime.
  if (EnvFlag("APRILTAG_VK_ALLOW_CPU")) options.allow_cpu_device = true;
  if (EnvFlag("APRILTAG_VK_VALIDATION")) options.enable_validation = true;
  int env_int = 0;
  if (EnvInt("APRILTAG_VK_DEVICE", &env_int)) options.device_index = env_int;
  if (EnvInt("APRILTAG_VK_WG", &env_int) && env_int > 0) {
    options.workgroup_size_override = static_cast<uint32_t>(env_int);
  }
  if (EnvInt("APRILTAG_VK_MAX_INVOCATIONS", &env_int) && env_int > 0) {
    options.max_invocations_override = static_cast<uint32_t>(env_int);
  }

  CreateInstance(options);
  SelectPhysicalDevice(options);
  CreateLogicalDevice();
  QueryCaps(options);
  CreateCommandResources();

  if (options.verbose) {
    std::cerr << DescribeDevice() << std::endl;
  }
}

Context::~Context() {
  if (device_) {
    // Nothing may still be in flight when we start destroying pools.
    vkDeviceWaitIdle(device_);
    for (size_t i = 0; i < kCommandRing; ++i) {
      if (fence_ring_[i]) vkDestroyFence(device_, fence_ring_[i], nullptr);
    }
    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Context::CreateInstance(const ContextOptions &options) {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "apriltag_vulkan";
  // Nothing in this pipeline uses a 1.2 core entry point, so asking for 1.1
  // keeps older mobile drivers (and older Panfrost) eligible.
  app_info.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char *> layers;
  if (options.enable_validation) {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available.data());
    const char *kValidation = "VK_LAYER_KHRONOS_validation";
    for (const auto &l : available) {
      if (std::strcmp(l.layerName, kValidation) == 0) {
        layers.push_back(kValidation);
        break;
      }
    }
    if (layers.empty()) {
      std::cerr << "apriltag_vulkan: APRILTAG_VK_VALIDATION set but "
                   "VK_LAYER_KHRONOS_validation is not installed; continuing without it."
                << std::endl;
    }
  }

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

  CheckVk(vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");
}

void Context::SelectPhysicalDevice(const ContextOptions &options) {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (count == 0) {
    throw std::runtime_error(
        "No Vulkan physical devices found. No usable Vulkan driver (ICD) is installed - "
        "check that a *_icd.json for your GPU exists under /usr/share/vulkan/icd.d and that "
        "vulkaninfo --summary lists your device.");
  }
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance_, &count, devices.data());

  // Gather properties once, for both selection and the diagnostic message.
  std::vector<VkPhysicalDeviceProperties> props(count);
  for (uint32_t i = 0; i < count; ++i) {
    vkGetPhysicalDeviceProperties(devices[i], &props[i]);
  }

  auto describe_all = [&]() {
    std::ostringstream os;
    for (uint32_t i = 0; i < count; ++i) {
      os << "\n  [" << i << "] " << props[i].deviceName << " ("
         << DeviceTypeName(props[i].deviceType) << ", Vulkan "
         << VK_API_VERSION_MAJOR(props[i].apiVersion) << "."
         << VK_API_VERSION_MINOR(props[i].apiVersion) << ")";
    }
    return os.str();
  };

  // Explicit index wins, but still has to be usable.
  if (options.device_index >= 0) {
    if (static_cast<uint32_t>(options.device_index) >= count) {
      throw std::runtime_error("APRILTAG_VK_DEVICE=" + std::to_string(options.device_index) +
                               " is out of range; devices are:" + describe_all());
    }
    uint32_t i = static_cast<uint32_t>(options.device_index);
    if (!HasComputeQueue(devices[i])) {
      throw std::runtime_error(std::string("Requested device '") + props[i].deviceName +
                               "' has no compute queue family.");
    }
    physical_device_ = devices[i];
    return;
  }

  int best_score = -1;
  bool saw_cpu_only_candidate = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (!HasComputeQueue(devices[i])) continue;
    if (props[i].apiVersion < VK_API_VERSION_1_1) continue;

    const bool is_cpu = props[i].deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    if (is_cpu && !options.allow_cpu_device) {
      saw_cpu_only_candidate = true;
      continue;
    }

    int score = ScoreDeviceType(props[i].deviceType);
    if (score > best_score) {
      best_score = score;
      physical_device_ = devices[i];
    }
  }

  if (physical_device_ == VK_NULL_HANDLE) {
    if (saw_cpu_only_candidate) {
      // This is the case that silently destroys performance, so say exactly
      // what happened and exactly how to proceed.
      throw std::runtime_error(
          "The only Vulkan device available is a CPU/software implementation (e.g. Mesa "
          "lavapipe/llvmpipe). Running this detector there is typically 50-500x slower than "
          "real hardware, so it is refused by default.\n"
          "  * To fix properly: install the Vulkan driver (ICD) for your GPU. On WSL2 with an "
          "NVIDIA card this means the driver's nvidia_icd.json; on Linux with AMD/Intel it "
          "means mesa-vulkan-drivers plus a working /dev/dri. Verify with "
          "'vulkaninfo --summary'.\n"
          "  * To benchmark or test on the CPU anyway: set APRILTAG_VK_ALLOW_CPU=1.\n"
          "Devices seen:" +
          describe_all());
    }
    throw std::runtime_error("No Vulkan 1.1+ device with a compute queue found. Devices seen:" +
                             describe_all());
  }
}

void Context::CreateLogicalDevice() {
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qfs.data());

  // A COMPUTE-capable family implicitly supports transfer operations, which is
  // all this pipeline needs (dispatch, copy, fill).
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

  // Deliberately enable *no* optional features: every shader here is written
  // against core Vulkan 1.1 compute so that Mali/Adreno parts lacking
  // shaderFloat64 / shaderInt64 / 8-bit storage still work.
  VkPhysicalDeviceFeatures enabled_features{};

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.pEnabledFeatures = &enabled_features;

  CheckVk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_), "vkCreateDevice");
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

void Context::QueryCaps(const ContextOptions &options) {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_device_, &features);
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props_);

  caps_.name = props.deviceName;
  caps_.type = props.deviceType;
  caps_.api_version = props.apiVersion;
  caps_.vendor_id = props.vendorID;

  const VkPhysicalDeviceLimits &l = props.limits;
  caps_.max_workgroup_invocations = l.maxComputeWorkGroupInvocations;
  for (int i = 0; i < 3; ++i) {
    caps_.max_workgroup_size[i] = l.maxComputeWorkGroupSize[i];
    caps_.max_workgroup_count[i] = l.maxComputeWorkGroupCount[i];
  }
  caps_.max_shared_memory_bytes = l.maxComputeSharedMemorySize;

  // Testing aid: simulate a more constrained device so the launch geometry a
  // Mali-class part would receive can be exercised on desktop hardware.
  if (options.max_invocations_override > 0) {
    caps_.max_workgroup_invocations =
        std::min(caps_.max_workgroup_invocations, options.max_invocations_override);
    caps_.max_workgroup_size[0] =
        std::min(caps_.max_workgroup_size[0], options.max_invocations_override);
  }
  caps_.has_shader_float64 = features.shaderFloat64 == VK_TRUE;
  caps_.has_shader_int64 = features.shaderInt64 == VK_TRUE;

  // Timestamp support: needs a non-zero period and a queue family that can
  // actually write timestamps.
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qfs.data());
  const uint32_t valid_bits =
      (queue_family_ < qf_count) ? qfs[queue_family_].timestampValidBits : 0;
  caps_.timestamp_period_ns = l.timestampPeriod;
  caps_.timestamps_supported = valid_bits > 0 && l.timestampPeriod > 0.0f;

  // Memory topology.
  caps_.unified_memory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                         props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
  caps_.has_host_cached = false;
  for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags f = mem_props_.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
      caps_.has_host_cached = true;
    }
  }

  // --- Derived launch geometry ---
  // 1D: 256 is a good default on desktop; Mali-class parts do better with
  // 128, and the hard ceiling is whatever the device reports.
  uint32_t want_1d = (caps_.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 128u : 256u;
  if (options.workgroup_size_override > 0) want_1d = options.workgroup_size_override;
  want_1d = std::min(want_1d, caps_.max_workgroup_invocations);
  want_1d = std::min(want_1d, caps_.max_workgroup_size[0]);
  // bitonic_local.comp stages three uint32 arrays of one element per
  // invocation in shared memory, so the workgroup must also fit that budget.
  want_1d = std::min(want_1d, std::max<uint32_t>(caps_.max_shared_memory_bytes / 12u, 1u));
  caps_.wg1d = std::max(FloorPow2(want_1d), 1u);

  // 2D: prefer 16x16, fall back to 8x8 on parts that cannot host 256
  // invocations per group.
  if (caps_.max_workgroup_invocations >= 256 && caps_.max_workgroup_size[0] >= 16 &&
      caps_.max_workgroup_size[1] >= 16) {
    caps_.wg2d_x = 16;
    caps_.wg2d_y = 16;
  } else {
    caps_.wg2d_x = 8;
    caps_.wg2d_y = 8;
  }

  // Scan: a bigger block means fewer scan levels, but it must fit both the
  // invocation limit and the shared-memory budget (4 bytes per element).
  uint32_t want_scan = std::min<uint32_t>(1024, caps_.max_workgroup_invocations);
  want_scan = std::min(want_scan, caps_.max_workgroup_size[0]);
  want_scan = std::min(want_scan, std::max<uint32_t>(caps_.max_shared_memory_bytes / 4u, 1u));
  caps_.scan_wg = std::max(FloorPow2(want_scan), 1u);
}

std::vector<DeviceCaps> Context::EnumerateDevices() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "apriltag_vulkan";
    // Nothing in this pipeline uses a 1.2 core entry point, so asking for 1.1
    // keeps older mobile drivers (and older Panfrost) eligible.
    app_info.apiVersion = VK_API_VERSION_1_1;
    
	// for this wee session to enumerate devices there is no need not to have the validation layer enabled, so we will enable it if it is available
    std::vector<const char*> layers;
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available.data());
    const char* kValidation = "VK_LAYER_KHRONOS_validation";
    for (const auto& l : available) {
        if (std::strcmp(l.layerName, kValidation) == 0) {
            layers.push_back(kValidation);
            break;
        }
    }
    if (layers.empty()) {
        std::cerr << "apriltag_vulkan: VK_LAYER_KHRONOS_validation is not installed; continuing without it." << std::endl;
    }

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

    VkInstance instance;
    CheckVk(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");

    
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error(
            "No Vulkan physical devices found. No usable Vulkan driver (ICD) is installed - "
            "check that a *_icd.json for your GPU exists under /usr/share/vulkan/icd.d and that "
            "vulkaninfo --summary lists your device.");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    // Gather properties once, for both selection and the diagnostic message.
    std::vector<VkPhysicalDeviceProperties> props(count);
    for (uint32_t i = 0; i < count; ++i) {
        vkGetPhysicalDeviceProperties(devices[i], &props[i]);
    }

	std::vector<DeviceCaps> caps_list;
    caps_list.reserve(count);

    for (const VkPhysicalDevice& device : devices) {
		DeviceCaps caps;
		VkPhysicalDeviceProperties device_props;
        vkGetPhysicalDeviceProperties(device, &device_props);
    
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(device, &features);
        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(device, &mem_props);

        caps.name = props.deviceName;
        caps.type = props.deviceType;
        caps.api_version = props.apiVersion;
        caps.vendor_id = props.vendorID;

        const VkPhysicalDeviceLimits& l = props.limits;
        caps.max_workgroup_invocations = l.maxComputeWorkGroupInvocations;
        for (int i = 0; i < 3; ++i) {
            caps.max_workgroup_size[i] = l.maxComputeWorkGroupSize[i];
            caps.max_workgroup_count[i] = l.maxComputeWorkGroupCount[i];
        }
        caps.max_shared_memory_bytes = l.maxComputeSharedMemorySize;

        // Testing aid: simulate a more constrained device so the launch geometry a
        // Mali-class part would receive can be exercised on desktop hardware.
        //if (options.max_invocations_override > 0) {
        //    caps.max_workgroup_invocations =
        //        std::min(caps.max_workgroup_invocations, options.max_invocations_override);
        //    caps.max_workgroup_size[0] =
        //        std::min(caps.max_workgroup_size[0], options.max_invocations_override);
        //}

        caps.has_shader_float64 = features.shaderFloat64 == VK_TRUE;
        caps.has_shader_int64 = features.shaderInt64 == VK_TRUE;

        // Timestamp support: needs a non-zero period and a queue family that can
        // actually write timestamps.
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &qf_count, qfs.data());

        int queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < qf_count; ++i) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == UINT32_MAX) {
			std::cerr << "Device " << props.deviceName << " has no compute queue family; skipping." << std::endl;
            continue;
        }

        const uint32_t valid_bits =
            (queue_family < qf_count) ? qfs[queue_family].timestampValidBits : 0;
        caps.timestamp_period_ns = l.timestampPeriod;
        caps.timestamps_supported = valid_bits > 0 && l.timestampPeriod > 0.0f;

        // Memory topology.
        caps.unified_memory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        caps.has_host_cached = false;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags f = mem_props.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                caps.has_host_cached = true;
            }
        }

        // --- Derived launch geometry ---
        // 1D: 256 is a good default on desktop; Mali-class parts do better with
        // 128, and the hard ceiling is whatever the device reports.
        uint32_t want_1d = (caps.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 128u : 256u;
        //if (options.workgroup_size_override > 0) want_1d = options.workgroup_size_override;
        want_1d = std::min(want_1d, caps.max_workgroup_invocations);
        want_1d = std::min(want_1d, caps.max_workgroup_size[0]);
        // bitonic_local.comp stages three uint32 arrays of one element per
        // invocation in shared memory, so the workgroup must also fit that budget.
        want_1d = std::min(want_1d, std::max<uint32_t>(caps.max_shared_memory_bytes / 12u, 1u));
        caps.wg1d = std::max(FloorPow2(want_1d), 1u);

        // 2D: prefer 16x16, fall back to 8x8 on parts that cannot host 256
        // invocations per group.
        if (caps.max_workgroup_invocations >= 256 && caps.max_workgroup_size[0] >= 16 &&
            caps.max_workgroup_size[1] >= 16) {
            caps.wg2d_x = 16;
            caps.wg2d_y = 16;
        }
        else {
            caps.wg2d_x = 8;
            caps.wg2d_y = 8;
        }

        // Scan: a bigger block means fewer scan levels, but it must fit both the
        // invocation limit and the shared-memory budget (4 bytes per element).
        uint32_t want_scan = std::min<uint32_t>(1024, caps.max_workgroup_invocations);
        want_scan = std::min(want_scan, caps.max_workgroup_size[0]);
        want_scan = std::min(want_scan, std::max<uint32_t>(caps.max_shared_memory_bytes / 4u, 1u));
        caps.scan_wg = std::max(FloorPow2(want_scan), 1u);
    
		caps_list.push_back(caps);
    }

    return caps_list;
}

void Context::CreateCommandResources() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_;
  CheckVk(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
          "vkCreateCommandPool");

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = static_cast<uint32_t>(kCommandRing);
  CheckVk(vkAllocateCommandBuffers(device_, &alloc_info, cmd_ring_),
          "vkAllocateCommandBuffers");

  // Created signaled so the first BeginCommands() wait is a no-op.
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < kCommandRing; ++i) {
    CheckVk(vkCreateFence(device_, &fence_info, nullptr, &fence_ring_[i]), "vkCreateFence");
  }
}

uint32_t Context::FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred) const {
  // Two passes: an exact match on required|preferred first, then required only.
  for (int pass = 0; pass < 2; ++pass) {
    const VkMemoryPropertyFlags want = (pass == 0) ? (required | preferred) : required;
    if (pass == 0 && preferred == 0) continue;
    for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) == 0) continue;
      if ((mem_props_.memoryTypes[i].propertyFlags & want) == want) return i;
    }
  }
  return UINT32_MAX;
}

uint32_t Context::FindMemoryTypeOrThrow(uint32_t type_bits, VkMemoryPropertyFlags required,
                                        VkMemoryPropertyFlags preferred) const {
  uint32_t index = FindMemoryType(type_bits, required, preferred);
  if (index == UINT32_MAX) {
    throw std::runtime_error("No Vulkan memory type satisfies the required properties (0x" +
                             std::to_string(required) + ")");
  }
  return index;
}

VkCommandBuffer Context::BeginCommands() const {
  const size_t i = ring_next_ % kCommandRing;
  ring_next_ = (ring_next_ + 1) % kCommandRing;
  ring_active_ = i;

  // Wait for this slot's previous submission (if any) to retire before reusing
  // the command buffer.
  CheckVk(vkWaitForFences(device_, 1, &fence_ring_[i], VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
  CheckVk(vkResetFences(device_, 1, &fence_ring_[i]), "vkResetFences");
  CheckVk(vkResetCommandBuffer(cmd_ring_[i], 0), "vkResetCommandBuffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CheckVk(vkBeginCommandBuffer(cmd_ring_[i], &begin_info), "vkBeginCommandBuffer");
  return cmd_ring_[i];
}

void Context::SubmitAndWait(VkCommandBuffer cmd) const {
  const size_t i = ring_active_;
  CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  CheckVk(vkQueueSubmit(queue_, 1, &submit_info, fence_ring_[i]), "vkQueueSubmit");
  ++submit_count;
  CheckVk(vkWaitForFences(device_, 1, &fence_ring_[i], VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
}

std::string Context::DescribeDevice() const {
  std::ostringstream os;
  os << "apriltag_vulkan: using " << caps_.name << " (" << DeviceTypeName(caps_.type)
     << ", Vulkan " << VK_API_VERSION_MAJOR(caps_.api_version) << "."
     << VK_API_VERSION_MINOR(caps_.api_version) << "."
     << VK_API_VERSION_PATCH(caps_.api_version) << ")";
  if (caps_.is_cpu_device()) {
    os << "\n  WARNING: this is a SOFTWARE Vulkan implementation running on the CPU. "
          "Timings here say nothing about GPU performance.";
  }
  os << "\n  limits: maxComputeWorkGroupInvocations=" << caps_.max_workgroup_invocations
     << ", maxComputeWorkGroupSize.x=" << caps_.max_workgroup_size[0]
     << ", maxComputeSharedMemorySize=" << caps_.max_shared_memory_bytes
     << ", float64=" << (caps_.has_shader_float64 ? "yes" : "no")
     << ", int64=" << (caps_.has_shader_int64 ? "yes" : "no");
  os << "\n  chosen geometry: wg1d=" << caps_.wg1d << ", wg2d=" << caps_.wg2d_x << "x"
     << caps_.wg2d_y << ", scan_wg=" << caps_.scan_wg
     << ", unified_memory=" << (caps_.unified_memory ? "yes" : "no")
     << ", host_cached=" << (caps_.has_host_cached ? "yes" : "no")
     << ", timestamps=" << (caps_.timestamps_supported ? "yes" : "no");
  return os.str();
}

std::string Context::DescribeDevice(const DeviceCaps &caps) {
    std::ostringstream os;
    os << "apriltag_vulkan: using " << caps.name << " (" << DeviceTypeName(caps.type)
        << ", Vulkan " << VK_API_VERSION_MAJOR(caps.api_version) << "."
        << VK_API_VERSION_MINOR(caps.api_version) << "."
        << VK_API_VERSION_PATCH(caps.api_version) << ")";
    if (caps.is_cpu_device()) {
        os << "\n  WARNING: this is a SOFTWARE Vulkan implementation running on the CPU. "
            "Timings here say nothing about GPU performance.";
    }
    os << "\n  limits: maxComputeWorkGroupInvocations=" << caps.max_workgroup_invocations
        << ", maxComputeWorkGroupSize.x=" << caps.max_workgroup_size[0]
        << ", maxComputeSharedMemorySize=" << caps.max_shared_memory_bytes
        << ", float64=" << (caps.has_shader_float64 ? "yes" : "no")
        << ", int64=" << (caps.has_shader_int64 ? "yes" : "no");
    os << "\n  chosen geometry: wg1d=" << caps.wg1d << ", wg2d=" << caps.wg2d_x << "x"
        << caps.wg2d_y << ", scan_wg=" << caps.scan_wg
        << ", unified_memory=" << (caps.unified_memory ? "yes" : "no")
        << ", host_cached=" << (caps.has_host_cached ? "yes" : "no")
        << ", timestamps=" << (caps.timestamps_supported ? "yes" : "no");
    return os.str();
}

}  // namespace apriltag_vulkan::vk
