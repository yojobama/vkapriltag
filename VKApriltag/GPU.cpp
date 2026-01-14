//
// Created by john on 1/7/26.
//

#include "GPU.h"
#include <vulkan/vulkan.hpp>

namespace apriltag {
    std::vector<GPU> getAvailableAccelerators() {
        std::vector<GPU> gpus;

        vk::ApplicationInfo appInfo("vulkan-apriltag", 1, nullptr, 0, VK_API_VERSION_1_1);
        // todo: remove all validation layers for release builds
        const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
        vk::InstanceCreateInfo instanceInfo(vk::InstanceCreateFlags(), &appInfo, layers);
        vk::Instance instance = vk::createInstance(instanceInfo);
        for (auto device : instance.enumeratePhysicalDevices()) {
            vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
            gpus.emplace_back(deviceProperties.deviceName);
        }
        return gpus;
    }
} // apriltag