//
// Created by john on 1/7/26.
//

#include "ApriltagDetector.h"

#include <ios>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

apriltag::ApriltagDetector::ApriltagDetector(Accelerator accelerator) {
    vk::ApplicationInfo appInfo("vulkan-apriltag", 1, nullptr, 0, VK_API_VERSION_1_1);
    // todo: remove all validation layers for release builds
    const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    vk::InstanceCreateInfo instanceInfo(vk::InstanceCreateFlags(), &appInfo, layers);
    instance = vk::createInstance(instanceInfo);
    for (auto &physicalDevice : instance.enumeratePhysicalDevices() ) {
        vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
        if (deviceProperties.deviceName == accelerator.GetName()) {
            this->physicalDevice = physicalDevice;
            break;
        }
    }
    if (!physicalDevice) {throw std::runtime_error("No suitable physical device found");}

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](const auto &queueFamily) {
        return queueFamily.queueFlags & vk::QueueFlagBits::eCompute;
    });
    if (propIt == queueFamilyProperties.end()) {throw std::runtime_error("No graphics queue family found");}
    const uint32_t queueFamilyIndex = std::distance(queueFamilyProperties.begin(), propIt);

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1);
    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo);
    device = physicalDevice.createDevice(deviceCreateInfo);

    PreProcessionShaderModule = createShaderModule("shaders/PreProcess.spv");

    // todo: memory allocation for the detector
}

vk::ShaderModule apriltag::ApriltagDetector::createShaderModule(const std::string &filename) const {
    std::vector<char> contents;
    if (std::ifstream shaderFile{filename, std::ios::binary | std::ios::ate}) {
        const size_t fileSize = shaderFile.tellg();
        shaderFile.seekg(0);
        contents.resize(fileSize, '\0');
        shaderFile.read(contents.data(), fileSize);
    }

    vk::ShaderModuleCreateInfo createInfo(
        vk::ShaderModuleCreateFlags(), contents.size(), reinterpret_cast<const uint32_t *>(contents.data())
    );

    return device.createShaderModule(createInfo);
}
