//
// Created by john on 1/7/26.
//

#include "ApriltagDetector.h"

#include <ios>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

apriltag::ApriltagDetector::ApriltagDetector(Accelerator accelerator, ImageSize imageSize) {
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

    // todo: check whether this is the correct size of the pixels, or should it be one byte of data per pixel?
    vk::BufferCreateInfo bufferCreateInfo(vk::BufferCreateFlags(), imageSize.width * imageSize.height * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
    outBuffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(outBuffer);

    vk::MemoryAllocateInfo allocInfo(memRequirements.size, static_cast<uint32_t>(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    vk::DeviceMemory outBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(outBuffer, outBufferMemory, 0); // todo: check what the memory offset parameter means and change it if necessary

    // todo: create descriptor sets and pipelines

    const std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };

    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings);
    vk::DescriptorSetLayout descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayout);
    vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
    vk::PipelineCache pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
    // todo: change the entry point of the shaderStageCreateInfo to the correct one (this is just the deafult option)
    vk::PipelineShaderStageCreateInfo shaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, PreProcessionShaderModule, "main");
    vk::ComputePipelineCreateInfo pipelineCreateInfo(vk::PipelineCreateFlags(), shaderStageCreateInfo, pipelineLayout);
    // todo: check the result of the operation and throw a runtime exception if the operation failed
    preProcessPipeline = device.createComputePipeline(pipelineCache, pipelineCreateInfo).value;
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
