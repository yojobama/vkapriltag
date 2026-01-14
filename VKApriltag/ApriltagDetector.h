//
// Created by john on 1/7/26.
//

#ifndef VKAPRILTAG_APRILTAGDETECTOR_H
#define VKAPRILTAG_APRILTAGDETECTOR_H

#include "Accelerator.h"
#include <vulkan/vulkan.hpp>

namespace apriltag {
    struct ImageSize {
        int height;
        int width;
    };

    class ApriltagDetector {
    public:
        ApriltagDetector(Accelerator accelerator, ImageSize imageSize);
    private:
        // todo: should these members be unique_ptr's?
        vk::Instance instance;
        vk::PhysicalDevice physicalDevice;
        vk::Device device;

        // shader moduls
        // todo: add all of the required shader modules
        vk::ShaderModule PreProcessionShaderModule;
        // vk::ShaderModule DecimationGrayscaleShaderModule;
        // vk::ShaderModule GaussianBlurShaderModule;
        // vk::ShaderModule AdaptiveThresholdingShaderModule;

        vk::ShaderModule createShaderModule(const std::string &filename) const;
    };
}



#endif //VKAPRILTAG_APRILTAGDETECTOR_H