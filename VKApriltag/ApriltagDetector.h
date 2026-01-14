//
// Created by john on 1/7/26.
//

#ifndef VKAPRILTAG_APRILTAGDETECTOR_H
#define VKAPRILTAG_APRILTAGDETECTOR_H

#include "GPU.h"
#include <vulkan/vulkan.hpp>
#include <memory>

namespace apriltag {
    struct ImageSize {
        int height;
        int width;
    };

    // todo: fill this struct with relevant stuff for apriltag detection
    struct ApriltagDetection {

    };

    class ApriltagDetector {
    public:
        ApriltagDetector(GPU accelerator, ImageSize imageSize);
        std::vector<ApriltagDetection> Detect(const char* ImageData);
    private:
        // todo: should these members be unique_ptr's?
        vk::Instance instance;
        vk::PhysicalDevice physicalDevice;
        vk::Device device;
        vk::Buffer outBuffer;
        vk::Pipeline preProcessPipeline;

        // shader moduls
        // todo: add all of the required shader modules
        vk::ShaderModule PreProcessionShaderModule;

        vk::ShaderModule createShaderModule(const std::string &filename) const;
    };
}



#endif //VKAPRILTAG_APRILTAGDETECTOR_H