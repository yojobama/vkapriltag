#include "Accelerator.h"

#include <utility>

apriltag::Accelerator::Accelerator(const std::string &name, std::vector<std::string> supportedExtensions) {
    m_Name = name;
    m_SupportedExtensions = std::move(supportedExtensions);
}

std::vector<std::string> apriltag::Accelerator::GetSupportedExtensions() {
    return m_SupportedExtensions;
}

std::vector<apriltag::Accelerator> apriltag::getAvailableAccelerators(const vk::Instance& instance) {
    std::vector<Accelerator> returnVector;

    for (const vk::PhysicalDevice& device : instance.enumeratePhysicalDevices()) {
        vk::PhysicalDeviceProperties props = device.getProperties();

        std::vector<vk::ExtensionProperties> extensions = device.enumerateDeviceExtensionProperties();

        std::vector<std::string> extensionsInStringFormat;

        for (vk::ExtensionProperties extension : extensions) {
            extensionsInStringFormat.push_back(extension.extensionName);
        }

        returnVector.push_back(Accelerator(props.deviceName, extensionsInStringFormat));
    }

    return returnVector;
}
