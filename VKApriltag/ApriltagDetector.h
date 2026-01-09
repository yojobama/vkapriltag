#include "Accelerator.h"
#include <memory>

namespace vk {
	class Device;
	class PhysicalDevice;
	class ShaderModule;
}

namespace apriltag {
	struct ApriltagDetection {
		int id;
		float hammingDistance;
		std::vector<std::pair<float, float>> corners;
	};

	struct DetectorSettings {
		Accelerator accelerator;
	};

	struct Image {
		int width;
		int height;
		std::vector<uint8_t> data;
	};
	
	class ApriltagDetector {
	public:
		ApriltagDetector(const DetectorSettings& settings);
		~ApriltagDetector();
		std::vector<ApriltagDetection> detect(const Image& image);
	private:
		DetectorSettings m_Settings;

		// Vulkan device and physical device
		std::unique_ptr<vk::Device> m_Device;
		std::unique_ptr<vk::PhysicalDevice> m_PhysicalDevice;

		// shader modules for detection
		std::unique_ptr<vk::ShaderModule> m_PreProcessingShaderModule;
		std::unique_ptr<vk::ShaderModule> m_GradientAndEdgeComputingShaderModule;
		std::unique_ptr<vk::ShaderModule> m_SegmentationAndQuadExtractionShaderModule;
	};
};