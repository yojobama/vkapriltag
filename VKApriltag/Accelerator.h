#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace apriltag {
	// struct AcceleratorExtensions {
	// 	bool ExternalMemoryHost;
	// 	bool ByteStorage;
	// 	bool ShaderGroupArithmatic;
	// 	bool ImageCompressionControl;
	// 	bool PushDescriptor;
	// 	bool DescriptorBuffer;
	// 	bool ShaderSubgroupExtendedTypes;
	// 	bool ShaderFloat16Int8;
	// };

	class Accelerator {
	public:
		Accelerator(const std::string& name, std::vector<std::string> supportedExtensions);
		// bool SetAcceleratorExtensions(AcceleratorExtensions acceleratorExtensions);
		// AcceleratorExtensions GetAcceleratorExtensions();
		std::vector<std::string> GetSupportedExtensions();
	private:
		std::string m_Name;
		std::vector<std::string> m_SupportedExtensions;
	};

	std::vector<Accelerator> getAvailableAccelerators(const vk::Instance& instance);
}