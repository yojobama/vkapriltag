#include <string>
#include <vector>

namespace apriltag {
	struct AcceleratorExtensions {
		bool ExternalMemoryHost;
		bool ByteStorage;
		bool ShaderGroupArithmatic;
		bool ImageCompressionControl;
		bool PushDescriptor;
		bool DescriptorBuffer;
		bool ShaderSubgroupExtendedTypes;
		bool ShaderFloat16Int8;
	};

	class Accelerator {
	public:
		Accelerator(const std::string& name, AcceleratorExtensions acceleratorExtensions);
		bool SetAcceleratorExtensions(AcceleratorExtensions acceleratorExtensions);
		AcceleratorExtensions GetAcceleratorExtensions();
	private:
		AcceleratorExtensions m_EnabledExtensions;
		std::string m_Name;
	};

	std::vector<Accelerator> getAvailableAccelerators();
}