#include <vector>
#include <string>

namespace apriltag {
	struct IApriltagFamily {
		std::vector<int> codes;
		std::string name;
	};
}