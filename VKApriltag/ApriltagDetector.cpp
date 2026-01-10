#include <vulkan/vulkan.hpp>
#include "ApriltagDetector.h"

apriltag::ApriltagDetector::ApriltagDetector(const DetectorSettings& settings) : m_Settings(settings) {
	m_Settings = settings;
}

std::vector<apriltag::ApriltagDetection> apriltag::ApriltagDetector::Detect(const Image &image) {
	std::vector<apriltag::ApriltagDetection> detections;
}
