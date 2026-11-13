#include "vkapriltag/apriltag_family.h"

#include <iostream>

bool setup_tag_family(const apriltag_vulkan::Family **tf, const char *famname) {
  *tf = apriltag_vulkan::Family::ByName(famname);
  if (*tf == nullptr) {
    std::cerr << "Unknown tag family: " << famname << std::endl;
    return false;
  }
  return true;
}

void teardown_tag_family(const apriltag_vulkan::Family **tf, const char *) { *tf = nullptr; }

void print_detections(const std::vector<apriltag_vulkan::Detection> &detections) {
  for (const apriltag_vulkan::Detection &det : detections) {
    std::cout << "tag #: " << det.id << "  hamming: " << det.hamming << "  margin: " << det.decision_margin
              << "  center: (" << det.c[0] << ", " << det.c[1] << ")" << std::endl;
  }
}
