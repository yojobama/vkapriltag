#pragma once

#include <vector>

#include "vkapriltag/Detection.h"
#include "vkapriltag/Family.h"

// Looks up a built-in tag family by name (see Family::ByName); false, with a message, if unknown.
bool setup_tag_family(const apriltag_vulkan::Family **tf, const char *famname);
// Releases nothing (the tables are static); clears *tf.
void teardown_tag_family(const apriltag_vulkan::Family **tf, const char *famname);
void print_detections(const std::vector<apriltag_vulkan::Detection> &detections);
