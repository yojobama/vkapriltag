#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace apriltag_vulkan {

// Loads a binary (P5) 8-bit PGM into a tightly packed buffer. Returns false, leaving the outputs
// unchanged, if the file cannot be read or is not a P5 PGM with maxval <= 255.
bool LoadGrayPgm(const std::string &path, std::vector<uint8_t> *out_pixels, uint32_t *out_width,
                 uint32_t *out_height);

}  // namespace apriltag_vulkan
