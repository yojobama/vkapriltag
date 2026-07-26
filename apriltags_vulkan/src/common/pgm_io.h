#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace apriltag_vulkan {

// Loads a binary (P5) grayscale PGM file into a tightly-packed 8-bit buffer.
// No image-library dependency (mirrors main.cpp's DumpPgm writer). Returns
// false (leaving out_pixels/out_width/out_height unchanged) if the file
// cannot be read or isn't a P5 PGM with maxval <= 255.
bool LoadGrayPgm(const std::string &path, std::vector<uint8_t> *out_pixels, uint32_t *out_width,
                 uint32_t *out_height);

}  // namespace apriltag_vulkan
