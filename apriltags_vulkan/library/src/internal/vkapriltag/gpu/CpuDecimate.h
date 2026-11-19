#pragma once

#include <cstdint>

namespace apriltag_vulkan {

// Decimates a tightly packed 8-bit frame by `decimation` into `dst` (width / decimation by
// height / decimation), bit-identical to decimate_box_u8.comp (box) or decimate_u8.comp (top-left).
// `wide` writes one uint32 per pixel (the layout without 8-bit storage), else one byte.
void CpuDecimate(const uint8_t *src, uint32_t width, uint32_t height, uint32_t decimation, bool box, bool wide,
                 void *dst);

}  // namespace apriltag_vulkan
