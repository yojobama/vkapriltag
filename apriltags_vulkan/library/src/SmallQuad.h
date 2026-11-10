#pragma once

#include <cstdint>

namespace apriltag_vulkan {

// Fits a quad to the dark blob inside a region of a full-resolution grey image: Otsu threshold, the
// largest dark component clear of the region's edge, then the minimum-area rectangle of its hull.
// The region is the box (x0, y0)-(x1, y1) in pixel-index coordinates, grown by a margin. Writes
// four corners, counter-clockwise on screen, and returns false if nothing quad-like is found.
bool FitSmallQuad(const uint8_t *grey, int width, int height, double x0, double y0, double x1,
                  double y1, double out[4][2]);

}  // namespace apriltag_vulkan
