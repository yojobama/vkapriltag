#pragma once

#include "vkapriltag/Types.h"

namespace apriltag_vulkan {

// Grey levels read around a tag's border, from the row-major homography H (9 values) of its black square
// (tag coordinates [-1, 1] across the W-cell square).
struct BorderLevels {
  double black = 0;  // median of the border cells
  double white = 0;  // median of the quiet ring half a cell outside the border
  int dark = 0;      // border cells darker than the midpoint
  int cells = 0;     // border cells read (4 * (W - 1) when all fall inside the image)
  // Each side's non-corner border cells, from the tag's black (0) to its white (1): near 0 for a
  // side fitted on the border, near 1 when it was fitted a cell too far out.
  double side_level[4] = {0, 0, 0, 0};
};

// Samples the border cells and the quiet ring. Returns false if any sample falls outside the image
// or the quiet ring is not brighter than the border.
bool ReadBorder(const double *H, const Image *im, int width_cells, BorderLevels *out);

}  // namespace apriltag_vulkan
