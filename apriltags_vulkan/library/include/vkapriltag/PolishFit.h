#pragma once

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Refits a decoded tag's homography to the whole tag: every cell edge of the known code, blurred by a
// fitted Gaussian per axis, with fitted black, white and outside levels and sensor clipping. Updates
// det->H, det->c and det->p and returns true; returns false, leaving det as it was, if the tag is
// unsupported or the fit is rejected. `im` is the full-resolution image.
bool PolishDetection(const image_u8_t *im, apriltag_detection_t *det);

}  // namespace apriltag_vulkan
