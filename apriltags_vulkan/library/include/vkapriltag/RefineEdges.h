#pragma once

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Implementation used for upstream's gradient-based edge refinement.
enum class RefineEdgesMethod {
  // Upstream's arithmetic in double with modf() replaced by trunc()+subtract; bit-identical.
  kExact,
  // Dense edge profiles (every ~1.5 px along a side) with the gradient lobe's centre of mass and a
  // strength-weighted line fit; more accurate corners at several times kExact's cost.
  kProfile,
};

// Resolves the method: an explicit argument wins, then APRILTAG_VK_REFINE ("exact", "profile"),
// then `configured`.
RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured);

// Refines `quad`'s corners in place against the full-resolution image `im`; `td` supplies the decimation.
void RefineEdges(RefineEdgesMethod method, apriltag_detector_t *td, image_u8_t *im,
                 struct quad *quad);

}  // namespace apriltag_vulkan
