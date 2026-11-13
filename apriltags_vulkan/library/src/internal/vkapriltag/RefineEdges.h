#pragma once

#include "vkapriltag/Types.h"

namespace apriltag_vulkan {

// Implementation used for upstream's gradient-based edge refinement.
enum class RefineEdgesMethod {
  // Upstream's arithmetic in double with modf() replaced by trunc()+subtract; bit-identical.
  kExact,
  // Dense edge profiles (every ~1.5 px along a side) with the gradient lobe's centre of mass and a
  // strength-weighted line fit; more accurate corners at several times kExact's cost.
  kProfile,
};

// Resolves the method: APRILTAG_VK_REFINE ("exact" or "profile") overrides `configured`.
RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured);

// Refines `quad`'s corners in place against the full-resolution image `im`. `decimation` is the detector's
// (it sets the search radius); `width_at_border` is the tag family's border width in cells, 0 if unknown.
void RefineEdges(RefineEdgesMethod method, int decimation, int width_at_border, const Image *im, Quad *quad);

}  // namespace apriltag_vulkan
