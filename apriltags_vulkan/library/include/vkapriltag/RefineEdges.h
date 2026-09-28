#pragma once

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Implementation used for upstream's gradient-based edge refinement.
enum class RefineEdgesMethod {
  // Upstream's compiled refine_edges() (apriltag.c); the reference.
  kUpstream,
  // Upstream's arithmetic in double with modf() replaced by trunc()+subtract; bit-identical.
  kExact,
  // As kExact, with the inner sampling loop in float; the line-fit accumulators stay double.
  kFast,
};

// Resolves the method: an explicit argument wins, then APRILTAG_VK_REFINE ("upstream", "exact",
// "fast"), then `configured`.
RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured);

// Refines `quad`'s corners in place against the full-resolution image `im`; `td` is used only for
// kUpstream.
void RefineEdges(RefineEdgesMethod method, apriltag_detector_t *td, image_u8_t *im,
                 struct quad *quad);

}  // namespace apriltag_vulkan
