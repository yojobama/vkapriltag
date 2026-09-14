#pragma once

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Selects which implementation of upstream's gradient-based edge refinement
// TagDecoder runs per quad. Profiling on an RX 9060 XT put upstream's
// refine_edges() at ~55% of all CPU cycles in the whole pipeline, with a
// further ~16% inside the glibc modf() it calls twice per interpolation step -
// making it the single most expensive stage, GPU phases included.
enum class RefineEdgesMethod {
  // Call upstream's compiled refine_edges() (apriltag.c). The reference.
  kUpstream,
  // Same arithmetic as upstream, in the same double precision, with modf()
  // replaced by trunc()+subtract. `modf(x, &i)` is defined to return x -
  // trunc(x) and both results are exactly representable, so this is
  // bit-identical to kUpstream by construction - no accuracy argument
  // required - while removing a non-inlinable libm call (which also stores
  // through a pointer, blocking vectorization) from the innermost loop.
  kExact,
  // kExact, but the innermost sampling loop - coordinate stepping, the two
  // bilinear interpolations and the gradient weighting - runs in float.
  //
  // The accumulators that actually need the range stay double: the per-edge
  // line fit builds a covariance from raw second moments (Cxx = Mxx/N - Ex*Ex)
  // at image coordinates, where Mxx reaches ~1e7 at 1080p and the subtraction
  // is a near-total cancellation. In float that loses most of the significant
  // digits of the variance. Only the inner loop - whose inputs are 8-bit pixels
  // and whose output is a single ratio Mn/Mcount - is narrowed.
  kFast,
};

// Resolves the method: an explicit argument wins, otherwise APRILTAG_VK_REFINE
// ("upstream", "exact", "fast"), otherwise `configured`.
RefineEdgesMethod ResolveRefineEdgesMethod(RefineEdgesMethod configured);

// Refines `quad`'s four corners in place against the full-resolution image
// `im`, using `quad_decimate` for the per-edge search radius exactly as
// upstream does. `td` is used only to reach upstream's implementation for
// kUpstream.
void RefineEdges(RefineEdgesMethod method, apriltag_detector_t *td, image_u8_t *im,
                 struct quad *quad);

}  // namespace apriltag_vulkan
