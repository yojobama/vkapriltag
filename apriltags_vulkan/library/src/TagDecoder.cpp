#include "vkapriltag/TagDecoder.h"

extern "C" {
#include "common/g2d.h"
#include "common/matd.h"

// Exposed as non-static entry points by cmake/patches/apriltag-expose-decode-steps.patch,
// applied to the fetched (unmodified upstream) apriltag library - not exposed via
// any header, forward-declared here exactly as the original CUDA
// apriltag_detect.cu does.
void quad_decode_index(apriltag_detector_t *td, struct quad *quad_original, image_u8_t *im,
                       image_u8_t *im_samples, zarray_t *detections);
void reconcile_detections(zarray_t *detections, zarray_t *poly0, zarray_t *poly1);
}

namespace apriltag_vulkan {
namespace {

int DetectionCompare(const void *_a, const void *_b) {
  apriltag_detection_t *a = *(apriltag_detection_t *const *)_a;
  apriltag_detection_t *b = *(apriltag_detection_t *const *)_b;
  return a->id - b->id;
}

void ClearDetections(zarray_t *detections) {
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    apriltag_detection_destroy(det);
  }
  zarray_truncate(detections, 0);
}

// Empties a per-quad scratch array WITHOUT destroying its elements. Every
// pointer a scratch array holds is also, after the merge step in Decode(),
// held by detections_ - which is the sole owner for destruction purposes.
// Calling ClearDetections (which destroys) on a scratch array would race
// ClearDetections(detections_) over the exact same objects: whichever runs
// second is a double-free.
void ResetScratch(zarray_t *scratch) { zarray_truncate(scratch, 0); }

}  // namespace

TagDecoder::TagDecoder(apriltag_detector_t *td, uint32_t cpu_threads)
    : td_(td), pool_(std::make_unique<WorkerPool>(ResolveThreadCount(cpu_threads))) {
  poly0_ = g2d_polygon_create_zeros(4);
  poly1_ = g2d_polygon_create_zeros(4);
  detections_ = zarray_create(sizeof(apriltag_detection_t *));
}

TagDecoder::~TagDecoder() {
  // detections_ is the sole owner of the detection objects (see ResetScratch's
  // comment) - destroy them here, then only free the per_quad_ arrays'
  // structure, not their (already-freed, aliased) elements.
  ClearDetections(detections_);
  zarray_destroy(detections_);
  for (zarray_t *z : per_quad_) {
    zarray_destroy(z);
  }
  zarray_destroy(poly1_);
  zarray_destroy(poly0_);
}

zarray_t *TagDecoder::Decode(const std::vector<DetectedQuad> &quads, const uint8_t *gray_frame,
                             uint32_t width, uint32_t height, bool reversed_border) {
  ClearDetections(detections_);

  image_u8_t im{
      .width = static_cast<int32_t>(width),
      .height = static_cast<int32_t>(height),
      .stride = static_cast<int32_t>(width),
      .buf = const_cast<uint8_t *>(gray_frame),
  };

  // One scratch zarray per quad (grown, never shrunk - see per_quad_'s
  // comment) so quad_decode_index's tasks share no mutable state except
  // td_->mutex, which it already takes internally around its own append.
  // This is exactly how upstream's own workpool calls quad_decode_index
  // (apriltag.c's quad_decode_task) - see the class comment.
  while (per_quad_.size() < quads.size()) {
    per_quad_.push_back(zarray_create(sizeof(apriltag_detection_t *)));
  }
  // Only over [0, quads.size()): per_quad_ never shrinks, so a frame with
  // fewer quads than some earlier frame leaves entries beyond this range
  // holding that earlier frame's now-destroyed pointers. Touching them here
  // would be a use-after-free; leaving them untouched is fine since the
  // merge loop below is bounded the same way and never looks at them.
  for (size_t i = 0; i < quads.size(); ++i) ResetScratch(per_quad_[i]);

  pool_->ParallelFor(quads.size(), [&](size_t i) {
    const DetectedQuad &q = quads[i];
    struct quad quad_original;
    for (int k = 0; k < 4; ++k) {
      quad_original.p[k][0] = static_cast<float>(q.p[k][0]);
      quad_original.p[k][1] = static_cast<float>(q.p[k][1]);
    }
    quad_original.reversed_border = reversed_border;
    quad_original.H = nullptr;
    quad_original.Hinv = nullptr;

    // quad_decode_index appends any successful decode(s) (one per matching
    // tag family) to per_quad_[i]; it computes quad->H/Hinv itself.
    quad_decode_index(td_, &quad_original, &im, /*im_samples=*/nullptr, per_quad_[i]);

    // ...and leaves both of those allocated on the quad we passed in.
    // Nothing inside frees them: upstream's apriltag_detector_detect keeps
    // its quads in a zarray and destroys them itself once the decode loop is
    // done, so ownership of H/Hinv lands on whoever supplied the quad. Ours
    // is a stack local, so skipping this leaked two matd_t per quad - four
    // allocations, since matd_create() callocs the header and the data
    // separately - every quad of every frame. At ~100 candidate quads and 30
    // fps that is roughly 2 GB/hour in a continuous camera loop, which is
    // exactly this library's intended workload.
    //
    // Safe unconditionally: quad_update_homographies only ever leaves H/Hinv
    // freshly allocated or NULL when they were NULL on entry, which the
    // initialisation above guarantees for every iteration (it is only
    // reusing a quad across calls that can leave Hinv dangling there).
    if (quad_original.H) matd_destroy(quad_original.H);
    if (quad_original.Hinv) matd_destroy(quad_original.Hinv);
  });

  // Merge in quad order, not completion order, so the result - and what
  // reconcile_detections below keeps when two candidates overlap - is
  // bit-identical to the fully serial version regardless of thread count.
  // Bounded to [0, quads.size()), matching the reset loop above - see its
  // comment on why entries beyond that must not be touched.
  for (size_t i = 0; i < quads.size(); ++i) {
    zarray_t *z = per_quad_[i];
    for (int j = 0; j < zarray_size(z); ++j) {
      apriltag_detection_t *det;
      zarray_get(z, j, &det);
      zarray_add(detections_, &det);
    }
  }

  reconcile_detections(detections_, poly0_, poly1_);
  zarray_sort(detections_, DetectionCompare);
  return detections_;
}

}  // namespace apriltag_vulkan
