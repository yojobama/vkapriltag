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

}  // namespace

TagDecoder::TagDecoder(apriltag_detector_t *td) : td_(td) {
  poly0_ = g2d_polygon_create_zeros(4);
  poly1_ = g2d_polygon_create_zeros(4);
  detections_ = zarray_create(sizeof(apriltag_detection_t *));
}

TagDecoder::~TagDecoder() {
  ClearDetections(detections_);
  zarray_destroy(detections_);
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

  for (const DetectedQuad &q : quads) {
    struct quad quad_original;
    for (int i = 0; i < 4; ++i) {
      quad_original.p[i][0] = static_cast<float>(q.p[i][0]);
      quad_original.p[i][1] = static_cast<float>(q.p[i][1]);
    }
    quad_original.reversed_border = reversed_border;
    quad_original.H = nullptr;
    quad_original.Hinv = nullptr;

    // quad_decode_index appends any successful decode(s) (one per matching
    // tag family) to detections_; it computes quad->H/Hinv itself.
    quad_decode_index(td_, &quad_original, &im, /*im_samples=*/nullptr, detections_);

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
  }

  reconcile_detections(detections_, poly0_, poly1_);
  zarray_sort(detections_, DetectionCompare);
  return detections_;
}

}  // namespace apriltag_vulkan
