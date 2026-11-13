#pragma once

#include <cstdint>
#include <memory>
#include <vector>

extern "C" {
#include "apriltag.h"
#include "common/image_u8.h"
#include "common/zarray.h"
}

// The decode of a quad into tags: bit sampling, code lookup, homography and reconcile. It gives the
// results of libapriltag's quad_decode_index and reconcile_detections bit for bit, without their
// per-attempt heap allocations. Derived from AprilRobotics apriltag (BSD-2-Clause); see NOTICE.
namespace apriltag_vulkan::own {

// Maps tag coordinates ([-1, 1] across the border square) to pixels, row-major.
struct Homography {
  double h[9];
};

// The homography of the quad with corners p; false if it is degenerate.
bool ComputeHomography(const float p[4][2], Homography *out);

inline void Project(const double *h, double x, double y, double *ox, double *oy) {
  const double xx = h[0] * x + h[1] * y + h[2];
  const double yy = h[3] * x + h[4] * y + h[5];
  const double zz = h[6] * x + h[7] * y + h[8];
  *ox = xx / zz;
  *oy = yy / zz;
}

// What one family's bit sampling read from a quad.
struct RawSample {
  float margin = -1.0f;  // the decision margin; negative if the quad was rejected
  uint64_t rcode = 0;    // the code word read (0 if rejected)
  bool valid = false;    // set once a family sampled the quad
};

struct DecodedTag {
  const apriltag_family_t *family = nullptr;
  int id = 0;
  int hamming = 0;
  int rotation = 0;
  float margin = 0.0f;
  double h[9] = {};
  double c[2] = {};
  double p[4][2] = {};
};

// One family's sampling grid, border sums and code index, built once.
class FamilyTables;

class Decoder {
 public:
  Decoder();
  ~Decoder();
  Decoder(const Decoder &) = delete;
  Decoder &operator=(const Decoder &) = delete;

  // Rebuilds the tables if the detector's family list changed. False if a family is not supported
  // (the caller then keeps to libapriltag for the whole run).
  bool Refresh(const apriltag_detector_t *td);

  // Samples the quad with the first family of its border polarity (libapriltag's quad_sample_code).
  // Returns false, with the homography invalid, if the quad is degenerate; *h is then unset.
  bool SampleCode(const float p[4][2], bool reversed_border, const image_u8_t *im, Homography *h,
                  RawSample *raw) const;

  // Decodes the quad against every family of its border polarity (libapriltag's quad_decode_index)
  // into out[0, count); `h` receives the quad's homography and `first` the first family's raw sample
  // (either may be null). Returns the count, or -1 if the quad is degenerate.
  static constexpr int kMaxFamilies = 8;
  int Decode(const float p[4][2], bool reversed_border, const image_u8_t *im, Homography *h, RawSample *first,
             DecodedTag *out) const;

  // The smallest Hamming distance from rcode to any code of the first family, over four rotations.
  int MinHamming(uint64_t rcode) const;

  // A new detection (calloc-ed, with a matd homography) that apriltag_detection_destroy frees.
  static apriltag_detection_t *NewDetection(const DecodedTag &tag);

  // Builds a DecodedTag from a decode: homography turned by `rotation`, centre and corners.
  static void FillTag(const apriltag_family_t *family, int id, int hamming, int rotation, float margin,
                      const double *h, DecodedTag *out);

 private:
  double sharpening_ = 0.0;
  std::vector<std::unique_ptr<FamilyTables>> families_;
};

// libapriltag's reconcile_detections: removes the lesser of two overlapping detections of one tag,
// in the same order, from an array of apriltag_detection_t pointers.
void Reconcile(zarray_t *detections);

// Sorts the array by tag id as the detector's output order does.
void SortById(zarray_t *detections);

}  // namespace apriltag_vulkan::own
