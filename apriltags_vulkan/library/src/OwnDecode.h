#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "vkapriltag/Detection.h"
#include "vkapriltag/Family.h"
#include "vkapriltag/Types.h"

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
  const Family *family = nullptr;
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

  // Rebuilds the tables if the family list (or a family's max_hamming) changed; `sharpening` is the
  // decode's sharpening weight. False if a family is not supported.
  bool Refresh(std::span<const Family *const> families, double sharpening);

  // Samples the quad with the first family of its border polarity (libapriltag's quad_sample_code).
  // Returns false, with the homography invalid, if the quad is degenerate; *h is then unset.
  bool SampleCode(const float p[4][2], bool reversed_border, const Image *im, Homography *h,
                  RawSample *raw) const;

  // Decodes the quad against every family of its border polarity (libapriltag's quad_decode_index)
  // into out[0, count); `h` receives the quad's homography and `first` the first family's raw sample
  // (either may be null). Returns the count, or -1 if the quad is degenerate.
  static constexpr int kMaxFamilies = 8;
  int Decode(const float p[4][2], bool reversed_border, const Image *im, Homography *h, RawSample *first,
             DecodedTag *out) const;

  // The smallest Hamming distance from rcode to any code of the first family, over four rotations.
  int MinHamming(uint64_t rcode) const;

  static Detection ToDetection(const DecodedTag &tag);

  // Builds a DecodedTag from a decode: homography turned by `rotation`, centre and corners.
  static void FillTag(const Family *family, int id, int hamming, int rotation, float margin,
                      const double *h, DecodedTag *out);

 private:
  double sharpening_ = 0.0;
  std::vector<std::unique_ptr<FamilyTables>> families_;
};

// libapriltag's reconcile_detections: removes the lesser of two overlapping detections of one tag,
// in the same order (a removed entry is replaced by the last one).
void Reconcile(DetectionList *detections);

// Sorts the list by tag id; detections with equal ids keep their order (stable).
void SortById(DetectionList *detections);

}  // namespace apriltag_vulkan::own
