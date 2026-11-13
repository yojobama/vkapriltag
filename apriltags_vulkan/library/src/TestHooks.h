#pragma once

#include <cstdint>

#include "DecodeSwitches.h"
#include "vkapriltag/TagDecoder.h"

// Entry points to helpers that are otherwise private to a source file, for the unit tests only.
namespace apriltag_vulkan::testing {

uint64_t Rotate90(uint64_t w, int nbits);
bool ErrorsConnected(const apriltag_family_t *f, uint64_t diff);
void BuildThirdBitIndex(const apriltag_family_t *f, ThirdBitIndex *index);
bool MatchThreeTouchingBits(const ThirdBitIndex &index, uint64_t rcode, int *id, int *rotation);
int MaxCorrectedBits(const apriltag_family_t *f);

// The decode steps on one quad (corners in pixel coordinates, top left first, clockwise on screen).
// `outcome` is 0 dropped, 1 stopped by the margin gate, 2 rejected before refining, 3 ran to the end.
struct StepResult {
  int outcome = 0;
  int count = 0;
  int id = -1;
  int hamming = -1;
  double corners[4][2] = {};  // the first detection's corners
};
bool PreRefineRejects(apriltag_detector_t *td, image_u8_t *im, const double corners[4][2], bool reversed_border,
                      const DecodeSwitches &switches);
StepResult RunCandidateSteps(apriltag_detector_t *td, image_u8_t *im, const double corners[4][2],
                             bool reversed_border, const DecodeSwitches &switches);

void IntersectLines(const double lines[4][4], struct quad *quad);
void FitSide(double *bxs, double *bys, double *ws, int used, double range, double *mx, double *my,
             double *normal_theta);

}  // namespace apriltag_vulkan::testing
