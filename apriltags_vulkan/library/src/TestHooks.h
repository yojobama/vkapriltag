#pragma once

#include <cstdint>

#include "vkapriltag/TagDecoder.h"

// Entry points to helpers that are otherwise private to a source file, for the unit tests only.
namespace apriltag_vulkan::testing {

uint64_t Rotate90(uint64_t w, int nbits);
bool ErrorsConnected(const apriltag_family_t *f, uint64_t diff);
void BuildThirdBitIndex(const apriltag_family_t *f, ThirdBitIndex *index);
bool MatchThreeTouchingBits(const ThirdBitIndex &index, uint64_t rcode, int *id, int *rotation);
int MaxCorrectedBits(const apriltag_family_t *f);

void IntersectLines(const double lines[4][4], struct quad *quad);
void FitSide(double *bxs, double *bys, double *ws, int used, double range, double *mx, double *my,
             double *normal_theta);

}  // namespace apriltag_vulkan::testing
