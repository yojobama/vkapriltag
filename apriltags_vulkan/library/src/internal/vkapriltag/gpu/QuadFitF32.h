#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vkapriltag/gpu/DetectorConfig.h"
#include "vkapriltag/gpu/Types.h"

namespace apriltag_vulkan::quadf32 {

// A blob's points as separate arrays, in half-pixel units.
struct Points {
  const int32_t *x, *y, *w;
};

// Central moments of a weighted point set: total weight W, count N, the mean as an offset from the
// blob origin, and the unnormalised second moments about the mean.
struct Centred {
  float W = 0, N = 0, mx = 0, my = 0, sxx = 0, sxy = 0, syy = 0;
};

// A line fitted to a Centred set: sum and mean of the squared error, and the unit normal.
struct LineFit {
  float err, mse, nx, ny;
};

// Moments of the circular, inclusive range [i0, i1] of p[0, n); i0 >= i1 wraps.
Centred RangeMoments(const Points &p, size_t n, size_t i0, size_t i1, int32_t blob_x0, int32_t blob_y0);

// Moments of the points [a0, a1) and [b0, b1) taken together.
Centred SegmentMoments(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t blob_x0,
                       int32_t blob_y0);

// The same without SSE2, for comparison: results are identical.
Centred SegmentMomentsScalar(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t blob_x0,
                             int32_t blob_y0);

// The set of one point with weight w at the offset (x, y).
Centred SinglePoint(float x, float y, float w);

// Parallel-axis merge of two disjoint sets.
Centred Merge(const Centred &a, const Centred &b);

// Line fit of one set, and of `count` sets given as separate arrays.
LineFit FitLine(const Centred &c);
void FitLines(const float *W, const float *N, const float *sxx, const float *sxy, const float *syy, size_t count,
              float *err, float *mse, float *nx, float *ny);

// Window error of every point of a closed contour: the line fit error of the 2k+1 points centred
// on it, wrapping round. Needs k >= 1 and n > 2k + 1.
struct WindowScratch;
void WindowErrors(const Points &p, size_t n, size_t k, WindowScratch &scratch, float *error);

// Point on the line (in half-pixel units) and unit normal of a side's fit.
void LineParams(const Centred &c, int32_t blob_x0, int32_t blob_y0, double out[4]);

// The four sides of an accepted fit.
struct Sides {
  int32_t x0 = 0, y0 = 0;
  Centred side[4];
};

// Reusable buffers of the peaks path.
struct WindowScratch {};
struct Scratch {
  WindowScratch window;
  std::vector<float> error, padded, filtered;
  std::vector<int32_t> mask;
  std::vector<std::pair<float, uint32_t>> peaks;
};

// Fits the four sides between the corner indices idx (circular order). False when a side's mean
// squared error exceeds max_mse.
bool FitArcs(const Points &p, size_t n, const uint32_t idx[4], float max_mse, Sides *out);

// Windowed error, filter, peaks and the search over the strongest ten, for a blob of n points.
// Fills indices and sides; false when no four sides pass the gates.
bool FitPeaks(const DetectorConfig &config, const Points &p, size_t n, Scratch &scratch, uint32_t indices[4],
              Sides *out);

}  // namespace apriltag_vulkan::quadf32
