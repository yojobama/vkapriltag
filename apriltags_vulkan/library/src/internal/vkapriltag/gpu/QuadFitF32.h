#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "vkapriltag/gpu/DetectorConfig.h"
#include "vkapriltag/gpu/Types.h"

namespace apriltag_vulkan::quadf32 {

// A blob's points as separate arrays, in half-pixel units. Each array needs 7 readable elements past its end.
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

// Splits n raw points into the three arrays.
void Unpack(const RawLineFitPoint *pts, size_t n, int32_t *x, int32_t *y, int32_t *w);

// As SegmentMoments about the given origin, which need not be a point of the range; zero weights drop points.
Centred SegmentMomentsAbout(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t ox, int32_t oy,
                            int32_t blob_x0, int32_t blob_y0);
Centred SegmentMomentsAboutScalar(const Points &p, size_t a0, size_t a1, size_t b0, size_t b1, int32_t ox, int32_t oy,
                                  int32_t blob_x0, int32_t blob_y0);

// Bounding box {lo x, hi x, lo y, hi y} of n points and, when keys is not null, their packed keys x << 14 | y.
void BoundsAndKeys(const Points &p, size_t n, int32_t bounds[4], uint32_t *keys);

// A hull edge: its start (a point of the blob's grid), unit direction, length and the half width of the
// strip searched for support, all in half-pixel units.
struct HullEdge {
  int32_t p0x, p0y;
  double ux, uy, len, window;
};

// Buffers of the hull side fit.
struct HullScratch {
  std::vector<int32_t> mask, wm, idx, hist;
  std::vector<float> nx, ny;
};

// Fits one hull side: the line of greatest support near the edge, then the moments of the points within
// band of it. False when fewer than 6 points are inside the band. simd false runs the portable loops.
bool FitHullSide(const Points &p, size_t n, int32_t blob_x0, int32_t blob_y0, const HullEdge &e, double band,
                 HullScratch &scratch, Centred *out, bool simd = true);

// Buffers of the hull geometry.
struct HullGeom {
  std::vector<int32_t> colmin, colmax, hx, hy;
  std::vector<char> keep;
};

// The points that can be vertices of the convex hull: the lowest and the highest point of every column
// x in [lo_x, hi_x], in (x, y) order, without duplicates.
void ColumnCandidates(const Points &p, size_t n, int32_t lo_x, int32_t hi_x, HullGeom &g,
                      std::vector<std::pair<int32_t, int32_t>> &out);

// The first vertex 1..m-1 of the largest squared distance from vertex 0.
size_t FarVertex(const int32_t *hx, const int32_t *hy, size_t m);

// Douglas-Peucker over the closed polygon (hx, hy), whose vertex 0 is repeated at index m, between vertices
// i and j: marks the vertices of the simplification in keep.
void SimplifyHull(const int32_t *hx, const int32_t *hy, size_t i, size_t j, double eps, char *keep);

// The four edges of the k <= 16 polygon edges (vertex v[e], direction (ux, uy), length len) in order of
// greatest total length whose neighbours are near-perpendicular and whose intersections lie close to the
// edges' ends. Returns the total length (0 when none) and fills best.
double ChooseEdges(const std::pair<double, double> *v, size_t k, const double *len, const double *ux, const double *uy,
                   size_t best[4]);

// Whether more than `fraction` of the n points are farther than `tolerance` from every side of the quad
// whose corners are in half-pixel units.
bool HasOutliers(const Points &p, size_t n, const double corners[4][2], double tolerance, double fraction,
                 bool simd = true);

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
void WindowErrors(const Points &p, size_t n, size_t k, float *error);

// Point on the line (in half-pixel units) and unit normal of a side's fit.
void LineParams(const Centred &c, int32_t blob_x0, int32_t blob_y0, double out[4]);

// The four sides of an accepted fit.
struct Sides {
  int32_t x0 = 0, y0 = 0;
  Centred side[4];
};

// Reusable buffers of the peaks path.
struct Scratch {
  HullScratch hull;
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
