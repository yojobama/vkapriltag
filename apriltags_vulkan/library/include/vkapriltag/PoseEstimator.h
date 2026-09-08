#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vkapriltag/common/WorkerPool.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Camera intrinsics for pose estimation, in pixels. Assumed pinhole: neither
// this library nor the CUDA original it is ported from implements lens
// undistortion, so a caller with a wide-angle lens should undistort the
// detection's corners before estimating pose. On such a lens that matters far
// more than any numerical detail in the solver below.
struct CameraIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

// The tag's pose in the camera optical frame: `t` is the tag's position,
// `R` its orientation. Same convention as libapriltag's apriltag_pose_t,
// which this is verified against - see tools/validate_pose.
//
// `error` is the object-space error of the fit (Lu 2000), the same scalar
// libapriltag's estimate_tag_pose returns, and is what selects between the
// two candidate solutions of the planar-pose ambiguity. `valid` is false
// when no pose could be computed at all (a degenerate homography, or a
// solution the ambiguity search legitimately declined to produce).
struct TagPose {
  double R[3][3] = {};
  double t[3] = {};
  double error = 0.0;
  bool valid = false;
};

// Both candidate solutions of the planar-pose ambiguity, before the caller
// (or Estimate) picks the lower-error one. Mirrors what libapriltag's
// estimate_tag_pose_orthogonal_iteration produces, so each half can be
// verified against it independently rather than only the final answer.
struct TagPosePair {
  TagPose solution1;  // orthogonal iteration from the homography seed
  TagPose solution2;  // the second local minimum, if one exists
};

// Tag pose from four detected corners, ported from libapriltag's
// apriltag_pose.c: a homography-based seed (Olson 2011) refined by
// orthogonal iteration (Lu/Hager/Mjolsness 2000), plus a search for the
// second local minimum of the planar-pose ambiguity (Schweighofer/Pinz 2006)
// with the lower-error solution winning.
//
// Same algorithm and same iteration counts as libapriltag, but with
// stack-allocated fixed-size 3x3/3x1 arithmetic instead of matd_t. That is
// the entire reason this exists: libapriltag's solver is built on matd_op(),
// a RUNTIME STRING-EXPRESSION INTERPRETER that per call scans the expression,
// heap-allocates an argument array plus a 2*exprlen garbage array, recursively
// parses it character by character allocating intermediates, copies the
// result, then frees everything. The solver issues ~16 of those per iteration
// across 100 iterations - roughly 1600 interpreted expression evaluations per
// pose - so essentially all of its cost is interpreter and allocator
// overhead, not the ~100k FLOPs of actual arithmetic. Measured on
// Mali-G610/RK3588: 1.309 ms per tag for libapriltag against 0.0175 ms for a
// single equivalent 50-iteration solve done this way.
//
// Deliberately CPU, not GPU. The arithmetic is ~1 us of work on this GPU but
// would need its own queue submission to reach it, and a submit boundary
// measures ~0.46 ms on this hardware (~0.19 ms GPU-clock plus ~0.27 ms
// host-side fence/readback). The 50 iterations are also strictly sequential -
// each consumes the previous R and t - and there are only 4 points to spread
// across a subgroup. GPU only overtakes a threaded CPU port past roughly 38
// simultaneous tags.
class PoseEstimator {
 public:
  // libapriltag's estimate_tag_pose uses 50; kept identical so results are
  // comparable to it.
  static constexpr int kDefaultIterations = 50;

  // `tagsize` is the full width of the tag's black border, in metres - the
  // same quantity libapriltag's apriltag_detection_info_t::tagsize takes.
  // Translation scales linearly with it, so a 1% error here is a 1% range
  // error, which dwarfs every numerical consideration in the solver.
  //
  // `cpu_threads` is the total degree of parallelism for EstimateAll (see
  // WorkerPool); 0 selects hardware_concurrency, overridable by
  // APRILTAG_CPU_THREADS via ResolveThreadCount - the same resolution
  // QuadDecode and TagDecoder use, so the env var means one thing across
  // every CPU-tail phase.
  PoseEstimator(CameraIntrinsics intrinsics, double tagsize, uint32_t cpu_threads = 0);

  PoseEstimator(const PoseEstimator &) = delete;
  PoseEstimator &operator=(const PoseEstimator &) = delete;

  // The homography-based initial estimate on its own, before any refinement.
  // Equivalent to libapriltag's estimate_pose_for_tag_homography. Exposed
  // because verifying the seed separately from the refinement is how a
  // divergence gets localized - see tools/validate_pose's ladder.
  TagPose EstimateSeed(const double H[3][3]) const;

  // Both candidate solutions, unranked. Equivalent to libapriltag's
  // estimate_tag_pose_orthogonal_iteration. `iterations` is exposed for the
  // same reason its nIters is: running with 1 isolates a single iteration
  // when chasing a numerical difference.
  TagPosePair EstimateBoth(const double corners[4][2], const double H[3][3],
                           int iterations = kDefaultIterations) const;

  // The lower-error of the two candidate solutions. Equivalent to
  // libapriltag's estimate_tag_pose.
  TagPose Estimate(const double corners[4][2], const double H[3][3]) const;

  // Estimate for every detection, in parallel across detections (the work is
  // fully independent per detection). Results are written in input order, so
  // the output is identical regardless of thread count.
  void EstimateAll(const std::vector<const apriltag_detection_t *> &detections,
                   std::vector<TagPose> &out) const;

  // Convenience overload for a zarray_t* of apriltag_detection_t*, which is
  // what TagDecoder::Decode returns.
  void EstimateAll(const zarray_t *detections, std::vector<TagPose> &out) const;

  const CameraIntrinsics &intrinsics() const { return intrinsics_; }
  double tagsize() const { return tagsize_; }
  unsigned threads() const { return pool_->threads(); }

 private:
  CameraIntrinsics intrinsics_;
  double tagsize_ = 0.0;
  std::unique_ptr<WorkerPool> pool_;
};

}  // namespace apriltag_vulkan
