#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vkapriltag/common/WorkerPool.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

// Pinhole camera intrinsics in pixels; lens distortion is not modelled.
struct CameraIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

// The tag's pose in the camera optical frame (`t` position, `R` orientation), as libapriltag's
// apriltag_pose_t. `error` is the object-space error of the fit; `valid` is false if no pose
// could be computed.
struct TagPose {
  double R[3][3] = {};
  double t[3] = {};
  double error = 0.0;
  bool valid = false;
  // Orthogonal-iteration steps run (the cap when early exit is disabled).
  int iterations = 0;
};

// Both candidate solutions of the planar-pose ambiguity, before the lower-error one is chosen.
struct TagPosePair {
  TagPose solution1;  // orthogonal iteration from the homography seed
  TagPose solution2;  // the second local minimum, if one exists
};

// Tag pose from four corners, ported from libapriltag's apriltag_pose.c: homography seed,
// orthogonal iteration, and a search for the second local minimum, with the lower-error solution
// winning. Uses fixed-size 3x3/3x1 arithmetic instead of matd_t. Runs on the CPU.
class PoseEstimator {
 public:
  // Iteration cap (libapriltag uses 50).
  static constexpr int kDefaultIterations = 50;

  // Orthogonal iteration stops when both the relative translation change and the largest
  // elementwise rotation change fall below this; 0 disables early exit and always runs the full
  // iteration count.
  static constexpr double kDefaultConvergenceTol = 1e-8;

  // `tagsize` is the full width of the tag's black border in metres, as in libapriltag.
  //
  // `cpu_threads` is the total parallelism for EstimateAll (0 = hardware_concurrency, or
  // APRILTAG_CPU_THREADS). `convergence_tol` is the early-exit threshold above.
  PoseEstimator(CameraIntrinsics intrinsics, double tagsize, uint32_t cpu_threads = 0,
                double convergence_tol = kDefaultConvergenceTol);

  PoseEstimator(const PoseEstimator &) = delete;
  PoseEstimator &operator=(const PoseEstimator &) = delete;

  // The homography-based initial estimate (libapriltag's estimate_pose_for_tag_homography).
  TagPose EstimateSeed(const double H[3][3]) const;

  // Both candidate solutions, unranked (libapriltag's estimate_tag_pose_orthogonal_iteration).
  TagPosePair EstimateBoth(const double corners[4][2], const double H[3][3],
                           int iterations = kDefaultIterations) const;

  // The lower-error candidate (libapriltag's estimate_tag_pose).
  TagPose Estimate(const double corners[4][2], const double H[3][3]) const;

  // Estimates every detection in parallel; results are in input order.
  void EstimateAll(const std::vector<const apriltag_detection_t *> &detections,
                   std::vector<TagPose> &out) const;

  // Overload for the zarray_t* of apriltag_detection_t* returned by TagDecoder::Decode.
  void EstimateAll(const zarray_t *detections, std::vector<TagPose> &out) const;

  const CameraIntrinsics &intrinsics() const { return intrinsics_; }
  double tagsize() const { return tagsize_; }
  double convergence_tol() const { return convergence_tol_; }
  unsigned threads() const { return pool_->threads(); }

 private:
  CameraIntrinsics intrinsics_;
  double tagsize_ = 0.0;
  double convergence_tol_ = kDefaultConvergenceTol;
  std::unique_ptr<WorkerPool> pool_;
};

}  // namespace apriltag_vulkan
