#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "vkapriltag/Detection.h"
#include "vkapriltag/Pose.h"
#include "vkapriltag/common/WorkerPool.h"

namespace apriltag_vulkan {

// Pinhole camera intrinsics in pixels; lens distortion is not modelled.
class PoseEstimator {
 public:
  // Iteration cap (libapriltag uses 50).
  static constexpr int kDefaultIterations = 50;

  // Orthogonal iteration stops when both the relative translation change and the largest
  // elementwise rotation change fall below this; 0 disables early exit and always runs the full
  // iteration count.
  static constexpr double kDefaultConvergenceTol = 1e-8;

  // `tagsize` is the full width of the tag's black border in metres, as in libapriltag. Throws
  // std::invalid_argument unless tagsize, fx and fy are finite and positive and cx, cy finite.
  //
  // `cpu_threads` is the total parallelism for EstimateAll (0 = hardware_concurrency, or
  // APRILTAG_CPU_THREADS). `convergence_tol` is the early-exit threshold above. `pool`, if given, is
  // used instead of a pool of its own (`cpu_threads` is then ignored).
  PoseEstimator(CameraIntrinsics intrinsics, double tagsize, uint32_t cpu_threads = 0,
                double convergence_tol = kDefaultConvergenceTol, std::shared_ptr<WorkerPool> pool = nullptr);

  PoseEstimator(const PoseEstimator &) = delete;
  PoseEstimator &operator=(const PoseEstimator &) = delete;

  // The homography-based initial estimate (libapriltag's estimate_pose_for_tag_homography).
  TagPose EstimateSeed(const double H[3][3]) const;

  // Both candidate solutions, unranked (libapriltag's estimate_tag_pose_orthogonal_iteration).
  TagPosePair EstimateBoth(const double corners[4][2], const double H[3][3],
                           int iterations = kDefaultIterations) const;

  // The lower-error candidate (libapriltag's estimate_tag_pose).
  TagPose Estimate(const double corners[4][2], const double H[3][3]) const;

  // Estimates every detection in parallel (a DetectionList converts); results are in input order.
  void EstimateAll(std::span<const Detection> detections, std::vector<TagPose> &out) const;

  const CameraIntrinsics &intrinsics() const { return intrinsics_; }
  double tagsize() const { return tagsize_; }
  double convergence_tol() const { return convergence_tol_; }
  unsigned threads() const { return pool_->threads(); }

 private:
  CameraIntrinsics intrinsics_;
  double tagsize_ = 0.0;
  double convergence_tol_ = kDefaultConvergenceTol;
  std::shared_ptr<WorkerPool> pool_;
};

}  // namespace apriltag_vulkan
