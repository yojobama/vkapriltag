#pragma once

namespace apriltag_vulkan {

// Pinhole camera intrinsics in pixels.
struct CameraIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

// A tag's pose in the camera frame: tag points p map to R * p + t, in the unit of the tag size.
struct TagPose {
  double R[3][3] = {};
  double t[3] = {};
  double error = 0.0;  // object-space error of the solution
  bool valid = false;
  int iterations = 0;
};

// The two candidate solutions of the planar pose ambiguity.
struct TagPosePair {
  TagPose solution1;  // orthogonal iteration from the homography seed
  TagPose solution2;  // the second local minimum, if one exists
};

}  // namespace apriltag_vulkan
