// End-to-end pose validation: for every image in a directory, runs BOTH
// pipelines fully independently, starting from the raw pixels and ending at a
// 6-DoF tag pose, then compares the poses of tags decoded by both sides:
//
//   (a) stock libapriltag: apriltag_detector_detect() + estimate_tag_pose()
//       (apriltag_pose.c), unmodified upstream code.
//   (b) vkapriltag: GpuDetector -> QuadDecode -> TagDecoder for detection,
//       then apriltag_vulkan::PoseEstimator::Estimate() for pose.
//
// Neither side's corners/homography are shared with the other - this is NOT
// the same thing tools/validate_pose does (which feeds one shared H into both
// pose solvers). Here, corner-localization differences between the two
// detectors flow into the pose comparison, which is the point: it measures
// what a user of the whole vkapriltag pipeline actually gets versus what they
// would have gotten from stock libapriltag, end to end.
//
// Modeled closely on validate_against_libapriltag_opencv.cpp (detection
// side) and validate_pose.cpp (RotationAngleDeg, pose plumbing).
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "vkapriltag/PoseEstimator.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "common/matd.h"
}

namespace {

using apriltag_vulkan::CameraIntrinsics;
using apriltag_vulkan::PoseEstimator;
using apriltag_vulkan::TagPose;

constexpr double kPi = 3.14159265358979323846;

double Norm3(const double v[3]) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Geodesic angle between two rotation matrices, via the axis-angle form of
// D = A' * B: atan2(|axis|, cos-part). Numerically robust near identity,
// unlike acos((trace-1)/2) - see tools/validate_pose/validate_pose.cpp,
// which this is copied from verbatim.
double RotationAngleDeg(const double A[3][3], const double B[3][3]) {
  double D[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += A[k][i] * B[k][j];
      D[i][j] = s;
    }
  }
  const double ax = D[2][1] - D[1][2];
  const double ay = D[0][2] - D[2][0];
  const double az = D[1][0] - D[0][1];
  const double sin_scaled = std::sqrt(ax * ax + ay * ay + az * az) * 0.5;
  const double cos_scaled = (D[0][0] + D[1][1] + D[2][2] - 1.0) * 0.5;
  return std::atan2(sin_scaled, cos_scaled) * 180.0 / kPi;
}

// One side's pose for one tag id, in a common representation so both the
// libapriltag matd_t* result and the PoseEstimator TagPose result can be
// compared uniformly.
struct PoseResult {
  int id = 0;
  double R[3][3] = {};
  double t[3] = {};
  bool valid = false;
};

struct MatchDelta {
  int id = 0;
  double rot_deg = 0.0;
  double dt_abs = 0.0;
  double dt_rel = 0.0;
};

struct ImageSummary {
  std::string file;
  int ref_count = 0;
  int our_count = 0;
  int matched = 0;
  double mean_rot = 0.0, worst_rot = 0.0;
  double mean_dt_rel = 0.0, worst_dt_rel = 0.0;
};

struct Stat3 {
  double mean = 0.0, median = 0.0, worst = 0.0;
};

Stat3 Summarize(std::vector<double> v) {
  Stat3 s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0.0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  s.median = v[v.size() / 2];
  s.worst = v.back();
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  std::string load_path;
  std::string family_name = "tag36h11";
  uint32_t decimation = 2;
  double tagsize = 0.1651;  // metres; matches validate_pose.cpp's default.
  // 0 means "derive per-image": fx = fy = image width, cx/cy = image center.
  // This tool is measuring pipeline-vs-pipeline agreement, not absolute
  // metric accuracy, so the exact intrinsics don't matter as long as both
  // sides get the identical values - see the header comment.
  double fx_override = 0.0, fy_override = 0.0, cx_override = -1.0, cy_override = -1.0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--data") {
      load_path = next("--data");
    } else if (arg == "--family") {
      family_name = next("--family");
    } else if (arg == "--decimation") {
      decimation = static_cast<uint32_t>(std::max(1, std::stoi(next("--decimation"))));
    } else if (arg == "--tagsize") {
      tagsize = std::stod(next("--tagsize"));
    } else if (arg == "--fx") {
      fx_override = std::stod(next("--fx"));
    } else if (arg == "--fy") {
      fy_override = std::stod(next("--fy"));
    } else if (arg == "--cx") {
      cx_override = std::stod(next("--cx"));
    } else if (arg == "--cy") {
      cy_override = std::stod(next("--cy"));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (load_path.empty()) {
    std::cerr << "Usage: apriltag_pose_e2e_validate --data <dir-or-file> [--family tag36h11] "
                 "[--decimation N] [--tagsize M] [--fx F --fy F --cx C --cy C]"
              << std::endl;
    return 1;
  }

  std::filesystem::path path = load_path;
  std::error_code ec;
  std::vector<std::string> files;
  if (std::filesystem::is_directory(path, ec)) {
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      if (!std::filesystem::is_regular_file(entry.path())) continue;
      cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        std::cerr << "Skipping (failed to load): " << entry.path().string() << std::endl;
        continue;
      }
      if (img.cols % 2 != 0 || img.rows % 2 != 0) {
        std::cerr << "Skipping (odd dimensions): " << entry.path().string() << std::endl;
        continue;
      }
      files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
  } else if (std::filesystem::is_regular_file(path, ec)) {
    files.push_back(path.string());
  } else {
    std::cerr << "Error: " << load_path << " is not a valid file or directory." << std::endl;
    return 1;
  }

  std::cout << "Testing " << files.size() << " image(s) from: " << load_path << std::endl;
  std::cout << "tagsize=" << tagsize << " m, decimation=" << decimation
            << ", family=" << family_name << std::endl;
  std::cout << "============================================================" << std::endl;

  std::vector<ImageSummary> summaries;
  std::vector<double> all_rot_deg, all_dt_rel;
  int images_zero_matched = 0;
  int images_failed = 0;

  for (const std::string &file : files) {
    cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      std::cerr << "Failed to load: " << file << std::endl;
      ++images_failed;
      continue;
    }
    const uint32_t width = static_cast<uint32_t>(image.cols);
    const uint32_t height = static_cast<uint32_t>(image.rows);

    CameraIntrinsics intr;
    intr.fx = fx_override > 0.0 ? fx_override : static_cast<double>(width);
    intr.fy = fy_override > 0.0 ? fy_override : static_cast<double>(width);
    intr.cx = cx_override >= 0.0 ? cx_override : width / 2.0;
    intr.cy = cy_override >= 0.0 ? cy_override : height / 2.0;

    apriltag_family_t *tf = nullptr;
    if (!setup_tag_family(&tf, family_name.c_str())) {
      std::cerr << "Failed to set up family '" << family_name << "' for " << file << std::endl;
      ++images_failed;
      continue;
    }

    // ---------------- vkapriltag pipeline: detection ----------------
    apriltag_detector_t *td_ours = apriltag_detector_create();
    apriltag_detector_add_family(td_ours, tf);
    td_ours->refine_edges = false;  // RefineEdges is not ported - see README.md.

    apriltag_vulkan::DetectorConfig config;
    config.width = width;
    config.height = height;
    config.decimation = decimation;
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;

    apriltag_vulkan::vk::Context ctx;
    apriltag_vulkan::GpuDetector detector(ctx, config);
    apriltag_vulkan::QuadDecode quad_decode(config);
    apriltag_vulkan::TagDecoder tag_decoder(td_ours);

    detector.Detect(image.data);
    std::vector<apriltag_vulkan::DetectedQuad> quads =
        quad_decode.Decode(detector.last_line_fit_points);
    zarray_t *ours = tag_decoder.Decode(quads, image.data, width, height, config.reversed_border);

    // ---------------- stock libapriltag pipeline: detection ----------------
    apriltag_detector_t *td_ref = apriltag_detector_create();
    apriltag_detector_add_family(td_ref, tf);
    td_ref->quad_decimate = static_cast<float>(decimation);
    td_ref->nthreads = 1;

    image_u8_t im{
        .width = static_cast<int32_t>(width),
        .height = static_cast<int32_t>(height),
        .stride = static_cast<int32_t>(width),
        .buf = image.data,
    };
    zarray_t *ref = apriltag_detector_detect(td_ref, &im);

    // ---------------- pose: reference side (apriltag_pose.c) ----------------
    std::map<int, PoseResult> ref_poses;
    for (int i = 0; i < zarray_size(ref); ++i) {
      apriltag_detection_t *det = nullptr;
      zarray_get(ref, i, &det);
      if (det == nullptr || det->H == nullptr) continue;
      apriltag_detection_info_t info{det, tagsize, intr.fx, intr.fy, intr.cx, intr.cy};
      apriltag_pose_t pose{};
      estimate_tag_pose(&info, &pose);
      PoseResult pr;
      pr.id = det->id;
      pr.valid = (pose.R != nullptr && pose.t != nullptr);
      if (pr.valid) {
        for (int a = 0; a < 3; ++a) {
          pr.t[a] = MATD_EL(pose.t, a, 0);
          for (int b = 0; b < 3; ++b) pr.R[a][b] = MATD_EL(pose.R, a, b);
        }
      }
      if (pose.R) matd_destroy(pose.R);
      if (pose.t) matd_destroy(pose.t);
      // A tag ID can (rarely) repeat within one image's detections; keep the
      // first, since duplicate IDs are a decoder-quality issue orthogonal to
      // pose and this tool's matching is by ID.
      ref_poses.emplace(pr.id, pr);
    }

    // ---------------- pose: vkapriltag side (PoseEstimator) ----------------
    const PoseEstimator est(intr, tagsize);
    std::map<int, PoseResult> our_poses;
    for (int i = 0; i < zarray_size(ours); ++i) {
      apriltag_detection_t *det = nullptr;
      zarray_get(ours, i, &det);
      if (det == nullptr || det->H == nullptr) continue;
      double H[3][3], corners[4][2];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) H[a][b] = MATD_EL(det->H, a, b);
      for (int c = 0; c < 4; ++c) {
        corners[c][0] = det->p[c][0];
        corners[c][1] = det->p[c][1];
      }
      const TagPose tp = est.Estimate(corners, H);
      PoseResult pr;
      pr.id = det->id;
      pr.valid = tp.valid;
      if (pr.valid) {
        for (int a = 0; a < 3; ++a) {
          pr.t[a] = tp.t[a];
          for (int b = 0; b < 3; ++b) pr.R[a][b] = tp.R[a][b];
        }
      }
      our_poses.emplace(pr.id, pr);
    }

    // ---------------- match by ID, compute deltas ----------------
    ImageSummary summary;
    summary.file = std::filesystem::path(file).filename().string();
    summary.ref_count = static_cast<int>(ref_poses.size());
    summary.our_count = static_cast<int>(our_poses.size());

    std::vector<double> img_rot, img_dt_rel;
    for (const auto &[id, rp] : ref_poses) {
      auto it = our_poses.find(id);
      if (it == our_poses.end()) continue;
      const PoseResult &op = it->second;
      if (!rp.valid || !op.valid) continue;  // detected both sides, but pose failed on one
      const double rot_deg = RotationAngleDeg(rp.R, op.R);
      double diff[3];
      for (int a = 0; a < 3; ++a) diff[a] = op.t[a] - rp.t[a];
      const double dt_abs = Norm3(diff);
      const double ref_norm = Norm3(rp.t);
      const double dt_rel = (ref_norm > 0.0) ? dt_abs / ref_norm : dt_abs;

      img_rot.push_back(rot_deg);
      img_dt_rel.push_back(dt_rel);
      all_rot_deg.push_back(rot_deg);
      all_dt_rel.push_back(dt_rel);
    }
    summary.matched = static_cast<int>(img_rot.size());
    if (!img_rot.empty()) {
      const Stat3 rs = Summarize(img_rot);
      const Stat3 ds = Summarize(img_dt_rel);
      summary.mean_rot = rs.mean;
      summary.worst_rot = rs.worst;
      summary.mean_dt_rel = ds.mean;
      summary.worst_dt_rel = ds.worst;
    } else {
      ++images_zero_matched;
    }
    summaries.push_back(summary);

    std::cout << summary.file << " (" << width << "x" << height << "): "
              << "libapriltag=" << summary.ref_count << " tags, vkapriltag=" << summary.our_count
              << " tags, matched=" << summary.matched;
    if (summary.matched > 0) {
      std::cout << " | rot deg mean=" << summary.mean_rot << " worst=" << summary.worst_rot
                << " | dt_rel mean=" << summary.mean_dt_rel << " worst=" << summary.worst_dt_rel;
    } else {
      std::cout << " | NO MATCHED TAGS";
    }
    std::cout << std::endl;

    apriltag_detector_destroy(td_ours);
    apriltag_detector_destroy(td_ref);
    teardown_tag_family(&tf, family_name.c_str());
  }

  std::cout << "============================================================" << std::endl;
  std::cout << "Images tested: " << files.size() << " (" << images_failed << " failed to load, "
            << images_zero_matched << " with zero matched tags)" << std::endl;

  if (!all_rot_deg.empty()) {
    const Stat3 rs = Summarize(all_rot_deg);
    const Stat3 ds = Summarize(all_dt_rel);
    std::cout << "Aggregate over " << all_rot_deg.size() << " matched tag(s) across all images:"
              << std::endl;
    std::printf("  rotation delta (deg):        mean=%.6f  median=%.6f  worst=%.6f\n", rs.mean,
                rs.median, rs.worst);
    std::printf("  translation delta (rel |t|): mean=%.6f  median=%.6f  worst=%.6f\n", ds.mean,
                ds.median, ds.worst);
  } else {
    std::cout << "No matched tags across any image - nothing to compare." << std::endl;
  }

  return 0;
}
