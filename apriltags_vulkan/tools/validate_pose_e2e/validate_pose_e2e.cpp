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
#include <limits>
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
//
// BOTH candidate solutions of the planar-pose ambiguity are kept, not just
// the winner. A large end-to-end rotation delta is usually not a solver
// disagreement at all but a BRANCH FLIP: the two solutions are near-tied in
// object-space error, so a sub-pixel corner difference decides which one
// wins, and the loser is tens of degrees away. Distinguishing that from a
// genuine corner divergence needs the runner-up on both sides.
struct PoseResult {
  int id = 0;
  double R[3][3] = {};   // the picked solution
  double t[3] = {};
  double alt_R[3][3] = {};  // the runner-up, if the ambiguity search found one
  double alt_t[3] = {};
  double err = 0.0;      // object-space error of the picked solution
  double alt_err = 0.0;
  bool has_alt = false;
  bool picked_alt = false;  // true when solution2 beat solution1
  double corners[4][2] = {};
  double px_size = 0.0;  // mean edge length in pixels: how big the tag is
  bool valid = false;
};

// Mean of the four edge lengths of the quad, in pixels. The ambiguity is
// near-tied (and so branch flips are likely) precisely when the tag is small
// and near fronto-parallel, so this is the covariate to report alongside a
// large delta.
double QuadPixelSize(const double p[4][2]) {
  double sum = 0.0;
  for (int c = 0; c < 4; ++c) {
    const double dx = p[(c + 1) % 4][0] - p[c][0];
    const double dy = p[(c + 1) % 4][1] - p[c][1];
    sum += std::sqrt(dx * dx + dy * dy);
  }
  return sum / 4.0;
}

// Mean signed corner offset. A constant non-zero value here (as opposed to
// zero-mean scatter) is a coordinate-convention difference between the two
// detectors, not corner noise: every corner displaced the same way.
void MeanCornerOffset(const double a[4][2], const double b[4][2], double *dx, double *dy) {
  *dx = 0.0;
  *dy = 0.0;
  for (int c = 0; c < 4; ++c) {
    *dx += a[c][0] - b[c][0];
    *dy += a[c][1] - b[c][1];
  }
  *dx /= 4.0;
  *dy /= 4.0;
}

double CornerRms(const double a[4][2], const double b[4][2]) {
  double sq = 0.0;
  for (int c = 0; c < 4; ++c) {
    const double dx = a[c][0] - b[c][0];
    const double dy = a[c][1] - b[c][1];
    sq += dx * dx + dy * dy;
  }
  return std::sqrt(sq / 4.0);
}

struct MatchDelta {
  int id = 0;
  double rot_deg = 0.0;
  double dt_abs = 0.0;
  double dt_rel = 0.0;
  // Same comparison against the reference's OTHER ambiguity solution. When
  // this is far smaller than rot_deg, the two pipelines agree on the pose and
  // merely disagree on which branch of the ambiguity to report.
  double rot_deg_vs_alt = 0.0;
  bool branch_flip = false;
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
  bool verbose = false;
  // libapriltag defaults refine_edges to ON, and vkapriltag does not implement
  // RefineEdges by default, though TagDecoder now supports it - see
  // --our-refine-edges below. Leaving both off is the apples-to-apples
  // comparison this tool defaults to; --ref-refine-edges alone measures the
  // cost of a real asymmetry (reference refining, ours not); both on is the
  // actual shipping-vs-shipping comparison.
  bool ref_refine_edges = false;
  bool our_refine_edges = false;

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
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--ref-refine-edges") {
      ref_refine_edges = (std::stoi(next("--ref-refine-edges")) != 0);
    } else if (arg == "--our-refine-edges") {
      our_refine_edges = (std::stoi(next("--our-refine-edges")) != 0);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (load_path.empty()) {
    std::cerr << "Usage: apriltag_pose_e2e_validate --data <dir-or-file> [--family tag36h11] "
                 "[--decimation N] [--tagsize M] [--fx F --fy F --cx C --cy C] "
                 "[--ref-refine-edges 0|1] [--our-refine-edges 0|1]"
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
  // Rotation delta after resolving ambiguity branch flips (i.e. comparing
  // against whichever of the reference's two minima our side actually
  // converged to). The gap between this and all_rot_deg is precisely the
  // damage done by branch selection rather than by pose accuracy.
  std::vector<double> all_rot_resolved, all_corner_rms;
  int branch_flips = 0;
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
    td_ours->refine_edges = our_refine_edges;

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
    apriltag_vulkan::TagDecoder tag_decoder(td_ours, decimation);

    detector.Detect(image.data);
    std::vector<apriltag_vulkan::DetectedQuad> quads =
        quad_decode.Decode(detector.last_line_fit_points);
    zarray_t *ours = tag_decoder.Decode(quads, image.data, width, height, config.reversed_border);

    // ---------------- stock libapriltag pipeline: detection ----------------
    apriltag_detector_t *td_ref = apriltag_detector_create();
    apriltag_detector_add_family(td_ref, tf);
    td_ref->quad_decimate = static_cast<float>(decimation);
    td_ref->nthreads = 1;
    td_ref->refine_edges = ref_refine_edges;

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
      // estimate_tag_pose() internally runs this and returns only the winner,
      // so calling the orthogonal-iteration entry point directly costs nothing
      // extra and additionally yields the runner-up needed to detect a branch
      // flip. Picking min(err1, err2) below reproduces estimate_tag_pose's own
      // choice exactly.
      apriltag_pose_t p1{}, p2{};
      double e1 = HUGE_VAL, e2 = HUGE_VAL;
      estimate_tag_pose_orthogonal_iteration(&info, &e1, &p1, &e2, &p2,
                                             PoseEstimator::kDefaultIterations);
      PoseResult pr;
      pr.id = det->id;
      for (int c = 0; c < 4; ++c) {
        pr.corners[c][0] = det->p[c][0];
        pr.corners[c][1] = det->p[c][1];
      }
      pr.px_size = QuadPixelSize(pr.corners);
      const bool has1 = (p1.R != nullptr && p1.t != nullptr);
      const bool has2 = (p2.R != nullptr && p2.t != nullptr) && std::isfinite(e2);
      pr.valid = has1 || has2;
      if (pr.valid) {
        const bool take2 = has2 && (!has1 || e2 < e1);
        const apriltag_pose_t &win = take2 ? p2 : p1;
        const apriltag_pose_t &lose = take2 ? p1 : p2;
        pr.picked_alt = take2;
        pr.err = take2 ? e2 : e1;
        pr.alt_err = take2 ? e1 : e2;
        for (int a = 0; a < 3; ++a) {
          pr.t[a] = MATD_EL(win.t, a, 0);
          for (int b = 0; b < 3; ++b) pr.R[a][b] = MATD_EL(win.R, a, b);
        }
        pr.has_alt = has1 && has2;
        if (pr.has_alt) {
          for (int a = 0; a < 3; ++a) {
            pr.alt_t[a] = MATD_EL(lose.t, a, 0);
            for (int b = 0; b < 3; ++b) pr.alt_R[a][b] = MATD_EL(lose.R, a, b);
          }
        }
      }
      if (p1.R) matd_destroy(p1.R);
      if (p1.t) matd_destroy(p1.t);
      if (p2.R) matd_destroy(p2.R);
      if (p2.t) matd_destroy(p2.t);
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
      // EstimateBoth + min(error) is exactly what Estimate() does internally,
      // so this is the same answer plus the runner-up.
      const apriltag_vulkan::TagPosePair pair = est.EstimateBoth(corners, H);
      PoseResult pr;
      pr.id = det->id;
      for (int c = 0; c < 4; ++c) {
        pr.corners[c][0] = corners[c][0];
        pr.corners[c][1] = corners[c][1];
      }
      pr.px_size = QuadPixelSize(pr.corners);
      const bool has1 = pair.solution1.valid;
      const bool has2 = pair.solution2.valid;
      pr.valid = has1 || has2;
      if (pr.valid) {
        const bool take2 = has2 && (!has1 || pair.solution2.error < pair.solution1.error);
        const TagPose &win = take2 ? pair.solution2 : pair.solution1;
        const TagPose &lose = take2 ? pair.solution1 : pair.solution2;
        pr.picked_alt = take2;
        pr.err = win.error;
        pr.alt_err = lose.error;
        for (int a = 0; a < 3; ++a) {
          pr.t[a] = win.t[a];
          for (int b = 0; b < 3; ++b) pr.R[a][b] = win.R[a][b];
        }
        pr.has_alt = has1 && has2;
        if (pr.has_alt) {
          for (int a = 0; a < 3; ++a) {
            pr.alt_t[a] = lose.t[a];
            for (int b = 0; b < 3; ++b) pr.alt_R[a][b] = lose.R[a][b];
          }
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

      // Our winner against the reference's RUNNER-UP. If the two pipelines
      // picked opposite branches of the planar-pose ambiguity, this is the
      // comparison that reflects how well they actually agree; rot_deg then
      // measures the gap between the ambiguity's two minima, which is a
      // property of the geometry, not of either implementation.
      const double rot_vs_alt =
          rp.has_alt ? RotationAngleDeg(rp.alt_R, op.R) : std::numeric_limits<double>::quiet_NaN();
      const bool flip = rp.has_alt && rot_vs_alt < rot_deg * 0.5;

      if (verbose) {
        const double crms = CornerRms(op.corners, rp.corners);
        // err_gap: how near-tied the reference's two minima are. A flip is
        // benign exactly when this is small - the selection is then decided by
        // corner noise far below the detector's own precision.
        const double err_gap =
            rp.has_alt && rp.alt_err > 0.0 ? std::fabs(rp.err - rp.alt_err) / rp.alt_err : 0.0;
        double mdx = 0.0, mdy = 0.0;
        MeanCornerOffset(op.corners, rp.corners, &mdx, &mdy);
        std::printf(
            "    id=%-4d px=%6.1f corner_rms=%7.4f px mean_d=(%+.4f,%+.4f) | rot=%8.3f deg"
            " vs_alt=%8.3f dt_rel=%.3e | ref_err=%.4e alt=%.4e gap=%.3f pick2=%d | our_pick2=%d%s\n",
            id, rp.px_size, crms, mdx, mdy, rot_deg, rot_vs_alt, dt_rel, rp.err, rp.alt_err,
            err_gap, rp.picked_alt ? 1 : 0, op.picked_alt ? 1 : 0,
            flip ? "  <== BRANCH FLIP" : "");
      }

      img_rot.push_back(rot_deg);
      img_dt_rel.push_back(dt_rel);
      all_rot_deg.push_back(rot_deg);
      all_dt_rel.push_back(dt_rel);
      all_corner_rms.push_back(CornerRms(op.corners, rp.corners));
      if (flip) {
        ++branch_flips;
        all_rot_resolved.push_back(rot_vs_alt);
      } else {
        all_rot_resolved.push_back(rot_deg);
      }
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
    const Stat3 rr = Summarize(all_rot_resolved);
    const Stat3 cs = Summarize(all_corner_rms);
    std::cout << "Aggregate over " << all_rot_deg.size() << " matched tag(s) across all images:"
              << std::endl;
    std::printf("  corner RMS (px):             mean=%.6f  median=%.6f  worst=%.6f\n", cs.mean,
                cs.median, cs.worst);
    std::printf("  rotation delta (deg):        mean=%.6f  median=%.6f  worst=%.6f\n", rs.mean,
                rs.median, rs.worst);
    std::printf("  translation delta (rel |t|): mean=%.6f  median=%.6f  worst=%.6f\n", ds.mean,
                ds.median, ds.worst);
    std::printf("  branch flips: %d of %zu matched tag(s)\n", branch_flips, all_rot_deg.size());
    std::printf("  rotation delta EXCLUDING branch flips (deg): mean=%.6f  median=%.6f  "
                "worst=%.6f\n",
                rr.mean, rr.median, rr.worst);
  } else {
    std::cout << "No matched tags across any image - nothing to compare." << std::endl;
  }

  return 0;
}
