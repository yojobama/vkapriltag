// Ground-truths the libapriltag-vs-vkapriltag pose delta: renders a tag at a chosen 6-DoF pose into a synthetic frame,
// runs both end-to-end pipelines (as validate_pose_e2e does) and compares each against the known pose and against each other.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
// OpenCV 5 moved getPerspectiveTransform() into a geometry module; include it only where it exists.
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

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

// Copied from validate_pose.cpp and validate_pose_e2e.cpp; stable near identity.
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

void RotY(double a, double out[3][3]) {
  const double c = std::cos(a), s = std::sin(a);
  const double m[3][3] = {{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
  std::memcpy(out, m, sizeof(m));
}
void RotZ(double a, double out[3][3]) {
  const double c = std::cos(a), s = std::sin(a);
  const double m[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
  std::memcpy(out, m, sizeof(m));
}
void Mul33(const double a[3][3], const double b[3][3], double out[3][3]) {
  double tmp[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += a[i][k] * b[k][j];
      tmp[i][j] = s;
    }
  std::memcpy(out, tmp, sizeof(tmp));
}

// --- canonical tag bitmap, built from the family's own data -------------

// Renders family/id as a solid-cell greyscale bitmap: `cell_px` pixels per cell, `total_width` cells square
// (border ring, data cells, quiet zone). Quiet zone and "1" bits are white (255), border ring and "0" bits black (0); reversed_border is not handled.
cv::Mat RenderTagBitmap(const apriltag_family_t *tf, uint32_t id, int cell_px, int *border_start_px,
                        int *border_width_px) {
  const int total_px = tf->total_width * cell_px;
  cv::Mat bitmap(total_px, total_px, CV_8UC1, cv::Scalar(255));  // quiet zone: white.

  const int border_start_cell = (tf->total_width - tf->width_at_border) / 2;
  const int bstart_px = border_start_cell * cell_px;
  const int bwidth_px = tf->width_at_border * cell_px;
  *border_start_px = bstart_px;
  *border_width_px = bwidth_px;

  // Solid black border ring; "1" data-bit cells punch white squares into it.
  cv::rectangle(bitmap, cv::Rect(bstart_px, bstart_px, bwidth_px, bwidth_px), cv::Scalar(0),
                cv::FILLED);

  const uint64_t code = tf->codes[id];
  for (uint32_t i = 0; i < tf->nbits; ++i) {
    const bool one = (code & (uint64_t(1) << (tf->nbits - i - 1))) != 0;
    if (!one) continue;  // "0" bits stay black (the border-ring fill above).
    const int cx = bstart_px + static_cast<int>(tf->bit_x[i]) * cell_px;
    const int cy = bstart_px + static_cast<int>(tf->bit_y[i]) * cell_px;
    cv::rectangle(bitmap, cv::Rect(cx, cy, cell_px, cell_px), cv::Scalar(255), cv::FILLED);
  }
  return bitmap;
}

// --- ground-truth pose + projected border-square corners -----------------

struct SynthCase {
  double truth_R[3][3];
  double truth_t[3];
  // Border-square corners in image pixels, ordered (-1,1),(1,1),(1,-1),(-1,-1) in the tag's object frame,
  // the order a detector's p[] uses.
  double corners[4][2];
  double px_size = 0.0;  // mean edge length of the four corners, in pixels
  bool ok = false;
};

SynthCase MakeSynthCase(const CameraIntrinsics &intr, double tagsize, double dist, double tilt,
                        double spin, double off_x, double off_y, int width, int height) {
  SynthCase c;
  double Ry[3][3], Rz[3][3];
  RotY(tilt, Ry);
  RotZ(spin, Rz);
  Mul33(Ry, Rz, c.truth_R);
  c.truth_t[0] = off_x;
  c.truth_t[1] = off_y;
  c.truth_t[2] = dist;

  const double s = tagsize / 2.0;
  const double obj[4][3] = {{-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};
  for (int i = 0; i < 4; ++i) {
    double cam[3];
    for (int a = 0; a < 3; ++a) {
      cam[a] = c.truth_R[a][0] * obj[i][0] + c.truth_R[a][1] * obj[i][1] +
               c.truth_R[a][2] * obj[i][2] + c.truth_t[a];
    }
    if (!(cam[2] > 1e-6)) return c;  // behind or on the camera plane
    c.corners[i][0] = intr.fx * cam[0] / cam[2] + intr.cx;
    c.corners[i][1] = intr.fy * cam[1] / cam[2] + intr.cy;
    // Reject anything falling even partly outside the frame.
    if (c.corners[i][0] < 0 || c.corners[i][0] >= width || c.corners[i][1] < 0 ||
        c.corners[i][1] >= height) {
      return c;
    }
  }
  double sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double dx = c.corners[(i + 1) % 4][0] - c.corners[i][0];
    const double dy = c.corners[(i + 1) % 4][1] - c.corners[i][1];
    sum += std::sqrt(dx * dx + dy * dy);
  }
  c.px_size = sum / 4.0;
  c.ok = true;
  return c;
}

cv::Mat RenderFrame(const cv::Mat &bitmap, int border_start_px, int border_width_px,
                    const double corners_image[4][2], int width, int height, double blur_sigma,
                    double noise_sigma, uint8_t background) {
  // Bitmap-space corners of the border square in the same order as corners_image.
  // Object +y maps to increasing pixel row, so obj (-1,+1) is the larger-y (bottom) bitmap corner.
  const cv::Point2f src[4] = {
      {static_cast<float>(border_start_px),
       static_cast<float>(border_start_px + border_width_px)},
      {static_cast<float>(border_start_px + border_width_px),
       static_cast<float>(border_start_px + border_width_px)},
      {static_cast<float>(border_start_px + border_width_px), static_cast<float>(border_start_px)},
      {static_cast<float>(border_start_px), static_cast<float>(border_start_px)}};
  cv::Point2f dst[4];
  for (int i = 0; i < 4; ++i) {
    dst[i] = {static_cast<float>(corners_image[i][0]), static_cast<float>(corners_image[i][1])};
  }
  const cv::Mat H = cv::getPerspectiveTransform(src, dst);

  cv::Mat frame(height, width, CV_8UC1, cv::Scalar(background));
  // BORDER_TRANSPARENT leaves the pre-filled flat background outside the tag's plane.
  cv::warpPerspective(bitmap, frame, H, frame.size(), cv::INTER_AREA, cv::BORDER_TRANSPARENT);

  // Blur and read noise, so corner estimates are sub-pixel as on a real camera.
  if (blur_sigma > 0.0) cv::GaussianBlur(frame, frame, cv::Size(0, 0), blur_sigma);
  if (noise_sigma > 0.0) {
    cv::Mat noise(frame.size(), CV_32F);
    cv::randn(noise, 0.0, noise_sigma);
    cv::Mat frame_f;
    frame.convertTo(frame_f, CV_32F);
    frame_f += noise;
    frame_f.convertTo(frame, CV_8UC1);
  }
  return frame;
}

// --- one side's pose + corners, for one case -----------------------------

struct SideResult {
  bool decoded = false;   // this side found exactly the tag id under test
  bool pose_valid = false;
  double R[3][3] = {};
  double t[3] = {};
  double corners[4][2] = {};
};

struct CaseResult {
  bool both_decoded = false;
  double px_size = 0.0;
  double lib_rot_vs_truth = 0.0, lib_dt_rel_vs_truth = 0.0, lib_corner_rms = 0.0;
  double vk_rot_vs_truth = 0.0, vk_dt_rel_vs_truth = 0.0, vk_corner_rms = 0.0;
  double mutual_rot = 0.0, mutual_dt_rel = 0.0;
};

double CornerRms(const double a[4][2], const double b[4][2]) {
  double sq = 0.0;
  for (int c = 0; c < 4; ++c) {
    const double dx = a[c][0] - b[c][0];
    const double dy = a[c][1] - b[c][1];
    sq += dx * dx + dy * dy;
  }
  return std::sqrt(sq / 4.0);
}

struct Stat3 {
  double mean = 0.0, median = 0.0, worst = 0.0;
  int n = 0;
};
Stat3 Summarize(std::vector<double> v) {
  Stat3 s;
  s.n = static_cast<int>(v.size());
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
  // Fixed seed so cv::randn() output, and hence the CI gate, is reproducible.
  cv::theRNG() = cv::RNG(0x5eed);

  std::string family_name = "tag36h11";
  int width = 1280, height = 800;
  double tagsize = 0.1651;
  double blur_sigma = 0.7, noise_sigma = 3.0;
  std::string dump_dir;
  std::vector<uint32_t> decimations = {1, 2};
  std::vector<uint32_t> ids = {0, 5};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--width") width = std::stoi(next("--width"));
    else if (arg == "--height") height = std::stoi(next("--height"));
    else if (arg == "--tagsize") tagsize = std::stod(next("--tagsize"));
    else if (arg == "--blur-sigma") blur_sigma = std::stod(next("--blur-sigma"));
    else if (arg == "--noise-sigma") noise_sigma = std::stod(next("--noise-sigma"));
    else if (arg == "--dump-dir") dump_dir = next("--dump-dir");
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }
  if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);

  CameraIntrinsics intr{static_cast<double>(width), static_cast<double>(width), width / 2.0,
                        height / 2.0};

  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, family_name.c_str())) {
    std::cerr << "Failed to set up family '" << family_name << "'" << std::endl;
    return 1;
  }

  // Sweep of target pixel sizes; tilt/spin add foreshortening and in-plane rotation.
  // dist is solved from the target size at tilt=0.
  const double target_px_sizes[] = {15, 20, 30, 50, 80, 150};
  const double tilts_deg[] = {0.0, 20.0, 40.0, 60.0};
  const double spins_deg[] = {0.0, 30.0, 60.0, 90.0};
  const double off_fracs[] = {0.0, 0.3};

  std::printf("Synthetic ground-truth sweep: %dx%d, tagsize=%.4f m, family=%s\n", width, height,
              tagsize, family_name.c_str());
  std::printf("blur_sigma=%.2f noise_sigma=%.2f, ids=", blur_sigma, noise_sigma);
  for (uint32_t id : ids) std::printf("%u ", id);
  std::printf("\n============================================================\n");

  // Bucketed by truth pixel size.
  struct Bucket {
    std::string label;
    double lo, hi;
    std::vector<double> lib_rot, vk_rot, lib_dt, vk_dt, lib_crms, vk_crms, mutual_rot, mutual_dt;
    int lib_missed = 0, vk_missed = 0, both_ok = 0;
  };
  std::vector<Bucket> buckets = {
      {"<20px", 0, 20, {}, {}, {}, {}, {}, {}, {}, {}},
      {"20-40px", 20, 40, {}, {}, {}, {}, {}, {}, {}, {}},
      {"40-70px", 40, 70, {}, {}, {}, {}, {}, {}, {}, {}},
      {"70-120px", 70, 120, {}, {}, {}, {}, {}, {}, {}, {}},
      {">=120px", 120, 1e18, {}, {}, {}, {}, {}, {}, {}, {}},
  };
  auto BucketFor = [&](double px) -> Bucket & {
    for (auto &b : buckets) {
      if (px >= b.lo && px < b.hi) return b;
    }
    return buckets.back();
  };

  int case_index = 0;
  for (uint32_t decimation : decimations) {
    apriltag_vulkan::DetectorConfig config;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);
    config.decimation = decimation;
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;

    apriltag_vulkan::vk::Context ctx;
    apriltag_vulkan::GpuDetector detector(ctx, config);
    apriltag_vulkan::QuadDecode quad_decode(config);

    apriltag_detector_t *td_ours = apriltag_detector_create();
    apriltag_detector_add_family(td_ours, tf);
    td_ours->refine_edges = false;  // RefineEdges is not ported - see README.md.
    apriltag_vulkan::TagDecoder tag_decoder(td_ours, decimation);

    apriltag_detector_t *td_ref = apriltag_detector_create();
    apriltag_detector_add_family(td_ref, tf);
    td_ref->quad_decimate = static_cast<float>(decimation);
    td_ref->nthreads = 1;
    td_ref->refine_edges = false;  // apples-to-apples: see validate_pose_e2e's default.

    const PoseEstimator est(intr, tagsize);

    for (uint32_t id : ids) {
      // Render the canonical bitmap once per id at 32px/cell so warpPerspective's INTER_AREA always minifies.
      int border_start_px = 0, border_width_px = 0;
      const cv::Mat bitmap = RenderTagBitmap(tf, id, 32, &border_start_px, &border_width_px);

      for (double target_px : target_px_sizes) {
        const double dist = tagsize * intr.fx / target_px;
        for (double tilt_deg : tilts_deg) {
          for (double spin_deg : spins_deg) {
            for (double off_frac : off_fracs) {
              const SynthCase sc =
                  MakeSynthCase(intr, tagsize, dist, tilt_deg * kPi / 180.0, spin_deg * kPi / 180.0,
                               off_frac * dist, -off_frac * dist * 0.5, width, height);
              if (!sc.ok) continue;
              ++case_index;

              const cv::Mat frame = RenderFrame(bitmap, border_start_px, border_width_px,
                                                sc.corners, width, height, blur_sigma, noise_sigma,
                                                200);
              if (!dump_dir.empty()) {
                char name[256];
                std::snprintf(name, sizeof(name), "%s/case_%04d_d%u_id%u_px%.0f_t%.0f_s%.0f.png",
                              dump_dir.c_str(), case_index, decimation, id, target_px, tilt_deg,
                              spin_deg);
                cv::imwrite(name, frame);
              }

              // ---------------- vkapriltag: detect + pose ----------------
              SideResult vk;
              detector.Detect(frame.data);
              std::vector<apriltag_vulkan::DetectedQuad> quads =
                  quad_decode.Decode(detector.last_line_fit_points);
              zarray_t *ours =
                  tag_decoder.Decode(quads, frame.data, width, height, config.reversed_border);
              for (int i = 0; i < zarray_size(ours); ++i) {
                apriltag_detection_t *det = nullptr;
                zarray_get(ours, i, &det);
                if (det == nullptr || det->H == nullptr || det->id != static_cast<int>(id))
                  continue;
                double H[3][3], corners[4][2];
                for (int a = 0; a < 3; ++a)
                  for (int b = 0; b < 3; ++b) H[a][b] = MATD_EL(det->H, a, b);
                for (int c = 0; c < 4; ++c) {
                  corners[c][0] = det->p[c][0];
                  corners[c][1] = det->p[c][1];
                }
                const TagPose tp = est.Estimate(corners, H);
                vk.decoded = true;
                vk.pose_valid = tp.valid;
                for (int c = 0; c < 4; ++c) {
                  vk.corners[c][0] = corners[c][0];
                  vk.corners[c][1] = corners[c][1];
                }
                if (tp.valid) {
                  for (int a = 0; a < 3; ++a) {
                    vk.t[a] = tp.t[a];
                    for (int b = 0; b < 3; ++b) vk.R[a][b] = tp.R[a][b];
                  }
                }
                break;
              }

              // ---------------- libapriltag: detect + pose ----------------
              SideResult lib;
              image_u8_t im{static_cast<int32_t>(width), static_cast<int32_t>(height),
                            static_cast<int32_t>(width), frame.data};
              zarray_t *ref = apriltag_detector_detect(td_ref, &im);
              for (int i = 0; i < zarray_size(ref); ++i) {
                apriltag_detection_t *det = nullptr;
                zarray_get(ref, i, &det);
                if (det == nullptr || det->H == nullptr || det->id != static_cast<int>(id))
                  continue;
                apriltag_detection_info_t info{det, tagsize, intr.fx, intr.fy, intr.cx, intr.cy};
                apriltag_pose_t pose{};
                estimate_tag_pose(&info, &pose);
                lib.decoded = true;
                lib.pose_valid = (pose.R != nullptr && pose.t != nullptr);
                for (int c = 0; c < 4; ++c) {
                  lib.corners[c][0] = det->p[c][0];
                  lib.corners[c][1] = det->p[c][1];
                }
                if (lib.pose_valid) {
                  for (int a = 0; a < 3; ++a) {
                    lib.t[a] = MATD_EL(pose.t, a, 0);
                    for (int b = 0; b < 3; ++b) lib.R[a][b] = MATD_EL(pose.R, a, b);
                  }
                }
                if (pose.R) matd_destroy(pose.R);
                if (pose.t) matd_destroy(pose.t);
                break;
              }

              Bucket &bucket = BucketFor(sc.px_size);
              if (!lib.decoded) ++bucket.lib_missed;
              if (!vk.decoded) ++bucket.vk_missed;
              if (lib.decoded && vk.decoded && lib.pose_valid && vk.pose_valid) {
                ++bucket.both_ok;
                bucket.lib_rot.push_back(RotationAngleDeg(lib.R, sc.truth_R));
                bucket.vk_rot.push_back(RotationAngleDeg(vk.R, sc.truth_R));
                double dlib[3], dvk[3];
                for (int a = 0; a < 3; ++a) {
                  dlib[a] = lib.t[a] - sc.truth_t[a];
                  dvk[a] = vk.t[a] - sc.truth_t[a];
                }
                const double tn = std::max(Norm3(sc.truth_t), 1e-300);
                bucket.lib_dt.push_back(Norm3(dlib) / tn);
                bucket.vk_dt.push_back(Norm3(dvk) / tn);
                bucket.lib_crms.push_back(CornerRms(lib.corners, sc.corners));
                bucket.vk_crms.push_back(CornerRms(vk.corners, sc.corners));
                bucket.mutual_rot.push_back(RotationAngleDeg(lib.R, vk.R));
                double dm[3];
                for (int a = 0; a < 3; ++a) dm[a] = vk.t[a] - lib.t[a];
                const double libn = std::max(Norm3(lib.t), 1e-300);
                bucket.mutual_dt.push_back(Norm3(dm) / libn);
              }
            }
          }
        }
      }
    }
    apriltag_detector_destroy(td_ours);
    apriltag_detector_destroy(td_ref);
  }
  teardown_tag_family(&tf, family_name.c_str());

  std::printf("%-9s %6s %8s %8s | %11s %11s %11s | %11s %11s | %10s %10s\n", "bucket", "n",
              "lib_miss", "vk_miss", "lib_rot", "vk_rot", "mutual_rot", "lib_dtrel", "vk_dtrel",
              "lib_crms", "vk_crms");
  std::printf("%-9s %6s %8s %8s | %11s %11s %11s | %11s %11s | %10s %10s\n", "", "", "", "",
              "mean/worst", "mean/worst", "mean/worst", "mean/worst", "mean/worst", "mean/worst",
              "mean/worst");
  int total_cases = 0, total_both_ok = 0;
  std::vector<double> all_lib_rot, all_vk_rot, all_mutual_rot;
  std::vector<double> all_lib_dt, all_vk_dt, all_mutual_dt;
  for (const Bucket &b : buckets) {
    total_cases += b.both_ok + std::max(b.lib_missed, b.vk_missed);
    total_both_ok += b.both_ok;
    if (b.both_ok == 0) {
      std::printf("%-9s %6d %8d %8d | (no cases both sides decoded and posed)\n", b.label.c_str(),
                  b.both_ok, b.lib_missed, b.vk_missed);
      continue;
    }
    const Stat3 lr = Summarize(b.lib_rot), vr = Summarize(b.vk_rot), mr = Summarize(b.mutual_rot);
    const Stat3 ld = Summarize(b.lib_dt), vd = Summarize(b.vk_dt);
    const Stat3 lc = Summarize(b.lib_crms), vc = Summarize(b.vk_crms);
    std::printf("%-9s %6d %8d %8d | %5.3f/%5.3f %5.3f/%5.3f %5.3f/%5.3f | %5.3f/%5.3f %5.3f/%5.3f | "
                "%4.2f/%4.2f %4.2f/%4.2f\n",
                b.label.c_str(), b.both_ok, b.lib_missed, b.vk_missed, lr.mean, lr.worst, vr.mean,
                vr.worst, mr.mean, mr.worst, ld.mean, ld.worst, vd.mean, vd.worst, lc.mean, lc.worst,
                vc.mean, vc.worst);
    all_lib_rot.insert(all_lib_rot.end(), b.lib_rot.begin(), b.lib_rot.end());
    all_vk_rot.insert(all_vk_rot.end(), b.vk_rot.begin(), b.vk_rot.end());
    all_mutual_rot.insert(all_mutual_rot.end(), b.mutual_rot.begin(), b.mutual_rot.end());
    all_lib_dt.insert(all_lib_dt.end(), b.lib_dt.begin(), b.lib_dt.end());
    all_vk_dt.insert(all_vk_dt.end(), b.vk_dt.begin(), b.vk_dt.end());
    all_mutual_dt.insert(all_mutual_dt.end(), b.mutual_dt.begin(), b.mutual_dt.end());
  }
  std::printf("============================================================\n");
  std::printf("Cases: %d generated, %d both sides decoded and produced a valid pose\n",
              total_cases, total_both_ok);

  // ---------------- VERDICT ----------------
  // Named PASS/FAIL checks; non-zero exit on any failure. Thresholds gate on the mean, not the worst case,
  // since the worst mutual delta is dominated by pose-ambiguity branch flips.
  bool all_ok = true;
  if (all_lib_rot.empty()) {
    std::printf("VERDICT: FAIL (no cases both sides decoded and produced a valid pose)\n");
    return 1;
  }
  const Stat3 lr = Summarize(all_lib_rot), vr = Summarize(all_vk_rot), mr = Summarize(all_mutual_rot);
  const Stat3 ld = Summarize(all_lib_dt), vd = Summarize(all_vk_dt), md = Summarize(all_mutual_dt);
  std::printf("Overall rotation error vs GROUND TRUTH (deg): libapriltag mean=%.4f worst=%.4f | "
              "vkapriltag mean=%.4f worst=%.4f\n",
              lr.mean, lr.worst, vr.mean, vr.worst);
  std::printf("Overall translation error vs GROUND TRUTH (rel |t|): libapriltag mean=%.4f "
              "worst=%.4f | vkapriltag mean=%.4f worst=%.4f\n",
              ld.mean, ld.worst, vd.mean, vd.worst);
  std::printf("Overall mutual rotation delta (deg, libapriltag vs vkapriltag): mean=%.4f "
              "worst=%.4f\n",
              mr.mean, mr.worst);
  std::printf("Overall mutual translation delta (rel |t|, libapriltag vs vkapriltag): mean=%.4f "
              "worst=%.4f\n",
              md.mean, md.worst);
  std::printf("Interpretation: if lib and vk error-vs-truth are close and both far below the "
              "mutual delta's worst case, the worst mutual deltas are pose-ambiguity branch "
              "flips (see validate_pose_e2e), not either side being systematically wrong.\n");

  auto Check = [&](const char *name, bool ok, const std::string &detail) {
    std::printf("  %-32s %s  (%s)\n", name, ok ? "PASS" : "FAIL", detail.c_str());
    all_ok = all_ok && ok;
  };

  std::printf("VERDICT\n");
  {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%d/%d, floor=550", total_both_ok, total_cases);
    Check("decode+pose rate", total_both_ok >= 550, buf);

    std::snprintf(buf, sizeof(buf), "mean=%.4f deg, limit=8.0 deg", mr.mean);
    Check("mutual rotation (mean)", mr.mean <= 8.0, buf);

    std::snprintf(buf, sizeof(buf), "mean=%.4f, limit=0.01", md.mean);
    Check("mutual translation (mean, rel |t|)", md.mean <= 0.01, buf);

    const double symmetry = std::fabs(vr.mean - lr.mean);
    std::snprintf(buf, sizeof(buf), "|%.4f - %.4f| = %.4f deg, limit=2.0 deg", vr.mean, lr.mean,
                 symmetry);
    Check("lib/vk symmetry (mean rot vs truth)", symmetry <= 2.0, buf);
  }
  std::printf("  %s\n", all_ok ? "ALL CHECKS PASS" : "AT LEAST ONE CHECK FAILED");
  return all_ok ? 0 : 1;
}
