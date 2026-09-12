// Verifies apriltag_vulkan::PoseEstimator against libapriltag's
// apriltag_pose.c, which is the reference implementation this is a port of.
//
// libapriltag exposes every intermediate stage publicly
// (estimate_pose_for_tag_homography, and estimate_tag_pose_orthogonal_iteration
// with both solutions, both errors and a settable nIters), so each half of the
// algorithm is checked on its own rather than only the final answer. A
// divergence therefore localizes to a stage instead of just showing up at the
// end. No patch to libapriltag is needed for any of this.
//
// Ladder:
//   L1  seed                vs estimate_pose_for_tag_homography
//   L2  solution 1 + err1   vs estimate_tag_pose_orthogonal_iteration
//   L3  solution 2 + err2   vs the same (exercises fix_pose_ambiguities)
//   L4  final pick          vs estimate_tag_pose
//
// Inputs:
//   * a synthetic sweep over distance / tilt / in-plane rotation / off-axis
//     translation, which additionally gives an ABSOLUTE error against known
//     ground truth - that catches the case where both implementations agree
//     with each other and are both wrong.
//   * deliberately degenerate geometry (t parallel to e_x, fronto-parallel,
//     extreme range, tiny tag, near-collinear corners).
//   * optionally, real detections from a .pgm run through the actual pipeline
//     (--data), so the H matrices are in-distribution rather than synthesized.
// The Vulkan headers pull in windows.h on MSVC, whose min/max macros would
// otherwise break every std::min/std::max below.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vkapriltag/PoseEstimator.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/common/pgm_io.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

extern "C" {
#include "apriltag_pose.h"
#include "common/homography.h"
#include "common/matd.h"
}

namespace {

using apriltag_vulkan::CameraIntrinsics;
using apriltag_vulkan::PoseEstimator;
using apriltag_vulkan::TagPose;
using apriltag_vulkan::TagPosePair;

// kPi is not in the C++ standard and MSVC omits it by default.
constexpr double kPi = 3.14159265358979323846;

// Below this the object-space error is numerically indistinguishable from
// zero (a synthetic case is fitted exactly), so a relative comparison of
// two such values carries no information.
constexpr double kErrFloor = 1e-12;

// --- comparison metrics ---------------------------------------------------

struct Delta {
  double dt_abs = 0.0;      // metres
  double dt_rel = 0.0;      // relative to |t_ref|
  double rot_deg = 0.0;     // geodesic angle between the two rotations
  double err_abs = 0.0;     // absolute difference of the object-space error
  double err_rel = 0.0;     // ...and relative, meaningful only above kErrFloor
  double err_ref = 0.0;     // the reference error itself, for scale
  bool both_valid = false;
  bool validity_mismatch = false;
};

double Norm3(const double v[3]) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Geodesic angle between two rotations, from the axis-angle form of
// D = A'B: atan2(|axis|, cos-part).
//
// NOT acos((trace(D) - 1) / 2), which is the textbook formula and is useless
// here. acos has an infinite derivative at 1, so for two nearly-identical
// rotations - exactly the case this tool spends its time measuring - a
// single-ULP error in the trace becomes ~4e-8 rad, i.e. ~2.4e-6 deg of
// phantom difference. That put a hard noise floor under every rotation
// number, visible as a nonzero angle when comparing a solver against
// ITSELF. The atan2 form below is well conditioned at both ends and reports
// a true zero for identical input.
double RotationAngleDeg(const double A[3][3], const double B[3][3]) {
  // D = A' * B
  double D[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += A[k][i] * B[k][j];
      D[i][j] = s;
    }
  }
  // The skew part carries sin(theta) * axis; the trace carries cos(theta).
  const double ax = D[2][1] - D[1][2];
  const double ay = D[0][2] - D[2][0];
  const double az = D[1][0] - D[0][1];
  const double sin_scaled = std::sqrt(ax * ax + ay * ay + az * az) * 0.5;
  const double cos_scaled = (D[0][0] + D[1][1] + D[2][2] - 1.0) * 0.5;
  return std::atan2(sin_scaled, cos_scaled) * 180.0 / kPi;
}

Delta Compare(const TagPose &ours, const matd_t *ref_R, const matd_t *ref_t, double ref_err,
              bool ref_valid) {
  Delta d;
  if (ours.valid != ref_valid) {
    d.validity_mismatch = true;
    return d;
  }
  if (!ours.valid) return d;
  d.both_valid = true;

  double ref_Rm[3][3], ref_tv[3];
  for (int a = 0; a < 3; ++a) {
    ref_tv[a] = MATD_EL(ref_t, a, 0);
    for (int b = 0; b < 3; ++b) ref_Rm[a][b] = MATD_EL(ref_R, a, b);
  }

  double diff[3];
  for (int a = 0; a < 3; ++a) diff[a] = ours.t[a] - ref_tv[a];
  d.dt_abs = Norm3(diff);
  const double ref_norm = Norm3(ref_tv);
  d.dt_rel = (ref_norm > 0.0) ? d.dt_abs / ref_norm : d.dt_abs;
  d.rot_deg = RotationAngleDeg(ours.R, ref_Rm);
  d.err_ref = ref_err;
  d.err_abs = std::fabs(ours.error - ref_err);
  const double err_scale = std::max(std::fabs(ref_err), 1e-300);
  d.err_rel = d.err_abs / err_scale;
  return d;
}

// Worst-case accumulator, so a single bad case cannot be averaged away.
struct Worst {
  // Explicit constructor rather than aggregate initialization: brace-init of
  // just the name leaves the remaining members unmentioned, which GCC
  // reports under -Wmissing-field-initializers even though the default
  // member initializers below cover them.
  explicit Worst(const char *n) : name(n) {}

  const char *name = "";
  double rot_deg = 0.0;
  double dt_rel = 0.0;
  double err_rel = 0.0;      // only over cases whose reference error clears kErrFloor
  double err_abs = 0.0;
  double max_err_ref = 0.0;
  int cases = 0;
  int err_rel_cases = 0;     // cases the relative error metric applies to
  int validity_mismatches = 0;
  std::string worst_label;

  void Add(const Delta &d, const std::string &label) {
    ++cases;
    if (d.validity_mismatch) {
      ++validity_mismatches;
      return;
    }
    if (!d.both_valid) return;
    if (d.rot_deg > rot_deg || d.dt_rel > dt_rel) {
      if (d.rot_deg > rot_deg) rot_deg = d.rot_deg;
      if (d.dt_rel > dt_rel) dt_rel = d.dt_rel;
      worst_label = label;
    }
    // A synthetic case is fitted essentially exactly, so both error scalars
    // land near zero and their ratio is noise. Only compare relatively once
    // the reference error is large enough for the ratio to mean anything.
    err_abs = std::max(err_abs, d.err_abs);
    max_err_ref = std::max(max_err_ref, std::fabs(d.err_ref));
    if (std::fabs(d.err_ref) > kErrFloor) {
      ++err_rel_cases;
      err_rel = std::max(err_rel, d.err_rel);
    }
  }

  void Print() const {
    std::printf("  %-22s cases=%-5d worst_rot=%.3e deg  worst_dt_rel=%.3e  "
                "worst_err_abs=%.3e (rel=%.3e over %d/%d cases, max|err_ref|=%.3e)",
                name, cases, rot_deg, dt_rel, err_abs, err_rel, err_rel_cases, cases,
                max_err_ref);
    if (validity_mismatches > 0) std::printf("  VALIDITY_MISMATCH=%d", validity_mismatches);
    if (!worst_label.empty()) std::printf("   [worst: %s]", worst_label.c_str());
    std::printf("\n");
  }
};

// --- reference wrapper ----------------------------------------------------

// Builds the apriltag_detection_t libapriltag's pose entry points consume.
struct RefDetection {
  apriltag_detection_t det{};
  ~RefDetection() {
    if (det.H) matd_destroy(det.H);
  }
};

void FillRefDetection(RefDetection *rd, const double corners[4][2], const double H[3][3]) {
  rd->det.id = 0;
  rd->det.hamming = 0;
  rd->det.decision_margin = 50.0f;
  for (int i = 0; i < 4; ++i) {
    rd->det.p[i][0] = corners[i][0];
    rd->det.p[i][1] = corners[i][1];
  }
  rd->det.c[0] = 0.0;
  rd->det.c[1] = 0.0;
  rd->det.H = matd_create(3, 3);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) MATD_EL(rd->det.H, a, b) = H[a][b];
}

apriltag_detection_info_t MakeInfo(RefDetection *rd, const CameraIntrinsics &intr,
                                   double tagsize) {
  apriltag_detection_info_t info;
  info.det = &rd->det;
  info.tagsize = tagsize;
  info.fx = intr.fx;
  info.fy = intr.fy;
  info.cx = intr.cx;
  info.cy = intr.cy;
  return info;
}

// --- synthetic case generation --------------------------------------------

struct SynthCase {
  double corners[4][2];
  double H[3][3];
  double truth_R[3][3];
  double truth_t[3];
  std::string label;
  bool ok = false;
};

void RotZ(double a, double out[3][3]) {
  const double c = std::cos(a), s = std::sin(a);
  const double m[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
  std::memcpy(out, m, sizeof(m));
}
void RotY(double a, double out[3][3]) {
  const double c = std::cos(a), s = std::sin(a);
  const double m[3][3] = {{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
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

// Projects the four tag corners for a known pose and derives H from them the
// same way the real pipeline would (homography_compute over the corner
// correspondences), so the reference and the port both see a realistic H.
SynthCase MakeSynthCase(const CameraIntrinsics &intr, double tagsize, double dist, double tilt,
                        double spin, double off_x, double off_y, const std::string &label) {
  SynthCase c;
  c.label = label;

  double Ry[3][3], Rz[3][3], R[3][3];
  RotY(tilt, Ry);
  RotZ(spin, Rz);
  Mul33(Ry, Rz, R);
  std::memcpy(c.truth_R, R, sizeof(R));
  c.truth_t[0] = off_x;
  c.truth_t[1] = off_y;
  c.truth_t[2] = dist;

  const double s = tagsize / 2.0;
  // Same corner order and object-space layout libapriltag's pose code uses.
  const double obj[4][3] = {{-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};

  for (int i = 0; i < 4; ++i) {
    double cam[3];
    for (int a = 0; a < 3; ++a) {
      cam[a] = R[a][0] * obj[i][0] + R[a][1] * obj[i][1] + R[a][2] * obj[i][2] + c.truth_t[a];
    }
    if (!(cam[2] > 1e-6)) return c;  // behind or on the camera plane
    c.corners[i][0] = intr.fx * cam[0] / cam[2] + intr.cx;
    c.corners[i][1] = intr.fy * cam[1] / cam[2] + intr.cy;
  }

  // H from normalized tag coordinates to pixels, as the detector produces it.
  zarray_t *corr = zarray_create(sizeof(float[4]));
  const float norm[4][2] = {{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
  for (int i = 0; i < 4; ++i) {
    float row[4] = {norm[i][0], norm[i][1], static_cast<float>(c.corners[i][0]),
                    static_cast<float>(c.corners[i][1])};
    zarray_add(corr, &row);
  }
  matd_t *H = homography_compute(corr, HOMOGRAPHY_COMPUTE_FLAG_SVD);
  zarray_destroy(corr);
  if (H == nullptr) return c;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) c.H[a][b] = MATD_EL(H, a, b);
  matd_destroy(H);
  c.ok = true;
  return c;
}

// --- one case, all four ladder levels ------------------------------------

struct LadderStats {
  Worst l1{"L1 seed"};
  Worst l2{"L2 solution1"};
  Worst l3{"L3 solution2"};
  Worst l4{"L4 final pick"};
  int branch_disagreements = 0;
  int branch_disagreements_tied = 0;
  double worst_truth_rot_ours = 0.0;
  double worst_truth_rot_ref = 0.0;
  double worst_truth_dt_rel_ours = 0.0;
  double worst_truth_dt_rel_ref = 0.0;
};

void RunCase(const PoseEstimator &est, const CameraIntrinsics &intr, double tagsize,
             const double corners[4][2], const double H[3][3], const std::string &label,
             LadderStats *st, const double truth_R[3][3], const double truth_t[3]) {
  RefDetection rd;
  FillRefDetection(&rd, corners, H);
  apriltag_detection_info_t info = MakeInfo(&rd, intr, tagsize);

  // L1: seed
  {
    apriltag_pose_t ref{};
    estimate_pose_for_tag_homography(&info, &ref);
    const TagPose ours = est.EstimateSeed(H);
    st->l1.Add(Compare(ours, ref.R, ref.t, 0.0, ref.R != nullptr), label);
    if (ref.R) matd_destroy(ref.R);
    if (ref.t) matd_destroy(ref.t);
  }

  // L2/L3: both solutions, unranked
  apriltag_pose_t ref1{}, ref2{};
  double ref_err1 = HUGE_VAL, ref_err2 = HUGE_VAL;
  estimate_tag_pose_orthogonal_iteration(&info, &ref_err1, &ref1, &ref_err2, &ref2,
                                         PoseEstimator::kDefaultIterations);
  const TagPosePair ours_both = est.EstimateBoth(corners, H);

  st->l2.Add(Compare(ours_both.solution1, ref1.R, ref1.t, ref_err1, ref1.R != nullptr), label);
  const bool ref2_valid = (ref2.R != nullptr) && std::isfinite(ref_err2);
  if (ours_both.solution2.valid == ref2_valid && ref2_valid) {
    st->l3.Add(Compare(ours_both.solution2, ref2.R, ref2.t, ref_err2, true), label);
  } else if (ours_both.solution2.valid != ref2_valid) {
    // A different count of ambiguity minima is a legitimate outcome of tiny
    // numerical differences (see the plan's quality-loss list), so it is
    // counted rather than treated as a pass or a hard failure.
    Delta d;
    d.validity_mismatch = true;
    st->l3.Add(d, label);
  } else {
    ++st->l3.cases;
  }

  // L4: final pick, and whether both sides chose the same branch
  {
    apriltag_pose_t ref_final{};
    const double ref_final_err = estimate_tag_pose(&info, &ref_final);
    const TagPose ours = est.Estimate(corners, H);
    st->l4.Add(Compare(ours, ref_final.R, ref_final.t, ref_final_err, ref_final.R != nullptr),
               label);

    const bool ref_took_2 = (ref_err2 < ref_err1);
    const bool ours_took_2 =
        ours_both.solution2.valid && (ours_both.solution2.error < ours_both.solution1.error);
    if (ref_took_2 != ours_took_2) {
      ++st->branch_disagreements;
      // A disagreement is benign exactly when the two errors are effectively
      // tied, because then the choice is decided by the last bits.
      const double e1 = ours_both.solution1.valid ? ours_both.solution1.error : HUGE_VAL;
      const double e2 = ours_both.solution2.valid ? ours_both.solution2.error : HUGE_VAL;
      const double scale = std::max(std::min(std::fabs(e1), std::fabs(e2)), 1e-300);
      if (std::isfinite(e1) && std::isfinite(e2) && std::fabs(e1 - e2) / scale < 1e-6) {
        ++st->branch_disagreements_tied;
      } else {
        std::printf("    branch disagreement (NOT a tie) on %s: ours e1=%.9g e2=%.9g | "
                    "ref e1=%.9g e2=%.9g\n",
                    label.c_str(), e1, e2, ref_err1, ref_err2);
      }
    }

    // Absolute accuracy against ground truth, when we have it.
    if (truth_R != nullptr && ours.valid && ref_final.R != nullptr) {
      double ref_Rm[3][3], ref_tv[3];
      for (int a = 0; a < 3; ++a) {
        ref_tv[a] = MATD_EL(ref_final.t, a, 0);
        for (int b = 0; b < 3; ++b) ref_Rm[a][b] = MATD_EL(ref_final.R, a, b);
      }
      double d_ours[3], d_ref[3];
      for (int a = 0; a < 3; ++a) {
        d_ours[a] = ours.t[a] - truth_t[a];
        d_ref[a] = ref_tv[a] - truth_t[a];
      }
      const double tn = std::max(Norm3(truth_t), 1e-300);
      st->worst_truth_rot_ours = std::max(st->worst_truth_rot_ours,
                                          RotationAngleDeg(ours.R, truth_R));
      st->worst_truth_rot_ref = std::max(st->worst_truth_rot_ref,
                                         RotationAngleDeg(ref_Rm, truth_R));
      st->worst_truth_dt_rel_ours = std::max(st->worst_truth_dt_rel_ours, Norm3(d_ours) / tn);
      st->worst_truth_dt_rel_ref = std::max(st->worst_truth_dt_rel_ref, Norm3(d_ref) / tn);
    }

    if (ref_final.R) matd_destroy(ref_final.R);
    if (ref_final.t) matd_destroy(ref_final.t);
  }

  if (ref1.R) matd_destroy(ref1.R);
  if (ref1.t) matd_destroy(ref1.t);
  if (ref2.R) matd_destroy(ref2.R);
  if (ref2.t) matd_destroy(ref2.t);
}

double NowMs() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
  CameraIntrinsics intr{1400.0, 1400.0, 960.0, 540.0};
  double tagsize = 0.1651;
  std::string data_path;
  uint32_t decimation = 2;
  int timing_iters = 2000;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--tagsize") tagsize = std::stod(next("--tagsize"));
    else if (a == "--fx") intr.fx = std::stod(next("--fx"));
    else if (a == "--fy") intr.fy = std::stod(next("--fy"));
    else if (a == "--cx") intr.cx = std::stod(next("--cx"));
    else if (a == "--cy") intr.cy = std::stod(next("--cy"));
    else if (a == "--data") data_path = next("--data");
    else if (a == "--decimation") decimation = static_cast<uint32_t>(std::stoul(next("--decimation")));
    else if (a == "--timing-iters") timing_iters = std::stoi(next("--timing-iters"));
    else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 1;
    }
  }

  // convergence_tol = 0: libapriltag always runs its full iteration count,
  // so the ladder below compares like with like. The early exit is a
  // deliberate deviation and is measured separately, further down.
  const PoseEstimator est(intr, tagsize, 0, 0.0);
  std::printf("PoseEstimator vs libapriltag apriltag_pose.c\n");
  std::printf("  intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f, tagsize=%.4f m, threads=%u\n\n",
              intr.fx, intr.fy, intr.cx, intr.cy, tagsize, est.threads());

  // ---------------- synthetic sweep ----------------
  LadderStats synth;
  int synth_cases = 0, synth_skipped = 0;
  const double dists[] = {0.3, 0.6, 1.0, 2.0, 3.5, 5.0};
  const double tilts_deg[] = {0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0};
  const double spins_deg[] = {0.0, 17.0, 45.0, 73.0, 90.0};
  const double offsets[] = {0.0, 0.25};
  for (double d : dists) {
    for (double tilt : tilts_deg) {
      for (double spin : spins_deg) {
        for (double off : offsets) {
          char label[160];
          std::snprintf(label, sizeof(label), "d=%.2f tilt=%.0f spin=%.0f off=%.2f", d, tilt,
                        spin, off);
          const SynthCase c =
              MakeSynthCase(intr, tagsize, d, tilt * kPi / 180.0, spin * kPi / 180.0,
                            off * d, -off * d * 0.5, label);
          if (!c.ok) {
            ++synth_skipped;
            continue;
          }
          RunCase(est, intr, tagsize, c.corners, c.H, label, &synth, c.truth_R, c.truth_t);
          ++synth_cases;
        }
      }
    }
  }
  std::printf("Synthetic sweep: %d cases (%d skipped as unprojectable)\n", synth_cases,
              synth_skipped);
  synth.l1.Print();
  synth.l2.Print();
  synth.l3.Print();
  synth.l4.Print();
  std::printf("  branch disagreements: %d (%d of them numerical ties)\n",
              synth.branch_disagreements, synth.branch_disagreements_tied);
  std::printf("  absolute error vs ground truth  ours: rot<=%.4f deg, |dt|/|t|<=%.3e\n",
              synth.worst_truth_rot_ours, synth.worst_truth_dt_rel_ours);
  std::printf("                                   ref: rot<=%.4f deg, |dt|/|t|<=%.3e\n\n",
              synth.worst_truth_rot_ref, synth.worst_truth_dt_rel_ref);

  // ---------------- degenerate geometry ----------------
  LadderStats degen;
  int degen_cases = 0;
  struct DegenSpec {
    const char *name;
    double dist, tilt_deg, spin_deg, off_x_frac, off_y_frac;
  };
  const DegenSpec degens[] = {
      // t nearly along +x, the Gram-Schmidt degeneracy in fix_pose_ambiguities
      {"t_parallel_to_ex", 0.02, 10.0, 0.0, 40.0, 0.0},
      {"fronto_parallel", 1.0, 0.0, 0.0, 0.0, 0.0},
      {"fronto_parallel_spun", 1.0, 0.0, 45.0, 0.0, 0.0},
      {"very_close", 0.12, 20.0, 30.0, 0.0, 0.0},
      {"very_far", 40.0, 25.0, 10.0, 0.0, 0.0},
      {"extreme_tilt", 1.0, 85.0, 20.0, 0.0, 0.0},
      {"extreme_tilt_2", 1.5, 88.0, 0.0, 0.1, 0.0},
      {"far_off_axis", 2.0, 30.0, 15.0, 0.9, 0.9},
  };
  for (const DegenSpec &s : degens) {
    const SynthCase c =
        MakeSynthCase(intr, tagsize, s.dist, s.tilt_deg * kPi / 180.0,
                      s.spin_deg * kPi / 180.0, s.off_x_frac * s.dist, s.off_y_frac * s.dist,
                      s.name);
    if (!c.ok) {
      std::printf("  %-22s unprojectable, skipped\n", s.name);
      continue;
    }
    RunCase(est, intr, tagsize, c.corners, c.H, s.name, &degen, c.truth_R, c.truth_t);
    ++degen_cases;
  }
  std::printf("Degenerate geometry: %d cases\n", degen_cases);
  degen.l1.Print();
  degen.l2.Print();
  degen.l3.Print();
  degen.l4.Print();
  std::printf("  branch disagreements: %d (%d of them numerical ties)\n\n",
              degen.branch_disagreements, degen.branch_disagreements_tied);

  // ---------------- real detections ----------------
  if (!data_path.empty()) {
    std::vector<uint8_t> gray;
    uint32_t width = 0, height = 0;
    if (!apriltag_vulkan::LoadGrayPgm(data_path, &gray, &width, &height)) {
      std::fprintf(stderr, "could not load %s as a P5 PGM\n", data_path.c_str());
      return 1;
    }
    apriltag_family_t *tf = nullptr;
    if (!setup_tag_family(&tf, "tag36h11")) return 1;
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->refine_edges = false;

    apriltag_vulkan::vk::Context ctx;
    apriltag_vulkan::DetectorConfig cfg;
    cfg.width = width;
    cfg.height = height;
    cfg.decimation = decimation;
    cfg.tag_width = static_cast<uint32_t>(tf->width_at_border);
    cfg.reversed_border = tf->reversed_border;
    cfg.normal_border = !tf->reversed_border;

    apriltag_vulkan::GpuDetector detector(ctx, cfg);
    apriltag_vulkan::QuadDecode quad_decode(cfg);
    apriltag_vulkan::TagDecoder tag_decoder(td, decimation);

    detector.Detect(gray.data());
    const std::vector<apriltag_vulkan::DetectedQuad> quads =
        quad_decode.Decode(detector.last_selected_extents, detector.last_line_fit_points);
    zarray_t *dets =
        tag_decoder.Decode(quads, gray.data(), width, height, cfg.reversed_border);

    LadderStats real;
    int real_cases = 0;
    for (int i = 0; i < zarray_size(dets); ++i) {
      apriltag_detection_t *det = nullptr;
      zarray_get(dets, i, &det);
      if (det == nullptr || det->H == nullptr) continue;
      double H[3][3], corners[4][2];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) H[a][b] = MATD_EL(det->H, a, b);
      for (int c = 0; c < 4; ++c) {
        corners[c][0] = det->p[c][0];
        corners[c][1] = det->p[c][1];
      }
      char label[64];
      std::snprintf(label, sizeof(label), "real tag id=%d", det->id);
      RunCase(est, intr, tagsize, corners, H, label, &real, nullptr, nullptr);
      ++real_cases;
    }
    std::printf("Real detections from %s (decimation %u): %d tag(s)\n", data_path.c_str(),
                decimation, real_cases);
    real.l1.Print();
    real.l2.Print();
    real.l3.Print();
    real.l4.Print();
    std::printf("  branch disagreements: %d (%d of them numerical ties)\n\n",
                real.branch_disagreements, real.branch_disagreements_tied);

    // ---------------- timing, on a real detection ----------------
    if (real_cases > 0 && timing_iters > 0) {
      apriltag_detection_t *det = nullptr;
      zarray_get(dets, 0, &det);
      double H[3][3], corners[4][2];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) H[a][b] = MATD_EL(det->H, a, b);
      for (int c = 0; c < 4; ++c) {
        corners[c][0] = det->p[c][0];
        corners[c][1] = det->p[c][1];
      }
      RefDetection rd;
      FillRefDetection(&rd, corners, H);
      apriltag_detection_info_t info = MakeInfo(&rd, intr, tagsize);

      for (int i = 0; i < 100; ++i) {
        apriltag_pose_t p{};
        estimate_tag_pose(&info, &p);
        matd_destroy(p.R);
        matd_destroy(p.t);
        est.Estimate(corners, H);
      }

      double ref_best = 1e30, ref_total = 0.0;
      for (int i = 0; i < timing_iters; ++i) {
        const double s = NowMs();
        apriltag_pose_t p{};
        estimate_tag_pose(&info, &p);
        const double dt = NowMs() - s;
        matd_destroy(p.R);
        matd_destroy(p.t);
        ref_best = std::min(ref_best, dt);
        ref_total += dt;
      }
      // Both configurations: `est` runs the full iteration count (the
      // libapriltag-parity setting the ladder above verifies), `early_est`
      // uses the default convergence tolerance. The gap between those two
      // rows is exactly what the early exit buys.
      const PoseEstimator early_est(intr, tagsize, 1);
      double our_best = 1e30, our_total = 0.0;
      double sink = 0.0;  // keeps the calls from being optimized away
      for (int i = 0; i < timing_iters; ++i) {
        const double s = NowMs();
        const TagPose p = est.Estimate(corners, H);
        const double dt = NowMs() - s;
        sink += p.t[2] + p.error;
        our_best = std::min(our_best, dt);
        our_total += dt;
      }
      double early_best = 1e30, early_total = 0.0;
      for (int i = 0; i < timing_iters; ++i) {
        const double s = NowMs();
        const TagPose p = early_est.Estimate(corners, H);
        const double dt = NowMs() - s;
        sink += p.t[2] + p.error;
        early_best = std::min(early_best, dt);
        early_total += dt;
      }
      // Consume `sink` so the loop body cannot be discarded, without a
      // zero-length printf (which warns under -Wformat-zero-length).
      if (!std::isfinite(sink)) {
        std::fprintf(stderr, "non-finite pose accumulated over timing loop\n");
        return 1;
      }
      const TagPosePair early_pair = early_est.EstimateBoth(corners, H);
      std::printf("Timing over %d calls (single tag):\n", timing_iters);
      std::printf("  libapriltag estimate_tag_pose      : best=%.5f ms  mean=%.5f ms\n",
                  ref_best, ref_total / timing_iters);
      std::printf("  PoseEstimator, fixed %2d iterations : best=%.5f ms  mean=%.5f ms  (%.1fx)\n",
                  PoseEstimator::kDefaultIterations, our_best, our_total / timing_iters,
                  our_total > 0.0 ? ref_total / our_total : 0.0);
      std::printf("  PoseEstimator, early exit %.0e    : best=%.5f ms  mean=%.5f ms  (%.1fx)"
                  "   [%d + %d iterations used]\n",
                  early_est.convergence_tol(), early_best, early_total / timing_iters,
                  early_total > 0.0 ? ref_total / early_total : 0.0,
                  early_pair.solution1.iterations, early_pair.solution2.iterations);
    }

    apriltag_detector_destroy(td);
    teardown_tag_family(&tf, "tag36h11");
  }

  // ---------------- EstimateAll: threaded batch path ----------------
  //
  // Distinct from everything above, which exercises the single-detection
  // entry points. EstimateAll writes results by index rather than in
  // completion order, so its output must be bit-identical to calling
  // Estimate() serially - at any thread count. That is what is asserted
  // here, since "it produced plausible poses" would not catch an
  // index/ordering bug.
  bool batch_ok = true;
  {
    // Reuse a spread of the synthetic sweep as a multi-tag frame.
    std::vector<SynthCase> cases;
    for (double d : {0.4, 0.8, 1.5, 3.0}) {
      for (double tilt : {0.0, 20.0, 50.0}) {
        for (double spin : {0.0, 30.0, 60.0}) {
          SynthCase c = MakeSynthCase(intr, tagsize, d, tilt * kPi / 180.0,
                                      spin * kPi / 180.0, 0.1 * d, -0.05 * d, "batch");
          if (c.ok) cases.push_back(c);
        }
      }
    }

    std::vector<RefDetection> rds(cases.size());
    std::vector<const apriltag_detection_t *> dets;
    dets.reserve(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
      FillRefDetection(&rds[i], cases[i].corners, cases[i].H);
      rds[i].det.id = static_cast<int>(i);
      dets.push_back(&rds[i].det);
    }

    // Serial reference: the single-detection path, one at a time.
    // Serial reference in the SHIPPING configuration (default tolerance),
    // single-threaded, so this isolates the batching from everything else.
    const PoseEstimator serial_est(intr, tagsize, 1);
    std::vector<TagPose> serial(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
      serial[i] = serial_est.Estimate(cases[i].corners, cases[i].H);
    }

    for (uint32_t threads : {1u, 2u, 4u, 8u}) {
      const PoseEstimator batched(intr, tagsize, threads);
      std::vector<TagPose> out;
      batched.EstimateAll(dets, out);
      if (out.size() != cases.size()) {
        std::printf("  EstimateAll(threads=%u): size %zu != %zu  FAIL\n", threads, out.size(),
                    cases.size());
        batch_ok = false;
        continue;
      }
      int mismatches = 0;
      for (size_t i = 0; i < cases.size(); ++i) {
        if (out[i].valid != serial[i].valid) {
          ++mismatches;
          continue;
        }
        if (!out[i].valid) continue;
        // Bit-identical is the requirement here, not "close": both sides run
        // the same code on the same input, so any difference means the batch
        // path mixed up results.
        bool same = (out[i].error == serial[i].error);
        for (int a = 0; a < 3 && same; ++a) {
          if (out[i].t[a] != serial[i].t[a]) same = false;
          for (int b = 0; b < 3 && same; ++b) {
            if (out[i].R[a][b] != serial[i].R[a][b]) same = false;
          }
        }
        if (!same) ++mismatches;
      }
      std::printf("  EstimateAll(threads=%u, %zu tags, pool=%u): %s\n", threads, cases.size(),
                  batched.threads(), mismatches == 0 ? "bit-identical to serial" : "MISMATCH");
      if (mismatches != 0) batch_ok = false;
    }
  }
  std::printf("\n");

  // ---------------- early exit (Phase B) ----------------
  //
  // The convergence early exit is NOT part of the libapriltag port -
  // libapriltag always runs its full 50 steps - so libapriltag is the wrong
  // oracle for it. It is checked two ways instead:
  //
  //   1. against the fixed-iteration solver (convergence_tol = 0), which is
  //      the configuration verified against libapriltag, and
  //   2. against synthetic ground truth, because agreeing with the
  //      fixed-iteration solver would mean nothing if both had drifted.
  //
  // The tolerance is swept rather than assumed, so the default is chosen from
  // the measured iterations/accuracy tradeoff instead of guessed.
  bool early_ok = true;
  {
    const PoseEstimator exact(intr, tagsize, 1, 0.0);  // fixed 50 steps

    // Ground-truth accuracy of the fixed-iteration solver, which is the bar
    // the early exit must not fall below.
    double base_truth_rot = 0.0, base_truth_dt = 0.0;

    struct Row {
      double tol;
      double rot_vs_fixed, dt_vs_fixed;
      double truth_rot, truth_dt;
      long iters;
      int max_iters, pick_flips, capped;
      double ms;  // wall clock over the whole case set, best of a few passes
    };
    std::vector<Row> rows;
    const double tols[] = {0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4};

    const double dists2[] = {0.3, 0.6, 1.0, 2.0, 3.5, 5.0};
    const double tilts2[] = {0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0};
    const double spins2[] = {0.0, 17.0, 45.0, 73.0, 90.0};

    std::vector<SynthCase> cases;
    for (double d : dists2)
      for (double tl : tilts2)
        for (double sp : spins2) {
          char label[160];
          std::snprintf(label, sizeof(label), "d=%.2f tilt=%.0f spin=%.0f", d, tl, sp);
          SynthCase c = MakeSynthCase(intr, tagsize, d, tl * kPi / 180.0, sp * kPi / 180.0,
                                      0.1 * d, -0.05 * d, label);
          if (c.ok) cases.push_back(c);
        }

    // Fixed-iteration reference results, computed once.
    std::vector<TagPose> ref(cases.size());
    std::vector<bool> ref_pick2(cases.size(), false);
    for (size_t i = 0; i < cases.size(); ++i) {
      ref[i] = exact.Estimate(cases[i].corners, cases[i].H);
      const TagPosePair b = exact.EstimateBoth(cases[i].corners, cases[i].H);
      ref_pick2[i] = b.solution2.valid && b.solution2.error < b.solution1.error;
      if (!ref[i].valid) continue;
      const double tn = std::max(Norm3(cases[i].truth_t), 1e-300);
      double d3[3];
      for (int a = 0; a < 3; ++a) d3[a] = ref[i].t[a] - cases[i].truth_t[a];
      base_truth_rot = std::max(base_truth_rot, RotationAngleDeg(ref[i].R, cases[i].truth_R));
      base_truth_dt = std::max(base_truth_dt, Norm3(d3) / tn);
    }

    for (double tol : tols) {
      const PoseEstimator est_t(intr, tagsize, 1, tol);
      Row r{tol, 0, 0, 0, 0, 0, 0, 0, 0, 0.0};
      // Wall clock over the whole case set. Iteration count is only a proxy
      // for cost - the seed, the per-point F precompute and the ambiguity
      // search (with its quartic solve) are fixed overhead the early exit
      // cannot touch - so measure the thing that matters directly.
      {
        double best_ms = 1e30, sink = 0.0;
        for (int pass = 0; pass < 5; ++pass) {
          const double t0 = NowMs();
          for (const SynthCase &c : cases) {
            const TagPose p = est_t.Estimate(c.corners, c.H);
            sink += p.t[2] + p.error;
          }
          const double dt = NowMs() - t0;
          if (dt < best_ms) best_ms = dt;
        }
        if (!std::isfinite(sink)) return 1;
        r.ms = best_ms;
      }
      for (size_t i = 0; i < cases.size(); ++i) {
        const TagPose p = est_t.Estimate(cases[i].corners, cases[i].H);
        if (!p.valid || !ref[i].valid) {
          if (p.valid != ref[i].valid) early_ok = false;
          continue;
        }
        r.rot_vs_fixed = std::max(r.rot_vs_fixed, RotationAngleDeg(p.R, ref[i].R));
        double dd[3];
        for (int a = 0; a < 3; ++a) dd[a] = p.t[a] - ref[i].t[a];
        r.dt_vs_fixed = std::max(r.dt_vs_fixed, Norm3(dd) / std::max(Norm3(ref[i].t), 1e-300));

        const double tn = std::max(Norm3(cases[i].truth_t), 1e-300);
        double d3[3];
        for (int a = 0; a < 3; ++a) d3[a] = p.t[a] - cases[i].truth_t[a];
        r.truth_rot = std::max(r.truth_rot, RotationAngleDeg(p.R, cases[i].truth_R));
        r.truth_dt = std::max(r.truth_dt, Norm3(d3) / tn);

        const TagPosePair b = est_t.EstimateBoth(cases[i].corners, cases[i].H);
        r.iters += b.solution1.iterations + b.solution2.iterations;
        r.max_iters = std::max(r.max_iters, b.solution1.iterations);
        r.max_iters = std::max(r.max_iters, b.solution2.iterations);
        if (b.solution1.iterations >= PoseEstimator::kDefaultIterations) ++r.capped;
        const bool pick2 = b.solution2.valid && b.solution2.error < b.solution1.error;
        if (pick2 != ref_pick2[i]) ++r.pick_flips;
      }
      rows.push_back(r);
    }

    std::printf("Early exit tolerance sweep over %zu synthetic cases\n", cases.size());
    std::printf("  fixed-iteration accuracy vs ground truth: rot<=%.6f deg, |dt|/|t|<=%.3e\n",
                base_truth_rot, base_truth_dt);
    std::printf("  %-9s %-12s %-11s %-13s %-11s %-8s %-7s %-7s %s\n", "tol", "rot_vs_fix",
                "dt_vs_fix", "truth_rot", "truth_dt", "iters", "worst", "capped", "ms/set");
    const long base_iters = rows.empty() ? 0 : rows[0].iters;
    for (const Row &r : rows) {
      char tolbuf[16];
      if (r.tol == 0.0) {
        std::snprintf(tolbuf, sizeof(tolbuf), "off");
      } else {
        std::snprintf(tolbuf, sizeof(tolbuf), "%.0e", r.tol);
      }
      std::printf("  %-9s %-12.3e %-11.3e %-13.6f %-11.3e %-8ld %-7d %-7d %.3f\n", tolbuf,
                  r.rot_vs_fixed, r.dt_vs_fixed, r.truth_rot, r.truth_dt, r.iters, r.max_iters,
                  r.capped, r.ms);
      (void)base_iters;
    }
    if (base_iters > 0) {
      std::printf("  iterations saved vs off:");
      for (const Row &r : rows) {
        if (r.tol == 0.0) continue;
        std::printf("  %.0e:%.0f%%", r.tol, 100.0 * (1.0 - double(r.iters) / double(base_iters)));
      }
      std::printf("\n");
    }
    const double base_ms = rows.empty() ? 0.0 : rows[0].ms;
    if (base_ms > 0.0) {
      std::printf("  WALL CLOCK saved vs off:");
      for (const Row &r : rows) {
        if (r.tol == 0.0) continue;
        std::printf("  %.0e:%.0f%%", r.tol, 100.0 * (1.0 - r.ms / base_ms));
      }
      std::printf("   (iterations are only a proxy; this is the real figure)\n");
    }

    // Gate on the DEFAULT tolerance only, and gate it on the right thing.
    //
    // Deviating from the fixed-iteration solver by roughly the tolerance is
    // what a tolerance MEANS, so a fixed absolute bound on that deviation
    // would just be a restatement of the tolerance. Two bounds that actually
    // carry information instead:
    //
    //   1. Accuracy against ground truth must not regress at all. This is
    //      the one that matters to a caller, and it is why "early exit and
    //      fixed iteration agree" would be insufficient on its own - they
    //      could agree and both have drifted.
    //   2. The deviation from the fixed solver must stay an order of
    //      magnitude below the accuracy limit the input itself imposes.
    //      Being closer to the fixed solver than the fixed solver is to
    //      reality means the early exit cannot be what limits the answer.
    //
    // Plus: the deviation must scale with the tolerance rather than exceed
    // it wildly, which catches an exit criterion that fires too early.
    for (const Row &r : rows) {
      if (r.tol != PoseEstimator::kDefaultConvergenceTol) continue;
      const bool truth_not_worse = r.truth_rot <= base_truth_rot + 1e-9 &&
                                   r.truth_dt <= base_truth_dt * (1.0 + 1e-6) + 1e-12;
      const bool far_below_accuracy_limit =
          r.rot_vs_fixed < base_truth_rot * 0.1 && r.dt_vs_fixed < base_truth_dt * 0.1;
      // Translation deviation is dimensionless-relative, same units as the
      // tolerance, so it should land within a small multiple of it.
      const bool consistent_with_tol = r.dt_vs_fixed < r.tol * 100.0;
      early_ok = early_ok && truth_not_worse && far_below_accuracy_limit &&
                 consistent_with_tol && r.pick_flips == 0;
      std::printf("  default tol %.0e: truth_not_worse=%d far_below_limit=%d "
                  "consistent_with_tol=%d pick_flips=%d\n",
                  r.tol, truth_not_worse ? 1 : 0, far_below_accuracy_limit ? 1 : 0,
                  consistent_with_tol ? 1 : 0, r.pick_flips);
    }
  }
  std::printf("\n");

  // ---------------- verdict ----------------
  //
  // Thresholds are calibrated against what the deliberate divergences from
  // libapriltag actually cost, with margin - not pulled from thin air:
  //
  //  * The seed (L1) is where the one intentional precision change lives:
  //    libapriltag computes homography_to_pose's scale factor with
  //    single-precision sqrtf, this port uses double. Measured effect on the
  //    seed is ~1.4e-7 relative in translation, so L1 gets its own looser
  //    translation bound. Orthogonal iteration then washes that out - by L2
  //    translation agrees to ~4e-10.
  //
  //  * The rotation bound is shared. The measured worst divergence is
  //    ~2.1e-6 deg, and 1e-4 deg leaves ~48x margin while still being ~30x
  //    TIGHTER than the error both implementations share against synthetic
  //    ground truth (~3.2e-3 deg). That is the honest framing: the port
  //    tracks libapriltag far more closely than either tracks reality.
  //
  //  * The object-space error is compared in absolute terms, because a
  //    synthetic case is fitted exactly and both scalars sit at ~1e-13 or
  //    below, where a ratio is pure noise. Relative agreement is asserted
  //    only over the cases clearing kErrFloor.
  auto LevelOk = [](const Worst &w, double max_rot_deg, double max_dt_rel) {
    return w.validity_mismatches == 0 && w.rot_deg < max_rot_deg && w.dt_rel < max_dt_rel &&
           w.err_abs < 1e-12 && w.err_rel < 1e-9;
  };
  constexpr double kMaxRotDeg = 1e-4;
  constexpr double kMaxDtRelSeed = 1e-5;   // the deliberate sqrtf -> sqrt change
  constexpr double kMaxDtRelRefined = 1e-7;

  auto SetOk = [&](const LadderStats &st, const char *what, bool require_l3) {
    bool ok = LevelOk(st.l1, kMaxRotDeg, kMaxDtRelSeed) &&
              LevelOk(st.l2, kMaxRotDeg, kMaxDtRelRefined) &&
              LevelOk(st.l4, kMaxRotDeg, kMaxDtRelRefined);
    // L3 exercises fix_pose_ambiguities, where a differing count of minima is
    // a documented legitimate outcome of last-bit differences; its validity
    // mismatches are reported but only gated where asked.
    if (require_l3) {
      ok = ok && st.l3.rot_deg < kMaxRotDeg && st.l3.dt_rel < kMaxDtRelRefined &&
           st.l3.err_abs < 1e-12 && st.l3.err_rel < 1e-9;
    }
    // Every branch disagreement must be a numerical tie.
    ok = ok && (st.branch_disagreements == st.branch_disagreements_tied);
    std::printf("  %-22s %s\n", what, ok ? "PASS" : "FAIL");
    return ok;
  };

  std::printf("VERDICT\n");
  bool all_ok = SetOk(synth, "synthetic sweep", true);
  all_ok = SetOk(degen, "degenerate geometry", true) && all_ok;
  std::printf("  %-22s %s\n", "EstimateAll batch", batch_ok ? "PASS" : "FAIL");
  std::printf("  %-22s %s\n", "early exit", early_ok ? "PASS" : "FAIL");
  all_ok = all_ok && batch_ok && early_ok;
  std::printf("  thresholds: rot < %.0e deg, |dt|/|t| < %.0e (seed %.0e), "
              "err_abs < 1e-12, err_rel < 1e-9\n",
              kMaxRotDeg, kMaxDtRelRefined, kMaxDtRelSeed);
  return all_ok ? 0 : 1;
}
