#include "vkapriltag/PoseEstimator.h"

#include <cmath>
#include <cstring>

namespace apriltag_vulkan {
namespace {

// Fixed-size stack arithmetic, replacing the matd_t/matd_op machinery
// libapriltag's apriltag_pose.c is built on. Every routine below writes into
// caller-provided storage; nothing here allocates.
//
// Layout is row-major: m[row][col].
using Mat3 = double[3][3];
using Vec3 = double[3];

constexpr int kN = 4;  // tag corners; libapriltag's n_points, always 4 here

void Mat3Identity(Mat3 out) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][j] = (i == j) ? 1.0 : 0.0;
}

void Mat3Copy(const Mat3 a, Mat3 out) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][j] = a[i][j];
}

// out = a * b
void Mat3Mul(const Mat3 a, const Mat3 b, Mat3 out) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  }
}

// out = a * b'
void Mat3MulTransposed(const Mat3 a, const Mat3 b, Mat3 out) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[i][k] * b[j][k];
      out[i][j] = s;
    }
  }
}

// out = a' * b
void Mat3TransposedMul(const Mat3 a, const Mat3 b, Mat3 out) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[k][i] * b[k][j];
      out[i][j] = s;
    }
  }
}

// out = a * x
void Mat3MulVec(const Mat3 a, const Vec3 x, Vec3 out) {
  for (int i = 0; i < 3; ++i) out[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
}

// out = a' * x
void Mat3TransposedMulVec(const Mat3 a, const Vec3 x, Vec3 out) {
  for (int i = 0; i < 3; ++i) out[i] = a[0][i] * x[0] + a[1][i] * x[1] + a[2][i] * x[2];
}

double Mat3Det(const Mat3 a) {
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

// Adjugate inverse. Returns false when `a` is singular to working precision.
bool Mat3Inverse(const Mat3 a, Mat3 out) {
  const double det = Mat3Det(a);
  if (!(std::fabs(det) > 0.0) || !std::isfinite(det)) return false;
  const double inv_det = 1.0 / det;
  out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
  out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
  out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
  out[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
  out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
  out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
  out[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
  out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
  out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (!std::isfinite(out[i][j])) return false;
  return true;
}

void Vec3Cross(const Vec3 a, const Vec3 b, Vec3 out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

double Vec3Dot(const Vec3 a, const Vec3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Normalizes in place. Returns false on a (near-)zero vector, which is the
// degenerate case libapriltag's matd_vec_normalize walks into silently.
bool Vec3Normalize(Vec3 v) {
  const double n = std::sqrt(Vec3Dot(v, v));
  if (!(n > 0.0) || !std::isfinite(n)) return false;
  for (int i = 0; i < 3; ++i) v[i] /= n;
  return true;
}

// --- Singular value decomposition of a 3x3 -------------------------------
//
// M = U * diag(sv) * V', singular values non-negative and sorted descending,
// matching matd_svd's own normalization (it sorts by descending magnitude and
// folds the sign into U).
//
// A real SVD is genuinely required by the caller below, not a
// polar-decomposition shortcut: orthogonal iteration's
// M3 = sum_j (q_j - q_mean) * p_res_j' is ALWAYS rank deficient, because the
// four tag corners are coplanar, so every p_res_j has a zero z component and
// M3's third column is identically zero. A Newton polar iteration needs
// M^-1 and collapses to a zero pose on exactly this input.
//
// Symmetric cyclic Jacobi on A = M'M yields V and the squared singular
// values; U's columns follow as M*v_i / sv_i. The column belonging to a zero
// singular value is unconstrained by that relation and is recovered as the
// cross product of the other two, which keeps U orthogonal.
void Svd3x3(const Mat3 M, Mat3 U, double sv[3], Mat3 V) {
  Mat3 A;
  Mat3TransposedMul(M, M, A);
  Mat3Identity(V);

  for (int sweep = 0; sweep < 24; ++sweep) {
    const double off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
    if (!(off > 1e-300)) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (!(std::fabs(A[p][q]) > 0.0)) continue;
        // Jacobi rotation zeroing A[p][q].
        const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < 3; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  // Eigenvalues of A are the squared singular values; sort descending.
  double lambda[3] = {A[0][0], A[1][1], A[2][2]};
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (lambda[order[j]] > lambda[order[i]]) {
        const int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
    }
  }

  Mat3 Vs;
  for (int c = 0; c < 3; ++c) {
    sv[c] = std::sqrt(lambda[order[c]] > 0.0 ? lambda[order[c]] : 0.0);
    for (int r = 0; r < 3; ++r) Vs[r][c] = V[r][order[c]];
  }
  Mat3Copy(Vs, V);

  const double tol = (sv[0] > 0.0 ? sv[0] : 1.0) * 1e-14;
  int deficient = -1;
  for (int c = 0; c < 3; ++c) {
    if (sv[c] > tol) {
      for (int r = 0; r < 3; ++r) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k) acc += M[r][k] * V[k][c];
        U[r][c] = acc / sv[c];
      }
    } else {
      // At most one deficient direction arises for the matrices this solver
      // produces (M3 loses exactly its third column). If more than one were
      // deficient the later columns would be filled from the earlier ones,
      // which is still orthogonal, just arbitrary - the same freedom
      // matd_svd has on such input.
      deficient = c;
    }
  }
  if (deficient >= 0) {
    const int a = (deficient + 1) % 3;
    const int b = (deficient + 2) % 3;
    Vec3 ua = {U[0][a], U[1][a], U[2][a]};
    Vec3 ub = {U[0][b], U[1][b], U[2][b]};
    Vec3 un;
    Vec3Cross(ua, ub, un);
    if (!Vec3Normalize(un)) {
      un[0] = 0.0;
      un[1] = 0.0;
      un[2] = 1.0;
    }
    for (int r = 0; r < 3; ++r) U[r][deficient] = un[r];
  }
}

// R = U * V' for M = U S V'. This is the orthogonal factor of M - the closest
// rotation (or reflection) to it.
void NearestOrthogonal(const Mat3 M, Mat3 out) {
  Mat3 U, V;
  double sv[3];
  Svd3x3(M, U, sv, V);
  Mat3MulTransposed(U, V, out);
}

// --- Polynomial helpers ---------------------------------------------------

// p(x) = p[0] + p[1]*x + ... by Horner.
//
// libapriltag's polyval sums p[i]*pow(x, i), i.e. one libm pow() call per
// term. That runs inside a 100-iteration Newton loop inside the recursive
// root finder below, and is the single largest avoidable cost in this file.
// Horner is also better conditioned, so root positions differ from
// libapriltag's in the last bits - see the tolerance discussion in
// tools/validate_pose.
double PolyEval(const double *p, int degree, double x) {
  double acc = p[degree];
  for (int i = degree - 1; i >= 0; --i) acc = acc * x + p[i];
  return acc;
}

// Ported from libapriltag's solve_poly_approx: recursively bracket roots
// between the derivative's roots, then a Newton/bisection hybrid per
// bracket. Approximate by construction - it discards roots beyond
// kMaxRoot and runs a fixed 100 iterations with no convergence assertion,
// so an unconverged root is still returned. Both behaviours are preserved
// deliberately; changing them would change which pose comes out.
void SolvePolyApprox(const double *p, int degree, double *roots, int *n_roots) {
  constexpr double kMaxRoot = 1000.0;

  if (degree == 1) {
    if (std::fabs(p[0]) > kMaxRoot * std::fabs(p[1])) {
      *n_roots = 0;
    } else {
      // Matches libapriltag exactly, including that an all-zero linear
      // polynomial yields a non-finite root rather than being rejected.
      roots[0] = -p[0] / p[1];
      *n_roots = 1;
    }
    return;
  }

  // Degree is at most 4 here (the quartic from FixPoseAmbiguities), so the
  // recursion depth and these buffers are bounded and can live on the stack.
  double p_der[5];
  for (int i = 0; i < degree; ++i) p_der[i] = (i + 1) * p[i + 1];

  double der_roots[5];
  int n_der_roots = 0;
  SolvePolyApprox(p_der, degree - 1, der_roots, &n_der_roots);

  *n_roots = 0;
  for (int i = 0; i <= n_der_roots; ++i) {
    const double min = (i == 0) ? -kMaxRoot : der_roots[i - 1];
    const double max = (i == n_der_roots) ? kMaxRoot : der_roots[i];

    const double f_min = PolyEval(p, degree, min);
    const double f_max = PolyEval(p, degree, max);

    if (f_min * f_max < 0.0) {
      double lower, upper;
      if (f_min < f_max) {
        lower = min;
        upper = max;
      } else {
        lower = max;
        upper = min;
      }
      double root = 0.5 * (lower + upper);
      double dx_old = upper - lower;
      double dx = dx_old;
      double f = PolyEval(p, degree, root);
      double df = PolyEval(p_der, degree - 1, root);

      for (int j = 0; j < 100; ++j) {
        if (((f + df * (upper - root)) * (f + df * (lower - root)) > 0.0) ||
            (std::fabs(2.0 * f) > std::fabs(dx_old * df))) {
          dx_old = dx;
          dx = 0.5 * (upper - lower);
          root = lower + dx;
        } else {
          dx_old = dx;
          dx = -f / df;
          root += dx;
        }

        if (root == upper || root == lower) break;

        f = PolyEval(p, degree, root);
        df = PolyEval(p_der, degree - 1, root);

        if (f > 0.0) {
          upper = root;
        } else {
          lower = root;
        }
      }
      roots[(*n_roots)++] = root;
    } else if (f_max == 0.0) {
      // Double/triple root.
      roots[(*n_roots)++] = max;
    }
  }
}

// --- Core solver ----------------------------------------------------------

// F_i = v_i v_i' / (v_i' v_i), libapriltag's calculate_F.
void CalculateF(const Vec3 v, Mat3 out) {
  const double den = Vec3Dot(v, v);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) out[a][b] = v[a] * v[b] / den;
}

// Orthogonal iteration (Lu/Hager/Mjolsness 2000), libapriltag's
// orthogonal_iteration with n_points fixed at 4.
//
// R and t are in/out: R must hold the initial guess. Returns the object-space
// error after the final step.
//
// Differences from libapriltag's structure, both pure hoisting with no change
// to the arithmetic performed:
//   - (F_j - I) is built once here; libapriltag re-derives it inside matd_op
//     on every iteration.
//   - R*p_j is computed once per point per iteration and reused by the
//     translation, rotation and error steps; libapriltag's expression
//     structure recomputes it three times.
double OrthogonalIteration(const Vec3 *v, const Vec3 *p, Vec3 t, Mat3 R, int n_steps) {
  Vec3 p_mean = {0.0, 0.0, 0.0};
  for (int i = 0; i < kN; ++i)
    for (int k = 0; k < 3; ++k) p_mean[k] += p[i][k];
  for (int k = 0; k < 3; ++k) p_mean[k] /= kN;

  Vec3 p_res[kN];
  for (int i = 0; i < kN; ++i)
    for (int k = 0; k < 3; ++k) p_res[i][k] = p[i][k] - p_mean[k];

  Mat3 F[kN];
  Mat3 F_minus_I[kN];
  Mat3 avg_F;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) avg_F[a][b] = 0.0;
  for (int i = 0; i < kN; ++i) {
    CalculateF(v[i], F[i]);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        avg_F[a][b] += F[i][a][b];
        F_minus_I[i][a][b] = F[i][a][b] - ((a == b) ? 1.0 : 0.0);
      }
    }
  }

  Mat3 M1, M1_inv;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) M1[a][b] = ((a == b) ? 1.0 : 0.0) - avg_F[a][b] / kN;
  if (!Mat3Inverse(M1, M1_inv)) return HUGE_VAL;

  double error = HUGE_VAL;
  for (int step = 0; step < n_steps; ++step) {
    Vec3 Rp[kN];
    for (int j = 0; j < kN; ++j) Mat3MulVec(R, p[j], Rp[j]);

    // t = M1_inv * mean_j[(F_j - I) * R * p_j]
    Vec3 M2 = {0.0, 0.0, 0.0};
    for (int j = 0; j < kN; ++j) {
      Vec3 term;
      Mat3MulVec(F_minus_I[j], Rp[j], term);
      for (int a = 0; a < 3; ++a) M2[a] += term[a];
    }
    for (int a = 0; a < 3; ++a) M2[a] /= kN;
    Mat3MulVec(M1_inv, M2, t);

    // q_j = F_j * (R p_j + t); M3 = sum_j (q_j - q_mean) p_res_j'
    Vec3 q[kN];
    Vec3 q_mean = {0.0, 0.0, 0.0};
    for (int j = 0; j < kN; ++j) {
      Vec3 sum;
      for (int a = 0; a < 3; ++a) sum[a] = Rp[j][a] + t[a];
      Mat3MulVec(F[j], sum, q[j]);
      for (int a = 0; a < 3; ++a) q_mean[a] += q[j][a];
    }
    for (int a = 0; a < 3; ++a) q_mean[a] /= kN;

    Mat3 M3;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) M3[a][b] = 0.0;
    for (int j = 0; j < kN; ++j)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) M3[a][b] += (q[j][a] - q_mean[a]) * p_res[j][b];

    NearestOrthogonal(M3, R);
    // libapriltag's proper-rotation fixup: negate R's third column when the
    // orthogonal factor came out as a reflection.
    if (Mat3Det(R) < 0.0) {
      for (int a = 0; a < 3; ++a) R[a][2] = -R[a][2];
    }

    // error = sum_j || (I - F_j)(R p_j + t) ||^2. Recomputed with the new R,
    // matching libapriltag's ordering (it updates R, then measures).
    error = 0.0;
    for (int j = 0; j < kN; ++j) {
      Vec3 Rp_new, sum, e;
      Mat3MulVec(R, p[j], Rp_new);
      for (int a = 0; a < 3; ++a) sum[a] = Rp_new[a] + t[a];
      // (I - F_j) = -(F_j - I)
      Mat3MulVec(F_minus_I[j], sum, e);
      error += Vec3Dot(e, e);
    }
  }
  return error;
}

// Second local minimum of the pose error (Schweighofer/Pinz 2006),
// libapriltag's fix_pose_ambiguities. Writes the second solution's rotation
// to out_R and returns true when one exists.
bool FixPoseAmbiguities(const Vec3 *v, const Vec3 *p, const Vec3 t, const Mat3 R, Mat3 out_R) {
  // 1. Build R_t, an orthonormal basis whose third row is t normalized.
  Vec3 R_t_3 = {t[0], t[1], t[2]};
  if (!Vec3Normalize(R_t_3)) return false;

  // Gram-Schmidt e_x against R_t_3. Degenerate when t is parallel to e_x,
  // which libapriltag walks into silently (matd_vec_normalize of a zero
  // vector); rejected here instead.
  Vec3 R_t_1 = {1.0 - R_t_3[0] * R_t_3[0], -R_t_3[0] * R_t_3[1], -R_t_3[0] * R_t_3[2]};
  if (!Vec3Normalize(R_t_1)) return false;

  Vec3 R_t_2;
  Vec3Cross(R_t_3, R_t_1, R_t_2);

  // Each basis vector becomes a ROW of R_t. libapriltag assembles this by
  // reading MATD_EL(R_t_1, 0, 1) and (0, 2) off a 3x1 column vector - outside
  // its declared column range, but landing on the right flat offsets in
  // row-major storage, so the effect is the vector's three components laid
  // out as a row. That intent is what is ported here.
  Mat3 R_t = {{R_t_1[0], R_t_1[1], R_t_1[2]},
              {R_t_2[0], R_t_2[1], R_t_2[2]},
              {R_t_3[0], R_t_3[1], R_t_3[2]}};

  // 2. R_z from the bottom row of R_t * R.
  Mat3 R_1_prime;
  Mat3Mul(R_t, R, R_1_prime);
  double r31 = R_1_prime[2][0];
  double r32 = R_1_prime[2][1];
  double hypotenuse = std::sqrt(r31 * r31 + r32 * r32);
  if (hypotenuse < 1e-100) {
    r31 = 1.0;
    r32 = 0.0;
    hypotenuse = 1.0;
  }
  Mat3 R_z = {{r31 / hypotenuse, -r32 / hypotenuse, 0.0},
              {r32 / hypotenuse, r31 / hypotenuse, 0.0},
              {0.0, 0.0, 1.0}};

  // 3. Parameters of the error function E(beta).
  Mat3 R_trans;
  Mat3Mul(R_1_prime, R_z, R_trans);
  const double sin_gamma = -R_trans[0][1];
  const double cos_gamma = R_trans[1][1];
  Mat3 R_gamma = {{cos_gamma, -sin_gamma, 0.0}, {sin_gamma, cos_gamma, 0.0}, {0.0, 0.0, 1.0}};

  const double sin_beta = -R_trans[2][0];
  const double cos_beta = R_trans[2][2];
  const double t_initial = std::atan2(sin_beta, cos_beta);

  Vec3 v_trans[kN], p_trans[kN];
  Mat3 F_trans[kN];
  Mat3 avg_F_trans;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) avg_F_trans[a][b] = 0.0;
  for (int i = 0; i < kN; ++i) {
    Mat3TransposedMulVec(R_z, p[i], p_trans[i]);
    Mat3MulVec(R_t, v[i], v_trans[i]);
    CalculateF(v_trans[i], F_trans[i]);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) avg_F_trans[a][b] += F_trans[i][a][b];
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) avg_F_trans[a][b] /= kN;

  Mat3 I_minus_avg, G;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) I_minus_avg[a][b] = ((a == b) ? 1.0 : 0.0) - avg_F_trans[a][b];
  if (!Mat3Inverse(I_minus_avg, G)) return false;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) G[a][b] /= kN;

  const Mat3 M1 = {{0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, {-2.0, 0.0, 0.0}};
  const Mat3 M2 = {{-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}};

  Vec3 b0 = {0.0, 0.0, 0.0}, b1 = {0.0, 0.0, 0.0}, b2 = {0.0, 0.0, 0.0};
  for (int i = 0; i < kN; ++i) {
    Mat3 Fi_minus_I;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) Fi_minus_I[a][b] = F_trans[i][a][b] - ((a == b) ? 1.0 : 0.0);

    Vec3 Rg_p, M1_p, M2_p, Rg_M1_p, Rg_M2_p, tmp;
    Mat3MulVec(R_gamma, p_trans[i], Rg_p);
    Mat3MulVec(M1, p_trans[i], M1_p);
    Mat3MulVec(M2, p_trans[i], M2_p);
    Mat3MulVec(R_gamma, M1_p, Rg_M1_p);
    Mat3MulVec(R_gamma, M2_p, Rg_M2_p);

    Mat3MulVec(Fi_minus_I, Rg_p, tmp);
    for (int a = 0; a < 3; ++a) b0[a] += tmp[a];
    Mat3MulVec(Fi_minus_I, Rg_M1_p, tmp);
    for (int a = 0; a < 3; ++a) b1[a] += tmp[a];
    Mat3MulVec(Fi_minus_I, Rg_M2_p, tmp);
    for (int a = 0; a < 3; ++a) b2[a] += tmp[a];
  }

  Vec3 b0_, b1_, b2_;
  Mat3MulVec(G, b0, b0_);
  Mat3MulVec(G, b1, b1_);
  Mat3MulVec(G, b2, b2_);

  double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
  for (int i = 0; i < kN; ++i) {
    Mat3 I_minus_Fi;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) I_minus_Fi[a][b] = ((a == b) ? 1.0 : 0.0) - F_trans[i][a][b];

    Vec3 Rg_p, M1_p, M2_p, Rg_M1_p, Rg_M2_p;
    Mat3MulVec(R_gamma, p_trans[i], Rg_p);
    Mat3MulVec(M1, p_trans[i], M1_p);
    Mat3MulVec(M2, p_trans[i], M2_p);
    Mat3MulVec(R_gamma, M1_p, Rg_M1_p);
    Mat3MulVec(R_gamma, M2_p, Rg_M2_p);

    Vec3 s0, s1, s2, c0, c1, c2;
    for (int a = 0; a < 3; ++a) {
      s0[a] = Rg_p[a] + b0_[a];
      s1[a] = Rg_M1_p[a] + b1_[a];
      s2[a] = Rg_M2_p[a] + b2_[a];
    }
    Mat3MulVec(I_minus_Fi, s0, c0);
    Mat3MulVec(I_minus_Fi, s1, c1);
    Mat3MulVec(I_minus_Fi, s2, c2);

    a0 += Vec3Dot(c0, c0);
    a1 += 2.0 * Vec3Dot(c0, c1);
    a2 += Vec3Dot(c1, c1) + 2.0 * Vec3Dot(c0, c2);
    a3 += 2.0 * Vec3Dot(c1, c2);
    a4 += Vec3Dot(c2, c2);
  }

  // 4. Minima of E: roots of its derivative, as a quartic in the half-angle
  // tangent.
  const double poly[5] = {a1, 2.0 * a2 - 4.0 * a0, 3.0 * a3 - 3.0 * a1, 4.0 * a4 - 2.0 * a2,
                          -a3};
  double roots[5];
  int n_roots = 0;
  SolvePolyApprox(poly, 4, roots, &n_roots);

  double minima[5];
  int n_minima = 0;
  for (int i = 0; i < n_roots; ++i) {
    const double t1 = roots[i];
    const double t2 = t1 * t1;
    const double t3 = t1 * t2;
    const double t4 = t1 * t3;
    const double t5 = t1 * t4;
    const double second_derivative = a2 - 2.0 * a0 + (3.0 * a3 - 6.0 * a1) * t1 +
                                     (6.0 * a4 - 8.0 * a2 + 10.0 * a0) * t2 +
                                     (-8.0 * a3 + 6.0 * a1) * t3 +
                                     (-6.0 * a4 + 3.0 * a2) * t4 + a3 * t5;
    if (second_derivative >= 0.0) {
      // Keep only a minimum qualitatively different from the one we already
      // have.
      const double t_cur = 2.0 * std::atan(roots[i]);
      if (std::fabs(t_cur - t_initial) > 0.1) minima[n_minima++] = roots[i];
    }
  }

  // 5. Recover the pose for the single new minimum. More than one means the
  // prior estimate was poor; libapriltag gives up in that case too.
  if (n_minima != 1) return false;

  const double t_cur = minima[0];
  Mat3 R_beta;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      // ((M2 * t_cur) + M1) * t_cur + I, all scaled by 1/(1 + t_cur^2).
      R_beta[a][b] = ((M2[a][b] * t_cur + M1[a][b]) * t_cur + ((a == b) ? 1.0 : 0.0)) /
                     (1.0 + t_cur * t_cur);
    }

  // out_R = R_t' * R_gamma * R_beta * R_z'
  Mat3 tmp1, tmp2;
  Mat3Mul(R_gamma, R_beta, tmp1);
  Mat3MulTransposed(tmp1, R_z, tmp2);
  Mat3TransposedMul(R_t, tmp2, out_R);
  return true;
}

// libapriltag's homography_to_pose, writing the rotation and translation
// separately instead of assembling a 4x4.
//
// The only deliberate change is precision: libapriltag computes the scale
// factor with single-precision sqrtf, which perturbs the seed by ~1e-7
// relative. Since orthogonal iteration is a local optimizer, that can very
// occasionally steer it into the other basin of the planar-pose ambiguity -
// the main reason a comparison against libapriltag is a tolerance, not an
// equality.
bool HomographyToPose(const double H[3][3], double fx, double fy, double cx, double cy, Mat3 R,
                      Vec3 t) {
  double R20 = H[2][0];
  double R21 = H[2][1];
  double TZ = H[2][2];
  double R00 = (H[0][0] - cx * R20) / fx;
  double R01 = (H[0][1] - cx * R21) / fx;
  double TX = (H[0][2] - cx * TZ) / fx;
  double R10 = (H[1][0] - cy * R20) / fy;
  double R11 = (H[1][1] - cy * R21) / fy;
  double TY = (H[1][2] - cy * TZ) / fy;

  // Scale so the rotation columns are unit length (geometric mean of the two
  // we have).
  const double length1 = std::sqrt(R00 * R00 + R10 * R10 + R20 * R20);
  const double length2 = std::sqrt(R01 * R01 + R11 * R11 + R21 * R21);
  const double denom = std::sqrt(length1 * length2);
  if (!(denom > 0.0) || !std::isfinite(denom)) return false;
  double s = 1.0 / denom;

  // Sign of s comes from requiring the tag to sit in front of the camera,
  // which looks along -Z.
  if (TZ > 0.0) s *= -1.0;

  R20 *= s;
  R21 *= s;
  TZ *= s;
  R00 *= s;
  R01 *= s;
  TX *= s;
  R10 *= s;
  R11 *= s;
  TY *= s;

  // Third column is the cross product of the first two.
  const double R02 = R10 * R21 - R20 * R11;
  const double R12 = R20 * R01 - R00 * R21;
  const double R22 = R00 * R11 - R10 * R01;

  // Make the rotation proper by polar decomposition. Note this takes the
  // orthogonal factor with NO determinant fixup, unlike orthogonal
  // iteration's use of the same decomposition - preserved as libapriltag has
  // it.
  const Mat3 raw = {{R00, R01, R02}, {R10, R11, R12}, {R20, R21, R22}};
  NearestOrthogonal(raw, R);

  t[0] = TX;
  t[1] = TY;
  t[2] = TZ;
  for (int a = 0; a < 3; ++a) {
    if (!std::isfinite(t[a])) return false;
    for (int b = 0; b < 3; ++b)
      if (!std::isfinite(R[a][b])) return false;
  }
  return true;
}

}  // namespace

PoseEstimator::PoseEstimator(CameraIntrinsics intrinsics, double tagsize, uint32_t cpu_threads)
    : intrinsics_(intrinsics),
      tagsize_(tagsize),
      pool_(std::make_unique<WorkerPool>(ResolveThreadCount(cpu_threads))) {}

TagPose PoseEstimator::EstimateSeed(const double H[3][3]) const {
  TagPose out;
  Mat3 R;
  Vec3 t;
  // Note the negated fx and the diag(1, -1, -1) correction below: both are
  // libapriltag's estimate_pose_for_tag_homography, not incidental.
  if (!HomographyToPose(H, -intrinsics_.fx, intrinsics_.fy, intrinsics_.cx, intrinsics_.cy, R,
                        t)) {
    return out;
  }

  const double scale = tagsize_ / 2.0;
  for (int a = 0; a < 3; ++a) t[a] *= scale;

  const double fix[3] = {1.0, -1.0, -1.0};
  for (int a = 0; a < 3; ++a) {
    out.t[a] = fix[a] * t[a];
    for (int b = 0; b < 3; ++b) out.R[a][b] = fix[a] * R[a][b];
  }
  out.valid = true;
  return out;
}

TagPosePair PoseEstimator::EstimateBoth(const double corners[4][2], const double H[3][3],
                                        int iterations) const {
  TagPosePair out;

  const double scale = tagsize_ / 2.0;
  const Vec3 p[kN] = {{-scale, scale, 0.0},
                      {scale, scale, 0.0},
                      {scale, -scale, 0.0},
                      {-scale, -scale, 0.0}};
  Vec3 v[kN];
  for (int i = 0; i < kN; ++i) {
    v[i][0] = (corners[i][0] - intrinsics_.cx) / intrinsics_.fx;
    v[i][1] = (corners[i][1] - intrinsics_.cy) / intrinsics_.fy;
    v[i][2] = 1.0;
  }

  const TagPose seed = EstimateSeed(H);
  if (!seed.valid) return out;

  Mat3 R1;
  Vec3 t1;
  for (int a = 0; a < 3; ++a) {
    t1[a] = seed.t[a];
    for (int b = 0; b < 3; ++b) R1[a][b] = seed.R[a][b];
  }
  const double err1 = OrthogonalIteration(v, p, t1, R1, iterations);
  if (std::isfinite(err1)) {
    out.solution1.error = err1;
    out.solution1.valid = true;
    for (int a = 0; a < 3; ++a) {
      out.solution1.t[a] = t1[a];
      for (int b = 0; b < 3; ++b) out.solution1.R[a][b] = R1[a][b];
    }
  } else {
    return out;
  }

  Mat3 R2;
  if (FixPoseAmbiguities(v, p, t1, R1, R2)) {
    Vec3 t2 = {0.0, 0.0, 0.0};
    const double err2 = OrthogonalIteration(v, p, t2, R2, iterations);
    if (std::isfinite(err2)) {
      out.solution2.error = err2;
      out.solution2.valid = true;
      for (int a = 0; a < 3; ++a) {
        out.solution2.t[a] = t2[a];
        for (int b = 0; b < 3; ++b) out.solution2.R[a][b] = R2[a][b];
      }
    }
  }
  return out;
}

TagPose PoseEstimator::Estimate(const double corners[4][2], const double H[3][3]) const {
  const TagPosePair both = EstimateBoth(corners, H);
  if (!both.solution1.valid) return both.solution1;
  if (!both.solution2.valid) return both.solution1;
  // Matches libapriltag's tie-break: solution 1 wins when the errors are
  // equal.
  return (both.solution1.error <= both.solution2.error) ? both.solution1 : both.solution2;
}

void PoseEstimator::EstimateAll(const std::vector<const apriltag_detection_t *> &detections,
                                std::vector<TagPose> &out) const {
  out.assign(detections.size(), TagPose{});
  pool_->ParallelFor(detections.size(), [&](size_t i) {
    const apriltag_detection_t *det = detections[i];
    if (det == nullptr || det->H == nullptr) return;
    double H[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) H[a][b] = MATD_EL(det->H, a, b);
    double corners[4][2];
    for (int c = 0; c < 4; ++c) {
      corners[c][0] = det->p[c][0];
      corners[c][1] = det->p[c][1];
    }
    // Each entry is written by exactly one task, so no synchronization is
    // needed, and writing by index keeps the output independent of thread
    // scheduling.
    out[i] = Estimate(corners, H);
  });
}

void PoseEstimator::EstimateAll(const zarray_t *detections, std::vector<TagPose> &out) const {
  std::vector<const apriltag_detection_t *> flat;
  flat.reserve(detections != nullptr ? static_cast<size_t>(zarray_size(detections)) : 0u);
  if (detections != nullptr) {
    for (int i = 0; i < zarray_size(detections); ++i) {
      apriltag_detection_t *det = nullptr;
      zarray_get(const_cast<zarray_t *>(detections), i, &det);
      flat.push_back(det);
    }
  }
  EstimateAll(flat, out);
}

}  // namespace apriltag_vulkan
