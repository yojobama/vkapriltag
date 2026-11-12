#pragma once

#include <cmath>
#include <cstdio>

// Minimal assertion helpers: a failed check prints its location and the run exits non-zero.
namespace testcheck {
inline int &Failures() {
  static int failures = 0;
  return failures;
}
}  // namespace testcheck

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
      ++testcheck::Failures();                                                     \
    }                                                                              \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                   \
  do {                                                                                          \
    const double va_ = (a), vb_ = (b), vt_ = (tol);                                             \
    if (!(std::fabs(va_ - vb_) <= vt_)) {                                                       \
      std::printf("FAIL %s:%d: %s = %.9g, expected %.9g +- %.3g\n", __FILE__, __LINE__, #a, va_, vb_, vt_); \
      ++testcheck::Failures();                                                                  \
    }                                                                                           \
  } while (0)
