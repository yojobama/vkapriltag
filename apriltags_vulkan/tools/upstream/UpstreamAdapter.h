#pragma once

// Access to the upstream libapriltag for the tools that compare against it (built only with
// VKAPRILTAG_UPSTREAM_VALIDATION=ON). The library itself does not include upstream headers.
#include <cstring>
#include <vector>

#include "vkapriltag/Detection.h"

extern "C" {
#include "apriltag.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

namespace upstream {

// A new upstream family of the given name, or null.
inline apriltag_family_t *CreateFamily(const char *name) {
  if (!std::strcmp(name, "tag36h11")) return tag36h11_create();
  if (!std::strcmp(name, "tag25h9")) return tag25h9_create();
  if (!std::strcmp(name, "tag16h5")) return tag16h5_create();
  if (!std::strcmp(name, "tagCircle21h7")) return tagCircle21h7_create();
  if (!std::strcmp(name, "tagCircle49h12")) return tagCircle49h12_create();
  if (!std::strcmp(name, "tagStandard41h12")) return tagStandard41h12_create();
  if (!std::strcmp(name, "tagStandard52h13")) return tagStandard52h13_create();
  if (!std::strcmp(name, "tagCustom48h12")) return tagCustom48h12_create();
  return nullptr;
}

inline void DestroyFamily(apriltag_family_t *f, const char *name) {
  if (f == nullptr) return;
  if (!std::strcmp(name, "tag36h11")) tag36h11_destroy(f);
  else if (!std::strcmp(name, "tag25h9")) tag25h9_destroy(f);
  else if (!std::strcmp(name, "tag16h5")) tag16h5_destroy(f);
  else if (!std::strcmp(name, "tagCircle21h7")) tagCircle21h7_destroy(f);
  else if (!std::strcmp(name, "tagCircle49h12")) tagCircle49h12_destroy(f);
  else if (!std::strcmp(name, "tagStandard41h12")) tagStandard41h12_destroy(f);
  else if (!std::strcmp(name, "tagStandard52h13")) tagStandard52h13_destroy(f);
  else if (!std::strcmp(name, "tagCustom48h12")) tagCustom48h12_destroy(f);
}

// The family of an upstream detection set.
inline apriltag_vulkan::Detection ToDetection(const apriltag_detection_t *d, const apriltag_vulkan::Family *family = nullptr) {
  apriltag_vulkan::Detection out;
  out.family = family;
  out.id = d->id;
  out.hamming = d->hamming;
  out.decision_margin = d->decision_margin;
  if (d->H != nullptr) std::memcpy(out.H, d->H->data, sizeof(out.H));
  out.c[0] = d->c[0];
  out.c[1] = d->c[1];
  for (int k = 0; k < 4; ++k) {
    out.p[k][0] = d->p[k][0];
    out.p[k][1] = d->p[k][1];
  }
  return out;
}

inline std::vector<apriltag_vulkan::Detection> ToDetections(const zarray_t *z, const apriltag_vulkan::Family *family = nullptr) {
  std::vector<apriltag_vulkan::Detection> out;
  for (int i = 0; i < zarray_size(z); ++i) {
    apriltag_detection_t *d = nullptr;
    zarray_get(const_cast<zarray_t *>(z), i, &d);
    out.push_back(ToDetection(d, family));
  }
  return out;
}

}  // namespace upstream
