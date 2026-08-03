#include "vkapriltag/apriltag_family.h"

#include <cstring>
#include <iostream>

extern "C" {
#include "common/matd.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

bool setup_tag_family(apriltag_family_t **tf, const char *famname) {
  if (!strcmp(famname, "tag36h11")) {
    *tf = tag36h11_create();
  } else if (!strcmp(famname, "tag25h9")) {
    *tf = tag25h9_create();
  } else if (!strcmp(famname, "tag16h5")) {
    *tf = tag16h5_create();
  } else if (!strcmp(famname, "tagCircle21h7")) {
    *tf = tagCircle21h7_create();
  } else if (!strcmp(famname, "tagCircle49h12")) {
    *tf = tagCircle49h12_create();
  } else if (!strcmp(famname, "tagStandard41h12")) {
    *tf = tagStandard41h12_create();
  } else if (!strcmp(famname, "tagStandard52h13")) {
    *tf = tagStandard52h13_create();
  } else if (!strcmp(famname, "tagCustom48h12")) {
    *tf = tagCustom48h12_create();
  } else {
    std::cerr << "Unknown tag family: " << famname << std::endl;
    return false;
  }
  return true;
}

void teardown_tag_family(apriltag_family_t **tf, const char *famname) {
  if (!strcmp(famname, "tag36h11")) {
    tag36h11_destroy(*tf);
  } else if (!strcmp(famname, "tag25h9")) {
    tag25h9_destroy(*tf);
  } else if (!strcmp(famname, "tag16h5")) {
    tag16h5_destroy(*tf);
  } else if (!strcmp(famname, "tagCircle21h7")) {
    tagCircle21h7_destroy(*tf);
  } else if (!strcmp(famname, "tagCircle49h12")) {
    tagCircle49h12_destroy(*tf);
  } else if (!strcmp(famname, "tagStandard41h12")) {
    tagStandard41h12_destroy(*tf);
  } else if (!strcmp(famname, "tagStandard52h13")) {
    tagStandard52h13_destroy(*tf);
  } else if (!strcmp(famname, "tagCustom48h12")) {
    tagCustom48h12_destroy(*tf);
  }
}

void print_detections(zarray_t *detections) {
  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    std::cout << "tag #: " << det->id << "  hamming: " << det->hamming
              << "  margin: " << det->decision_margin << "  center: (" << det->c[0] << ", "
              << det->c[1] << ")" << std::endl;
  }
}
