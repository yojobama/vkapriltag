#pragma once

extern "C" {
#include "apriltag.h"
}

// Thin, OpenCV-free port of apriltags_cuda's setup_tag_family /
// teardown_tag_family / print_detections (apriltag_utils.cu). Owns the
// creation/destruction of a single named tag family from the fetched
// `apriltag` C library.
bool setup_tag_family(apriltag_family_t **tf, const char *famname);
void teardown_tag_family(apriltag_family_t **tf, const char *famname);
void print_detections(zarray_t *detections);
