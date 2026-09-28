#pragma once

extern "C" {
#include "apriltag.h"
}

// Creates and destroys a single named tag family from the `apriltag` C library.
bool setup_tag_family(apriltag_family_t **tf, const char *famname);
void teardown_tag_family(apriltag_family_t **tf, const char *famname);
void print_detections(zarray_t *detections);
