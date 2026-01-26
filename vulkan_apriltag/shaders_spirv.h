#pragma once

// SPIR-V bytecode for compute shaders compiled from shaders/*.comp at build time

#include <stddef.h>
#include <stdint.h>


// Threshold shader SPIR-V
extern const uint32_t threshold_comp_spirv[];
extern const size_t threshold_comp_spirv_size;

// Connected components shader SPIR-V
extern const uint32_t connected_components_comp_spirv[];
extern const size_t connected_components_comp_spirv_size;

// Gradient shader SPIR-V
extern const uint32_t gradient_comp_spirv[];
extern const size_t gradient_comp_spirv_size;

// Decimation shader SPIR-V
extern const uint32_t decimate_comp_spirv[];
extern const size_t decimate_comp_spirv_size;

// Blur shader SPIR-V
extern const uint32_t blur_comp_spirv[];
extern const size_t blur_comp_spirv_size;