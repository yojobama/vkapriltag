#pragma once

// Minimal shader interface for the demo
// In a full implementation, this would contain the actual compiled SPIR-V bytecode

#include <stdint.h>

// Placeholder SPIR-V data - would be replaced with actual compiled shaders
extern const uint32_t threshold_comp_spirv[];
extern const size_t threshold_comp_spirv_size;

extern const uint32_t connected_components_comp_spirv[];
extern const size_t connected_components_comp_spirv_size;

extern const uint32_t gradient_comp_spirv[];
extern const size_t gradient_comp_spirv_size;

extern const uint32_t decimate_comp_spirv[];
extern const size_t decimate_comp_spirv_size;

extern const uint32_t blur_comp_spirv[];
extern const size_t blur_comp_spirv_size;