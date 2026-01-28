#include "vulkan_apriltag.h"
#include "vulkan_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Include embedded shader SPIR-V
#include "shaders_spirv.h"

static VkResult load_shader_with_fallback(vulkan_apriltag_context_t* ctx,
                                          const char* shader_name,
                                          const uint32_t* embedded_data,
                                          size_t embedded_size,
                                          VkShaderModule* shader_module) {
    const char* shader_dir = getenv("APRILTAG_VULKAN_SHADER_DIR");
#ifdef VULKAN_SHADER_DIR
    if (!shader_dir || shader_dir[0] == '\0') {
        shader_dir = VULKAN_SHADER_DIR;
    }
#endif

    if (shader_dir && shader_name && shader_name[0] != '\0') {
        char shader_path[512];
        int written = snprintf(shader_path, sizeof(shader_path), "%s/%s.spv", shader_dir, shader_name);
        if (written > 0 && written < (int)sizeof(shader_path)) {
            FILE* f = fopen(shader_path, "rb");
            if (f) {
                fclose(f);
                VkResult file_result = load_shader_module(ctx->device, shader_path, shader_module);
                if (file_result == VK_SUCCESS) {
                    return VK_SUCCESS;
                }
            }
        }
    }

    return load_shader_from_spirv(ctx->device, embedded_data, embedded_size, shader_module);
}

static VkResult create_image_buffer(vulkan_apriltag_context_t* ctx, uint32_t width, uint32_t height,
                                   VkFormat format, VkImageUsageFlags usage, vulkan_image_buffer_t* buffer) {
    buffer->width = width;
    buffer->height = height;
    buffer->size = width * height * sizeof(uint8_t); // Assuming 8-bit per channel for now
    
    return create_storage_buffer(ctx->device, ctx->memory_properties, buffer->size, 
                                &buffer->buffer, &buffer->memory);
}

static VkResult create_compute_buffer(vulkan_apriltag_context_t* ctx, uint32_t element_count, 
                                     uint32_t element_size, vulkan_compute_buffer_t* buffer) {
    buffer->element_count = element_count;
    buffer->size = element_count * element_size;
    
    return create_storage_buffer(ctx->device, ctx->memory_properties, buffer->size,
                                &buffer->buffer, &buffer->memory);
}

static VkResult setup_pipelines(vulkan_apriltag_context_t* ctx) {
    VkResult result;
    
    // Create shader modules from embedded SPIR-V
    result = load_shader_with_fallback(ctx, "threshold_comp", threshold_comp_spirv,
                                   threshold_comp_spirv_size, &ctx->threshold_shader);
    if (result != VK_SUCCESS) return result;
    
    result = load_shader_with_fallback(ctx, "connected_components_comp", connected_components_comp_spirv,
                                   connected_components_comp_spirv_size, &ctx->connected_components_shader);
    if (result != VK_SUCCESS) return result;
    
    result = load_shader_with_fallback(ctx, "gradient_comp", gradient_comp_spirv,
                                   gradient_comp_spirv_size, &ctx->gradient_shader);
    if (result != VK_SUCCESS) return result;
    
    result = load_shader_with_fallback(ctx, "decimate_comp", decimate_comp_spirv,
                                   decimate_comp_spirv_size, &ctx->decimate_shader);
    if (result != VK_SUCCESS) return result;
    
    result = load_shader_with_fallback(ctx, "blur_comp", blur_comp_spirv,
                                   blur_comp_spirv_size, &ctx->blur_shader);
    if (result != VK_SUCCESS) return result;
    
    // Create descriptor set layouts
    VkDescriptorSetLayoutBinding threshold_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    
    result = create_descriptor_set_layout(ctx->device, 3, threshold_bindings, &ctx->threshold_desc_layout);
    if (result != VK_SUCCESS) return result;
    
    VkDescriptorSetLayoutBinding connected_components_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    
    result = create_descriptor_set_layout(ctx->device, 4, connected_components_bindings, &ctx->connected_components_desc_layout);
    if (result != VK_SUCCESS) return result;
    
    VkDescriptorSetLayoutBinding gradient_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    
    result = create_descriptor_set_layout(ctx->device, 5, gradient_bindings, &ctx->gradient_desc_layout);
    if (result != VK_SUCCESS) return result;
    
    VkDescriptorSetLayoutBinding decimate_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    
    result = create_descriptor_set_layout(ctx->device, 3, decimate_bindings, &ctx->decimate_desc_layout);
    if (result != VK_SUCCESS) return result;
    
    VkDescriptorSetLayoutBinding blur_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }
    };
    
    result = create_descriptor_set_layout(ctx->device, 4, blur_bindings, &ctx->blur_desc_layout);
    if (result != VK_SUCCESS) return result;
    
    // Create compute pipelines
    result = create_compute_pipeline(ctx->device, ctx->threshold_shader, ctx->threshold_desc_layout,
                                    &ctx->threshold_layout, &ctx->threshold_pipeline);
    if (result != VK_SUCCESS) return result;
    
    result = create_compute_pipeline(ctx->device, ctx->connected_components_shader, ctx->connected_components_desc_layout,
                                    &ctx->connected_components_layout, &ctx->connected_components_pipeline);
    if (result != VK_SUCCESS) return result;
    
    result = create_compute_pipeline(ctx->device, ctx->gradient_shader, ctx->gradient_desc_layout,
                                    &ctx->gradient_layout, &ctx->gradient_pipeline);
    if (result != VK_SUCCESS) return result;
    
    result = create_compute_pipeline(ctx->device, ctx->decimate_shader, ctx->decimate_desc_layout,
                                    &ctx->decimate_layout, &ctx->decimate_pipeline);
    if (result != VK_SUCCESS) return result;
    
    result = create_compute_pipeline(ctx->device, ctx->blur_shader, ctx->blur_desc_layout,
                                    &ctx->blur_layout, &ctx->blur_pipeline);
    if (result != VK_SUCCESS) return result;

    result = create_compute_pipeline(ctx->device, ctx->line_fit_shader, ctx->line_fit_desc_layout,
                                    &ctx->line_fit_layout, &ctx->line_fit_pipeline);
    if (result != VK_SUCCESS) return result;

    return VK_SUCCESS;
}

static void destroy_image_buffer(vulkan_apriltag_context_t* ctx, vulkan_image_buffer_t* buf) {
    if (buf->buffer) vkDestroyBuffer(ctx->device, buf->buffer, NULL);
    if (buf->memory) vkFreeMemory(ctx->device, buf->memory, NULL);
    memset(buf, 0, sizeof(*buf));
}

static void destroy_compute_buffer(vulkan_apriltag_context_t* ctx, vulkan_compute_buffer_t* buf) {
    if (buf->buffer) vkDestroyBuffer(ctx->device, buf->buffer, NULL);
    if (buf->memory) vkFreeMemory(ctx->device, buf->memory, NULL);
    memset(buf, 0, sizeof(*buf));
}

vulkan_apriltag_detector_t* vulkan_apriltag_detector_create(
    vulkan_apriltag_context_t* ctx,
    uint32_t max_width,
    uint32_t max_height) {
    
    if (!ctx) return NULL;
    
    vulkan_apriltag_detector_t* detector = calloc(1, sizeof(vulkan_apriltag_detector_t));
    if (!detector) return NULL;
    
    detector->vulkan_ctx = ctx;
    
    // Create base CPU detector for non-GPU operations
    detector->base_detector = apriltag_detector_create();
    if (!detector->base_detector) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Setup pipelines if not already done
    if (ctx->threshold_pipeline == VK_NULL_HANDLE) {
        if (setup_pipelines(ctx) != VK_SUCCESS) {
            fprintf(stderr, "Failed to setup Vulkan pipelines\n");
            vulkan_apriltag_detector_destroy(detector);
            return NULL;
        }
    }
    
    // Create GPU buffers
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R8_UINT, 
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->input_image) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width/2, max_height/2, VK_FORMAT_R8_UINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->decimated_image) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R8_UINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->blurred_image) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R8_UINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->threshold_image) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R16_SINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->gradient_x) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R16_SINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->gradient_y) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R16_UINT,
                           VK_IMAGE_USAGE_STORAGE_BIT, &detector->gradient_mag) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Create compute buffers
    uint32_t max_components = max_width * max_height / 4; // Conservative estimate
    if (create_compute_buffer(ctx, max_components, sizeof(uint32_t), &detector->connected_components) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    uint32_t max_edges = max_width * max_height / 8;
    if (create_compute_buffer(ctx, max_edges, sizeof(uint32_t) * 2, &detector->edge_points) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    uint32_t max_lines = max_edges / 10;
    if (create_compute_buffer(ctx, max_lines, sizeof(float) * 4, &detector->line_segments) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    uint32_t max_quads = max_lines / 4;
    if (create_compute_buffer(ctx, max_quads, sizeof(float) * 8, &detector->quad_candidates) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Create staging buffers
    if (create_image_buffer(ctx, max_width, max_height, VK_FORMAT_R8_UINT,
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &detector->staging_input) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    if (create_compute_buffer(ctx, max_quads, sizeof(float) * 8, &detector->staging_output) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Allocate command buffers
    if (allocate_command_buffer(ctx->device, ctx->command_pool, &detector->preprocess_cmd) != VK_SUCCESS ||
        allocate_command_buffer(ctx->device, ctx->command_pool, &detector->threshold_cmd) != VK_SUCCESS ||
        allocate_command_buffer(ctx->device, ctx->command_pool, &detector->connected_components_cmd) != VK_SUCCESS ||
        allocate_command_buffer(ctx->device, ctx->command_pool, &detector->gradient_cmd) != VK_SUCCESS ||
        allocate_command_buffer(ctx->device, ctx->command_pool, &detector->line_fit_cmd) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Create synchronization objects
    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    if (vkCreateSemaphore(ctx->device, &semaphore_info, NULL, &detector->preprocess_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(ctx->device, &semaphore_info, NULL, &detector->threshold_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(ctx->device, &semaphore_info, NULL, &detector->connected_components_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(ctx->device, &semaphore_info, NULL, &detector->gradient_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(ctx->device, &semaphore_info, NULL, &detector->line_fit_semaphore) != VK_SUCCESS ||
        vkCreateFence(ctx->device, &fence_info, NULL, &detector->completion_fence) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    // Create timestamp query pool
    if (create_timestamp_query_pool(ctx->device, 16, &detector->timestamp_pool) != VK_SUCCESS) {
        vulkan_apriltag_detector_destroy(detector);
        return NULL;
    }
    
    return detector;
}

void vulkan_apriltag_detector_destroy(vulkan_apriltag_detector_t* detector) {
    if (!detector) return;
    
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    if (ctx && ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        
        // Destroy synchronization objects
        if (detector->preprocess_semaphore) vkDestroySemaphore(ctx->device, detector->preprocess_semaphore, NULL);
        if (detector->threshold_semaphore) vkDestroySemaphore(ctx->device, detector->threshold_semaphore, NULL);
        if (detector->connected_components_semaphore) vkDestroySemaphore(ctx->device, detector->connected_components_semaphore, NULL);
        if (detector->gradient_semaphore) vkDestroySemaphore(ctx->device, detector->gradient_semaphore, NULL);
        if (detector->line_fit_semaphore) vkDestroySemaphore(ctx->device, detector->line_fit_semaphore, NULL);
        if (detector->completion_fence) vkDestroyFence(ctx->device, detector->completion_fence, NULL);
        
        if (detector->timestamp_pool) vkDestroyQueryPool(ctx->device, detector->timestamp_pool, NULL);
        
        // Destroy buffers and memory
        destroy_image_buffer(ctx, &detector->input_image);
        destroy_image_buffer(ctx, &detector->decimated_image);
        destroy_image_buffer(ctx, &detector->blurred_image);
        destroy_image_buffer(ctx, &detector->threshold_image);
        destroy_image_buffer(ctx, &detector->gradient_x);
        destroy_image_buffer(ctx, &detector->gradient_y);
        destroy_image_buffer(ctx, &detector->gradient_mag);
        destroy_image_buffer(ctx, &detector->staging_input);
        
        destroy_compute_buffer(ctx, &detector->connected_components);
        destroy_compute_buffer(ctx, &detector->edge_points);
        destroy_compute_buffer(ctx, &detector->line_segments);
        destroy_compute_buffer(ctx, &detector->quad_candidates);
        destroy_compute_buffer(ctx, &detector->staging_output);
    }
    
    if (detector->base_detector) {
        apriltag_detector_destroy(detector->base_detector);
    }
    
    free(detector);
}

void vulkan_apriltag_detector_add_family(vulkan_apriltag_detector_t* detector, apriltag_family_t* family) {
    if (detector && detector->base_detector && family) {
        apriltag_detector_add_family(detector->base_detector, family);
    }
}

void vulkan_apriltag_detector_add_family_bits(vulkan_apriltag_detector_t* detector, apriltag_family_t* family, int bits_corrected) {
    if (detector && detector->base_detector && family) {
        apriltag_detector_add_family_bits(detector->base_detector, family, bits_corrected);
    }
}

// Configuration functions
void vulkan_apriltag_detector_set_quad_decimate(vulkan_apriltag_detector_t* detector, float decimate) {
    if (detector && detector->base_detector) {
        detector->base_detector->quad_decimate = decimate;
    }
}

void vulkan_apriltag_detector_set_quad_sigma(vulkan_apriltag_detector_t* detector, float sigma) {
    if (detector && detector->base_detector) {
        detector->base_detector->quad_sigma = sigma;
    }
}

void vulkan_apriltag_detector_set_refine_edges(vulkan_apriltag_detector_t* detector, bool refine_edges) {
    if (detector && detector->base_detector) {
        detector->base_detector->refine_edges = refine_edges;
    }
}

void vulkan_apriltag_detector_set_decode_sharpening(vulkan_apriltag_detector_t* detector, double sharpening) {
    if (detector && detector->base_detector) {
        detector->base_detector->decode_sharpening = sharpening;
    }
}

void vulkan_apriltag_detector_set_debug(vulkan_apriltag_detector_t* detector, bool debug) {
    if (detector && detector->base_detector) {
        detector->base_detector->debug = debug;
    }
}

uint64_t vulkan_apriltag_detector_get_gpu_time_ns(vulkan_apriltag_detector_t* detector) {
    return detector ? detector->vulkan_ctx->gpu_time_ns : 0;
}

void vulkan_apriltag_detector_print_performance_stats(vulkan_apriltag_detector_t* detector) {
    if (!detector || !detector->vulkan_ctx) return;
    
    printf("Vulkan AprilTag Performance Stats:\n");
    printf("  GPU Time: %.3f ms\n", detector->vulkan_ctx->gpu_time_ns / 1000000.0);
    printf("  Device: ");
    vulkan_apriltag_print_device_info(detector->vulkan_ctx);
}