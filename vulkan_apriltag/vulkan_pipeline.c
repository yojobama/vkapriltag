#include "vulkan_apriltag.h"
#include "vulkan_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// GPU-accelerated quad detection pipeline
static VkResult execute_preprocessing_pipeline(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    VkResult result;
    
    // Upload image to GPU
    memcpy(ctx->staging_buffer.mapped, image->buf, image->width * image->height);
    
    result = copy_buffer(ctx->device, ctx->compute_queue, ctx->command_pool,
                        ctx->staging_buffer.buffer, detector->input_image.buffer,
                        image->width * image->height);
    if (result != VK_SUCCESS) return result;
    
    // Begin command buffer recording
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    result = vkBeginCommandBuffer(detector->preprocess_cmd, &begin_info);
    if (result != VK_SUCCESS) return result;
    
    // Add timestamp query
    vkCmdResetQueryPool(detector->preprocess_cmd, detector->timestamp_pool, 0, 2);
    vkCmdWriteTimestamp(detector->preprocess_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       detector->timestamp_pool, 0);
    
    // Decimation pass (if needed)
    if (detector->base_detector->quad_decimate > 1.0f) {
        vkCmdBindPipeline(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->decimate_pipeline);
        
        // Set up uniform buffer for decimation
        struct {
            uint32_t input_width;
            uint32_t input_height;
            uint32_t output_width;
            uint32_t output_height;
            float decimate_factor;
        } decimate_params = {
            image->width, image->height,
            (uint32_t)(image->width / detector->base_detector->quad_decimate),
            (uint32_t)(image->height / detector->base_detector->quad_decimate),
            detector->base_detector->quad_decimate
        };
        
        // Update push constants with decimation parameters
        vkCmdPushConstants(detector->preprocess_cmd, ctx->decimate_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(decimate_params), &decimate_params);
        
        // Bind descriptor sets for input and output buffers
        vkCmdBindDescriptorSets(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->decimate_layout, 0, 1, &detector->input_image.descriptor_set,
                               0, NULL);
        
        uint32_t group_x = (decimate_params.output_width + 15) / 16;
        uint32_t group_y = (decimate_params.output_height + 15) / 16;
        vkCmdDispatch(detector->preprocess_cmd, group_x, group_y, 1);
        
        // Memory barrier
        VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };

        vkCmdPipelineBarrier(detector->preprocess_cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, NULL, 0, NULL);
    }
    
    // Blur pass (if needed)
    if (detector->base_detector->quad_sigma > 0.0f) {
        vkCmdBindPipeline(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->blur_pipeline);
        
        // Calculate kernel size from sigma
        uint32_t kernel_size = (uint32_t)(6 * detector->base_detector->quad_sigma + 1);
        if (kernel_size % 2 == 0) kernel_size++;
        
        struct {
            uint32_t width;
            uint32_t height;
            uint32_t kernel_size;
            uint32_t direction;
            float sigma;
        } blur_params = {
            image->width, image->height,
            kernel_size, 0, // Horizontal pass first
            detector->base_detector->quad_sigma
        };
        
        // Memory barrier
        VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };


        // Update push constants with blur parameters
        vkCmdPushConstants(detector->preprocess_cmd, ctx->blur_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_params), &blur_params);
        
        // Bind descriptor sets for blur input/output
        VkDescriptorSet blur_desc_set = detector->base_detector->quad_decimate > 1.0f ?
            detector->decimated_image.descriptor_set : detector->input_image.descriptor_set;
        vkCmdBindDescriptorSets(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->blur_layout, 0, 1, &blur_desc_set, 0, NULL);
        
        // Horizontal pass
        uint32_t group_count = (image->width + 255) / 256;
        vkCmdDispatch(detector->preprocess_cmd, group_count, image->height, 1);
        
        vkCmdPipelineBarrier(detector->preprocess_cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, NULL, 0, NULL);
        
        
        // Vertical pass
        blur_params.direction = 1;
        vkCmdPushConstants(detector->preprocess_cmd, ctx->blur_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_params), &blur_params);
        
        group_count = (image->height + 255) / 256;
        vkCmdDispatch(detector->preprocess_cmd, image->width, group_count, 1);
        
        vkCmdPipelineBarrier(detector->preprocess_cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, NULL, 0, NULL);
    }
    
    vkCmdWriteTimestamp(detector->preprocess_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                       detector->timestamp_pool, 1);
    
    result = vkEndCommandBuffer(detector->preprocess_cmd);
    if (result != VK_SUCCESS) return result;
    
    // Submit preprocessing
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &detector->preprocess_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &detector->preprocess_semaphore
    };

    return vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
}

static VkResult execute_threshold_pipeline(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    VkResult result;
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    result = vkBeginCommandBuffer(detector->threshold_cmd, &begin_info);
    if (result != VK_SUCCESS) return result;
    
    vkCmdBindPipeline(detector->threshold_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->threshold_pipeline);
    
    // Set up threshold parameters
    struct {
        uint32_t width;
        uint32_t height;
        uint32_t tile_size;
        uint32_t min_white_black_diff;
    } threshold_params = {
        image->width, image->height,
        16, // Tile size for adaptive thresholding
        (uint32_t)detector->base_detector->qtp.min_white_black_diff
    };
    
    
    // Update push constants with threshold parameters
    vkCmdPushConstants(detector->threshold_cmd, ctx->threshold_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(threshold_params), &threshold_params);
    
    // Bind descriptor sets for input image
    // Use blurred image if blur was applied, otherwise use decimated or original
    VkDescriptorSet input_desc_set;
    if (detector->base_detector->quad_sigma > 0.0f) {
        input_desc_set = detector->blurred_image.descriptor_set;
    } else if (detector->base_detector->quad_decimate > 1.0f) {
        input_desc_set = detector->decimated_image.descriptor_set;
    } else {
        input_desc_set = detector->input_image.descriptor_set;
    }
    vkCmdBindDescriptorSets(detector->threshold_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->threshold_layout, 0, 1, &input_desc_set, 0, NULL);

    uint32_t group_x = (image->width + 15) / 16;
    uint32_t group_y = (image->height + 15) / 16;
    vkCmdDispatch(detector->threshold_cmd, group_x, group_y, 1);
    
    result = vkEndCommandBuffer(detector->threshold_cmd);
    if (result != VK_SUCCESS) return result;
    
    // Submit threshold pass
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &detector->preprocess_semaphore,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &detector->threshold_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &detector->threshold_semaphore
    };

    return vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
}

static VkResult execute_connected_components_pipeline(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    VkResult result;
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    result = vkBeginCommandBuffer(detector->connected_components_cmd, &begin_info);
    if (result != VK_SUCCESS) return result;
    
    vkCmdBindPipeline(detector->connected_components_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                     ctx->connected_components_pipeline);
    
    // Run multiple iterations for convergence
    uint32_t max_iterations = 10;
    
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        struct {
            uint32_t width;
            uint32_t height;
            uint32_t iteration;
            uint32_t max_iterations;
        } cc_params = {
            image->width, image->height,
            iteration, max_iterations
        };
        
        
        // Update push constants with connected components parameters
        vkCmdPushConstants(detector->connected_components_cmd, ctx->connected_components_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cc_params), &cc_params);
        
        // Bind descriptor sets for threshold image and components buffer
        vkCmdBindDescriptorSets(detector->connected_components_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->connected_components_layout, 0, 1,
                               &detector->threshold_image.descriptor_set, 0, NULL);
        
        uint32_t group_x = (image->width + 15) / 16;
        uint32_t group_y = (image->height + 15) / 16;
        vkCmdDispatch(detector->connected_components_cmd, group_x, group_y, 1);
        
        if (iteration < max_iterations - 1) {
            VkMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };

            vkCmdPipelineBarrier(detector->connected_components_cmd,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }
    
    result = vkEndCommandBuffer(detector->connected_components_cmd);
    if (result != VK_SUCCESS) return result;
    
    // Submit connected components pass
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &detector->threshold_semaphore,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &detector->connected_components_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &detector->connected_components_semaphore
    };

    return vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
}

static VkResult execute_gradient_pipeline(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    VkResult result;
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    result = vkBeginCommandBuffer(detector->gradient_cmd, &begin_info);
    if (result != VK_SUCCESS) return result;
    
    vkCmdBindPipeline(detector->gradient_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->gradient_pipeline);
    
    struct {
        uint32_t width;
        uint32_t height;
        float scale;
    } gradient_params = {
        image->width, image->height,
        1.0f / 8.0f // Sobel normalization
    };
    
    
    // Update push constants with gradient parameters
    vkCmdPushConstants(detector->gradient_cmd, ctx->gradient_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gradient_params), &gradient_params);
    
    // Bind descriptor sets for input image (use preprocessed image)
    VkDescriptorSet input_desc_set;
    if (detector->base_detector->quad_sigma > 0.0f) {
        input_desc_set = detector->blurred_image.descriptor_set;
    } else if (detector->base_detector->quad_decimate > 1.0f) {
        input_desc_set = detector->decimated_image.descriptor_set;
    } else {
        input_desc_set = detector->input_image.descriptor_set;
    }
    vkCmdBindDescriptorSets(detector->gradient_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->gradient_layout, 0, 1, &input_desc_set, 0, NULL);
    
    uint32_t group_x = (image->width + 15) / 16;
    uint32_t group_y = (image->height + 15) / 16;
    vkCmdDispatch(detector->gradient_cmd, group_x, group_y, 1);
    
    result = vkEndCommandBuffer(detector->gradient_cmd);
    if (result != VK_SUCCESS) return result;
    
    // Submit gradient pass
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &detector->connected_components_semaphore,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &detector->gradient_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &detector->gradient_semaphore
    };

    return vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
}

// CPU-based final processing of GPU results
static zarray_t* process_gpu_results(vulkan_apriltag_detector_t* detector/*, image_u8_t* image*/) {
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    
    // Wait for GPU completion
    vkWaitForFences(ctx->device, 1, &detector->completion_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &detector->completion_fence);
    
    // Read back GPU results
    // TODO: Copy results from GPU buffers to staging buffers
    
    // Get timestamp results for performance measurement
    uint64_t timestamps[2];
    vkGetQueryPoolResults(ctx->device, detector->timestamp_pool, 0, 2,
                         sizeof(timestamps), timestamps, sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    
    double timestamp_period = get_timestamp_period_ns(ctx->physical_device);
    ctx->gpu_time_ns = (timestamps[1] - timestamps[0]) * timestamp_period;
    
    // TODO: Process connected components and extract quad candidates
    // For now, return empty array - this would be replaced with actual quad processing
    zarray_t* detections = zarray_create(sizeof(apriltag_detection_t*));
    
    return detections;
}

zarray_t* vulkan_apriltag_detector_detect(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    if (!detector || !image) {
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    
    // Check if image dimensions are within supported limits
    if ((uint32_t)image->width > ctx->max_image_width || (uint32_t)image->height > ctx->max_image_height) {
        fprintf(stderr, "Image dimensions exceed GPU limits\n");
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    // Execute GPU pipeline stages
    VkResult result;
    
    result = execute_preprocessing_pipeline(detector, image);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Preprocessing pipeline failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    result = execute_threshold_pipeline(detector, image);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Threshold pipeline failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    result = execute_connected_components_pipeline(detector, image);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Connected components pipeline failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    result = execute_gradient_pipeline(detector, image);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Gradient pipeline failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    // Final fence for completion
    VkSubmitInfo final_submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &detector->gradient_semaphore,
    };
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    final_submit.pWaitDstStageMask = &wait_stage;
    
    result = vkQueueSubmit(ctx->compute_queue, 1, &final_submit, detector->completion_fence);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Final submit failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    // Process results on CPU
    return process_gpu_results(detector/*, image*/);
}