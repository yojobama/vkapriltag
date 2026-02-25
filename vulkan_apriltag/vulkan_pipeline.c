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
    VkDeviceSize image_size = (VkDeviceSize)image->width * image->height;
    if (image_size > ctx->staging_buffer.size) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    uint8_t* dst = (uint8_t*)ctx->staging_buffer.mapped;
    for (int y = 0; y < image->height; y++) {
        memcpy(dst + (size_t)y * image->width, image->buf + (size_t)y * image->stride, image->width);
    }
    
    result = copy_buffer_to_image(ctx->device, ctx->compute_queue, ctx->command_pool,
                                 ctx->staging_buffer.buffer, detector->input_image.image,
                                 image->width, image->height);
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
        decimate_params decimateParams = {
            image->width, image->height,
            (uint32_t)(image->width / detector->base_detector->quad_decimate),
            (uint32_t)(image->height / detector->base_detector->quad_decimate),
            detector->base_detector->quad_decimate
        };
        
        // Update push constants with decimation parameters
        vkCmdPushConstants(detector->preprocess_cmd, ctx->decimate_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(decimate_params), &decimateParams);
        
        // Bind descriptor sets for input and output buffers
        VkDescriptorImageInfo decimate_input_info = { .imageView = detector->input_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo decimate_output_info = { .imageView = detector->decimated_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet decimate_writes[] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->decimate_desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &decimate_input_info },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->decimate_desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &decimate_output_info }
        };
        vkUpdateDescriptorSets(ctx->device, (uint32_t)(sizeof(decimate_writes) / sizeof(decimate_writes[0])), decimate_writes, 0, NULL);
        vkCmdBindDescriptorSets(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->decimate_layout, 0, 1, &detector->decimate_desc_set,
                               0, NULL);
        
        uint32_t group_x = (decimateParams.output_width + 15) / 16;
        uint32_t group_y = (decimateParams.output_height + 15) / 16;
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
        if (kernel_size > 31) kernel_size = 31;
        uint32_t blur_width = detector->base_detector->quad_decimate > 1.0f ?
            (uint32_t)(image->width / detector->base_detector->quad_decimate) : image->width;
        uint32_t blur_height = detector->base_detector->quad_decimate > 1.0f ?
            (uint32_t)(image->height / detector->base_detector->quad_decimate) : image->height;
        blur_params blurParams = {
            blur_width, blur_height,
            kernel_size, 0, // Horizontal pass first
            detector->base_detector->quad_sigma
        };
        
        // Memory barrier
        VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };


        // Update blur kernel weights
        uint32_t weight_count = kernel_size;
        if (weight_count > 31) weight_count = 31;
        float weights[31] = { 0.0f };
        float sum = 0.0f;
        int radius = (int)weight_count / 2;
        for (uint32_t i = 0; i < weight_count; i++) {
            int x = (int)i - radius;
            float w = expf(-(x * x) / (2.0f * detector->base_detector->quad_sigma * detector->base_detector->quad_sigma));
            weights[i] = w;
            sum += w;
        }
        for (uint32_t i = 0; i < weight_count; i++) {
            weights[i] /= sum;
        }
        copy_data_to_buffer(ctx->device, detector->blur_kernel_memory, weights, sizeof(weights), 0);

        // Update push constants with blur parameters
        vkCmdPushConstants(detector->preprocess_cmd, ctx->blur_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_params), &blurParams);
        
        // Bind descriptor sets for blur input/output
        VkDescriptorImageInfo blur_input_info = {
            .imageView = detector->base_detector->quad_decimate > 1.0f ? detector->decimated_image.view : detector->input_image.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL
        };
        VkDescriptorImageInfo blur_output_info = { .imageView = detector->blurred_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo blur_kernel_info = { .buffer = detector->blur_kernel_buffer, .offset = 0, .range = sizeof(float) * 31 };
        VkWriteDescriptorSet blur_writes[] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->blur_desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &blur_input_info },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->blur_desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &blur_output_info },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->blur_desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &blur_kernel_info }
        };
        vkUpdateDescriptorSets(ctx->device, (uint32_t)(sizeof(blur_writes) / sizeof(blur_writes[0])), blur_writes, 0, NULL);
        vkCmdBindDescriptorSets(detector->preprocess_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx->blur_layout, 0, 1, &detector->blur_desc_set, 0, NULL);
        
        // Horizontal pass
        uint32_t group_count = (blur_width + 255) / 256;
        vkCmdDispatch(detector->preprocess_cmd, group_count, blur_height, 1);
        
        vkCmdPipelineBarrier(detector->preprocess_cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, NULL, 0, NULL);
        
        
        // Vertical pass
        blurParams.direction = 1;
        vkCmdPushConstants(detector->preprocess_cmd, ctx->blur_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_params), &blurParams);
        
        group_count = (blur_height + 255) / 256;
        vkCmdDispatch(detector->preprocess_cmd, blur_width, group_count, 1);
        
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
    uint32_t process_width = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->width / detector->base_detector->quad_decimate) : image->width;
    uint32_t process_height = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->height / detector->base_detector->quad_decimate) : image->height;
    threshold_params thresholdParams = {
        process_width, process_height,
        16, // Tile size for adaptive thresholding
        (uint32_t)detector->base_detector->qtp.min_white_black_diff
    };
    
    
    // Update push constants with threshold parameters
    vkCmdPushConstants(detector->threshold_cmd, ctx->threshold_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(threshold_params), &thresholdParams);
    
    // Bind descriptor sets for input image
    // Use blurred image if blur was applied, otherwise use decimated or original
    VkDescriptorImageInfo threshold_input_info = {
        .imageView = detector->base_detector->quad_sigma > 0.0f ? detector->blurred_image.view :
                     (detector->base_detector->quad_decimate > 1.0f ? detector->decimated_image.view : detector->input_image.view),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    VkDescriptorImageInfo threshold_output_info = { .imageView = detector->threshold_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet threshold_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->threshold_desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &threshold_input_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->threshold_desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &threshold_output_info }
    };
    vkUpdateDescriptorSets(ctx->device, (uint32_t)(sizeof(threshold_writes) / sizeof(threshold_writes[0])), threshold_writes, 0, NULL);
    vkCmdBindDescriptorSets(detector->threshold_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->threshold_layout, 0, 1, &detector->threshold_desc_set, 0, NULL);

    uint32_t group_x = (process_width + 15) / 16;
    uint32_t group_y = (process_height + 15) / 16;
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

    vkCmdFillBuffer(detector->connected_components_cmd, detector->connected_components.buffer, 0,
                    detector->connected_components.size, 0);
    VkMemoryBarrier clear_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    };
    vkCmdPipelineBarrier(detector->connected_components_cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &clear_barrier, 0, NULL, 0, NULL);
    
    vkCmdBindPipeline(detector->connected_components_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                     ctx->connected_components_pipeline);
    
    uint32_t process_width = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->width / detector->base_detector->quad_decimate) : image->width;
    uint32_t process_height = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->height / detector->base_detector->quad_decimate) : image->height;

    // Run multiple iterations for convergence
    uint32_t max_iterations = 10;

    VkDescriptorImageInfo threshold_info = { .imageView = detector->threshold_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo labels_info = { .imageView = detector->labels_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo components_info = { .buffer = detector->connected_components.buffer, .offset = 0, .range = detector->connected_components.size };
    VkWriteDescriptorSet cc_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->connected_components_desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &threshold_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->connected_components_desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &labels_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->connected_components_desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &components_info }
    };
    vkUpdateDescriptorSets(ctx->device, (uint32_t)(sizeof(cc_writes) / sizeof(cc_writes[0])), cc_writes, 0, NULL);
    vkCmdBindDescriptorSets(detector->connected_components_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->connected_components_layout, 0, 1,
                           &detector->connected_components_desc_set, 0, NULL);
    
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        cc_params ccParams = {
            process_width, process_height,
            iteration, max_iterations
        };
        
        
        // Update push constants with connected components parameters
        vkCmdPushConstants(detector->connected_components_cmd, ctx->connected_components_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cc_params), &ccParams);
        
        uint32_t group_x = (process_width + 15) / 16;
        uint32_t group_y = (process_height + 15) / 16;
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
    
    uint32_t process_width = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->width / detector->base_detector->quad_decimate) : image->width;
    uint32_t process_height = detector->base_detector->quad_decimate > 1.0f ?
        (uint32_t)(image->height / detector->base_detector->quad_decimate) : image->height;
    gradient_params gradientParams = {
        process_width, process_height,
        1.0f / 8.0f // Sobel normalization
    };
    
    
    // Update push constants with gradient parameters
    vkCmdPushConstants(detector->gradient_cmd, ctx->gradient_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gradient_params), &gradientParams);
    
    VkDescriptorImageInfo gradient_input_info = {
        .imageView = detector->base_detector->quad_sigma > 0.0f ? detector->blurred_image.view :
                     (detector->base_detector->quad_decimate > 1.0f ? detector->decimated_image.view : detector->input_image.view),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    VkDescriptorImageInfo gradient_x_info = { .imageView = detector->gradient_x.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo gradient_y_info = { .imageView = detector->gradient_y.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo gradient_mag_info = { .imageView = detector->gradient_mag.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet gradient_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->gradient_desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &gradient_input_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->gradient_desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &gradient_x_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->gradient_desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &gradient_y_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = detector->gradient_desc_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &gradient_mag_info }
    };
    vkUpdateDescriptorSets(ctx->device, (uint32_t)(sizeof(gradient_writes) / sizeof(gradient_writes[0])), gradient_writes, 0, NULL);
    vkCmdBindDescriptorSets(detector->gradient_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->gradient_layout, 0, 1, &detector->gradient_desc_set, 0, NULL);
    
    uint32_t group_x = (process_width + 15) / 16;
    uint32_t group_y = (process_height + 15) / 16;
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
static zarray_t* process_gpu_results(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
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
    
    if (detector->base_detector && image) {
        return apriltag_detector_detect(detector->base_detector, image);
    }

    return zarray_create(sizeof(apriltag_detection_t*));
}

zarray_t* vulkan_apriltag_detector_detect(vulkan_apriltag_detector_t* detector, image_u8_t* image) {
    if (!detector || !image) {
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    vulkan_apriltag_context_t* ctx = detector->vulkan_ctx;
    
    // Check if image dimensions are within supported limits
    if ((uint32_t)image->width > ctx->max_image_width || (uint32_t)image->height > ctx->max_image_height ||
        (uint32_t)image->width > detector->input_image.width || (uint32_t)image->height > detector->input_image.height) {
        fprintf(stderr, "Image dimensions exceed GPU limits\n");
        if (detector->base_detector) {
            return apriltag_detector_detect(detector->base_detector, image);
        }
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    // Execute GPU pipeline stages
    VkResult result;

    result = execute_preprocessing_pipeline(detector, image);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Preprocessing pipeline failed: %s\n", vulkan_result_to_string(result));
        if (detector->base_detector) {
            return apriltag_detector_detect(detector->base_detector, image);
        }
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

    result = vkResetFences(ctx->device, 1, &detector->completion_fence);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to reset completion fence: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    result = vkQueueSubmit(ctx->compute_queue, 1, &final_submit, detector->completion_fence);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Final submit failed: %s\n", vulkan_result_to_string(result));
        return zarray_create(sizeof(apriltag_detection_t*));
    }
    
    // Process results on CPU
    return process_gpu_results(detector, image);
}