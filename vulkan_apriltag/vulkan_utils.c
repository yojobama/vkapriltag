#include "vulkan_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

VkResult load_shader_from_spirv(VkDevice device, const uint32_t* spirv_data, size_t spirv_size, VkShaderModule* shader_module) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv_data
    };

    return vkCreateShaderModule(device, &create_info, NULL, shader_module);
}

VkResult load_shader_module(VkDevice device, const char* filename, VkShaderModule* shader_module) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filename);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    uint32_t* buffer = malloc(file_size);
    if (!buffer) {
        fclose(file);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    
    fread(buffer, 1, file_size, file);
    fclose(file);
    
    VkResult result = load_shader_from_spirv(device, buffer, file_size, shader_module);
    free(buffer);
    
    return result;
}

static uint32_t find_memory_type(VkPhysicalDeviceMemoryProperties memory_properties, 
                                uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkResult create_storage_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties memory_properties,
                              VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* buffer_memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    
    VkResult result = vkCreateBuffer(device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) return result;
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, *buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = find_memory_type(memory_properties, mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device, *buffer, NULL);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    result = vkAllocateMemory(device, &alloc_info, NULL, buffer_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, *buffer, NULL);
        return result;
    }
    
    vkBindBufferMemory(device, *buffer, *buffer_memory, 0);
    return VK_SUCCESS;
}

VkResult create_uniform_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties memory_properties,
                              VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* buffer_memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    
    VkResult result = vkCreateBuffer(device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) return result;
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, *buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = find_memory_type(memory_properties, mem_requirements.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    
    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device, *buffer, NULL);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    result = vkAllocateMemory(device, &alloc_info, NULL, buffer_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, *buffer, NULL);
        return result;
    }
    
    vkBindBufferMemory(device, *buffer, *buffer_memory, 0);
    return VK_SUCCESS;
}

VkResult create_descriptor_set_layout(VkDevice device, uint32_t binding_count, 
                                     VkDescriptorSetLayoutBinding* bindings,
                                     VkDescriptorSetLayout* layout) {
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindings = bindings,
    };
    return vkCreateDescriptorSetLayout(device, &layout_info, NULL, layout);
}

VkResult allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptor_pool,
                                VkDescriptorSetLayout layout, VkDescriptorSet* descriptor_set) {
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    return vkAllocateDescriptorSets(device, &alloc_info, descriptor_set);
}

VkResult create_compute_pipeline(VkDevice device, VkShaderModule shader_module,
                                VkDescriptorSetLayout descriptor_set_layout,
                                VkPipelineLayout* pipeline_layout,
                                VkPipeline* pipeline) {
    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    VkResult result = vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, pipeline_layout);
    if (result != VK_SUCCESS) return result;
    
    // Create compute pipeline
    VkPipelineShaderStageCreateInfo compute_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
    };
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = compute_stage_info,
        .layout = *pipeline_layout,
    };
    
    return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
}

VkResult allocate_command_buffer(VkDevice device, VkCommandPool command_pool, VkCommandBuffer* command_buffer) {
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    
    return vkAllocateCommandBuffers(device, &alloc_info, command_buffer);
}

VkResult begin_single_time_commands(VkDevice device, VkCommandPool command_pool, VkCommandBuffer* command_buffer) {
    VkResult result = allocate_command_buffer(device, command_pool, command_buffer);
    if (result != VK_SUCCESS) return result;
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    
    return vkBeginCommandBuffer(*command_buffer, &begin_info);
}

VkResult end_single_time_commands(VkDevice device, VkQueue queue, VkCommandPool command_pool, VkCommandBuffer command_buffer) {
    VkResult result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) return result;
    
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };
    
    result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) return result;
    
    result = vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    
    return result;
}

VkResult copy_buffer(VkDevice device, VkQueue queue, VkCommandPool command_pool,
                    VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
    VkCommandBuffer command_buffer;
    VkResult result = begin_single_time_commands(device, command_pool, &command_buffer);
    if (result != VK_SUCCESS) return result;
    
    VkBufferCopy copy_region = {
        .size = size,
    };
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);
    
    return end_single_time_commands(device, queue, command_pool, command_buffer);
}

VkResult copy_data_to_buffer(VkDevice device, VkDeviceMemory buffer_memory, 
                           const void* data, VkDeviceSize size, VkDeviceSize offset) {
    void* mapped_memory;
    VkResult result = vkMapMemory(device, buffer_memory, offset, size, 0, &mapped_memory);
    if (result != VK_SUCCESS) return result;
    
    memcpy(mapped_memory, data, size);
    vkUnmapMemory(device, buffer_memory);
    
    return VK_SUCCESS;
}

VkResult create_timestamp_query_pool(VkDevice device, uint32_t query_count, VkQueryPool* query_pool) {
    VkQueryPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = query_count,
    };
    
    return vkCreateQueryPool(device, &pool_info, NULL, query_pool);
}

double get_timestamp_period_ns(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    return properties.limits.timestampPeriod;
}

const char* vulkan_result_to_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "UNKNOWN_ERROR";
    }
}

void check_vk_result(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Vulkan error in %s: %s\n", operation, vulkan_result_to_string(result));
    }
}