#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shader loading utilities
VkResult load_shader_module(VkDevice device, const char* filename, VkShaderModule* shader_module);
VkResult load_shader_from_spirv(VkDevice device, const uint32_t* spirv_data, size_t spirv_size, VkShaderModule* shader_module);

// Buffer utilities
VkResult create_storage_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties memory_properties,
                              VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* buffer_memory);
VkResult create_uniform_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties memory_properties,
                              VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* buffer_memory);

// Descriptor set utilities
VkResult create_descriptor_set_layout(VkDevice device, uint32_t binding_count, 
                                     VkDescriptorSetLayoutBinding* bindings,
                                     VkDescriptorSetLayout* layout);
VkResult allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptor_pool,
                                VkDescriptorSetLayout layout, VkDescriptorSet* descriptor_set);

// Pipeline utilities
VkResult create_compute_pipeline(VkDevice device, VkShaderModule shader_module,
                                VkDescriptorSetLayout descriptor_set_layout,
                                VkPipelineLayout* pipeline_layout,
                                VkPipeline* pipeline, uint32_t push_constants_size);

// Command buffer utilities
VkResult allocate_command_buffer(VkDevice device, VkCommandPool command_pool, VkCommandBuffer* command_buffer);
VkResult begin_single_time_commands(VkDevice device, VkCommandPool command_pool, VkCommandBuffer* command_buffer);
VkResult end_single_time_commands(VkDevice device, VkQueue queue, VkCommandPool command_pool, VkCommandBuffer command_buffer);

// Memory utilities
VkResult copy_buffer(VkDevice device, VkQueue queue, VkCommandPool command_pool,
                    VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
VkResult copy_data_to_buffer(VkDevice device, VkDeviceMemory buffer_memory, 
                           const void* data, VkDeviceSize size, VkDeviceSize offset);
VkResult copy_buffer_to_image(VkDevice device, VkQueue queue, VkCommandPool command_pool,
                             VkBuffer src_buffer, VkImage dst_image,
                             uint32_t width, uint32_t height);
VkResult transition_image_layout(VkDevice device, VkQueue queue, VkCommandPool command_pool,
                                VkImage image, VkImageLayout old_layout, VkImageLayout new_layout);

// Timing utilities
VkResult create_timestamp_query_pool(VkDevice device, uint32_t query_count, VkQueryPool* query_pool);
double get_timestamp_period_ns(VkPhysicalDevice physical_device);

// Error checking
const char* vulkan_result_to_string(VkResult result);
void check_vk_result(VkResult result, const char* operation);

#ifdef __cplusplus
}
#endif