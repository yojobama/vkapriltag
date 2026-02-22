#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Include Vulkan headers when building full implementation
#ifdef BUILD_VULKAN_FULL
#include <vulkan/vulkan.h>
#else
// Forward declarations for Vulkan types when not building full implementation
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;
typedef void* VkDescriptorPool;
typedef void* VkShaderModule;
typedef void* VkPipeline;
typedef void* VkPipelineLayout;
typedef void* VkDescriptorSetLayout;
typedef void* VkBuffer;
typedef void* VkImage;
typedef void* VkImageView;
typedef void* VkDeviceMemory;
typedef void* VkDescriptorSet;
typedef void* VkCommandBuffer;
typedef void* VkSemaphore;
typedef void* VkFence;
typedef void* VkQueryPool;
typedef void* VkDebugUtilsMessengerEXT;
typedef struct { int dummy; } VkPhysicalDeviceMemoryProperties;
typedef uint64_t VkDeviceSize;
#endif

#include <stdint.h>
#include <stdbool.h>

#include "../apriltag.h"
#include "../common/image_u8.h"
#include "../common/zarray.h"

// Vulkan context for AprilTag detection
typedef struct vulkan_apriltag_context {
    // Vulkan core objects
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool command_pool;
    VkDescriptorPool descriptor_pool;
    
    // Compute queue family index
    uint32_t compute_queue_family_index;
    
    // Memory allocator
    VkPhysicalDeviceMemoryProperties memory_properties;
    
    // Shader modules for different pipeline stages
    VkShaderModule threshold_shader;
    VkShaderModule connected_components_shader;
    VkShaderModule gradient_shader;
    VkShaderModule line_fit_shader;
    VkShaderModule decimate_shader;
    VkShaderModule blur_shader;
    
    // Compute pipelines
    VkPipeline threshold_pipeline;
    VkPipeline connected_components_pipeline;
    VkPipeline gradient_pipeline;
    VkPipeline line_fit_pipeline;
    VkPipeline decimate_pipeline;
    VkPipeline blur_pipeline;
    
    // Pipeline layouts
    VkPipelineLayout threshold_layout;
    VkPipelineLayout connected_components_layout;
    VkPipelineLayout gradient_layout;
    VkPipelineLayout line_fit_layout;
    VkPipelineLayout decimate_layout;
    VkPipelineLayout blur_layout;
    
    // Descriptor set layouts
    VkDescriptorSetLayout threshold_desc_layout;
    VkDescriptorSetLayout connected_components_desc_layout;
    VkDescriptorSetLayout gradient_desc_layout;
    VkDescriptorSetLayout line_fit_desc_layout;
    VkDescriptorSetLayout decimate_desc_layout;
    VkDescriptorSetLayout blur_desc_layout;
    
    // Buffer management
    struct {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped;
        VkDeviceSize size;
    } staging_buffer;
    
    // Maximum image dimensions supported
    uint32_t max_image_width;
    uint32_t max_image_height;
    
    // Performance counters
    uint64_t gpu_time_ns;
    
    // Debug mode
    bool debug_enabled;
    VkDebugUtilsMessengerEXT debug_messenger;
    
} vulkan_apriltag_context_t;

// push constant parameters

typedef struct {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    float decimate_factor;
} decimate_params;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t kernel_size;
    uint32_t direction;
    float sigma;
} blur_params;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t tile_size;
    uint32_t min_white_black_diff;
} threshold_params;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t iteration;
    uint32_t max_iterations;
} cc_params;

typedef struct {
    uint32_t width;
    uint32_t height;
    float scale;
} gradient_params;


// GPU buffer for image data
typedef struct vulkan_image_buffer {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint32_t width;
    uint32_t height;
    VkDeviceSize size;
} vulkan_image_buffer_t;

// GPU buffer for intermediate results
typedef struct vulkan_compute_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorSet descriptor_set;
    VkDeviceSize size;
    uint32_t element_count;
} vulkan_compute_buffer_t;

// Forward declaration of the detector structure
typedef struct vulkan_apriltag_detector vulkan_apriltag_detector_t;

// Vulkan-accelerated detector
struct vulkan_apriltag_detector {
    apriltag_detector_t* base_detector;
    vulkan_apriltag_context_t* vulkan_ctx;
    
    // GPU buffers for pipeline stages
    vulkan_image_buffer_t input_image;
    vulkan_image_buffer_t decimated_image;
    vulkan_image_buffer_t blurred_image;
    vulkan_image_buffer_t threshold_image;
    vulkan_image_buffer_t gradient_x;
    vulkan_image_buffer_t gradient_y;
    vulkan_image_buffer_t gradient_mag;
    vulkan_image_buffer_t labels_image;
    
    vulkan_compute_buffer_t connected_components;
    vulkan_compute_buffer_t edge_points;
    vulkan_compute_buffer_t line_segments;
    vulkan_compute_buffer_t quad_candidates;
    
    // Staging buffers for CPU-GPU transfer
    vulkan_image_buffer_t staging_input;
    vulkan_compute_buffer_t staging_output;

    // Descriptor sets for pipeline stages
    VkDescriptorSet threshold_desc_set;
    VkDescriptorSet connected_components_desc_set;
    VkDescriptorSet gradient_desc_set;
    VkDescriptorSet decimate_desc_set;
    VkDescriptorSet blur_desc_set;

    // Blur kernel weights buffer
    VkBuffer blur_kernel_buffer;
    VkDeviceMemory blur_kernel_memory;
    
    // Command buffers for different stages
    VkCommandBuffer preprocess_cmd;
    VkCommandBuffer threshold_cmd;
    VkCommandBuffer connected_components_cmd;
    VkCommandBuffer gradient_cmd;
    VkCommandBuffer line_fit_cmd;
    
    // Synchronization
    VkSemaphore preprocess_semaphore;
    VkSemaphore threshold_semaphore;
    VkSemaphore connected_components_semaphore;
    VkSemaphore gradient_semaphore;
    VkSemaphore line_fit_semaphore;
    VkFence completion_fence;
    
    // Timing
    VkQueryPool timestamp_pool;
    
};

// Initialization and cleanup
vulkan_apriltag_context_t* vulkan_apriltag_context_create(bool enable_debug);
void vulkan_apriltag_context_destroy(vulkan_apriltag_context_t* ctx);

vulkan_apriltag_detector_t* vulkan_apriltag_detector_create(
    vulkan_apriltag_context_t* ctx,
    uint32_t max_width,
    uint32_t max_height
);
void vulkan_apriltag_detector_destroy(vulkan_apriltag_detector_t* detector);

// Add tag families (same as regular detector)
void vulkan_apriltag_detector_add_family(vulkan_apriltag_detector_t* detector, apriltag_family_t* family);
void vulkan_apriltag_detector_add_family_bits(vulkan_apriltag_detector_t* detector, apriltag_family_t* family, int bits_corrected);

// Main detection function
zarray_t* vulkan_apriltag_detector_detect(vulkan_apriltag_detector_t* detector, image_u8_t* image);

// Configuration
void vulkan_apriltag_detector_set_quad_decimate(vulkan_apriltag_detector_t* detector, float decimate);
void vulkan_apriltag_detector_set_quad_sigma(vulkan_apriltag_detector_t* detector, float sigma);
void vulkan_apriltag_detector_set_refine_edges(vulkan_apriltag_detector_t* detector, bool refine_edges);
void vulkan_apriltag_detector_set_decode_sharpening(vulkan_apriltag_detector_t* detector, double sharpening);
void vulkan_apriltag_detector_set_debug(vulkan_apriltag_detector_t* detector, bool debug);

// Performance monitoring
uint64_t vulkan_apriltag_detector_get_gpu_time_ns(vulkan_apriltag_detector_t* detector);
void vulkan_apriltag_detector_print_performance_stats(vulkan_apriltag_detector_t* detector);

// Utility functions
bool vulkan_apriltag_is_supported(void);
void vulkan_apriltag_print_device_info(vulkan_apriltag_context_t* ctx);

#ifdef __cplusplus
}
#endif