#include "vulkan_apriltag.h"
#include "vulkan_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

// Validation layers for debug builds
static const char* validation_layers[] = {
    "VK_LAYER_KHRONOS_validation"
};

static const char* device_extensions[1] = { NULL };

// static const uint32_t validation_layer_count = sizeof(validation_layers) / sizeof(validation_layers[0]);
static const uint32_t validation_layer_count = 1;
static const uint32_t device_extension_count = 0; // Empty array

// Debug callback for validation layers
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "Vulkan validation layer: %s\n", callback_data->pMessage);
    }
    return VK_FALSE;
}

// Helper function to find memory type
static uint32_t find_memory_type(vulkan_apriltag_context_t* ctx, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < ctx->memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (ctx->memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Helper function to create buffer
static VkResult create_buffer(vulkan_apriltag_context_t* ctx, VkDeviceSize size, 
                             VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                             VkBuffer* buffer, VkDeviceMemory* buffer_memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    
    VkResult result = vkCreateBuffer(ctx->device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) return result;
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = find_memory_type(ctx, mem_requirements.memoryTypeBits, properties)
    };
    
    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    result = vkAllocateMemory(ctx->device, &alloc_info, NULL, buffer_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        return result;
    }
    
    vkBindBufferMemory(ctx->device, *buffer, *buffer_memory, 0);
    return VK_SUCCESS;
}

static bool check_validation_layer_support(void) {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, NULL);
    
    VkLayerProperties* available_layers = malloc(layer_count * sizeof(VkLayerProperties));
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers);
    
    for (uint32_t i = 0; i < validation_layer_count; i++) {
        bool layer_found = false;
        for (uint32_t j = 0; j < layer_count; j++) {
            if (strcmp(validation_layers[i], available_layers[j].layerName) == 0) {
                layer_found = true;
                break;
            }
        }
        if (!layer_found) {
            free(available_layers);
            return false;
        }
    }
    
    free(available_layers);
    return true;
}

static VkResult create_instance(vulkan_apriltag_context_t* ctx, bool enable_validation) {
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan AprilTag",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1
    };
    
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    
    // Extensions for debug
    const char* extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    
    if (enable_validation) {
        if (!check_validation_layer_support()) {
            fprintf(stderr, "Validation layers requested but not available\n");
            return VK_ERROR_LAYER_NOT_PRESENT;
        }
        create_info.enabledLayerCount = validation_layer_count;
        create_info.ppEnabledLayerNames = validation_layers;
        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = extensions;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.enabledExtensionCount = 0;
    }
    
    return vkCreateInstance(&create_info, NULL, &ctx->instance);
}

static VkResult setup_debug_messenger(vulkan_apriltag_context_t* ctx) {
    if (!ctx->debug_enabled) return VK_SUCCESS;
    
    VkDebugUtilsMessengerCreateInfoEXT create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback
    };
    
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT");
    
    if (func != NULL) {
        return func(ctx->instance, &create_info, NULL, &ctx->debug_messenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static VkResult pick_physical_device(vulkan_apriltag_context_t* ctx) {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    
    if (device_count == 0) {
        return VK_ERROR_DEVICE_LOST;
    }
    
    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);
    
    // Score devices and pick the best one
    int best_score = -1;
    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties properties;
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        vkGetPhysicalDeviceFeatures(devices[i], &features);
        
        int score = 0;
        
        // Prefer discrete GPU
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }
        
        // Check for compute queue support
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, NULL);
        
        VkQueueFamilyProperties* queue_families = malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, queue_families);
        
        bool has_compute_queue = false;
        for (uint32_t j = 0; j < queue_family_count; j++) {
            if (queue_families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute_queue = true;
                ctx->compute_queue_family_index = j;
                break;
            }
        }
        
        free(queue_families);
        
        if (!has_compute_queue) {
            continue;
        }
        
        score += properties.limits.maxComputeWorkGroupCount[0];
        score += properties.limits.maxComputeWorkGroupSize[0];
        
        if (score > best_score) {
            best_score = score;
            best_device = devices[i];
        }
    }
    
    free(devices);
    
    if (best_device == VK_NULL_HANDLE) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    ctx->physical_device = best_device;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &ctx->memory_properties);
    
    return VK_SUCCESS;
}

static VkResult create_logical_device(vulkan_apriltag_context_t* ctx) {
    VkDeviceQueueCreateInfo queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_queue_family_index,
        .queueCount = 1
    };
    
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    
    VkDeviceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = NULL,
        .enabledExtensionCount = device_extension_count,
        .ppEnabledExtensionNames = device_extensions
    };
    
    if (ctx->debug_enabled) {
        create_info.enabledLayerCount = validation_layer_count;
        create_info.ppEnabledLayerNames = validation_layers;
    } else {
        create_info.enabledLayerCount = 0;
    }
    
    VkResult result = vkCreateDevice(ctx->physical_device, &create_info, NULL, &ctx->device);
    if (result != VK_SUCCESS) return result;
    
    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family_index, 0, &ctx->compute_queue);
    
    return VK_SUCCESS;
}

static VkResult create_command_pool(vulkan_apriltag_context_t* ctx) {
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->compute_queue_family_index
    };
    
    return vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);
}

static VkResult create_descriptor_pool(vulkan_apriltag_context_t* ctx) {
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 50 }
    };
    
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
        .pPoolSizes = pool_sizes,
        .maxSets = 1000
    };
    
    return vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->descriptor_pool);
}

vulkan_apriltag_context_t* vulkan_apriltag_context_create(bool enable_debug) {
    vulkan_apriltag_context_t* ctx = calloc(1, sizeof(vulkan_apriltag_context_t));
    if (!ctx) return NULL;
    
    ctx->debug_enabled = enable_debug;
    
    // Create Vulkan instance
    if (create_instance(ctx, enable_debug) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    // Setup debug messenger
    if (setup_debug_messenger(ctx) != VK_SUCCESS && enable_debug) {
        fprintf(stderr, "Failed to setup debug messenger\n");
    }
    
    // Pick physical device
    if (pick_physical_device(ctx) != VK_SUCCESS) {
        fprintf(stderr, "Failed to find suitable GPU\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    // Create logical device
    if (create_logical_device(ctx) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create logical device\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    // Create command pool
    if (create_command_pool(ctx) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create command pool\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    // Create descriptor pool
    if (create_descriptor_pool(ctx) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    // Get device limits
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &properties);
    ctx->max_image_width = properties.limits.maxComputeWorkGroupCount[0] * properties.limits.maxComputeWorkGroupSize[0];
    ctx->max_image_height = properties.limits.maxComputeWorkGroupCount[1] * properties.limits.maxComputeWorkGroupSize[1];
    
    // Create staging buffer (32MB default)
    VkDeviceSize staging_size = 32 * 1024 * 1024;
    if (create_buffer(ctx, staging_size, 
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &ctx->staging_buffer.buffer, &ctx->staging_buffer.memory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create staging buffer\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    ctx->staging_buffer.size = staging_size;
    
    // Map staging buffer
    if (vkMapMemory(ctx->device, ctx->staging_buffer.memory, 0, staging_size, 0, &ctx->staging_buffer.mapped) != VK_SUCCESS) {
        fprintf(stderr, "Failed to map staging buffer\n");
        vulkan_apriltag_context_destroy(ctx);
        return NULL;
    }
    
    return ctx;
}

void vulkan_apriltag_context_destroy(vulkan_apriltag_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        
        if (ctx->staging_buffer.memory != VK_NULL_HANDLE) {
            if (ctx->staging_buffer.mapped) {
                vkUnmapMemory(ctx->device, ctx->staging_buffer.memory);
            }
            vkFreeMemory(ctx->device, ctx->staging_buffer.memory, NULL);
        }
        if (ctx->staging_buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->staging_buffer.buffer, NULL);
        }
        
        // Destroy pipelines and layouts
        if (ctx->threshold_pipeline) vkDestroyPipeline(ctx->device, ctx->threshold_pipeline, NULL);
        if (ctx->connected_components_pipeline) vkDestroyPipeline(ctx->device, ctx->connected_components_pipeline, NULL);
        if (ctx->gradient_pipeline) vkDestroyPipeline(ctx->device, ctx->gradient_pipeline, NULL);
        if (ctx->line_fit_pipeline) vkDestroyPipeline(ctx->device, ctx->line_fit_pipeline, NULL);
        if (ctx->decimate_pipeline) vkDestroyPipeline(ctx->device, ctx->decimate_pipeline, NULL);
        if (ctx->blur_pipeline) vkDestroyPipeline(ctx->device, ctx->blur_pipeline, NULL);
        
        if (ctx->threshold_layout) vkDestroyPipelineLayout(ctx->device, ctx->threshold_layout, NULL);
        if (ctx->connected_components_layout) vkDestroyPipelineLayout(ctx->device, ctx->connected_components_layout, NULL);
        if (ctx->gradient_layout) vkDestroyPipelineLayout(ctx->device, ctx->gradient_layout, NULL);
        if (ctx->line_fit_layout) vkDestroyPipelineLayout(ctx->device, ctx->line_fit_layout, NULL);
        if (ctx->decimate_layout) vkDestroyPipelineLayout(ctx->device, ctx->decimate_layout, NULL);
        if (ctx->blur_layout) vkDestroyPipelineLayout(ctx->device, ctx->blur_layout, NULL);
        
        // Destroy descriptor set layouts
        if (ctx->threshold_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->threshold_desc_layout, NULL);
        if (ctx->connected_components_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->connected_components_desc_layout, NULL);
        if (ctx->gradient_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->gradient_desc_layout, NULL);
        if (ctx->line_fit_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->line_fit_desc_layout, NULL);
        if (ctx->decimate_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->decimate_desc_layout, NULL);
        if (ctx->blur_desc_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->blur_desc_layout, NULL);
        
        // Destroy shader modules
        if (ctx->threshold_shader) vkDestroyShaderModule(ctx->device, ctx->threshold_shader, NULL);
        if (ctx->connected_components_shader) vkDestroyShaderModule(ctx->device, ctx->connected_components_shader, NULL);
        if (ctx->gradient_shader) vkDestroyShaderModule(ctx->device, ctx->gradient_shader, NULL);
        if (ctx->line_fit_shader) vkDestroyShaderModule(ctx->device, ctx->line_fit_shader, NULL);
        if (ctx->decimate_shader) vkDestroyShaderModule(ctx->device, ctx->decimate_shader, NULL);
        if (ctx->blur_shader) vkDestroyShaderModule(ctx->device, ctx->blur_shader, NULL);
        
        if (ctx->descriptor_pool) vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
        if (ctx->command_pool) vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
    }
    
    if (ctx->debug_messenger != VK_NULL_HANDLE && ctx->instance != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(ctx->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != NULL) {
            func(ctx->instance, ctx->debug_messenger, NULL);
        }
    }
    
    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx->instance, NULL);
    }
    
    free(ctx);
}

bool vulkan_apriltag_is_supported(void) {
    // Check if Vulkan loader is available
    VkInstance test_instance;
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_1
    };
    
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    
    VkResult result = vkCreateInstance(&create_info, NULL, &test_instance);
    if (result == VK_SUCCESS) {
        vkDestroyInstance(test_instance, NULL);
        return true;
    }
    
    return false;
}

void vulkan_apriltag_print_device_info(vulkan_apriltag_context_t* ctx) {
    if (!ctx || ctx->physical_device == VK_NULL_HANDLE) {
        printf("No Vulkan device available\n");
        return;
    }
    
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &properties);
    
    printf("Vulkan Device Info:\n");
    printf("  Device Name: %s\n", properties.deviceName);
    printf("  Device Type: ");
    switch (properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: printf("Integrated GPU\n"); break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: printf("Discrete GPU\n"); break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: printf("Virtual GPU\n"); break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: printf("CPU\n"); break;
        default: printf("Unknown\n"); break;
    }
    printf("  API Version: %d.%d.%d\n", 
           VK_VERSION_MAJOR(properties.apiVersion),
           VK_VERSION_MINOR(properties.apiVersion),
           VK_VERSION_PATCH(properties.apiVersion));
    printf("  Driver Version: %d.%d.%d\n",
           VK_VERSION_MAJOR(properties.driverVersion),
           VK_VERSION_MINOR(properties.driverVersion),
           VK_VERSION_PATCH(properties.driverVersion));
    printf("  Max Compute Work Group Size: [%d, %d, %d]\n",
           properties.limits.maxComputeWorkGroupSize[0],
           properties.limits.maxComputeWorkGroupSize[1],
           properties.limits.maxComputeWorkGroupSize[2]);
    printf("  Max Compute Work Group Count: [%d, %d, %d]\n",
           properties.limits.maxComputeWorkGroupCount[0],
           properties.limits.maxComputeWorkGroupCount[1],
           properties.limits.maxComputeWorkGroupCount[2]);
    printf("  Max Memory Allocation Count: %d\n", properties.limits.maxMemoryAllocationCount);
    printf("  Max Buffer Size: %llu bytes\n", (unsigned long long)properties.limits.maxStorageBufferRange);
}