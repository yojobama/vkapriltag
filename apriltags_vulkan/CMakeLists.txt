cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ApriltagVulkan LANGUAGES C CXX)

# The fetched apriltag library's own CMakeLists.txt declares a very old
# cmake_minimum_required(); this policy floor lets modern CMake configure it
# without erroring out, without weakening any policy in our own targets.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# The AprilTag CPU decode library.  This is a pure-C library (no CUDA/OpenCV
# dependency) that performs the final quad -> tag id/pose decode step, exactly
# like the original CUDA implementation does (it also ran this part on the
# CPU).  We fetch the same fork used by the CUDA project because it contains
# the quad_decode_index()/reconcile_detections() helper entry points the
# GPU pipeline needs.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  apriltag
  GIT_REPOSITORY https://github.com/cgpadwick/apriltag.git
  GIT_TAG 3.3.0
)
FetchContent_MakeAvailable(apriltag)

# Newer GCC (15+) defaults to a C standard where implicit/incompatible
# pointer-type function calls (used throughout this older C library, e.g.
# common/getopt.c's zarray_vmap(..., free)) are hard errors instead of
# warnings. Pin it to gnu17 so it builds unmodified.
if(TARGET apriltag)
  target_compile_options(apriltag PRIVATE -std=gnu17)
endif()


include(cmake/CompileShaders.cmake)

set(SHADER_SOURCES
  shaders/decimate.comp
  shaders/block_minmax.comp
  shaders/block_filter.comp
  shaders/threshold.comp
  shaders/uf_init.comp
  shaders/uf_merge.comp
  shaders/uf_compress.comp
  shaders/uf_final.comp
  shaders/blob_diff.comp
  shaders/fill_max_key.comp
  shaders/mark_heads.comp
  shaders/scan_block.comp
  shaders/scan_add_offsets.comp
  shaders/init_extents.comp
  shaders/reduce_extents.comp
  shaders/select_blobs.comp
  shaders/rewrite_index_points.comp
  shaders/bitonic_sort.comp
  shaders/bitonic_local.comp
  shaders/radix_histogram.comp
  shaders/radix_scan_hist.comp
  shaders/radix_scatter.comp
  shaders/gather_generic.comp
  shaders/compute_line_fit_points.comp
)

compile_shaders(apriltag_vulkan_shaders "${SHADER_SOURCES}" SHADER_OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Core detection pipeline (Vulkan compute GPU stages + CPU QuadDecode/
# TagDecoder tail), shared between the interactive apriltag_vulkan executable
# (V4L2 camera capture) and the apriltag_vulkan_validate tool (static PGM
# image input, used to cross-check final decoded tag IDs against the
# fetched apriltag library's own unmodified reference detector).
# ---------------------------------------------------------------------------
add_library(apriltag_vulkan_core STATIC
  src/vk/Context.cpp
  src/vk/Buffer.cpp
  src/vk/Shader.cpp
  src/vk/ComputePipeline.cpp
  src/gpu/GpuDetector.cpp
  src/gpu/QuadDecode.cpp
  src/apriltag_family.cpp
  src/TagDecoder.cpp
  src/common/pgm_io.cpp
  src/common/WorkerPool.cpp
)

add_dependencies(apriltag_vulkan_core apriltag_vulkan_shaders)

target_include_directories(apriltag_vulkan_core PUBLIC src ${SHADER_OUTPUT_DIR})
target_compile_definitions(apriltag_vulkan_core PUBLIC SHADER_DIR="${SHADER_OUTPUT_DIR}")

target_link_libraries(apriltag_vulkan_core PUBLIC
  Vulkan::Vulkan
  Threads::Threads
  apriltag
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(apriltag_vulkan
  src/main.cpp
  src/common/v4l2_capture.cpp
)
target_link_libraries(apriltag_vulkan PRIVATE apriltag_vulkan_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Static-image validation tool: compares our pipeline's decoded tag IDs
# against the official (unmodified) apriltag_detector_detect() reference.
add_executable(apriltag_vulkan_validate
  src/tools/validate_against_libapriltag.cpp
)
target_link_libraries(apriltag_vulkan_validate PRIVATE apriltag_vulkan_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

enable_testing()
