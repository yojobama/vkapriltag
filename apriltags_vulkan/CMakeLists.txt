cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ApriltagVulkan VERSION 1.0.0 LANGUAGES C CXX)

# The fetched apriltag library's own CMakeLists.txt declares a very old
# cmake_minimum_required(); this policy floor lets modern CMake configure it
# without erroring out, without weakening any policy in our own targets.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)

option(VKAPRILTAG_BUILD_APPS "Build the interactive sample app (V4L2 camera capture)" ON)
option(VKAPRILTAG_BUILD_TOOLS "Build the libapriltag cross-validation tool" ON)

# The vkapriltag library: Vulkan compute GPU detection pipeline + CPU
# QuadDecode/TagDecoder tail. Installable and consumable standalone via
# find_package(vkapriltag) / the vkapriltag::vkapriltag target.
add_subdirectory(library)

if(VKAPRILTAG_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(VKAPRILTAG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

enable_testing()
