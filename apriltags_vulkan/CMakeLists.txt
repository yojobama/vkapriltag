cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ApriltagVulkan VERSION 1.4.0 LANGUAGES C CXX)

# Policy floor so the fetched apriltag's old cmake_minimum_required() configures.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)

if(WIN32)
  # Stop <windows.h> defining min()/max() macros and pulling in winsock; set before add_subdirectory.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

option(VKAPRILTAG_BUILD_APPS "Build the interactive sample app (V4L2 camera capture)" ON)
option(VKAPRILTAG_BUILD_TOOLS "Build the libapriltag cross-validation tool" ON)

# The vkapriltag library: Vulkan compute detection pipeline + CPU QuadDecode/TagDecoder tail.
add_subdirectory(library)

if(VKAPRILTAG_BUILD_APPS AND UNIX)
  add_subdirectory(apps)
endif()

if(VKAPRILTAG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

enable_testing()

# ---------------------------------------------------------------------------
# CPack: packages the install() rules of library/CMakeLists.txt; inert unless cpack is invoked.
# ---------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "vkapriltag")
# Defaults to the in-tree version unless CPACK_PACKAGE_VERSION is already set.
if(NOT CPACK_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
endif()
set(CPACK_PACKAGE_VENDOR "vkapriltag")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "Vulkan-compute AprilTag detection pipeline for embedded/mobile GPUs")
set(CPACK_PACKAGE_CONTACT "https://github.com/yojobama/vkapriltag")
set(CPACK_GENERATOR "DEB")
# dpkg-shlibdeps-derived Depends: (requires dpkg-dev).
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
# <name>_<version>_<arch>.deb; the architecture is auto-detected from the build host.
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
include(CPack)
