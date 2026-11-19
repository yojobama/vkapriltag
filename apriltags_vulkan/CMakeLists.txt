cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ApriltagVulkan VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildTuning.cmake)

if(WIN32)
  # Stop <windows.h> defining min()/max() macros and pulling in winsock; set before add_subdirectory.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# The version in Version.h is the one users see; it must match the project version.
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/library/include/vkapriltag/Version.h VKAPRILTAG_VERSION_LINE REGEX "VKAPRILTAG_VERSION_STRING")
if(NOT VKAPRILTAG_VERSION_LINE MATCHES "\"${PROJECT_VERSION}\"")
  message(FATAL_ERROR "library/include/vkapriltag/Version.h (${VKAPRILTAG_VERSION_LINE}) does not match the project version ${PROJECT_VERSION}")
endif()

option(VKAPRILTAG_BUILD_APPS "Build the interactive sample app (V4L2 camera capture)" ON)
option(VKAPRILTAG_BUILD_TOOLS "Build the evaluation, benchmark and validation tools" ON)
# The library, its tests and the packages use no upstream libapriltag and fetch nothing. With this ON, stock upstream
# is fetched for the reference validation tools that compare against it (tools/upstream).
option(VKAPRILTAG_UPSTREAM_VALIDATION "Fetch upstream libapriltag and build the reference validation tools" OFF)
# Also patches upstream and builds gen_decode_vectors, which rewrites tests/own_decode_vectors.bin.
option(VKAPRILTAG_REGEN_VECTORS "Build the generator of the recorded decode test vectors (needs VKAPRILTAG_UPSTREAM_VALIDATION=ON)" OFF)
if(VKAPRILTAG_REGEN_VECTORS AND NOT VKAPRILTAG_UPSTREAM_VALIDATION)
  message(FATAL_ERROR "VKAPRILTAG_REGEN_VECTORS needs VKAPRILTAG_UPSTREAM_VALIDATION=ON")
endif()

# The vkapriltag library: Vulkan compute detection pipeline + CPU QuadDecode/TagDecoder tail.
add_subdirectory(library)

if(VKAPRILTAG_BUILD_APPS AND UNIX)
  add_subdirectory(apps)
endif()

option(VKAPRILTAG_BUILD_TESTS "Build and register the ctest unit and golden-detection tests" ON)

enable_testing()

if(VKAPRILTAG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(VKAPRILTAG_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Python (and, from the same interface, Java and C#) bindings of the public Detector.
option(VKAPRILTAG_BUILD_SWIG "Build the SWIG Python bindings (needs SWIG 4.1 or later and Python development files)" OFF)
if(VKAPRILTAG_BUILD_SWIG)
  add_subdirectory(bindings/swig)
endif()

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
