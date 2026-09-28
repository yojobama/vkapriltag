cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ApriltagVulkan VERSION 1.4.0 LANGUAGES C CXX)

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

if(WIN32)
  # <windows.h> - pulled in transitively on Windows by FetchContent'd
  # apriltag's common/pthreads_cross.h (included from apriltag.h) as well as
  # by the Vulkan headers - defines min()/max() macros unless NOMINMAX is
  # set, which break every std::min/std::max call in both our own code and
  # apriltag's. WIN32_LEAN_AND_MEAN keeps it from dragging in winsock etc.
  # Set at the top level (before add_subdirectory) so it also reaches
  # apriltag, which is fetched from inside library/CMakeLists.txt.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

option(VKAPRILTAG_BUILD_APPS "Build the interactive sample app (V4L2 camera capture)" ON)
option(VKAPRILTAG_BUILD_TOOLS "Build the libapriltag cross-validation tool" ON)

# The vkapriltag library: Vulkan compute GPU detection pipeline + CPU
# QuadDecode/TagDecoder tail. Installable and consumable standalone via
# find_package(vkapriltag) / the vkapriltag::vkapriltag target.
add_subdirectory(library)

if(VKAPRILTAG_BUILD_APPS AND UNIX)
  add_subdirectory(apps)
endif()

if(VKAPRILTAG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

enable_testing()

# ---------------------------------------------------------------------------
# CPack: packages exactly what library/CMakeLists.txt's install() rules
# cover (static lib, public headers, compiled SPIR-V shader corpus, CMake
# package config) - no tools/apps, matching the release workflow's
# -DVKAPRILTAG_BUILD_APPS=OFF -DVKAPRILTAG_BUILD_TOOLS=OFF. This block is
# inert unless `cpack` is actually invoked.
# ---------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "vkapriltag")
# Overridable via -DCPACK_PACKAGE_VERSION=X.Y.Z at configure time (the
# release workflow derives this from the pushed git tag rather than the
# in-tree development version above). A plain set() here would shadow that
# command-line cache value for the rest of this configure run regardless of
# how CPACK_PACKAGE_VERSION was set, so only default it when unset.
if(NOT CPACK_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
endif()
set(CPACK_PACKAGE_VENDOR "vkapriltag")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "Vulkan-compute AprilTag detection pipeline for embedded/mobile GPUs")
set(CPACK_PACKAGE_CONTACT "https://github.com/yojobama/vkapriltag")
set(CPACK_GENERATOR "DEB")
# dpkg-shlibdeps-derived Depends: (e.g. libvulkan1); requires dpkg-dev on
# the packaging host.
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
# Standard <name>_<version>_<arch>.deb naming; CPACK_DEBIAN_PACKAGE_ARCHITECTURE
# is left unset so CPack auto-detects it from `dpkg --print-architecture` on
# the build host - correct as long as packaging always happens natively
# (amd64 host -> amd64 package, arm64 host -> arm64 package), never
# cross-compiled.
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
include(CPack)
