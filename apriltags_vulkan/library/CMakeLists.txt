# ---------------------------------------------------------------------------
# vkapriltag: the Vulkan-compute GPU detection pipeline + CPU QuadDecode/
# TagDecoder tail, packaged as a standalone, installable library. Consumers
# link against the `vkapriltag::vkapriltag` target (or plain `vkapriltag` when
# add_subdirectory()'d in-tree).
# ---------------------------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# The AprilTag CPU decode library.  This is a pure-C library (no CUDA/OpenCV
# dependency) that performs the final quad -> tag id/pose decode step, exactly
# like the original CUDA implementation does (it also ran this part on the
# CPU). We fetch the official upstream AprilRobotics library and apply a small
# patch that extracts two blocks apriltag_detector_detect() already contains
# (the per-quad family decode, and the cross-family duplicate-reconciliation
# pass) into non-static entry points - quad_decode_index()/
# reconcile_detections() - so the GPU pipeline can drive them directly with
# its own externally-detected quads instead of apriltag_detector_detect()'s
# own (CPU-only) quad-detection step. The patch changes no behavior of
# apriltag_detector_detect() itself.
# ---------------------------------------------------------------------------
include(FetchContent)
set(APRILTAG_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/patches/apriltag-expose-decode-steps.patch)
FetchContent_Declare(
  apriltag
  GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git
  GIT_TAG v3.4.5
  # Idempotent: skips re-applying if the patch is already present (needed
  # since FetchContent may re-run the patch step across reconfigures).
  PATCH_COMMAND git apply --reverse --check ${APRILTAG_PATCH_FILE} || git apply ${APRILTAG_PATCH_FILE}
)
FetchContent_MakeAvailable(apriltag)

# Newer GCC (15+) defaults to a C standard where implicit/incompatible
# pointer-type function calls (used throughout this older C library, e.g.
# common/getopt.c's zarray_vmap(..., free)) are hard errors instead of
# warnings. Pin it to gnu17 so it builds unmodified.
if(TARGET apriltag)
  target_compile_options(apriltag PRIVATE -std=gnu17)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompileShaders.cmake)

set(SHADER_SOURCES
  shaders/decimate.comp
  shaders/decimate_u8.comp
  shaders/block_minmax.comp
  shaders/block_minmax_u8.comp
  shaders/block_filter.comp
  shaders/threshold.comp
  shaders/threshold_u8.comp
  shaders/uf_init.comp
  shaders/uf_init_u8.comp
  shaders/uf_merge.comp
  shaders/uf_merge_u8.comp
  shaders/uf_compress.comp
  shaders/uf_final.comp
  shaders/uf_final_subgroup.comp
  shaders/label_pixels.comp
  shaders/blob_diff.comp
  shaders/blob_diff_u8.comp
  shaders/blob_diff_subgroup.comp
  shaders/blob_diff_u8_subgroup.comp
  shaders/scan_block.comp
  shaders/scan_add_offsets.comp
  shaders/init_extents.comp
  shaders/select_blobs.comp
  shaders/extract_blob_counts.comp
  shaders/sort_points_local.comp
  shaders/sort_points_local_u8.comp
  shaders/hash_group.comp
  shaders/build_indirect_args.comp
  shaders/reduce_extents_hash.comp
  shaders/reduce_extents_hash_subgroup.comp
  shaders/scatter_index_points.comp
  shaders/compute_moments_prefix.comp
)

compile_shaders(vkapriltag_shaders "${SHADER_SOURCES}" SHADER_OUTPUT_DIR SHADER_SPIRV_BINARIES)

# ---------------------------------------------------------------------------
# Compile the SPIR-V corpus into the library, so a deployed binary needs
# nothing on disk beside it. This is what makes the library usable from a
# context with no install prefix at all - most concretely a JNI shared object
# extracted from a jar, which is how PhotonVision consumes native code. Turn
# it off to fall back to loading .spv files from SHADER_DIR at runtime; the
# selection happens at runtime via vk::HasEmbeddedShaders(), so both paths
# stay compiled and exercised.
# ---------------------------------------------------------------------------
option(VKAPRILTAG_EMBED_SHADERS "Compile the SPIR-V corpus into the library" ON)

set(EMBEDDED_SHADERS_CPP ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.cpp)
set(GENERATE_EMBEDDED_SHADERS_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GenerateEmbeddedShaders.cmake)

# '|'-joined: a ';'-separated list would be split into separate arguments on
# the way through -D.
string(REPLACE ";" "|" SHADER_SPIRV_BINARIES_ARG "${SHADER_SPIRV_BINARIES}")

add_custom_command(
  OUTPUT ${EMBEDDED_SHADERS_CPP}
  COMMAND ${CMAKE_COMMAND}
          "-DSPV_FILES=${SHADER_SPIRV_BINARIES_ARG}"
          "-DOUTPUT=${EMBEDDED_SHADERS_CPP}"
          "-DEMBED=${VKAPRILTAG_EMBED_SHADERS}"
          -P ${GENERATE_EMBEDDED_SHADERS_SCRIPT}
  DEPENDS ${SHADER_SPIRV_BINARIES} ${GENERATE_EMBEDDED_SHADERS_SCRIPT}
  COMMENT "Embedding SPIR-V corpus into libvkapriltag"
  VERBATIM
)

add_library(vkapriltag STATIC
  ${EMBEDDED_SHADERS_CPP}
  src/vk/Context.cpp
  src/vk/Buffer.cpp
  src/vk/Shader.cpp
  src/vk/ComputePipeline.cpp
  src/vk/PipelineCache.cpp
  src/vk/QueryPool.cpp
  src/gpu/GpuDetector.cpp
  src/gpu/QuadDecode.cpp
  src/apriltag_family.cpp
  src/TagDecoder.cpp
  src/common/pgm_io.cpp
  src/common/WorkerPool.cpp
)
add_library(vkapriltag::vkapriltag ALIAS vkapriltag)

add_dependencies(vkapriltag vkapriltag_shaders)

target_include_directories(vkapriltag PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# The shader SPIR-V lives next to the build tree during development, and next
# to the installed library (under share/vkapriltag/shaders) once installed.
target_compile_definitions(vkapriltag PUBLIC
  "SHADER_DIR=\"$<BUILD_INTERFACE:${SHADER_OUTPUT_DIR}>$<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_DATADIR}/vkapriltag/shaders>\""
)

target_link_libraries(vkapriltag PUBLIC
  Vulkan::Vulkan
  Threads::Threads
  apriltag
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vkapriltag PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

set_target_properties(vkapriltag PROPERTIES OUTPUT_NAME vkapriltag)

# ---------------------------------------------------------------------------
# Install rules: library, public headers, compiled shaders, and a CMake
# package config so downstream projects can find_package(vkapriltag).
# ---------------------------------------------------------------------------
install(TARGETS vkapriltag
  EXPORT vkapriltagTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/vkapriltag DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${SHADER_SPIRV_BINARIES}
  DESTINATION ${CMAKE_INSTALL_DATADIR}/vkapriltag/shaders
)

install(EXPORT vkapriltagTargets
  FILE vkapriltagTargets.cmake
  NAMESPACE vkapriltag::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkapriltag
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vkapriltagConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/vkapriltagConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkapriltag
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/vkapriltagConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/vkapriltagConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/vkapriltagConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkapriltag
)
