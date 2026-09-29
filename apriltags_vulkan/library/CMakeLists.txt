# ---------------------------------------------------------------------------
# vkapriltag: Vulkan compute detection pipeline + CPU QuadDecode/TagDecoder tail, as an installable library
# (target vkapriltag::vkapriltag, or vkapriltag in-tree).
# ---------------------------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Upstream AprilRobotics apriltag, patched to expose quad_decode_index() and reconcile_detections()
# so the GPU pipeline can feed its own quads to the CPU decode.
# ---------------------------------------------------------------------------
include(FetchContent)
set(APRILTAG_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/patches/apriltag-expose-decode-steps.patch)
# Patch scoping the pthreads_cross emulation to "_WIN32 and not MinGW" so MinGW uses its real pthread.h.
set(APRILTAG_MINGW_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/patches/apriltag-mingw-real-pthreads.patch)
FetchContent_Declare(
  apriltag
  GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git
  GIT_TAG v3.4.5
  # Applied via ApplyPatches.cmake, which skips patches already present.
  # Comma-separated, not semicolon-separated.
  PATCH_COMMAND ${CMAKE_COMMAND}
        "-DPATCHES=${APRILTAG_PATCH_FILE},${APRILTAG_MINGW_PATCH_FILE}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/patches/ApplyPatches.cmake
)
# Disable upstream's example binaries so they are not installed.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(apriltag)

# Pin gnu17: newer GCC defaults turn the old C sources' incompatible pointer calls into errors.
if(TARGET apriltag)
  target_compile_options(apriltag PRIVATE -std=gnu17)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompileShaders.cmake)

set(SHADER_SOURCES
  shaders/decimate.comp
  shaders/decimate_u8.comp
  shaders/decimate_yuyv.comp
  shaders/decimate_yuyv_u8.comp
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
  shaders/uf_final_u8.comp
  shaders/label_pixels.comp
  shaders/label_pixels_u8.comp
  shaders/blob_diff.comp
  shaders/scan_block.comp
  shaders/scan_add_offsets.comp
  shaders/init_extents.comp
  shaders/merge_extents.comp
  shaders/select_blobs.comp
  shaders/extract_blob_counts.comp
  shaders/sort_points_local.comp
  shaders/sort_points_local_u8.comp
  shaders/hash_group.comp
  shaders/build_indirect_args.comp
  shaders/reduce_extents_hash.comp
  shaders/reduce_extents_hash_atomic64.comp
  shaders/reduce_extents_hash_subgroup.comp
  shaders/scatter_index_points.comp
)

compile_shaders(vkapriltag_shaders "${SHADER_SOURCES}" SHADER_OUTPUT_DIR SHADER_SPIRV_BINARIES)

# ---------------------------------------------------------------------------
# Compile the SPIR-V corpus into the library; OFF loads .spv files from SHADER_DIR at runtime
# (selected by vk::HasEmbeddedShaders()).
# ---------------------------------------------------------------------------
option(VKAPRILTAG_EMBED_SHADERS "Compile the SPIR-V corpus into the library" ON)

set(EMBEDDED_SHADERS_CPP ${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders.cpp)
set(GENERATE_EMBEDDED_SHADERS_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GenerateEmbeddedShaders.cmake)

# '|'-joined so -D does not split the list.
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
  src/RefineEdges.cpp
  src/FramePipeline.cpp
  src/PoseEstimator.cpp
  src/common/pgm_io.cpp
  src/common/WorkerPool.cpp
)
add_library(vkapriltag::vkapriltag ALIAS vkapriltag)

add_dependencies(vkapriltag vkapriltag_shaders)

target_include_directories(vkapriltag PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# SPIR-V location: the build tree in development, share/vkapriltag/shaders once installed.
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
# Install rules: library, public headers, compiled shaders and the CMake package config.
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
