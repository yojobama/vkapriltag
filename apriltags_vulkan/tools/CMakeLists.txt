# The tools that compare against upstream libapriltag include upstream/UpstreamAdapter.h and link `apriltag`;
# they are built only when upstream is fetched (VKAPRILTAG_UPSTREAM_VALIDATION).
if(VKAPRILTAG_UPSTREAM_VALIDATION)
  add_subdirectory(upstream)
endif()
function(vkapriltag_use_upstream target)
  target_link_libraries(${target} PRIVATE apriltag)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # Windows has no rpath: put the apriltag DLL beside the executable.
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:apriltag> $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()

# Static-image validation tool: compares decoded tag IDs against apriltag_detector_detect().
find_package(OpenCV QUIET)

if (NOT TARGET apriltag)
  message(STATUS "Reference validation tools skipped: VKAPRILTAG_UPSTREAM_VALIDATION is OFF")
elseif (OpenCV_FOUND)
  message(STATUS "Building OpenCV Compatible validation tool")
  add_executable(apriltag_vulkan_validate
          validate_against_libapriltag/validate_against_libapriltag_opencv.cpp
  )

  target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::internal ${OpenCV_LIBS})
  vkapriltag_use_upstream(apriltag_vulkan_validate)
  # --renderdoc-capture needs RenderDoc's in-application API header; without it the option warns.
  find_path(RENDERDOC_APP_INCLUDE_DIR renderdoc_app.h
            PATHS "$ENV{ProgramFiles}/RenderDoc" /usr/include /usr/local/include /opt/renderdoc/include)
  if (RENDERDOC_APP_INCLUDE_DIR)
    target_include_directories(apriltag_vulkan_validate PRIVATE ${RENDERDOC_APP_INCLUDE_DIR})
    target_compile_definitions(apriltag_vulkan_validate PRIVATE VKAPRILTAG_HAVE_RENDERDOC)
    target_link_libraries(apriltag_vulkan_validate PRIVATE ${CMAKE_DL_LIBS})
  endif()

  # Corpus generator: rescales a source image to a spread of resolutions. Needs OpenCV only.
  add_executable(make_corpus make_corpus/make_corpus.cpp)
  target_link_libraries(make_corpus PRIVATE ${OpenCV_LIBS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(make_corpus PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  install(TARGETS make_corpus RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  # End-to-end pose validation: compares the 6-DoF pose of stock libapriltag and the full vkapriltag pipeline
  # on the same image, matched by tag ID. Needs OpenCV.
  add_executable(apriltag_pose_e2e_validate
          validate_pose_e2e/validate_pose_e2e.cpp
  )
  target_link_libraries(apriltag_pose_e2e_validate PRIVATE vkapriltag::internal ${OpenCV_LIBS})
  vkapriltag_use_upstream(apriltag_pose_e2e_validate)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_pose_e2e_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()

  # Renders a tag at a known 6-DoF pose and compares both pipelines against it. Needs OpenCV.
  add_executable(apriltag_synth_ground_truth
          synth_pose_ground_truth/synth_pose_ground_truth.cpp
  )
  target_link_libraries(apriltag_synth_ground_truth PRIVATE vkapriltag::internal ${OpenCV_LIBS})
  vkapriltag_use_upstream(apriltag_synth_ground_truth)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_synth_ground_truth PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
else()
  message(STATUS "Building validation tool without OpenCV (using pgm)")
  add_executable(apriltag_vulkan_validate
    validate_against_libapriltag/validate_against_libapriltag.cpp
  )
  target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::internal)
  vkapriltag_use_upstream(apriltag_vulkan_validate)
endif()

if(TARGET apriltag_vulkan_validate)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_vulkan_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# Throughput of concurrent detector streams on one GPU, each on a chosen queue family. Needs no OpenCV.
add_executable(bench_multiqueue bench_multiqueue/bench_multiqueue.cpp)
target_link_libraries(bench_multiqueue PRIVATE vkapriltag::internal)
install(TARGETS bench_multiqueue RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# What a pipeline barrier costs: K independent dispatches replayed with no barrier, an execution-only
# barrier and a memory barrier between them (see the header comment). Needs no OpenCV.
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/shaders/barrier_probe.comp.spv
  COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} --target-env vulkan1.1
          -o ${CMAKE_BINARY_DIR}/shaders/barrier_probe.comp.spv
          ${CMAKE_CURRENT_SOURCE_DIR}/bench_barrier/barrier_probe.comp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench_barrier/barrier_probe.comp
  COMMENT "Compiling barrier_probe.comp to SPIR-V"
  VERBATIM
)
add_executable(bench_barrier bench_barrier/bench_barrier.cpp
  ${CMAKE_BINARY_DIR}/shaders/barrier_probe.comp.spv)
target_compile_definitions(bench_barrier PRIVATE
  BARRIER_PROBE_SPV="${CMAKE_BINARY_DIR}/shaders/barrier_probe.comp.spv")
target_link_libraries(bench_barrier PRIVATE vkapriltag::internal)
install(TARGETS bench_barrier RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(UNIX)
  # Imports a dma-buf (dma-heap allocation or V4L2 capture buffer) as the detector's input frame.
  add_executable(dmabuf_import_test dmabuf_import_test/dmabuf_import_test.cpp)
  target_link_libraries(dmabuf_import_test PRIVATE vkapriltag::internal)
endif()

# Pose verification: compares PoseEstimator with libapriltag's apriltag_pose.c stage by stage over a synthetic sweep,
# degenerate geometry and (with --data) real detections. Needs no OpenCV.
if(TARGET apriltag)
  add_executable(apriltag_pose_validate validate_pose/validate_pose.cpp)
  target_link_libraries(apriltag_pose_validate PRIVATE vkapriltag::internal)
  vkapriltag_use_upstream(apriltag_pose_validate)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_pose_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# Records upstream's decode of a fixed set of quads as tests/own_decode_vectors.bin (needs the patched upstream).
if(VKAPRILTAG_REGEN_VECTORS)
  add_executable(gen_decode_vectors upstream/gen_decode_vectors.cpp)
  target_include_directories(gen_decode_vectors PRIVATE ${PROJECT_SOURCE_DIR}/library/src ${PROJECT_SOURCE_DIR}/tests ${PROJECT_SOURCE_DIR}/tools/vk_eval)
  target_link_libraries(gen_decode_vectors PRIVATE vkapriltag::internal)
  vkapriltag_use_upstream(gen_decode_vectors)
endif()

# Mali hardware-counter tools (Linux, Arm Mali): hwcnt_run prints every counter over a command's
# run, hwcnt_timeline samples selected counters periodically. Built against Arm's libGPUCounters.
option(VKAPRILTAG_BUILD_MALI_COUNTERS "Build the Mali hardware-counter tools (fetches libGPUCounters)" OFF)
if (VKAPRILTAG_BUILD_MALI_COUNTERS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(FetchContent)
  set(HWCPIPE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWCPIPE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    libgpucounters
    GIT_REPOSITORY https://github.com/ARM-software/libGPUCounters.git
    GIT_TAG 378c7b59e2dae87cbbffb0ac9cf96ed13c939823
  )
  FetchContent_MakeAvailable(libgpucounters)
  add_executable(hwcnt_run mali_counters/hwcnt_run.cpp)
  target_link_libraries(hwcnt_run PRIVATE hwcpipe)
  add_executable(hwcnt_timeline mali_counters/hwcnt_timeline.cpp)
  target_link_libraries(hwcnt_timeline PRIVATE hwcpipe)
endif()

# Runs a frame sequence through TrackingDetector with tracking off and on, interleaved, and reports times and paths.
add_executable(track_eval track_eval/track_eval.cpp)
target_link_libraries(track_eval PRIVATE vkapriltag::internal)
install(TARGETS track_eval RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Evaluation harness: runs a frame list through the GPU pipeline and CPU tail and writes the detection dump
# that napa_corpus/score_corpus.py scores. Needs no OpenCV.
add_executable(vk_eval vk_eval/vk_eval.cpp)
target_include_directories(vk_eval PRIVATE vk_eval)
target_link_libraries(vk_eval PRIVATE vkapriltag::internal)
# --ref (the unmodified libapriltag CPU detector) needs upstream.
if(TARGET apriltag)
  target_sources(vk_eval PRIVATE vk_eval/vk_eval_ref.cpp)
  target_compile_definitions(vk_eval PRIVATE VKAPRILTAG_VK_EVAL_REF=1)
  vkapriltag_use_upstream(vk_eval)
endif()
