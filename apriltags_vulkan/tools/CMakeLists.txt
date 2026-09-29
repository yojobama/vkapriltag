# Static-image validation tool: compares decoded tag IDs against apriltag_detector_detect().
find_package(OpenCV QUIET)

if (OpenCV_FOUND)
  message(STATUS "Building OpenCV Compatible validation tool")
  add_executable(apriltag_vulkan_validate
          validate_against_libapriltag/validate_against_libapriltag_opencv.cpp
  )

  target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::vkapriltag ${OpenCV_LIBS})

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
  target_link_libraries(apriltag_pose_e2e_validate PRIVATE vkapriltag::vkapriltag ${OpenCV_LIBS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_pose_e2e_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  install(TARGETS apriltag_pose_e2e_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  # Renders a tag at a known 6-DoF pose and compares both pipelines against it. Needs OpenCV.
  add_executable(apriltag_synth_ground_truth
          synth_pose_ground_truth/synth_pose_ground_truth.cpp
  )
  target_link_libraries(apriltag_synth_ground_truth PRIVATE vkapriltag::vkapriltag ${OpenCV_LIBS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_synth_ground_truth PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  install(TARGETS apriltag_synth_ground_truth RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
  message(STATUS "Building validation tool without OpenCV (using pgm)")
  add_executable(apriltag_vulkan_validate
    validate_against_libapriltag/validate_against_libapriltag.cpp
  )
  target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::vkapriltag)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

install(TARGETS apriltag_vulkan_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Throughput of concurrent detector streams on one GPU, each on a chosen queue family. Needs no OpenCV.
add_executable(bench_multiqueue bench_multiqueue/bench_multiqueue.cpp)
target_link_libraries(bench_multiqueue PRIVATE vkapriltag::vkapriltag)
install(TARGETS bench_multiqueue RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(UNIX)
  # Imports a dma-buf (dma-heap allocation or V4L2 capture buffer) as the detector's input frame.
  add_executable(dmabuf_import_test dmabuf_import_test/dmabuf_import_test.cpp)
  target_link_libraries(dmabuf_import_test PRIVATE vkapriltag::vkapriltag)
endif()

# Pose verification: compares PoseEstimator with libapriltag's apriltag_pose.c stage by stage over a synthetic sweep,
# degenerate geometry and (with --data) real detections. Needs no OpenCV.
add_executable(apriltag_pose_validate validate_pose/validate_pose.cpp)
target_link_libraries(apriltag_pose_validate PRIVATE vkapriltag::vkapriltag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_pose_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
install(TARGETS apriltag_pose_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
