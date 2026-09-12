# Static-image validation tool: compares the vkapriltag pipeline's decoded
# tag IDs against the official (unmodified) apriltag_detector_detect()
# reference.
find_package(OpenCV QUIET)

if (OpenCV_FOUND)
  message(STATUS "Building OpenCV Compatible validation tool")
  add_executable(apriltag_vulkan_validate
          validate_against_libapriltag/validate_against_libapriltag_opencv.cpp
  )

  target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::vkapriltag ${OpenCV_LIBS})

  # Phase 0 corpus generator: rescales a tracked source image into a spread
  # of resolutions so the geometric-prefilter thresholds (item 2) can be
  # swept against varying effective tag pixel sizes. Only needs OpenCV, not
  # the vkapriltag library itself.
  add_executable(make_corpus make_corpus/make_corpus.cpp)
  target_link_libraries(make_corpus PRIVATE ${OpenCV_LIBS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(make_corpus PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  install(TARGETS make_corpus RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  # End-to-end pose validation: independently runs the stock libapriltag
  # pipeline (apriltag_detector_detect + estimate_tag_pose) and the full
  # vkapriltag pipeline (GpuDetector -> QuadDecode -> TagDecoder ->
  # PoseEstimator) on the same raw image, matches detections by tag ID, and
  # reports the 6-DoF pose delta between the two - unlike tools/validate_pose,
  # which feeds one shared H into both pose solvers. Needs OpenCV for
  # directory/image loading, same as apriltag_vulkan_validate above.
  add_executable(apriltag_pose_e2e_validate
          validate_pose_e2e/validate_pose_e2e.cpp
  )
  target_link_libraries(apriltag_pose_e2e_validate PRIVATE vkapriltag::vkapriltag ${OpenCV_LIBS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag_pose_e2e_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  install(TARGETS apriltag_pose_e2e_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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

# Pose verification: compares PoseEstimator against libapriltag's
# apriltag_pose.c at every stage of the algorithm, over a synthetic sweep with
# known ground truth, deliberately degenerate geometry, and (with --data)
# real detections. Needs no OpenCV - it loads PGMs via the library's own
# pgm_io, so it builds and runs identically on both dev machines.
add_executable(apriltag_pose_validate validate_pose/validate_pose.cpp)
target_link_libraries(apriltag_pose_validate PRIVATE vkapriltag::vkapriltag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_pose_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
install(TARGETS apriltag_pose_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
