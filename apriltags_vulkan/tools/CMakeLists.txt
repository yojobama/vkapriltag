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
