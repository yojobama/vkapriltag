# Static-image validation tool: compares the vkapriltag pipeline's decoded
# tag IDs against the official (unmodified) apriltag_detector_detect()
# reference.
add_executable(apriltag_vulkan_validate
  validate_against_libapriltag/validate_against_libapriltag.cpp
)
target_link_libraries(apriltag_vulkan_validate PRIVATE vkapriltag::vkapriltag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan_validate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

install(TARGETS apriltag_vulkan_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
