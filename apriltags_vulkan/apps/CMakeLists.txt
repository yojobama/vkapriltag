# Interactive sample app (V4L2 camera capture) built on top of the vkapriltag
# library.
add_executable(apriltag_vulkan
  apriltag_vulkan/main.cpp
  apriltag_vulkan/v4l2_capture.cpp
)
target_link_libraries(apriltag_vulkan PRIVATE vkapriltag::vkapriltag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(apriltag_vulkan PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

install(TARGETS apriltag_vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
