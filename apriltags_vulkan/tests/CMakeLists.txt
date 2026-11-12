# CPU-only unit tests (no Vulkan device) and the golden-detection test (runs vk_eval on a committed
# corpus subset and compares the dumps with the committed goldens).
find_package(Python3 COMPONENTS Interpreter)

add_executable(unit_tests unit_tests.cpp)
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/library/src)
target_link_libraries(unit_tests PRIVATE vkapriltag::vkapriltag)

foreach(group decode geometry geom edges quadfit options steps)
  add_test(NAME unit_${group} COMMAND unit_tests ${group})
endforeach()

if(TARGET vk_eval AND Python3_FOUND)
  add_test(NAME golden_detections
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/golden_test.py
            --vk-eval $<TARGET_FILE:vk_eval>
            --corpus ${CMAKE_CURRENT_SOURCE_DIR}/corpus
            --golden ${CMAKE_CURRENT_SOURCE_DIR}/golden)
  set_tests_properties(golden_detections PROPERTIES TIMEOUT 600)
  add_test(NAME options_lint
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/options_lint.py
            --root ${PROJECT_SOURCE_DIR} --vk-eval $<TARGET_FILE:vk_eval>)
else()
  message(STATUS "golden_detections test skipped: needs the vk_eval tool and Python 3")
endif()

# Windows has no rpath: put the apriltag DLL beside the executables.
if(WIN32 AND TARGET apriltag)
  add_custom_command(TARGET unit_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:apriltag> $<TARGET_FILE_DIR:unit_tests>)
endif()
