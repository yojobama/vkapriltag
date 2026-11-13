# CPU-only unit tests (no Vulkan device) and the golden-detection test (runs vk_eval on a committed
# corpus subset and compares the dumps with the committed goldens).
find_package(Python3 COMPONENTS Interpreter)

add_executable(unit_tests unit_tests.cpp)
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/library/src)
target_link_libraries(unit_tests PRIVATE vkapriltag::vkapriltag)

# The differential test of the decode against the patched upstream library.
if(TARGET apriltag)
  add_executable(own_decode_test own_decode_test.cpp)
  target_include_directories(own_decode_test PRIVATE ${PROJECT_SOURCE_DIR}/library/src ${PROJECT_SOURCE_DIR}/tools/vk_eval ${PROJECT_SOURCE_DIR}/tools)
  target_link_libraries(own_decode_test PRIVATE vkapriltag::vkapriltag apriltag)
  add_test(NAME own_decode_random COMMAND own_decode_test random)
  add_test(NAME own_decode_reconcile COMMAND own_decode_test reconcile)
  add_test(NAME families_match_upstream COMMAND own_decode_test families)
  add_test(NAME own_decode_corpus
    COMMAND own_decode_test corpus ${CMAKE_CURRENT_SOURCE_DIR}/corpus ${CMAKE_CURRENT_SOURCE_DIR}/own_decode_quads.txt)
  add_test(NAME own_decode_comparison_can_fail COMMAND own_decode_test random "" "" tag16h5)
  set_tests_properties(own_decode_comparison_can_fail PROPERTIES WILL_FAIL TRUE ENVIRONMENT OWN_DECODE_TEST_SABOTAGE=1)
  set_tests_properties(own_decode_random own_decode_reconcile own_decode_corpus PROPERTIES TIMEOUT 600)
  # Windows has no rpath: put the apriltag DLL beside the executable.
  if(WIN32)
    add_custom_command(TARGET own_decode_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:apriltag> $<TARGET_FILE_DIR:own_decode_test>)
  endif()
endif()

foreach(group decode geometry geom edges quadfit options steps pool)
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

if(Python3_FOUND)
  add_test(NAME deps_lint
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/deps_lint.py --root ${PROJECT_SOURCE_DIR})
endif()
