# CPU-only unit tests (no Vulkan device) and the golden-detection test (runs vk_eval on a committed
# corpus subset and compares the dumps with the committed goldens).
find_package(Python3 COMPONENTS Interpreter)

add_executable(unit_tests unit_tests.cpp)
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/library/src)
target_link_libraries(unit_tests PRIVATE vkapriltag::vkapriltag)

# The differential test of the decode against the upstream results recorded in own_decode_vectors.bin
# (tools/upstream/gen_decode_vectors.cpp writes the file). It needs no upstream.
add_executable(own_decode_test own_decode_test.cpp)
target_include_directories(own_decode_test PRIVATE ${PROJECT_SOURCE_DIR}/library/src ${PROJECT_SOURCE_DIR}/tools/vk_eval)
target_link_libraries(own_decode_test PRIVATE vkapriltag::vkapriltag)
set(OWN_DECODE_VECTORS ${CMAKE_CURRENT_SOURCE_DIR}/own_decode_vectors.bin)
add_test(NAME own_decode_random COMMAND own_decode_test random ${OWN_DECODE_VECTORS})
add_test(NAME own_decode_reconcile COMMAND own_decode_test reconcile ${OWN_DECODE_VECTORS})
add_test(NAME families_match_upstream COMMAND own_decode_test families ${OWN_DECODE_VECTORS})
add_test(NAME own_decode_corpus COMMAND own_decode_test corpus ${OWN_DECODE_VECTORS} ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
add_test(NAME own_decode_comparison_can_fail COMMAND own_decode_test random ${OWN_DECODE_VECTORS} "" tag16h5)
set_tests_properties(own_decode_comparison_can_fail PROPERTIES WILL_FAIL TRUE ENVIRONMENT OWN_DECODE_TEST_SABOTAGE=1)
set_tests_properties(own_decode_random own_decode_reconcile own_decode_corpus PROPERTIES TIMEOUT 600)

# The recorded GPU work of the first frame, per device (tests/gpu_trace/<device>/<profile>.txt; devices without goldens skip).
add_executable(gpu_trace_test gpu_trace_test.cpp)
target_include_directories(gpu_trace_test PRIVATE ${PROJECT_SOURCE_DIR}/library/src ${PROJECT_SOURCE_DIR}/tools/vk_eval)
target_link_libraries(gpu_trace_test PRIVATE vkapriltag::vkapriltag)
function(add_gpu_trace name decimation)
  add_test(NAME gpu_trace_${name}
    COMMAND gpu_trace_test --image ${CMAKE_CURRENT_SOURCE_DIR}/corpus/one-tag/one-tag_640x480_000.png
            --golden-root ${CMAKE_CURRENT_SOURCE_DIR}/gpu_trace --profile ${name} --decimation ${decimation})
  set_tests_properties(gpu_trace_${name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  if(ARGN)
    set_tests_properties(gpu_trace_${name} PROPERTIES ENVIRONMENT "${ARGN}")
  endif()
endfunction()
add_gpu_trace(default_d2 2)
add_gpu_trace(default_d1 1)
add_gpu_trace(default_d4 4)
add_gpu_trace(no_subgroup_d2 2 "APRILTAG_VK_FORCE_NO_SUBGROUP=1")
add_gpu_trace(no_int64_d2 2 "APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1")
add_gpu_trace(no_subgroup_no_int64_d2 2 "APRILTAG_VK_FORCE_NO_SUBGROUP=1;APRILTAG_VK_FORCE_NO_INT64_ATOMIC=1")
add_gpu_trace(no_8bit_d2 2 "APRILTAG_VK_FORCE_NO_8BIT=1")
add_gpu_trace(integrated_like_d2 2 "APRILTAG_VK_COUNT_RUNS=1;APRILTAG_VK_SPARSE=1;APRILTAG_VK_MERGE_X4=1;APRILTAG_VK_LEAN_TAIL=3;APRILTAG_VK_MERGE_BAND=0")
add_gpu_trace(discrete_like_d2 2 "APRILTAG_VK_COUNT_RUNS=0;APRILTAG_VK_SPARSE=0;APRILTAG_VK_MERGE_X4=0;APRILTAG_VK_LEAN_TAIL=0;APRILTAG_VK_MERGE_BAND=8")
add_gpu_trace(unfused_d2 2 "APRILTAG_VK_FUSE_SUBMITS=0")
add_gpu_trace(unfused_d1 1 "APRILTAG_VK_FUSE_SUBMITS=0")
add_gpu_trace(unfused_no_8bit_d2 2 "APRILTAG_VK_FUSE_SUBMITS=0;APRILTAG_VK_FORCE_NO_8BIT=1")
add_gpu_trace(box_decimate_off_d2 2 "APRILTAG_VK_BOX_DECIMATE=0")
add_gpu_trace(adaptive_floor_off_d2 2 "APRILTAG_VK_ADAPTIVE_FLOOR=0")
add_gpu_trace(fused_threshold_off_d2 2 "APRILTAG_VK_FUSED_THRESHOLD=0")
add_gpu_trace(uf_init_scan_off_d2 2 "APRILTAG_VK_UF_INIT_SCAN=0")
add_gpu_trace(prerecord_off_d2 2 "APRILTAG_VK_PRERECORD=0")

foreach(group decode geometry geom edges quadfit options steps pool)
  add_test(NAME unit_${group} COMMAND unit_tests ${group})
endforeach()
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/plan)
add_test(NAME unit_plan COMMAND unit_tests plan ${CMAKE_CURRENT_SOURCE_DIR}/plan)

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

# Installs the build to a temporary prefix, checks it holds nothing from upstream libapriltag, and builds and runs
# tests/consumer against it with find_package(vkapriltag).
if(Python3_FOUND AND TARGET vk_eval)
  add_test(NAME installed_package_consumer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/consumer_test.py
            --cmake ${CMAKE_COMMAND} --build-dir ${CMAKE_BINARY_DIR} --config $<CONFIG>
            --generator ${CMAKE_GENERATOR} --make-program ${CMAKE_MAKE_PROGRAM} --cxx ${CMAKE_CXX_COMPILER}
            --consumer ${CMAKE_CURRENT_SOURCE_DIR}/consumer --stb-dir ${PROJECT_SOURCE_DIR}/tools/vk_eval
            --image ${CMAKE_CURRENT_SOURCE_DIR}/corpus/one-tag/one-tag_640x480_000.png --expect-id 316)
  set_tests_properties(installed_package_consumer PROPERTIES TIMEOUT 600)
endif()
