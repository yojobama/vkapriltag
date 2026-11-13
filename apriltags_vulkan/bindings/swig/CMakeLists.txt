# SWIG bindings of the public Detector. Python is built and tested here; Java and C# can be generated from the same
# interface file with the SWIG command line (see README.md).
find_package(SWIG 4.1 REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
include(UseSWIG)

set_source_files_properties(vkapriltag.i PROPERTIES
  CPLUSPLUS ON
  SWIG_MODULE_NAME vkapriltag
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/library/include
  SWIG_FLAGS "-threads")
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/python)
set(SWIG_OUTFILE_DIR ${CMAKE_CURRENT_BINARY_DIR}/python)

swig_add_library(vkapriltag_python TYPE MODULE LANGUAGE python SOURCES vkapriltag.i)
target_link_libraries(vkapriltag_python PRIVATE vkapriltag::vkapriltag Python3::Module)
set_target_properties(vkapriltag_python PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python)

add_test(NAME swig_python
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/swig_test.py
          ${CMAKE_CURRENT_BINARY_DIR}/python ${PROJECT_SOURCE_DIR}/tests/corpus)
set_tests_properties(swig_python PROPERTIES TIMEOUT 120)
