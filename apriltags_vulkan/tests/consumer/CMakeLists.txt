# A project that uses an installed vkapriltag through find_package (tests/consumer_test.py builds and runs it).
cmake_minimum_required(VERSION 3.16)
project(vkapriltag_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vkapriltag CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_include_directories(consumer PRIVATE ${STB_DIR})
target_link_libraries(consumer PRIVATE vkapriltag::vkapriltag)
