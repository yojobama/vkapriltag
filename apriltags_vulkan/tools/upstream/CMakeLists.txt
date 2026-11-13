# Upstream AprilRobotics apriltag, fetched only for VKAPRILTAG_UPSTREAM_VALIDATION=ON: the reference validation
# tools link it. It is never part of the library, the tests or the install rules.
include(FetchContent)
enable_language(C)

set(APRILTAG_PATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patches)
set(APRILTAG_PATCHES "")
# The vector generator needs the decode steps exposed (patched sources); the other tools use stock upstream.
if(VKAPRILTAG_REGEN_VECTORS)
  list(APPEND APRILTAG_PATCHES ${APRILTAG_PATCH_DIR}/apriltag-expose-decode-steps.patch ${APRILTAG_PATCH_DIR}/apriltag-expose-code-probe.patch)
endif()
# Scopes the pthreads_cross emulation to "_WIN32 and not MinGW" so MinGW uses its real pthread.h.
if(MINGW)
  list(APPEND APRILTAG_PATCHES ${APRILTAG_PATCH_DIR}/apriltag-mingw-real-pthreads.patch)
endif()

set(APRILTAG_FETCH_ARGS GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git GIT_TAG v3.4.5)
if(APRILTAG_PATCHES)
  string(JOIN "," APRILTAG_PATCH_LIST ${APRILTAG_PATCHES})
  # Comma-separated, not semicolon-separated; ApplyPatches.cmake skips patches already present.
  list(APPEND APRILTAG_FETCH_ARGS PATCH_COMMAND ${CMAKE_COMMAND} "-DPATCHES=${APRILTAG_PATCH_LIST}" -P ${APRILTAG_PATCH_DIR}/ApplyPatches.cmake)
endif()
# Upstream's examples, Python wrapper and install rules are not wanted.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
  FetchContent_Declare(apriltag ${APRILTAG_FETCH_ARGS} EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(apriltag)
else()
  FetchContent_Declare(apriltag ${APRILTAG_FETCH_ARGS})
  FetchContent_GetProperties(apriltag)
  if(NOT apriltag_POPULATED)
    FetchContent_Populate(apriltag)
    add_subdirectory(${apriltag_SOURCE_DIR} ${apriltag_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

# Pin gnu17: newer GCC defaults turn the old C sources' incompatible pointer calls into errors.
if(TARGET apriltag)
  target_compile_options(apriltag PRIVATE -std=gnu17)
  # No fused multiply-adds, so upstream computes what OwnDecode.cpp (same flag) computes.
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(apriltag PRIVATE -ffp-contract=off)
  endif()
endif()
