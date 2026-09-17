# Applies each patch file passed as a -DPATCHES="a,b,c" argument (comma-,
# NOT semicolon-separated: PATCH_COMMAND is itself stored as a CMake list,
# i.e. semicolon-joined, by the FetchContent/ExternalProject machinery that
# regenerates it into a subbuild CMakeLists.txt, so an embedded, escaped
# `\;` does not reliably survive that round-trip and silently splits back
# into two separate command-line arguments instead - comma sidesteps the
# whole problem), in order, each idempotently: a patch already applied
# (e.g. a stale reconfigure that re-runs FetchContent's PATCH_COMMAND
# against an already-patched checkout) is silently skipped instead of
# erroring `git apply` out.
#
# Invoked via `${CMAKE_COMMAND} -P ApplyPatches.cmake` rather than a chained
# shell command line, so the idempotency logic doesn't depend on the
# PATCH_COMMAND's shell (cmd.exe vs sh) agreeing on &&/|| grouping.
string(REPLACE "," ";" PATCHES "${PATCHES}")
foreach(patch ${PATCHES})
  execute_process(
    COMMAND git apply --reverse --check "${patch}"
    RESULT_VARIABLE already_applied
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT already_applied EQUAL 0)
    execute_process(
      COMMAND git apply "${patch}"
      RESULT_VARIABLE apply_result
    )
    if(NOT apply_result EQUAL 0)
      message(FATAL_ERROR "Failed to apply patch: ${patch}")
    endif()
  endif()
endforeach()
