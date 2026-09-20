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
  # Check whether the reverse of the patch applies cleanly (meaning the
  # patch is already present). Capture output for diagnostics.
  execute_process(
    COMMAND git apply --reverse --check "${patch}"
    RESULT_VARIABLE already_applied
    OUTPUT_VARIABLE _git_reverse_out
    ERROR_VARIABLE _git_reverse_err
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(already_applied EQUAL 0)
    message(STATUS "Patch already applied: ${patch}")
  else()
    # Try applying the patch and capture stdout/stderr for diagnostics.
    execute_process(
      COMMAND git apply "${patch}"
      RESULT_VARIABLE apply_result
      OUTPUT_VARIABLE _git_apply_out
      ERROR_VARIABLE _git_apply_err
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT apply_result EQUAL 0)
      # As a fallback, try to apply with rejects and whitespace fixes so the
      # user can inspect .rej files if hunks fail.
      execute_process(
        COMMAND git apply --reject --whitespace=fix "${patch}"
        RESULT_VARIABLE reject_result
        OUTPUT_VARIABLE _git_reject_out
        ERROR_VARIABLE _git_reject_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      message(STATUS "git apply output: ${_git_apply_out} ${_git_apply_err}")
      if(NOT reject_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply patch: ${patch}\n\ngit apply output:\n${_git_apply_out}\n${_git_apply_err}\n\ngit apply --reject output:\n${_git_reject_out}\n${_git_reject_err}")
      else()
        message(WARNING "Patch applied with --reject (some hunks may have been rejected): ${patch}")
      endif()
    else()
      message(STATUS "Applied patch: ${patch}")
    endif()
  endif()
endforeach()
