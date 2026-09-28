# Applies each patch in -DPATCHES="a,b,c" (comma-separated) in order, skipping any already applied.
# Invoked via cmake -P.
string(REPLACE "," ";" PATCHES "${PATCHES}")
foreach(patch ${PATCHES})
  # Already applied if the reverse patch checks cleanly.
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
    # Apply the patch, capturing diagnostics.
    execute_process(
      COMMAND git apply "${patch}"
      RESULT_VARIABLE apply_result
      OUTPUT_VARIABLE _git_apply_out
      ERROR_VARIABLE _git_apply_err
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT apply_result EQUAL 0)
      # Fallback: apply with rejects and whitespace fixes.
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
