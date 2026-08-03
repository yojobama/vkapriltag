# Compiles a list of GLSL compute shaders to SPIR-V using glslangValidator,
# targeting the Vulkan 1.1 environment.
#
# 1.1 rather than 1.2 on purpose: nothing in these shaders needs a 1.2
# feature, and targeting 1.1 keeps older mobile drivers (Mali/Adreno, older
# Panfrost) eligible. The shaders also avoid every optional device feature -
# no shaderFloat64, no shaderInt64, no 8-bit storage - so the resulting
# SPIR-V loads unmodified on desktop NVIDIA/AMD/Intel and on mobile parts
# alike. Workgroup sizes are specialization constants, chosen at runtime from
# the device's reported limits.
find_program(GLSLANG_VALIDATOR_EXECUTABLE
  NAMES glslangValidator
  HINTS Vulkan::glslangValidator
)

if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
  message(FATAL_ERROR "glslangValidator not found - required to compile GLSL compute shaders to SPIR-V")
endif()

function(compile_shaders TARGET_NAME SHADER_LIST OUT_DIR_VAR)
  # Optional 4th argument: name of a variable (set in the caller's scope) to
  # receive the full list of compiled .spv paths, e.g. for install(FILES ...).
  set(OUT_BINARIES_VAR "${ARGV3}")

  set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

  set(SPIRV_BINARIES)
  foreach(SHADER_SOURCE ${SHADER_LIST})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SPIRV_OUTPUT "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV_OUTPUT}
      COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE}
              --target-env vulkan1.1
              -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
              -o ${SPIRV_OUTPUT}
              ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
      COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
      VERBATIM
    )
    list(APPEND SPIRV_BINARIES ${SPIRV_OUTPUT})
  endforeach()

  add_custom_target(${TARGET_NAME} DEPENDS ${SPIRV_BINARIES})
  set(${OUT_DIR_VAR} "${SHADER_OUT_DIR}" PARENT_SCOPE)
  if(OUT_BINARIES_VAR)
    set(${OUT_BINARIES_VAR} "${SPIRV_BINARIES}" PARENT_SCOPE)
  endif()
endfunction()
