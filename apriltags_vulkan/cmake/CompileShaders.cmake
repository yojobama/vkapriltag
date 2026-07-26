# Compiles a list of GLSL compute shaders to SPIR-V using glslangValidator,
# targeting the Vulkan 1.2 environment.  We deliberately avoid any
# vendor-specific extensions in the shaders themselves so the resulting
# SPIR-V runs unmodified on any Vulkan 1.2 conformant driver (AMD/RADV,
# NVIDIA, Intel, etc).
find_program(GLSLANG_VALIDATOR_EXECUTABLE
  NAMES glslangValidator
  HINTS Vulkan::glslangValidator
)

if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
  message(FATAL_ERROR "glslangValidator not found - required to compile GLSL compute shaders to SPIR-V")
endif()

function(compile_shaders TARGET_NAME SHADER_LIST OUT_DIR_VAR)
  set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

  set(SPIRV_BINARIES)
  foreach(SHADER_SOURCE ${SHADER_LIST})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SPIRV_OUTPUT "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV_OUTPUT}
      COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE}
              --target-env vulkan1.2
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
endfunction()
