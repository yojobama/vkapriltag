# Compiles GLSL compute shaders to SPIR-V with glslangValidator, targeting Vulkan 1.1.
# Workgroup sizes are specialisation constants.
find_program(GLSLANG_VALIDATOR_EXECUTABLE
  NAMES glslangValidator
  HINTS Vulkan::glslangValidator
)

if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
  message(FATAL_ERROR "glslangValidator not found - required to compile GLSL compute shaders to SPIR-V")
endif()

function(compile_shaders TARGET_NAME SHADER_LIST OUT_DIR_VAR)
  # Optional 4th argument: variable (in the caller's scope) receiving the list of compiled .spv paths.
  set(OUT_BINARIES_VAR "${ARGV3}")

  set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

  # Shared headers included via GL_GOOGLE_include_directive; globbed so edits to them recompile the .comp files.
  file(GLOB SHADER_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")

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
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE} ${SHADER_HEADERS}
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
