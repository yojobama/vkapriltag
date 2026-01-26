# Convert SPIR-V binary to C header file

if(NOT SPIRV_FILE OR NOT HEADER_FILE OR NOT SHADER_NAME)
    message(FATAL_ERROR "Required variables not set: SPIRV_FILE=${SPIRV_FILE}, HEADER_FILE=${HEADER_FILE}, SHADER_NAME=${SHADER_NAME}")
endif()

# Check if input file exists
if(NOT EXISTS ${SPIRV_FILE})
    message(FATAL_ERROR "SPIR-V file does not exist: ${SPIRV_FILE}")
endif()

# Read the SPIR-V file as binary
file(READ ${SPIRV_FILE} SPIRV_DATA HEX)

# Calculate file size
file(SIZE ${SPIRV_FILE} SPIRV_SIZE)

# Convert hex string to proper array format
string(LENGTH ${SPIRV_DATA} HEX_LENGTH)
math(EXPR WORD_COUNT "${HEX_LENGTH} / 8")

set(SPIRV_ARRAY "")
set(BYTE_COUNT 0)

while(BYTE_COUNT LESS HEX_LENGTH)
    # Extract 8 hex characters (4 bytes) = 1 SPIR-V word
    string(SUBSTRING ${SPIRV_DATA} ${BYTE_COUNT} 8 HEX_WORD)
    
    # Convert to little-endian uint32_t
    string(SUBSTRING ${HEX_WORD} 6 2 BYTE0)
    string(SUBSTRING ${HEX_WORD} 4 2 BYTE1)
    string(SUBSTRING ${HEX_WORD} 2 2 BYTE2)
    string(SUBSTRING ${HEX_WORD} 0 2 BYTE3)
    
    set(WORD "0x${BYTE0}${BYTE1}${BYTE2}${BYTE3}")
    
    if(NOT SPIRV_ARRAY STREQUAL "")
        set(SPIRV_ARRAY "${SPIRV_ARRAY},\n    ")
    else()
        set(SPIRV_ARRAY "    ")
    endif()
    
    set(SPIRV_ARRAY "${SPIRV_ARRAY}${WORD}")
    
    math(EXPR BYTE_COUNT "${BYTE_COUNT} + 8")
endwhile()

# Create header content
set(HEADER_CONTENT 
"// Auto-generated SPIR-V header for ${SHADER_NAME}
// Generated from: ${SPIRV_FILE}
// Size: ${SPIRV_SIZE} bytes (${WORD_COUNT} words)

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern \"C\" {
#endif

extern const uint32_t ${SHADER_NAME}_comp_spirv[];
extern const size_t ${SHADER_NAME}_comp_spirv_size;

#ifdef __cplusplus
}
#endif
")

# Create source content
set(SOURCE_CONTENT
"// Auto-generated SPIR-V source for ${SHADER_NAME}
#include \"${SHADER_NAME}_spirv.h\"

const uint32_t ${SHADER_NAME}_comp_spirv[] = {
${SPIRV_ARRAY}
};

const size_t ${SHADER_NAME}_comp_spirv_size = sizeof(${SHADER_NAME}_comp_spirv);
")

# Write header file
file(WRITE ${HEADER_FILE} ${HEADER_CONTENT})

# Write source file
string(REGEX REPLACE "\\.h$" ".c" SOURCE_FILE ${HEADER_FILE})
file(WRITE ${SOURCE_FILE} ${SOURCE_CONTENT})

message(STATUS "Generated SPIR-V header: ${HEADER_FILE}")
message(STATUS "Generated SPIR-V source: ${SOURCE_FILE}")
message(STATUS "Shader ${SHADER_NAME}: ${WORD_COUNT} words (${SPIRV_SIZE} bytes)")