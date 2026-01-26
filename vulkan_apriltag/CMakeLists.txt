# Vulkan-accelerated AprilTag extension
set(COMPILE_WARNING_AS_ERROR OFF)
option(BUILD_VULKAN_APRILTAG "Build Vulkan-accelerated AprilTag detector" ON)

if(BUILD_VULKAN_APRILTAG)
    # Find Vulkan
    find_package(Vulkan QUIET)
    
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")
        
        # For now, use stub implementation due to compiler issues
        # TODO: Enable full Vulkan implementation once compiler issues are resolved
        set(USE_VULKAN_STUB OFF)
        
        if(USE_VULKAN_STUB)
            message(STATUS "Using Vulkan stub implementation (CPU fallback)")
            
            # Stub implementation
            set(VULKAN_APRILTAG_SOURCES
                vulkan_apriltag_stub.c
            )
            
            set(VULKAN_APRILTAG_HEADERS
                vulkan_apriltag.h
            )
            
            # Create Vulkan AprilTag library
            add_library(vulkan_apriltag ${VULKAN_APRILTAG_SOURCES} ${VULKAN_APRILTAG_HEADERS})
            
            target_link_libraries(vulkan_apriltag 
                PUBLIC apriltag
            )
            
        else()
            # Build full Vulkan implementation and embed real SPIR-V generated from shaders/*.comp
            set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
            set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders_spv)
            file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

            find_program(GLSLANG_VALIDATOR glslangValidator)
            find_program(GLSLC glslc)
            if(NOT GLSLANG_VALIDATOR AND NOT GLSLC)
                message(FATAL_ERROR "Neither glslangValidator nor glslc found; cannot build Vulkan shaders")
            endif()

            set(SHADER_NAMES threshold connected_components gradient decimate blur)
            set(GENERATED_SHADER_SOURCES)
            set(GENERATED_SHADER_HEADERS)
            foreach(SHADER_NAME IN LISTS SHADER_NAMES)
                set(SHADER_SRC ${SHADER_SOURCE_DIR}/${SHADER_NAME}.comp)
                set(SHADER_SPV ${SHADER_BINARY_DIR}/${SHADER_NAME}.spv)

                if(GLSLANG_VALIDATOR)
                    add_custom_command(
                        OUTPUT ${SHADER_SPV}
                        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC} -o ${SHADER_SPV}
                        DEPENDS ${SHADER_SRC}
                        COMMENT "Compiling ${SHADER_SRC} -> ${SHADER_SPV}"
                        VERBATIM)
                else()
                    add_custom_command(
                        OUTPUT ${SHADER_SPV}
                        COMMAND ${GLSLC} ${SHADER_SRC} -o ${SHADER_SPV}
                        DEPENDS ${SHADER_SRC}
                        COMMENT "Compiling ${SHADER_SRC} -> ${SHADER_SPV}"
                        VERBATIM)
                endif()

                set(SHADER_HEADER ${SHADER_BINARY_DIR}/${SHADER_NAME}_spirv.h)
                set(SHADER_SOURCE ${SHADER_BINARY_DIR}/${SHADER_NAME}_spirv.c)
                add_custom_command(
                    OUTPUT ${SHADER_HEADER} ${SHADER_SOURCE}
                    COMMAND ${CMAKE_COMMAND}
                        -DSPIRV_FILE=${SHADER_SPV}
                        -DHEADER_FILE=${SHADER_HEADER}
                        -DSHADER_NAME=${SHADER_NAME}
                        -P ${CMAKE_SOURCE_DIR}/CMake/spirv_to_header.cmake
                    DEPENDS ${SHADER_SPV} ${CMAKE_SOURCE_DIR}/CMake/spirv_to_header.cmake
                    COMMENT "Embedding ${SHADER_NAME}.spv into C source"
                    VERBATIM)

                list(APPEND GENERATED_SHADER_SOURCES ${SHADER_SOURCE})
                list(APPEND GENERATED_SHADER_HEADERS ${SHADER_HEADER})
            endforeach()

            add_custom_target(vulkan_shaders_embedded ALL
                DEPENDS ${GENERATED_SHADER_SOURCES} ${GENERATED_SHADER_HEADERS})
            set_source_files_properties(${GENERATED_SHADER_SOURCES} ${GENERATED_SHADER_HEADERS} PROPERTIES GENERATED TRUE)

            # Full Vulkan implementation
            set(VULKAN_APRILTAG_SOURCES
                vulkan_apriltag.c
                vulkan_utils.c
                vulkan_detector.c
                vulkan_pipeline.c
                shaders_spirv.c
            )
            
            set(VULKAN_APRILTAG_HEADERS
                vulkan_apriltag.h
                vulkan_utils.h
                shaders_spirv.h
            )
            
            # Create Vulkan AprilTag library
            add_library(vulkan_apriltag ${VULKAN_APRILTAG_SOURCES} ${VULKAN_APRILTAG_HEADERS})
            add_dependencies(vulkan_apriltag vulkan_shaders_embedded)

            target_link_libraries(vulkan_apriltag 
                PUBLIC apriltag
                PUBLIC Vulkan::Vulkan
            )
            
            # Define BUILD_VULKAN_FULL to properly handle Vulkan type declarations
            target_compile_definitions(vulkan_apriltag PRIVATE BUILD_VULKAN_FULL)
            target_compile_definitions(vulkan_apriltag PRIVATE VULKAN_SHADER_DIR="${SHADER_BINARY_DIR}")
            target_include_directories(vulkan_apriltag PRIVATE ${SHADER_BINARY_DIR})

            if(WIN32)
                target_compile_definitions(vulkan_apriltag PRIVATE VK_USE_PLATFORM_WIN32_KHR)
            endif()
        endif()
        
        target_include_directories(vulkan_apriltag 
            PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/apriltag>"
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
        set_target_properties(vulkan_apriltag PROPERTIES 
            SOVERSION 1 
            VERSION 1.0.0
            C_STANDARD 99
        )
        
        # Install Vulkan AprilTag
        install(TARGETS vulkan_apriltag EXPORT apriltagTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
        
        install(FILES ${VULKAN_APRILTAG_HEADERS} 
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/apriltag/vulkan_apriltag)
        install(DIRECTORY ${SHADER_BINARY_DIR}/ DESTINATION ${CMAKE_INSTALL_LIBDIR}/apriltag/shaders FILES_MATCHING PATTERN "*.spv")

        # Vulkan demo
        if(BUILD_EXAMPLES)
            set(_OpenCV_REQUIRED_COMPONENTS core imgproc videoio highgui)
            find_package(OpenCV COMPONENTS ${_OpenCV_REQUIRED_COMPONENTS} QUIET CONFIG)
            if(OpenCV_FOUND)
                enable_language(CXX)
                add_executable(vulkan_demo ../example/vulkan_demo.cpp)
                target_link_libraries(vulkan_demo vulkan_apriltag ${OpenCV_LIBRARIES})
                set_target_properties(vulkan_demo PROPERTIES CXX_STANDARD 11)
                install(TARGETS vulkan_demo RUNTIME DESTINATION bin)
            endif()
        endif()
        
    else()
        message(STATUS "Vulkan not found. Vulkan AprilTag will not be built.")
    endif()
endif()