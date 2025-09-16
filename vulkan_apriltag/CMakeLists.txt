# Vulkan-accelerated AprilTag extension
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
            
            target_link_libraries(vulkan_apriltag 
                PUBLIC apriltag
                PUBLIC Vulkan::Vulkan
            )
            
            # Define BUILD_VULKAN_FULL to properly handle Vulkan type declarations
            target_compile_definitions(vulkan_apriltag PRIVATE BUILD_VULKAN_FULL)
            
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