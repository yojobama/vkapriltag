@echo off
REM Vulkan AprilTag Build Script for Windows

echo Building Vulkan AprilTag...

REM Check for Vulkan SDK
if not defined VULKAN_SDK (
    echo ERROR: VULKAN_SDK environment variable not set.
    echo Please install the Vulkan SDK and restart your terminal.
    exit /b 1
)

echo Using Vulkan SDK: %VULKAN_SDK%

REM Check for glslangValidator
where glslangValidator >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: glslangValidator not found in PATH.
    echo Shader compilation will be skipped.
    echo Add %VULKAN_SDK%\Bin to your PATH to enable shader compilation.
) else (
    echo Found glslangValidator, compiling shaders...
    
    REM Create output directory
    if not exist "spirv" mkdir spirv
    
    REM Compile shaders
    for %%f in (vulkan_apriltag\shaders\*.comp) do (
        echo Compiling %%f...
        glslangValidator -V %%f -o spirv\%%~nf.spv
        if %ERRORLEVEL% neq 0 (
            echo ERROR: Failed to compile %%f
            exit /b 1
        )
    )
    
    echo Shader compilation complete.
)

REM Configure CMake
echo Configuring CMake...
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_VULKAN_APRILTAG=ON

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed
    exit /b 1
)

REM Build
echo Building...
cmake --build build --config Release

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)

echo Build complete!
echo.
echo Executables:
echo   build\Release\apriltag_demo.exe     - Basic AprilTag demo (CPU)
echo   build\Release\opencv_demo.exe       - OpenCV demo (CPU)
if exist "build\Release\vulkan_demo.exe" (
    echo   build\Release\vulkan_demo.exe       - Vulkan demo (GPU accelerated)
)

pause