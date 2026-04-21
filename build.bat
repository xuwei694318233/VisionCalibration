@echo off
REM VisionCalibration Build Script for Visual Studio 2022

setlocal EnableDelayedExpansion

if "%1"=="help" goto :help
if "%1"=="clean" goto :clean
if "%1"=="debug" goto :build_debug
if "%1"=="release" goto :build_release
if "%1"=="" goto :build_all
goto :help

:help
echo ========================================
echo VisionCalibration Build Script
echo ========================================
echo.
echo Usage:
echo   build.bat [option]
echo.
echo Options:
echo   clean     - Clean build directory
echo   debug     - Build Debug only
echo   release   - Build Release only
echo   all       - Build Debug and Release (default)
echo   help      - Show this help
echo.
echo Examples:
echo   build.bat           - Build all
echo   build.bat debug     - Build Debug
echo   build.bat release   - Build Release
echo.
exit /b 0

:clean
echo ========================================
echo Cleaning build directory
echo ========================================
if exist "build" (
    rmdir /s /q build
    echo Build directory cleaned!
) else (
    echo Build directory does not exist
)
echo.
exit /b 0

:check_cmake
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake not found in PATH
    echo Please install CMake and add it to your PATH
    exit /b 1
)
goto :eof

:setup_build
call :check_cmake
set BUILD_DIR=build
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"
goto :eof

:build_release
call :setup_build
echo.
echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% NEQ 0 (echo Error: CMake failed & cd .. & exit /b 1)
echo.
echo Building Release...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (echo Error: Build failed & cd .. & exit /b 1)
cd ..
echo Release build completed!
echo Output: build\bin\Release\
exit /b 0

:build_debug
call :setup_build
echo.
echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% NEQ 0 (echo Error: CMake failed & cd .. & exit /b 1)
echo.
echo Building Debug...
cmake --build . --config Debug
if %ERRORLEVEL% NEQ 0 (echo Error: Build failed & cd .. & exit /b 1)
cd ..
echo Debug build completed!
echo Output: build\bin\Debug\
exit /b 0

:build_all
call :setup_build
echo.
echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% NEQ 0 (echo Error: CMake failed & cd .. & exit /b 1)
echo.
echo Building Release...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (echo Error: Build failed & cd .. & exit /b 1)
echo.
echo Building Debug...
cmake --build . --config Debug
if %ERRORLEVEL% NEQ 0 (echo Error: Build failed & cd .. & exit /b 1)
cd ..
echo.
echo ========================================
echo Build completed!
echo ========================================
echo Release: build\bin\Release\
echo Debug:   build\bin\Debug\
echo.
echo Run test: build\bin\Debug\test_camera_calibrator.exe
exit /b 0