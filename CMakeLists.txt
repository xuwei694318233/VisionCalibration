cmake_minimum_required(VERSION 3.16)
project(VisionCalibration VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(MSVC)
    add_compile_options(/W3 /utf-8)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==================== nlohmann/json 库配置 (本地) ====================

set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json" CACHE PATH "nlohmann/json include directory")

if(NOT EXISTS "${JSON_INCLUDE_DIR}/json.hpp")
    message(FATAL_ERROR "json.hpp not found in ${JSON_INCLUDE_DIR}\nPlease ensure thirdparty/json/json.hpp exists")
endif()

message(STATUS "nlohmann/json include: ${JSON_INCLUDE_DIR}")

# 为nlohmann_json创建接口库（仅头文件库）
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${JSON_INCLUDE_DIR})

# ==================== OpenCV 配置 ====================

set(OPENCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv" CACHE PATH "OpenCV安装目录")

find_path(OPENCV_INCLUDE_DIR
    NAMES opencv2/opencv.hpp
    PATHS
        "${OPENCV_DIR}/include"
        "${OPENCV_DIR}"
    NO_DEFAULT_PATH
)

if(NOT OPENCV_INCLUDE_DIR)
    message(FATAL_ERROR "OpenCV include directory not found in ${OPENCV_DIR}")
endif()

message(STATUS "OpenCV include: ${OPENCV_INCLUDE_DIR}")

# 查找库文件 - 支持多配置生成器
find_library(OPENCV_LIBRARY_RELEASE
    NAMES opencv_world460
    PATHS "${OPENCV_DIR}/bin/Release"
    NO_DEFAULT_PATH
)

find_library(OPENCV_LIBRARY_DEBUG
    NAMES opencv_world460d
    PATHS "${OPENCV_DIR}/bin/Debug"
    NO_DEFAULT_PATH
)

# 设置库文件
if(OPENCV_LIBRARY_RELEASE AND OPENCV_LIBRARY_DEBUG)
    set(OPENCV_LIBRARIES
        optimized ${OPENCV_LIBRARY_RELEASE}
        debug ${OPENCV_LIBRARY_DEBUG}
    )
    message(STATUS "OpenCV library (Release): ${OPENCV_LIBRARY_RELEASE}")
    message(STATUS "OpenCV library (Debug): ${OPENCV_LIBRARY_DEBUG}")
elseif(OPENCV_LIBRARY_RELEASE)
    set(OPENCV_LIBRARIES ${OPENCV_LIBRARY_RELEASE})
    message(STATUS "OpenCV library: ${OPENCV_LIBRARY_RELEASE}")
elseif(OPENCV_LIBRARY_DEBUG)
    set(OPENCV_LIBRARIES ${OPENCV_LIBRARY_DEBUG})
    message(STATUS "OpenCV library: ${OPENCV_LIBRARY_DEBUG}")
else()
    message(FATAL_ERROR "OpenCV library not found in ${OPENCV_DIR}/bin")
endif()

# ==================== 包含目录 ====================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENCV_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

# ==================== 库源文件 ====================

set(CALIB_SOURCES
    src/CameraCalibrator.cpp
    src/HandEyeCalibrator.cpp
    src/HandEyeCalibrator2D.cpp
    src/HandEyeCalibrator25D.cpp
    src/Utils.cpp
)

set(CALIB_HEADERS
    include/CalibrationConfig.h
    include/CameraCalibrator.h
    include/HandEyeCalibrator.h
    include/HandEyeCalibrator2D.h
    include/HandEyeCalibrator25D.h
    include/Utils.h
)

# ==================== 创建动态库 ====================

add_library(VisionCalibration SHARED ${CALIB_SOURCES} ${CALIB_HEADERS})

# 定义导出宏
if(WIN32)
    target_compile_definitions(VisionCalibration
        PUBLIC
            $<$<CONFIG:Debug>:VISION_CALIBRATION_EXPORTS>
            $<$<CONFIG:Release>:VISION_CALIBRATION_EXPORTS>
    )
else()
    target_compile_definitions(VisionCalibration PUBLIC VISION_CALIBRATION_EXPORTS)
endif()

target_include_directories(VisionCalibration
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OPENCV_INCLUDE_DIR}
)

target_link_libraries(VisionCalibration
    PRIVATE
        ${OPENCV_LIBRARIES}
        nlohmann_json
)

# 设置输出目录
if(MSVC)
    set_target_properties(VisionCalibration PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/Debug"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release"
    )

    # 复制 OpenCV DLL
    add_custom_command(TARGET VisionCalibration POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DIR}/bin/$<CONFIG>/opencv_world460$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:VisionCalibration>"
    )
endif()

# ==================== 测试程序 ====================

option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
    add_executable(test_camera_calibrator test/test_camera_calibrator.cpp)

target_include_directories(test_camera_calibrator
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OPENCV_INCLUDE_DIR}
            ${JSON_INCLUDE_DIR}
    )

    target_link_libraries(test_camera_calibrator
        PRIVATE
            VisionCalibration
            ${OPENCV_LIBRARIES}
    )

    if(MSVC)
        set_target_properties(test_camera_calibrator PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
        )

        # 复制 OpenCV DLL
        add_custom_command(TARGET test_camera_calibrator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DIR}/bin/$<CONFIG>/opencv_world460$<$<CONFIG:Debug>:d>.dll"
                "$<TARGET_FILE_DIR:test_camera_calibrator>"
        )
    endif()
endif()

# ==================== 安装配置 ====================

install(TARGETS VisionCalibration
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==================== 摘要 ====================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "VisionCalibration v${PROJECT_VERSION} Configuration")
message(STATUS "========================================")
message(STATUS "  OpenCV: ${OPENCV_INCLUDE_DIR}")
message(STATUS "  JSON:   ${JSON_INCLUDE_DIR}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  CameraCalibrator: AVAILABLE")
message(STATUS "  HandEyeCalibrator: AVAILABLE")
message(STATUS "  HandEyeCalibrator2D: AVAILABLE")
message(STATUS "  HandEyeCalibrator25D: AVAILABLE")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "========================================")