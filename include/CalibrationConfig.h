/**
 * @file CalibrationConfig.h
 * @brief 相机标定库导出宏配置
 * 
 * 定义跨平台动态库导出宏
 * 
 * @author 系统开发组
 * @date 2026-04-20
 * @version 1.0.0
 */

#pragma once

// 动态库导出宏定义
#if defined(_WIN32)
    #ifdef VISION_CALIBRATION_EXPORTS
        #define CALIBRATION_API __declspec(dllexport)
    #else
        #define CALIBRATION_API __declspec(dllimport)
    #endif
#else
    #define CALIBRATION_API __attribute__((visibility("default")))
#endif

// 库版本信息
#define VISION_CALIBRATION_VERSION_MAJOR 1
#define VISION_CALIBRATION_VERSION_MINOR 0
#define VISION_CALIBRATION_VERSION_PATCH 0