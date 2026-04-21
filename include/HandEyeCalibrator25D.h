/**
 * @file HandEyeCalibrator25D.h
 * @brief 2.5D手眼标定器头文件
 * 
 * 用于2.5D手眼标定，结合2D图像特征和深度信息进行标定
 * 支持基于Z轴距离的相机与机器人坐标系转换
 * 
 * @author 系统开发组
 * @date 2026-04-20
 * @version 1.0.0
 * 
 * @section 功能特性
 * - 基于图像中心点坐标和特征尺度的2.5D标定
 * - 支持机器人Z轴距离与图像特征的线性关系标定
 * - 计算相机与机器人坐标系的转换参数
 * 
 * @section 使用说明
 * 1. 创建HandEyeCalibrator25D实例
 * 2. 调用addObservation()添加观测数据
 * 3. 调用calibrate()执行标定计算
 * 4. 获取标定结果参数
 */

#pragma once

#include "CalibrationConfig.h"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 2.5D手眼标定器类
 * 
 * 实现基于Z轴距离的2.5D手眼标定算法，用于标定相机与机器人之间的转换关系
 * 采用线性回归方法计算相机坐标系与机器人坐标系的转换参数
 */
class CALIBRATION_API HandEyeCalibrator25D
{
public:
    /**
     * @brief 默认构造函数
     * 
     * 初始化所有参数为默认值
     */
    HandEyeCalibrator25D();

    /**
     * @brief 析构函数
     */
    ~HandEyeCalibrator25D();

    /**
     * @brief 添加观测数据
     * @param center 图像中心坐标 (像素)
     * @param scale 特征尺度
     * @param robotZ 机器人Z轴坐标 (mm)
     * 
     * 添加一组观测数据，用于后续标定计算
     */
    void addObservation(const cv::Point2f &center, double scale, double robotZ);

    /**
     * @brief 执行标定计算
     * @return true-标定成功，false-标定失败
     * 
     * 基于已添加的观测数据执行标定计算，使用线性回归方法
     * 至少需要2组观测数据才能进行标定
     */
    bool calibrate();

    /**
     * @brief 获取标定结果 - X轴偏移
     * @return X轴偏移量 (像素)
     * 
     * 返回相机坐标系相对于机器人坐标系的X轴偏移量
     */
    double getCx() const;

    /**
     * @brief 获取标定结果 - Y轴偏移
     * @return Y轴偏移量 (像素)
     * 
     * 返回相机坐标系相对于机器人坐标系的Y轴偏移量
     */
    double getCy() const;

    /**
     * @brief 获取标定结果 - 尺度因子
     * @return 尺度因子 (像素/mm)
     * 
     * 返回机器人Z轴变化与图像尺度变化的比例关系
     */
    double getKScale() const;

    /**
     * @brief 获取标定结果 - 尺度偏移
     * @return 尺度偏移量 (像素)
     * 
     * 返回尺度关系中的偏移项
     */
    double getS0() const;

    /**
     * @brief 清空所有观测数据
     * 
     * 清空所有已添加的观测数据，重新开始标定流程
     */
    void clear();

private:
    std::vector<cv::Point2f> m_centers;  ///< 图像中心坐标集合
    std::vector<double> m_scales;         ///< 特征尺度集合
    std::vector<double> m_robotZs;        ///< 机器人Z轴坐标集合

    double m_cx;     ///< X轴偏移量 (像素)
    double m_cy;     ///< Y轴偏移量 (像素)
    double m_kScale; ///< 尺度因子 (像素/mm)
    double m_S0;     ///< 尺度偏移量 (像素)
};