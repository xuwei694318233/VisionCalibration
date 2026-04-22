#ifndef __HAND_EYE_CALIBRATOR_2D_H__
#define __HAND_EYE_CALIBRATOR_2D_H__

#include "CalibrationConfig.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "Utils.h"

class CALIBRATION_API HandEyeCalibrator2D {
public:
    enum CalibrationMode {
        EYE_IN_HAND,
        EYE_TO_HAND
    };

    HandEyeCalibrator2D();
    ~HandEyeCalibrator2D();

    // 设置标定模式
    void setMode(CalibrationMode mode);

    // 添加数据点
    // pixel: 图像坐标 (u, v)
    // robotPose: 机器人位姿 (x, y, theta_rad)
    void addObservation(const cv::Point2f& pixel, const cv::Vec3f& robotPose);

    // 执行标定
    bool calibrate();

    // 获取结果矩阵 (3x3)
    cv::Mat getResult() const;

    // 获取重投影误差
    double getReprojectionError() const;

    // 清除数据
    void clear();

private:
    // Eye-to-Hand 标定算法
    bool calibrateEyeToHand();

    // Eye-in-Hand 标定算法
    bool calibrateEyeInHand();

    // 辅助：将 (x, y, theta) 转换为 3x3 矩阵
    cv::Mat poseToMatrix(const cv::Vec3f& pose);

private:
    CalibrationMode m_mode;
    std::vector<cv::Point2f> m_pixels;
    std::vector<cv::Vec3f> m_robotPoses; // x, y, theta
    
    cv::Mat m_resultMatrix; // 3x3 Affine Matrix
    double m_reprojectionError;
};

#endif // __HAND_EYE_CALIBRATOR_2D_H__