#pragma once

#include "CalibrationConfig.h"
#include <opencv2/opencv.hpp>
#include <vector>

class CALIBRATION_API HandEyeCalibrator25D {
public:
    HandEyeCalibrator25D();
    ~HandEyeCalibrator25D();

    void addObservation(const cv::Point2f& center, double scale, double robotZ);
    bool calibrate();

    double getCx() const;
    double getCy() const;
    double getKScale() const;
    double getS0() const;

    void clear();

private:
    std::vector<cv::Point2f> m_centers;
    std::vector<double> m_scales;
    std::vector<double> m_robotZs;

    double m_cx;
    double m_cy;
    double m_kScale;
    double m_S0;
};