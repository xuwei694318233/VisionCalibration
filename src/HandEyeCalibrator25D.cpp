#include "HandEyeCalibrator25D.h"

HandEyeCalibrator25D::HandEyeCalibrator25D() : m_cx(0), m_cy(0), m_kScale(0), m_S0(0) {}
HandEyeCalibrator25D::~HandEyeCalibrator25D() {}

void HandEyeCalibrator25D::addObservation(const cv::Point2f& center, double scale, double robotZ) {
    m_centers.push_back(center);
    m_scales.push_back(scale);
    m_robotZs.push_back(robotZ);
}

bool HandEyeCalibrator25D::calibrate() {
    int n = (int)m_robotZs.size();
    if (n < 2) return false;

    cv::Mat A(n, 2, CV_64F);
    cv::Mat b_u(n, 1, CV_64F);
    cv::Mat b_v(n, 1, CV_64F);
    cv::Mat b_scale(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = m_robotZs[i];
        A.at<double>(i, 1) = 1.0;
        
        b_u.at<double>(i, 0) = m_centers[i].x;
        b_v.at<double>(i, 0) = m_centers[i].y;
        b_scale.at<double>(i, 0) = m_scales[i];
    }

    cv::Mat res_u, res_v, res_scale;
    if (!cv::solve(A, b_u, res_u, cv::DECOMP_SVD)) return false;
    if (!cv::solve(A, b_v, res_v, cv::DECOMP_SVD)) return false;
    if (!cv::solve(A, b_scale, res_scale, cv::DECOMP_SVD)) return false;

    m_cx = res_u.at<double>(0, 0);
    m_cy = res_v.at<double>(0, 0);
    m_kScale = res_scale.at<double>(0, 0);
    m_S0 = res_scale.at<double>(1, 0);

    return true;
}

double HandEyeCalibrator25D::getCx() const { return m_cx; }
double HandEyeCalibrator25D::getCy() const { return m_cy; }
double HandEyeCalibrator25D::getKScale() const { return m_kScale; }
double HandEyeCalibrator25D::getS0() const { return m_S0; }

void HandEyeCalibrator25D::clear() {
    m_centers.clear();
    m_scales.clear();
    m_robotZs.clear();
}