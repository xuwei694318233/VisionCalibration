#include "HandEyeCalibrator2D.h"
#include <iostream>
#include <numeric>

HandEyeCalibrator2D::HandEyeCalibrator2D() 
    : m_mode(EYE_TO_HAND), m_reprojectionError(0.0)
{
    m_resultMatrix = cv::Mat::eye(3, 3, CV_64F);
}

HandEyeCalibrator2D::~HandEyeCalibrator2D() {
}

void HandEyeCalibrator2D::setMode(CalibrationMode mode) {
    m_mode = mode;
}

void HandEyeCalibrator2D::addObservation(const cv::Point2f& pixel, const cv::Vec3f& robotPose) {
    m_pixels.push_back(pixel);
    m_robotPoses.push_back(robotPose);
}

void HandEyeCalibrator2D::clear() {
    m_pixels.clear();
    m_robotPoses.clear();
    m_resultMatrix = cv::Mat::eye(3, 3, CV_64F);
    m_reprojectionError = 0.0;
}

cv::Mat HandEyeCalibrator2D::getResult() const {
    return m_resultMatrix.clone();
}

double HandEyeCalibrator2D::getReprojectionError() const {
    return m_reprojectionError;
}

cv::Mat HandEyeCalibrator2D::poseToMatrix(const cv::Vec3f& pose) {
    float x = pose[0];
    float y = pose[1];
    float theta = pose[2];

    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 0) = cos(theta);
    T.at<double>(0, 1) = -sin(theta);
    T.at<double>(0, 2) = x;
    T.at<double>(1, 0) = sin(theta);
    T.at<double>(1, 1) = cos(theta);
    T.at<double>(1, 2) = y;
    return T;
}

bool HandEyeCalibrator2D::calibrate() {
    if (m_pixels.size() < 3 || m_pixels.size() != m_robotPoses.size()) {
        std::cerr << "Insufficient data for 2D calibration. Need at least 3 points." << std::endl;
        return false;
    }

    if (m_mode == EYE_TO_HAND) {
        return calibrateEyeToHand();
    } else {
        return calibrateEyeInHand();
    }
}

bool HandEyeCalibrator2D::calibrateEyeToHand() {
    // Eye-to-Hand: Camera is fixed, target is moved by robot.
    // Assuming target is at the robot tool origin (or standard fixed offset mapped to pixel).
    // Equation: P_base = H_cam2base * P_pixel
    // t_base_i = R_h * P_pixel_i + t_h
    // We solve for rigid H_cam2base = [R_h, t_h] avoiding affine non-rigid distortion.

    size_t N = m_pixels.size();
    if (N < 3) return false;

    // 1. Solve for Rotation (c, s) using relative translations to eliminate t_h
    // t_i - t_j = R_h * (P_pixel_i - P_pixel_j)
    // [dx] = [ du  -dv ] [ c ]
    // [dy]   [ dv   du ] [ s ]
    cv::Mat A_rot = cv::Mat::zeros(2 * (N - 1), 2, CV_64F);
    cv::Mat B_rot = cv::Mat::zeros(2 * (N - 1), 1, CV_64F);

    int row = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        double dx = m_robotPoses[i][0] - m_robotPoses[i+1][0];
        double dy = m_robotPoses[i][1] - m_robotPoses[i+1][1];
        
        double du = m_pixels[i].x - m_pixels[i+1].x;
        double dv = m_pixels[i].y - m_pixels[i+1].y;

        A_rot.at<double>(row, 0) = du;
        A_rot.at<double>(row, 1) = -dv;
        B_rot.at<double>(row, 0) = dx;

        A_rot.at<double>(row + 1, 0) = dv;
        A_rot.at<double>(row + 1, 1) = du;
        B_rot.at<double>(row + 1, 0) = dy;
        
        row += 2;
    }

    cv::Mat X_rot;
    cv::solve(A_rot, B_rot, X_rot, cv::DECOMP_SVD);

    double c = X_rot.at<double>(0, 0);
    double s = X_rot.at<double>(1, 0);
    double angle = std::atan2(s, c);
    double c_opt = std::cos(angle);
    double s_opt = std::sin(angle);

    // 2. Solve for Translation (t_x, t_y) using the orthogonalized rotation
    // t_h = t_i - R_h * P_pixel_i
    double sum_tx = 0, sum_ty = 0;
    for (size_t i = 0; i < N; ++i) {
        double px = m_pixels[i].x;
        double py = m_pixels[i].y;
        double rx = m_robotPoses[i][0];
        double ry = m_robotPoses[i][1];

        sum_tx += (rx - (c_opt * px - s_opt * py));
        sum_ty += (ry - (s_opt * px + c_opt * py));
    }
    double tx_opt = sum_tx / N;
    double ty_opt = sum_ty / N;

    // 3. Construct Result Matrix
    m_resultMatrix = cv::Mat::eye(3, 3, CV_64F);
    m_resultMatrix.at<double>(0, 0) = c_opt;
    m_resultMatrix.at<double>(0, 1) = -s_opt;
    m_resultMatrix.at<double>(0, 2) = tx_opt;
    m_resultMatrix.at<double>(1, 0) = s_opt;
    m_resultMatrix.at<double>(1, 1) = c_opt;
    m_resultMatrix.at<double>(1, 2) = ty_opt;

    // 4. Calculate Error
    double totalErr = 0;
    for (size_t i = 0; i < N; ++i) {
        cv::Mat p = (cv::Mat_<double>(3, 1) << m_pixels[i].x, m_pixels[i].y, 1.0);
        cv::Mat p_est = m_resultMatrix * p;
        
        double err_x = p_est.at<double>(0) - m_robotPoses[i][0];
        double err_y = p_est.at<double>(1) - m_robotPoses[i][1];
        totalErr += std::sqrt(err_x * err_x + err_y * err_y);
    }
    m_reprojectionError = totalErr / N;

    return true;
}

bool HandEyeCalibrator2D::calibrateEyeInHand() {
    // Eye-in-Hand: Camera is on the robot. Target is fixed in the world.
    // P_world = T_robot * H_cam2tool * P_pixel
    // H_cam2tool = [R_h, t_h] (Rigid 2D transform)
    // T_robot = [R_i, t_i]
    // Equation: R_i (R_h P_pixel_i + t_h) + t_i = R_j (R_h P_pixel_j + t_h) + t_j
    // (R_i R_h P_pixel_i - R_j R_h P_pixel_j) + (R_i - R_j) t_h = t_j - t_i

    size_t N = m_pixels.size();
    if (N < 3) return false;

    // 1. Setup linear system A * X = B
    // X = [c, s, tx, ty]^T
    int numPairs = N - 1; 
    cv::Mat A = cv::Mat::zeros(2 * numPairs, 4, CV_64F);
    cv::Mat B = cv::Mat::zeros(2 * numPairs, 1, CV_64F);

    int row = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        size_t j = i + 1;
        cv::Mat T_i = poseToMatrix(m_robotPoses[i]);
        cv::Mat T_j = poseToMatrix(m_robotPoses[j]);

        cv::Mat R_i = T_i(cv::Rect(0, 0, 2, 2));
        cv::Mat R_j = T_j(cv::Rect(0, 0, 2, 2));
        cv::Mat t_i = T_i(cv::Rect(2, 0, 1, 2));
        cv::Mat t_j = T_j(cv::Rect(2, 0, 1, 2));

        // Let M_i = R_i * [u_i, -v_i; v_i, u_i]
        cv::Mat px_i = (cv::Mat_<double>(2, 2) << m_pixels[i].x, -m_pixels[i].y, m_pixels[i].y, m_pixels[i].x);
        cv::Mat px_j = (cv::Mat_<double>(2, 2) << m_pixels[j].x, -m_pixels[j].y, m_pixels[j].y, m_pixels[j].x);

        cv::Mat K_i = R_i * px_i;
        cv::Mat K_j = R_j * px_j;
        cv::Mat K_diff = K_i - K_j;
        cv::Mat L_diff = R_i - R_j;
        cv::Mat t_diff = t_j - t_i;

        K_diff.copyTo(A(cv::Rect(0, row, 2, 2)));
        L_diff.copyTo(A(cv::Rect(2, row, 2, 2)));
        t_diff.copyTo(B(cv::Rect(0, row, 1, 2)));

        row += 2;
    }

    // 2. Initial solve using SVD
    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);

    // 3. Orthogonalize rotation
    double c = X.at<double>(0, 0);
    double s = X.at<double>(1, 0);
    double angle = std::atan2(s, c);
    double c_opt = std::cos(angle);
    double s_opt = std::sin(angle);

    // 4. Polish translation (t_x, t_y) using orthogonalized rotation
    // (R_i - R_j) t_h = t_j - t_i - (K_i - K_j) * [c_opt, s_opt]^T
    cv::Mat A_t = cv::Mat::zeros(2 * numPairs, 2, CV_64F);
    cv::Mat B_t = cv::Mat::zeros(2 * numPairs, 1, CV_64F);
    cv::Mat rot_opt = (cv::Mat_<double>(2, 1) << c_opt, s_opt);

    row = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        size_t j = i + 1;
        
        cv::Mat L_diff = A(cv::Rect(2, row, 2, 2));
        cv::Mat K_diff = A(cv::Rect(0, row, 2, 2));
        cv::Mat t_diff = B(cv::Rect(0, row, 1, 2));

        cv::Mat target_t = t_diff - K_diff * rot_opt;

        L_diff.copyTo(A_t(cv::Rect(0, row, 2, 2)));
        target_t.copyTo(B_t(cv::Rect(0, row, 1, 2)));

        row += 2;
    }

    cv::Mat X_t;
    cv::solve(A_t, B_t, X_t, cv::DECOMP_SVD);
    double tx_opt = X_t.at<double>(0, 0);
    double ty_opt = X_t.at<double>(1, 0);

    // 5. Construct result matrix
    m_resultMatrix = cv::Mat::eye(3, 3, CV_64F);
    m_resultMatrix.at<double>(0, 0) = c_opt;
    m_resultMatrix.at<double>(0, 1) = -s_opt;
    m_resultMatrix.at<double>(0, 2) = tx_opt;
    m_resultMatrix.at<double>(1, 0) = s_opt;
    m_resultMatrix.at<double>(1, 1) = c_opt;
    m_resultMatrix.at<double>(1, 2) = ty_opt;

    // 6. Calculate P_world to measure reprojection error
    cv::Point2d P_world(0, 0);
    for (size_t i = 0; i < N; ++i) {
        cv::Mat T = poseToMatrix(m_robotPoses[i]);
        cv::Mat p_pix = (cv::Mat_<double>(3, 1) << m_pixels[i].x, m_pixels[i].y, 1.0);
        cv::Mat p_est = T * m_resultMatrix * p_pix;
        P_world.x += p_est.at<double>(0);
        P_world.y += p_est.at<double>(1);
    }
    P_world.x /= N;
    P_world.y /= N;

    // 7. Calculate Error
    double totalErr = 0;
    for (size_t i = 0; i < N; ++i) {
        cv::Mat T = poseToMatrix(m_robotPoses[i]);
        cv::Mat p_pix = (cv::Mat_<double>(3, 1) << m_pixels[i].x, m_pixels[i].y, 1.0);
        cv::Mat p_est = T * m_resultMatrix * p_pix;
        
        double dx = p_est.at<double>(0) - P_world.x;
        double dy = p_est.at<double>(1) - P_world.y;
        totalErr += std::sqrt(dx*dx + dy*dy);
    }
    m_reprojectionError = totalErr / N;

    return true;
}
