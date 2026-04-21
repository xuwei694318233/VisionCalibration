#include "Utils.h"


const double _epsilon = 1e-6;

/**
 * @brief 将旋转矩阵与平移向量合成齐次矩阵（自动类型推导版）
 * @param R 输入3x3旋转矩阵（CV_32F/CV_64F）
 * @param T 输入3x1平移向量（CV_32F/CV_64F）
 * @return 4x4齐次矩阵（类型与输入一致）
 * @note 自动处理类型转换，若R/T类型不一致则转为更高精度（double优先）
 */
bool R_T2H(const cv::Mat& R, const cv::Mat& T, cv::Mat& H_out) {
    // 1. 检查输入维度和类型
    if (R.rows != 3 || R.cols != 3) {
        LOG_OUTPUT(Error, "Input rotation matrix dimension error");
        return false;
    }
    // 2. 检查平移向量维度 (3x1 或 1x3)
    if (!((T.rows == 3 && T.cols == 1) || (T.rows == 1 && T.cols == 3))) {
        LOG_OUTPUT(Error, "Input translation vector dimension error, should be 3x1 or 1x3");
        return false;
    }

    // 3. 构建齐次矩阵
    cv::Mat H = cv::Mat::eye(4, 4, R.type());
    R.copyTo(H(cv::Rect(0, 0, 3, 3)));
    T.copyTo(H(cv::Rect(3, 0, 1, 3)));
    H_out = H;
    return true;
}

/**
 * @brief 将齐次矩阵分解为旋转矩阵与平移向量（自动类型推导版）
 * @param H  输入4x4齐次矩阵（支持CV_32F/CV_64F）
 * @param R  输出3x3旋转矩阵（与H同类型）
 * @param T  输出3x1平移向量（与H同类型）
 * @note 内部自动处理类型转换，但会严格检查旋转矩阵正交性
 */
bool H2R_T(const cv::Mat& H, cv::Mat& R, cv::Mat& T) {
    // 1. 检查维度
    if (H.rows != 4 || H.cols != 4) {
        LOG_OUTPUT(Error, "Input homogeneous matrix dimension error");
        return false;
    }
    if (!(H.type() == CV_32F || H.type() == CV_64F)) {
        LOG_OUTPUT(Error, "Input homogeneous matrix type error, only CV_32F or CV_64F is supported");
        return false;
    }

    // 2. 提取旋转和平移部分（保持原类型）
    R = H(cv::Rect(0, 0, 3, 3)).clone();
    T = H(cv::Rect(3, 0, 1, 3)).clone();

    // 3. 宽进严出：转换为double验证正交性
    cv::Mat R_double;
    R.convertTo(R_double, CV_64F); // 统一用高精度验证
    if (!isRotationMatrix(R_double)) {
        LOG_OUTPUT(Warn, "The extracted rotation matrix is not orthogonal");
    }
    return true;
}

/**
 * @brief 检查矩阵是否为有效的旋转矩阵（正交且行列式为+1）
 * @param R 输入3x3矩阵，类型需为 CV_32F 或 CV_64F
 * @return true 如果是旋转矩阵，否则 false
 */
bool isRotationMatrix(const cv::Mat& R) {
    const double epsilon = 1e-6;  // 定义判断阈值

    // 1. 检查维度
    if (R.rows != 3 || R.cols != 3) {
        std::stringstream ss;
        ss << "Rotation matrix must be 3x3, got " << R.rows << "x" << R.cols;
        LOG_OUTPUT(Error, ss.str().c_str());
        return false;
    }

    // 2. 检查类型
    int mat_type = R.type();
    if (mat_type != CV_32F && mat_type != CV_64F) {
        std::stringstream ss;
        ss << "Rotation matrix type must be CV_32F or CV_64F, got " << mat_type;
        LOG_OUTPUT(Error, ss.str().c_str());
        return false;
    }

    // 3. 计算R^T * R
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;

    // 4. 创建单位矩阵用于比较
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    // 5. 计算差异
    double norm = cv::norm(I, shouldBeIdentity);
    double det = cv::determinant(R);

    // 调试输出（可选）
    // std::cout << "Orthogonality check: " << norm 
    //          << ", Determinant: " << det << std::endl;

    // 6. 验证正交性和行列式
    return (norm < epsilon) && (fabs(det - 1.0) < epsilon);
}

//bool isRotationMatrix(const Eigen::Matrix3d& R) {
//    if (fabs(R.determinant() - 1.0) > _epsilon) {
//        return false;
//    }
//
//    Eigen::Matrix3d RtR = R.transpose() * R;
//    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//    return (RtR - I).norm() < _epsilon;
//}

/**
 * @brief 将欧拉角转换为旋转矩阵
 * @param euler_angles 输入欧拉角 [rx, ry, rz]（单位：弧度）
 * @param rotation_order 旋转顺序（如 "ZYX"），默认为 "ZYX"
 * @param dtype 输出矩阵类型（CV_32F 或 CV_64F），默认为 CV_64F
 * @return 3x3旋转矩阵
 * @note 支持的旋转顺序：XYZ, XZY, YXZ, YZX, ZXY, ZYX
 */
bool eulerToRotationMatrix(const cv::Vec3d& euler_angles,
    cv::Mat& R, const std::string& rotation_order, int dtype)
{
    // 1. 检查输入
    if (!(dtype == CV_32F || dtype == CV_64F)) {
        LOG_OUTPUT(Error, "Input matrix type error, only CV_32F or CV_64F is supported");
        return false;
    }
    if (rotation_order.size() != 3) {
        LOG_OUTPUT(Error, "Rotation order length error, only 3 characters are supported");
        return false;
    }
    std::string order_upper = std::string(rotation_order);
    // 转换为大写，实现大小写不敏感
    std::transform(order_upper.begin(), order_upper.end(),
        order_upper.begin(), ::toupper);

    // 2. 提取欧拉角
    const double rx = euler_angles[0];
    const double ry = euler_angles[1];
    const double rz = euler_angles[2];

    // 3. 计算各轴旋转矩阵
    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(rx), -sin(rx),
        0, sin(rx), cos(rx));

    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
        cos(ry), 0, sin(ry),
        0, 1, 0,
        -sin(ry), 0, cos(ry));

    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        cos(rz), -sin(rz), 0,
        sin(rz), cos(rz), 0,
        0, 0, 1);
    // 4. 按指定顺序组合旋转矩阵
    if (order_upper == "XYZ") {
        R = Rx * Ry * Rz;
    }
    else if (order_upper == "XZY") {
        R = Rx * Rz * Ry;
    }
    else if (order_upper == "YXZ") {
        R = Ry * Rx * Rz;
    }
    else if (order_upper == "YZX") {
        R = Ry * Rz * Rx;
    }
    else if (order_upper == "ZXY") {
        R = Rz * Rx * Ry;
    }
    else if (order_upper == "ZYX") {
        R = Rz * Ry * Rx;
    }
    else {
        std::string msg("Unsupported rotation order: " + order_upper);
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }
    // 5. 检查旋转矩阵是否正交
    if (!isRotationMatrix(R)) {
        LOG_OUTPUT(Error, "Internal generated rotation matrix is not orthogonal, algorithm logic error");
        return false;
    }
    // 6. 检查欧拉角是否出现死锁
    if (order_upper == "ZYX" || order_upper == "XYZ") {
        if (fabs(fabs(ry) - CV_PI / 2) < _epsilon) {
            std::string msg("Euler angles are close to gimbal lock (ry approx +-90 deg), rotation order: " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }
    else if (order_upper == "ZXY" || order_upper == "YZX") {
        if (fabs(fabs(rx) - CV_PI / 2) < _epsilon) {
            std::string msg("Euler angles are close to gimbal lock (rx approx +-90 deg), rotation order: " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }
    else if (order_upper == "YXZ" || order_upper == "XZY") {
        if (fabs(fabs(rz) - CV_PI / 2) < _epsilon) {
            std::string msg("Euler angles are close to gimbal lock(rz approx + -90 deg), rotation order : " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }
    // 7. 转换为目标类型
    cv::Mat R_output;
    R.convertTo(R_output, dtype);
    return true;
}

/**
 * @brief 将欧拉角转换为旋转矩阵 (仅支持float类型)
 * @param euler_angles 输入欧拉角 [rx, ry, rz]（单位：弧度），3x1或1x3的CV_32F矩阵
 * @param R 输出的3x3旋转矩阵 (CV_32F)
 * @param rotation_order 旋转顺序（如 "ZYX"），默认为 "ZYX"
 * @return 成功返回true，失败返回false
 * @note 支持的旋转顺序：XYZ, XZY, YXZ, YZX, ZXY, ZYX
 */
bool eulerToRotationMatrix(const cv::Mat& euler_angles,
    cv::Mat& R,
    const std::string& rotation_order)
{
    // 1. 检查输入矩阵
    if (euler_angles.empty() ||
        (euler_angles.rows != 3 && euler_angles.cols != 3) ||
        euler_angles.type() != CV_32F)
    {
        LOG_OUTPUT(Error, "Input euler angles should be a 3x1 or 1x3 CV_32F matrix");
        return false;
    }

    // 2. 检查旋转顺序
    if (rotation_order.size() != 3) {
        LOG_OUTPUT(Error, "Rotation order length error, only 3 characters are supported");
        return false;
    }
    std::string order_upper = rotation_order;
    std::transform(order_upper.begin(), order_upper.end(),
        order_upper.begin(), ::toupper);

    // 3. 提取欧拉角
    const float rx = euler_angles.at<float>(0);
    const float ry = euler_angles.at<float>(1);
    const float rz = euler_angles.at<float>(2);

    // 4. 计算各轴旋转矩阵
    cv::Mat Rx = (cv::Mat_<float>(3, 3) <<
        1.0f, 0.0f, 0.0f,
        0.0f, cosf(rx), -sinf(rx),
        0.0f, sinf(rx), cosf(rx));

    cv::Mat Ry = (cv::Mat_<float>(3, 3) <<
        cosf(ry), 0.0f, sinf(ry),
        0.0f, 1.0f, 0.0f,
        -sinf(ry), 0.0f, cosf(ry));

    cv::Mat Rz = (cv::Mat_<float>(3, 3) <<
        cosf(rz), -sinf(rz), 0.0f,
        sinf(rz), cosf(rz), 0.0f,
        0.0f, 0.0f, 1.0f);

    // 5. 按指定顺序组合旋转矩阵
    if (order_upper == "XYZ") {
        R = Rx * Ry * Rz;
    }
    else if (order_upper == "XZY") {
        R = Rx * Rz * Ry;
    }
    else if (order_upper == "YXZ") {
        R = Ry * Rx * Rz;
    }
    else if (order_upper == "YZX") {
        R = Ry * Rz * Rx;
    }
    else if (order_upper == "ZXY") {
        R = Rz * Rx * Ry;
    }
    else if (order_upper == "ZYX") {
        R = Rz * Ry * Rx;
    }
    else {
        std::string msg("Unsupported rotation order: " + order_upper);
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }

    // 6. 检查旋转矩阵是否正交
    if (!isRotationMatrix(R)) {
        LOG_OUTPUT(Error, "Generated rotation matrix is not orthogonal");
        return false;
    }

    // 7. 检查欧拉角死锁
    const float epsilon = 1e-5f;
    if (order_upper == "ZYX" || order_upper == "XYZ") {
        if (fabsf(fabsf(ry) - CV_PI / 2) < epsilon) {
            std::string msg("Euler angles are close to gimbal lock (ry approx +-90 deg), rotation order: " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }
    else if (order_upper == "ZXY" || order_upper == "YZX") {
        if (fabsf(fabsf(rx) - CV_PI / 2) < epsilon) {
            std::string msg("Euler angles are close to gimbal lock (rx approx +-90 deg), rotation order: " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }
    else if (order_upper == "YXZ" || order_upper == "XZY") {
        if (fabsf(fabsf(rz) - CV_PI / 2) < epsilon) {
            std::string msg("Euler angles are close to gimbal lock (rz approx +-90 deg), rotation order: " + order_upper);
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }

    return true;
}

//bool eulerToRotationMatrix(const Eigen::Vector3d& euler_angle,
//    Eigen::Matrix3d& R, const std::string& rotation_order
//) {
//    if (rotation_order.size() != 3) {
//        LOG_OUTPUT(Error, std::string("Rotation order length error, only 3 characters are supported"));
//        return false;
//    }
//    std::string order_upper = std::string(rotation_order);
//    std::transform(order_upper.begin(), order_upper.end(), order_upper.begin(), ::toupper);
//    const double rx = euler_angle[0];
//    const double ry = euler_angle[1];
//    const double rz = euler_angle[2];
//    if (order_upper == "ZYX") {
//        R = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
//            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
//            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX());
//    }
//    else if (order_upper == "XYZ") {
//        R = Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
//            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
//            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ());
//    }
//    else if (order_upper == "XZY") {
//        R = Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
//            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
//            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY());
//    }
//    else if (order_upper == "YXZ") {
//        R = Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
//            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
//            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ());
//    }
//    else if (order_upper == "YZX") {
//        R = Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
//            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
//            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX());
//    }
//    else if (order_upper == "ZXY") {
//        R = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
//            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
//            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY());
//    }
//    else {
//        LOG_OUTPUT(Error, "Unsupported rotation order: " + order_upper);
//        return false;
//    }
//    if (!isRotationMatrix(R)) {
//        LOG_OUTPUT(Error, std::string("Internal generated rotation matrix is not orthogonal, algorithm logic error"));
//        return false;
//    }
//
//    if (order_upper == "ZYX" || order_upper == "XYZ") {
//        if (fabs(fabs(ry) - CV_PI / 2) < _epsilon) {
//            LOG_OUTPUT(Warn, "Euler angles are close to gimbal lock (ry approx +-90 deg), rotation order: " + order_upper);
//        }
//    }
//    else if (order_upper == "ZXY" || order_upper == "YZX") {
//        if (fabs(fabs(rx) - CV_PI / 2) < _epsilon) {
//            LOG_OUTPUT(Warn, "Euler angles are close to gimbal lock (rx approx +-90 deg), rotation order: " + order_upper);
//        }
//    }
//    else if (order_upper == "YXZ" || order_upper == "XZY") {
//        if (fabs(fabs(rz) - CV_PI / 2) < _epsilon) {
//            LOG_OUTPUT(Warn, "Euler angles are close to gimbal lock (rz approx +-90 deg), rotation order: " + order_upper);
//        }
//    }
//    return true;
//
//}

/**
 * @brief 将四元数转换为旋转矩阵
 * @param q 输入四元数，格式为 [w, x, y, z]
 * @param dtype 输出矩阵类型，CV_32F 或 CV_64F（默认为CV_64F）
 * @return 3x3旋转矩阵
 */
bool quaternionToRotationMatrix(const cv::Vec4d& q, cv::Mat& R_output, int dtype) {
    // 1. 参数检查
    if (!(dtype == CV_32F || dtype == CV_64F)) {
        LOG_OUTPUT(Error, "Output rotation matrix type error");
        return false;
    }
    if (!(q.val[0] * q.val[0] + q.val[1] * q.val[1] +
        q.val[2] * q.val[2] + q.val[3] * q.val[3] > 0)) {
        LOG_OUTPUT(Error, "Input quaternion norm is 0, please check input");
        return false;
    }

    // 2. 归一化四元数（确保单位四元数）
    double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double w = q[0] / norm;
    double x = q[1] / norm;
    double y = q[2] / norm;
    double z = q[3] / norm;

    // 3. 计算旋转矩阵元素（根据四元数转旋转矩阵公式）
    double xx = x * x;
    double yy = y * y;
    double zz = z * z;
    double xy = x * y;
    double xz = x * z;
    double yz = y * z;
    double wx = w * x;
    double wy = w * y;
    double wz = w * z;

    // 4. 构建旋转矩阵
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));

    // 5. 验证是否为有效旋转矩阵
    if (!isRotationMatrix(R)) {
        LOG_OUTPUT(Error, "Generated rotation matrix is invalid, please check quaternion input");
        return false;
    }

    // 6. 转换为目标类型
    R.convertTo(R_output, dtype);
    return true;
}

/**
 * @brief 将采集的原始数据转换为齐次矩阵（从机器人控制器中获得）
 * @param pose_data 输入位姿数据，支持三种格式：
 *                  1. 1x6矩阵 [x, y, z, rx, ry, rz] (欧拉角，单位弧度)
 *                  2. 1x7矩阵 [x, y, z, q0, q1, q2, q3] (四元数)
 *                  3. 1x10矩阵 [x, y, z, rx, ry, rz, q0, q1, q2, q3]
 * @param rotation_order 欧拉角的旋转顺序，默认为"ZYX"
 * @param dtype 输出矩阵类型，默认为CV_64F
 * @return 4x4齐次变换矩阵
 * @note 宽进严出：输入数据会被严格校验，输出保证是有效的齐次变换矩阵
 */
cv::Mat poseToHomogeneousMatrix(const cv::Mat& pose_data,
    const std::string& rotation_order,
    int dtype) {
    // 1. Input validation
    if (pose_data.rows != 1) {
        LOG_OUTPUT(Error, "Input pose_data must have 1 row.");
        return cv::Mat();
    }
    if (pose_data.cols != 6 && pose_data.cols != 7) {
        LOG_OUTPUT(Error, "Input pose_data must have 6 (euler) or 7 (quaternion) columns.");
        return cv::Mat();
    }
    if (dtype != CV_32F && dtype != CV_64F) {
        LOG_OUTPUT(Error, "Output dtype must be CV_32F or CV_64F.");
        return cv::Mat();
    }

    int input_type = pose_data.type();
    if (input_type != CV_32F && input_type != CV_64F) {
        LOG_OUTPUT(Error, "Input pose_data type must be CV_32F or CV_64F.");
        return cv::Mat();
    }

    // 2. Extract translation and rotation data based on input type
    cv::Mat T;
    cv::Mat R;
    bool conversion_ok = false;

    if (input_type == CV_32F) {
        T = (cv::Mat_<double>(3, 1) <<
            (double)pose_data.at<float>(0, 0),
            (double)pose_data.at<float>(0, 1),
            (double)pose_data.at<float>(0, 2));

        if (pose_data.cols == 6) {
            cv::Vec3d euler_angles(
                (double)pose_data.at<float>(0, 3),
                (double)pose_data.at<float>(0, 4),
                (double)pose_data.at<float>(0, 5));
            conversion_ok = eulerToRotationMatrix(euler_angles, R, rotation_order);
        }
        else { // cols == 7
            cv::Vec4d quaternion(
                (double)pose_data.at<float>(0, 3), // w
                (double)pose_data.at<float>(0, 4), // x
                (double)pose_data.at<float>(0, 5), // y
                (double)pose_data.at<float>(0, 6)); // z
            conversion_ok = quaternionToRotationMatrix(quaternion, R);
        }
    }
    else { // CV_64F
        T = (cv::Mat_<double>(3, 1) <<
            pose_data.at<double>(0, 0),
            pose_data.at<double>(0, 1),
            pose_data.at<double>(0, 2));

        if (pose_data.cols == 6) {
            cv::Vec3d euler_angles(
                pose_data.at<double>(0, 3),
                pose_data.at<double>(0, 4),
                pose_data.at<double>(0, 5));
            conversion_ok = eulerToRotationMatrix(euler_angles, R, rotation_order);
        }
        else { // cols == 7
            cv::Vec4d quaternion(
                pose_data.at<double>(0, 3), // w
                pose_data.at<double>(0, 4), // x
                pose_data.at<double>(0, 5), // y
                pose_data.at<double>(0, 6)); // z
            conversion_ok = quaternionToRotationMatrix(quaternion, R);
        }
    }

    // 3. Check if rotation matrix was created successfully
    if (!conversion_ok || R.empty()) {
        LOG_OUTPUT(Error, "Failed to create rotation matrix from pose data.");
        return cv::Mat();
    }

    // 4. Combine into homogeneous matrix
    cv::Mat H;
    if (!R_T2H(R, T, H)) {
        LOG_OUTPUT(Error, "Failed to combine R and T into homogeneous matrix.");
        return cv::Mat();
    }

    // 5. Ensure correct output type
    if (H.type() != dtype) {
        H.convertTo(H, dtype);
    }
    return H;
}

cv::Mat inverseHomogeneous(const cv::Mat& H) {
    if (H.empty() || H.rows != 4 || H.cols != 4) {
        LOG_OUTPUT(Error, "Input homogeneous matrix for inversion is invalid.");
        return cv::Mat();
    }
    cv::Mat R = H(cv::Rect(0, 0, 3, 3));
    cv::Mat T = H(cv::Rect(3, 0, 1, 3));

    cv::Mat U, S, Vt;
    cv::SVDecomp(R, S, U, Vt, cv::SVD::FULL_UV);
    cv::Mat R_inv = Vt.t() * U.t();
    cv::Mat T_inv = -R_inv * T;
    cv::Mat H_inv = cv::Mat::eye(4, 4, H.type());
    R_inv.copyTo(H_inv(cv::Rect(0, 0, 3, 3)));
    T_inv.copyTo(H_inv(cv::Rect(3, 0, 1, 3)));
    return H_inv;
}

/**
 * @brief 将位姿数据转换为旋转矩阵和平移向量
 * @param pose 输入位姿数据 [x, y, z, roll, pitch, yaw] (1x6 CV_32F矩阵)
 * @param R 输出的3x3旋转矩阵 (CV_32F)
 * @param t 输出的3x1平移向量 (CV_32F)
 * @return 成功返回true，失败返回false
 */
bool poseToRT_rad(const cv::Mat& pose, cv::Mat& R, cv::Mat& t)
{
    // 1. 检查输入格式
    if (pose.empty() || pose.rows != 1 || pose.cols != 6 || pose.type() != CV_32F) {
        LOG_OUTPUT(Error, "Input pose must be 1x6 CV_32F matrix");
        return false;
    }

    // 2. 提取平移部分 (x,y,z)
    t = pose.colRange(0, 3).clone().reshape(1, 3);  // 转换为3x1矩阵

    // 3. 提取欧拉角 (roll,pitch,yaw)
    cv::Mat euler = (cv::Mat_<float>(1, 3) <<
        pose.at<float>(0, 3), pose.at<float>(0, 4), pose.at<float>(0, 5));
    if (eulerToRotationMatrix(euler, R, "ZYX", CV_32F)) {
        LOG_OUTPUT(Error, "Euler to rotation matrix failed");
        return false;
    }

    return true;
}

bool poseToRT_deg(const cv::Mat& pose, cv::Mat& R, cv::Mat& T) {
    if (pose.empty() || pose.rows != 1 || pose.cols != 6 || pose.type() != CV_32F) {
        LOG_OUTPUT(Error, "Invalid pose matrix format");
        return false;
    }

    // 提取欧拉角（角度制）并转换为弧度
    float rx = pose.at<float>(0, 3) * CV_PI / 180.0f;  // roll
    float ry = pose.at<float>(0, 4) * CV_PI / 180.0f;  // pitch
    float rz = pose.at<float>(0, 5) * CV_PI / 180.0f;  // yaw

    // 调试输出转换前后的值
    std::cout << "Angles (deg): " << pose.at<float>(0, 3) << ", "
        << pose.at<float>(0, 4) << ", " << pose.at<float>(0, 5) << std::endl;
    std::cout << "Angles (rad): " << rx << ", " << ry << ", " << rz << std::endl;

    // 计算各轴旋转矩阵
    cv::Mat Rx = (cv::Mat_<float>(3, 3) <<
        1, 0, 0,
        0, cosf(rx), -sinf(rx),
        0, sinf(rx), cosf(rx));

    cv::Mat Ry = (cv::Mat_<float>(3, 3) <<
        cosf(ry), 0, sinf(ry),
        0, 1, 0,
        -sinf(ry), 0, cosf(ry));

    cv::Mat Rz = (cv::Mat_<float>(3, 3) <<
        cosf(rz), -sinf(rz), 0,
        sinf(rz), cosf(rz), 0,
        0, 0, 1);

    // 组合旋转矩阵（ZYX顺序）
    R = Rz * Ry * Rx;

    // 提取平移向量
    T = (cv::Mat_<float>(3, 1) <<
        pose.at<float>(0, 0),
        pose.at<float>(0, 1),
        pose.at<float>(0, 2));

    return true;
}

bool Utils::naturalCompare(const std::string& a, const std::string& b) {
    std::string::const_iterator itA = a.begin(), itB = b.begin();

    while (itA != a.end() && itB != b.end()) {
        if (isdigit(*itA) && isdigit(*itB)) {
            long long numA = 0, numB = 0;
            std::string numStrA, numStrB;

            while (itA != a.end() && isdigit(*itA)) {
                numStrA += *itA++;
            }
            while (itB != b.end() && isdigit(*itB)) {
                numStrB += *itB++;
            }

            try {
                numA = std::stoll(numStrA);
                numB = std::stoll(numStrB);
            }
            catch (const std::out_of_range&) {
                // Handle very large numbers by comparing as strings
                if (numStrA.length() != numStrB.length()) {
                    return numStrA.length() < numStrB.length();
                }
                if (numStrA != numStrB) {
                    return numStrA < numStrB;
                }
                continue; // They are "equal" as numbers, continue with rest of string
            }

            if (numA != numB) {
                return numA < numB;
            }
        }
        else {
            if (*itA != *itB) {
                return *itA < *itB;
            }
            ++itA;
            ++itB;
        }
    }
    return itA == a.end() && itB != b.end();
}

bool saveMatrixToJsonInOutput(
    const cv::Mat& mat,
    const std::string& fileName,
    const std::string& key,
    int floatPrecision) {
    if (mat.empty()) {
        LOG_OUTPUT(Error, "Input matrix is empty");
        return false;
    }
    if (mat.dims != 2) {
        LOG_OUTPUT(Error, "Input matrix must be 2-dimensional");
        return false;
    }

    // Try to locate the solution root directory by walking up until a .sln is found
    auto tryFindSolutionRoot = []() -> std::filesystem::path {
        std::filesystem::path cur = std::filesystem::current_path();
        for (int i = 0; i < 10 && !cur.empty(); ++i) {
            if (std::filesystem::exists(cur / "Visionbasepro.sln")) {
                return cur;
            }
            // generic: any .sln
            bool hasSln = false;
            for (const auto& entry : std::filesystem::directory_iterator(cur)) {
                if (entry.is_regular_file() && entry.path().extension() == ".sln") {
                    hasSln = true;
                    break;
                }
            }
            if (hasSln) return cur;
            cur = cur.parent_path();
        }
        // Fallback: directory containing both known project folders
        cur = std::filesystem::current_path();
        for (int i = 0; i < 10 && !cur.empty(); ++i) {
            if (std::filesystem::exists(cur / "procedurealgo") &&
                std::filesystem::exists(cur / "proceduretools")) {
                return cur;
            }
            cur = cur.parent_path();
        }
        return std::filesystem::current_path();
    };

    const std::filesystem::path solutionRoot = tryFindSolutionRoot();
    std::filesystem::path outDir = solutionRoot / "output";
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec); // ignore error if exists

    std::filesystem::path outPath = outDir / fileName;

    // Build JSON array
    nlohmann::json arr = nlohmann::json::array();
    double scale = 1.0;
    if (floatPrecision > 0) {
        scale = std::pow(10.0, static_cast<double>(floatPrecision));
    }
    for (int r = 0; r < mat.rows; ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < mat.cols; ++c) {
            double v = 0.0;
            switch (mat.type()) {
            case CV_64F: v = mat.at<double>(r, c); break;
            case CV_32F: v = static_cast<double>(mat.at<float>(r, c)); break;
            case CV_32S: v = static_cast<double>(mat.at<int>(r, c)); break;
            case CV_16S: v = static_cast<double>(mat.at<short>(r, c)); break;
            case CV_16U: v = static_cast<double>(mat.at<unsigned short>(r, c)); break;
            case CV_8S:  v = static_cast<double>(mat.at<char>(r, c)); break;
            case CV_8U:  v = static_cast<double>(mat.at<unsigned char>(r, c)); break;
            default:
                LOG_OUTPUT(Error, "Unsupported cv::Mat type for JSON export");
                return false;
            }
            if (floatPrecision >= 0) {
                v = std::round(v * scale) / scale;
            }
            row.push_back(v);
        }
        arr.push_back(row);
    }

    nlohmann::json root;
    root[key] = arr;

    std::ofstream ofs(outPath.string());
    if (!ofs.is_open()) {
        std::string msg("Failed to open file for writing: " + outPath.string());
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }
    ofs << std::setw(2) << root;
    ofs.close();
    LOG_OUTPUT(Info, ("Successfully wirte matrix in file: " + outPath.string()).c_str());
    return true;
}

std::vector<cv::Mat> LoadRobotPoses(const std::string& filePath) {
    std::vector<cv::Mat> poses;
    std::ifstream poseStream(filePath);
    if (!poseStream.is_open()) {
        std::string msg = "Cannot open pose file: " + filePath;
        LOG_OUTPUT(Error, msg.c_str());
        return poses;
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(poseStream, line)) {
        lineNum++;
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::vector<float> values;
        float val;
        while (iss >> val) {
            values.push_back(val);
        }

        if (values.empty()) continue;

        // Check if we have a valid number of elements (6 for Euler, 7 for Quaternion)
        // We allow some flexibility but warn if it looks weird
        if (values.size() < 6) {
            std::string msg = "Line " + std::to_string(lineNum) + " has fewer than 6 elements, skipping.";
            LOG_OUTPUT(Warn, msg.c_str());
            continue;
        }

        cv::Mat poseMat(1, static_cast<int>(values.size()), CV_32F);
        for (size_t i = 0; i < values.size(); ++i) {
            poseMat.at<float>(0, static_cast<int>(i)) = values[i];
        }
        poses.push_back(poseMat);
    }
    poseStream.close();
    
    std::string msg = "Successfully loaded " + std::to_string(poses.size()) + " poses from " + filePath;
    LOG_OUTPUT(Info, msg.c_str());
    
    return poses;
}