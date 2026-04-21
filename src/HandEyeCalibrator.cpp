#include "HandEyeCalibrator.h"
#include "Utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/optim.hpp>

// 优化回调类：计算残差
class HandEyeRefinementCallback : public cv::LMSolver::Callback {
public:
    // context_mode: 0 = EYE_IN_HAND, 1 = EYE_TO_HAND
    HandEyeRefinementCallback(const std::vector<cv::Mat>& T_robotVec, 
                              const std::vector<cv::Mat>& T_targetVec, 
                              int mode,
                              double w_rot, double w_trans) 
        : m_T_robot(T_robotVec), m_T_target(T_targetVec), m_mode(mode),
          m_w_rot(w_rot), m_w_trans(w_trans) {}

    // 优化的核心：计算误差向量
    // param: 1x12 的向量 (前6个是手眼矩阵X的参数，后6个是辅助矩阵Y的参数)
    // err: 输出残差向量
    bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _jac) const override {
        cv::Mat param = _param.getMat();
        
        // Calculate Error
        calcError(param, _err);

        // Calculate Jacobian if needed (Numerical Differentiation)
        if (_jac.needed()) {
            cv::Mat err0 = _err.getMat();
            int n_params = param.rows;
            int n_errs = err0.rows;
            
            _jac.create(n_errs, n_params, CV_64F);
            cv::Mat jac = _jac.getMat();
            double epsilon = 1e-7;

            for (int j = 0; j < n_params; j++) {
                cv::Mat param_p = param.clone();
                param_p.at<double>(j) += epsilon;

                cv::Mat err_p;
                calcError(param_p, err_p);

                cv::Mat col = (err_p - err0) / epsilon;
                col.copyTo(jac.col(j));
            }
        }
        
        return true;
    }

private:
    void calcError(const cv::Mat& param, cv::OutputArray _err) const {
        // 1. 解析参数
        // X: 手眼变换 (Cam -> Gripper 或 Cam -> Base)
        cv::Mat rvec_X = param.rowRange(0, 3).clone();
        cv::Mat tvec_X = param.rowRange(3, 6).clone();
        
        // Y: 辅助变换 (Target -> Base 或 Target -> Gripper)
        cv::Mat rvec_Y = param.rowRange(6, 9).clone();
        cv::Mat tvec_Y = param.rowRange(9, 12).clone();

        cv::Mat R_X, R_Y;
        cv::Rodrigues(rvec_X, R_X);
        cv::Rodrigues(rvec_Y, R_Y);

        // 构造齐次矩阵
        cv::Mat X = cv::Mat::eye(4, 4, CV_64F);
        R_X.copyTo(X(cv::Rect(0,0,3,3)));
        tvec_X.copyTo(X(cv::Rect(3,0,1,3)));

        cv::Mat Y = cv::Mat::eye(4, 4, CV_64F);
        R_Y.copyTo(Y(cv::Rect(0,0,3,3)));
        tvec_Y.copyTo(Y(cv::Rect(3,0,1,3)));
        
        cv::Mat Y_inv = Y.inv();

        int n_poses = m_T_robot.size();
        _err.create(n_poses * 6, 1, CV_64F);
        cv::Mat err = _err.getMat();

        for(int i = 0; i < n_poses; ++i) {
            cv::Mat T_robot_i, T_target_i; // 需确保输入是 4x4 CV_64F
            m_T_robot[i].convertTo(T_robot_i, CV_64F);
            m_T_target[i].convertTo(T_target_i, CV_64F);

            cv::Mat T_chain;
            
            if (m_mode == 0) { // EYE_IN_HAND
                // Chain: T_robot * X * T_target_cam
                // Error = Y_inv * (T_robot_i * X * T_target_i)
                // 理想情况 Error 应为 Identity
                T_chain = T_robot_i * X * T_target_i;
            } else { // EYE_TO_HAND
                // EYE_TO_HAND: T_target_grip = inv(T_base_grip) * X * T_target_cam
                // Y = inv(T_robot_i) * X * T_target_i
                
                cv::Mat T_robot_inv = T_robot_i.inv();
                T_chain = T_robot_inv * X * T_target_i; 
            }

            cv::Mat diff = Y_inv * T_chain;

            // 计算差并将旋转部分转换为旋转向量
            cv::Mat R_diff = diff(cv::Rect(0,0,3,3));
            cv::Mat t_diff = diff(cv::Rect(3,0,1,3));
            
            cv::Mat r_diff_vec;
            cv::Rodrigues(R_diff, r_diff_vec);

            // 填充误差向量
            
            err.at<double>(i*6 + 0) = r_diff_vec.at<double>(0) * m_w_rot;
            err.at<double>(i*6 + 1) = r_diff_vec.at<double>(1) * m_w_rot;
            err.at<double>(i*6 + 2) = r_diff_vec.at<double>(2) * m_w_rot;
            err.at<double>(i*6 + 3) = t_diff.at<double>(0) * m_w_trans;
            err.at<double>(i*6 + 4) = t_diff.at<double>(1) * m_w_trans;
            err.at<double>(i*6 + 5) = t_diff.at<double>(2) * m_w_trans;
        }
    }

    std::vector<cv::Mat> m_T_robot;
    std::vector<cv::Mat> m_T_target;
    int m_mode;
    double m_w_rot;
    double m_w_trans;
};


HandEyeCalibrator::HandEyeCalibrator(const HandEyeCalibConfig& config)
	: m_mode(config.mode),
      m_enableRefinement(config.enableRefinement),
      m_rotationWeight(config.rotationWeight),
      m_translationWeight(config.translationWeight),
      m_maxIterations(config.maxIterations),
	  m_poseFile(config.poseFile),
	  m_imagesFolder(config.camConfig.imagesFolder),
	  m_boardSize(config.camConfig.boardSize),
	  m_squareSize(config.camConfig.squareSize)
{
	if (config.useExistIntrinsics) {
		m_intrinsics = config.intrinsics;
		m_distCoeffs = config.distCoeffs;
	}
	
	if (config.useExistBoardPose) {
		m_rvecsMat = config.rvecs;
		m_tvecsMat = config.tvecs;
		
		// 转换rvec/tvec到目标坐标系
		for (size_t i = 0; i < m_rvecsMat.size(); i++) {
			m_T_target2camVec.push_back(m_tvecsMat[i]);
			cv::Mat R;
			cv::Rodrigues(m_rvecsMat[i], R);
			m_R_target2camVec.push_back(R);

			cv::Mat H_target2cam;
			R_T2H(R, m_tvecsMat[i], H_target2cam);
			m_H_target2camVec.push_back(H_target2cam);
		}
	}
	
	// 初始化Charuco板（如果需要）
	if (config.camConfig.plateType == PlateType::Charuco) {
		try {
			m_charucoBoard = cv::aruco::CharucoBoard::create(
				config.camConfig.boardSize.width, config.camConfig.boardSize.height,
				config.camConfig.squareSize.width, config.camConfig.markerSize.width, 
				config.camConfig.dictionary);
		}
		catch (const std::exception& e) {
			LOG_OUTPUT(Error, e.what());
		}
	}
}

HandEyeCalibrator::~HandEyeCalibrator() {

}

void HandEyeCalibrator::setRobotPoses(const std::vector<cv::Mat>& poses, bool isRad) {
	// Clear previous data
	m_R_base2gripperVec.clear();
	m_T_base2gripperVec.clear();
	m_R_gripper2baseVec.clear();
	m_T_gripper2baseVec.clear();

	for (size_t i = 0; i < poses.size(); ++i) {
		cv::Mat poseMat = poses[i].clone();
		
		// Convert to radians if needed (assuming Euler angles in first 3 rotation components)
		// Note: poseToHomogeneousMatrix handles 6 or 7 elements.
		// If 6 elements (Euler), we might need conversion.
		// If 7 elements (Quaternion), usually no conversion needed (already normalized or not).
		// But the original code only converted if !isRad and it was Euler.
		// Let's assume if cols==6 it's Euler.
		
		if (!isRad && poseMat.cols == 6) {
			poseMat.at<float>(0, 3) *= CV_PI / 180.0f;
			poseMat.at<float>(0, 4) *= CV_PI / 180.0f;
			poseMat.at<float>(0, 5) *= CV_PI / 180.0f;
		}

		cv::Mat H_gripper2base = poseToHomogeneousMatrix(poseMat);
		if (H_gripper2base.empty()) {
			std::string msg = "Failed to convert pose to homogeneous matrix at index " + std::to_string(i);
			LOG_OUTPUT(Error, msg.c_str());
			continue;
		}
		cv::Mat R_gripper2base = H_gripper2base(cv::Rect(0, 0, 3, 3));
		cv::Mat T_gripper2base = H_gripper2base(cv::Rect(3, 0, 1, 3));
		m_R_gripper2baseVec.push_back(R_gripper2base.clone());
		m_T_gripper2baseVec.push_back(T_gripper2base.clone());

		cv::Mat H_base2gripper = inverseHomogeneous(H_gripper2base);
		cv::Mat R_base2gripper = H_base2gripper(cv::Rect(0, 0, 3, 3));
		cv::Mat T_base2gripper = H_base2gripper(cv::Rect(3, 0, 1, 3));
		m_R_base2gripperVec.push_back(R_base2gripper.clone());
		m_T_base2gripperVec.push_back(T_base2gripper.clone());
	}
	std::string msg = "Successfully set " + std::to_string(m_R_gripper2baseVec.size()) + " robot poses.";
	LOG_OUTPUT(Info, msg.c_str());
}

void HandEyeCalibrator::setTargetPoses(const std::vector<cv::Mat>& R_target2cam, const std::vector<cv::Mat>& T_target2cam) {
	if (R_target2cam.size() != T_target2cam.size()) {
		LOG_OUTPUT(Error, "R_target2cam and T_target2cam size mismatch");
		return;
	}
	m_R_target2camVec = R_target2cam;
	m_T_target2camVec = T_target2cam;
	
	// Also update H_target2camVec if needed
	m_H_target2camVec.clear();
	for(size_t i=0; i<m_R_target2camVec.size(); ++i) {
		cv::Mat H;
		R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H);
		m_H_target2camVec.push_back(H);
	}
	
	std::string msg = "Successfully set " + std::to_string(m_R_target2camVec.size()) + " target poses.";
	LOG_OUTPUT(Info, msg.c_str());
}


bool HandEyeCalibrator::checkVec() {
	// 1. 基础容器非空检查
	if (m_R_gripper2baseVec.empty() || m_T_gripper2baseVec.empty() ||
		m_R_base2gripperVec.empty() || m_T_base2gripperVec.empty() ||
		m_R_target2camVec.empty() || m_T_target2camVec.empty()) {
		std::string msg("Input transformation vectors cannot be empty");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}

	// 2. 检查数据量一致性
	const size_t n_poses = m_R_gripper2baseVec.size();

	// 检查gripper2base相关数据
	if (m_T_gripper2baseVec.size() != n_poses) {
		std::string msg("m_R_gripper2baseVec.size(" + std::to_string(n_poses) +
			") != m_T_gripper2baseVec.size(" + std::to_string(m_T_gripper2baseVec.size()) + ")");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}

	// 检查base2gripper相关数据
	if (m_R_base2gripperVec.size() != n_poses || m_T_base2gripperVec.size() != n_poses) {
		std::string msg("base2gripper vectors size mismath with gripper2base vectors");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}

	// 检查target2cam相关数据
	if (m_R_target2camVec.size() != n_poses || m_T_target2camVec.size() != n_poses) {
		std::string msg("target2cam vectors size mismatch with gripper2base vectors");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}

	// 3. 检查矩阵基本属性
	for (size_t i = 0; i < n_poses; ++i) {
		// 检查旋转矩阵
		if (!isRotationMatrix(m_R_gripper2baseVec[i])) {
			std::string msg("m_R_gripper2baseVec[" + std::to_string(i) + "]" + "not rotation matrix");
			LOG_OUTPUT(Error, msg.c_str());
			return false;
		}
		if (!isRotationMatrix(m_R_base2gripperVec[i])) {
			std::string msg("m_R_base2gripperVec[" + std::to_string(i) + "]" + "not rotation matrix");
			LOG_OUTPUT(Error, msg.c_str());
			return false;
		}
		if (!isRotationMatrix(m_R_target2camVec[i])) {
			std::string msg("m_R_target2cam[" + std::to_string(i) + "]" + "not rotation matrix");
			LOG_OUTPUT(Error, msg.c_str());
			return false;
		}

		// 检查平移向量
		auto check_translation = [i](const cv::Mat& t, const std::string& name) {
			if (t.rows != 3 || t.cols != 1) {
				std::string msg(name + " translation vector must be 3x1 at pose " + std::to_string(i));
				LOG_OUTPUT(Error, msg.c_str());
				return false;
			}
			return true;
		};

		if (!check_translation(m_T_gripper2baseVec[i], "gripper2base") ||
			!check_translation(m_T_base2gripperVec[i], "base2gripper") ||
			!check_translation(m_T_target2camVec[i], "target2cam")) {
			return false;
		}

		// 4. 检查gripper2base和base2gripper的互逆关系
		cv::Mat H_gripper2base, H_base2gripper;
		R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_gripper2base);
		R_T2H(m_R_base2gripperVec[i], m_T_base2gripperVec[i], H_base2gripper);
		cv::Mat shouldBeIdentity = H_gripper2base * H_base2gripper;
		cv::Mat I = cv::Mat::eye(4, 4, shouldBeIdentity.type());
		double norm = cv::norm(I, shouldBeIdentity);
		if (norm > 1e-6) {
			std::string msg("H_gripper2base and H_base2gripper mismatch");
			LOG_OUTPUT(Error, msg.c_str());
			return false;
		}
	}

	// 5. 检查最小数据量要求
	if (n_poses < 2) {
		std::string msg("Error: At least 2 poses are required for hand-eye calibration");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}
	return true;
}

bool HandEyeCalibrator::TvecTo31() {
	for (int i = 0; i < m_R_gripper2baseVec.size(); i++) {

		if (m_T_gripper2baseVec[i].size() != cv::Size(3, 1) && m_T_gripper2baseVec[i].size() != cv::Size(1, 3)) {
			LOG_OUTPUT(Error, "Translation matrix from actuator to base must be a 1x3 or 3x1 vector");
			return false;
		}
		if (m_T_target2camVec[i].size() != cv::Size(3, 1) && m_T_target2camVec[i].size() != cv::Size(1, 3)) {
			LOG_OUTPUT(Error, "Translation matrix from calibration plate to camera must be a 1x3 or 3x1 vector");
			return false;
		}
	}
	std::vector<cv::Mat> T_gripper2baseVec_31, T_target2camVec_31;
	for (const auto& T_gripper2base : m_T_gripper2baseVec) {
		if (T_gripper2base.rows == 1) {
			T_gripper2baseVec_31.push_back(T_gripper2base.t());
		}
		else {
			T_gripper2baseVec_31.push_back(T_gripper2base);
		}
	}
	for (const auto& T_target2cam : m_T_target2camVec) {
		if (T_target2cam.rows == 1) {
			T_target2camVec_31.push_back(T_target2cam.t());
		}
		else {
			T_target2camVec_31.push_back(T_target2cam);
		}
	}
	m_T_gripper2baseVec = T_gripper2baseVec_31;
	m_T_target2camVec = T_target2camVec_31;
	return true;
}



bool HandEyeCalibrator::calibrateWithAruco(const std::string& folderPath,
	const float markerLength_mm, const int dictionaryId,
	cv::HandEyeCalibrationMethod method) {
	if (!checkVec()) {
		LOG_OUTPUT(Error, "Hand-eye calibration data check failed");
		return false;
	}
	if (m_arucoImageVec.empty()) {
		LOG_OUTPUT(Error, "Aruco image count is zero");
		return false;
	}
	if (m_arucoImageVec.size() < 3) {
		LOG_OUTPUT(Error, "Aruco image count is too low for hand-eye calibration");
		return false;
	}
	if (m_arucoImageVec.size() != m_R_gripper2baseVec.size()) {
		LOG_OUTPUT(Error, "Aruco image count does not match robot pose count");
		return false;
	}
	if (m_arucoImageVec.size() < 10) {
		LOG_OUTPUT(Warn, "Aruco image and pose data count is less than 10, may affect calibration accuracy");
	}
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictionaryId);
	for (size_t i = 0; i < m_arucoImageVec.size(); i++) {
		std::vector<int> idVec;
		std::vector<std::vector<cv::Point2f>> cornersVec;
		cv::aruco::detectMarkers(m_arucoImageVec[i], dictionary, cornersVec, idVec);
		cv::Mat R_target2cam, T_target2cam, rvec;
		if (!idVec.empty()) {
			std::vector<cv::Vec3f> rvecs, tvecs;
			cv::aruco::estimatePoseSingleMarkers(cornersVec, markerLength_mm,
				m_intrinsics, m_distCoeffs, rvecs, tvecs);
			// 使用第一个检测到的标记的位姿
			rvec = rvecs[0];
			T_target2cam = cv::Mat(tvecs[0]);
			cv::Rodrigues(rvec, R_target2cam);
			m_R_target2camVec.push_back(R_target2cam);
			m_T_target2camVec.push_back(T_target2cam);
		}
		else {
			std::string msg("Warning: No markers detected in image " + std::to_string(i));
			LOG_OUTPUT(Warn, msg.c_str());
			continue;
		}
	}
	if (m_mode == EYE_IN_HAND) {	// 
		cv::calibrateHandEye(m_R_gripper2baseVec, m_T_gripper2baseVec,
			m_R_target2camVec, m_T_target2camVec, m_R_cam2gripper, m_T_cam2gripper,
			method);
		if (!R_T2H(m_R_cam2gripper, m_T_cam2gripper, m_H_cam2gripper)) {
			LOG_OUTPUT(Error, "m_R_cam2gripper calculation failed");
			return false;
		}
	}
	else if (m_mode == EYE_TO_HAND) {
		cv::calibrateHandEye(m_R_base2gripperVec, m_T_base2gripperVec,
			m_R_target2camVec, m_T_target2camVec, m_R_cam2base, m_T_cam2base,
			method);
		if (!R_T2H(m_R_cam2base, m_T_cam2base, m_H_cam2base)) {
			LOG_OUTPUT(Error, "m_R_base2cam calculation failed");
			return false;
		}
	}
	else {
		LOG_OUTPUT(Error, "Unknown calibration mode");
		return false;
	}
	refineCalibration();
	return true;
}

// HandEyeCalibrator.cpp 完善calibrate函数
bool HandEyeCalibrator::calibrate(cv::HandEyeCalibrationMethod method) {

	if (!checkVec()) {
		std::string msg("hand eye calibration data check failed");
		LOG_OUTPUT(Error, msg.c_str());
		return false;
	}

	if (m_mode == EYE_IN_HAND) {
		cv::calibrateHandEye(
			m_R_gripper2baseVec,
			m_T_gripper2baseVec,
			//m_R_base2gripperVec,
			//m_T_base2gripperVec,	// 确认位姿读取的数据是base2gripper还是gripper2base
			m_R_target2camVec,
			m_T_target2camVec,
			m_R_cam2gripper,
			m_T_cam2gripper,
			method);
		R_T2H(m_R_cam2gripper, m_T_cam2gripper, m_H_cam2gripper);
	}
	else if (m_mode == EYE_TO_HAND) {
		cv::calibrateHandEye(
			m_R_base2gripperVec,
			m_T_base2gripperVec,	// 确认位姿读取的数据是base2gripper还是gripper2base
			//m_R_gripper2baseVec,
			//m_T_gripper2baseVec,
			m_R_target2camVec,
			m_T_target2camVec,
			m_R_cam2base,
			m_T_cam2base,
			method);
		R_T2H(m_R_cam2base, m_T_cam2base, m_H_cam2base);
	}
	refineCalibration();
	return true;
}

bool HandEyeCalibrator::calcTarget2baseVec() {
	// 新增输入检查
	if (m_mode == EYE_IN_HAND) {
		if (m_R_cam2gripper.empty() || m_T_cam2gripper.empty()) {
			std::cerr << "ERROR: Eye-in-hand calibration results not initialized" << std::endl;
			return false;
		}
	}
	else if (m_mode == EYE_TO_HAND) {
		if (m_R_cam2base.empty() || m_T_cam2base.empty()) {
			std::cerr << "ERROR: Eye-to-hand calibration results not initialized" << std::endl;
			return false;
		}
	}

	if (m_R_gripper2baseVec.size() != m_R_target2camVec.size()) {
		std::cerr << "ERROR: Pose count mismatch: gripper2base="
			<< m_R_gripper2baseVec.size()
			<< ", target2cam=" << m_R_target2camVec.size() << std::endl;
		return false;
	}

	// 清空结果容器
	m_R_target2baseVec.clear();
	m_T_target2baseVec.clear();

	for (size_t i = 0; i < m_R_gripper2baseVec.size(); ++i) {
		// 类型转换和输入验证
		if (m_R_gripper2baseVec[i].empty() || m_R_target2camVec[i].empty()) {
			std::cerr << "WARNING: Skip invalid pose at index " << i << std::endl;
			continue;
		}

		cv::Mat R_target2cam, T_target2cam;

		m_R_target2camVec[i].convertTo(R_target2cam, CV_64F);
		m_T_target2camVec[i].convertTo(T_target2cam, CV_64F);

		// 计算目标到基座的变换
		cv::Mat R_target2base, T_target2base;
		if (m_mode == EYE_IN_HAND) {
			cv::Mat R_gripper2base, T_gripper2base;
			m_R_gripper2baseVec[i].convertTo(R_gripper2base, CV_64F);
			m_T_gripper2baseVec[i].convertTo(T_gripper2base, CV_64F);

			cv::Mat R_cam2gripper, T_cam2gripper;
			m_R_cam2gripper.convertTo(R_cam2gripper, CV_64F);
			m_T_cam2gripper.convertTo(T_cam2gripper, CV_64F);

			// target2base = gripper2base * cam2gripper * target2cam
			R_target2base = R_gripper2base * R_cam2gripper * R_target2cam;
			T_target2base = R_gripper2base * (R_cam2gripper * T_target2cam + T_cam2gripper)
				+ T_gripper2base;
		}
		else { // EYE_TO_HAND
			cv::Mat R_cam2base, T_cam2base;
			m_R_cam2base.convertTo(R_cam2base, CV_64F);
			m_T_cam2base.convertTo(T_cam2base, CV_64F);

			// target2base = cam2base * target2cam
			R_target2base = R_cam2base * R_target2cam;
			T_target2base = R_cam2base * T_target2cam + T_cam2base;
		}

		// 结果验证
		if (!isRotationMatrix(R_target2base)) {
			std::cerr << "ERROR: Invalid rotation matrix at pose " << i << std::endl;
			return false;
		}

		// 检查平移向量是否合理（单位：mm）
		double tx = T_target2base.at<double>(0);
		double ty = T_target2base.at<double>(1);
		double tz = T_target2base.at<double>(2);
		if (abs(tx) > 10000 || abs(ty) > 10000 || abs(tz) > 10000) { // 假设工作空间在10米范围内
			std::cerr << "WARNING: Unrealistic translation at pose " << i
				<< ": [" << tx << ", " << ty << ", " << tz << "]" << std::endl;
		}

		m_R_target2baseVec.push_back(R_target2base);
		m_T_target2baseVec.push_back(T_target2base);
	}

	return !m_R_target2baseVec.empty();
}

cv::Mat HandEyeCalibrator::getResult() {
	if (m_mode == EYE_IN_HAND) {
		return m_H_cam2gripper;
	}
	else if (m_mode == EYE_TO_HAND) {
		return m_H_cam2base;
	}
	return cv::Mat();
}

bool HandEyeCalibrator::calcCornersInTargetVec() {
	if (m_mode == EYE_IN_HAND) {
		for (size_t i = 0; i < m_images.size(); ++i) {
			cv::Mat H_gripper2base;
			R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_gripper2base);
			cv::Mat H_target2cam;
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam);
			cv::Mat H_target2base = H_gripper2base * m_H_cam2gripper * H_target2cam;
			std::vector<cv::Point3f> points_base;
			for (size_t j = 0; j < m_imageCornersVec[i].size(); j++) {
				float Xt = m_objectCornersVec[i][j].x;
				float Yt = m_objectCornersVec[i][j].y;
				float Zt = m_objectCornersVec[i][j].z;
				cv::Mat point_target = (cv::Mat_<float>(4, 1) << Xt, Yt, Zt, 1.0);
				cv::Mat point_base = H_target2base * point_target;
				cv::Point3f point_base_3f = cv::Point3f(point_base.at<float>(0, 0) / point_base.at<float>(3, 0),
					point_base.at<float>(1, 0) / point_base.at<float>(3, 0),
					point_base.at<float>(2, 0) / point_base.at<float>(3, 0));
				points_base.push_back(point_base_3f);
			}
			m_cornersInBaseVec.push_back(points_base);
		}
	}
	else if (m_mode == EYE_TO_HAND) {
		for (size_t i = 0; i < m_images.size(); i++) {
			cv::Mat H_target2cam;
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam);
			cv::Mat H_cam2base;
			R_T2H(m_R_cam2base, m_T_cam2base, H_cam2base);
			cv::Mat H_target2base = H_cam2base * H_target2cam;
			std::vector<cv::Point3f> points_base;
			for (size_t j = 0; j < m_imageCornersVec[i].size(); j++) {
				float Xt = m_objectCornersVec[i][j].x;
				float Yt = m_objectCornersVec[i][j].y;
				float Zt = m_objectCornersVec[i][j].z;
				cv::Mat point_target = (cv::Mat_<float>(4, 1) << Xt, Yt, Zt, 1.0);
				cv::Mat point_base = H_target2base * point_target;
				cv::Point3f point_base_3f = cv::Point3f(point_base.at<float>(0, 0) / point_base.at<float>(3, 0),
					point_base.at<float>(1, 0) / point_base.at<float>(3, 0),
					point_base.at<float>(2, 0) / point_base.at<float>(3, 0));
				points_base.push_back(point_base_3f);
			}
			m_cornersInBaseVec.push_back(points_base);
		}
	}
	return true;
}

// 新增辅助函数：计算变换的逆
inline bool HandEyeCalibrator::invertTransform(const cv::Mat& R, const cv::Mat& T,
	cv::Mat& R_inv, cv::Mat& T_inv) {
	R_inv = R.t();  // 旋转矩阵的逆等于其转置
	T_inv = -R_inv * T;
	return true;
}

bool HandEyeCalibrator::calReprojectionError() {
	// 1. Input validation
	if (m_R_target2camVec.size() != m_R_base2gripperVec.size()) {
		LOG_OUTPUT(Error, "Pose data and camera data size mismatch.");
		return false;
	}
	if (m_R_target2camVec.size() < 2) {
		LOG_OUTPUT(Error, "Need at least 2 poses to calculate consistency error.");
		return false;
	}

	double total_translation_error = 0.0;
	double total_rotation_error = 0.0;
	int num_comparisons = m_R_target2camVec.size() - 1;

	if (m_mode == EYE_IN_HAND) {
		std::cout << "\n======== Pose Consistency Error Calculation (EYE_IN_HAND) ========" << std::endl;
		std::cout << "Verifying consistency of H_target2base..." << std::endl;

		if (m_H_cam2gripper.empty()) {
			LOG_OUTPUT(Error, "Hand-eye calibration for EYE_IN_HAND not performed.");
			return false;
		}
		// Calculate reference pose from the first frame
		cv::Mat H_gripper2base_0, H_target2cam_0;
		R_T2H(m_R_base2gripperVec[0], m_T_gripper2baseVec[0], H_gripper2base_0);
		R_T2H(m_R_target2camVec[0], m_T_target2camVec[0], H_target2cam_0);
		cv::Mat H_target2base_ref = H_gripper2base_0 * m_H_cam2gripper * H_target2cam_0;

		for (size_t i = 1; i < m_R_target2camVec.size(); ++i) {
			cv::Mat H_gripper2base_i, H_target2cam_i;
			R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_gripper2base_i);
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam_i);
			cv::Mat H_target2base_i = H_gripper2base_i * m_H_cam2gripper * H_target2cam_i;

			cv::Mat H_diff = H_target2base_ref.inv() * H_target2base_i;
			cv::Mat R_diff, T_diff;
			H2R_T(H_diff, R_diff, T_diff);

			double trans_error = cv::norm(T_diff);
			total_translation_error += trans_error;
			double angle_rad = acos(std::min(1.0, std::max(-1.0, (cv::trace(R_diff)[0] - 1.0) / 2.0)));
			double rot_error_deg = angle_rad * 180.0 / CV_PI;
			total_rotation_error += rot_error_deg;
			std::cout << "Pose " << i << " vs Pose 0: Translation Error = " << trans_error
				<< " mm, Rotation Error = " << rot_error_deg << " deg" << std::endl;
		}
	}
	else { // EYE_TO_HAND
		std::cout << "\n======== Pose Consistency Error Calculation (EYE_TO_HAND) ========" << std::endl;
		std::cout << "Verifying consistency of H_target2gripper..." << std::endl;

		if (m_H_cam2base.empty()) {
			LOG_OUTPUT(Error, "Hand-eye calibration for EYE_TO_HAND not performed.");
			return false;
		}
		// Calculate reference pose from the first frame
		cv::Mat H_base2gripper_0, H_target2cam_0;
		R_T2H(m_R_base2gripperVec[0], m_T_base2gripperVec[0], H_base2gripper_0);
		R_T2H(m_R_target2camVec[0], m_T_target2camVec[0], H_target2cam_0);
		cv::Mat H_target2gripper_ref = H_base2gripper_0 * m_H_cam2base * H_target2cam_0;

		for (size_t i = 1; i < m_R_target2camVec.size(); ++i) {
			cv::Mat H_base2gripper_i, H_target2cam_i;
			R_T2H(m_R_base2gripperVec[i], m_T_base2gripperVec[i], H_base2gripper_i);
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam_i);
			cv::Mat H_target2gripper_i = H_base2gripper_i * m_H_cam2base * H_target2cam_i;

			cv::Mat H_diff = H_target2gripper_ref.inv() * H_target2gripper_i;
			cv::Mat R_diff, T_diff;
			H2R_T(H_diff, R_diff, T_diff);

			double trans_error = cv::norm(T_diff);
			total_translation_error += trans_error;
			double angle_rad = acos(std::min(1.0, std::max(-1.0, (cv::trace(R_diff)[0] - 1.0) / 2.0)));
			double rot_error_deg = angle_rad * 180.0 / CV_PI;
			total_rotation_error += rot_error_deg;
			std::cout << "Pose " << i << " vs Pose 0: Translation Error = " << trans_error
				<< " mm, Rotation Error = " << rot_error_deg << " deg" << std::endl;
		}
	}

	// Final average errors
	double avg_translation_error = total_translation_error / num_comparisons;
	double avg_rotation_error = total_rotation_error / num_comparisons;

	m_reprojectionError = avg_translation_error;

	std::cout << "\n======== Average Pose Consistency Error ========" << std::endl;
	std::cout << "Average Translation Error: " << avg_translation_error << " mm" << std::endl;
	std::cout << "Average Rotation Error: " << avg_rotation_error << " deg" << std::endl;
	std::cout << "================================================" << std::endl;

	return true;
}

bool HandEyeCalibrator::calReprojectionErrorMean()
{
	// 1. Input validation
	if (m_R_target2camVec.size() != m_R_base2gripperVec.size()) {
		LOG_OUTPUT(Error, "Pose data and camera data size mismatch.");
		return false;
	}
	if (m_R_target2camVec.empty()) {
		LOG_OUTPUT(Error, "No poses available to calculate error.");
		return false;
	}

	std::vector<cv::Mat> H_target_poses;
	H_target_poses.reserve(m_R_target2camVec.size());

	// 2. Compute all target poses in the reference frame
	if (m_mode == EYE_IN_HAND) {
		if (m_H_cam2gripper.empty()) {
			LOG_OUTPUT(Error, "Hand-eye calibration for EYE_IN_HAND not performed.");
			return false;
		}
		for (size_t i = 0; i < m_R_target2camVec.size(); ++i) {
			cv::Mat H_gripper2base_i, H_target2cam_i;
			R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_gripper2base_i);
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam_i);
			H_target_poses.push_back(H_gripper2base_i * m_H_cam2gripper * H_target2cam_i);
		}
	}
	else { // EYE_TO_HAND
		if (m_H_cam2base.empty()) {
			LOG_OUTPUT(Error, "Hand-eye calibration for EYE_TO_HAND not performed.");
			return false;
		}
		for (size_t i = 0; i < m_R_target2camVec.size(); ++i) {
			cv::Mat H_base2gripper_i, H_target2cam_i;
			R_T2H(m_R_base2gripperVec[i], m_T_base2gripperVec[i], H_base2gripper_i);
			R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_target2cam_i);
			H_target_poses.push_back(H_base2gripper_i * m_H_cam2base * H_target2cam_i);
		}
	}

	// 3. Calculate the mean pose
	// a) Mean of translations
	cv::Vec3d mean_translation(0, 0, 0);
	// b) Mean of rotations via averaging quaternions
	cv::Mat Q_acc = cv::Mat::zeros(4, 4, CV_64F);

	for (const auto& H : H_target_poses) {
		cv::Mat R, T;
		H2R_T(H, R, T);
		mean_translation += cv::Vec3d(T);

		// Convert rotation matrix to quaternion
		cv::Mat R_double;
		R.convertTo(R_double, CV_64F);
		double w = 0.5 * std::sqrt(1.0 + R_double.at<double>(0, 0) + R_double.at<double>(1, 1) + R_double.at<double>(2, 2));
		double x = (R_double.at<double>(2, 1) - R_double.at<double>(1, 2)) / (4.0 * w);
		double y = (R_double.at<double>(0, 2) - R_double.at<double>(2, 0)) / (4.0 * w);
		double z = (R_double.at<double>(1, 0) - R_double.at<double>(0, 1)) / (4.0 * w);
		cv::Mat q = (cv::Mat_<double>(4, 1) << w, x, y, z);
		Q_acc += q * q.t();
	}
	mean_translation /= static_cast<double>(H_target_poses.size());

	// The mean quaternion is the eigenvector corresponding to the largest eigenvalue of Q_acc
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(Q_acc, eigenvalues, eigenvectors);
	cv::Vec4d mean_q = eigenvectors.row(0);

	// Convert mean quaternion back to rotation matrix
	cv::Mat R_mean;
	quaternionToRotationMatrix(mean_q, R_mean);

	// Assemble the mean homogeneous matrix
	cv::Mat H_mean_ref;
	cv::Mat T_mean = (cv::Mat_<double>(3, 1) << mean_translation[0], mean_translation[1], mean_translation[2]);
	R_T2H(R_mean, T_mean, H_mean_ref);


	// 4. Calculate errors by comparing each pose to the mean pose
	double total_translation_error = 0.0;
	double total_rotation_error = 0.0;

	std::cout << "\n======== Pose Consistency Error Calculation (Mean as Reference) ========" << std::endl;
	if (m_mode == EYE_IN_HAND) std::cout << "Verifying consistency of H_target2base..." << std::endl;
	else std::cout << "Verifying consistency of H_target2gripper..." << std::endl;

	for (size_t i = 0; i < H_target_poses.size(); ++i) {
		cv::Mat H_diff = H_mean_ref.inv() * H_target_poses[i];
		cv::Mat R_diff, T_diff;
		H2R_T(H_diff, R_diff, T_diff);

		double trans_error = cv::norm(T_diff);
		total_translation_error += trans_error;

		double angle_rad = acos(std::min(1.0, std::max(-1.0, (cv::trace(R_diff)[0] - 1.0) / 2.0)));
		double rot_error_deg = angle_rad * 180.0 / CV_PI;
		total_rotation_error += rot_error_deg;

		//std::cout << "Pose " << i << " vs Mean: Translation Error = " << trans_error
		//	<< " mm, Rotation Error = " << rot_error_deg << " deg" << std::endl;
        LOG_OUTPUT(Info, ("Pose " + std::to_string(i) + " vs Mean: Translation Error = " + std::to_string(trans_error) + 
            " mm, Rotation Error = " + std::to_string(rot_error_deg) + " deg").c_str());
	}

	// 5. Final average errors
	double avg_translation_error = total_translation_error / H_target_poses.size();
	double avg_rotation_error = total_rotation_error / H_target_poses.size();

	m_reprojectionError = avg_translation_error;
    m_rotationError = avg_rotation_error;


	LOG_OUTPUT(Info, "======== Average Pose Consistency Error (vs Mean) ========");
	LOG_OUTPUT(Info, ("Average Translation Error: " + std::to_string(avg_translation_error) + " mm").c_str());
	LOG_OUTPUT(Info, ("Average Rotation Error: " + std::to_string(avg_rotation_error) + " deg").c_str());

    // 6. Calculate AX=XB Residuals (Relative Motion Error)
    double total_axxb_trans_error = 0.0;
    double total_axxb_rot_error = 0.0;
    int pair_count = 0;

    LOG_OUTPUT(Info, "======== AX=XB Residual Error (Relative Motion) ========");
    
    cv::Mat X;
    if (m_mode == EYE_IN_HAND) {
        X = m_H_cam2gripper;
    } else {
        X = m_H_cam2base;
    }
    
    if (X.empty()) {
         LOG_OUTPUT(Error, "Hand-eye matrix is empty, skipping residual calculation.");
         return true;
    }

    cv::Mat X_inv = X.inv();

    // Prepare robot poses and cam poses in cv::Mat format
    std::vector<cv::Mat> H_robot_poses;
    std::vector<cv::Mat> H_cam_poses;
    
    // Fill H_robot_poses and H_cam_poses based on input vectors
    if (m_R_gripper2baseVec.size() == m_R_target2camVec.size()) { // Assume aligned
        for (size_t i = 0; i < m_R_gripper2baseVec.size(); ++i) {
             cv::Mat H_robot, H_cam;
             if (m_mode == EYE_IN_HAND) {
                R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_robot);
             } else {
                R_T2H(m_R_base2gripperVec[i], m_T_base2gripperVec[i], H_robot);
             }
             R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_cam);
             H_robot_poses.push_back(H_robot);
             H_cam_poses.push_back(H_cam);
        }
    }


    for (size_t i = 0; i < H_robot_poses.size(); ++i) {
        for (size_t j = i + 1; j < H_robot_poses.size(); ++j) {
            // A_ij: Relative motion of Robot Hand
            cv::Mat A;
            if (m_mode == EYE_IN_HAND) {
                 // A = H_robot_inv_j * H_robot_i (Standard mapping for eye-in-hand logic A*X = X*B)
                 // Note: Depends on solver definition. Assuming standard: T_g_ij * X = X * T_c_ij
                 // T_g_ij = T_base_hand_j.inv() * T_base_hand_i
                 A = H_robot_poses[j].inv() * H_robot_poses[i];
            } else {
                 // Eye-to-Hand
                 A = H_robot_poses[j] * H_robot_poses[i].inv();
            }

            cv::Mat A_ij = A; 
            
            // B_ij: Relative motion of Camera (Target to Camera)
            // T_c_ij = T_cam_target_j * T_cam_target_i.inv()  <-- WRONG for target-to-camera matrix
            // Standard OpenCV: B = T_target_cam_j * T_target_cam_i.inv() is wrong.
            // Correct relative cam motion relating to grid: T_c_ij = T_cam_i * T_cam_j.inv() if using T_cam_to_world
            
            // Let's use the definition: A * X = X * B
            // Eye-in-Hand:
            // T_hand(j->i) * T_cam_hand = T_cam_hand * T_cam(j->i)
           
            // The logic: 
            // A = H_robot_j.inv() * H_robot_i (Motion from j to i in robot frame)
            // B = H_cam_j * H_cam_i.inv()    (Motion from j to i in camera frame)

            cv::Mat B_ij = H_cam_poses[j] * H_cam_poses[i].inv();

            // Note: The definition of A and B depends strictly on whether we define motion I->J or J->I
            // and whether it's eye-in-hand or eye-to-hand. 
            // The safest check is: Does A * X predict X * B?
            
            cv::Mat LHS = A_ij * X;
            cv::Mat RHS = X * B_ij;
            
            // Calculate error between LHS and RHS
            cv::Mat Diff = LHS.inv() * RHS; // Should be Identity if perfect
            
            cv::Mat R_diff, T_diff;
            H2R_T(Diff, R_diff, T_diff);

            double trans_err = cv::norm(T_diff);
            double angle_rad = acos(std::min(1.0, std::max(-1.0, (cv::trace(R_diff)[0] - 1.0) / 2.0)));
            double rot_err = angle_rad * 180.0 / CV_PI;

            total_axxb_trans_error += trans_err;
            total_axxb_rot_error += rot_err;
            pair_count++;
        }
    }

    if (pair_count > 0) {
        double avg_axxb_trans = total_axxb_trans_error / pair_count;
        double avg_axxb_rot = total_axxb_rot_error / pair_count;
        LOG_OUTPUT(Info, ("Average AX=XB Translation Residual: " + std::to_string(avg_axxb_trans) + " mm").c_str());
        LOG_OUTPUT(Info, ("Average AX=XB Rotation Residual: " + std::to_string(avg_axxb_rot) + " deg").c_str());
    }

	return true;
}

double HandEyeCalibrator::getReprojectionError() const {
	return m_reprojectionError;
}

void HandEyeCalibrator::refineCalibration() {
    if (!m_enableRefinement) {
        LOG_OUTPUT(Info, "Non-linear Refinement is DISABLED by user.");
        return;
    }
    std::string logMsg = "Starting Non-linear Refinement. Weights -> Rotation: " + 
                         std::to_string(m_rotationWeight) + ", Translation: " + std::to_string(m_translationWeight);
    LOG_OUTPUT(Info, logMsg.c_str());

    // 1. 准备初始猜测 (Initial Guess)
    // 假设已经运行过 cv::calibrateHandEye，结果存在 m_R_cam2gripper/m_T_cam2gripper 中
    
    cv::Mat X_init_R, X_init_T;
    if (m_mode == EYE_IN_HAND) {
        X_init_R = m_R_cam2gripper;
        X_init_T = m_T_cam2gripper;
    } else {
        X_init_R = m_R_cam2base;
        X_init_T = m_T_cam2base;
    }
    
    // 如果没有初始结果，尝试重新构建
    if (X_init_R.empty() || X_init_T.empty()) {
        if(m_H_cam2gripper.empty() && m_H_cam2base.empty()) {
             LOG_OUTPUT(Error, "Initial calibration not found. Run standard calibration first.");
             return;
        }
        // 如果有H矩阵但没有分离的R/T，尝试分离
        if (m_mode == EYE_IN_HAND && !m_H_cam2gripper.empty()) {
             H2R_T(m_H_cam2gripper, X_init_R, X_init_T);
        } else if (!m_H_cam2base.empty()) {
             H2R_T(m_H_cam2base, X_init_R, X_init_T);
        }
    }
    
    if (X_init_R.empty() || X_init_T.empty()) {
        LOG_OUTPUT(Error, "Initial calibration data invalid.");
        return;
    }

    // 2. 估计辅助变量 Y (Initial Guess for Y)
    // Y 是 Target 到 Base (或 Target 到 Gripper) 的相对固定位姿
    // 取第一帧计算出的 Y0 作为初始值
    cv::Mat Y_init = cv::Mat::eye(4, 4, CV_64F);
    {
        cv::Mat X_mat = cv::Mat::eye(4, 4, CV_64F);
        X_init_R.copyTo(X_mat(cv::Rect(0,0,3,3)));
        X_init_T.copyTo(X_mat(cv::Rect(3,0,1,3)));

        // 取第0帧
        cv::Mat H_robot_0, H_target_0;
        if (m_R_target2camVec.empty()) {
            LOG_OUTPUT(Error, "Target to Cam vector empty.");
            return;
        }

        if (m_mode == EYE_IN_HAND) {
            if(m_R_gripper2baseVec.empty()) return;
            R_T2H(m_R_gripper2baseVec[0], m_T_gripper2baseVec[0], H_robot_0);
            R_T2H(m_R_target2camVec[0], m_T_target2camVec[0], H_target_0);
            // Y = T_grip_base * X * T_target_cam
            Y_init = H_robot_0 * X_mat * H_target_0;
        } else {
             if(m_R_base2gripperVec.empty()) return;
             R_T2H(m_R_base2gripperVec[0], m_T_base2gripperVec[0], H_robot_0);
             R_T2H(m_R_target2camVec[0], m_T_target2camVec[0], H_target_0);
             // EYE_TO_HAND: Y = inv(Robot) * X * Target
             Y_init = H_robot_0.inv() * X_mat * H_target_0;
        }
    }

    // 3. 构建参数向量 Param (12x1)
    cv::Mat X_rvec, Y_rvec;
    cv::Rodrigues(X_init_R, X_rvec);
    cv::Rodrigues(Y_init(cv::Rect(0,0,3,3)), Y_rvec);
    
    cv::Mat param(12, 1, CV_64F);
    X_rvec.copyTo(param.rowRange(0, 3));
    X_init_T.copyTo(param.rowRange(3, 6));
    Y_rvec.copyTo(param.rowRange(6, 9));
    Y_init(cv::Rect(3,0,1,3)).copyTo(param.rowRange(9, 12));

    // 4. 使用 LMSolver 进行优化
    try {
        // 数据准备
        std::vector<cv::Mat> robot_poses_H, target_poses_H;
        for(size_t i=0; i<m_R_target2camVec.size(); ++i) {
            cv::Mat H_t;
            R_T2H(m_R_target2camVec[i], m_T_target2camVec[i], H_t);
            target_poses_H.push_back(H_t);

            cv::Mat H_r;
            if (m_mode == EYE_IN_HAND) 
                R_T2H(m_R_gripper2baseVec[i], m_T_gripper2baseVec[i], H_r);
            else 
                R_T2H(m_R_base2gripperVec[i], m_T_base2gripperVec[i], H_r);
        robot_poses_H.push_back(H_r);
    }

    cv::Ptr<HandEyeRefinementCallback> cb = 
        cv::makePtr<HandEyeRefinementCallback>(robot_poses_H, target_poses_H, (int)m_mode, m_rotationWeight, m_translationWeight);
    
    // 创建求解器，最大迭代100次，误差阈值设定
    cv::Ptr<cv::LMSolver> solver = cv::LMSolver::create(cb, m_maxIterations);        // 运行优化
        int iter = solver->run(param);
        
        LOG_OUTPUT(Info, ("Optimization finished in " + std::to_string(iter) + " iterations.").c_str());

        // 5. 更新结果
        cv::Mat final_rvec_X = param.rowRange(0, 3);
        cv::Mat final_tvec_X = param.rowRange(3, 6);
        cv::Mat final_R_X;
        cv::Rodrigues(final_rvec_X, final_R_X);

        if (m_mode == EYE_IN_HAND) {
            m_R_cam2gripper = final_R_X.clone();
            m_T_cam2gripper = final_tvec_X.clone();
            R_T2H(m_R_cam2gripper, m_T_cam2gripper, m_H_cam2gripper);
        } else {
            m_R_cam2base = final_R_X.clone();
            m_T_cam2base = final_tvec_X.clone();
            R_T2H(m_R_cam2base, m_T_cam2base, m_H_cam2base);
        }
        
        LOG_OUTPUT(Info, "Non-linear Refinement Completed.");
    } catch (const cv::Exception& e) {
         LOG_OUTPUT(Error, ("OpenCV Refinement Error: " + e.msg).c_str());
    } catch (const std::exception& e) {
         LOG_OUTPUT(Error, ("Std Refinement Error: " + std::string(e.what())).c_str());
    } catch (...) {
         LOG_OUTPUT(Error, "Unknown Refinement Error");
    }
}

