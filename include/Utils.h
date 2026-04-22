#ifndef __ULTILS_H__
#define __ULTILS_H__

#include "CalibrationConfig.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "json.hpp"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cmath>


//#define _ALGRAPH

#ifndef LOG_OUTPUT
#ifdef _ALGRAPH
#include <modLogWraper.h>
#define LOG_OUTPUT(level, msg) LogAlgoRun##level(msg)
#else
#define LOG_OUTPUT(level, msg) std::cout << #level << ": " << msg << std::endl
#endif
#endif

bool CALIBRATION_API R_T2H(const cv::Mat& R, const cv::Mat& T, cv::Mat& H_out);

bool CALIBRATION_API H2R_T(const cv::Mat& H, cv::Mat& R, cv::Mat& T);

bool CALIBRATION_API isRotationMatrix(const cv::Mat& R);

//bool isRotationMatrix(const Eigen::Matrix3d& R);

bool CALIBRATION_API eulerToRotationMatrix(const cv::Vec3d& euler_angles,
	cv::Mat& R, const std::string& rotation_order = "ZYX", int dtype = CV_64F);

//bool eulerToRotationMatrix(const Eigen::Vector3d& euler_angle,
//    Eigen::Matrix3d& R, const std::string& rotation_order = "ZYX");

bool CALIBRATION_API quaternionToRotationMatrix(const cv::Vec4d& q, cv::Mat& R_output, int dtype = CV_64F);

cv::Mat CALIBRATION_API poseToHomogeneousMatrix(
	const cv::Mat& pose_data, const std::string& rotation_order = "ZYX", int dtype = CV_64F);

cv::Mat CALIBRATION_API inverseHomogeneous(const cv::Mat& H);

bool CALIBRATION_API eulerToRotationMatrix(const cv::Mat& euler_angles,
	cv::Mat& R, const std::string& rotation_order = "ZYX");
bool CALIBRATION_API poseToRT_deg(const cv::Mat& pose, cv::Mat& R, cv::Mat& t);
bool CALIBRATION_API poseToRT_rad(const cv::Mat& pose, cv::Mat& R, cv::Mat& t);

// Save cv::Mat to project root's "output" directory as JSON.
// Only file name is required. Default key is "T_base_camera".
bool CALIBRATION_API saveMatrixToJsonInOutput(
	const cv::Mat& mat,
	const std::string& fileName = "T_base_camera.json",
	const std::string& key = "T_base_camera",
	int floatPrecision = 6);

/**
 * @brief Load robot poses from a text file.
 * @param filePath Path to the text file containing poses.
 * @return A vector of cv::Mat, where each Mat is a 1xN pose vector (usually 1x6 or 1x7).
 */
CALIBRATION_API std::vector<cv::Mat> LoadRobotPoses(const std::string& filePath);

namespace Utils {
	bool naturalCompare(const std::string& a, const std::string& b);
	/**
	 * @brief read csv file which only contains double type data
	 * @param filePath file path of the csv file
	 */
}
#endif