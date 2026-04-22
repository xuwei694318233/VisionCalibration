#ifndef __CAMERA_CALIBRATOR_H__
#define __CAMERA_CALIBRATOR_H__

#include <CalibrationConfig.h>
#include <Utils.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>


#include <stdexcept>
#include <future>
#include <atomic>
#include <fstream>

// 定义标定板类型
enum class PlateType {
	Chessboard,
	Charuco,
	Aruco,
	AprilTag
};

// 相机标定配置结构体
struct CameraCalibConfig {
	PlateType plateType = PlateType::Chessboard;
	cv::Size boardSize;             // 角点数 (cols, rows)
	cv::Size2f squareSize;          // 方格物理尺寸 (mm)
	cv::Size2f markerSize;          // Charuco 标记尺寸 (mm)
	float markerLength = 0.0f;      // Aruco 标记长度 (mm)
	cv::Ptr<cv::aruco::Dictionary> dictionary; // 字典
	std::string imagesFolder;       // 图片文件夹路径
};

struct CalibrateResult {
	cv::Mat intrinsics;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> tvecsMat;
	std::vector<cv::Mat> rvecsMat;
	std::vector<float> errForImage;
	std::vector<cv::Mat> draws;
	std::vector<std::vector<std::vector<cv::Point2f>>> markerCornersVec;
	std::vector<std::vector<cv::Point2f>> chessCornersVec;
};


class CALIBRATION_API CameraCalibrator {
public:
	CameraCalibrator() = default;

	// 新增：基于配置结构体的构造函数
	explicit CameraCalibrator(const CameraCalibConfig& config);

	virtual ~CameraCalibrator();

	bool addImages();

	bool calibrate(int calibrateCameraFlags = 0,
		const cv::TermCriteria& calibrateCameraC = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON));

	static bool undistortImage(const cv::Mat& inputImage,
		const cv::Mat& intrinsics, const cv::Mat& distCoeffs, cv::Mat& undistortedImage,
		cv::InterpolationFlags flags = cv::INTER_LINEAR);

	bool getCornersSB(int findChessBoardCornersFlags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);

	bool getCorners(int findChessboardCornersFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE /*| cv::CALIB_CB_FAST_CHECK*/);

	//bool pixelToCameraCoordinates(const cv::Point2d& pixel, const cv::Mat& rvec,
	//	const cv::Mat& tvec, cv::Mat& point);

	inline bool checkResultSize();


	inline bool checkType(const std::vector<cv::Mat>& matrixs, int& mainType);

	//template<typename T>
	//bool pixelToCameraCoordinatesImpl(const cv::Point_<T>& pixel,
	//	const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& point);

	std::vector<cv::Mat> getDraws() const;

	CalibrateResult getCalibrateResult() const;

	bool genOjbectCorners();

	bool calReprojectionError();


	float getError() const;

	bool calcReprojectionErrorReal();

	bool pixelToWorld(const cv::Point2f& undistoredPoint, const cv::Mat& rvec, const cv::Mat& tvec,
		cv::Point3f& worldPoint);

	bool calcWorldPointsWithoutPnP();

	bool calcWorldPoints();

	bool showCornersImages(float scaleFactor = 0.0f, int waitTime = 0) const;

	bool calibrateCameraWithCharuco();
	bool calibrateCameraWithAruco();
    bool calibrateCameraWithArucoBoard();
    bool calibrateCameraWithAprilTag();
	std::vector<cv::Mat> getImages() const;

	std::vector<std::vector<cv::Point2f>> getImageCornersVec() const;

	std::vector<std::vector<cv::Point3f>> getObjectCornersVec() const;

    std::vector<int> getValidIndices() const;

private:


	cv::Ptr<cv::aruco::CharucoBoard> m_board;
	cv::Ptr<cv::aruco::Dictionary> m_dictionary;
	cv::Size m_boardSize;
	cv::Size2f m_squareSize;
	float m_markerLength;
	CalibrateResult m_calibrateResult;
	cv::Size m_imageSize;
	int m_imageType;
	std::vector<cv::Mat> m_images;
	std::string m_imagesFolder;

	std::vector<std::vector<cv::Point2f>> m_imageCornersVec;
	std::vector<cv::Mat> m_debugDraws;
	std::vector<std::vector<cv::Point3f>> m_objectCornersVec;
	std::vector<std::vector<cv::Point2f>> m_reprojectionCornersVec;
	std::vector<cv::Mat> m_rvecs_base2camera;
	std::vector<cv::Mat> m_tvecs_base2camera;
	std::vector<cv::Mat> m_R_base2camera;
	std::vector<cv::Mat> m_T_base2camera;

	std::vector<std::vector<cv::Point3f>> m_worldPointsVec;

	float m_err;

	std::vector<double> m_perImageErrors;
	double m_averageError;

	std::vector<std::vector<cv::Point2f>> m_markerCornersVec;
    std::vector<int> m_validIndices;
    std::vector<std::string> m_imagePaths;
    cv::Ptr<cv::aruco::GridBoard> m_arucoBoard;
};

#endif // __CAMERA_CALIBRATOR_H__
