#ifndef __HAND_EYE_CALIBRATOR_H__
#define __HAND_EYE_CALIBRATOR_H__ 

#include "CalibrationConfig.h"

#include <fstream>
#include <algorithm>



#include "Utils.h"
#include "CameraCalibrator.h"

enum CalibrationMode {
    EYE_IN_HAND = 0,
    EYE_TO_HAND = 1
};

// 手眼标定配置结构体
struct HandEyeCalibConfig {
    CalibrationMode mode = EYE_IN_HAND;
    std::string poseFile;         // 机器人位姿文件路径
    
    // 包含相机标定配置
    CameraCalibConfig camConfig;

    // 可选：预先存在的内参（如果不需要重新标定相机）
    bool useExistIntrinsics = false;
    cv::Mat intrinsics;
    cv::Mat distCoeffs;

    // 可选：预先存在的标定板位姿（如果不需要从图片计算）
    bool useExistBoardPose = false;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    bool enableRefinement = true;
    double rotationWeight = 10.0;
    double translationWeight = 1.0;
    int maxIterations = 100;
};

class CALIBRATION_API HandEyeCalibrator {

public:
    HandEyeCalibrator() = default;

    // 新增：基于配置结构体的构造函数
    explicit HandEyeCalibrator(const HandEyeCalibConfig& config);

    virtual ~HandEyeCalibrator();

    bool calibrate(cv::HandEyeCalibrationMethod method = cv::CALIB_HAND_EYE_TSAI);

    bool checkVec();

    bool TvecTo31();

    bool calReprojectionError();

    inline bool invertTransform(const cv::Mat& R, const cv::Mat& T,
        cv::Mat& R_inv, cv::Mat& T_inv);

    bool solveTargetPose(bool useExtrinsicGuess = false, int flags = cv::SOLVEPNP_ITERATIVE);

    double getReprojectionError() const;

    bool calibrateWithAruco(const std::string& folderPath, const float markerLegth_mm,
        const int dictionaryId = cv::aruco::DICT_6X6_250, cv::HandEyeCalibrationMethod method = cv::CALIB_HAND_EYE_TSAI);

    cv::Mat getResult();

    bool calcCornersInTargetVec();

    bool calcTarget2baseVec();

    bool calcReprojectionErrorWithCharuco();

    bool calcReprojectionErrorOnlyChess();

    bool calReprojectionErrorMean();

    double getRotationError() const { 
        return m_rotationError; 
    }

    // Data Injection Methods
    void setRobotPoses(const std::vector<cv::Mat>& poses, bool isRad = true);
    void setTargetPoses(const std::vector<cv::Mat>& R_target2cam, const std::vector<cv::Mat>& T_target2cam);
    
    // Non-linear refinement
    void refineCalibration();

public:
    // CameraCalibrator m_cameraCalibrator; // Decoupled

private:

    cv::Ptr<cv::aruco::CharucoBoard> m_charucoBoard;
    CalibrationMode m_mode;
    std::string m_poseFile;
    std::string m_imagesFolder;
    cv::Size m_boardSize;
    cv::Size2f m_squareSize;
    std::vector<cv::Mat> m_images;
    cv::Size m_imageSize;
    int m_imageType;

    std::vector<std::vector<cv::Point2f>> m_imageCornersVec;
    std::vector<std::vector<cv::Point3f>> m_objectCornersVec;
    std::vector<std::vector<std::vector<cv::Point2f>>> m_markerCornersVec;
    std::vector<std::vector<cv::Point2f>> m_chessCornersVec;
    cv::Mat m_intrinsics;
    cv::Mat m_distCoeffs;
    std::vector<cv::Mat> m_R_target2camVec;
    std::vector<cv::Mat> m_T_target2camVec;
    std::vector<cv::Mat> m_H_target2camVec;
    cv::HandEyeCalibrationMethod m_method = cv::CALIB_HAND_EYE_TSAI;
    double m_reprojectionError = 0.0;
    double m_rotationError = 0.0;

    std::vector<cv::Mat> m_R_target2baseVec;
    std::vector<cv::Mat> m_T_target2baseVec;
    std::vector<cv::Mat> m_H_base2gripperVec;
    std::vector<cv::Mat> m_rvecsMat;
    std::vector<cv::Mat> m_tvecsMat;


    std::vector<cv::Mat> m_arucoImageVec;
    std::vector<cv::Mat> m_poseVec;

    cv::Mat m_debugDraws;

    // ======================================================
    std::vector<cv::Mat> m_R_gripper2baseVec;    // 眼在手上
    std::vector<cv::Mat> m_T_gripper2baseVec;    // 眼在手上


    std::vector<cv::Mat> m_R_base2gripperVec;    // 眼在手外
    std::vector<cv::Mat> m_T_base2gripperVec;    // 眼在手外


    cv::Mat m_R_cam2gripper;    // 眼在手上
    cv::Mat m_T_cam2gripper;    // 眼在手上
    cv::Mat m_H_cam2gripper;    // 眼在手上

    cv::Mat m_R_cam2base;    // 眼在手外
    cv::Mat m_T_cam2base;    // 眼在手外
    cv::Mat m_H_cam2base;    // 眼在手外

    std::vector<std::vector<cv::Point3f>> m_cornersInBaseVec;

    bool m_enableRefinement = true;
    double m_rotationWeight = 10.0;
    double m_translationWeight = 1.0;
    int m_maxIterations = 100;
};

#endif