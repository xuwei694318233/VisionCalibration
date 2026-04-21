/**
 * @file CameraCalibrator.h
 * @brief 相机标定器头文件 - 声明相机标定相关类和数据结构
 * 
 * 本文件定义了CameraCalibrator类，用于实现多种标定板(棋盘格、Charuco、Aruco、AprilTag)
 * 的相机内参标定功能。支持从图像文件夹自动加载图片、角点检测、标定计算等完整流程。
 * 
 * @author 系统开发组
 * @date 2026-04-20
 * @version 1.0.0
 * 
 * @section 功能特性
 * - 支持多种标定板类型：棋盘格、Charuco、Aruco、AprilTag
 * - 自动图像加载和角点检测
 * - 内参矩阵和畸变系数计算
 * - 重投影误差计算和分析
 * - 图像去畸变处理
 * 
 * @section 使用说明
 * 1. 创建CameraCalibrator实例并配置标定参数
 * 2. 调用addImages()加载标定图像
 * 3. 调用calibrate()进行标定计算
 * 4. 获取标定结果和误差分析
 * 
 * @section 注意事项
 * - 标定需要至少3幅有效图像
 * - 图像应包含完整标定板且避免过曝/欠曝
 * - 推荐使用10-20幅图像以获得更好的标定精度
 */

#pragma once

#include "CalibrationConfig.h"
#include <Utils.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <stdexcept>
#include <future>
#include <atomic>
#include <fstream>

/**
 * @brief 标定板类型枚举
 * 
 * 定义支持的标定板类型，不同标定板对应不同的角点检测算法
 */
enum class PlateType
{
    Chessboard,  ///< 标准棋盘格标定板，使用findChessboardCorners检测
    Charuco,     ///< Charuco标定板，结合ArUco标记和棋盘格
    Aruco,       ///< ArUco标记标定板，使用detectMarkers检测
    AprilTag     ///< AprilTag标定板，使用AprilTag检测算法
};

/**
 * @brief 相机标定配置参数结构体
 * 
 * 包含标定所需的全部配置参数，用于初始化CameraCalibrator
 */
struct CameraCalibConfig
{
    PlateType plateType = PlateType::Chessboard;          ///< 标定板类型，默认为棋盘格
    cv::Size boardSize;                                   ///< 标定板角点数 (cols, rows)
    cv::Size2f squareSize;                                ///< 方格物理尺寸 (mm)
    cv::Size2f markerSize;                                ///< Charuco标记尺寸 (mm)
    float markerLength = 0.0f;                            ///< Aruco标记长度 (mm)，只对Aruco类型有效
    cv::Ptr<cv::aruco::Dictionary> dictionary;            ///< ArUco字典，用于Charuco/Aruco标定
    std::string imagesFolder;                             ///< 标定图片文件夹路径
};

struct CalibrateResult
{
    cv::Mat intrinsics;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> tvecsMat;
    std::vector<cv::Mat> rvecsMat;
    std::vector<float> errForImage;
    std::vector<cv::Mat> draws;
    std::vector<std::vector<std::vector<cv::Point2f>>> markerCornersVec;
    std::vector<std::vector<cv::Point2f>> chessCornersVec;
};

class CALIBRATION_API CameraCalibrator
{
public:
    /**
     * @brief 默认构造函数
     */
    CameraCalibrator() = default;

    /**
     * @brief 带配置参数的构造函数
     * @param config 相机标定配置参数
     * 
     * 使用预定义的配置参数初始化标定器
     */
    explicit CameraCalibrator(const CameraCalibConfig &config);

    /**
     * @brief 虚析构函数
     */
    virtual ~CameraCalibrator();

    /**
     * @brief 从配置路径加载标定图像
     * @return true-加载成功 false-加载失败
     * 
     * 从预配置的图片文件夹路径加载所有支持的图像格式
     * 支持格式：jpg, jpeg, png, bmp, tiff等
     */
    bool addImages();

    /**
     * @brief 执行相机标定计算
     * @param calibrateCameraFlags 标定标志位，控制标定算法行为
     * @param calibrateCameraC 标定终止条件
     * @return true-标定成功 false-标定失败
     * 
     * 基于已加载的图像和检测到的角点，计算相机内参和畸变系数
     */
    bool calibrate(int calibrateCameraFlags = 0,
                   const cv::TermCriteria &calibrateCameraC = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON));

    /**
     * @brief 图像去畸变处理（静态函数）
     * @param inputImage 输入待去畸变图像
     * @param intrinsics 相机内参矩阵（3x3）
     * @param distCoeffs 畸变系数向量（4,5,8或12个元素）
     * @param undistortedImage 输出去畸变后的图像
     * @param flags 插值方法，默认线性插值
     * @return true-去畸变成功 false-参数错误
     * 
     * 使用标定得到的相机参数对图像进行去畸变处理
     */
    static bool undistortImage(const cv::Mat &inputImage,
                               const cv::Mat &intrinsics, const cv::Mat &distCoeffs, cv::Mat &undistortedImage,
                               cv::InterpolationFlags flags = cv::INTER_LINEAR);

    /**
     * @brief 使用改进的棋盘格角点检测算法
     * @param findChessBoardCornersFlags 角点检测标志位
     * @return true-检测成功 false-检测失败
     * 
     * 使用findChessboardCornersSB算法进行角点检测，相对于标准算法精度更高
     */
    bool getCornersSB(int findChessBoardCornersFlags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);

    /**
     * @brief 使用标准棋盘格角点检测算法
     * @param findChessboardCornersFlags 角点检测标志位
     * @return true-检测成功 false-检测失败
     * 
     * 使用OpenCV的标准findChessboardCorners算法进行角点检测
     */
    bool getCorners(int findChessboardCornersFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE /*| cv::CALIB_CB_FAST_CHECK*/);

    // bool pixelToCameraCoordinates(const cv::Point2d& pixel, const cv::Mat& rvec,
    //    const cv::Mat& tvec, cv::Mat& point);

    /**
     * @brief 检查标定结果数据完整性
     * @return true-数据完整 false-数据缺失或不一致
     * 
     * 验证标定结果中各种数据数组的大小是否匹配
     */
    inline bool checkResultSize();

    /**
     * @brief 检查矩阵向量中的数据类型一致性
     * @param matrixs 矩阵向量
     * @param mainType 输出主导数据类型
     * @return true-类型一致 false-类型不一致
     * 
     * 验证矩阵向量中所有矩阵的数据类型是否相同
     */
    inline bool checkType(const std::vector<cv::Mat> &matrixs, int &mainType);

    // template<typename T>      
    // bool pixelToCameraCoordinatesImpl(const cv::Point_<T>& pixel,
    //    const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& point);

    /**
     * @brief 获取标记了角点的图像
     * @return 包含角点标记的图像向量
     * 
     * 返回在角点检测过程中绘制的带标记图像
     */
    std::vector<cv::Mat> getDraws() const;

    /**
     * @brief 获取完整的标定结果
     * @return 包含所有标定计算结果的结构体
     */
    CalibrateResult getCalibrateResult() const;

    /**
     * @brief 生成标定板3D空间角点坐标
     * @return true-生成成功 false-参数错误
     * 
     * 根据配置的标定板参数生成对应的世界坐标系角点坐标
     */
    bool genObjectCorners();

    /**
     * @brief 计算重投影误差
     * @return true-计算成功 false-计算失败
     * 
     * 基于标定结果计算每个图像的重投影误差，评估标定精度
     */
    bool calReprojectionError();

    /**
     * @brief 获取平均重投影误差
     * @return 平均重投影误差（像素单位）
     * 
     * 返回所有有效图像的平均重投影误差，用于评估标定精度
     */
    float getError() const;

    /**
     * @brief 计算真实的重投影误差
     * @return true-计算成功 false-计算失败
     * 
     * 基于实际检测结果重新计算重投影误差，提供更准确的精度评估
     */
    bool calcReprojectionErrorReal();

    /**
     * @brief 像素坐标到世界坐标转换
     * @param undistortedPoint 去畸变后的像素坐标
     * @param rvec 旋转向量
     * @param tvec 平移向量
     * @param worldPoint 输出的世界坐标
     * @return true-转换成功 false-参数错误
     * 
     * 使用PnP原理将二维像素点反投影到三维世界坐标系
     */
    bool pixelToWorld(const cv::Point2f &undistortedPoint, const cv::Mat &rvec, const cv::Mat &tvec,
                      cv::Point3f &worldPoint);

    /**
     * @brief 无PNP约束的世界坐标计算
     * @return true-计算成功 false-计算失败
     * 
     * 不依赖PNP计算的世界坐标系点生成，用于特殊场景下的坐标转换
     */
    bool calcWorldPointsWithoutPnP();

    /**
     * @brief 计算世界坐标系中的标定点坐标
     * @return true-计算成功 false-计算失败
     * 
     * 通过PNP算法计算标定板在世界坐标系中的精确位置
     */
    bool calcWorldPoints();

    /**
     * @brief 显示带角点标记的图像
     * @param scaleFactor 图像缩放比例，0表示不缩放
     * @param waitTime 图像显示等待时间（毫秒），0表示等待按键
     * @return true-显示成功 false-无图像或显示失败
     * 
     * 可视化展示角点检测结果，用于调试和验证
     */
    bool showCornersImages(float scaleFactor = 0.0f, int waitTime = 0) const;

    /**
     * @brief Charuco标定板专用相机标定
     * @return true-标定成功 false-标定失败
     * 
     * 针对Charuco标定板的特定标定算法实现
     */
    bool calibrateCameraWithCharuco();
    
    /**
     * @brief Aruco标记标定
     * @return true-标定成功 false-标定失败
     * 
     * 使用ArUco标记进行相机标定的特定实现
     */
    bool calibrateCameraWithAruco();
    
    /**
     * @brief Aruco板标定
     * @return true-标定成功 false-标定失败
     * 
     * 使用ArUco板（多个标记组成的标定板）进行标定
     */
    bool calibrateCameraWithArucoBoard();
    
    /**
     * @brief AprilTag标定板标定
     * @return true-标定成功 false-标定失败
     * 
     * 针对AprilTag标定板的特定标定算法实现
     */
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
