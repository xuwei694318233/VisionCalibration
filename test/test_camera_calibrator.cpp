/**
 * @file test_camera_calibrator.cpp
 * @brief 相机标定器和手眼标定器测试程序
 *
 * 本测试程序不依赖gtest框架，使用原生C++实现。
 * 测试需要外部提供标定图像。
 *
 * @author 测试开发组
 * @date 2026-04-21
 * @version 2.0.0
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "CameraCalibrator.h"
#include "HandEyeCalibrator.h"
#include "HandEyeCalibrator2D.h"
#include "HandEyeCalibrator25D.h"
#include "Utils.h"

// 测试结果统计
static int g_testPassed = 0;
static int g_testFailed = 0;

// 打印测试结果宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            std::cout << "[  PASSED  ] " << message << std::endl; \
            g_testPassed++; \
        } else { \
            std::cout << "[  FAILED  ] " << message << std::endl; \
            g_testFailed++; \
        } \
    } while (0)

#define TEST_INFO(message) \
    std::cout << "[   INFO   ] " << message << std::endl;

// 测试配置
struct TestConfig {
    // 相机标定配置
    std::string calibImagesFolder = "calibration_images";
    cv::Size boardSize = cv::Size(9, 6);
    cv::Size2f squareSize = cv::Size2f(25.0f, 25.0f);
    PlateType plateType = PlateType::Chessboard;

    // 手眼标定配置
    std::string robotPosesFile = "robot_poses.txt";
    CalibrationMode handEyeMode = CalibrationMode::EYE_IN_HAND;
};

// 基础功能测试
void testConstructor() {
    std::cout << std::endl << "[测试 1] 构造函数测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibrator calibrator1;
    TEST_ASSERT(true, "默认构造函数创建成功");

    CameraCalibConfig config;
    config.boardSize = cv::Size(9, 6);
    config.squareSize = cv::Size2f(25.0f, 25.0f);
    CameraCalibrator calibrator2(config);
    TEST_ASSERT(true, "带参构造函数创建成功");
}

// 图像加载测试
void testAddImages(const TestConfig& config) {
    std::cout << std::endl << "[测试 2] 图像加载测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    bool result = calibrator.addImages();
    TEST_ASSERT(result, "图像加载");

    if (result) {
        auto images = calibrator.getImages();
        TEST_ASSERT(images.size() >= 3, "图像数量: " + std::to_string(images.size()) + " (至少3张)");
    }
}

// 角点检测测试
void testCornerDetection(const TestConfig& config) {
    std::cout << std::endl << "[测试 3] 角点检测测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    calibrator.addImages();

    bool result = calibrator.getCorners();
    TEST_ASSERT(result, "标准角点检测");

    bool resultSB = calibrator.getCornersSB();
    TEST_ASSERT(resultSB, "高精度角点检测");

    auto corners = calibrator.getImageCornersVec();
    TEST_ASSERT(corners.size() >= 3, "有效角点组数: " + std::to_string(corners.size()));
}

// 标定算法测试
void testCalibration(const TestConfig& config) {
    std::cout << std::endl << "[测试 4] 标定算法测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    calibrator.addImages();
    calibrator.getCorners();

    bool result = calibrator.calibrate();
    TEST_ASSERT(result, "相机标定");

    if (result) {
        auto calibResult = calibrator.getCalibrateResult();

        bool hasIntrinsics = !calibResult.intrinsics.empty();
        TEST_ASSERT(hasIntrinsics, "内参矩阵存在");

        if (hasIntrinsics) {
            TEST_ASSERT(calibResult.intrinsics.rows == 3 && calibResult.intrinsics.cols == 3,
                        "内参矩阵维度: 3x3");
            std::cout << "  内参矩阵:" << std::endl;
            std::cout << "    fx=" << calibResult.intrinsics.at<double>(0, 0)
                      << "  fy=" << calibResult.intrinsics.at<double>(1, 1) << std::endl;
            std::cout << "    cx=" << calibResult.intrinsics.at<double>(0, 2)
                      << "  cy=" << calibResult.intrinsics.at<double>(1, 2) << std::endl;
        }

        bool hasDistCoeffs = !calibResult.distCoeffs.empty();
        TEST_ASSERT(hasDistCoeffs, "畸变系数存在");

        if (hasDistCoeffs) {
            std::cout << "  畸变系数: ";
            for (int i = 0; i < calibResult.distCoeffs.cols; ++i) {
                std::cout << calibResult.distCoeffs.at<double>(0, i) << " ";
            }
            std::cout << std::endl;
        }
    }
}

// 重投影误差测试
void testReprojectionError(const TestConfig& config) {
    std::cout << std::endl << "[测试 5] 重投影误差测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    calibrator.addImages();
    calibrator.getCorners();
    calibrator.calibrate();

    bool result = calibrator.calReprojectionError();
    TEST_ASSERT(result, "重投影误差计算");

    bool resultReal = calibrator.calcReprojectionErrorReal();
    TEST_ASSERT(resultReal, "真实重投影误差计算");

    float error = calibrator.getError();
    TEST_ASSERT(error >= 0.0f, "平均误差: " + std::to_string(error) + " 像素");
    TEST_ASSERT(error < 2.0f, "误差小于2像素 (良好标定)");
}

// 像素坐标转换测试
void testPixelToWorld(const TestConfig& config) {
    std::cout << std::endl << "[测试 6] 像素坐标转世界坐标测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    calibrator.addImages();
    calibrator.getCorners();
    calibrator.calibrate();
    calibrator.calcWorldPoints();

    auto calibResult = calibrator.getCalibrateResult();

    if (!calibResult.rvecsMat.empty() && !calibResult.tvecsMat.empty()) {
        cv::Point2f testPixel(320, 240);
        cv::Point3f worldPoint;

        bool result = calibrator.pixelToWorld(testPixel,
                                               calibResult.rvecsMat[0],
                                               calibResult.tvecsMat[0],
                                               worldPoint);

        TEST_ASSERT(result, "像素到世界坐标转换");
        if (result) {
            std::cout << "  像素坐标: (" << testPixel.x << ", " << testPixel.y << ")" << std::endl;
            std::cout << "  世界坐标: (" << worldPoint.x << ", " << worldPoint.y << ", " << worldPoint.z << ")" << std::endl;
        }
    } else {
        TEST_ASSERT(false, "缺少标定结果数据");
    }
}

// 图像去畸变测试
void testUndistortImage(const TestConfig& config) {
    std::cout << std::endl << "[测试 7] 图像去畸变测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);
    calibrator.addImages();
    calibrator.getCorners();
    calibrator.calibrate();

    auto calibResult = calibrator.getCalibrateResult();
    auto images = calibrator.getImages();

    if (!images.empty() && !calibResult.intrinsics.empty()) {
        cv::Mat undistorted;
        bool result = CameraCalibrator::undistortImage(images[0],
                                                       calibResult.intrinsics,
                                                       calibResult.distCoeffs,
                                                       undistorted);

        TEST_ASSERT(result, "图像去畸变");
        TEST_ASSERT(!undistorted.empty(), "去畸变图像非空");
        TEST_ASSERT(undistorted.size() == images[0].size(), "图像尺寸一致");
    } else {
        TEST_ASSERT(false, "无图像或标定数据");
    }
}

// 空图像处理测试
void testEmptyImages() {
    std::cout << std::endl << "[测试 8] 空图像处理测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibrator calibrator;

    bool result = calibrator.getCorners();
    TEST_ASSERT(!result, "空图像角点检测应失败");

    bool calibResult = calibrator.calibrate();
    TEST_ASSERT(!calibResult, "空图像标定应失败");
}

// 手眼标定测试 - Eye-in-Hand 模式
void testHandEyeCalibration(const TestConfig& config) {
    std::cout << std::endl << "[测试 9] 手眼标定测试 (Eye-in-Hand)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  注意: 需要提供实际的机器人位姿和相机位姿数据" << std::endl;
    std::cout << "  此处仅测试接口可用性" << std::endl;

    HandEyeCalibConfig handEyeConfig;
    handEyeConfig.mode = CalibrationMode::EYE_IN_HAND;

    HandEyeCalibrator handEyeCalibrator(handEyeConfig);

    // 测试空数据标定（应该失败）
    bool result = handEyeCalibrator.calibrate();
    TEST_ASSERT(!result, "空数据手眼标定应失败");

    TEST_ASSERT(true, "HandEyeCalibrator 创建成功");
}

// 手眼标定测试 - Eye-to-Hand 模式
void testHandEyeCalibrationEyeToHand(const TestConfig& config) {
    std::cout << std::endl << "[测试 10] 手眼标定测试 (Eye-to-Hand)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    HandEyeCalibConfig handEyeConfig;
    handEyeConfig.mode = CalibrationMode::EYE_TO_HAND;

    HandEyeCalibrator handEyeCalibrator(handEyeConfig);

    // 测试空数据标定（应该失败）
    bool result = handEyeCalibrator.calibrate();
    TEST_ASSERT(!result, "空数据手眼标定应失败");

    TEST_ASSERT(true, "HandEyeCalibrator (Eye-to-Hand) 创建成功");
}

// 2D 手眼标定测试
void testHandEyeCalibrator2D() {
    std::cout << std::endl << "[测试 11] 2D 手眼标定测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    HandEyeCalibrator2D calibrator2D;

    // 添加观测数据: 像素坐标 + 机器人位姿 (x, y, theta)
    for (int i = 0; i < 5; ++i) {
        cv::Point2f pixel(100 + i * 50, 200 + i * 30);
        cv::Vec3f robotPose(i * 50.0f, i * 30.0f, i * 0.1f);  // x, y, theta
        calibrator2D.addObservation(pixel, robotPose);
    }

    bool result = calibrator2D.calibrate();
    TEST_ASSERT(result, "2D 手眼标定");

    if (result) {
        cv::Mat affineMatrix = calibrator2D.getResult();
        TEST_ASSERT(!affineMatrix.empty(), "仿射矩阵非空");
        TEST_ASSERT(affineMatrix.rows == 2 && affineMatrix.cols == 3, "仿射矩阵维度 2x3");

        double error = calibrator2D.getReprojectionError();
        std::cout << "  重投影误差: " << error << " 像素" << std::endl;
    }
}

// 2.5D 手眼标定测试
void testHandEyeCalibrator25D() {
    std::cout << std::endl << "[测试 12] 2.5D 手眼标定测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    HandEyeCalibrator25D calibrator25D;

    // 添加观测数据: 图像中心 + 尺度 + 机器人Z轴
    for (int i = 0; i < 5; ++i) {
        cv::Point2f center(320 + i * 20, 240 + i * 15);
        double scale = 100.0 + i * 10;
        double robotZ = 500.0 + i * 50;
        calibrator25D.addObservation(center, scale, robotZ);
    }

    bool result = calibrator25D.calibrate();
    TEST_ASSERT(result, "2.5D 手眼标定");

    if (result) {
        std::cout << "  图像中心: cx=" << calibrator25D.getCx() << "  cy=" << calibrator25D.getCy() << std::endl;
        std::cout << "  尺度因子: kScale=" << calibrator25D.getKScale() << std::endl;
        std::cout << "  S0: " << calibrator25D.getS0() << std::endl;
    }
}

// 工具函数测试
void testUtils() {
    std::cout << std::endl << "[测试 13] 工具函数测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 测试旋转矩阵转齐次矩阵
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T = (cv::Mat_<double>(3, 1) << 10, 20, 30);
    cv::Mat H;

    bool result = R_T2H(R, T, H);
    TEST_ASSERT(result, "R_T2H 函数");

    if (result) {
        TEST_ASSERT(H.rows == 4 && H.cols == 4, "齐次矩阵维度 4x4");
    }

    // 测试齐次矩阵转 R/T
    cv::Mat R_out, T_out;
    result = H2R_T(H, R_out, T_out);
    TEST_ASSERT(result, "H2R_T 函数");

    // 测试欧拉角转旋转矩阵
    cv::Mat R_euler;
    result = eulerToRotationMatrix(cv::Vec3d(45, 30, 0), R_euler);
    TEST_ASSERT(result, "eulerToRotationMatrix 函数");

    if (result) {
        TEST_ASSERT(isRotationMatrix(R_euler), "旋转矩阵有效性");
    }
}

// 性能测试
void testPerformance(const TestConfig& config) {
    std::cout << std::endl << "[测试 14] 性能测试" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CameraCalibConfig calibConfig;
    calibConfig.plateType = config.plateType;
    calibConfig.boardSize = config.boardSize;
    calibConfig.squareSize = config.squareSize;
    calibConfig.imagesFolder = config.calibImagesFolder;

    CameraCalibrator calibrator(calibConfig);

    auto start = std::chrono::high_resolution_clock::now();

    calibrator.addImages();
    calibrator.getCorners();
    calibrator.calibrate();
    calibrator.calReprojectionError();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "  耗时: " << duration.count() << " ms" << std::endl;
    TEST_ASSERT(duration.count() < 30000, "性能测试通过 (<30秒)");
}

// 打印使用说明
void printUsage() {
    std::cout << "========================================" << std::endl;
    std::cout << "   VisionCalibration 测试程序" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "用法: test_camera_calibrator.exe [图像文件夹] [棋盘格列数] [棋盘格行数] [方格尺寸]" << std::endl;
    std::cout << std::endl;
    std::cout << "参数说明:" << std::endl;
    std::cout << "  图像文件夹    - 标定图像所在文件夹 (默认: calibration_images)" << std::endl;
    std::cout << "  棋盘格列数    - 内部角点列数 (默认: 9)" << std::endl;
    std::cout << "  棋盘格行数    - 内部角点行数 (默认: 6)" << std::endl;
    std::cout << "  方格尺寸      - 方格物理尺寸 mm (默认: 25)" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  test_camera_calibrator.exe" << std::endl;
    std::cout << "  test_camera_calibrator.exe my_images 11 8 30" << std::endl;
    std::cout << std::endl;
}

// 主函数
int main(int argc, char** argv) {
    // 设置控制台输出编码为 UTF-8
    #ifdef _WIN32
    system("chcp 65001 >nul");
    #endif

    cv::setNumThreads(0);

    TestConfig config;

    // 解析命令行参数
    if (argc >= 2) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            printUsage();
            return 0;
        }
        config.calibImagesFolder = argv[1];
    }
    if (argc >= 4) {
        config.boardSize.width = std::atoi(argv[2]);
        config.boardSize.height = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        config.squareSize.width = std::atof(argv[4]);
        config.squareSize.height = config.squareSize.width;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "   VisionCalibration 测试套件" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "配置信息:" << std::endl;
    std::cout << "  图像文件夹: " << config.calibImagesFolder << std::endl;
    std::cout << "  棋盘格尺寸: " << config.boardSize.width << "x" << config.boardSize.height << std::endl;
    std::cout << "  方格尺寸: " << config.squareSize.width << " mm" << std::endl;
    std::cout << std::endl;

    // 检查图像文件夹是否存在
    if (!std::filesystem::exists(config.calibImagesFolder)) {
        std::cout << "错误: 图像文件夹不存在: " << config.calibImagesFolder << std::endl;
        std::cout << "请创建文件夹并放入标定图像，或指定正确的路径" << std::endl;
        printUsage();
        return 1;
    }

    // 执行各项测试
    testConstructor();
    testAddImages(config);
    testCornerDetection(config);
    testCalibration(config);
    testReprojectionError(config);
    testPixelToWorld(config);
    testUndistortImage(config);
    testEmptyImages();
    testHandEyeCalibration(config);
    testHandEyeCalibrationEyeToHand(config);
    testHandEyeCalibrator2D();
    testHandEyeCalibrator25D();
    testUtils();
    testPerformance(config);

    // 输出统计结果
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "   测试结果统计" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "通过: " << g_testPassed << std::endl;
    std::cout << "失败: " << g_testFailed << std::endl;
    std::cout << "总计: " << (g_testPassed + g_testFailed) << std::endl;

    if (g_testFailed == 0) {
        std::cout << std::endl << "所有测试通过!" << std::endl;
    } else {
        std::cout << std::endl << "存在 " << g_testFailed << " 个测试失败!" << std::endl;
    }

    return g_testFailed > 0 ? 1 : 0;
}