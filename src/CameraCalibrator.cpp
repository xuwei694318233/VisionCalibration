/**
 * @file CameraCalibrator.cpp
 * @brief 相机标定器实现文件 - 实现相机标定的核心逻辑和算法
 * 
 * 本文件实现了CameraCalibrator类的所有功能，包括图像加载、角点检测、标定计算、
 * 误差分析等完整的相机标定流程。支持多种标定板类型和标定算法。
 * 
 * @author 系统开发组
 * @date 2026-04-20
 * @version 1.0.0
 * 
 * @section 实现要点
 * - 使用OpenCV标定算法进行相机参数计算
 * - 支持棋盘格、Charuco、Aruco、AprilTag标定板
 * - 自动图像加载和角点检测
 * - 重投影误差计算和精度评估
 * - 像素到世界坐标转换功能
 * 
 * @section 算法流程
 * 1. 构造函数：根据配置初始化相应标定板
 * 2. addImages()：从文件夹加载标定图像
 * 3. getCorners()/getCornersSB()：角点检测
 * 4. calibrate()：进行标定计算
 * 5. calReprojectionError()：计算重投影误差
 */

#include "CameraCalibrator.h"
#include "Utils.h"

/**
 * @brief 带配置参数的构造函数
 * @param config 相机标定配置参数
 */
CameraCalibrator::CameraCalibrator(const CameraCalibConfig& config)
    : m_boardSize(config.boardSize),
      m_squareSize(config.squareSize),
      m_markerLength(config.markerLength),
      m_dictionary(config.dictionary),
      m_imagesFolder(config.imagesFolder)
{
    if (config.plateType == PlateType::Charuco)
    {
        // 确保使用正确的正方形尺寸和标记尺寸
        // 注意：CharucoBoard::create接收的是网格尺寸（squaresX, squaresY），UI传入的是角点数，需+1
        m_board = cv::aruco::CharucoBoard::create(
            config.boardSize.width + 1, config.boardSize.height + 1,
            config.squareSize.width, config.markerSize.width, 
            config.dictionary);
            
        if (config.markerSize.width >= config.squareSize.width) {
            LOG_OUTPUT(Warn, "Marker size should be smaller than square size for proper Charuco board");
        }
    }
    else if (config.plateType == PlateType::Aruco)
    {
        float markerSeparation = config.squareSize.width - config.markerSize.width;
        if (markerSeparation < 0) markerSeparation = 0;
        
        m_arucoBoard = cv::aruco::GridBoard::create(
            config.boardSize.width + 1, config.boardSize.height + 1,
            config.markerSize.width, markerSeparation,
            config.dictionary);
    }
}

/**
 * @brief 析构函数
 * 
 * 清理类实例占用的资源，释放OpenCV相关的智能指针和容器
 */
CameraCalibrator::~CameraCalibrator()
{

}

std::vector<std::vector<cv::Point2f>> CameraCalibrator::getImageCornersVec() const
{
    return m_imageCornersVec;
}

std::vector<std::vector<cv::Point3f>> CameraCalibrator::getObjectCornersVec() const
{
    return  m_objectCornersVec;
}

std::vector<int> CameraCalibrator::getValidIndices() const
{
    return m_validIndices;
}

bool CameraCalibrator::undistortImage(const cv::Mat& inputImage,
    const cv::Mat& intrinsics, const cv::Mat& distCoeffs, cv::Mat& undistortedImage,
    cv::InterpolationFlags flags)
{
    // 输入检查
    if (inputImage.empty())
    {
        LOG_OUTPUT(Error, "Input image for undistortion is empty");
        return false;
    }
    if (intrinsics.rows != 3 || intrinsics.cols != 3) {
        LOG_OUTPUT(Error, "Camera intrinsic matrix must be 3x3");
        return false;
    }
    const int distCoeffsCount = distCoeffs.total();
    if (distCoeffsCount != 5 && distCoeffsCount != 8) {
        LOG_OUTPUT(Error, "Distortion coefficients count error, should be 5 or 8");
        return false;
    }
    if (intrinsics.type() != CV_32F && intrinsics.type() != CV_64F) {
        LOG_OUTPUT(Error, "Camera intrinsic matrix type error");
        return false;
    }
    if (distCoeffs.type() != CV_32F && distCoeffs.type() != CV_64F) {
        LOG_OUTPUT(Error, "Distortion coefficients type error");
        return false;
    }

    // 统一数据类型
    const int output_type = (intrinsics.type() == CV_64F || distCoeffs.type() == CV_64F) ? CV_64F : CV_32F;
    cv::Mat intrinsics_unified, distCoeffs_unified;
    intrinsics.convertTo(intrinsics_unified, output_type);
    distCoeffs.convertTo(distCoeffs_unified, output_type);

    // 预计算映射表
    const int map_type = (output_type == CV_64F) ? CV_64FC2 : CV_32FC2;
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(
        intrinsics_unified, distCoeffs_unified, cv::Mat(),
        intrinsics_unified, inputImage.size(), map_type, map1, map2
    );

    // 执行畸变校正
    cv::remap(inputImage, undistortedImage, map1, map2, flags);

    return true;
}

bool CameraCalibrator::getCorners(int findChessBoardCornersFlags)
{
    // 检查输入有效性
    if (m_images.empty()) {
        LOG_OUTPUT(Error, "Image data is empty, cannot perform calibration");
        return false;
    }
    if (m_boardSize.width <= 0 || m_boardSize.height <= 0) {
        LOG_OUTPUT(Error, "Chessboard size setting error");
        return false;
    }

    // 清空之前的结果
    m_imageCornersVec.clear();
    m_debugDraws.clear();
    m_validIndices.clear();

    int failedCount = 0;

    // 定义结果结构体用于并行处理
    struct DetectionResult {
        bool found = false;
        std::vector<cv::Point2f> corners;
        cv::Mat debugImage;
    };
    std::vector<DetectionResult> results(m_images.size());

    // 并行检测角点
    cv::parallel_for_(cv::Range(0, m_images.size()), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            auto& image = m_images[i];
            cv::Mat grayImage;
            
            if (image.channels() > 1) {
                cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            }
            else {
                grayImage = image;
            }

            results[i].found = findChessboardCorners(grayImage, m_boardSize, results[i].corners,
                findChessBoardCornersFlags);

            if (results[i].found) {
                cv::find4QuadCornerSubpix(grayImage, results[i].corners, cv::Size(5, 5));

                // 绘制并保存结果
                results[i].debugImage = image.clone();
                try {
                    cv::drawChessboardCorners(results[i].debugImage, m_boardSize, results[i].corners, results[i].found);
                }
                catch (const cv::Exception&) {
                    // 并行中不方便打印日志，忽略绘制错误
                }
            }
        }
    });

    // 汇总结果
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].found) {
            m_imageCornersVec.push_back(results[i].corners);
            m_debugDraws.push_back(results[i].debugImage);
            m_validIndices.push_back(i);
        }
        else {
            failedCount++;
            std::string msg("No corners detected in image " + std::to_string(i));
            if (i < m_imagePaths.size()) {
                msg += " (" + m_imagePaths[i] + ")";
            }
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }

    // 检查有效数据量
    if (m_imageCornersVec.size() < 3) {
        std::string msg("Insufficient corner data (" + std::to_string(m_imageCornersVec.size()) +
            " groups), need at least 3 groups");
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }

    if (m_imageCornersVec.size() < 10) {
        std::string msg("Insufficient corner data (only " + std::to_string(m_imageCornersVec.size()) +
            " groups), recommend at least 10 groups");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    LOG_OUTPUT(Info, "Successfully get corners");
    return true;
}

bool CameraCalibrator::getCornersSB(int findChessBoardCornersFlags)
{
    // 检查输入有效性
    if (m_images.empty()) {
        LOG_OUTPUT(Error, "Image data is empty, cannot perform calibration");
        return false;
    }
    if (m_boardSize.width <= 0 || m_boardSize.height <= 0) {
        LOG_OUTPUT(Error, "Chessboard size setting error");
        return false;
    }

    // 清空之前的结果
    m_imageCornersVec.clear();
    m_debugDraws.clear();
    m_validIndices.clear();

    std::vector<cv::Point2f> corners;
    cv::Mat grayImage;
    bool found;
    int failedCount = 0;

    for (size_t i = 0; i < m_images.size(); ++i) {
        auto& image = m_images[i];

        // 转换为灰度图
        if (image.channels() > 1) {
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        }
        else {
            grayImage = image;
        }

        // 使用findChessboardCornersSB检测角点
        found = findChessboardCornersSB(grayImage, m_boardSize, corners,
            findChessBoardCornersFlags);

        if (found) {
            m_imageCornersVec.push_back(corners);
            m_validIndices.push_back(i);

            // 绘制并保存结果
            cv::Mat debugImage = image.clone();

            try {
                cv::drawChessboardCorners(debugImage, m_boardSize, corners, found);
            }
            catch (const cv::Exception& e) {
                LOG_OUTPUT(Error, e.what());
                std::string msg("Image type" + std::to_string(debugImage.type()) + ", Corners type: " + std::to_string(cv::Mat(corners).type()));
                LOG_OUTPUT(Error, msg.c_str());
            }


            m_debugDraws.push_back(debugImage);

        }
        else {
            failedCount++;
            std::string msg("No corners detected in image " + std::to_string(i));
            LOG_OUTPUT(Warn, msg.c_str());
        }
    }

    // 检查有效数据量
    if (m_imageCornersVec.size() < 3) {
        std::string msg("Insufficient corner data (" + std::to_string(m_imageCornersVec.size()) +
            " groups), need at least 3 groups");
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }

    if (m_imageCornersVec.size() < 10) {
        std::string msg("Insufficient corner data (only " + std::to_string(m_imageCornersVec.size()) +
            " groups), recommend at least 10 groups");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    LOG_OUTPUT(Info, "Successfully get corners");
    return true;
}

bool CameraCalibrator::genObjectCorners()
{
    if (m_imageCornersVec.empty()) {
        LOG_OUTPUT(Error, "No image corners available");
        return false;
    }
    
    m_objectCornersVec.clear();
    for (size_t imgIdx = 0; imgIdx < m_imageCornersVec.size(); imgIdx++) {
        std::vector<cv::Point3f> tempPoints;
        for (int row = 0; row < m_boardSize.height; row++) {
            for (int col = 0; col < m_boardSize.width; col++) {
                tempPoints.emplace_back(
                    col * m_squareSize.width,
                    row * m_squareSize.height,
                    0.0f
                );
            }
        }
        if (tempPoints.size() != m_imageCornersVec[imgIdx].size()) {
            LOG_OUTPUT(Error, "Object points and image corner points count mismatch");
            return false;
        }
        m_objectCornersVec.push_back(tempPoints);
    }
    LOG_OUTPUT(Info, "Generate object corners successfully");
    return true;
}

/**
 * @brief 显示带有棋盘格角点的图像（自动缩放适应屏幕）
 * @param scaleFactor 可选缩放因子，0表示自动计算
 * @param waitTime 显示时间(ms)，0表示等待按键
 */
bool CameraCalibrator::showCornersImages(float scaleFactor, int waitTime) const {
    if (m_debugDraws.empty()) {
        LOG_OUTPUT(Warn, "No corner images to display");
        return false;
    }

    // 获取屏幕分辨率
    cv::Size screenSize(1920, 1080); // 默认1080p
    try {
        // 尝试获取实际屏幕分辨率
        cv::namedWindow("temp", cv::WINDOW_NORMAL);
        screenSize.width = cv::getWindowImageRect("temp").width;
        screenSize.height = cv::getWindowImageRect("temp").height;
        cv::destroyWindow("temp");
    }
    catch (...) {
        // 使用默认分辨率
    }

    // 计算自动缩放因子
    if (scaleFactor <= 0) {
        const cv::Size& imgSize = m_debugDraws[0].size();
        float widthScale = static_cast<float>(screenSize.width * 0.8) / imgSize.width;
        float heightScale = static_cast<float>(screenSize.height * 0.8) / imgSize.height;
        scaleFactor = std::min(widthScale, heightScale);
        scaleFactor = std::min(scaleFactor, 1.0f); // 不超过原始大小
    }

    std::cout << "Displaying " << m_debugDraws.size()
        << " images with scale factor: " << scaleFactor << std::endl;

    // 存储所有窗口名称
    std::vector<std::string> windowNames;
    for (size_t i = 0; i < m_debugDraws.size(); ++i) {
        cv::Mat displayImage;
        if (scaleFactor != 1.0f) {
            cv::resize(m_debugDraws[i], displayImage,
                cv::Size(), scaleFactor, scaleFactor,
                cv::INTER_AREA);
        }
        else {
            displayImage = m_debugDraws[i];
        }

        std::string windowName = "Chessboard Corners " + std::to_string(i + 1) +
            "/" + std::to_string(m_debugDraws.size());
        windowNames.push_back(windowName);

        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, displayImage);

        // 调整窗口大小以适应缩放后的图像
        cv::resizeWindow(windowName, displayImage.cols, displayImage.rows);
    }

    cv::waitKey(waitTime);

    // 安全清理窗口
    for (const auto& windowName : windowNames) {
        try {
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 0) {
                cv::destroyWindow(windowName);
            }
        }
        catch (const cv::Exception& e) {
            LOG_OUTPUT(Warn, ("Warning: Failed to destroy window '" + windowName + "': " + e.what()).c_str());
        }
    }

    return true;
}

/**
 * @brief 执行相机标定计算
 * @param calibrateCameraFlags 标定算法控制标志位
 * @param calibrateCameraC 算法终止条件
 * @return true-标定成功 false-标定失败
 * 
 * 本函数执行完整的相机标定流程：
 * 1. 数据准备：检查角点数据和标定板参数有效性
 * 2. 执行标定：调用OpenCV的calibrateCamera算法
 * 3. 结果存储：保存内参矩阵、畸变系数、外参向量等
 * 4. 精度评估：计算重投影误差RMS并记录
 * 
 * @note 需要调用角点检测函数(addCorners)和标定板坐标生成函数(addObjectCorners)进行数据准备
 * @note 标定成功RMS值应小于2.0像素，过大表示标定精度不足
 */
bool CameraCalibrator::calibrate(int calibrateCameraFlags,
    const cv::TermCriteria& calibrateCameraC) {


    // 2. 执行相机标定
    cv::Mat intrinsics = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat distCoeffs = cv::Mat(1, 5, CV_32F, cv::Scalar::all(0));
    std::vector<cv::Mat> tvecsMat;
    std::vector<cv::Mat> rvecsMat;

    double rms = -1.0;
    try {
        rms = cv::calibrateCamera(
            m_objectCornersVec, m_imageCornersVec, m_imageSize,
            m_calibrateResult.intrinsics, m_calibrateResult.distCoeffs,
            m_calibrateResult.rvecsMat, m_calibrateResult.tvecsMat,
            calibrateCameraFlags, calibrateCameraC
        );
    }
    catch (const cv::Exception& e) {
        LOG_OUTPUT(Error, e.what());
        return false;
    }

    if (rms < 0) {
        LOG_OUTPUT(Error, "camera calibrate failed");
        return false;
    }
    m_err = (float)rms;

    LOG_OUTPUT(Info, "sucessfully calibrate camera");
    LOG_OUTPUT(Info, ("RMS reprojection error: " + std::to_string(rms) + " pixels").c_str());
    //std::cout << "intrinsics:\n" << m_calibrateResult.intrinsics << std::endl;
    //std::cout << "distCoeffs:\n" << m_calibrateResult.distCoeffs << std::endl;
    LOG_OUTPUT(Info, ("RMS: " + std::to_string(rms)).c_str());

    double fx = 0, fy = 0, cx = 0, cy = 0;
    if (m_calibrateResult.intrinsics.depth() == CV_64F) {
        fx = m_calibrateResult.intrinsics.at<double>(0, 0);
        fy = m_calibrateResult.intrinsics.at<double>(1, 1);
        cx = m_calibrateResult.intrinsics.at<double>(0, 2);
        cy = m_calibrateResult.intrinsics.at<double>(1, 2);
    }
    else {
        fx = m_calibrateResult.intrinsics.at<float>(0, 0);
        fy = m_calibrateResult.intrinsics.at<float>(1, 1);
        cx = m_calibrateResult.intrinsics.at<float>(0, 2);
        cy = m_calibrateResult.intrinsics.at<float>(1, 2);
    }
    std::string intLog = "Intrinsics: fx=" + std::to_string(fx) + " fy=" + std::to_string(fy) +
        " cx=" + std::to_string(cx) + " cy=" + std::to_string(cy);
    LOG_OUTPUT(Info, intLog.c_str());
    
    return true;
}

bool CameraCalibrator::calReprojectionError()
{
    // 1. 输入验证
    if (m_objectCornersVec.empty() || m_imageCornersVec.empty()) {
        LOG_OUTPUT(Error, "Object points or image corner points data is empty");
        return false;
    }
    m_reprojectionCornersVec.clear();
    m_calibrateResult.errForImage.clear();

    std::vector<cv::Point2f> reprojectionCorners;
    float total_err = 0.0;
    for (size_t i = 0; i < m_imageCornersVec.size(); i++) {
        cv::projectPoints(m_objectCornersVec[i], m_calibrateResult.rvecsMat[i],
            m_calibrateResult.tvecsMat[i], m_calibrateResult.intrinsics,
            m_calibrateResult.distCoeffs, reprojectionCorners);
        m_reprojectionCornersVec.push_back(reprojectionCorners);
        float errForImage = 0.0;

        for (size_t j = 0; j < m_imageCornersVec[i].size(); j++) {

            errForImage += cv::norm(reprojectionCorners[j] - m_imageCornersVec[i][j]);
        }
        m_calibrateResult.errForImage.push_back(errForImage / m_imageCornersVec[i].size());
        total_err += errForImage;
    }

    m_err = total_err / m_imageCornersVec.size();

    return true;
}

/**
 * @brief 从配置文件夹加载所有标定图像
 * @return true-加载成功 false-加载失败
 * 
 * 本函数执行以下操作：
 * 1. 检查图片文件夹路径是否有效
 * 2. 扫描文件夹查找所有支持的图片格式（jpg, jpeg, png, bmp, tif, tiff）
 * 3. 对图片文件进行自然排序（考虑数字序列）
 * 4. 逐一读取并验证图片有效性
 * 5. 检查图片尺寸和类型一致性
 * 
 * @note 要求至少3幅有效图片才能进行标定
 * @note 图片应包含完整的标定板且避免过曝/欠曝
 * @note 支持批量加载，自动跳过损坏的图片文件
 */
bool CameraCalibrator::addImages()
{
    if (m_imagesFolder.empty()) {
        LOG_OUTPUT(Error, "Image folder path not set");
        return false;
    }
    m_images.clear();

    // 获取所有支持的图片文件
    std::vector<cv::String> imagePaths;
    const std::vector<std::string> supportedExtensions = {
        "/*.jpg", "/*.jpeg", "/*.png", "/*.bmp", "/*.tif", "/*.tiff"
    };

    try {
        for (const auto& ext : supportedExtensions) {
            std::vector<cv::String> currentPaths;
            cv::glob(m_imagesFolder + ext, currentPaths, false);
            imagePaths.insert(imagePaths.end(), currentPaths.begin(), currentPaths.end());
        }

        // 合并结果并进行自然排序
        std::sort(imagePaths.begin(), imagePaths.end(),
            [](const cv::String& a, const cv::String& b) {
                return Utils::naturalCompare(a, b);
            });
    }
    catch (const cv::Exception& e) {
        LOG_OUTPUT(Error, ("Failed to read image folder: " + std::string(e.what()) +
            ", path: " + m_imagesFolder).c_str());
        return false;
    }

    if (imagePaths.empty()) {
        std::string msg("there is no images in path: " + m_imagesFolder);
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }


    std::cout << "Loading images in order:" << std::endl;
    for (const auto& path : imagePaths) {
        std::cout << path << std::endl;
    }
    // 加载并检查图片
    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::string msg("cannot load image, skip" + path);
            LOG_OUTPUT(Warn, msg.c_str());
            continue;
        }

        // 检查第一张图片的尺寸和类型
        if (m_images.empty()) {
            m_imageSize = img.size();
            m_imageType = img.type();
        }
        // 检查后续图片是否匹配
        else if (img.size() != m_imageSize) {
            std::string msg(("images size mismatch, skip " + path +
                "，expect size:" + std::to_string(m_imageSize.width) +
                "x" + std::to_string(m_imageSize.height) +
                "，real size:" + std::to_string(img.cols) +
                "x" + std::to_string(img.rows)));
            LOG_OUTPUT(Warn, msg.c_str());
            continue;
        }
        else if (img.type() != m_imageType) {
            std::string msg("Skipping mismatched type image: " + path);
            LOG_OUTPUT(Warn, msg.c_str());
            continue;
        }
        m_images.push_back(std::move(img)); // 使用移动语义避免拷贝
        m_imagePaths.push_back(path);
    }

    if (m_images.empty()) {
        std::string msg("No valid images loaded");
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }

    // 可选：检查图片数量是否足够
    const size_t minImagesRequired = 3;
    if (m_images.size() < minImagesRequired) {
        std::string msg(("Only(" + std::to_string(m_images.size()) +
            ") images loaded (minimum" + std::to_string(minImagesRequired) +
            ") recommended"));
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }
    std::string msg("Info: Successfully loaded " + std::to_string(m_images.size()) + " images");
    LOG_OUTPUT(Info, msg.c_str());
    return true;
}

/**
 * @brief 检查标定结果矩阵尺寸是否正确
 * @return true-尺寸正确 false-尺寸错误
 */
inline bool CameraCalibrator::checkResultSize()
{
    // 检查内参矩阵尺寸
    if (m_calibrateResult.intrinsics.rows != 3 || m_calibrateResult.intrinsics.cols != 3) {
        LOG_OUTPUT(Error, "Camera intrinsic matrix size error, should be 3x3 matrix");
        return false;
    }

    // 检查畸变系数矩阵尺寸
    const int distCoeffsCount = m_calibrateResult.distCoeffs.total();
    if (distCoeffsCount != 5 && distCoeffsCount != 8) {
        std::string msg("Distortion coefficients matrix size error, should be 5 or 8 elements, actual: " + std::to_string(distCoeffsCount));
        LOG_OUTPUT(Error, msg.c_str());
        return false;
    }

    return true;
}

/**
 * @brief 检查矩阵类型一致性并确定主类型
 * @param matrixs 需要检查的矩阵列表
 * @param[out] mainType 输出的主类型(CV_32F或CV_64F)
 * @return true-检查通过 false-检查失败
 */
inline bool CameraCalibrator::checkType(const std::vector<cv::Mat>& matrixs, int& mainType)
{
    if (matrixs.empty()) {
        LOG_OUTPUT(Error, "Input matrix list is empty");
        return false;
    }

    // 检查所有矩阵是否为CV_32F或CV_64F类型
    for (const auto& matrix : matrixs) {
        const int type = matrix.type();
        if (type != CV_32F && type != CV_64F) {
            std::string msg("Unsupported data type (" + std::to_string(type) + ") found, only CV_32F or CV_64F supported");
            LOG_OUTPUT(Error, msg.c_str());
            return false;
        }
    }

    // 确定主类型(只要有一个是CV_64F就使用CV_64F)
    mainType = CV_32F;
    for (const auto& matrix : matrixs) {
        if (matrix.type() == CV_64F) {
            mainType = CV_64F;
            break;
        }
    }

    return true;
}

std::vector<cv::Mat> CameraCalibrator::getDraws() const {
    return m_debugDraws;
}

CalibrateResult CameraCalibrator::getCalibrateResult() const {
    return m_calibrateResult;
}



float CameraCalibrator::getError() const {
    return m_err;
}



bool CameraCalibrator::pixelToWorld(const cv::Point2f& undistoredPoint,
    const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3f& worldPoint) {
    double fx = m_calibrateResult.intrinsics.at<double>(0, 0);
    double fy = m_calibrateResult.intrinsics.at<double>(1, 1);
    double cx = m_calibrateResult.intrinsics.at<double>(0, 2);
    double cy = m_calibrateResult.intrinsics.at<double>(1, 2);
    cv::Mat R;
    try {
        cv::Rodrigues(rvec, R);
    }
    catch (cv::Exception& e) {
        LOG_OUTPUT(Error, e.what());
    }
    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        R.at<double>(0, 0), R.at<double>(0, 1), (undistoredPoint.x - cx) / fx,
        R.at<double>(1, 0), R.at<double>(1, 1), (undistoredPoint.y - cy) / fy,
        R.at<double>(2, 0), R.at<double>(2, 1), 1);
    cv::Mat b = (cv::Mat_<double>(3, 1) << -1.0 * tvec.at<double>(0),
        -1.0 * tvec.at<double>(1), -1.0 * tvec.at<double>(2));
    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt);
    cv::Mat sigma_inv_mat = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) {
        if (w.at<double>(i) > 1e-6) {
            sigma_inv_mat.at<double>(i, i) = 1.0f / w.at<double>(i);
        }
    }
    cv::Mat X = vt.t() * sigma_inv_mat * u.t() * b;
    worldPoint = cv::Point3f(
        static_cast<float>(X.at<double>(0)),
        static_cast<float>(X.at<double>(1)),
        0.0
    );
    return true;
}


bool CameraCalibrator::calcWorldPoints()
{
    std::vector<cv::Point2f> undistortedPoints;
    cv::Point3f worldPoint;
    m_worldPointsVec.clear();
    m_rvecs_base2camera.clear();
    m_tvecs_base2camera.clear();
    for (size_t i = 0; i < m_imageCornersVec.size(); ++i) {
        try {
            cv::undistortPoints(m_imageCornersVec[i], undistortedPoints, m_calibrateResult.intrinsics,
                m_calibrateResult.distCoeffs, cv::noArray(), m_calibrateResult.intrinsics);
        }
        catch (const cv::Exception& e) {
            LOG_OUTPUT(Error, ("undistortPoints failed: " + std::string(e.what())).c_str());
            return false;
        }
        cv::Mat rvec, tvec;
        cv::solvePnP(m_objectCornersVec[i], undistortedPoints, m_calibrateResult.intrinsics,
            m_calibrateResult.distCoeffs, rvec, tvec);
        std::vector<cv::Point3f> worldPoints;
        for (size_t j = 0; j < undistortedPoints.size(); ++j) {
            pixelToWorld(undistortedPoints[j], rvec, tvec,
                worldPoint);
            worldPoints.push_back(worldPoint);
        }
        m_worldPointsVec.push_back(worldPoints);
    }
    return true;
}

bool CameraCalibrator::calcWorldPointsWithoutPnP()
{
    std::vector<cv::Point2f> undistortedPoints;
    cv::Point3f worldPoint;
    m_worldPointsVec.clear();
    m_rvecs_base2camera.clear();
    m_tvecs_base2camera.clear();
    for (size_t i = 0; i < m_imageCornersVec.size(); ++i) {
        try {
            cv::undistortPoints(m_imageCornersVec[i], undistortedPoints, m_calibrateResult.intrinsics,
                m_calibrateResult.distCoeffs, cv::noArray(), m_calibrateResult.intrinsics);
        }
        catch (const cv::Exception& e) {
            LOG_OUTPUT(Error, ("undistortPoints failed: " + std::string(e.what())).c_str());
            return false;
        }
        std::vector<cv::Point3f> worldPoints;
        for (size_t j = 0; j < undistortedPoints.size(); ++j) {
            pixelToWorld(undistortedPoints[j], m_calibrateResult.rvecsMat[i], m_calibrateResult.tvecsMat[i],
                worldPoint);
            worldPoints.push_back(worldPoint);
        }
        m_worldPointsVec.push_back(worldPoints);
    }
    return true;
}

bool CameraCalibrator::calcReprojectionErrorReal()
{
    // 1. 参数校验
    if (m_worldPointsVec.empty()) {
        LOG_OUTPUT(Error, "World points vector is empty");
        return false;
    }

    // 2. 清空旧数据
    m_perImageErrors.clear();
    m_averageError = 0.0;

    // 3. 逐图计算误差
    double totalError = 0.0;
    size_t totalPoints = 0;

    for (size_t i = 0; i < m_worldPointsVec.size(); ++i) {
        const auto& worldPoints = m_worldPointsVec[i];
        const auto& objectCorners = m_objectCornersVec[i];

        if (worldPoints.size() != objectCorners.size()) {
            //LogAlgoRunError(("第" + std::to_string(i + 1) + "张图的点数量不匹配").c_str());
            LOG_OUTPUT(Error, "worldPointsVec or objectCornersVec size is not match");
            continue;
        }

        // 计算单图误差
        double imageError = 0.0;
        for (size_t j = 0; j < worldPoints.size(); ++j) {
            // 计算欧氏距离
            double dx = worldPoints[j].x - objectCorners[j].x;
            double dy = worldPoints[j].y - objectCorners[j].y;
            double dz = worldPoints[j].z - objectCorners[j].z;
            imageError += std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        // 记录单图平均误差
        double avgImageError = imageError / worldPoints.size();
        m_perImageErrors.push_back(avgImageError);
        totalError += imageError;
        totalPoints += worldPoints.size();

        // 调试输出
        std::string msg("Image" + std::to_string(i + 1) + std::to_string(avgImageError) + " (total points: " + std::to_string(worldPoints.size()) + ")");
        LOG_OUTPUT(Info, msg.c_str());
    }

    // 4. 计算全局平均误差
    if (totalPoints > 0) {
        m_averageError = totalError / totalPoints;
        std::cout << "Average reprojection error: " << m_averageError
            << " (total images: " << m_worldPointsVec.size()
            << ", total points: " << totalPoints << ")" << std::endl;
    }
    else {
        //LogAlgoRunError("没有有效数据点计算误差");
        std::cerr << "no valid data point to calculate error" << std::endl;
        return false;
    }

    return true;
}

bool CameraCalibrator::calibrateCameraWithCharuco()
{
    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::Size imageSize;

    m_debugDraws.clear();
    m_imageCornersVec.clear();
    m_objectCornersVec.clear();
    m_validIndices.clear();

    // 打印Charuco板的配置信息
    std::cout << "\n======== Charuco Board Configuration ========" << std::endl;
    std::cout << "Board size: " << m_board->getChessboardSize().width << "x" << m_board->getChessboardSize().height << std::endl;
    std::cout << "Square size: " << m_board->getSquareLength() << std::endl;
    std::cout << "Marker size: " << m_board->getMarkerLength() << std::endl;
    std::cout << "Total images: " << m_images.size() << std::endl;

    int validImageCount = 0;
    int imageIndex = 0;
    
    // Configure detector parameters for subpixel refinement
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; 
    parameters->polygonalApproxAccuracyRate = 0.05;
    parameters->adaptiveThreshWinSizeStep = 2;
    parameters->cornerRefinementMaxIterations = 100;
    parameters->cornerRefinementMinAccuracy = 0.01;

    for (const cv::Mat& img : m_images) {
        std::vector<int> markerIds;

        cv::Mat img_filtered;
        cv::medianBlur(img, img_filtered, 3);

        cv::aruco::detectMarkers(img_filtered, m_board->dictionary, markerCorners, markerIds, parameters);
        if (markerIds.empty()) {

            std::string msg = "No marker detected, skip image " + std::to_string(imageIndex);
            if (imageIndex < m_imagePaths.size()) {
                msg += " (" + m_imagePaths[imageIndex] + ")";
            }
            LOG_OUTPUT(Warn, msg.c_str());
            imageIndex++;
            continue;
        }
        m_calibrateResult.markerCornersVec.push_back(markerCorners);

        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img_filtered, m_board,
            charucoCorners, charucoIds);
            
        // 对插值结果进行亚像素级极限二次优化，强制使用纯净无模糊的原始图像 img 的灰度图
        if (!charucoCorners.empty()) {
            cv::Mat gray;
            if (img.channels() == 3) {
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = img;
            }
            cv::TermCriteria subpixCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-4);
            cv::cornerSubPix(gray, charucoCorners, cv::Size(5, 5), cv::Size(-1, -1), subpixCriteria);
        }
            
        if (charucoIds.empty()) {
            std::string msg = "No charuco corners, skip image " + std::to_string(imageIndex);
            if (imageIndex < m_imagePaths.size()) {
                msg += " (" + m_imagePaths[imageIndex] + ")";
            }
            LOG_OUTPUT(Warn, msg.c_str());
            imageIndex++;
            continue;
        }

        // 打印检测到的角点信息
        std::cout << "Image " << validImageCount + 1 << ": detected "
            << charucoCorners.size() << " Charuco corners" << std::endl;

        m_calibrateResult.chessCornersVec.push_back(charucoCorners);
        allCharucoCorners.push_back(charucoCorners);
        allCharucoIds.push_back(charucoIds);
        
        // Populate member vectors for external use (e.g. HandEye Calibration)
        m_imageCornersVec.push_back(charucoCorners);
        std::vector<cv::Point3f> currentObjectPoints;
        for(int id : charucoIds) {
            if(id >= 0 && id < m_board->chessboardCorners.size()) {
                currentObjectPoints.push_back(m_board->chessboardCorners[id]);
            }
        }
        m_objectCornersVec.push_back(currentObjectPoints);
        m_validIndices.push_back(imageIndex);

        imageSize = img.size();
        validImageCount++;
        imageIndex++;

        cv::Mat debugImage = img.clone();
        try {
            cv::aruco::drawDetectedMarkers(debugImage, markerCorners, markerIds);
            if (!charucoCorners.empty()) {
                cv::aruco::drawDetectedCornersCharuco(debugImage, charucoCorners, charucoIds);
            }
        }
        catch (const cv::Exception& e) {
            std::string msg(e.what());
            LOG_OUTPUT(Error, msg.c_str());
            msg = "Image type: " + std::to_string(debugImage.type());
            LOG_OUTPUT(Error, msg.c_str());
        }
        m_debugDraws.push_back(debugImage);
    }

    if (allCharucoCorners.empty()) {
        LOG_OUTPUT(Error, "No valid ChArUco corners detected in any image");
        return false;
    }

    std::cout << "\nCalibrating with " << validImageCount << " valid images..." << std::endl;
    std::cout << "Image size: " << imageSize.width << "x" << imageSize.height << std::endl;

    // 设置标定参数
    int calibrationFlags = cv::CALIB_RATIONAL_MODEL;  // 启用有理模型，拟合 k4, k5, k6
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-9); // 暴力迭代1000次

    double reprojError = cv::aruco::calibrateCameraCharuco(
        allCharucoCorners, allCharucoIds, m_board, imageSize,
        m_calibrateResult.intrinsics, m_calibrateResult.distCoeffs,
        m_calibrateResult.rvecsMat, m_calibrateResult.tvecsMat,
        calibrationFlags, criteria
    );
    m_err = (float)reprojError;

    std::cout << "\n======== Calibration Results ========" << std::endl;
    std::cout << "Reprojection error: " << reprojError << " pixels" << std::endl;
    std::cout << "Intrinsics:\n" << m_calibrateResult.intrinsics << std::endl;
    std::cout << "Distortion coefficients:\n" << m_calibrateResult.distCoeffs << std::endl;

    LOG_OUTPUT(Info, ("RMS reprojection error: " + std::to_string(reprojError) + " pixels").c_str());

    if (reprojError > 1.0) {
        std::string msg("High reprojection error detected: " + std::to_string(reprojError) + " pixels");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    return reprojError >= 0;
}

bool CameraCalibrator::calibrateCameraWithAruco()
{
    // These will be the input for calibrateCamera
    std::vector<std::vector<cv::Point2f>> allCorners;
    std::vector<std::vector<cv::Point3f>> allObjectPoints;

    cv::Size imageSize;

    // From detectMarkers
    std::vector<std::vector<cv::Point2f>> markerCorners;

    // Create the 3D object points for a single marker
    std::vector<cv::Point3f> singleMarkerObjectPoints;
    singleMarkerObjectPoints.push_back(cv::Point3f(-m_markerLength / 2.f, m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(m_markerLength / 2.f, m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(m_markerLength / 2.f, -m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(-m_markerLength / 2.f, -m_markerLength / 2.f, 0));

    m_debugDraws.clear();
    m_imageCornersVec.clear();
    m_objectCornersVec.clear();
    m_validIndices.clear();
    int validImageCount = 0;
    int imageIndex = 0;
    
    // Configure detector parameters for subpixel refinement
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; 

    for (const cv::Mat& img : m_images) {
        std::vector<int> markerIds;

        cv::aruco::detectMarkers(img, m_dictionary, markerCorners, markerIds, parameters);

        if (!markerIds.empty()) {
            validImageCount++;
            std::vector<cv::Point2f> currentImageCorners;
            std::vector<cv::Point3f> currentImageObjectPoints;

            for (size_t i = 0; i < markerIds.size(); i++) {
                for (int j = 0; j < 4; ++j) {
                    currentImageCorners.push_back(markerCorners[i][j]);
                    currentImageObjectPoints.push_back(singleMarkerObjectPoints[j]);
                }
            }
            allCorners.push_back(currentImageCorners);
            allObjectPoints.push_back(currentImageObjectPoints);
            
            // Populate member vectors
            m_imageCornersVec.push_back(currentImageCorners);
            m_objectCornersVec.push_back(currentImageObjectPoints);
            m_validIndices.push_back(imageIndex);
        }
        else
        {
            std::string msg = "No markers detected in image " + std::to_string(imageIndex);
            if (imageIndex < m_imagePaths.size()) {
                msg += " (" + m_imagePaths[imageIndex] + ")";
            }
            LOG_OUTPUT(Warn, msg.c_str());
        }

        imageSize = img.size();
        imageIndex++;

        // For debugging
        cv::Mat debugImage = img.clone();
        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(debugImage, markerCorners, markerIds);
        }
        m_debugDraws.push_back(debugImage);
    }

    if (validImageCount < 4) { // Need at least 4 views for calibration
        LOG_OUTPUT(Error, "Not enough views for calibration");
        return false;
    }

    // Now calibrate
    int calibrationFlags = cv::CALIB_RATIONAL_MODEL;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-9);

    double rms = cv::calibrateCamera(allObjectPoints, allCorners, imageSize,
        m_calibrateResult.intrinsics, m_calibrateResult.distCoeffs,
        m_calibrateResult.rvecsMat, m_calibrateResult.tvecsMat,
        calibrationFlags, criteria);
    m_err = (float)rms;

    std::cout << "\n======== Aruco Calibration Results ========" << std::endl;
    std::cout << "Reprojection error: " << rms << " pixels" << std::endl;
    std::cout << "Intrinsics:\n" << m_calibrateResult.intrinsics << std::endl;
    std::cout << "Distortion coefficients:\n" << m_calibrateResult.distCoeffs << std::endl;

    LOG_OUTPUT(Info, ("RMS reprojection error: " + std::to_string(rms) + " pixels").c_str());

    if (rms > 1.0) {
        std::string msg("High reprojection error detected: " + std::to_string(rms) + " pixels");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    return rms >= 0;
}

bool CameraCalibrator::calibrateCameraWithArucoBoard()
{
    std::vector<std::vector<cv::Point2f>> allCorners;
    std::vector<std::vector<cv::Point3f>> allObjectPoints;
    cv::Size imageSize;

    m_debugDraws.clear();
    m_imageCornersVec.clear();
    m_objectCornersVec.clear();
    m_validIndices.clear();

    std::cout << "\n======== ArUco Board Configuration ========" << std::endl;
    std::cout << "Board size: " << m_arucoBoard->getGridSize().width << "x" << m_arucoBoard->getGridSize().height << std::endl;
    std::cout << "Marker size: " << m_arucoBoard->getMarkerLength() << std::endl;
    std::cout << "Marker separation: " << m_arucoBoard->getMarkerSeparation() << std::endl;
    std::cout << "Total images: " << m_images.size() << std::endl;

    int validImageCount = 0;
    int imageIndex = 0;

    // Configure detector parameters for subpixel refinement
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; 

    for (const cv::Mat& img : m_images) {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;

        cv::aruco::detectMarkers(img, m_dictionary, markerCorners, markerIds, parameters);

        if (!markerIds.empty()) {
            // Refine detected markers (optional but recommended for boards)
            cv::aruco::refineDetectedMarkers(img, m_arucoBoard, markerCorners, markerIds, cv::noArray());

            // Collect data for this image
            std::vector<cv::Point2f> currentImageCorners;
            std::vector<cv::Point3f> currentObjectPoints;
            
            // Match detected markers to board object points
            // GridBoard::objPoints is a vector<vector<Point3f>> where index corresponds to marker ID in the dictionary?
            // Actually, GridBoard stores ids and objPoints.
            // We need to find which marker in the board corresponds to the detected markerId.
            
            // In OpenCV 4.x, Board::ids contains the ids of the markers in the board.
            // Board::objPoints contains the object points for each marker in the board.
            
            const std::vector<int>& boardIds = m_arucoBoard->ids;
            const std::vector<std::vector<cv::Point3f>>& boardObjPoints = m_arucoBoard->objPoints;

            for (size_t i = 0; i < markerIds.size(); ++i) {
                int detectedId = markerIds[i];
                
                // Find this id in the board
                auto it = std::find(boardIds.begin(), boardIds.end(), detectedId);
                if (it != boardIds.end()) {
                    size_t boardIdx = std::distance(boardIds.begin(), it);
                    
                    // Append corners and object points
                    const auto& corners = markerCorners[i];
                    const auto& objPts = boardObjPoints[boardIdx];
                    
                    currentImageCorners.insert(currentImageCorners.end(), corners.begin(), corners.end());
                    currentObjectPoints.insert(currentObjectPoints.end(), objPts.begin(), objPts.end());
                }
            }

            if (!currentImageCorners.empty()) {
                validImageCount++;
                allCorners.push_back(currentImageCorners);
                allObjectPoints.push_back(currentObjectPoints);
                
                m_imageCornersVec.push_back(currentImageCorners);
                m_objectCornersVec.push_back(currentObjectPoints);
                m_validIndices.push_back(imageIndex);
            } else {
                 std::string msg = "No valid board markers detected in image " + std::to_string(imageIndex);
                 LOG_OUTPUT(Warn, msg.c_str());
            }
        } else {
            std::string msg = "No markers detected in image " + std::to_string(imageIndex);
            if (imageIndex < m_imagePaths.size()) {
                msg += " (" + m_imagePaths[imageIndex] + ")";
            }
            LOG_OUTPUT(Warn, msg.c_str());
        }

        imageSize = img.size();
        imageIndex++;

        // Debug draw
        cv::Mat debugImage = img.clone();
        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(debugImage, markerCorners, markerIds);
        }
        m_debugDraws.push_back(debugImage);
    }

    if (validImageCount < 3) {
        LOG_OUTPUT(Error, "Not enough valid images for ArUco Board calibration (min 3)");
        return false;
    }

    std::cout << "\nCalibrating with " << validImageCount << " valid images..." << std::endl;

    int calibrationFlags = cv::CALIB_RATIONAL_MODEL;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-9);

    double rms = cv::calibrateCamera(allObjectPoints, allCorners, imageSize,
        m_calibrateResult.intrinsics, m_calibrateResult.distCoeffs,
        m_calibrateResult.rvecsMat, m_calibrateResult.tvecsMat,
        calibrationFlags, criteria);
    m_err = (float)rms;

    std::cout << "\n======== ArUco Board Calibration Results ========" << std::endl;
    std::cout << "Reprojection error: " << rms << " pixels" << std::endl;
    std::cout << "Intrinsics:\n" << m_calibrateResult.intrinsics << std::endl;
    std::cout << "Distortion coefficients:\n" << m_calibrateResult.distCoeffs << std::endl;

    LOG_OUTPUT(Info, ("RMS reprojection error: " + std::to_string(rms) + " pixels").c_str());

    if (rms > 1.0) {
        std::string msg("High reprojection error detected: " + std::to_string(rms) + " pixels");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    return rms >= 0;
}

bool CameraCalibrator::calibrateCameraWithAprilTag()
{
    // These will be the input for calibrateCamera
    std::vector<std::vector<cv::Point2f>> allCorners;
    std::vector<std::vector<cv::Point3f>> allObjectPoints;

    cv::Size imageSize;

    // From detectMarkers
    std::vector<std::vector<cv::Point2f>> markerCorners;

    // Create the 3D object points for a single marker
    std::vector<cv::Point3f> singleMarkerObjectPoints;
    singleMarkerObjectPoints.push_back(cv::Point3f(-m_markerLength / 2.f, m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(m_markerLength / 2.f, m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(m_markerLength / 2.f, -m_markerLength / 2.f, 0));
    singleMarkerObjectPoints.push_back(cv::Point3f(-m_markerLength / 2.f, -m_markerLength / 2.f, 0));

    m_debugDraws.clear();
    m_imageCornersVec.clear();
    m_objectCornersVec.clear();
    m_validIndices.clear();
    int validImageCount = 0;
    int imageIndex = 0;

    // Aadd detector parameters for robustness
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;

    for (const cv::Mat& img : m_images) {
        std::vector<int> markerIds;

        cv::aruco::detectMarkers(img, m_dictionary, markerCorners, markerIds, parameters);

        if (!markerIds.empty()) {
             validImageCount++;
             std::vector<cv::Point2f> currentImageCorners;
             std::vector<cv::Point3f> currentImageObjectPoints;

             for (size_t i = 0; i < markerIds.size(); i++) {
                 for (int j = 0; j < 4; ++j) {
                     currentImageCorners.push_back(markerCorners[i][j]);
                     currentImageObjectPoints.push_back(singleMarkerObjectPoints[j]);
                 }
             }
             allCorners.push_back(currentImageCorners);
             allObjectPoints.push_back(currentImageObjectPoints);
             
             // Populate member vectors
             m_imageCornersVec.push_back(currentImageCorners);
             m_objectCornersVec.push_back(currentImageObjectPoints);
             m_validIndices.push_back(imageIndex);
         } else {
             std::string msg = "No AprilTags detected in image " + std::to_string(imageIndex);
             if (imageIndex < m_imagePaths.size()) {
                 msg += " (" + m_imagePaths[imageIndex] + ")";
             }
             LOG_OUTPUT(Warn, msg.c_str());
         }

        imageSize = img.size();
        imageIndex++;

        // For debugging
        cv::Mat debugImage = img.clone();
        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(debugImage, markerCorners, markerIds);
        }
        m_debugDraws.push_back(debugImage);
    }

    if (validImageCount < 4) { // Need at least 4 views for calibration
        LOG_OUTPUT(Error, "Not enough views for AprilTag calibration");
        return false;
    }

    // Now calibrate
    int calibrationFlags = cv::CALIB_RATIONAL_MODEL;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1e-9);

    double rms = cv::calibrateCamera(allObjectPoints, allCorners, imageSize,
        m_calibrateResult.intrinsics, m_calibrateResult.distCoeffs,
        m_calibrateResult.rvecsMat, m_calibrateResult.tvecsMat,
        calibrationFlags, criteria);
    m_err = (float)rms;

    std::cout << "\n======== AprilTag Calibration Results ========" << std::endl;
    std::cout << "Reprojection error: " << rms << " pixels" << std::endl;
    std::cout << "Intrinsics:\n" << m_calibrateResult.intrinsics << std::endl;
    std::cout << "Distortion coefficients:\n" << m_calibrateResult.distCoeffs << std::endl;

    LOG_OUTPUT(Info, ("RMS reprojection error: " + std::to_string(rms) + " pixels").c_str());

    if (rms > 1.0) {
        std::string msg("High reprojection error detected: " + std::to_string(rms) + " pixels");
        LOG_OUTPUT(Warn, msg.c_str());
    }

    return rms >= 0;
}

std::vector<cv::Mat> CameraCalibrator::getImages() const {
    return m_images;
}

