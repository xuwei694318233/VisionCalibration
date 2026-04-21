# VisionCalibration

一个功能完整的相机标定和手眼标定库，支持多种标定板类型和标定方法。

## 功能特性

### 相机标定 (CameraCalibrator)

- 支持多种标定板类型：
  - 棋盘格 (Chessboard)
  - Charuco 标定板
  - ArUco 标记
  - AprilTag 标记
- 自动图像加载和角点检测
- 亚像素角点精化
- 内参矩阵和畸变系数计算
- 重投影误差计算和分析
- 图像去畸变处理
- 像素坐标到世界坐标转换

### 手眼标定 (HandEyeCalibrator)

- 支持眼在手 (Eye-in-Hand) 模式
- 支持眼在手外 (Eye-to-Hand) 模式
- 多种手眼标定算法 (Tsai, Park, Horaud 等)
- 非线性优化精化
- 重投影误差评估

### 2D 手眼标定 (HandEyeCalibrator2D)

- 基于 2D 像素坐标和机器人 pose (x, y, theta)
- 输出 3x3 仿射变换矩阵

### 2.5D 手眼标定 (HandEyeCalibrator25D)

- 结合 2D 图像特征和 Z 轴深度信息
- 输出标定参数 (cx, cy, kScale, S0)

### 工具函数 (Utils)

- 旋转矩阵与欧拉角、四元数转换
- 位姿矩阵操作 (R_T2H, H2R_T)
- 旋转矩阵有效性验证
- JSON 格式数据保存和加载

## 系统要求

- Windows 10/11
- Visual Studio 2022
- CMake 3.16 或更高版本
- OpenCV 4.6.0

## 目录结构

```
VisionCalibration/
├── CMakeLists.txt          # CMake 构建配置
├── build.bat              # Windows 编译脚本
├── README.md              # 项目文档
├── include/               # 头文件目录
│   ├── CalibrationConfig.h
│   ├── CameraCalibrator.h
│   ├── HandEyeCalibrator.h
│   ├── HandEyeCalibrator2D.h
│   ├── HandEyeCalibrator25D.h
│   └── Utils.h
├── src/                   # 源代码目录
│   ├── CameraCalibrator.cpp
│   ├── HandEyeCalibrator.cpp
│   ├── HandEyeCalibrator2D.cpp
│   ├── HandEyeCalibrator25D.cpp
│   └── Utils.cpp
├── test/                  # 测试代码目录
│   └── test_camera_calibrator.cpp
└── thirdparty/            # 第三方依赖
    ├── json/
    │   └── json.hpp
    └── opencv/
        └── include/
```

## 编译说明

### 使用 build.bat 脚本编译 (推荐)

1. 确保已安装 Visual Studio 2022 和 CMake，并将 CMake 添加到系统 PATH
2. 在项目根目录下运行 `build.bat`
3. 编译完成后，输出文件将位于 `build/bin/` 目录下

### 手动编译

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake 项目 (Visual Studio 2022)
cmake -G "Visual Studio 17 2022" -A x64 ..

# 编译 Release 版本
cmake --build . --config Release

# 编译 Debug 版本
cmake --build . --config Debug
```

## 运行测试

### 命令行参数

```bash
test_camera_calibrator.exe [图像文件夹] [棋盘格列数] [棋盘格行数] [方格尺寸]
```

参数说明：

- `图像文件夹` - 标定图像所在文件夹 (默认: calibration_images)
- `棋盘格列数` - 内部角点列数 (默认: 9)
- `棋盘格行数` - 内部角点行数 (默认: 6)
- `方格尺寸` - 方格物理尺寸 mm (默认: 25)

### 示例

```bash
# 使用默认参数
test_camera_calibrator.exe

# 指定自定义参数
test_camera_calibrator.exe my_images 11 8 30

# 运行测试
build\bin\Debug\test_camera_calibrator.exe
build\bin\Release\test_camera_calibrator.exe
```

### 测试内容

测试程序包含以下测试项：

1. 构造函数测试
2. 图像加载测试
3. 角点检测测试
4. 标定算法测试
5. 重投影误差测试
6. 像素坐标转换测试
7. 图像去畸变测试
8. 空图像处理测试
9. 手眼标定测试 (Eye-in-Hand)
10. 手眼标定测试 (Eye-to-Hand)
11. 2D 手眼标定测试
12. 2.5D 手眼标定测试
13. 工具函数测试
14. 性能测试

## 使用示例

### 相机标定示例

```cpp
#include <CameraCalibrator.h>

// 创建标定配置
CameraCalibConfig config;
config.plateType = PlateType::Chessboard;
config.boardSize = cv::Size(9, 6);  // 内部角点数
config.squareSize = cv::Size2f(20.0f, 20.0f);  // 方格尺寸 (mm)
config.imagesFolder = "./calibration_images";

// 创建标定器
CameraCalibrator calibrator(config);

// 加载图像
calibrator.addImages();

// 检测角点
calibrator.getCorners();

// 执行标定
calibrator.calibrate();

// 获取标定结果
CalibrateResult result = calibrator.getCalibrateResult();
cv::Mat intrinsics = result.intrinsics;
cv::Mat distCoeffs = result.distCoeffs;

// 获取平均重投影误差
float error = calibrator.getError();

// 图像去畸变
cv::Mat undistorted;
CameraCalibrator::undistortImage(inputImage, intrinsics, distCoeffs, undistorted);
```

### 手眼标定示例

```cpp
#include <HandEyeCalibrator.h>

// 创建手眼标定配置
HandEyeCalibConfig config;
config.mode = EYE_IN_HAND;
config.poseFile = "./robot_poses.txt";

// 配置相机标定参数
config.camConfig.plateType = PlateType::Chessboard;
config.camConfig.boardSize = cv::Size(9, 6);
config.camConfig.squareSize = cv::Size2f(20.0f, 20.0f);
config.camConfig.imagesFolder = "./calibration_images";

// 创建手眼标定器
HandEyeCalibrator handEyeCalibrator(config);

// 执行标定
handEyeCalibrator.calibrate();

// 获取标定结果 (4x4 齐次变换矩阵)
cv::Mat result = handEyeCalibrator.getResult();

// 获取重投影误差
double error = handEyeCalibrator.getReprojectionError();
```

### 2D 手眼标定示例

```cpp
#include <HandEyeCalibrator2D.h>

HandEyeCalibrator2D calibrator2D;

// 添加观测数据
for (int i = 0; i < n; i++) {
    cv::Point2f pixel(u, v);           // 像素坐标
    cv::Vec3f robotPose(x, y, theta);  // 机器人位姿 (x, y, theta)
    calibrator2D.addObservation(pixel, robotPose);
}

// 执行标定
calibrator2D.calibrate();

// 获取结果 (3x3 仿射矩阵)
cv::Mat affineMatrix = calibrator2D.getResult();
```

### 2.5D 手眼标定示例

```cpp
#include <HandEyeCalibrator25D.h>

HandEyeCalibrator25D calibrator25D;

// 添加观测数据
for (int i = 0; i < n; i++) {
    cv::Point2f center(cx, cy);  // 图像中心坐标
    double scale = 100.0;        // 特征尺度
    double robotZ = 500.0;       // 机器人 Z 轴位置
    calibrator25D.addObservation(center, scale, robotZ);
}

// 执行标定
calibrator25D.calibrate();

// 获取结果
auto result = calibrator25D.getResult();
std::cout << "cx=" << result.cx << " cy=" << result.cy << " kScale=" << result.kScale;
```

## 依赖说明

### OpenCV

项目使用 OpenCV 4.6.0，需要将 OpenCV 安装在 `thirdparty/opencv` 目录下。

```
thirdparty/opencv/
├── include/
│   └── opencv2/
│       └── opencv.hpp
└── bin/
    ├── Release/
    │   └── opencv_world460.dll
    └── Debug/
        └── opencv_world460d.dll
```

### nlohmann/json

项目使用本地 nlohmann/json 库，需要确保 `thirdparty/json/json.hpp` 文件存在。

```
thirdparty/json/
└── json.hpp
```

## 注意事项

1. 标定需要至少 3 幅有效图像，推荐使用 10-20 幅图像以获得更好的标定精度
2. 图像应包含完整标定板且避免过曝/欠曝
3. 手眼标定需要提供机器人位姿文件，格式为文本文件，每行一个位姿
4. 编译时会自动复制 OpenCV DLL 到输出目录
5. 测试程序需要用户提供真实的标定图像

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。
