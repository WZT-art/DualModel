#include <iostream>
#include <vector>
#include <chrono>

// Boost
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

// Eigen
#include <Eigen/Dense>

// #include <Windows.h>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>  // 可选：测试 cuBLAS

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

// cuDNN
#include <cudnn.h>

// pybind11
// #include <pybind11/embed.h>
// #include <pybind11/stl.h>

// ========== 1. Boost 测试 ==========
void test_boost() {
    std::cout << "=== Testing Boost ===" << std::endl;
    
    // 测试文件系统
    boost::filesystem::path dir("test_dir");
    if (!boost::filesystem::exists(dir)) {
        boost::filesystem::create_directory(dir);
        std::cout << "Created directory: " << dir << std::endl;
    }

    // 测试线程
    boost::thread t([]() {
        std::cout << "Boost thread is running!" << std::endl;
    });
    t.join();

    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << "[Boost] Current time: " << now << std::endl;

}

void test_pcl() {
    // 2. 创建简单点云（XYZ）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (float z = -1.0; z <= 1.0; z += 0.1f) {
        cloud->push_back(pcl::PointXYZ(0.0, 0.0, z));
    }
    std::cout << "[PCL] Original cloud points: " << cloud->size() << std::endl;

    // 3. VoxelGrid 滤波（体素滤波器）
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(0.2f, 0.2f, 0.2f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxel.filter(*cloud_filtered);
    std::cout << "[PCL] Filtered cloud points: " << cloud_filtered->size() << std::endl;

    // 4. 使用 KdTreeFLANN + 最近邻查找
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_filtered);
    pcl::PointXYZ search_point(0.0, 0.0, 0.0);

    std::vector<int> indices(3);
    std::vector<float> distances(3);
    if (kdtree.nearestKSearch(search_point, 3, indices, distances) > 0) {
        std::cout << "[Flann] Nearest neighbors to (0,0,0):" << std::endl;
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto& pt = cloud_filtered->points[indices[i]];
            std::cout << "  " << i << ": (" << pt.x << ", " << pt.y << ", " << pt.z 
                      << "), dist: " << distances[i] << std::endl;
        }
    }

    // 5. Eigen 矩阵计算测试
    Eigen::Matrix3f mat;
    mat << 1, 2, 3,
           0, 1, 4,
           5, 6, 0;
    Eigen::Matrix3f result = mat.inverse();
    std::cout << "[Eigen] Inverse of matrix:\n" << result << std::endl;
    std::cout<<"test pcl success!"<<std::endl;

}

// ========== 2. OpenCV 测试 ==========
void test_opencv() {
    std::cout << "\n=== Testing OpenCV ===" << std::endl;
    
    // 生成测试图像
    cv::Mat img(300, 300, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::circle(img, cv::Point(150, 150), 100, cv::Scalar(255, 0, 0), 2);
    
    // 显示图像
    cv::imshow("OpenCV Test", img);
    cv::waitKey(1000);
    cv::destroyAllWindows();

    // CUDA-accelerated OpenCV
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(img);
    cv::cuda::cvtColor(gpu_img, gpu_img, cv::COLOR_BGR2GRAY);
    std::cout << "OpenCV + CUDA test passed!" << std::endl;
}

// ========== 3. CUDA 测试（仅 Runtime API） ==========
void test_cuda() {
    std::cout << "\n=== Testing CUDA Runtime API ===" << std::endl;
    
    const int N = 5;
    float *h_a = new float[N]{1, 2, 3, 4, 5};
    float *h_b = new float[N]{10, 20, 30, 40, 50};
    float *h_c = new float[N];

    float *d_a, *d_b, *d_c;

    // 分配 Device 内存
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 拷贝数据到 Device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 使用 cuBLAS 计算点积（替代核函数测试）
    cublasHandle_t handle;
    cublasCreate(&handle);
    float dot_result = 0;
    cublasSdot(handle, N, d_a, 1, d_b, 1, &dot_result);
    cublasDestroy(handle);

    // 拷贝回 Host（仅示例，实际 dot_result 已在 Device 计算）
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "CUDA + cuBLAS dot product result: " << dot_result << std::endl;
    std::cout << "(Expected: 1*10 + 2*20 + 3*30 + 4*40 + 5*50 = 550)" << std::endl;

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ========== 4. TensorRT 测试 ==========
void test_tensorrt() {
    std::cout << "\n=== Testing TensorRT ===" << std::endl;

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    } logger;

    // 创建 Builder（需手动 delete）
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder!" << std::endl;
        return;
    }

    // 创建 Network（需手动 delete）
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::cerr << "Failed to create TensorRT network!" << std::endl;
        delete builder;  // 注意先释放 builder
        return;
    }

    // 构建一个简单的网络（输入 -> ReLU -> 输出）
    nvinfer1::ITensor* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::IActivationLayer* relu = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
    relu->getOutput(0)->setName("output");
    network->markOutput(*relu->getOutput(0));

    // 创建 Config（需手动 delete）
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "Failed to create TensorRT config!" << std::endl;
        delete network;
        delete builder;
        return;
    }

    // 构建 Engine（需手动 delete）
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine!" << std::endl;
    } else {
        std::cout << "TensorRT engine built successfully!" << std::endl;
        delete engine;  // 释放 engine
    }

    // 按依赖顺序释放对象
    delete config;
    delete network;
    delete builder;
}

// ========== 5. cuDNN 测试 ==========
void test_cudnn() {
    std::cout << "\n=== Testing cuDNN ===" << std::endl;

    cudnnHandle_t cudnn;
    if (cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuDNN handle!" << std::endl;
        return;
    }

    // 创建一个简单的 Tensor 描述符
    cudnnTensorDescriptor_t tensor_desc;
    cudnnCreateTensorDescriptor(&tensor_desc);
    cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    std::cout << "cuDNN initialized successfully!" << std::endl;

    cudnnDestroyTensorDescriptor(tensor_desc);
    cudnnDestroy(cudnn);
}

// namespace py = pybind11;

// ========== 主函数 ==========
int main(int argc, char* argv[]) {

    test_boost();
    test_pcl();
    test_opencv();
    test_cuda();
    test_tensorrt();
    test_cudnn();
    std::cout<<"Program is completed"<<std::endl;
    std::cout << "Exiting program..." << std::endl;
    return 0;
}

