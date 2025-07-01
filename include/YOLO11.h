#pragma once

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_utils.h>
#include <macros.h>
#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <preprocess.h>
#include <chrono>
#include <execution>
#include <filesystem>
#define NOMINMAX
// #include <Windows.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <cmath>
#include "AsyncLogger.h"

namespace fs = std::filesystem;

struct Box
{
	float x; // 深度
	float y; // 宽度方向（抓取点中心）
	float z; // 高度
	float angle;
	float width;
	float height;
};

struct Group
{
	float centerX;
	float centerY;
	float minZ;
	float groupLeft;
    float width;
	float height;
    bool isVerticalSuction;
	std::vector<Box> mBox;
	std::vector<bool> cupsEnabled;
};

struct TensorRT {

    std::string modelPath;

	std::unique_ptr<nvinfer1::ICudaEngine, std::function<void(nvinfer1::ICudaEngine*)>> engine;
	std::unique_ptr<nvinfer1::IExecutionContext, std::function<void(nvinfer1::IExecutionContext*)>> context;
	cudaStream_t stream;

	float* gpuBuffers[2];
	int numAttributes, numDetections, numClasses;
	int* hostNumAnchors;
	int* deviceNumAnchors;
	Detection* deviceFilteredDetections;
	std::vector<Detection> results;

    uint8_t* pinnedMem;
    uint8_t* deviceMem;
    uint16_t* depthMatHost;

    TensorRT() = default;
    ~TensorRT() {
        if (deviceFilteredDetections) {
            CUDA_CHECK(cudaFree(deviceFilteredDetections));
            deviceFilteredDetections = nullptr;
        }
        if (deviceNumAnchors) {
            CUDA_CHECK(cudaFree(deviceNumAnchors));
            deviceNumAnchors = nullptr;
        }
        if (hostNumAnchors) {
            CUDA_CHECK(cudaFreeHost(hostNumAnchors));
            hostNumAnchors = nullptr;
        }
        for (auto& buffer : gpuBuffers) {
            if (buffer) {
                CUDA_CHECK(cudaFree(buffer));
                buffer = nullptr;
            }
        }
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
};

class YOLO11 {
public:
    YOLO11(TensorRT& trt);

    ~YOLO11();

    std::vector<Detection> modelProcess(const std::pair<cv::Mat, cv::Mat>& pair);

    std::pair<int, std::vector<float>> detectProcess(cv::Mat& depthMat, std::vector<Detection> output, bool openVerify);

    // void draw(cv::Mat& img, std::vector<Detection>& output);
    void drawRotated(cv::Mat image, std::vector<Detection>& output);

    float getAngle(const cv::Mat& depthMat);

    float confThreshold_ = 0.5f;
    float nmsThreshold_  = 0.2f;
    int inputWidth_, inputHeight_, imageWidth_, imageHeight_;
    float scaleX_, scaleY_;
private:

    void lostReplenish(cv::Mat& depthMat, std::vector<Detection>& result);  //补漏
	static bool sortRuleWY(const Detection p1, const Detection p2);
	static bool sortRuleWZ(const std::vector<Detection> p1, const std::vector<Detection> p2);

    uint8_t** pinnedMemPtr_;
    uint8_t** deviceMemPtr_;
    uint16_t** depthMatHostPtr_;
    float* gpuBuffers_[2];

    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, std::function<void(nvinfer1::ICudaEngine*)>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, std::function<void(nvinfer1::IExecutionContext*)>> context_;

    int numAttributes_, numDetections_, numClasses_;
    float ratioH_, ratioW_;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    std::vector<Detection> thresholdResult_;
    int* deviceNumAnchors_;
    int* hostNumAnchors_;
    Detection* deviceFilteredDetections_;

    CameraParams params_;
    CameraParams* deviceParams_;
};
