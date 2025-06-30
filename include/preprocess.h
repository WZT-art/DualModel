/**
 * @file cuda_preprocess.h
 * @brief Header file for CUDA-based image preprocessing functions.
 *
 * This file contains functions for initializing, destroying, and running image preprocessing
 * using CUDA for accelerating operations like resizing and data format conversion.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

 /**
  * @brief Initialize CUDA resources for image preprocessing.
  *
  * Allocates resources and sets up the necessary environment for performing image preprocessing
  * on the GPU. This function should be called once before using any preprocessing functions.
  *
  * @param max_image_size The maximum image size (in pixels) that will be processed.
  */
void cuda_preprocess_init(int max_image_size,
                          uint8_t** pinnedMemPtr,
                          uint8_t** deviceMemPtr,
                          uint16_t** depthMatHostPtr);

/**
 * @brief Clean up and release CUDA resources.
 *
 * Frees any memory and resources allocated during initialization. This function should be
 * called when the preprocessing operations are no longer needed.
 */
void cuda_preprocess_destroy(uint8_t** pinnedMemPtr, uint8_t** deviceMemPtr, uint16_t** depthMatHostPtr);

/**
 * @brief Preprocess an image using CUDA.
 *
 * This function resizes and converts the input image data (from uint8 to float) using CUDA
 * for faster processing. The result is stored in a destination buffer, ready for inference.
 *
 * @param src Pointer to the source image data in uint8 format.
 * @param src_width The width of the source image.
 * @param src_height The height of the source image.
 * @param dst Pointer to the destination buffer to store the preprocessed image in float format.
 * @param dst_width The desired width of the output image.
 * @param dst_height The desired height of the output image.
 * @param stream The CUDA stream to execute the preprocessing operation asynchronously.
 */
void cuda_preprocess(int src_width,
                     int src_height,
                     float* dst,
                     int dst_width,
                     int dst_height,
                     cudaStream_t& stream,
                     uint8_t** pinnedMemPtr,
                     uint8_t** deviceMemPtr,
                     uint16_t** depthMatHostPtr);

// struct Detection {
//   float conf;
//   int classId = 0;
//   cv::RotatedRect rbox;  
//   int modelID = 0;
// };

struct Detection {
    float conf;
    int classId = 0;
    cv::RotatedRect rbox;  
    int modelID = 0;
    // cv::Point2f bottomPoint;
    float bottomX;
    float bottomY;
    float centerDepth;

    float camX;
    float camY;
    float camZ;

    float worldWidth;  // World coordinates
    float worldHeight; 
    float worldBX;   
    float worldBY;   
    float worldBZ;   
};

struct CameraParams {
    // 相机内参
    float fx, fy, cx, cy;
    
    // 相机外参 - 位置
    float tx, ty, tz;
    
    // 相机外参 - 旋转 (欧拉角，单位：度)
    float rx, ry, rz;
    
    // 旋转矩阵 (3x3)
    float R[9];
    
    // 旋转逆矩阵 (3x3)
    float Rinv[9];
};

void filterDetection (
    float* gpuDetections,
    int numAttributeSize,
    int numDetections,
    float confThreshold,
    cudaStream_t& stream,
    int* deviceNumAnchors,
    Detection* deviceFilteredDetections
);

void initCameraParams(CameraParams& paramsPtr);