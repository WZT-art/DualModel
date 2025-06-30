#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"

// Host and device pointers for image buffers
static uint8_t* img_buffer_host = nullptr;    // Pinned memory on the host for faster transfers
static uint8_t* img_buffer_device = nullptr;  // Memory on the device (GPU)
static uint8_t* pinnedMemPtr = nullptr;
static uint8_t* deviceMemPtr = nullptr;

// Structure to represent a 2x3 affine transformation matrix
struct AffineMatrix {
    float value[6]; // [m00, m01, m02, m10, m11, m12]
};

AffineMatrix s2d, d2s;
__constant__ CameraParams globalParams_;
__device__ uint16_t depthMatDevice_[1624*1240];  //! 1624*1240, 3248*2480

void initCameraParams(CameraParams& paramsPtr) {
    CUDA_CHECK(cudaMemcpyToSymbol(
        globalParams_,     
        &paramsPtr,         
        sizeof(CameraParams)
    ));  
    
    // debugPrintParams<<<1,1>>>();
    // cudaDeviceSynchronize(); // 确保打印完成
}

// CUDA kernel to perform affine warp on the image
__global__ void warpaffine_kernel(
    uint8_t* src,           // Source image on device
    int src_line_size,      // Number of bytes per source image row
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination image on device (output)
    int dst_width,          // Destination image width
    int dst_height,         // Destination image height
    uint8_t const_value_st, // Constant value for out-of-bound pixels  
    AffineMatrix d2s_device,       // Affine transformation matrix (destination to source)
    int edge                // Total number of pixels to process
) {
    
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return; // Exit if position exceeds total pixels

    // Extract affine matrix elements
    float m_x1 = d2s_device.value[0];
    float m_y1 = d2s_device.value[1];
    float m_z1 = d2s_device.value[2];
    float m_x2 = d2s_device.value[3];
    float m_y2 = d2s_device.value[4];
    float m_z2 = d2s_device.value[5];

    // Calculate the destination coordinates
    int dx = position % dst_width;
    int dy = position / dst_width;

    // Apply affine transformation to get source coordinates
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2; // Color channels (B, G, R)

    // Check if the source coordinates are out of bounds
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // Assign constant value if out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        // Perform bilinear interpolation

        // Get the integer parts of the source coordinates
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // Initialize constant values for out-of-bound pixels
        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };

        // Calculate the fractional parts
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        // Compute the weights for the four surrounding pixels
        float w1 = hy * hx; // Top-left
        float w2 = hy * lx; // Top-right
        float w3 = ly * hx; // Bottom-left
        float w4 = ly * lx; // Bottom-right

        // Initialize pointers to the four surrounding pixels
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        // Top-left pixel
        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;
            // Top-right pixel
            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        // Bottom-left and Bottom-right pixels
        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // Perform bilinear interpolation for each color channel
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]; // Blue
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]; // Green
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]; // Red
    }

    // Convert from BGR to RGB by swapping channels
    float t = c2;
    c2 = c0;
    c0 = t;

    // Normalize the color values to [0, 1]
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // Rearrange the output format from interleaved RGB to separate channels
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;        // Red channel
    float* pdst_c1 = pdst_c0 + area;                   // Green channel
    float* pdst_c2 = pdst_c1 + area;                   // Blue channel

    // Assign the normalized color values to the destination buffers
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// Host function to perform CUDA-based preprocessing
void cuda_preprocess(
    // uint8_t* src,        // Source image data on host
    int src_width,       // Source image width
    int src_height,      // Source image height
    float* dst,          // Destination buffer on device
    int dst_width,       // Destination image width
    int dst_height,      // Destination image height
    cudaStream_t& stream,  // CUDA stream for asynchronous execution
    uint8_t** pinnedMemPtr,
    uint8_t** deviceMemPtr,
    uint16_t** depthMatHostPtr
) {
    // Calculate the size of the image in bytes (3 channels: BGR)
    int img_size = src_width * src_height * 3;

    // Copy source image data to pinned host memory for faster transfer
    // memcpy(*pinnedMemPtr, src, img_size);

    // Asynchronously copy image data from host to device memory
    CUDA_CHECK(cudaMemcpyAsync(
        *deviceMemPtr,
        *pinnedMemPtr,
        img_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        depthMatDevice_,
        *depthMatHostPtr,
        1624*1240 * sizeof(uint16_t),
        0,
        cudaMemcpyHostToDevice,
        stream
    ));

    // Define affine transformation matrices
    // AffineMatrix s2d, d2s; // Source to destination and vice versa

    // Calculate the scaling factor to maintain aspect ratio
    float scale = std::min(
        dst_height / (float)src_height,
        dst_width / (float)src_width
    );

    // Initialize source-to-destination affine matrix (s2d)
    s2d.value[0] = scale;                  // m00
    s2d.value[1] = 0;                      // m01
    s2d.value[2] = -scale * src_width * 0.5f + dst_width * 0.5f; // m02
    s2d.value[3] = 0;                      // m10
    s2d.value[4] = scale;                  // m11
    s2d.value[5] = -scale * src_height * 0.5f + dst_height * 0.5f; // m12

    // Create OpenCV matrices for affine transformation
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);

    // Invert the source-to-destination matrix to get destination-to-source
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    // Copy the inverted matrix back to d2s
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // std::cout<<"d2s.value[0]: "<<d2s.value[0]<<" d2s.value[4]: "<<d2s.value[4]<<std::endl;

    // Calculate the total number of pixels to process
    int jobs = dst_height * dst_width;

    // Define the number of threads per block
    int threads = 256;

    // Calculate the number of blocks needed
    int blocks = ceil(jobs / (float)threads);

    // Launch the warp affine kernel
    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        *deviceMemPtr,           // Source image on device
        src_width * 3,               // Source line size (bytes per row)
        src_width,                   // Source width
        src_height,                  // Source height
        dst,                         // Destination buffer on device
        dst_width,                   // Destination width
        dst_height,                  // Destination height
        128,                         // Constant value for out-of-bounds (gray)
        d2s,                         // Destination to source affine matrix
        jobs                         // Total number of pixels
        );

    // Optionally, you might want to check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

// Initialize CUDA preprocessing by allocating memory
void cuda_preprocess_init(int max_image_size,
                          uint8_t** pinnedMemPtr,
                          uint8_t** deviceMemPtr,
                          uint16_t** depthMatHostPtr
                        ) {
    // Allocate pinned (page-locked) memory on the host for faster transfers
    CUDA_CHECK(cudaMallocHost((void**)pinnedMemPtr, max_image_size * 3));

    // Allocate memory on the device (GPU) for the image
    CUDA_CHECK(cudaMalloc((void**)deviceMemPtr, max_image_size * 3));

    // Allocate memory for depth matrix on device
    CUDA_CHECK(cudaMallocHost((void**)depthMatHostPtr, max_image_size * sizeof(uint16_t)));
}

void cuda_preprocess_destroy(uint8_t** pinnedMemPtr, uint8_t** deviceMemPtr, uint16_t** depthMatHostPtr) {
    // std::cout<<"thread id: "<<std::this_thread::get_id()<<std::endl;

    // Free global memory on device
    CUDA_CHECK(cudaFree(*deviceMemPtr));
    // Free pinned memory on host
    CUDA_CHECK(cudaFreeHost(*pinnedMemPtr));

    CUDA_CHECK(cudaFreeHost(*depthMatHostPtr));

    // Set pointers to nullptr
    *deviceMemPtr = nullptr;
    *pinnedMemPtr = nullptr;
    *depthMatHostPtr = nullptr;
}

__device__ void pixel2World(
    float u, float v,
    float& worldBottomX, float& worldBottomY, float& worldBottomZ, float& imageD,
    float cu, float cv
) {
    int iu = static_cast<int>(cu);
    int iv = static_cast<int>(cv);

    if (iu < 0 || iu >= 1624 || iv < 0 || iv >= 1240) {
        worldBottomX = worldBottomY = worldBottomZ = -1.0f;
        printf("Invalid pixel coordinates: (%d, %d)\n", iu, iv);
        return;
    }

    int idx = iv * 1624 + iu;
    uint16_t depth = depthMatDevice_[idx];
    // printf("depth: %u\n", depth);

    float norm_x = (u - globalParams_.cx) / globalParams_.fx;
    float norm_y = (v - globalParams_.cy) / globalParams_.fy;
    
    float x_cam = norm_x * depth;
    float y_cam = norm_y * depth;
    float z_cam = depth;

    // 世界坐标 = Rinv * (x_cam, y_cam, z_cam) + T
    worldBottomX = globalParams_.R[0]*x_cam + globalParams_.R[1]*y_cam 
                 + globalParams_.R[2]*z_cam + globalParams_.tx;
    worldBottomY = globalParams_.R[3]*x_cam + globalParams_.R[4]*y_cam
                 + globalParams_.R[5]*z_cam + globalParams_.ty;
    worldBottomZ = globalParams_.R[6]*x_cam + globalParams_.R[7]*y_cam
                 + globalParams_.R[8]*z_cam + globalParams_.tz;
    imageD = depth;

    // worldX = globalParams_.R[0]*x_cam + globalParams_.R[3]*y_cam 
    //        + globalParams_.R[6]*z_cam + globalParams_.tx;
    // worldY = globalParams_.R[1]*x_cam + globalParams_.R[4]*y_cam
    //        + globalParams_.R[7]*z_cam + globalParams_.ty;
    // worldZ = globalParams_.R[2]*x_cam + globalParams_.R[5]*y_cam
    //        + globalParams_.R[8]*z_cam + globalParams_.tz;
}

__device__ void pixel2World(
    float u, float v,
    float& worldBottomX, float& worldBottomY, float& worldBottomZ, float& imageD,
    float& camX, float& camY, float& camZ,
    float cu, float cv
) {
    int iu = static_cast<int>(cu);
    int iv = static_cast<int>(cv);

    if (iu < 0 || iu >= 1624 || iv < 0 || iv >= 1240) {
        worldBottomX = worldBottomY = worldBottomZ = -1.0f;
        printf("Invalid pixel coordinates: (%d, %d)\n", iu, iv);
        return;
    }

    int idx = iv * 1624 + iu;
    uint16_t depth = depthMatDevice_[idx];
    // printf("depth: %u\n", depth);

    float norm_x = (u - globalParams_.cx) / globalParams_.fx;
    float norm_y = (v - globalParams_.cy) / globalParams_.fy;
    
    camX = norm_x * depth;
    camY = norm_y * depth;
    camZ = depth;

    // 世界坐标 = Rinv * (x_cam, y_cam, z_cam) + T
    worldBottomX = globalParams_.R[0]*camX + globalParams_.R[1]*camY 
                 + globalParams_.R[2]*camZ + globalParams_.tx;
    worldBottomY = globalParams_.R[3]*camX + globalParams_.R[4]*camY
                 + globalParams_.R[5]*camZ + globalParams_.ty;
    worldBottomZ = globalParams_.R[6]*camX + globalParams_.R[7]*camY
                 + globalParams_.R[8]*camZ + globalParams_.tz;
    imageD = depth;
}

__global__ void filterDetectionKernel (
    //! const float* gpuDetections,
    float (*gpuDetections)[8400],
    int numDetections,
    int numAttributeSize,
    float confThreshold,
    Detection* filteredDetections,  
    int* numAnchors,
    AffineMatrix d2s_device
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) {
        // printf("preprocess.cu: 295 | Thread exited: idx=%d\n", idx);
        return;
    }

    //! const float* currDetection = *gpuDetections + idx;
    //! const float confidence = currDetection[4*8400];
    const float confidence = gpuDetections[4][idx];

    if (confidence > confThreshold) {
        int currIndex = atomicAdd(numAnchors, 1);

        // float x = currDetection[0];
        // float y = currDetection[1*8400];
        // float w = currDetection[2*8400];
        // float h = currDetection[3*8400];

        float x_dst = gpuDetections[0][idx];
        float y_dst = gpuDetections[1][idx];
        float w_dst = gpuDetections[2][idx];
        float h_dst = gpuDetections[3][idx];
        float θ     = gpuDetections[5][idx] / 3.14 * 180;

        float x_src = d2s_device.value[0]*x_dst + d2s_device.value[1]*y_dst + d2s_device.value[2];
        float y_src = d2s_device.value[3]*x_dst + d2s_device.value[4]*y_dst + d2s_device.value[5];
        float w_src = w_dst * d2s_device.value[0];
        float h_src = h_dst * d2s_device.value[0];
        float bottomX = x_src;
        float bottomY = (θ > 50 && θ < 130) ? (y_src + w_src / 2) : (y_src + h_src / 2);

        // world coordinates
        float worldBottomX, worldBottomY, worldBottomZ, imageCenterD, camBottomX, camBottomY, camBottomZ;
        // float worldBottomX, worldBottomY, worldBottomZ;

        // pixel2World(x_src, y_src, worldCenterX, worldCenterY, worldCenterZ);

        // pixel2World(bottomX, bottomY, worldBottomX, worldBottomY, worldBottomZ, imageCenterD, x_src, y_src);
        pixel2World(bottomX, bottomY, worldBottomX, worldBottomY, worldBottomZ, imageCenterD,
                    camBottomX, camBottomY, camBottomZ, x_src, y_src);


        /* 计算世界坐标系中的宽度和高度 */
        // 计算矩形四个角点
        float cosθ = cosf(θ * 3.1416 / 180.0f);
        float sinθ = sinf(θ * 3.1416 / 180.0f);
        float halfW = w_src / 2.0f;
        float halfH = h_src / 2.0f;

        float corners[4][2] = {
            {x_src - halfW * cosθ - halfH * sinθ, y_src - halfW * sinθ + halfH * cosθ}, // LT
            {x_src + halfW * cosθ - halfH * sinθ, y_src + halfW * sinθ + halfH * cosθ}, // RT
            {x_src + halfW * cosθ + halfH * sinθ, y_src + halfW * sinθ - halfH * cosθ}, // RB
            {x_src - halfW * cosθ + halfH * sinθ, y_src - halfW * sinθ - halfH * cosθ}  // LB
        };

        // 将四个角点转换为世界坐标
        float worldCorners[4][3]; // [x, y, z]
        float temp;
        for (int i = 0; i < 4; i++) {
            pixel2World(corners[i][0], corners[i][1], 
                worldCorners[i][0], worldCorners[i][1], worldCorners[i][2], temp, x_src, y_src);
            }
            
            float dx, dy, dz;
            
            dx = worldCorners[0][0] - worldCorners[1][0];
            dy = worldCorners[0][1] - worldCorners[1][1];
            dz = worldCorners[0][2] - worldCorners[1][2];
            float edge1 = __fsqrt_rn(dx * dx + dy * dy + dz * dz); 
            //! __fsqrt_rn(__powf(dx, 2) + __powf(dy, 2)); 
            //! __powf(dx, 2) 内部可能会以log的形式计算平方，如果是负数，则会输出nan
            
            dx = worldCorners[1][0] - worldCorners[2][0];
            dy = worldCorners[1][1] - worldCorners[2][1];
            dz = worldCorners[1][2] - worldCorners[2][2];
            float edge2 = __fsqrt_rn(dx * dx + dy * dy + dz * dz);
            
            dx = worldCorners[2][0] - worldCorners[3][0];
            dy = worldCorners[2][1] - worldCorners[3][1];
            dz = worldCorners[2][2] - worldCorners[3][2];
            float edge3 = __fsqrt_rn(dx * dx + dy * dy + dz * dz);
            
            dx = worldCorners[3][0] - worldCorners[0][0];
            dy = worldCorners[3][1] - worldCorners[0][1];
            dz = worldCorners[3][2] - worldCorners[0][2];
            float edge4 = __fsqrt_rn(dx * dx + dy * dy + dz * dz);
            
            // printf("edge1=%f, edge2=%f, edge3=%f, edge4=%f\n", 
            //     edge1, edge2, edge3, edge4);
            float worldWidth  = (θ > 50 && θ < 130) ? (edge2 + edge4) / 2.0f : (edge1 + edge3) / 2.0f;
            float worldHeight = (θ > 50 && θ < 130) ? (edge1 + edge3) / 2.0f : (edge2 + edge4) / 2.0f;
            
            /* for original, no obb
            filteredDetections[currIndex].bbox.x = static_cast<int>(x - w/2);
            filteredDetections[currIndex].bbox.y = static_cast<int>(y - h/2);
            */
           filteredDetections[currIndex].rbox.center.x    = static_cast<int>(x_src);
           filteredDetections[currIndex].rbox.center.y    = static_cast<int>(y_src);
           filteredDetections[currIndex].rbox.size.width  = static_cast<int>(w_src);
           filteredDetections[currIndex].rbox.size.height = static_cast<int>(h_src);
           filteredDetections[currIndex].rbox.angle       = θ;
           filteredDetections[currIndex].conf             = confidence;
           filteredDetections[currIndex].bottomX          = x_src;
           filteredDetections[currIndex].bottomY          = bottomY;
           filteredDetections[currIndex].centerDepth      = imageCenterD;
           // camera corrdinates
           filteredDetections[currIndex].camX             = camBottomX;
           filteredDetections[currIndex].camY             = camBottomY;
           filteredDetections[currIndex].camZ             = camBottomZ;
           // world coordinates
           filteredDetections[currIndex].worldWidth       = worldWidth;
           filteredDetections[currIndex].worldHeight      = worldHeight;
           filteredDetections[currIndex].worldBX          = worldBottomX;
           filteredDetections[currIndex].worldBY          = worldBottomY;  
           filteredDetections[currIndex].worldBZ          = worldBottomZ;
        //    printf("θ: %f\n", θ);
    }
}

void filterDetection (
    float* gpuDetections,
    int numDetections,
    int numAttributeSize,
    float confThreshold,
    cudaStream_t& stream,
    int* deviceNumAnchors,
    Detection* deviceFilteredDetections
) {
    // int* deviceNumAnchors;  //device_numAnchors
    int hostNumAnchors = 0;  //host_numAnchors

    // CUDA_CHECK(cudaMalloc(&deviceNumAnchors, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(deviceNumAnchors, 0, sizeof(int), stream));

    // Detection* deviceFilteredDetections;
    // CUDA_CHECK(cudaMalloc(&deviceFilteredDetections, numDetections/10 * sizeof(Detection)));

    const int blockSize = 256;
    const int gridSize = (numDetections + blockSize -1) / blockSize;

    // CameraParams hostParams;
    // CUDA_CHECK(cudaMemcpy(&hostParams, &deviceParams, sizeof(CameraParams), cudaMemcpyDeviceToHost));

    // std::cout<<"d2s.value[0]: "<<d2s.value[0]<<" d2s.value[4]: "<<d2s.value[4]<<std::endl;
    // std::cout<<"preprocess.cu 415: deviceParams.fx: "<< deviceParams.fx<<std::endl;

    filterDetectionKernel<<<gridSize, blockSize, 0, stream>>>(
        reinterpret_cast<float(*)[8400]>(gpuDetections),  // 6*8400
        numDetections,
        numAttributeSize,
        confThreshold,
        deviceFilteredDetections,
        deviceNumAnchors,
        d2s
    );
    // std::cout<<"preprocess.cu: 307 HERE"<<std::endl; 

    // CUDA_CHECK(cudaMemcpy(&hostNumAnchors, deviceNumAnchors, sizeof(int), cudaMemcpyDeviceToHost));

    // // std::cout<<"preprocess.cu: 311 , hostNumAnchors: "<<hostNumAnchors<<std::endl;

    // if(hostNumAnchors == 0) {
    //     CUDA_CHECK(cudaFree(deviceNumAnchors));
    //     CUDA_CHECK(cudaFree(deviceFilteredDetections));
    //     return std::vector<Detection>();
    // } else if (hostNumAnchors < 0) {
    //     CUDA_CHECK(cudaFree(deviceNumAnchors));
    //     CUDA_CHECK(cudaFree(deviceFilteredDetections)); 
    //     throw std::runtime_error("Error: Number of anchors is negative");      
    // }

    // std::vector<Detection> hostFilteredDetections(hostNumAnchors);
    // CUDA_CHECK(cudaMemcpy(
    //     hostFilteredDetections.data(),
    //     deviceFilteredDetections,
    //     hostNumAnchors * sizeof(Detection),
    //     cudaMemcpyDeviceToHost
    // ));

    // CUDA_CHECK(cudaFree(deviceNumAnchors));
    // CUDA_CHECK(cudaFree(deviceFilteredDetections));

    // return hostFilteredDetections;
}