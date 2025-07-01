#include "YOLO11.h"

// static uint8_t* pinnedMemPtr00 = nullptr;
// static uint8_t* deviceMemPtr00 = nullptr;
// static uint16_t* depthMatHostPtr00 = nullptr;

class TLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: LOG_ERROR("TensorRT: ", msg); break;
            case Severity::kERROR:          LOG_ERROR("TensorRT: ", msg); break;
            case Severity::kWARNING:        LOG_WARNING("TensorRT: ", msg); break;
            case Severity::kINFO:           LOG_INFO("TensorRT: ", msg); break;
            case Severity::kVERBOSE:        LOG_DEBUG("TensorRT: ", msg); break;
            default:                        LOG_TRACE("TensorRT: ", msg); break;            
        }
    }
}TRTlog;

// typedef unsigned short ushort;

YOLO11::YOLO11(TensorRT& trt){
    
    std::ifstream engineStream(trt.modelPath, std::ios::binary); 
    engineStream.seekg(0, std::ios::end);			
	LOG_TRACE("Open the engine file successfully");

    const size_t modelSize = engineStream.tellg();
	LOG_INFO("engine size: ", modelSize);
    
    engineStream.seekg(0, std::ios::beg);			
    std::unique_ptr<char[]> engineData(new char[modelSize]);	
    engineStream.read(engineData.get(), modelSize);		
    engineStream.close();
	LOG_TRACE("Read the engine file successfully");

    runtime_ = std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>>(
        nvinfer1::createInferRuntime(TRTlog),
        [](nvinfer1::IRuntime* r) { 
            // if(r) r->destroy(); 
			LOG_TRACE("Runtime destroyed");
        }
    );
    trt.engine = std::unique_ptr<nvinfer1::ICudaEngine, std::function<void(nvinfer1::ICudaEngine*)>>(
        runtime_->deserializeCudaEngine(engineData.get(), modelSize),
        [](nvinfer1::ICudaEngine* e) { 
            // if(e) e->destroy(); 
			LOG_TRACE("Engine destroyed");
        }
    );
    trt.context = std::unique_ptr<nvinfer1::IExecutionContext, std::function<void(nvinfer1::IExecutionContext*)>>(
        trt.engine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* c) {
            // if(c) c->destroy();
			LOG_TRACE("Context destroyed");
        }
    );
	LOG_TRACE("Create tensorRT runtime, engine, context successfully");

	auto inputName  = trt.engine->getIOTensorName(0);  
	auto outputName = trt.engine->getIOTensorName(1); 

	auto inputShape = trt.engine->getTensorShape(inputName);
	inputHeight_ 	= inputShape.d[2];
	inputWidth_  	= inputShape.d[3];

	auto outputShape  = trt.engine->getTensorShape(outputName);
	trt.numAttributes = outputShape.d[1];  // 6 (for OBB)
	trt.numDetections = outputShape.d[2];  // 8400
    trt.numClasses    = numAttributes_ - 5;  // (x, y, w, h, angle)

    CUDA_CHECK(cudaMalloc(&(trt.gpuBuffers[0]), 3*640*640*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&(trt.gpuBuffers[1]), 6*8400*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&(trt.deviceNumAnchors), sizeof(int)));
	CUDA_CHECK(cudaMemset(trt.deviceNumAnchors, 0, sizeof(int)));
	CUDA_CHECK(cudaMallocHost((void**)&(trt.hostNumAnchors), sizeof(int)));
	memset(trt.hostNumAnchors, 0, sizeof(int));
	CUDA_CHECK(cudaMalloc(&(trt.deviceFilteredDetections), 840 * sizeof(Detection)));
    cuda_preprocess_init(
		MAX_IMAGE_SIZE,
		&(trt.pinnedMem),
		&(trt.deviceMem),
		&(trt.depthMatHost)
	);
    CUDA_CHECK(cudaStreamCreate(&(trt.stream)));
	trt.context -> setOptimizationProfileAsync(0, trt.stream);
	trt.context -> setInputTensorAddress(inputName, trt.gpuBuffers[0]);  
	trt.context -> setOutputTensorAddress(outputName, trt.gpuBuffers[1]);

	std::ifstream file("Parameter.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open Parameter.txt" << std::endl;
        throw std::runtime_error("Error: Could not open Parameter.txt");
    }

    std::string line;
    std::vector<float> values;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stof(token));
        }
    }
	file.close();

    if (values.size() < 10) {
        std::cerr << "Error: Not enough values in Parameter.txt" << std::endl;
        throw std::runtime_error("Error: Not enough values in Parameter.txt");
    }

	params_.fx = values[0];
    params_.fy = values[1];
    params_.cx = values[2];
    params_.cy = values[3];

    params_.tx = values[4];
    params_.ty = values[5];
    params_.tz = values[6];

    params_.rx = values[7];
    params_.ry = values[8];
    params_.rz = values[9];

	std::cout << "fx: " << params_.fx << ", fy: " << params_.fy << ", cx: " << params_.cx << ", cy: " << params_.cy << std::endl;
    std::cout << "tx: " << params_.tx << ", ty: " << params_.ty << ", tz: " << params_.tz << std::endl;
    std::cout << "rx: " << params_.rx << ", ry: " << params_.ry << ", rz: " << params_.rz << std::endl;
	std::cout<<"sizeof(CameraParams): "<<sizeof(CameraParams)<<std::endl;

	float rx_rad = params_.rx * CV_PI / 180.0f;
    float ry_rad = params_.ry * CV_PI / 180.0f;
    float rz_rad = params_.rz * CV_PI / 180.0f;

	// X旋转矩阵
    float rotX[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, cosf(rx_rad), -sinf(rx_rad),
        0.0f, sinf(rx_rad), cosf(rx_rad)
    };
    
    // Y旋转矩阵
    float rotY[9] = {
        cosf(ry_rad), 0.0f, sinf(ry_rad),
        0.0f, 1.0f, 0.0f,
        -sinf(ry_rad), 0.0f, cosf(ry_rad)
    };
    
    // Z旋转矩阵
    float rotZ[9] = {
        cosf(rz_rad), -sinf(rz_rad), 0.0f,
        sinf(rz_rad), cosf(rz_rad), 0.0f,
        0.0f, 0.0f, 1.0f
    };

	/* R = Rz * Ry * Rx */ 
    // Ry * Rx
    float temp[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                temp[i*3+j] += rotY[i*3+k] * rotX[k*3+j];
            }
        }
    }
    
    // Rz * (Ry * Rx)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            params_.R[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                params_.R[i*3+j] += rotZ[i*3+k] * temp[k*3+j];
            }
        }
    }
    
    // 计算旋转矩阵的逆（对于正交矩阵，逆等于转置）
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            params_.Rinv[i*3+j] = params_.R[j*3+i];
        }
    }

	initCameraParams(params_);
}

YOLO11::~YOLO11() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamDestroy(stream_));

    CUDA_CHECK(cudaFree(gpuBuffers_[0]));
    CUDA_CHECK(cudaFree(gpuBuffers_[1]));
	CUDA_CHECK(cudaFree(deviceNumAnchors_));
	CUDA_CHECK(cudaFree(deviceFilteredDetections_));
	CUDA_CHECK(cudaFreeHost(hostNumAnchors_));
	// CUDA_CHECK(cudaFree(deviceParams_));

    cuda_preprocess_destroy(pinnedMemPtr_, deviceMemPtr_, depthMatHostPtr_);
	// nvmlShutdown();
    // std::cout<<"YOLO11 destruct complete"<<std::endl;
	LOG_TRACE("YOLO11 destruct complete");
}

std::vector<Detection> YOLO11::modelProcess(const std::pair<cv::Mat, cv::Mat>& pair) {
	cv::Mat colorMat = pair.first;
	cv::Mat depthMat = pair.second;
	int img_size = colorMat.cols*colorMat.rows;

	memcpy(*pinnedMemPtr_, colorMat.ptr(), img_size*3);
	memcpy(*depthMatHostPtr_, depthMat.ptr(), img_size*sizeof(uint16_t));

	cuda_preprocess(
		// dummyImg,
		colorMat.cols,
		colorMat.rows,
		gpuBuffers_[0],
		inputWidth_,
		inputHeight_,
		stream_,
		pinnedMemPtr_,
		deviceMemPtr_,
		depthMatHostPtr_
	);

    // context_ -> enqueueV2((void**)gpuBuffers_, stream_, nullptr);
	context_ -> enqueueV3(stream_);

	filterDetection(
        gpuBuffers_[1],
        numDetections_,
        numAttributes_,
        confThreshold_,
        stream_,
		deviceNumAnchors_,
		deviceFilteredDetections_
    );

	CUDA_CHECK(cudaMemcpy(
		hostNumAnchors_,
		deviceNumAnchors_,
		sizeof(int),
		cudaMemcpyDeviceToHost
	));
	LOG_INFO("hostNumAnchors: ", *hostNumAnchors_);

	if (*hostNumAnchors_ == 0) {
		LOG_INFO("No detection results");
		// nvtxRangePop();
		return std::vector<Detection> {};
	} else if (*hostNumAnchors_ < 0) {
		LOG_ERROR("Error in detection results");
		throw std::runtime_error("Error: Number of anchors is negative");
	}

	thresholdResult_.resize(*hostNumAnchors_);
	CUDA_CHECK(cudaMemcpy(
		thresholdResult_.data(),
		deviceFilteredDetections_,
		*hostNumAnchors_ * sizeof(Detection),
		cudaMemcpyDeviceToHost
	));	

    std::vector<cv::RotatedRect> rboxes;
    std::vector<float> scores;
    for (const auto& det : thresholdResult_) {
        rboxes.push_back(det.rbox);
        scores.push_back(det.conf);
    }

    std::vector<int> nmsResult; 
    if(!rboxes.empty()) {
        cv::dnn::NMSBoxes(rboxes, scores, confThreshold_, nmsThreshold_, nmsResult);
    }

    std::vector<Detection> output;
    for(const auto nms : nmsResult) {
        Detection det;
        det.conf    = scores[nms];
        det.classId = thresholdResult_[nms].classId;
        det.rbox    = rboxes[nms];
		det.bottomX = thresholdResult_[nms].bottomX;
		det.bottomY = thresholdResult_[nms].bottomY;
		det.camX	= thresholdResult_[nms].camX;
		det.camY	= thresholdResult_[nms].camY;
		det.camZ	= thresholdResult_[nms].camZ;
		det.worldWidth = thresholdResult_[nms].worldWidth;
		det.worldHeight = thresholdResult_[nms].worldHeight;
		det.worldBX = thresholdResult_[nms].worldBX;
		det.worldBY = thresholdResult_[nms].worldBY;
		det.worldBZ = thresholdResult_[nms].worldBZ;
        output.push_back(det);
    }
	// for (const auto& out : output) {
	// 	std::cout<<"out.worldCZ: "<<out.worldCZ<<" out.worldBX: "<<out.worldBX <<" out.worldBY: "<<out.worldBY
	// 	<<" out.worldW: "<<out.worldWidth<<" out.worldH: "<<out.worldHeight <<std::endl;
	// }

    return output;
}


std::pair<int, std::vector<float>> YOLO11::detectProcess(cv::Mat& depthMat, std::vector<Detection> output, bool openVerify) {
	int num = 0;
	std::vector<float> outputData;

	// if(openVerify)
	// 	lostReplenish(depthMat, output);

	//顺序处理
	std::vector<Detection> outputCopy = output;
	std::vector<std::vector<Detection>> outputRowsContainer; //以“行”为组存放
	std::vector<Detection> outputRow; //当前行暂存
	int zErr = 50; //同一行Z坐标允许的误差值 (mm)
	float minDepth = 10000;

	for (unsigned int i = 0; i < outputCopy.size(); i++) {
		for (unsigned int j = i + 1; j < outputCopy.size(); j++)
		{
			if (abs(outputCopy[i].worldBZ - outputCopy[j].worldBZ) <= zErr)
			{
				outputRow.push_back(outputCopy[j]);
				outputCopy.erase(outputCopy.begin() + j);
				j--;
			}
		}
		outputRow.push_back(outputCopy[i]);//最后将比较的点放入Vector
		std::sort(outputRow.begin(), outputRow.end(), &sortRuleWY);//对同一行的点位进行排序
		outputRowsContainer.push_back(outputRow);
		outputRow.clear();	//清空，准备下一次排序
	}

	std::sort(outputRowsContainer.begin(), outputRowsContainer.end(), &sortRuleWZ);
	std::vector<Detection> temp;
	for (unsigned int i = 0; i < outputRowsContainer.size(); i++)
	{
		if (i % 2 == 1) {
			for (int j = outputRowsContainer.at(i).size() - 1; j >= 0; j--) {
				temp.push_back(outputRowsContainer.at(i).at(j));
				if(outputRowsContainer[i][j].worldBX < minDepth)
					minDepth = outputRowsContainer[i][j].worldBX;
			}
		}
		else {
			for (int j = 0; j < outputRowsContainer.at(i).size(); j++) {
				temp.push_back(outputRowsContainer.at(i).at(j));
				if(outputRowsContainer[i][j].worldBX < minDepth)
					minDepth = outputRowsContainer[i][j].worldBX;
			}
		}
	}
	output = temp;
	temp.clear();
	// drawImg(bottomPoints);
	// for(auto p : bottomPoints) {
	// 	std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
	// }


	// std::cout << "minDepth: " << minDepth <<std::endl;
	LOG_INFO("minDepth: ", minDepth);

	//过滤后面
	int depthErr = 100; //范围内视为同一面
	std::vector<Box> boxes;

	for (int i = 0; i < output.size(); i++) {
		if (abs(output[i].worldBX - minDepth) < depthErr) {
			num++;
			Box box = {output[i].worldBX, output[i].worldBY, output[i].worldBZ, 0, output[i].worldWidth, output[i].worldHeight};
			boxes.push_back(box);
		}
		// else {
		// 	std::cout << i+1 << "out: " << output[i].worldBX << " " << output[i].worldBY << " " << output[i].worldBZ
		// 	<< " " << output[i].worldWidth << " " << output[i].worldHeight << std::endl;
		// }
	}

	//卸箱方案
	float headWidth = 600;
	float dropThreshold = 30;
	float cupDiameter = 65;
	float cupSpacing = 110;
	float leftSide = 875;
	float rightSide = -875;
	const int cupCount = 5;
	float cupsLeftBia[cupCount] = { 0 };

	double offset = (cupCount - 1) / 2.0;
	for (int i = 0; i < cupCount; ++i) {
		cupsLeftBia[i] = (offset - i) * cupSpacing + cupDiameter / 2; // 居中排布
	}

	std::vector<Group> groups;

	// std::cout<<"卸箱方案"<<std::endl;
	LOG_TRACE("unload solution");

	int index = 0;
	while (index < boxes.size()) {
		float totalWidth = boxes[index].width;
		std::vector<Box> mBox;
		mBox.push_back(boxes[index]);
		std::vector<bool> cupEnabled(5, false);
		bool verticalFlag = boxes[index].height > 90 ? false : true;
		Group group = { boxes[index].x, boxes[index].y, boxes[index].z, boxes[index].y + boxes[index].width / 2,
						boxes[index].width, boxes[index].height, verticalFlag, mBox, cupEnabled};
		for (int i = index + 1; i < boxes.size(); i++) {
			float tmpWidth = totalWidth + abs(boxes[i].y - boxes[i - 1].y) - boxes[i - 1].width / 2 + boxes[i].width / 2;
			if (tmpWidth > headWidth || abs(boxes[index].z - boxes[i].z) > dropThreshold) {
				break;
			}
			else {
				mBox.push_back(boxes[i]);
				totalWidth = tmpWidth;
				if (boxes[i].y + boxes[i].width / 2 > group.groupLeft) group.groupLeft = boxes[i].y + boxes[i].width / 2;
				if (boxes[i].z < group.minZ) {
					float tmpHeight = group.minZ - boxes[i].z + group.height;
					group.height = tmpHeight > boxes[i].height ? tmpHeight : boxes[i].height;
					group.minZ = boxes[i].z;
				}
				else {
					float tmpHeight = boxes[i].z - group.minZ + boxes[i].height;
					group.height = tmpHeight > group.height ? tmpHeight : group.height;
				}
				group.centerY = group.groupLeft - totalWidth / 2;
				group.width = totalWidth;
				group.isVerticalSuction = false;
			}
		}
		group.mBox = mBox;
		groups.push_back(group);
		index += mBox.size();
	}

	// std::cout<<"groups size: "<<groups.size()<<std::endl;
	LOG_INFO("groups size: ", groups.size());

	for (int i = 0; i < groups.size(); i++) {
		if (groups[i].centerY > leftSide) {
			groups[i].centerY = groups[i].groupLeft - headWidth / 2;
		}
		if (groups[i].centerY < rightSide) {
			groups[i].centerY = 2 * groups[i].centerY - groups[i].groupLeft + headWidth / 2;
		}
		bool canGrabAll = true;
		for (auto& box : groups[i].mBox) {
			int enaledCount = 0;
			float rangeLeft = box.y + box.width / 2;
			float rangeRight = box.y - box.width / 2;
			for (int j = 0; j < cupCount; j++) {
				float cupLeftY = groups[i].centerY + cupsLeftBia[j];
				float cupRightY = cupLeftY - cupDiameter;
				if (cupLeftY<rangeLeft && cupRightY>rangeRight) {
					groups[i].cupsEnabled[j] = true;
					enaledCount++;
				}
			}
			if (enaledCount == 0) {
				canGrabAll = false;
				break;
			}
		}
		if (!canGrabAll) {
			for (int j = 0; j < groups[i].mBox.size(); j++) {
				std::vector<Box> mBox = { groups[i].mBox[j] };
				std::vector<bool> cupEnabled(5, false);
				bool verticalFlag = groups[i].mBox[j].height > 90 ? false : true;
				Group subGroup = { groups[i].mBox[j].x, groups[i].mBox[j].y, groups[i].mBox[j].z,
					groups[i].mBox[j].y + groups[i].mBox[j].width / 2,
					groups[i].mBox[j].width, groups[i].mBox[j].height, verticalFlag, mBox, cupEnabled };
				groups.insert(groups.begin() + i + j + 1, subGroup);
			}
			groups.erase(groups.begin() + i);
			i -= 1;
		}
		// std::cout << "Processing group " << i << ", group size: " << groups.size() << std::endl;
		LOG_INFO("Process group: ", i, ", group size: ", groups.size());

	}

		// std::cout<<"485卸箱方案"<<std::endl;
		LOG_INFO("485");

	for (auto& group : groups) {
		outputData.push_back(group.centerX);  //世界坐标X
		outputData.push_back(group.centerY);  //世界坐标Y
		outputData.push_back(group.minZ);     //世界坐标Z
		outputData.push_back(0);              //世界坐标rx
		outputData.push_back(0);              //世界坐标ry
		outputData.push_back(0);              //世界坐标rz
		outputData.push_back(group.width);    //宽度
		outputData.push_back(group.height);   //高度
		for (int i = 0; i < group.cupsEnabled.size(); i++) outputData.push_back(group.cupsEnabled[i]);  //吸盘开启标志
		outputData.push_back(group.isVerticalSuction);  //垂直吸取标志
	}

	std::cout<<"outputData: ";
	// LOG_INFO("outputData: ", outputData);
	for (auto& output : outputData) 
		// LOG_INFO(output);
		std::cout<<output<<' ';
	// drawImg(bottomPointsPro);
	std::cout<<std::endl;
	// LOG_INFO("ready to send");
	//!-------------------------------------------------------------------------------------
	// int num = 1;
	// std::vector<float> outputData;
	// outputData.push_back(0.0);
	// outputData.push_back(1.0);
	// outputData.push_back(2.0);
	// outputData.push_back(3.0);

	// LOG_INFO("output.size: ", outputData.size());
	// std::cout<<"output.size: "<<outputData.size()<<std::endl;
	//!-------------------------------------------------------------------------------------

	return std::make_pair(num, outputData);    
}

float YOLO11::getAngle(const cv::Mat& depthMat) {
	cv::Mat depthFloat;
	depthMat.convertTo(depthFloat, CV_32F, 1.0 / 1000.0); // m -> mm

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	int width = depthMat.cols;
	int height = depthMat.rows;

	int roiSize = 300;
	int u_Start = std::max(0, static_cast<int>(params_.cx - roiSize / 2));
	int v_Start = std::max(0, static_cast<int>(params_.cy - roiSize / 2));
	int u_End	= std::min(width, u_Start + roiSize);
	int v_End	= std::min(height, v_Start + roiSize);

	for (int v=v_Start; v<v_End; v++) {
		for (int u=u_Start; u<u_End; u++) {
			float z = depthFloat.at<float>(v, u);
			if (z == 0) continue;
			float x = (u - params_.cx) * z / params_.fx;
			float y = (v - params_.cy) * z / params_.fy;
			cloud->points.emplace_back(x, y, z);
		}
	}
	cloud->width = cloud->points.size();
	cloud->height = 1;
	cloud->is_dense = false;

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01); // 10mm
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.empty()) {
		std::cerr<<"No plane found"<<std::endl;
		return 0.0f;
	}

	Eigen::Vector3f normal(coefficients->values[0],
		   coefficients->values[1], coefficients->values[2]);
	normal.normalize();

	Eigen::Vector3f zAxis(0.0f, 0.0f, 1.0f);
	float angle_rad = std::acos(normal.dot(zAxis));
	float angle_deg = angle_rad * 180.0f / CV_PI;

	std::cout<<"Plane normal: "<<normal.transpose()<<", angle: "<<angle_deg<<" degrees"<<std::endl;
	LOG_INFO("Plane normal: ", normal.transpose(), ", angle: ", angle_deg, " degrees");

	return angle_deg;
}

void YOLO11::lostReplenish(cv::Mat& depthMat, std::vector<Detection>& result)
{
	//模型漏检检测，补充疑似缺失框（矩形）
	int missingCount = 0;
	int top = depthMat.rows * 0.13;
	int down = 0;
	int left = depthMat.cols;
	int right = 0;
	float camMinDepth = 10000;
	int depthErr = 100; //范围内视为同一面
	for(int i=0; i < result.size(); i++)
	{
		if(result[i].centerDepth < camMinDepth) camMinDepth = result[i].centerDepth;
		std::vector<cv::Point2f> pts;
		result[i].rbox.points(pts);
		for (int p = 0; p < pts.size(); p++) {
			if (pts[p].x < left)
				left = pts[p].x;
			if (pts[p].x > right)
				right = pts[p].x;
			if (pts[p].y > down)
				down = pts[p].y;
		}
	}
	std::cout << "top: " << top << " down: " << down << " left: " << left << " right: " << right << " camMinDepth: " << camMinDepth << std::endl;
	//深度连通域
	cv::Mat judgeMat = cv::Mat::zeros(depthMat.size(), CV_8UC1);
	for(int i=0; i < depthMat.rows; i++)
	{
		for(int j=0; j < depthMat.cols; j++)
		{
			if(abs(depthMat.at<ushort>(i,j) - camMinDepth) < depthErr) {
				judgeMat.at<uchar>(i,j) = 255;
			}
		}
	}
	//cv::imwrite("F:\\02.17Ainstec_obb_test\\depth_all.png", judgeMat);
	for(int i = 0; i < result.size(); i++)
	{
		std::vector<cv::Point2f> pts;
		result[i].rbox.points(pts);
		std::vector<cv::Point> cPts;
		for(int n=0; n<pts.size(); n++) cPts.push_back(cv::Point((int)pts[n].x, (int)pts[n].y));
		cv::fillConvexPoly(judgeMat, cPts, cv::Scalar(0));
	}
	//cv::imwrite("F:\\02.17Ainstec_obb_test\\depth_over.png", judgeMat);
	cv::Mat croppedImage = judgeMat(cv::Range(top, down), cv::Range(left, right)); //裁剪
	cv::Mat eroDst;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); // 定义结构元素
	cv::erode(croppedImage, eroDst, element); // 腐蚀操作
	cv::Mat out, stats, centroids;
	//统计图像中连通域的个数
	int number = connectedComponentsWithStats(eroDst, out, stats, centroids, 8, CV_16U);

	std::vector<std::vector<float>> bottomPoints;
	for (int i = 1; i < number; ++i) {
		//中心位置
		int center_x = centroids.at<double>(i, 0);
		int center_y = centroids.at<double>(i, 1);
		//矩形边框
		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int area = stats.at<int>(i, cv::CC_STAT_AREA);

		//外接矩形
		cv::Rect rect(x, y, w, h);
		std::cout << i << ": " << area << "  rect: " << rect.area() << std::endl;
		if (area > 5000 && area < 40000) {
			//中心位置绘制
			//circle(croppedImage, cv::Point(center_x, center_y), 2, cv::Scalar(0, 255, 0), 2, 0, 0);
			if ((rect.area() - area) > 2000) {
				int semiH = 0, semiW = 0;
				for (int sub_y = center_y; sub_y < y + h; sub_y++) {
					if (eroDst.at<uchar>(sub_y, center_x) == 0) {
						semiH = sub_y - center_y;
						break;
					}
				}
				for (int sub_x = center_x; sub_x < x + w; sub_x++) {
					if (eroDst.at<uchar>(center_y, sub_x) == 0) {
						semiW = sub_x - center_x;
						break;
					}
				}
				rect = cv::Rect(center_x - semiW, center_y - semiH, semiW * 2, semiH * 2);
				//cv::rectangle(croppedImage, rect, cv::Scalar(255, 0, 255), 3, 0, 0);
			}
			float bottomX = rect.x + rect.width / 2 + left;
			float bottomY = rect.y + rect.height + top;
			float centerZ = depthMat.at<ushort>(cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2));
			cv::Point pts[3] = { cv::Point(rect.x, rect.y + rect.height), 
								 cv::Point(rect.x + rect.width,rect.y + rect.height), 
								 cv::Point(rect.x + rect.width,rect.y) };
			std::vector<float> bottomPoint = {bottomX, bottomY, centerZ, 0, (float)pts[0].x, (float)pts[0].y,
											 (float)pts[1].x, (float)pts[1].y, (float)pts[2].x, (float)pts[2].y,};

			bottomPoints.push_back(bottomPoint);
			missingCount++;
		}
	}
	std::cout << "missingCount: " << missingCount << std::endl;

	//转换矩阵
	std::ifstream convertParam("Parameter.txt");
	if (!convertParam.is_open()) {
		//LOG_WARNING("Cannot open Parameter.txt");
		std::cout<<"Cannot open Parameter.txt"<<std::endl;
	}
	std::string line;
	std::vector<float> cameraParam;
	std::vector<float> convertVec;
	int idxLine = 1;
	while (std::getline(convertParam, line)) {
		std::stringstream ss(line);
		std::string s;
		while (getline(ss, s, ',')) {
			float value = std::stof(s);
			switch (idxLine)
			{
			case 1:
				cameraParam.push_back(value);
				break;
			case 2:
				convertVec.push_back(value);
				break;
			default:
				break;
			}
		}
		idxLine++;
	}
	convertParam.close();
	double xRad = convertVec[3] * CV_PI / 180.0;
	double yRad = convertVec[4] * CV_PI / 180.0;
	double zRad = convertVec[5] * CV_PI / 180.0;
	cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
	    1, 0, 0,
	    0, cos(xRad), -sin(xRad),
	    0, sin(xRad), cos(xRad));
	cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
	    cos(yRad), 0, sin(yRad),
	    0, 1, 0,
	    -sin(yRad), 0, cos(yRad));
	cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
	    cos(zRad), -sin(zRad), 0,
	    sin(zRad), cos(zRad), 0,
	    0, 0, 1);
	cv::Mat rmat = R_z * R_y * R_x;
	cv::Mat convertRT = (cv::Mat_<float>(4, 4) <<
 	   rmat.at<double>(0, 0), rmat.at<double>(0, 1), rmat.at<double>(0, 2), convertVec[0],
 	   rmat.at<double>(1, 0), rmat.at<double>(1, 1), rmat.at<double>(1, 2), convertVec[1],
 	   rmat.at<double>(2, 0), rmat.at<double>(2, 1), rmat.at<double>(2, 2), convertVec[2],
  	   0, 0, 0, 1);


	//坐标转换
	for (auto bottomPoint : bottomPoints) {
		//num++;
		float cameraZ = bottomPoint[2];
	 	float cameraX = cameraZ * (bottomPoint[0] - cameraParam[2]) / cameraParam[0];
	 	float cameraY = cameraZ * (bottomPoint[1] - cameraParam[3]) / cameraParam[1];
		cv::Mat cameraCoord = (cv::Mat_<float>(4,1) << cameraX, cameraY, cameraZ, 1.0); 
	 	cv::Mat worldCoord = convertRT * cameraCoord;

		//求解w、h
		std::vector<cv::Point3f> worldPts;
		for(int i=4; i<10; i+=2)
		{
			float pX = cameraZ * (bottomPoint[i] - cameraParam[2]) / cameraParam[0];
	 		float pY = cameraZ * (bottomPoint[i+1] - cameraParam[3]) / cameraParam[1];
			cv::Mat camPoint = (cv::Mat_<float>(4,1) << pX, pY, cameraZ, 1.0); 
			cv::Mat worldPoint = convertRT * camPoint;
			worldPts.push_back(cv::Point3f(worldPoint.at<float>(0,0), worldPoint.at<float>(1,0), worldPoint.at<float>(2,0)));
		}
		float w = sqrtf(powf((worldPts[0].x - worldPts[1].x), 2) + powf((worldPts[0].y - worldPts[1].y), 2) + powf((worldPts[0].z - worldPts[1].z), 2));
		float h = sqrtf(powf((worldPts[2].x - worldPts[1].x), 2) + powf((worldPts[2].y - worldPts[1].y), 2) + powf((worldPts[2].z - worldPts[1].z), 2));

		Detection lostDet = {0,0,cv::RotatedRect(),0,bottomPoint[0],bottomPoint[1],bottomPoint[2],w,h,
							worldCoord.at<float>(0,0), worldCoord.at<float>(1,0), worldCoord.at<float>(2,0)};
		result.push_back(lostDet);
	}
}


bool YOLO11::sortRuleWY(const Detection p1, const Detection p2)
{
	if (p1.worldBY < p2.worldBY)
		return true;
	else
		return false;
}

bool YOLO11::sortRuleWZ(const std::vector<Detection> p1, const std::vector<Detection> p2)
{
	if (p1[0].worldBZ > p2[0].worldBZ)
		return true;
	else
		return false;
}

void YOLO11::drawRotated(cv::Mat image, std::vector<Detection>& output) {
	auto colorMat = image.clone();

	cv::Scalar color;

	for (const auto& out : output) {
		cv::Point2f vertices[4];
		out.rbox.points(vertices);

		if (out.modelID == 1)
			color = cv::Scalar(0, 0, 255); 
		else if (out.modelID == 2)
			color = cv::Scalar(0, 255, 0); 
		else
			color = cv::Scalar(255, 255, 255); // 未知 ID 使用白色

		for(int i = 0; i < 4; i++) 
			cv::line(colorMat, vertices[i], vertices[(i + 1) % 4], color, 2);
		std::string label = std::string("box ") + std::to_string(out.conf).substr(0, 4);
		// cv::rectangle(
		// 	colorMat,
		// 	cv::Point(box.x, box.y - 25),
		// 	cv::Point(box.x + label.length() * 15, box.y),
		// 	color,
		// 	cv::FILLED);
		cv::Point labelPos(vertices[0].x, vertices[0].y - 10);
		cv::putText(
			colorMat,
			label,
			labelPos,
			cv::FONT_HERSHEY_SIMPLEX,
			0.75,
			color,
			2);

		cv::circle(colorMat, cv::Point2f(out.bottomX, out.bottomY), 5, color, -1); // 实心圆，半径为5
	}

	auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto ms = now_ms.time_since_epoch().count() % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S")
    << "_" << std::setfill('0') << std::setw(3) << ms;
    std::string timestamp = ss.str();
    std::string filename = "result_" + timestamp + ".jpg";

    // 保存图像
    cv::imwrite(filename, colorMat);
}

