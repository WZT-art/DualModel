#pragma once

#include "YOLO11.h"
#include "ThreadPool.h"

class ModelManager {
public:
    ModelManager() = default;
    ~ModelManager() = default;

    void addModel(const std::string& modelPath) {
        auto trt = std::make_unique<TensorRT>();
        trt->modelPath = modelPath;

        modelMap_[modelPath] = std::move(trt);
    } 

    bool initModel(const std::string& modelPath) {
        auto it = modelMap_.find(modelPath);
        if (it == modelMap_.end()) {
            LOG_ERROR("Model[", modelPath, "] not found");
            return false; // Model not found
        }

        
    }
private:
    ThreadPool threadPool_;
    std::map<std::string, std::unique_ptr<TensorRT>> modelMap_;
    std::mutex modelMutex_;
    std::unique_ptr<YOLO11> yolo11_;
};