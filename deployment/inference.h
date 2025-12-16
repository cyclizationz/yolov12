#pragma once

#define RET_OK nullptr

#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,
    YOLO_SEG_V12 = 4, // Added for YOLOv12 Segmentation

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 5,
    YOLO_POSE_V8_HALF = 6,
    YOLO_CLS_HALF = 7,
    YOLO_SEG_V12_HALF = 8 // Added for YOLOv12 Segmentation (FP16)
};

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_SEG_V12;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.25;
    float iouThreshold = 0.45;
    float maskConfidenceThreshold = 0.5; // Added mask threshold
    int	keyPointsNum = 2;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
    cv::Mat boxMask; // Added for segmentation mask
    std::string seiPath; // For pixel mode / SEI: relative template path key for stitching
    std::array<float, 32> maskCoeff{}; // YOLOv12-seg mask embedding (latent vector), normalized later if needed
} DL_RESULT;

class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();

public:
    char* CreateSession(DL_INIT_PARAM& iParams);
    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    char* WarmUpSession();
    
    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    
    std::vector<std::string> classes{};

private:
    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float maskConfidenceThreshold; // Added
    float resizeScales;
    
    // Padding values for coordinate mapping
    int padW = 0;
    int padH = 0;
};
