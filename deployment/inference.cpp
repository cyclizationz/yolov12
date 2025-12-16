#include "inference.h"
#include <regex>
#include <iostream>

#define benchmark
// #define min(a,b) (((a) < (b)) ? (a) : (b)) // Removed to avoid conflict with std::min

YOLO_V8::YOLO_V8() {
    session = nullptr;
}

YOLO_V8::~YOLO_V8() {
    if (session) {
        delete session;
        session = nullptr;
    }
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    // Letterbox padding (Center alignment)
    // This matches Ultralytics behavior and should fix coordinate shifts
    float r = std::min(iImgSize.at(0) / (float)iImg.rows, iImgSize.at(1) / (float)iImg.cols);
    
    int new_unpad_w = int(round(iImg.cols * r));
    int new_unpad_h = int(round(iImg.rows * r));
    
    if (new_unpad_w != oImg.cols || new_unpad_h != oImg.rows) {
        cv::resize(oImg, oImg, cv::Size(new_unpad_w, new_unpad_h));
    }
    
    // Calculate padding
    int dw = iImgSize.at(1) - new_unpad_w;
    int dh = iImgSize.at(0) - new_unpad_h;
    
    dw /= 2; // divide padding by 2 for center
    dh /= 2;
    
    // Store padding values for coordinate mapping
    this->padW = dw;
    this->padH = dh;
    this->resizeScales = r; // Scale factor (target/orig)
    
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    // Add 114 padding color if needed, but zeros is fine for now (or 114 for gray)
    tempImg = cv::Scalar(114, 114, 114); // Ultralytics uses 114 for padding
    
    oImg.copyTo(tempImg(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
    oImg = tempImg;
    
    return RET_OK;
}

char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        maskConfidenceThreshold = iParams.maskConfidenceThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            cudaEnable = iParams.cudaEnable;
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char* modelPath = iParams.modelPath.c_str();

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[YOLO_V8]: " << e.what() << std::endl;
        return (char*)"[YOLO_V8]:Create session failed.";
    }
}

char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif 

    char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 5) // FLOAT32 models
    {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return Ret;
}

template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif 
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif 

    // Output processing
    auto& output0 = outputTensor[0];
    auto output0_info = output0.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output0_dims = output0_info.GetShape();
    
    delete[] blob;

    switch (modelType)
    {
    case YOLO_SEG_V12:
    case YOLO_SEG_V12_HALF:
    {
        // Dimensions: (1, 4+cls+32, 8400) e.g. (1, 37, 8400) for 1 class
        int channels = output0_dims[1]; // 37
        int anchors = output0_dims[2];  // 8400
        
        // Get Proto masks: (1, 32, 160, 160)
        auto& output1 = outputTensor[1];
        auto output1_info = output1.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output1_dims = output1_info.GetShape();
        int proto_channels = output1_dims[1]; // 32
        int proto_height = output1_dims[2];   // 160
        int proto_width = output1_dims[3];    // 160
        
        float* data0 = nullptr;
        float* data1 = nullptr;
        
        if (modelType == YOLO_SEG_V12) {
            data0 = output0.GetTensorMutableData<float>();
            data1 = output1.GetTensorMutableData<float>();
        }

        // Transpose output0 to (8400, 37) for easier row access
        cv::Mat rawData = cv::Mat(channels, anchors, CV_32F, data0);
        rawData = rawData.t(); // (8400, 37)
        float* data = (float*)rawData.data;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> mask_coeffs;

        for (int i = 0; i < anchors; i++) {
            float* row = data + i * channels;
            float* classes_scores = row + 4;
            
            int num_classes = channels - 4 - 32; 
            
            cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > rectConfidenceThreshold) {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];

                // Map back to original image with center padding
                // (x - padW) / scale
                
                float left_f = (x - 0.5 * w - padW) / resizeScales;
                float top_f = (y - 0.5 * h - padH) / resizeScales;
                float width_f = w / resizeScales;
                float height_f = h / resizeScales;

                int left = int(left_f);
                int top = int(top_f);
                int width = int(width_f);
                int height = int(height_f);
                
                // Clip boxes
                left = std::max(0, left);
                top = std::max(0, top);
                width = std::min(width, iImg.cols - left);
                height = std::min(height, iImg.rows - top);

                if (width > 0 && height > 0) {
                    boxes.push_back(cv::Rect(left, top, width, height));
                    
                    // Extract mask coefficients
                    std::vector<float> coeffs;
                    for (int j = 0; j < 32; j++) {
                        coeffs.push_back(row[4 + num_classes + j]);
                    }
                    mask_coeffs.push_back(coeffs);
                }
            }
        }

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

        // Process Masks
        cv::Mat proto = cv::Mat(proto_channels, proto_height * proto_width, CV_32F, data1);

        for (int idx : nmsResult) {
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            
            // Generate mask
            // 1. Matrix multiply coeffs (1, 32) * proto (32, 25600) -> (1, 25600)
            cv::Mat coeffMat = cv::Mat(1, 32, CV_32F, mask_coeffs[idx].data());
            cv::Mat maskMat = coeffMat * proto; // (1, 25600)
            maskMat = maskMat.reshape(1, proto_height); // (160, 160)
            
            // 2. Sigmoid
            cv::exp(-maskMat, maskMat);
            maskMat = 1.0 / (1.0 + maskMat);
            
            // 3. Resize to input image size (640x640)
            cv::Mat resizedMask;
            cv::resize(maskMat, resizedMask, cv::Size(imgSize[1], imgSize[0]));
            
            // Crop valid area (excluding padding)
            // valid area in resizedMask (640x640) corresponds to [padW, padH, unpad_w, unpad_h]
            // Actually, resizedMask corresponds to the padded image.
            // We want the part that corresponds to the image.
            
            // padW, padH are in 640x640 space.
            int unpad_w = imgSize[1] - 2 * padW;
            int unpad_h = imgSize[0] - 2 * padH;
            
            cv::Rect valid_rect(padW, padH, unpad_w, unpad_h);
            // Clip to bounds
            valid_rect = valid_rect & cv::Rect(0, 0, resizedMask.cols, resizedMask.rows);
            
            if (valid_rect.area() > 0) {
                cv::Mat croppedMask = resizedMask(valid_rect);
                
                // Resize cropped mask to original image size
                cv::resize(croppedMask, result.boxMask, iImg.size());
                
                // Threshold
                result.boxMask = result.boxMask > 0.5;
                
                // Mask inside box only
                cv::Mat finalMask = cv::Mat::zeros(iImg.size(), CV_8UC1);
                cv::Rect safeBox = result.box & cv::Rect(0, 0, iImg.cols, iImg.rows);
                if (safeBox.area() > 0) {
                    result.boxMask(safeBox).copyTo(finalMask(safeBox));
                }
                result.boxMask = finalMask;
            } else {
                result.boxMask = cv::Mat::zeros(iImg.size(), CV_8UC1);
            }

            oResult.push_back(result);
        }
        break;
    }
    default:
        std::cout << "[YOLO_V8]: Not support model type." << std::endl;
    }
    return RET_OK;
}

char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    
    float* blob = new float[iImg.total() * 3];
    BlobFromImage(processedImg, blob);
    std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
    auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    delete[] blob;
    
    clock_t starttime_4 = clock();
    double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
    if (cudaEnable)
    {
        std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
    }
    return RET_OK;
}
