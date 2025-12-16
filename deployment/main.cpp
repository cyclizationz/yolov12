#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unistd.h> 
#include <getopt.h> // Use getopt_long

#include "inference.h"
#include "offline_processor.h" 

namespace fs = std::filesystem;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  -i <input_video_path>   Input video file (default: /home/tiehangz/proj/datasets/racing/fm6.mkv)\n"
              << "  -o <output_dir>         Output directory for video, metadata, and dictionary (default: ../outputs)\n"
              << "  -m <model_path>         Path to the ONNX model (default: ../yolov12n-seg.onnx)\n"
              << "  -c <confidence>         Detection confidence threshold (default: 0.25)\n"
              << "  -t <iou_threshold>      NMS IOU threshold (default: 0.5)\n"
              << "  -s <mask_threshold>     Segmentation mask confidence threshold (default: 0.5)\n"
              << "  -d                      Enable CUDA (GPU) inference (default: CPU)\n"
              << "  -e                      Enable offline evaluation (default: false) [Ignored in this version, always runs]\n"
              << "  -p, --pixel             Use template-matching detector for pixel games (skip YOLO model)\n"
              << "  -T <templates_dir>      Directory containing multiple pixel templates (defaults from TM_TEMPLATE_PATH or ./)\n"
              << "  --yolo-class-color      YOLO mode: paint masks with class-consistent colors (instead of hash colors)\n"
              << "  --timing                Record per-frame timing stats into report.json\n"
              << "  --latent-key            YOLO offline sim: assign template_id by maskCoeff embedding (no client hashing)\n"
              << "  --latent-thr <float>    Cosine similarity threshold for template reuse (default: 0.95)\n"
              << "  --dump-latents[=PATH]   Dump per-frame latent embeddings to JSONL (default: <outDir>/latents.jsonl)\n"
              << "  --cpu                   Force CPU inference (explicit flag)\n"
              << "  -h                      Show this help message\n";
}

int main(int argc, char* argv[]) {
    OfflineOptions opt;
    opt.source = "/home/tiehangz/proj/datasets/racing/fm6.mkv"; // Default input video
    opt.model = "../yolov12n-seg.onnx"; // Default model path
    opt.outDir = "../outputs"; // Default output directory

    static struct option long_options[] = {
        {"cpu",   no_argument, 0, 'C'},
        {"pixel", no_argument, 0, 'p'},
        {"yolo-class-color", no_argument, 0, 'Y'},
        {"timing", no_argument, 0, 'R'},
        {"latent-key", no_argument, 0, 'L'},
        {"latent-thr", required_argument, 0, 'Z'},
        {"dump-latents", optional_argument, 0, 'D'},
        {0, 0, 0, 0}
    };

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "i:o:m:c:t:s:de:hpT:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'i':
                opt.source = optarg;
                break;
            case 'o':
                opt.outDir = optarg;
                break;
            case 'm':
                opt.model = optarg;
                break;
            case 'c':
                opt.confThreshold = std::stof(optarg);
                break;
            case 't':
                opt.phashThreshold = std::stoi(optarg);
                break;
            case 's':
                opt.maskThreshold = std::stof(optarg);
                break;
            case 'd':
                opt.useCuda = true;
                break;
            case 'e':
                // Ignored
                break;
            case 'C':
                opt.useCuda = false;
                break;
        case 'p':
                opt.pixelMode = true;
                break;
            case 'T':
                opt.templatesDir = optarg;
                break;
            case 'Y':
                opt.yoloClassConsistentColor = true;
                break;
            case 'R':
                opt.recordTiming = true;
                break;
            case 'L':
                opt.yoloLatentKey = true;
                break;
            case 'Z':
                opt.latentCosineThreshold = std::stof(optarg);
                break;
            case 'D':
                opt.dumpLatents = true;
                if (optarg) opt.latentsOutPath = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                // Handle unknown options safely
                break;
        }
    }
    
    // Ensure output directories exist
    fs::create_directories(opt.outDir);
    opt.dictDir = opt.outDir + "/dict";
    fs::create_directories(opt.dictDir);

    YOLO_V8 yoloDetector;
    if (!opt.pixelMode) {
        std::cout << "Initializing YOLOv12 Segmentation model...\n";
        DL_INIT_PARAM params;
        params.modelPath = opt.model;
        params.modelType = YOLO_SEG_V12; 
        params.imgSize = {640, 640};
        params.rectConfidenceThreshold = opt.confThreshold;
        params.iouThreshold = 0.5; 
        params.maskConfidenceThreshold = opt.maskThreshold;
        params.cudaEnable = opt.useCuda;

        if (yoloDetector.CreateSession(params) != RET_OK) {
            std::cerr << "Failed to create session: " << std::endl;
            return 1;
        }
    } else {
        std::cout << "Pixel mode enabled: using template-matching detector instead of YOLO model.\n";
    }

    std::cout << "Processing " << opt.source << "...\n";
    return run_offline_evaluation(opt, yoloDetector);
}
