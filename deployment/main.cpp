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
              << "  --pixel-bootstrap <n>   Pixel: global re-bootstrap interval in frames (default: 30)\n"
              << "  --pixel-topk <n>        Pixel: keep top-K templates after bootstrap (default: 3)\n"
              << "  --pixel-roi-pad <px>    Pixel: ROI padding around predicted box (default: 40)\n"
              << "  --pixel-min-score <f>   Pixel: min NCC score to accept ROI match (default: 0.65)\n"
              << "  --pixel-band-pad-y <px> Pixel: vertical band pad for multi-peak scan (default: 20)\n"
              << "  --pixel-max-peaks <n>   Pixel: max detections per frame in band scan (default: 30)\n"
              << "  --pixel-peak-nms <px>   Pixel: suppression radius around a peak (default: 6)\n"
              << "  --yolo-class-color      YOLO mode: paint masks with class-consistent colors (instead of hash colors)\n"
              << "  --timing                Record per-frame timing stats into report.json\n"
              << "  --latent-key            YOLO offline sim: assign template_id by maskCoeff embedding (no client hashing)\n"
              << "  --latent-thr <float>    Cosine similarity threshold for template reuse (default: 0.95)\n"
              << "  --latent-period <int>   Base sampling period (default: 2 => ~50% templates)\n"
              << "  --latent-period-skip <f> Skip periodic mint if best cosine >= this (default: disabled)\n"
              << "  --latent-merge <f>      When minting, merge into existing if best cosine >= this (default: disabled)\n"
              << "  --latent-motion-iou <f> Motion trigger IoU threshold (default: 0.6)\n"
              << "  --latent-motion-center <f> Motion trigger center shift in px (default: 20)\n"
              << "  --latent-motion-scale <f> Motion trigger area ratio delta (default: 0.25)\n"
              << "  --latent-motion-boost <int> Boost window frames (default: 6)\n"
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
        {"pixel-bootstrap", required_argument, 0, 'b'},
        {"pixel-topk", required_argument, 0, 'k'},
        {"pixel-roi-pad", required_argument, 0, 'r'},
        {"pixel-min-score", required_argument, 0, 'q'},
        {"pixel-band-pad-y", required_argument, 0, 'u'},
        {"pixel-max-peaks", required_argument, 0, 'v'},
        {"pixel-peak-nms", required_argument, 0, 'w'},
        {"latent-key", no_argument, 0, 'L'},
        {"latent-thr", required_argument, 0, 'Z'},
        {"latent-period", required_argument, 0, 'P'},
        {"latent-period-skip", required_argument, 0, 'K'},
        {"latent-merge", required_argument, 0, 'M'},
        {"latent-motion-iou", required_argument, 0, 'I'},
        {"latent-motion-center", required_argument, 0, 'O'},
        {"latent-motion-scale", required_argument, 0, 'S'},
        {"latent-motion-boost", required_argument, 0, 'B'},
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
            case 'b':
                opt.pixelBootstrapInterval = std::stoi(optarg);
                break;
            case 'k':
                opt.pixelTopK = std::stoi(optarg);
                break;
            case 'r':
                opt.pixelRoiPad = std::stoi(optarg);
                break;
            case 'q':
                opt.pixelMinScore = std::stof(optarg);
                break;
            case 'u':
                opt.pixelBandPadY = std::stoi(optarg);
                break;
            case 'v':
                opt.pixelMaxPeaks = std::stoi(optarg);
                break;
            case 'w':
                opt.pixelPeakNms = std::stoi(optarg);
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
            case 'P':
                opt.latentSamplePeriod = std::stoi(optarg);
                break;
            case 'K':
                opt.latentPeriodSkipThr = std::stof(optarg);
                break;
            case 'M':
                opt.latentMergeThr = std::stof(optarg);
                break;
            case 'I':
                opt.latentMotionIouThr = std::stof(optarg);
                break;
            case 'O':
                opt.latentMotionCenterPx = std::stof(optarg);
                break;
            case 'S':
                opt.latentMotionScaleThr = std::stof(optarg);
                break;
            case 'B':
                opt.latentMotionBoostFrames = std::stoi(optarg);
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
