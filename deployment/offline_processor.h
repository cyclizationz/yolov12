#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "inference.h" // For YOLO_V8

// Options structure for offline processing
struct OfflineOptions {
    std::string source; // path to video file
    std::string model = "./yolov12n-seg.onnx"; // Default model path
    bool useCuda = false; // Default to CPU
    int cudaID = 0;
    float confThreshold = 0.25f; // Lowered default confidence threshold
    float maskThreshold = 0.5f; // Default mask threshold
    std::string outDir = "../outputs"; // Output directory for processed video, metadata, etc.
    std::string dictDir = "../outputs/dict"; // Dictionary directory
    std::string reportPath = "../outputs/report.json"; // Report path
    std::string stitchedVideoOut = ""; // evaluation processed output
    std::string originalVideoOut = ""; // evaluation original output
    int maxFrames = -1; // unlimited
    float paintAlpha = 0.6f; // alpha for deterministic color painting
    int phashThreshold = 8; // Tolerant matching threshold (Hamming distance on 64-bit phash)
    bool pixelMode = false;  // If true, use template-matching detector for pixel games instead of YOLO
    std::string templatesDir; // Directory containing multiple templates for pixel mode
    bool yoloClassConsistentColor = false; // YOLO mode: paint masks with class-consistent colors (better motion pred)
    bool recordTiming = false; // Record per-frame timing stats into report.json (no extra printing)
    bool yoloLatentKey = false; // YOLO offline simulator: assign template_id via maskCoeff embedding, no hashing on client
    float latentCosineThreshold = 0.95f; // cosine similarity threshold to reuse template_id

    // Latent-key hybrid sampling policy:
    // - Normally mint a new template every `latentSamplePeriod` frames (e.g. 2 => ~50%).
    // - If motion is significant, mint every frame for `latentMotionBoostFrames` frames.
    int latentSamplePeriod = 2;          // base sampling period (frames)
    // NOTE: In high-motion rendered games, aggressive server-side reuse/compaction can increase perceptual
    // artifacts (flash/mismatch). We keep these disabled by default; prefer hybrid sampling + client fallback reuse.
    float latentPeriodSkipThr = 2.0f;    // disabled by default (cosine in [-1,1])
    float latentMergeThr = 2.0f;         // disabled by default (cosine in [-1,1])
    float latentMotionIouThr = 0.6f;     // trigger boost if IoU(last_box, cur_box) < this
    float latentMotionCenterPx = 20.0f;  // trigger boost if center shift (px) > this
    float latentMotionScaleThr = 0.25f;  // trigger boost if area ratio differs by > this (e.g. 0.25 => 25%)
    int latentMotionBoostFrames = 6;     // boost window length in frames

    // Reliability: only mask when the region is recoverable with client-known templates.
    // This avoids green-edge artifacts when a reused template's alpha mask doesn't match
    // the current frame's segmentation mask.
    //
    // Policy when enabled:
    // - New templates (minted this frame) are treated as NOT recoverable (client may not have them yet).
    // - Reused templates must pass a mask agreement check vs current mask inside bbox.
    // - If not recoverable, we keep original pixels (do not paint mask, do not emit SEI region).
    bool yoloHealOnly = false;
    float yoloHealMaskIouThr = 0.995f;   // IoU(template_alpha, current_mask) within bbox
    float yoloHealMaskExtraThr = 0.005f; // allow small alpha "extra" outside current mask (ratio)

    // Experiment support: dump per-frame latent embeddings to analyze similarity/thresholds offline.
    bool dumpLatents = false;
    std::string latentsOutPath = ""; // defaults to <outDir>/latents.jsonl when enabled

    // Pixel mode tracking (Kalman + ROI template matching)
    int pixelBootstrapInterval = 30; // frames between global re-bootstrap
    int pixelTopK = 3;              // keep top-K templates after bootstrap
    int pixelRoiPad = 40;           // ROI padding around predicted box (px)
    float pixelMinScore = 0.45f;    // min combined score to accept ROI match (edge NCC + color check)
    int pixelLostMax = 5;           // re-bootstrap after N consecutive low-score frames

    // Pixel multi-instance expansion: scan a narrow Y band but full width, output multiple peaks
    int pixelBandPadY = 20;         // vertical pad around predicted y (px)
    int pixelMaxPeaks = 30;         // max detections per frame from band scan
    int pixelPeakNms = 6;           // suppress radius (px) around a selected peak in response map

    // Pixel band selection + flow stabilization
    int pixelNumBands = 2;          // how many horizontal bands to scan per frame
    int pixelBandMinSep = 24;       // minimum separation between band centers (px)
    bool pixelUseFlow = true;       // enable sparse optical flow stabilization
    int pixelFlowMaxPts = 120;      // max points for LK flow

    // Pixel multi-template scanning
    int pixelBandTopK = 2;          // run band scan for top-K templates (ranked by ROI score). 0 => scan all active

    // Pixel adaptive thresholding (reduces confusion by forcing high-score templates to be higher)
    bool pixelAdaptiveThr = true;
    float pixelThrK = 0.85f;        // accept threshold = clamp(pixelThrK * calib_max, pixelThrLo, pixelThrHi)
    float pixelThrLo = 0.30f;
    float pixelThrHi = 0.65f;
};

struct DictItem {
    std::string hash;
    std::string path;
    int width{0};
    int height{0};
    int cls{0};
    uint64_t file_size{0};
    uint64_t count{0};
    uint64_t phash64{0};
};

// Core processing class to handle dictionary management and evaluation
class OfflineProcessor {
public:
    OfflineProcessor(const std::string& outDir, const std::string& dictDir);
    
    // Dictionary management
    bool loadDictionary();
    void saveDictionary();
    bool addToDictionary(const std::string& hash, const cv::Mat& mask, const cv::Rect& box, int cls, const cv::Mat& frame);
    
    // Reconstruction helpers
    void loadTemplates(); // Load all template images into memory for reconstruction
    void overlayTemplate(cv::Mat& frame, const std::string& hash, const cv::Rect& box);

    // Mask hashing helpers
    static std::string sha256_norm(const cv::Mat &mat);
    static uint64_t mask_phash64(const cv::Mat &mask);
    static int hamming64(uint64_t a, uint64_t b);
    static cv::Vec3b deterministic_color_from_hash(const std::string &hash);
    
    // Rendering helpers
    static void paint_mask_color(cv::Mat &dst_bgr, const cv::Mat &mask, const cv::Rect &bbox, const cv::Vec3b &color, float alpha);
    static cv::Mat extract_object_rgba(const cv::Mat &frame_bgr, const cv::Mat &mask, const cv::Rect &bbox);
    
    // Metrics
    static double computeSSIM(const cv::Mat &i1, const cv::Mat &i2);

    // Get dictionary access
    const std::unordered_map<std::string, DictItem>& getDict() const { return dict; }
    std::unordered_map<std::string, DictItem>& getDictMutable() { return dict; }

private:
    std::string outDir;
    std::string dictDir;
    std::string debugDir; // Directory for debug masks
    int debugSavedCount = 0; // Counter for saved debug images
    std::unordered_map<std::string, DictItem> dict;
    std::unordered_map<std::string, cv::Mat> loadedTemplates; // Cache for template images
};

// Function to run the offline evaluation process
int run_offline_evaluation(const OfflineOptions &opt, YOLO_V8& yoloDetector);
