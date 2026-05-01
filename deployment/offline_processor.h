#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#else
// Some distros install headers under /usr/include/opencv4 without adding include flags to tooling.
#include <opencv4/opencv2/opencv.hpp>
#endif
#include <nlohmann/json.hpp>
#include "inference.h" // For YOLO_V8

// Options structure for offline processing
struct OfflineOptions {
    std::string source; // path to video file
    std::string model = "./yolov12n-seg.onnx"; // Default model path
    // Full argv for reproducibility (recorded into report.json and often run.log).
    std::string commandline;
    bool useCuda = false; // Default to CPU
    int cudaID = 0;
    float confThreshold = 0.2f; // Default confidence threshold
    float maskThreshold = 0.6f; // Default mask threshold
    std::string outDir = "../outputs"; // Output directory for processed video, metadata, etc.
    std::string dictDir = "../outputs/dict"; // Dictionary directory
    std::string reportPath = "../outputs/report.json"; // Report path
    std::string stitchedVideoOut = ""; // evaluation processed output
    std::string originalVideoOut = ""; // evaluation original output
    int maxFrames = -1; // unlimited
    float paintAlpha = 1.0f; // alpha for deterministic color painting (1.0 => solid mask)
    int phashThreshold = 8; // Tolerant matching threshold (Hamming distance on 64-bit phash)
    bool pixelMode = false;  // If true, use template-matching detector for pixel games instead of YOLO
    std::string templatesDir; // Directory containing multiple templates for pixel mode
    bool yoloClassConsistentColor = false; // YOLO mode: paint masks with class-consistent colors (better motion pred)
    // YOLO matching policy for single-channel offline ablations.
    // - "latent_key": maskCoeff embedding cosine reuse
    // - "phash": legacy pHash reuse on cropped ROI
    // - "iou_only": nearest previous box / IoU-only reuse
    // - "rgb_hist": nearest-template RGB histogram reuse
    // Empty keeps backward-compatible behavior: --latent-key => latent_key, otherwise phash.
    std::string yoloMatcher = "";
    // Mask paint color for encoder bitrate experiments / visualization.
    // - "green" (default): legacy constant green
    // - "black": constant black (often compresses better and avoids edge halos perception)
    // - "class": use class-consistent palette (equivalent to --yolo-class-color)
    // - "brown": constant brown
    // - "dominant": compute dominant color from frame histogram every maskColorPeriod frames
    // - "rgb:R,G,B": constant color specified in RGB (0-255 each)
    std::string maskColor = "green";
    int maskColorPeriod = 200; // only used when maskColor == "dominant"
    int maskColorR = 0, maskColorG = 255, maskColorB = 0; // used for rgb: and named constants
    bool recordTiming = false; // Record per-frame timing stats into report.json (no extra printing)
    bool yoloLatentKey = false; // YOLO offline simulator: assign template_id via maskCoeff embedding, no hashing on client
    float latentCosineThreshold = 0.95f; // cosine similarity threshold to reuse template_id

    // Build-index mode: prefill template pool (dict + latent_bank.json) from the original stream.
    // This simulates an offline/first-pass server analysis to populate a stable pool before enabling heal-only.
    bool buildIndexOnly = false;
    bool latentNoMint = false;           // disallow minting new templates (match-only run from prebuilt pool)
    std::string latentBankLoadPath = ""; // optional: load latent bank JSON before processing (defaults to <dictDir>/latent_bank.json if exists)

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
    // Defaults are intentionally not ultra-strict because real segmentations have 1-2px jitter.
    // Set to 0 to disable IoU gating and rely on spill control (extra_thr) instead.
    float yoloHealMaskIouThr = 0.0f;     // IoU(template_alpha, current_mask) within bbox
    float yoloHealMaskExtraThr = 0.02f;  // allow small alpha "extra" outside current mask (ratio)

    // Heal-only fallback: if strict latent reuse threshold is too strict, allow using the most similar
    // recently-used latent template within a short temporal window (vision persistence).
    int yoloHealFallbackWindowFrames = 24;
    float yoloHealFallbackMinSim = 0.85f;
    float yoloHealFallbackMinSimI = 0.80f; // more aggressive on GOP boundaries
    int yoloHealMaxShiftPx = 8;            // max translation (px) to align template to current frame
    float yoloHealMinMaskCoverage = 0.15f; // require some overlap between seg mask and template alpha after alignment
    float yoloHealAppearanceMseThr = 700.0f; // masked RGB MSE(template, frame ROI) threshold for acceptance

    // Approximate GOP boundary (I-frame) by frame index modulo gopSize.
    // NOTE: This is an offline approximation; actual encoder I-frames depend on codec settings.
    int gopSize = 60;

    // Upper-bound experiment: paint every YOLO mask regardless of template match / healability.
    // This is ONLY for measuring potential H.264 bitrate savings (recovery will not be correct).
    bool yoloForceMaskAll = false;

    // Experiment support: dump per-frame latent embeddings to analyze similarity/thresholds offline.
    bool dumpLatents = false;
    std::string latentsOutPath = ""; // defaults to <outDir>/latents.jsonl when enabled
    // When enabled, dump pre-encode original/masked BGR frames to raw cache files.
    // This is used by offline multi-variant experiments that need to re-encode
    // alternate timelines without introducing a second lossy encode stage.
    bool dumpFrameCache = false;
    // Full-frame temporal hysteresis on the final raw/ref frame-mode decision.
    // When enabled, a new mode must be requested for N consecutive frames before
    // the emitted stream flips to that mode.
    bool frameModeHysteresis = true;
    int frameModeHysteresisConfirmFrames = 2;

    // ---------------------------------------------------------------------
    // Controlled encoding (server-side): use FFmpeg/libx264 via stdin pipe.
    // This avoids OpenCV VideoWriter's opaque encoder settings and lets us
    // fix GOP/keyint, scenecut, bframes, preset/profile/crf.
    // ---------------------------------------------------------------------
    bool useFfmpegEncoder = true;
    int encGop = 18;                 // keyint / GOP size (frames)
    int encBFrames = 0;              // bframes
    int encCrf = 18;                 // CRF quality
    double encBitrateMbps = 0.0;     // if >0, enable bitrate-targeted mode
    double encMaxrateMbps = 0.0;     // if <=0 and bitrate mode enabled, defaults to encBitrateMbps
    double encBufsizeMbits = 0.0;    // if <=0 and bitrate mode enabled, defaults to 2x bitrate
    std::string encPreset = "superfast";
    std::string encTune = "zerolatency";
    std::string encProfile = "baseline";
    std::string encLevel = "4.2";
    bool encNoScenecut = true;       // scenecut=0
    bool encRepeatHeaders = true;    // repeat_headers=1
    bool encAud = true;              // aud=1 (helps parsers)

    // Encoder codec (ffmpeg -c:v). Default keeps existing behavior.
    // Examples: libx264, libx265, libsvtav1, libaom-av1
    std::string encCodec = "libx264";

    // ---------------------------------------------------------------------
    // Mask fill + feathering (edge-aware masking for bitrate experiments)
    // ---------------------------------------------------------------------
    // fillMode:
    // - "solid": constant color (uses --mask-color or rgb)
    // - "blur": fill from a heavily blurred version of the original frame
    // - "bg_ema": fill from a running EMA background estimate (updated only on unmasked pixels)
    // - "inpaint": boundary-matched fill using OpenCV inpaint (Telea / Navier-Stokes)
    std::string fillMode = "solid";
    int featherPx = 0;               // 0 disables feather; else blend to original across band (px)
    float fillBlurSigma = 8.0f;      // blur fill sigma (pixels) when fillMode=blur
    float fillBgEmaAlpha = 0.98f;    // EMA alpha for bg_ema (closer to 1 => slower adaptation)
    int fillInpaintRadius = 5;       // inpaint radius (pixels)
    std::string fillInpaintMethod = "telea"; // telea|ns

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
    bool pixelUseKalman = true;     // enable Kalman prediction/correction in ROI tracking
    int pixelFlowMaxPts = 120;      // max points for LK flow

    // Pixel multi-template scanning
    int pixelBandTopK = 2;          // run band scan for top-K templates (ranked by ROI score). 0 => scan all active

    // Pixel adaptive thresholding (reduces confusion by forcing high-score templates to be higher)
    bool pixelAdaptiveThr = true;
    float pixelThrK = 0.85f;        // accept threshold = clamp(pixelThrK * calib_max, pixelThrLo, pixelThrHi)
    float pixelThrLo = 0.30f;
    float pixelThrHi = 0.65f;

    // Pixel template scaling:
    // - 0.0 (default): auto-calibrate per template from the first frame (may choose !=1.0)
    // - >0: force a fixed scale for all templates (e.g., 1.0 uses native template size)
    float pixelForceScale = 0.0f;
    // Optional safety cap for pixel-template loading. 0 disables the cap.
    int pixelMaxTemplates = 0;
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
