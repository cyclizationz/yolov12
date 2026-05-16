#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

#include "inference.h"
#include "offline_processor.h"
#include "offline_processor_duo.h"

namespace fs = std::filesystem;

static void print_usage(const char *program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  -i <input_video_path>   Input video file (default: /home/tiehangz/proj/datasets/racing/fm6.mkv)\n"
              << "  -o <output_dir>         Output directory for comparison artifacts (default: <repo>/record/duo_channel)\n"
              << "  -m <model_path>         Path to the ONNX model (default: <repo>/deployment/yolov12n-seg.onnx)\n"
              << "  -c <confidence>         Detection confidence threshold (default: 0.2)\n"
              << "  -t <iou_threshold>      NMS IOU threshold (default: 0.5)\n"
              << "  -s <mask_threshold>     Segmentation mask confidence threshold (default: 0.6)\n"
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
              << "  --pixel-bands <n>       Pixel: number of horizontal bands to scan (default: 2)\n"
              << "  --pixel-band-sep <px>   Pixel: min separation between band centers (default: 24)\n"
              << "  --pixel-flow <0|1>      Pixel: enable sparse optical flow stabilization (default: 1)\n"
              << "  --pixel-kalman <0|1>    Pixel: enable Kalman ROI tracking (default: 1)\n"
              << "  --pixel-flow-pts <n>    Pixel: max LK points (default: 120)\n"
              << "  --pixel-band-topk <n>   Pixel: scan top-K templates per frame in band scan (0=all active, default: 2)\n"
              << "  --pixel-adaptive-thr <0|1> Pixel: adaptive per-template thresholding (default: 1)\n"
              << "  --pixel-thr-k <f>       Pixel: threshold factor for adaptive threshold (default: 0.85)\n"
              << "  --pixel-thr-lo <f>      Pixel: adaptive threshold clamp low (default: 0.30)\n"
              << "  --pixel-thr-hi <f>      Pixel: adaptive threshold clamp high (default: 0.65)\n"
              << "  --pixel-force-scale <f> Pixel: force template scale (0=auto, 1.0=native size)\n"
              << "  --pixel-mask-pad <px>   Pixel: expand emitted mask/recovery boxes by this many px (default: 0)\n"
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
              << "  --yolo-heal-only        YOLO mode: only mask regions recoverable from client templates\n"
              << "  --yolo-heal-iou <f>     YOLO heal-only: require IoU(template_alpha, current_mask) >= f (default: 0.0 disabled)\n"
              << "  --yolo-heal-extra <f>   YOLO heal-only: allow alpha spill outside mask <= f (default: 0.02)\n"
              << "  --yolo-heal-fallback-window <n> Heal-only: fallback to most-similar latent used within last n frames (default: 24)\n"
              << "  --yolo-heal-fallback-minsim <f> Heal-only: min cosine sim for fallback reuse (default: 0.85)\n"
              << "  --yolo-heal-fallback-minsim-i <f> Heal-only: min cosine sim on GOP boundary frames (default: 0.80)\n"
              << "  --yolo-heal-max-shift <n> Heal-only: max translation search in px for template alignment (default: 8)\n"
              << "  --yolo-heal-min-coverage <f> Heal-only: min seg-mask coverage of template alpha after alignment (default: 0.15)\n"
              << "  --yolo-heal-mse <f>      Heal-only: masked RGB MSE threshold for template acceptance (default: 700)\n"
              << "  --build-index            Prefill template pool (dict + latent_bank.json); no masking/encoding\n"
              << "  --latent-bank <PATH>     Load latent bank JSON before processing (default: <dictDir>/latent_bank.json if present)\n"
              << "  --latent-no-mint         Disallow minting new templates (match-only from prebuilt pool)\n"
              << "  --gop <n>               Approximate GOP size for offline analysis (default: 60)\n"
              << "  --enc-gop <n>           Encoder keyint/GOP size (default: 18)\n"
              << "  --enc-crf <n>           Encoder CRF (default: 18)\n"
              << "  --enc-bitrate-mbps <f>  Encoder target bitrate in Mbps (enables bitrate mode)\n"
              << "  --enc-maxrate-mbps <f>  Encoder maxrate in Mbps (defaults to bitrate)\n"
              << "  --enc-bufsize-mbits <f> Encoder VBV bufsize in Mbits (defaults to 2x maxrate)\n"
              << "  --enc-preset <s>        Encoder preset (default: superfast)\n"
              << "  --enc-tune <s>          Encoder tune (default: zerolatency)\n"
              << "  --enc-profile <s>       Encoder profile (default: baseline)\n"
              << "  --enc-level <s>         Encoder level (default: 4.2)\n"
              << "  --enc-scenecut <0|1>    Encoder scenecut (default: 0)\n"
              << "  --enc-aud <0|1>         Encoder access unit delimiters (default: 1)\n"
              << "  --enc-repeat-headers <0|1> Encoder repeat_headers (default: 1)\n"
              << "  --enc-codec <s>         Encoder codec (ffmpeg -c:v), e.g. libx264|libx265|libsvtav1|libaom-av1 (default: libx264)\n"
              << "  --no-ffmpeg-enc         Use OpenCV VideoWriter instead of FFmpeg pipe encoder\n"
              << "  --yolo-force-mask-all   YOLO debug: paint every segmentation mask (upper-bound bitrate test; breaks recovery)\n"
              << "  --mask-color <s>        Mask paint color: green|black|brown|class|dominant|rgb:R,G,B (default: green)\n"
              << "  --mask-color-period <n> Recompute dominant color every N frames (default: 200; only for mask-color=dominant)\n"
              << "  --fill-mode <s>         Fill mode: solid|blur|bg_ema|inpaint (default: solid)\n"
              << "  --feather-px <n>        Feather band (px): blend fill->original across boundary, 0 disables (default: 0)\n"
              << "  --fill-blur-sigma <f>   Blur sigma (px) for fill-mode=blur (default: 8.0)\n"
              << "  --fill-bg-ema-alpha <f> EMA alpha for fill-mode=bg_ema (default: 0.98)\n"
              << "  --fill-inpaint-radius <n> Inpaint radius (px) for fill-mode=inpaint (default: 5)\n"
              << "  --fill-inpaint-method <s> Inpaint method telea|ns (default: telea)\n"
              << "  --paint-alpha <f>       Mask paint alpha for visualization/forced-masking (default: 1.0)\n"
              << "  --max-frames <n>        Stop after N frames (default: -1 unlimited)\n"
              << "  --cpu                   Force CPU inference (explicit flag)\n"
              << "  -h                      Show this help message\n";
}

int main(int argc, char *argv[]) {
    fs::path deploymentDir = fs::path(__FILE__).parent_path();
    fs::path repoRoot = deploymentDir.parent_path();
    OfflineOptions opt;
    opt.source = "/home/tiehangz/proj/datasets/racing/fm6.mkv";
    opt.model = (deploymentDir / "yolov12n-seg.onnx").string();
    opt.outDir = (repoRoot / "record" / "duo_channel").string();

    {
        std::ostringstream oss;
        for (int i = 0; i < argc; i++) {
            if (i) oss << ' ';
            oss << argv[i];
        }
        opt.commandline = oss.str();
    }

    enum {
        OPT_PIXEL_BAND_TOPK = 1001,
        OPT_PIXEL_ADAPTIVE_THR = 1002,
        OPT_PIXEL_THR_K = 1003,
        OPT_PIXEL_THR_LO = 1004,
        OPT_PIXEL_THR_HI = 1005,
        OPT_YOLO_FORCE_MASK_ALL = 1006,
        OPT_YOLO_HEAL_FALLBACK_WINDOW = 1007,
        OPT_YOLO_HEAL_FALLBACK_MINSIM = 1008,
        OPT_YOLO_HEAL_FALLBACK_MINSIM_I = 1009,
        OPT_YOLO_HEAL_MAX_SHIFT = 1010,
        OPT_YOLO_HEAL_MIN_COVERAGE = 1011,
        OPT_YOLO_HEAL_MSE = 1012,
        OPT_BUILD_INDEX = 1013,
        OPT_LATENT_BANK = 1014,
        OPT_LATENT_NO_MINT = 1015,
        OPT_GOP_SIZE = 1016,
        OPT_ENC_GOP = 1017,
        OPT_ENC_CRF = 1018,
        OPT_ENC_BITRATE_MBPS = 1036,
        OPT_ENC_MAXRATE_MBPS = 1037,
        OPT_ENC_BUFSIZE_MBITS = 1038,
        OPT_ENC_PRESET = 1019,
        OPT_ENC_TUNE = 1020,
        OPT_ENC_PROFILE = 1021,
        OPT_ENC_LEVEL = 1022,
        OPT_ENC_SCENECUT = 1023,
        OPT_ENC_AUD = 1024,
        OPT_ENC_REPEAT_HEADERS = 1025,
        OPT_NO_FFMPEG_ENC = 1026,
        OPT_MASK_COLOR = 1027,
        OPT_MASK_COLOR_PERIOD = 1028,
        OPT_ENC_CODEC = 1029,
        OPT_FILL_MODE = 1030,
        OPT_FEATHER_PX = 1031,
        OPT_FILL_BLUR_SIGMA = 1032,
        OPT_FILL_BG_EMA_ALPHA = 1033,
        OPT_FILL_INPAINT_RADIUS = 1034,
        OPT_FILL_INPAINT_METHOD = 1035,
        OPT_DUMP_FRAME_CACHE = 1039,
        OPT_PIXEL_FORCE_SCALE = 1100,
        OPT_PIXEL_MASK_PAD = 1102
    };

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"cpu", no_argument, 0, 'C'},
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
        {"pixel-bands", required_argument, 0, 'g'},
        {"pixel-band-sep", required_argument, 0, 'j'},
        {"pixel-flow", required_argument, 0, 'f'},
        {"pixel-kalman", required_argument, 0, 1040},
        {"pixel-flow-pts", required_argument, 0, 'x'},
        {"pixel-band-topk", required_argument, 0, OPT_PIXEL_BAND_TOPK},
        {"pixel-adaptive-thr", required_argument, 0, OPT_PIXEL_ADAPTIVE_THR},
        {"pixel-thr-k", required_argument, 0, OPT_PIXEL_THR_K},
        {"pixel-thr-lo", required_argument, 0, OPT_PIXEL_THR_LO},
        {"pixel-thr-hi", required_argument, 0, OPT_PIXEL_THR_HI},
        {"pixel-force-scale", required_argument, 0, OPT_PIXEL_FORCE_SCALE},
        {"pixel-mask-pad", required_argument, 0, OPT_PIXEL_MASK_PAD},
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
        {"yolo-heal-only", no_argument, 0, 'H'},
        {"yolo-heal-iou", required_argument, 0, 'J'},
        {"yolo-heal-extra", required_argument, 0, 'Q'},
        {"yolo-heal-fallback-window", required_argument, 0, OPT_YOLO_HEAL_FALLBACK_WINDOW},
        {"yolo-heal-fallback-minsim", required_argument, 0, OPT_YOLO_HEAL_FALLBACK_MINSIM},
        {"yolo-heal-fallback-minsim-i", required_argument, 0, OPT_YOLO_HEAL_FALLBACK_MINSIM_I},
        {"yolo-heal-max-shift", required_argument, 0, OPT_YOLO_HEAL_MAX_SHIFT},
        {"yolo-heal-min-coverage", required_argument, 0, OPT_YOLO_HEAL_MIN_COVERAGE},
        {"yolo-heal-mse", required_argument, 0, OPT_YOLO_HEAL_MSE},
        {"build-index", no_argument, 0, OPT_BUILD_INDEX},
        {"latent-bank", required_argument, 0, OPT_LATENT_BANK},
        {"latent-no-mint", no_argument, 0, OPT_LATENT_NO_MINT},
        {"gop", required_argument, 0, OPT_GOP_SIZE},
        {"enc-gop", required_argument, 0, OPT_ENC_GOP},
        {"enc-crf", required_argument, 0, OPT_ENC_CRF},
        {"enc-bitrate-mbps", required_argument, 0, OPT_ENC_BITRATE_MBPS},
        {"enc-maxrate-mbps", required_argument, 0, OPT_ENC_MAXRATE_MBPS},
        {"enc-bufsize-mbits", required_argument, 0, OPT_ENC_BUFSIZE_MBITS},
        {"enc-preset", required_argument, 0, OPT_ENC_PRESET},
        {"enc-tune", required_argument, 0, OPT_ENC_TUNE},
        {"enc-profile", required_argument, 0, OPT_ENC_PROFILE},
        {"enc-level", required_argument, 0, OPT_ENC_LEVEL},
        {"enc-scenecut", required_argument, 0, OPT_ENC_SCENECUT},
        {"enc-aud", required_argument, 0, OPT_ENC_AUD},
        {"enc-repeat-headers", required_argument, 0, OPT_ENC_REPEAT_HEADERS},
        {"enc-codec", required_argument, 0, OPT_ENC_CODEC},
        {"no-ffmpeg-enc", no_argument, 0, OPT_NO_FFMPEG_ENC},
        {"yolo-force-mask-all", no_argument, 0, OPT_YOLO_FORCE_MASK_ALL},
        {"mask-color", required_argument, 0, OPT_MASK_COLOR},
        {"mask-color-period", required_argument, 0, OPT_MASK_COLOR_PERIOD},
        {"fill-mode", required_argument, 0, OPT_FILL_MODE},
        {"feather-px", required_argument, 0, OPT_FEATHER_PX},
        {"fill-blur-sigma", required_argument, 0, OPT_FILL_BLUR_SIGMA},
        {"fill-bg-ema-alpha", required_argument, 0, OPT_FILL_BG_EMA_ALPHA},
        {"fill-inpaint-radius", required_argument, 0, OPT_FILL_INPAINT_RADIUS},
        {"fill-inpaint-method", required_argument, 0, OPT_FILL_INPAINT_METHOD},
        {"paint-alpha", required_argument, 0, 'A'},
        {"max-frames", required_argument, 0, 'N'},
        {0, 0, 0, 0}
    };

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "i:o:m:c:t:s:A:de:hpT:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'i': opt.source = optarg; break;
            case 'o': opt.outDir = optarg; break;
            case 'm': opt.model = optarg; break;
            case 'c': opt.confThreshold = std::stof(optarg); break;
            case 't': opt.phashThreshold = std::stoi(optarg); break;
            case 's': opt.maskThreshold = std::stof(optarg); break;
            case 'd': opt.useCuda = true; break;
            case 'e': break;
            case 'C': opt.useCuda = false; break;
            case 'p': opt.pixelMode = true; break;
            case 'T': opt.templatesDir = optarg; break;
            case 'b': opt.pixelBootstrapInterval = std::stoi(optarg); break;
            case 'k': opt.pixelTopK = std::stoi(optarg); break;
            case 'r': opt.pixelRoiPad = std::stoi(optarg); break;
            case 'q': opt.pixelMinScore = std::stof(optarg); break;
            case 'u': opt.pixelBandPadY = std::stoi(optarg); break;
            case 'v': opt.pixelMaxPeaks = std::stoi(optarg); break;
            case 'w': opt.pixelPeakNms = std::stoi(optarg); break;
            case 'g': opt.pixelNumBands = std::stoi(optarg); break;
            case 'j': opt.pixelBandMinSep = std::stoi(optarg); break;
            case 'f': opt.pixelUseFlow = (std::stoi(optarg) != 0); break;
            case 1040: opt.pixelUseKalman = (std::stoi(optarg) != 0); break;
            case 'x': opt.pixelFlowMaxPts = std::stoi(optarg); break;
            case OPT_PIXEL_BAND_TOPK: opt.pixelBandTopK = std::stoi(optarg); break;
            case OPT_PIXEL_ADAPTIVE_THR: opt.pixelAdaptiveThr = (std::stoi(optarg) != 0); break;
            case OPT_PIXEL_THR_K: opt.pixelThrK = std::stof(optarg); break;
            case OPT_PIXEL_THR_LO: opt.pixelThrLo = std::stof(optarg); break;
            case OPT_PIXEL_THR_HI: opt.pixelThrHi = std::stof(optarg); break;
            case OPT_PIXEL_FORCE_SCALE: opt.pixelForceScale = std::max(0.0f, std::stof(optarg)); break;
            case OPT_PIXEL_MASK_PAD: opt.pixelMaskPadPx = std::max(0, std::stoi(optarg)); break;
            case 'Y': opt.yoloClassConsistentColor = true; break;
            case 'R': opt.recordTiming = true; break;
            case 'L': opt.yoloLatentKey = true; break;
            case 'Z': opt.latentCosineThreshold = std::stof(optarg); break;
            case 'P': opt.latentSamplePeriod = std::stoi(optarg); break;
            case 'K': opt.latentPeriodSkipThr = std::stof(optarg); break;
            case 'M': opt.latentMergeThr = std::stof(optarg); break;
            case 'I': opt.latentMotionIouThr = std::stof(optarg); break;
            case 'O': opt.latentMotionCenterPx = std::stof(optarg); break;
            case 'S': opt.latentMotionScaleThr = std::stof(optarg); break;
            case 'B': opt.latentMotionBoostFrames = std::stoi(optarg); break;
            case 'D': opt.dumpLatents = true; if (optarg) opt.latentsOutPath = optarg; break;
            case 'H': opt.yoloHealOnly = true; break;
            case 'J': opt.yoloHealMaskIouThr = std::stof(optarg); break;
            case 'Q': opt.yoloHealMaskExtraThr = std::stof(optarg); break;
            case OPT_YOLO_HEAL_FALLBACK_WINDOW: opt.yoloHealFallbackWindowFrames = std::stoi(optarg); break;
            case OPT_YOLO_HEAL_FALLBACK_MINSIM: opt.yoloHealFallbackMinSim = std::stof(optarg); break;
            case OPT_YOLO_HEAL_FALLBACK_MINSIM_I: opt.yoloHealFallbackMinSimI = std::stof(optarg); break;
            case OPT_YOLO_HEAL_MAX_SHIFT: opt.yoloHealMaxShiftPx = std::stoi(optarg); break;
            case OPT_YOLO_HEAL_MIN_COVERAGE: opt.yoloHealMinMaskCoverage = std::stof(optarg); break;
            case OPT_YOLO_HEAL_MSE: opt.yoloHealAppearanceMseThr = std::stof(optarg); break;
            case OPT_BUILD_INDEX: opt.buildIndexOnly = true; break;
            case OPT_LATENT_BANK: opt.latentBankLoadPath = optarg; break;
            case OPT_LATENT_NO_MINT: opt.latentNoMint = true; break;
            case OPT_GOP_SIZE: opt.gopSize = std::stoi(optarg); break;
            case OPT_ENC_GOP: opt.encGop = std::stoi(optarg); break;
            case OPT_ENC_CRF: opt.encCrf = std::stoi(optarg); break;
            case OPT_ENC_BITRATE_MBPS: opt.encBitrateMbps = std::max(0.0, std::stod(optarg)); break;
            case OPT_ENC_MAXRATE_MBPS: opt.encMaxrateMbps = std::max(0.0, std::stod(optarg)); break;
            case OPT_ENC_BUFSIZE_MBITS: opt.encBufsizeMbits = std::max(0.0, std::stod(optarg)); break;
            case OPT_ENC_PRESET: opt.encPreset = optarg; break;
            case OPT_ENC_TUNE: opt.encTune = optarg; break;
            case OPT_ENC_PROFILE: opt.encProfile = optarg; break;
            case OPT_ENC_LEVEL: opt.encLevel = optarg; break;
            case OPT_ENC_SCENECUT: opt.encNoScenecut = (std::stoi(optarg) == 0); break;
            case OPT_ENC_AUD: opt.encAud = (std::stoi(optarg) != 0); break;
            case OPT_ENC_REPEAT_HEADERS: opt.encRepeatHeaders = (std::stoi(optarg) != 0); break;
            case OPT_NO_FFMPEG_ENC: opt.useFfmpegEncoder = false; break;
            case OPT_YOLO_FORCE_MASK_ALL: opt.yoloForceMaskAll = true; break;
            case OPT_MASK_COLOR: {
                std::string v = optarg ? std::string(optarg) : std::string();
                for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                if (v == "class") {
                    opt.maskColor = "class";
                    opt.yoloClassConsistentColor = true;
                } else if (v == "black") {
                    opt.maskColor = "black";
                    opt.maskColorR = 0; opt.maskColorG = 0; opt.maskColorB = 0;
                } else if (v == "brown") {
                    opt.maskColor = "brown";
                    opt.maskColorR = 165; opt.maskColorG = 42; opt.maskColorB = 42;
                } else if (v == "dominant") {
                    opt.maskColor = "dominant";
                } else if (v.rfind("rgb:", 0) == 0) {
                    int r = 0, g = 255, b = 0;
                    try {
                        std::string rest = v.substr(4);
                        size_t p1 = rest.find(',');
                        size_t p2 = (p1 == std::string::npos) ? std::string::npos : rest.find(',', p1 + 1);
                        if (p1 != std::string::npos && p2 != std::string::npos) {
                            r = std::stoi(rest.substr(0, p1));
                            g = std::stoi(rest.substr(p1 + 1, p2 - (p1 + 1)));
                            b = std::stoi(rest.substr(p2 + 1));
                        }
                    } catch (...) {}
                    auto clamp = [](int x){ return std::max(0, std::min(255, x)); };
                    opt.maskColor = "rgb";
                    opt.maskColorR = clamp(r);
                    opt.maskColorG = clamp(g);
                    opt.maskColorB = clamp(b);
                } else {
                    opt.maskColor = "green";
                    opt.maskColorR = 0; opt.maskColorG = 255; opt.maskColorB = 0;
                }
                break;
            }
            case OPT_MASK_COLOR_PERIOD: opt.maskColorPeriod = std::max(1, std::stoi(optarg)); break;
            case OPT_ENC_CODEC: opt.encCodec = optarg ? std::string(optarg) : std::string("libx264"); break;
            case OPT_FILL_MODE: {
                std::string v = optarg ? std::string(optarg) : std::string();
                for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                opt.fillMode = v.empty() ? "solid" : v;
                break;
            }
            case OPT_FEATHER_PX: opt.featherPx = std::max(0, std::stoi(optarg)); break;
            case OPT_FILL_BLUR_SIGMA: opt.fillBlurSigma = std::max(0.0f, std::stof(optarg)); break;
            case OPT_FILL_BG_EMA_ALPHA: {
                float a = std::stof(optarg);
                if (a < 0.0f) a = 0.0f;
                if (a > 0.9999f) a = 0.9999f;
                opt.fillBgEmaAlpha = a;
                break;
            }
            case OPT_FILL_INPAINT_RADIUS: opt.fillInpaintRadius = std::max(1, std::stoi(optarg)); break;
            case OPT_FILL_INPAINT_METHOD: {
                std::string v = optarg ? std::string(optarg) : std::string();
                for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                opt.fillInpaintMethod = v.empty() ? "telea" : v;
                break;
            }
            case 'A': opt.paintAlpha = std::stof(optarg); break;
            case 'N': opt.maxFrames = std::stoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            case '?': break;
        }
    }

    fs::create_directories(opt.outDir);
    opt.dictDir = opt.outDir + "/dict";
    fs::create_directories(opt.dictDir);

    std::cout << "[Command]";
    for (int i = 0; i < argc; i++) std::cout << " " << argv[i];
    std::cout << std::endl;

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

    std::cout << "[Config] source=" << opt.source
              << " outDir=" << opt.outDir
              << " maxFrames=" << opt.maxFrames
              << " healOnly=" << (opt.yoloHealOnly ? 1 : 0)
              << " latentNoMint=" << (opt.latentNoMint ? 1 : 0)
              << " encCodec=" << opt.encCodec
              << " encCrf=" << opt.encCrf
              << " encBitrateMbps=" << opt.encBitrateMbps
              << " encMaxrateMbps=" << opt.encMaxrateMbps
              << " encBufsizeMbits=" << opt.encBufsizeMbits
              << " encGop=" << opt.encGop
              << " bframes=" << opt.encBFrames
              << " scenecut=" << (opt.encNoScenecut ? 0 : 1)
              << " tune=" << opt.encTune
              << " maskColor=" << opt.maskColor
              << " fillMode=" << opt.fillMode
              << " featherPx=" << opt.featherPx
              << " fillBlurSigma=" << opt.fillBlurSigma
              << " fillBgEmaAlpha=" << opt.fillBgEmaAlpha
              << " fillInpaintRadius=" << opt.fillInpaintRadius
              << " fillInpaintMethod=" << opt.fillInpaintMethod
              << std::endl;

    std::cout << "Processing " << opt.source << "...\n";
    return run_duo_channel_evaluation(opt, yoloDetector);
}
