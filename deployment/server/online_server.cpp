#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "inference.h"
#include "offline_processor.h"

namespace fs = std::filesystem;

namespace {

void print_usage(const char *program_name) {
    std::cout
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "\n"
        << "RESPAWN online-server scaffold. This wraps the current evaluator in\n"
        << "server terminology and writes the artifacts needed by the deferred\n"
        << "online/Exp5 client path.\n"
        << "\n"
        << "Options:\n"
        << "  -i, --input <path>          Input video path\n"
        << "  -o, --output <dir>          Output artifact directory\n"
        << "  -m, --model <path>          YOLO segmentation ONNX model\n"
        << "      --cuda                  Enable CUDA inference\n"
        << "      --cpu                   Force CPU inference\n"
        << "      --pixel                 Use pixel-template detector mode\n"
        << "      --templates <dir>       Pixel-template directory\n"
        << "      --latent-key            Use latent-key template identifiers\n"
        << "      --latent-bank <path>    Load a prebuilt latent bank\n"
        << "      --latent-no-mint        Match only against prebuilt templates\n"
        << "      --heal-only             Only mask recoverable regions\n"
        << "      --cache-mode <mode>     cold|warm|partial-warm (recorded/logged)\n"
        << "      --max-frames <n>        Stop after n frames\n"
        << "      --enc-crf <n>           Encoder CRF\n"
        << "      --enc-bitrate-mbps <f>  Enable target bitrate mode\n"
        << "      --enc-preset <s>        Encoder preset\n"
        << "      --enc-tune <s>          Encoder tune, or none\n"
        << "      --enc-profile <s>       Encoder profile, or none\n"
        << "      --enc-level <s>         Encoder level, or none\n"
        << "      --enc-open-gop-defaults Leave keyint/scenecut defaults open\n"
        << "      --enc-aud <0|1>         Emit access unit delimiters\n"
        << "      --enc-repeat-headers <0|1> Repeat stream headers\n"
        << "      --mask-color <s>        Mask color strategy\n"
        << "      --mask-color-period <n> Dominant color refresh period\n"
        << "      --mask-color-local-pad <n> Local dominant-color pad\n"
        << "      --mask-color-local-per-region Use per-region local fill color\n"
        << "      --mask-color-local-stat <s> Local dominant statistic\n"
        << "      --fill-mode <s>         solid|blur|bg_ema|inpaint\n"
        << "      --feather-px <n>        Feather band in pixels\n"
        << "      --pixel-*               Common Mario pixel-mode knobs\n"
        << "      --help                  Show this help\n";
}

bool require_value(int &i, int argc, char **argv, std::string &out) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value for " << argv[i] << "\n";
        return false;
    }
    out = argv[++i];
    return true;
}

} // namespace

int main(int argc, char **argv) {
    OfflineOptions opt;
    opt.source = "/home/tiehangz/proj/datasets/racing/fm6.mkv";
    opt.model = "../yolov12n-seg.onnx";
    opt.outDir = "../outputs/online_server";
    opt.recordTiming = true;
    opt.yoloLatentKey = true;

    std::string cache_mode = "cold";
    {
        std::ostringstream oss;
        for (int i = 0; i < argc; ++i) {
            if (i) oss << ' ';
            oss << argv[i];
        }
        opt.commandline = oss.str();
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        if (arg == "-i" || arg == "--input") {
            if (!require_value(i, argc, argv, opt.source)) return 2;
        } else if (arg == "-o" || arg == "--output") {
            if (!require_value(i, argc, argv, opt.outDir)) return 2;
        } else if (arg == "-m" || arg == "--model") {
            if (!require_value(i, argc, argv, opt.model)) return 2;
        } else if (arg == "--cuda") {
            opt.useCuda = true;
        } else if (arg == "--cpu") {
            opt.useCuda = false;
        } else if (arg == "--pixel") {
            opt.pixelMode = true;
        } else if (arg == "--templates") {
            if (!require_value(i, argc, argv, opt.templatesDir)) return 2;
        } else if (arg == "-T") {
            if (!require_value(i, argc, argv, opt.templatesDir)) return 2;
        } else if (arg == "--latent-key") {
            opt.yoloLatentKey = true;
            opt.yoloMatcher = "latent_key";
        } else if (arg == "--latent-bank") {
            if (!require_value(i, argc, argv, opt.latentBankLoadPath)) return 2;
        } else if (arg == "--latent-no-mint") {
            opt.latentNoMint = true;
        } else if (arg == "--heal-only") {
            opt.yoloHealOnly = true;
        } else if (arg == "--cache-mode") {
            if (!require_value(i, argc, argv, cache_mode)) return 2;
            if (cache_mode != "cold" && cache_mode != "warm" && cache_mode != "partial-warm") {
                std::cerr << "Unsupported cache mode: " << cache_mode << "\n";
                return 2;
            }
        } else if (arg == "--max-frames") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.maxFrames = std::stoi(value);
        } else if (arg == "--enc-crf") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.encCrf = std::stoi(value);
        } else if (arg == "--enc-bitrate-mbps") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.encBitrateMbps = std::stod(value);
        } else if (arg == "--enc-preset") {
            if (!require_value(i, argc, argv, opt.encPreset)) return 2;
        } else if (arg == "--enc-tune") {
            if (!require_value(i, argc, argv, opt.encTune)) return 2;
        } else if (arg == "--enc-profile") {
            if (!require_value(i, argc, argv, opt.encProfile)) return 2;
        } else if (arg == "--enc-level") {
            if (!require_value(i, argc, argv, opt.encLevel)) return 2;
        } else if (arg == "--enc-open-gop-defaults") {
            opt.encOpenGopDefaults = true;
        } else if (arg == "--enc-aud") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.encAud = (std::stoi(value) != 0);
        } else if (arg == "--enc-repeat-headers") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.encRepeatHeaders = (std::stoi(value) != 0);
        } else if (arg == "--mask-color") {
            if (!require_value(i, argc, argv, opt.maskColor)) return 2;
        } else if (arg == "--mask-color-period") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.maskColorPeriod = std::stoi(value);
        } else if (arg == "--mask-color-local-pad") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.maskColorLocalPadPx = std::stoi(value);
        } else if (arg == "--mask-color-local-per-region") {
            opt.maskColorLocalPerRegion = true;
        } else if (arg == "--mask-color-local-stat") {
            if (!require_value(i, argc, argv, opt.maskColorLocalStat)) return 2;
        } else if (arg == "--fill-mode") {
            if (!require_value(i, argc, argv, opt.fillMode)) return 2;
        } else if (arg == "--feather-px") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.featherPx = std::stoi(value);
        } else if (arg == "--pixel-force-scale") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelForceScale = std::stof(value);
        } else if (arg == "--pixel-mask-pad") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelMaskPadPx = std::stoi(value);
        } else if (arg == "--pixel-grid-header") {
            opt.pixelGridHeader = true;
        } else if (arg == "--pixel-grid-snap") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelGridSnapTolPx = std::stoi(value);
        } else if (arg == "--pixel-band-pad-y") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelBandPadY = std::stoi(value);
        } else if (arg == "--pixel-bands") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelNumBands = std::stoi(value);
        } else if (arg == "--pixel-max-peaks") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelMaxPeaks = std::stoi(value);
        } else if (arg == "--pixel-bootstrap") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelBootstrapInterval = std::stoi(value);
        } else if (arg == "--pixel-flow") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelUseFlow = (std::stoi(value) != 0);
        } else if (arg == "--pixel-adaptive-thr") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelAdaptiveThr = (std::stoi(value) != 0);
        } else if (arg == "--pixel-thr-k") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelThrK = std::stof(value);
        } else if (arg == "--pixel-thr-lo") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelThrLo = std::stof(value);
        } else if (arg == "--pixel-thr-hi") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelThrHi = std::stof(value);
        } else if (arg == "--pixel-min-score") {
            if (!require_value(i, argc, argv, value)) return 2;
            opt.pixelMinScore = std::stof(value);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    fs::create_directories(opt.outDir);
    opt.dictDir = opt.outDir + "/dict";
    opt.reportPath = opt.outDir + "/report.json";
    fs::create_directories(opt.dictDir);

    std::cout << "[OnlineServer] cache_mode=" << cache_mode
              << " source=" << opt.source
              << " outDir=" << opt.outDir
              << " dictDir=" << opt.dictDir
              << "\n";
    std::cout << "[OnlineServer] This scaffold emits segmented_output.mp4, "
              << "msk1_payloads.bin, dict/, recovered_output.mp4, and report.json. "
              << "Live network transport is not enabled in this target.\n";

    YOLO_V8 yolo;
    if (!opt.pixelMode) {
        DL_INIT_PARAM params;
        params.modelPath = opt.model;
        params.modelType = YOLO_SEG_V12;
        params.imgSize = {640, 640};
        params.rectConfidenceThreshold = opt.confThreshold;
        params.iouThreshold = 0.5;
        params.maskConfidenceThreshold = opt.maskThreshold;
        params.cudaEnable = opt.useCuda;
        if (yolo.CreateSession(params) != RET_OK) {
            std::cerr << "Failed to initialize YOLO session\n";
            return 1;
        }
    }

    return run_offline_evaluation(opt, yolo);
}
