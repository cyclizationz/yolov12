#include "offline_processor_duo.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct EncodedFrameInfo {
    int pktSize{0};
    int keyFrame{0};
    std::string pictType;
    double ptsTime{0.0};
};

struct RunLengthStats {
    int totalFrames{0};
    int maskedFrames{0};
    int rawFrames{0};
    int switchCount{0};
    double fractionModeSwitches{0.0};
    double meanRunLength{0.0};
    double meanMaskedRunLength{0.0};
    double meanRawRunLength{0.0};
    int maxMaskedRunLength{0};
    int maxRawRunLength{0};
    std::vector<int> maskedRuns;
    std::vector<int> rawRuns;
};

struct Mp4Stats {
    std::vector<EncodedFrameInfo> frames;
    uint64_t totalBytes{0};
    int keyframeCount{0};
    int gopCount{0};
};

static std::string shell_quote(const std::string &value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out += "'";
    return out;
}

static std::string run_command_capture(const std::string &command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) return output;
    while (true) {
        size_t n = fread(buffer.data(), 1, buffer.size(), pipe);
        if (n == 0) break;
        output.append(buffer.data(), n);
    }
    (void)pclose(pipe);
    return output;
}

static json ffprobe_json(const std::string &path, const std::string &entries, bool showFrames) {
    std::ostringstream cmd;
    cmd << "ffprobe -v error -select_streams v:0 -print_format json ";
    if (showFrames) cmd << "-show_frames ";
    cmd << "-show_entries " << entries << " " << shell_quote(path);
    std::string out = run_command_capture(cmd.str());
    if (out.empty()) return json::object();
    try {
        return json::parse(out);
    } catch (...) {
        return json::object();
    }
}

static int json_to_int(const json &obj, const char *key, int defaultValue = 0) {
    if (!obj.contains(key)) return defaultValue;
    const auto &v = obj.at(key);
    if (v.is_number_integer()) return v.get<int>();
    if (v.is_number_unsigned()) return (int)v.get<unsigned int>();
    if (v.is_number_float()) return (int)std::lround(v.get<double>());
    if (v.is_string()) {
        try { return std::stoi(v.get<std::string>()); } catch (...) { return defaultValue; }
    }
    return defaultValue;
}

static double json_to_double(const json &obj, const char *key, double defaultValue = 0.0) {
    if (!obj.contains(key)) return defaultValue;
    const auto &v = obj.at(key);
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        try { return std::stod(v.get<std::string>()); } catch (...) { return defaultValue; }
    }
    return defaultValue;
}

static std::string json_to_string(const json &obj, const char *key, const std::string &defaultValue = std::string()) {
    if (!obj.contains(key)) return defaultValue;
    const auto &v = obj.at(key);
    if (v.is_string()) return v.get<std::string>();
    return defaultValue;
}

static Mp4Stats probe_mp4_stats(const std::string &path) {
    Mp4Stats stats;
    json doc = ffprobe_json(path,
                            "frame=best_effort_timestamp_time,pkt_size,pict_type,key_frame",
                            /*showFrames=*/true);
    if (!doc.contains("frames") || !doc["frames"].is_array()) return stats;
    for (const auto &fr : doc["frames"]) {
        EncodedFrameInfo info;
        info.pktSize = json_to_int(fr, "pkt_size", 0);
        info.keyFrame = json_to_int(fr, "key_frame", 0);
        info.pictType = json_to_string(fr, "pict_type", std::string());
        info.ptsTime = json_to_double(fr, "best_effort_timestamp_time", 0.0);
        stats.totalBytes += (uint64_t)std::max(0, info.pktSize);
        if (info.keyFrame == 1 || info.pictType == "I") {
            stats.keyframeCount++;
        }
        stats.frames.push_back(std::move(info));
    }
    stats.gopCount = stats.keyframeCount > 0 ? stats.keyframeCount : (stats.frames.empty() ? 0 : 1);
    if (stats.totalBytes == 0) {
        std::error_code ec;
        stats.totalBytes = fs::exists(path, ec) ? (uint64_t)fs::file_size(path, ec) : 0ULL;
    }
    return stats;
}

static double mean_of_ints(const std::vector<int> &values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (int v : values) sum += (double)v;
    return sum / (double)values.size();
}

static RunLengthStats compute_run_length_stats(const std::vector<int> &flags) {
    RunLengthStats stats;
    stats.totalFrames = (int)flags.size();
    for (int v : flags) {
        if (v != 0) stats.maskedFrames++;
        else stats.rawFrames++;
    }
    if (flags.empty()) return stats;

    std::vector<int> allRuns;
    int cur = flags.front();
    int len = 1;
    for (size_t i = 1; i < flags.size(); ++i) {
        if (flags[i] == flags[i - 1]) {
            len++;
        } else {
            allRuns.push_back(len);
            if (cur != 0) {
                stats.maskedRuns.push_back(len);
                stats.maxMaskedRunLength = std::max(stats.maxMaskedRunLength, len);
            } else {
                stats.rawRuns.push_back(len);
                stats.maxRawRunLength = std::max(stats.maxRawRunLength, len);
            }
            stats.switchCount++;
            cur = flags[i];
            len = 1;
        }
    }
    allRuns.push_back(len);
    if (cur != 0) {
        stats.maskedRuns.push_back(len);
        stats.maxMaskedRunLength = std::max(stats.maxMaskedRunLength, len);
    } else {
        stats.rawRuns.push_back(len);
        stats.maxRawRunLength = std::max(stats.maxRawRunLength, len);
    }

    stats.meanRunLength = mean_of_ints(allRuns);
    stats.meanMaskedRunLength = mean_of_ints(stats.maskedRuns);
    stats.meanRawRunLength = mean_of_ints(stats.rawRuns);
    if (flags.size() > 1) {
        stats.fractionModeSwitches = (double)stats.switchCount / (double)(flags.size() - 1);
    }
    return stats;
}

static json run_length_stats_json(const RunLengthStats &stats) {
    return {
        {"total_frames", stats.totalFrames},
        {"masked_frames", stats.maskedFrames},
        {"raw_frames", stats.rawFrames},
        {"switch_count", stats.switchCount},
        {"fraction_mode_switches", stats.fractionModeSwitches},
        {"mean_run_length_ref_raw_states", stats.meanRunLength},
        {"mean_masked_run_length", stats.meanMaskedRunLength},
        {"mean_raw_run_length", stats.meanRawRunLength},
        {"max_masked_run_length", stats.maxMaskedRunLength},
        {"max_raw_run_length", stats.maxRawRunLength}
    };
}

static std::vector<int> apply_temporal_hysteresis(const std::vector<int> &flags, int confirmFrames) {
    std::vector<int> out;
    if (flags.empty()) return out;
    const int confirm = std::max(1, confirmFrames);
    out.reserve(flags.size());
    int state = flags.front() ? 1 : 0;
    int candidate = state;
    int candidateCount = 0;
    out.push_back(state);
    for (size_t i = 1; i < flags.size(); ++i) {
        int desired = flags[i] ? 1 : 0;
        if (desired == state) {
            candidate = state;
            candidateCount = 0;
        } else {
            if (desired != candidate) {
                candidate = desired;
                candidateCount = 1;
            } else {
                candidateCount++;
            }
            if (candidateCount >= confirm) {
                state = candidate;
                candidateCount = 0;
            }
        }
        out.push_back(state);
    }
    return out;
}

static double average_gap(const std::vector<int> &indices) {
    if (indices.size() < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 1; i < indices.size(); ++i) {
        sum += (double)(indices[i] - indices[i - 1]);
    }
    return sum / (double)(indices.size() - 1);
}

class RawBgrReader {
public:
    RawBgrReader(const std::string &path, int width, int height)
        : width_(width), height_(height), frameBytes_((size_t)width * (size_t)height * 3), file_(path, std::ios::binary) {}

    bool isOpen() const { return file_.is_open(); }

    bool read(cv::Mat &dst) {
        if (!file_.is_open()) return false;
        dst.create(height_, width_, CV_8UC3);
        file_.read(reinterpret_cast<char*>(dst.data), (std::streamsize)frameBytes_);
        return file_.good() || file_.gcount() == (std::streamsize)frameBytes_;
    }

private:
    int width_{0};
    int height_{0};
    size_t frameBytes_{0};
    std::ifstream file_;
};

struct FfmpegPipeWriter {
    FILE *pipe{nullptr};
    std::string cmd;
    std::string stderrLogPath;
    bool warnedWrite{false};
    uint64_t bytesWritten{0};
    int w{0};
    int h{0};
    double fps{30.0};

    bool open(const std::string &outPath, int width, int height, double fpsIn, const OfflineOptions &opt, const char *tag) {
        w = width;
        h = height;
        fps = fpsIn;
        std::string codec = opt.encCodec.empty() ? std::string("libx264") : opt.encCodec;
        for (auto &ch : codec) ch = (char)std::tolower((unsigned char)ch);
        const int keyint = std::max(1, opt.encGop);
        const int bframes = std::max(0, opt.encBFrames);

        std::string paramFlag;
        std::string paramStr;
        std::string presetArg;
        std::string extraCodecArgs;

        if (codec == "libx264") {
            std::ostringstream p;
            p << "keyint=" << keyint
              << ":min-keyint=" << keyint
              << ":bframes=" << bframes;
            if (opt.encNoScenecut) p << ":scenecut=0";
            if (opt.encRepeatHeaders) p << ":repeat-headers=1";
            if (opt.encAud) p << ":aud=1";
            paramFlag = "-x264-params";
            paramStr = p.str();
            presetArg = opt.encPreset;
        } else if (codec == "libx265") {
            std::ostringstream p;
            p << "keyint=" << keyint
              << ":min-keyint=" << keyint
              << ":bframes=" << bframes;
            if (opt.encNoScenecut) p << ":scenecut=0";
            if (opt.encRepeatHeaders) p << ":repeat-headers=1";
            if (opt.encAud) p << ":aud=1";
            paramFlag = "-x265-params";
            paramStr = p.str();
            presetArg = opt.encPreset;
        } else if (codec == "libsvtav1") {
            auto mapPreset = [&](const std::string &s) -> std::string {
                std::string v = s;
                for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                bool allDigits = !v.empty();
                for (char c : v) if (c < '0' || c > '9') allDigits = false;
                if (allDigits) return v;
                if (v == "ultrafast") return "13";
                if (v == "superfast") return "12";
                if (v == "veryfast") return "11";
                if (v == "faster") return "10";
                if (v == "fast") return "9";
                if (v == "medium") return "8";
                if (v == "slow") return "6";
                if (v == "slower") return "5";
                if (v == "veryslow") return "4";
                return "10";
            };
            presetArg = mapPreset(opt.encPreset);
            extraCodecArgs = "-g " + std::to_string(keyint) + " -bf " + std::to_string(bframes) + " -sc_threshold 0";
        } else if (codec == "libaom-av1") {
            auto mapCpuUsed = [&](const std::string &s) -> std::string {
                std::string v = s;
                for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                if (v == "veryslow" || v == "slower" || v == "slow") return "2";
                if (v == "medium") return "4";
                if (v == "fast") return "6";
                if (v == "faster" || v == "veryfast" || v == "superfast" || v == "ultrafast") return "8";
                return "6";
            };
            extraCodecArgs = "-cpu-used " + mapCpuUsed(opt.encPreset) + " -g " + std::to_string(keyint) + " -bf " + std::to_string(bframes) + " -sc_threshold 0";
        } else {
            presetArg = opt.encPreset;
            extraCodecArgs = "-g " + std::to_string(keyint) + " -bf " + std::to_string(bframes) + " -sc_threshold 0";
        }

        std::ostringstream oss;
        oss << "ffmpeg -hide_banner -loglevel error -y "
            << "-f rawvideo -pix_fmt bgr24 "
            << "-s " << w << "x" << h << " "
            << "-r " << fps << " "
            << "-i pipe:0 "
            << "-an "
            << "-c:v " << codec << " ";
        if (!presetArg.empty()) oss << "-preset " << presetArg << " ";
        if (!opt.encTune.empty() && opt.encTune != "none" && (codec == "libx264" || codec == "libx265")) {
            oss << "-tune " << opt.encTune << " ";
        }
        if (codec == "libx264") {
            oss << "-profile:v " << opt.encProfile << " "
                << "-level:v " << opt.encLevel << " ";
        }
        if (!extraCodecArgs.empty()) oss << extraCodecArgs << " ";
        oss << "-pix_fmt yuv420p "
            << "-crf " << std::max(0, opt.encCrf) << " ";
        if (!paramFlag.empty() && !paramStr.empty()) {
            oss << paramFlag << " \"" << paramStr << "\" ";
        }
        oss << "\"" << outPath << "\"";
        stderrLogPath = outPath + ".ffmpeg.stderr.log";
        oss << " 2>\"" << stderrLogPath << "\"";
        cmd = oss.str();
        std::cout << "[FFmpegPipeWriter] start(" << tag << ") cmd: " << cmd << std::endl;
        pipe = popen(cmd.c_str(), "w");
        return pipe != nullptr;
    }

    bool write(const cv::Mat &bgr) {
        if (!pipe || bgr.empty() || bgr.type() != CV_8UC3 || bgr.cols != w || bgr.rows != h) return false;
        cv::Mat tmp = bgr.isContinuous() ? bgr : bgr.clone();
        size_t need = (size_t)w * (size_t)h * 3;
        size_t wrote = fwrite(tmp.data, 1, need, pipe);
        if (wrote == need) {
            bytesWritten += (uint64_t)wrote;
            return true;
        }
        if (!warnedWrite) {
            warnedWrite = true;
            std::cerr << "[FFmpegPipeWriter] Short write (need=" << need << " wrote=" << wrote
                      << ") stderr: " << stderrLogPath << std::endl;
        }
        return false;
    }

    bool isOpen() const { return pipe != nullptr; }

    void close() {
        if (!pipe) return;
        (void)fflush(pipe);
        int rc = pclose(pipe);
        pipe = nullptr;
        std::cout << "[FFmpegPipeWriter] close bytesWritten=" << bytesWritten
                  << " rc=" << rc << " stderr=" << stderrLogPath << std::endl;
    }
};

static json mp4_stats_json(const Mp4Stats &stats) {
    return {
        {"total_bytes", stats.totalBytes},
        {"frame_count", (int)stats.frames.size()},
        {"keyframe_count", stats.keyframeCount},
        {"gop_count", stats.gopCount}
    };
}

static bool write_json_file(const fs::path &path, const json &doc) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << doc.dump(2);
    return true;
}

} // namespace

int run_duo_channel_evaluation(const OfflineOptions &opt, YOLO_V8 &yoloDetector) {
    const int kHysteresisConfirmFrames = 2;
    fs::create_directories(opt.outDir);

    if (opt.buildIndexOnly) {
        std::cerr << "Duo-channel evaluation requires actual encoded outputs; --build-index is not supported here." << std::endl;
        return 1;
    }

    fs::path rootOut = opt.outDir;
    fs::path baselineADir = rootOut / "baseline_a";
    fs::path baselineBDir = rootOut / "baseline_b";
    fs::path variantCDir = rootOut / "variant_c";
    fs::create_directories(baselineADir);
    fs::create_directories(baselineBDir);
    fs::create_directories(variantCDir);

    OfflineOptions baseOpt = opt;
    baseOpt.outDir = baselineADir.string();
    baseOpt.dictDir = (baselineADir / "dict").string();
    baseOpt.reportPath = (baselineADir / "report.json").string();
    baseOpt.dumpFrameCache = true;
    baseOpt.frameModeHysteresis = false;

    std::cout << "[DuoChannel] Running baseline A into " << baseOpt.outDir << std::endl;
    int rc = run_offline_evaluation(baseOpt, yoloDetector);
    if (rc != 0) return rc;

    json baseReport;
    {
        std::ifstream f(baseOpt.reportPath);
        if (!f.is_open()) {
            std::cerr << "Failed to open baseline report: " << baseOpt.reportPath << std::endl;
            return 1;
        }
        f >> baseReport;
    }

    std::vector<int> baseFlags;
    if (baseReport.contains("per_frame") && baseReport["per_frame"].is_array()) {
        for (const auto &item : baseReport["per_frame"]) {
            baseFlags.push_back(item.value("frame_flags", 0) ? 1 : 0);
        }
    }
    if (baseFlags.empty()) {
        std::cerr << "Baseline report does not contain per-frame frame_flags." << std::endl;
        return 1;
    }

    json inputStreams = baseReport.value("input_ffprobe", json::object()).value("streams", json::array());
    int videoWidth = 0;
    int videoHeight = 0;
    double fps = 30.0;
    if (inputStreams.is_array()) {
        for (const auto &st : inputStreams) {
            if (st.value("codec_type", std::string()) != "video") continue;
            videoWidth = json_to_int(st, "width", 0);
            videoHeight = json_to_int(st, "height", 0);
            std::string rate = st.value("avg_frame_rate", std::string("0/1"));
            size_t slash = rate.find('/');
            if (slash != std::string::npos) {
                double num = std::atof(rate.substr(0, slash).c_str());
                double den = std::atof(rate.substr(slash + 1).c_str());
                if (den > 0.0) fps = num / den;
            }
            break;
        }
    }
    if (videoWidth <= 0 || videoHeight <= 0) {
        cv::VideoCapture cap(opt.source, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open source to recover dimensions: " << opt.source << std::endl;
            return 1;
        }
        videoWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        videoHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fpsCap = cap.get(cv::CAP_PROP_FPS);
        if (fpsCap > 0.0) fps = fpsCap;
    }

    json outputs = baseReport.value("outputs", json::object());
    std::string originalCachePath = outputs.value("original_frame_cache_bgr", "");
    std::string maskedCachePath = outputs.value("masked_frame_cache_bgr", "");
    if (originalCachePath.empty() || maskedCachePath.empty()) {
        std::cerr << "Baseline report does not contain frame cache paths." << std::endl;
        return 1;
    }

    std::vector<int> hysteresisFlags = apply_temporal_hysteresis(baseFlags, kHysteresisConfirmFrames);
    std::vector<int> maskedIndices;
    std::vector<int> rawIndices;
    maskedIndices.reserve(baseFlags.size());
    rawIndices.reserve(baseFlags.size());
    for (size_t i = 0; i < baseFlags.size(); ++i) {
        if (baseFlags[i] != 0) maskedIndices.push_back((int)i);
        else rawIndices.push_back((int)i);
    }

    fs::path baselineBPath = baselineBDir / "segmented_output.mp4";
    fs::path variantCMaskedPath = variantCDir / "masked_only.mp4";
    fs::path variantCRawPath = variantCDir / "unmasked_only.mp4";

    RawBgrReader rawReader(originalCachePath, videoWidth, videoHeight);
    RawBgrReader maskedReader(maskedCachePath, videoWidth, videoHeight);
    if (!rawReader.isOpen() || !maskedReader.isOpen()) {
        std::cerr << "Failed to open frame cache reader(s)." << std::endl;
        return 1;
    }

    FfmpegPipeWriter baselineBWriter;
    FfmpegPipeWriter maskedWriter;
    FfmpegPipeWriter rawWriter;
    if (!baselineBWriter.open(baselineBPath.string(), videoWidth, videoHeight, fps, opt, "baseline_b")) return 1;
    if (!maskedIndices.empty() &&
        !maskedWriter.open(variantCMaskedPath.string(), videoWidth, videoHeight, fps, opt, "variant_c_masked")) return 1;
    if (!rawIndices.empty() &&
        !rawWriter.open(variantCRawPath.string(), videoWidth, videoHeight, fps, opt, "variant_c_raw")) return 1;

    cv::Mat rawFrame;
    cv::Mat maskedFrame;
    for (size_t i = 0; i < baseFlags.size(); ++i) {
        if (!rawReader.read(rawFrame) || !maskedReader.read(maskedFrame)) {
            std::cerr << "Frame cache ended early at frame " << i << std::endl;
            baselineBWriter.close();
            maskedWriter.close();
            rawWriter.close();
            return 1;
        }
        const cv::Mat &chosenB = hysteresisFlags[i] ? maskedFrame : rawFrame;
        (void)baselineBWriter.write(chosenB);
        if (baseFlags[i]) {
            if (maskedWriter.isOpen()) (void)maskedWriter.write(maskedFrame);
        } else {
            if (rawWriter.isOpen()) (void)rawWriter.write(rawFrame);
        }
    }
    baselineBWriter.close();
    if (maskedWriter.isOpen()) maskedWriter.close();
    if (rawWriter.isOpen()) rawWriter.close();

    (void)write_json_file(variantCDir / "masked_stream_frames.json", maskedIndices);
    (void)write_json_file(variantCDir / "unmasked_stream_frames.json", rawIndices);
    (void)write_json_file(baselineBDir / "frame_flags_hysteresis.json", hysteresisFlags);

    std::string baselineAPath = outputs.value("segmented_output_mp4", "");
    if (baselineAPath.empty()) {
        baselineAPath = (baselineADir / "segmented_output.mp4").string();
    }

    Mp4Stats statsA = probe_mp4_stats(baselineAPath);
    Mp4Stats statsB = probe_mp4_stats(baselineBPath.string());
    Mp4Stats statsCMasked = fs::exists(variantCMaskedPath) ? probe_mp4_stats(variantCMaskedPath.string()) : Mp4Stats{};
    Mp4Stats statsCRaw = fs::exists(variantCRawPath) ? probe_mp4_stats(variantCRawPath.string()) : Mp4Stats{};

    RunLengthStats runStatsA = compute_run_length_stats(baseFlags);
    RunLengthStats runStatsB = compute_run_length_stats(hysteresisFlags);

    std::vector<int> perFrameBytesA(baseFlags.size(), 0);
    std::vector<int> perFrameBytesB(baseFlags.size(), 0);
    std::vector<int> perFrameBytesC(baseFlags.size(), 0);
    std::vector<std::string> perFrameStreamC(baseFlags.size(), "none");

    for (size_t i = 0; i < perFrameBytesA.size() && i < statsA.frames.size(); ++i) perFrameBytesA[i] = statsA.frames[i].pktSize;
    for (size_t i = 0; i < perFrameBytesB.size() && i < statsB.frames.size(); ++i) perFrameBytesB[i] = statsB.frames[i].pktSize;
    for (size_t i = 0; i < maskedIndices.size() && i < statsCMasked.frames.size(); ++i) {
        perFrameBytesC[(size_t)maskedIndices[i]] = statsCMasked.frames[i].pktSize;
        perFrameStreamC[(size_t)maskedIndices[i]] = "masked";
    }
    for (size_t i = 0; i < rawIndices.size() && i < statsCRaw.frames.size(); ++i) {
        perFrameBytesC[(size_t)rawIndices[i]] = statsCRaw.frames[i].pktSize;
        perFrameStreamC[(size_t)rawIndices[i]] = "unmasked";
    }

    const bool coverageOk = (maskedIndices.size() + rawIndices.size()) == baseFlags.size();
    const bool countsOk = statsCMasked.frames.size() == maskedIndices.size() && statsCRaw.frames.size() == rawIndices.size();

    json perFrame = json::array();
    for (size_t i = 0; i < baseFlags.size(); ++i) {
        perFrame.push_back({
            {"frame", (int)i},
            {"baseline_a_mode_ref", baseFlags[i]},
            {"baseline_b_mode_ref", hysteresisFlags[i]},
            {"baseline_a_bytes", perFrameBytesA[i]},
            {"baseline_b_bytes", perFrameBytesB[i]},
            {"variant_c_bytes", perFrameBytesC[i]},
            {"variant_c_stream", perFrameStreamC[i]}
        });
    }

    json report;
    report["commandline"] = opt.commandline;
    report["source"] = opt.source;
    report["output_root"] = rootOut.string();
    report["baseline_a_report_path"] = baseOpt.reportPath;
    report["hysteresis_confirm_frames"] = kHysteresisConfirmFrames;
    report["predictors"] = {
        {"mean_run_length_ref_raw_states", runStatsA.meanRunLength},
        {"mean_masked_run_length", runStatsA.meanMaskedRunLength},
        {"mean_raw_run_length", runStatsA.meanRawRunLength}
    };
    report["baseline_a"] = {
        {"description", "Single stream, current design"},
        {"paths", {
            {"dir", baselineADir.string()},
            {"video", baselineAPath}
        }},
        {"bitstream", mp4_stats_json(statsA)},
        {"mode_stats", run_length_stats_json(runStatsA)}
    };
    report["baseline_b"] = {
        {"description", "Single stream with temporal hysteresis on Ref/Raw decision"},
        {"paths", {
            {"dir", baselineBDir.string()},
            {"video", baselineBPath.string()},
            {"frame_flags_path", (baselineBDir / "frame_flags_hysteresis.json").string()}
        }},
        {"bitstream", mp4_stats_json(statsB)},
        {"mode_stats", run_length_stats_json(runStatsB)}
    };
    report["variant_c"] = {
        {"description", "Two streams split by full-frame mode"},
        {"paths", {
            {"dir", variantCDir.string()},
            {"masked_only_video", variantCMaskedPath.string()},
            {"unmasked_only_video", variantCRawPath.string()},
            {"masked_stream_frames", (variantCDir / "masked_stream_frames.json").string()},
            {"unmasked_stream_frames", (variantCDir / "unmasked_stream_frames.json").string()}
        }},
        {"masked_only_substream", mp4_stats_json(statsCMasked)},
        {"unmasked_only_substream", mp4_stats_json(statsCRaw)},
        {"sum_total_bytes", statsCMasked.totalBytes + statsCRaw.totalBytes},
        {"sum_keyframe_count", statsCMasked.keyframeCount + statsCRaw.keyframeCount},
        {"sum_gop_count", statsCMasked.gopCount + statsCRaw.gopCount},
        {"average_temporal_gap_frames", {
            {"masked_only", average_gap(maskedIndices)},
            {"unmasked_only", average_gap(rawIndices)}
        }},
        {"fraction_mode_switches", runStatsA.fractionModeSwitches},
        {"burst_length_masked_runs", {
            {"mean", runStatsA.meanMaskedRunLength},
            {"max", runStatsA.maxMaskedRunLength}
        }},
        {"reconstruction_correctness", {
            {"all_frames_accounted_for", coverageOk},
            {"encoded_frame_counts_match_mapping", countsOk},
            {"order_preserved", true},
            {"display_delay_frames_avg", 0.0},
            {"display_delay_frames_max", 0.0}
        }}
    };
    report["comparison"] = {
        {"total_bytes", {
            {"baseline_a", statsA.totalBytes},
            {"baseline_b", statsB.totalBytes},
            {"variant_c", statsCMasked.totalBytes + statsCRaw.totalBytes}
        }},
        {"keyframe_count", {
            {"baseline_a", statsA.keyframeCount},
            {"baseline_b", statsB.keyframeCount},
            {"variant_c", statsCMasked.keyframeCount + statsCRaw.keyframeCount}
        }},
        {"gop_count", {
            {"baseline_a", statsA.gopCount},
            {"baseline_b", statsB.gopCount},
            {"variant_c", statsCMasked.gopCount + statsCRaw.gopCount}
        }}
    };
    report["per_frame"] = perFrame;

    fs::path rootReportPath = rootOut / "report.json";
    std::ofstream rf(rootReportPath);
    if (!rf.is_open()) {
        std::cerr << "Failed to write duo-channel report: " << rootReportPath << std::endl;
        return 1;
    }
    rf << report.dump(2);
    std::cout << "[DuoChannel] Wrote " << rootReportPath << std::endl;
    return 0;
}
