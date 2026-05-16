#include "offline_processor.h"
#include <fstream>
#include <cstdio>
#include <iostream>
#include <openssl/evp.h>
#include <iomanip>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <array>
#include <opencv2/dnn.hpp>
#include "sei_parser.h"
#include <cstdio>
#include <sstream>
#include <csignal>
#include <climits>

namespace fs = std::filesystem;
using json = nlohmann::json;

static json ffprobe_input_summary_json(const std::string &path) {
    // Best-effort: run ffprobe and parse JSON. If ffprobe is missing or parsing fails,
    // return an empty object.
    // We keep the full ffprobe JSON so later analysis can decide which fields matter.
    std::string cmd = "ffprobe -v error -print_format json -show_format -show_streams \"" + path + "\"";
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) return json::object();
    std::string out;
    char buf[8192];
    while (true) {
        size_t n = fread(buf, 1, sizeof(buf), p);
        if (n > 0) out.append(buf, buf + n);
        if (n < sizeof(buf)) break;
    }
    (void)pclose(p);
    try {
        return json::parse(out);
    } catch (...) {
        return json::object();
    }
}

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
};

static RunLengthStats compute_run_length_stats(const std::vector<int> &flags) {
    RunLengthStats stats;
    stats.totalFrames = (int)flags.size();
    if (flags.empty()) return stats;
    for (int v : flags) {
        if (v != 0) stats.maskedFrames++;
        else stats.rawFrames++;
    }
    std::vector<int> allRuns;
    std::vector<int> maskedRuns;
    std::vector<int> rawRuns;
    int cur = flags.front();
    int len = 1;
    for (size_t i = 1; i < flags.size(); ++i) {
        if (flags[i] == flags[i - 1]) {
            len++;
            continue;
        }
        allRuns.push_back(len);
        if (cur != 0) {
            maskedRuns.push_back(len);
            stats.maxMaskedRunLength = std::max(stats.maxMaskedRunLength, len);
        } else {
            rawRuns.push_back(len);
            stats.maxRawRunLength = std::max(stats.maxRawRunLength, len);
        }
        stats.switchCount++;
        cur = flags[i];
        len = 1;
    }
    allRuns.push_back(len);
    if (cur != 0) {
        maskedRuns.push_back(len);
        stats.maxMaskedRunLength = std::max(stats.maxMaskedRunLength, len);
    } else {
        rawRuns.push_back(len);
        stats.maxRawRunLength = std::max(stats.maxRawRunLength, len);
    }
    auto mean_of = [](const std::vector<int> &vals) -> double {
        if (vals.empty()) return 0.0;
        double sum = 0.0;
        for (int v : vals) sum += (double)v;
        return sum / (double)vals.size();
    };
    stats.meanRunLength = mean_of(allRuns);
    stats.meanMaskedRunLength = mean_of(maskedRuns);
    stats.meanRawRunLength = mean_of(rawRuns);
    if (flags.size() > 1) {
        stats.fractionModeSwitches = (double)stats.switchCount / (double)(flags.size() - 1);
    }
    return stats;
}

// -----------------------------------------------------------------------------
// Multi-template matching \"detector\" for pixel-style games (pixelMode)
// -----------------------------------------------------------------------------
//
// We implement a lightweight multi-scale template matcher that:
//  - loads all templates from a directory (PNG/JPG, etc.),
//  - precomputes Canny edges for each template,
//  - runs matchTemplate for each template at multiple scales,
//  - applies NMS and emits DL_RESULT detections with classId = template index.
//
// Configuration precedence for templates directory:
//  1) OfflineOptions::templatesDir (if non-empty)
//  2) Directory of TM_TEMPLATE_PATH (if set)
//  3) \"./\" (current working directory) as a last resort.
// -----------------------------------------------------------------------------

struct PixelTemplate {
    std::string name;       // filename
    std::string relPath;    // relative to templatesDir, e.g. kind/foo.png
    std::string kindName;   // first path component, e.g. kind
    int kindId{-1};         // stable id per kindName
    cv::Mat bgr;            // original BGR image
    cv::Mat edges;          // Canny edges
};

static std::vector<PixelTemplate> g_pixel_templates;
static bool g_pixel_templates_initialized = false;
static std::vector<double> g_pixel_template_scales; // per-template chosen scale
static std::vector<double> g_pixel_template_calib_max; // per-template best NCC max score on first frame
static bool g_pixel_scales_calibrated = false;
static std::string g_pixel_templates_root_dir;
static std::unordered_map<std::string, int> g_pixel_relpath_to_index;
static std::unordered_map<std::string, int> g_pixel_kind_to_id;
static std::vector<cv::Vec3b> g_pixel_kind_id_to_color_bgr;
static bool g_pixel_kind_colors_loaded = false;

static std::string get_env_template_path() {
    const char *env = std::getenv("TM_TEMPLATE_PATH");
    if (env && *env) {
        return std::string(env);
    }
    return std::string();
}

static std::string choose_templates_dir(const OfflineOptions &opt) {
    // Highest priority: explicit CLI / config
    if (!opt.templatesDir.empty()) {
        return opt.templatesDir;
    }
    // Next: directory of TM_TEMPLATE_PATH (for backward compatibility)
    std::string envPath = get_env_template_path();
    if (!envPath.empty()) {
        try {
            fs::path p(envPath);
            if (p.has_parent_path()) {
                return p.parent_path().string();
            }
        } catch (...) {
            // fall through
        }
    }
    // Default for pixel-game experiments in this project
    return std::string("/home/tiehangz/proj/datasets/pixel/templates_rescale/day");
}

static void load_kind_colors_once(const fs::path &templatesDirPath,
                                 const std::vector<std::string> &kindNamesSorted) {
    if (g_pixel_kind_colors_loaded) return;
    g_pixel_kind_colors_loaded = true;

    // Assign deterministic kind_id based on sorted kind names
    g_pixel_kind_to_id.clear();
    for (size_t i = 0; i < kindNamesSorted.size(); ++i) {
        g_pixel_kind_to_id[kindNamesSorted[i]] = static_cast<int>(i);
    }
    g_pixel_kind_id_to_color_bgr.assign(kindNamesSorted.size(), cv::Vec3b(0, 255, 0)); // fallback green

    // Load explicit RGB colors: { "coin": [255,255,0], ... }
    fs::path colorPath = templatesDirPath / "kind_colors.json";
    if (!fs::exists(colorPath)) {
        std::cerr << "[PixelTemplates] kind_colors.json not found at " << colorPath
                  << " (fallback: green for all kinds)" << std::endl;
        return;
    }

    try {
        std::ifstream f(colorPath);
        json j; f >> j;
        if (!j.is_object()) {
            std::cerr << "[PixelTemplates] kind_colors.json is not a JSON object (fallback colors used)\n";
            return;
        }
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string kind = it.key();
            if (!it.value().is_array() || it.value().size() != 3) continue;
            int r = it.value()[0].get<int>();
            int g = it.value()[1].get<int>();
            int b = it.value()[2].get<int>();
            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));
            auto idIt = g_pixel_kind_to_id.find(kind);
            if (idIt == g_pixel_kind_to_id.end()) continue;
            int kid = idIt->second;
            if (kid >= 0 && kid < static_cast<int>(g_pixel_kind_id_to_color_bgr.size())) {
                g_pixel_kind_id_to_color_bgr[kid] = cv::Vec3b((uchar)b, (uchar)g, (uchar)r); // RGB -> BGR
            }
        }
        std::cout << "[PixelTemplates] Loaded kind colors from " << colorPath << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[PixelTemplates] Failed to load kind colors: " << e.what()
                  << " (fallback colors used)\n";
    }
}

static bool load_pixel_templates_once(const OfflineOptions &opt) {
    if (g_pixel_templates_initialized) {
        return !g_pixel_templates.empty();
    }

    g_pixel_templates_initialized = true;
    g_pixel_templates.clear();
    g_pixel_relpath_to_index.clear();
    g_pixel_kind_to_id.clear();
    g_pixel_kind_id_to_color_bgr.clear();
    g_pixel_kind_colors_loaded = false;
    g_pixel_scales_calibrated = false;
    g_pixel_template_scales.clear();
    g_pixel_template_calib_max.clear();

    std::string dir = choose_templates_dir(opt);
    fs::path dirPath(dir);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "[PixelTemplates] Invalid templatesDir: " << dir << std::endl;
        return false;
    }
    g_pixel_templates_root_dir = dirPath.string();

    const std::vector<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp"};
    std::vector<fs::path> files;
    for (auto &entry : fs::recursive_directory_iterator(
             dirPath,
             fs::directory_options::follow_directory_symlink | fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        fs::path p = entry.path();
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
        files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    if (opt.pixelMaxTemplates > 0 && static_cast<int>(files.size()) > opt.pixelMaxTemplates) {
        std::cout << "[PixelTemplates] Limiting template load from " << files.size()
                  << " to " << opt.pixelMaxTemplates << " files for memory control." << std::endl;
        files.resize(static_cast<size_t>(opt.pixelMaxTemplates));
    }

    // Discover kinds from subfolders (first path component of relative path)
    std::set<std::string> kindSet;
    struct Pending {
        fs::path absPath;
        std::string relPath;
        std::string name;
        std::string kind;
    };
    std::vector<Pending> pending;
    pending.reserve(files.size());

    for (const auto &p : files) {
        // NOTE: std::filesystem::relative may resolve symlinks (breaking keys when templatesDir uses symlinks).
        // We want a stable "key-based" relative path within templatesDir, so use lexical relative instead.
        fs::path rel = p.lexically_relative(dirPath);
        if (rel.empty()) {
            // Fallback: best-effort key (should be rare)
            rel = fs::path("root") / p.filename();
        }
        std::string relStr = rel.generic_string();
        std::string name = p.filename().string();

        std::string kind = "root";
        auto it = rel.begin();
        if (it != rel.end()) {
            kind = it->string();
        }
        kindSet.insert(kind);
        pending.push_back(Pending{p, relStr, name, kind});
    }

    std::vector<std::string> kindNames(kindSet.begin(), kindSet.end());
    std::sort(kindNames.begin(), kindNames.end());
    load_kind_colors_once(dirPath, kindNames);

    for (const auto &item : pending) {
        const fs::path &p = item.absPath;

        cv::Mat bgr = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) continue;

        cv::Mat gray, edges;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);

        PixelTemplate tmpl;
        tmpl.name = item.name;
        tmpl.relPath = item.relPath;
        tmpl.kindName = item.kind;
        auto kidIt = g_pixel_kind_to_id.find(tmpl.kindName);
        tmpl.kindId = (kidIt != g_pixel_kind_to_id.end()) ? kidIt->second : -1;
        tmpl.bgr = std::move(bgr);
        tmpl.edges = std::move(edges);
        g_pixel_relpath_to_index[tmpl.relPath] = static_cast<int>(g_pixel_templates.size());
        g_pixel_templates.push_back(std::move(tmpl));
    }

    std::cout << "[PixelTemplates] Loaded " << g_pixel_templates.size()
              << " templates from " << dir
              << " (" << g_pixel_kind_to_id.size() << " kinds)" << std::endl;
    return !g_pixel_templates.empty();
}

static cv::Mat get_pixel_template_rgba_for_detection(const DL_RESULT &d) {
    if (d.seiPath.empty()) return cv::Mat();
    auto it = g_pixel_relpath_to_index.find(d.seiPath);
    if (it == g_pixel_relpath_to_index.end()) return cv::Mat();
    int idx = it->second;
    if (idx < 0 || idx >= static_cast<int>(g_pixel_templates.size())) return cv::Mat();
    const auto &tmpl = g_pixel_templates[idx];
    if (tmpl.bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat bgr_rs;
    if (d.box.width > 0 && d.box.height > 0) {
        cv::resize(tmpl.bgr, bgr_rs, d.box.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        bgr_rs = tmpl.bgr;
    }
    cv::Mat rgba;
    cv::cvtColor(bgr_rs, rgba, cv::COLOR_BGR2BGRA);
    return rgba;
}

static cv::Rect expand_pixel_mask_box(const cv::Rect &box, int pad_px, const cv::Size &frame_size) {
    if (frame_size.width <= 0 || frame_size.height <= 0) return cv::Rect();
    const cv::Rect bounds(0, 0, frame_size.width, frame_size.height);
    const cv::Rect clamped = box & bounds;
    if (clamped.area() <= 0) return cv::Rect();
    if (pad_px <= 0) return clamped;

    const int x0 = std::max(0, clamped.x - pad_px);
    const int y0 = std::max(0, clamped.y - pad_px);
    const int x1 = std::min(frame_size.width, clamped.x + clamped.width + pad_px);
    const int y1 = std::min(frame_size.height, clamped.y + clamped.height + pad_px);
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

static cv::Vec3b kind_color_from_kind_id(int kindId) {
    if (kindId >= 0 && kindId < static_cast<int>(g_pixel_kind_id_to_color_bgr.size())) {
        return g_pixel_kind_id_to_color_bgr[kindId];
    }
    return cv::Vec3b(0, 255, 0); // fallback green
}

static cv::Vec3b yolo_class_color(int classId) {
    // Small stable palette (BGR) to encourage temporal consistency.
    static const cv::Vec3b palette[] = {
        {0, 255, 0},     // green
        {0, 255, 255},   // yellow
        {255, 0, 0},     // blue
        {255, 0, 255},   // magenta
        {255, 255, 0},   // cyan
        {0, 128, 255},   // orange-ish
        {128, 0, 255},   // purple-ish
        {255, 128, 0},   // light blue-ish
    };
    int n = (int)(sizeof(palette) / sizeof(palette[0]));
    int idx = (classId < 0) ? 0 : (classId % n);
    return palette[idx];
}

static std::string find_best_phash_match(const std::unordered_map<std::string, DictItem> &dict,
                                        uint64_t phash,
                                        int cls,
                                        const cv::Size &wh,
                                        int phashThreshold) {
    // Simple linear scan; dict sizes here are small enough for offline eval.
    int bestDist = phashThreshold + 1;
    std::string bestHash;
    for (const auto &kv : dict) {
        const DictItem &it = kv.second;
        if (it.cls != cls) continue;
        // Allow small bbox jitter across frames.
        if (std::abs(it.width - wh.width) > 2 || std::abs(it.height - wh.height) > 2) continue;
        int d = OfflineProcessor::hamming64(phash, it.phash64);
        if (d < bestDist) {
            bestDist = d;
            bestHash = it.hash;
            if (bestDist == 0) break;
        }
    }
    return bestHash;
}

static std::string resolve_yolo_matcher(const OfflineOptions &opt) {
    if (!opt.yoloMatcher.empty()) return opt.yoloMatcher;
    return opt.yoloLatentKey ? "latent_key" : "phash";
}

static cv::Mat compute_rgb_hist_feature(const cv::Mat &bgr, const cv::Mat &mask_u8) {
    if (bgr.empty()) return cv::Mat();
    cv::Mat hist;
    int channels[] = {0, 1, 2};
    int hist_size[] = {8, 8, 8};
    float range[] = {0.0f, 256.0f};
    const float *ranges[] = {range, range, range};
    cv::calcHist(&bgr, 1, channels, mask_u8, hist, 3, hist_size, ranges, true, false);
    if (!hist.empty()) {
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    }
    return hist;
}

static std::string find_best_rgb_hist_match(const std::unordered_map<std::string, DictItem> &dict,
                                            const cv::Mat &roi_bgr,
                                            const cv::Mat &roi_mask_u8,
                                            int cls,
                                            const cv::Size &wh,
                                            std::unordered_map<std::string, cv::Mat> &hist_cache,
                                            double min_score = 0.70) {
    cv::Mat query_hist = compute_rgb_hist_feature(roi_bgr, roi_mask_u8);
    if (query_hist.empty()) return {};
    double best_score = min_score;
    std::string best_hash;
    for (const auto &kv : dict) {
        const DictItem &it = kv.second;
        if (it.cls != cls) continue;
        if (std::abs(it.width - wh.width) > 2 || std::abs(it.height - wh.height) > 2) continue;
        cv::Mat cand_hist;
        auto hit = hist_cache.find(it.hash);
        if (hit != hist_cache.end()) {
            cand_hist = hit->second;
        } else {
            cv::Mat tpl = cv::imread(it.path, cv::IMREAD_UNCHANGED);
            if (tpl.empty()) continue;
            cv::Mat tpl_bgr;
            cv::cvtColor(tpl, tpl_bgr, cv::COLOR_BGRA2BGR);
            cv::Mat tpl_mask;
            cv::extractChannel(tpl, tpl_mask, 3);
            cand_hist = compute_rgb_hist_feature(tpl_bgr, tpl_mask);
            if (cand_hist.empty()) continue;
            hist_cache.emplace(it.hash, cand_hist);
        }
        double score = cv::compareHist(query_hist, cand_hist, cv::HISTCMP_CORREL);
        if (score > best_score) {
            best_score = score;
            best_hash = it.hash;
        }
    }
    return best_hash;
}

static float iou_rect(const cv::Rect &a, const cv::Rect &b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int ua = a.area() + b.area() - inter;
    return ua > 0 ? (float)inter / (float)ua : 0.0f;
}

static float center_dist_px(const cv::Rect &a, const cv::Rect &b) {
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

static float area_ratio_delta(const cv::Rect &a, const cv::Rect &b) {
    float aa = (float)std::max(1, a.area());
    float ba = (float)std::max(1, b.area());
    float r = ba / aa;
    // delta from 1.0 (e.g. 1.25 => 0.25, 0.8 => 0.2)
    return std::abs(r - 1.0f);
}

static std::array<float, 32> l2_normalize_32(const std::array<float, 32> &v) {
    double s2 = 0.0;
    for (float x : v) s2 += (double)x * (double)x;
    double inv = (s2 > 1e-12) ? (1.0 / std::sqrt(s2)) : 0.0;
    std::array<float, 32> out{};
    for (size_t i = 0; i < 32; ++i) out[i] = (float)(v[i] * inv);
    return out;
}

static float cosine_sim_32(const std::array<float, 32> &a, const std::array<float, 32> &b) {
    double dot = 0.0;
    for (size_t i = 0; i < 32; ++i) dot += (double)a[i] * (double)b[i];
    return (float)dot;
}

static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static float pixel_template_accept_thr(int ti, const OfflineOptions &opt) {
    if (!opt.pixelAdaptiveThr) return opt.pixelMinScore;
    if (ti < 0 || (size_t)ti >= g_pixel_template_calib_max.size()) return opt.pixelMinScore;
    float calib = (float)g_pixel_template_calib_max[(size_t)ti];
    if (calib <= 0.0f) return opt.pixelMinScore;
    float thr = opt.pixelThrK * calib;
    return clampf(thr, opt.pixelThrLo, opt.pixelThrHi);
}

// Calibrate best scale per template using the first frame's edges.
// We test multiple candidate scales and keep the one with the highest max NCC score.
static void calibrate_template_scales(const cv::Mat &frame_edges, const OfflineOptions &opt) {
    if (g_pixel_scales_calibrated) return;
    if (g_pixel_templates.empty()) return;

    // Finer-grained candidate scales; only used on the first frame.
    const std::vector<double> candidate_scales = {
        0.6, 0.7, 0.8, 0.9,
        1.0,
        1.1, 1.2, 1.3, 1.4
    };

    g_pixel_template_scales.assign(g_pixel_templates.size(), 1.0);
    g_pixel_template_calib_max.assign(g_pixel_templates.size(), -1.0);

    for (size_t ti = 0; ti < g_pixel_templates.size(); ++ti) {
        const auto &tmpl = g_pixel_templates[ti];
        if (tmpl.edges.empty()) continue;

        double best_scale = 1.0;
        double best_score = -1.0;

        // Force fixed scale if requested (prevents half-size boxes due to noisy auto-calibration).
        if (opt.pixelForceScale > 0.0f) {
            double s = (double)opt.pixelForceScale;
            int th = static_cast<int>(tmpl.edges.rows * s);
            int tw = static_cast<int>(tmpl.edges.cols * s);
            if (th >= 8 && tw >= 8 && th < frame_edges.rows && tw < frame_edges.cols) {
                cv::Mat tmpl_rs;
                cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
                cv::Mat res;
                cv::matchTemplate(frame_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
                if (!res.empty()) {
                    double minVal, maxVal;
                    cv::minMaxLoc(res, &minVal, &maxVal, nullptr, nullptr);
                    best_score = maxVal;
                    best_scale = s;
                }
            }
        } else {

            for (double s : candidate_scales) {
                int th = static_cast<int>(tmpl.edges.rows * s);
                int tw = static_cast<int>(tmpl.edges.cols * s);
                if (th < 8 || tw < 8) continue;
                if (th >= frame_edges.rows || tw >= frame_edges.cols) continue;

                cv::Mat tmpl_rs;
                cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th),
                           0, 0, cv::INTER_AREA);

                cv::Mat res;
                cv::matchTemplate(frame_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
                if (res.empty()) continue;

                double minVal, maxVal;
                cv::minMaxLoc(res, &minVal, &maxVal, nullptr, nullptr);
                if (maxVal > best_score) {
                    best_score = maxVal;
                    best_scale = s;
                }
            }
        }

        g_pixel_template_scales[ti] = best_scale;
        g_pixel_template_calib_max[ti] = best_score;
        std::cout << "[PixelTemplates] Calibrated template \"" << tmpl.relPath
                  << "\" best_scale=" << best_scale << " max_score=" << best_score
                  << std::endl;
    }

    g_pixel_scales_calibrated = true;
}

static void template_match_detections(const cv::Mat &frame_bgr,
                                      std::vector<DL_RESULT> &dets,
                                      const OfflineOptions &opt) {
    dets.clear();
    if (frame_bgr.empty()) return;
    if (!load_pixel_templates_once(opt)) return;

    // Preprocess frame: grayscale + Canny edges
    cv::Mat frame_gray, frame_edges;
    cv::cvtColor(frame_bgr, frame_gray, cv::COLOR_BGR2GRAY);
    cv::Canny(frame_gray, frame_edges, 50, 150);

    // On first frame, calibrate per-template best scale using finer-grained candidates.
    if (!g_pixel_scales_calibrated) {
        calibrate_template_scales(frame_edges, opt);
    }
    const float nms_iou = 0.3f;        // IOU for NMS
    const double zscore = 2.5;         // Threshold: mean + zscore * std

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> tmpl_indices;

    for (size_t ti = 0; ti < g_pixel_templates.size(); ++ti) {
        const auto &tmpl = g_pixel_templates[ti];
        if (tmpl.edges.empty()) continue;

        double s = 1.0;
        if (g_pixel_scales_calibrated && ti < g_pixel_template_scales.size()) {
            s = g_pixel_template_scales[ti];
        }

        int th = static_cast<int>(tmpl.edges.rows * s);
        int tw = static_cast<int>(tmpl.edges.cols * s);
        if (th < 8 || tw < 8) continue;
        if (th >= frame_edges.rows || tw >= frame_edges.cols) continue;

        cv::Mat tmpl_rs;
        cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th),
                   0, 0, cv::INTER_AREA);

        cv::Mat res;
        cv::matchTemplate(frame_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
        if (res.empty()) continue;

        // Compute dynamic threshold for this scale: mu + zscore*sd
        cv::Scalar mean, stddev;
        cv::meanStdDev(res, mean, stddev);
        double mu = mean[0];
        double sd = std::max(stddev[0], 1e-6);
        double thr = mu + zscore * sd;
        thr = std::min(thr, 0.99); // Clamp to valid NCC range

        for (int y = 0; y < res.rows; ++y) {
            const float *row = res.ptr<float>(y);
            for (int x = 0; x < res.cols; ++x) {
                float v = row[x];
                if (v >= thr) {
                    boxes.emplace_back(x, y, tw, th);
                    scores.push_back(v);
                    tmpl_indices.push_back(static_cast<int>(ti));
                }
            }
        }
    }

    if (boxes.empty()) {
        return;
    }

    // Non-maximum suppression to prune overlapping boxes
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, /*score_threshold=*/0.0f,
                      /*nms_threshold=*/nms_iou, keep);

    for (int idx : keep) {
        const cv::Rect &b = boxes[idx];
        if (b.width <= 0 || b.height <= 0) continue;

        DL_RESULT d{};
        int tIdx = tmpl_indices[idx];
        if (tIdx < 0 || tIdx >= static_cast<int>(g_pixel_templates.size())) continue;
        const auto &tmpl = g_pixel_templates[tIdx];
        d.classId = tmpl.kindId;         // kind id (stable across frames)
        d.seiPath = tmpl.relPath;        // template key for stitching (relative path)
        d.confidence = scores[idx];
        d.box = b;

        // Create a simple rectangular mask covering the detected box
        d.boxMask = cv::Mat::zeros(frame_bgr.size(), CV_8UC1);
        cv::Rect safeBox = b & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
        if (safeBox.area() > 0) {
            cv::rectangle(d.boxMask, safeBox, cv::Scalar(255), cv::FILLED);
            dets.push_back(std::move(d));
        }
    }
}

// -----------------------------------------------------------------------------
// Pixel tracker: Kalman + ROI template matching with periodic re-bootstrap
// -----------------------------------------------------------------------------
struct PixelTracker {
    bool inited{false};
    int lastBootstrapFrame{-999999};
    int lostCount{0};
    std::vector<int> activeTemplates; // indices into g_pixel_templates
    cv::Rect lastBox{};
    float lastScore{0.0f};

    // Kalman: state [cx, cy, vx, vy], measurement [cx, cy]
    cv::KalmanFilter kf;
    cv::Mat prev_gray;

    // Persisted multi-peak detections between re-bootstrap intervals to avoid
    // frame-to-frame region flicker (which breaks block/motion prediction).
    std::vector<cv::Rect> persisted_peaks;
    std::vector<int> persisted_peak_ti;
    int persisted_peaks_frame{-999999};

    static float color_score_mean_bgr(const cv::Mat &tpl_bgr, const cv::Mat &roi_bgr) {
        if (tpl_bgr.empty() || roi_bgr.empty()) return 0.0f;
        cv::Scalar mt = cv::mean(tpl_bgr);
        cv::Scalar mr = cv::mean(roi_bgr);
        double db = mt[0] - mr[0];
        double dg = mt[1] - mr[1];
        double dr = mt[2] - mr[2];
        double dist = std::sqrt(db * db + dg * dg + dr * dr);
        // convert to [0,1] where 1 = identical mean color
        // sigma tuned for 8-bit BGR space (larger sigma => less penalty)
        double sigma = 80.0;
        return (float)std::exp(-dist / sigma);
    }

    // Row-energy band selection: find top-N peaks in per-row edge density.
    static void select_bands_from_edges(const cv::Mat &edges, int numBands, int minSep,
                                        std::vector<int> &outY, double &row_ms) {
        auto t0 = std::chrono::steady_clock::now();
        outY.clear();
        if (edges.empty()) { row_ms = 0.0; return; }
        numBands = std::max(1, numBands);
        minSep = std::max(1, minSep);

        // Row energy = fraction of edge pixels in that row
        std::vector<float> energy(edges.rows, 0.0f);
        for (int y = 0; y < edges.rows; ++y) {
            const uchar *p = edges.ptr<uchar>(y);
            int cnt = 0;
            for (int x = 0; x < edges.cols; ++x) cnt += (p[x] != 0);
            energy[y] = (float)cnt / (float)std::max(1, edges.cols);
        }

        // Simple peak picking: repeatedly take max, then suppress +-minSep
        std::vector<float> work = energy;
        for (int i = 0; i < numBands; ++i) {
            int bestY = -1;
            float bestV = 0.0f;
            for (int y = 0; y < (int)work.size(); ++y) {
                if (work[y] > bestV) { bestV = work[y]; bestY = y; }
            }
            if (bestY < 0 || bestV <= 1e-6f) break;
            outY.push_back(bestY);
            int y0 = std::max(0, bestY - minSep);
            int y1 = std::min((int)work.size(), bestY + minSep + 1);
            for (int y = y0; y < y1; ++y) work[y] = 0.0f;
        }

        auto t1 = std::chrono::steady_clock::now();
        row_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Windowed band selection: restrict band candidates around a predicted center Y.
    // This avoids picking unrelated strong edges (HUD/text) and keeps masking focused
    // on the tracked sprite/tile rows, improving stability and coverage across the frame.
    static void select_bands_from_edges_window(const cv::Mat &edges,
                                               int centerY, int halfWindow,
                                               int numBands, int minSep,
                                               std::vector<int> &outY, double &row_ms) {
        auto t0 = std::chrono::steady_clock::now();
        outY.clear();
        if (edges.empty()) { row_ms = 0.0; return; }
        numBands = std::max(1, numBands);
        minSep = std::max(1, minSep);
        halfWindow = std::max(1, halfWindow);
        centerY = std::max(0, std::min(edges.rows - 1, centerY));
        int y0 = std::max(0, centerY - halfWindow);
        int y1 = std::min(edges.rows, centerY + halfWindow + 1);
        if (y1 <= y0 + 1) {
            outY.push_back(centerY);
            row_ms = 0.0;
            return;
        }

        // Row energy in window
        std::vector<float> energy((size_t)(y1 - y0), 0.0f);
        for (int y = y0; y < y1; ++y) {
            const uchar *p = edges.ptr<uchar>(y);
            int cnt = 0;
            for (int x = 0; x < edges.cols; ++x) cnt += (p[x] != 0);
            energy[(size_t)(y - y0)] = (float)cnt / (float)std::max(1, edges.cols);
        }

        // Peak picking in window, then add y0 offset
        std::vector<float> work = energy;
        for (int i = 0; i < numBands; ++i) {
            int bestJ = -1;
            float bestV = 0.0f;
            for (int j = 0; j < (int)work.size(); ++j) {
                if (work[j] > bestV) { bestV = work[j]; bestJ = j; }
            }
            if (bestJ < 0 || bestV <= 1e-6f) break;
            outY.push_back(y0 + bestJ);
            int j0 = std::max(0, bestJ - minSep);
            int j1 = std::min((int)work.size(), bestJ + minSep + 1);
            for (int j = j0; j < j1; ++j) work[j] = 0.0f;
        }

        auto t1 = std::chrono::steady_clock::now();
        row_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Sparse optical flow: estimate global shift (dx,dy) between prev_gray and cur_gray.
    // We use the median LK displacement to be robust to outliers.
    static bool estimate_flow_dxdy(const cv::Mat &prev_gray, const cv::Mat &cur_gray,
                                int maxPts, float &outDx, float &outDy, double &flow_ms) {
        auto t0 = std::chrono::steady_clock::now();
        outDy = 0.0f;
        outDx = 0.0f;
        if (prev_gray.empty() || cur_gray.empty()) { flow_ms = 0.0; return false; }
        std::vector<cv::Point2f> p0;
        cv::goodFeaturesToTrack(prev_gray, p0, std::max(10, maxPts), 0.01, 8);
        if (p0.empty()) { flow_ms = 0.0; return false; }
        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, p1, status, err,
                                 cv::Size(21, 21), 3);
        std::vector<float> dxs;
        std::vector<float> dys;
        dxs.reserve(p0.size());
        dys.reserve(p0.size());
        for (size_t i = 0; i < p0.size(); ++i) {
            if (!status[i]) continue;
            dxs.push_back(p1[i].x - p0[i].x);
            dys.push_back(p1[i].y - p0[i].y);
        }
        if (dys.size() < 8) { flow_ms = 0.0; return false; }
        if (dxs.size() < 8) { flow_ms = 0.0; return false; }
        std::nth_element(dys.begin(), dys.begin() + dys.size() / 2, dys.end());
        std::nth_element(dxs.begin(), dxs.begin() + dxs.size() / 2, dxs.end());
        outDx = dxs[dxs.size() / 2];
        outDy = dys[dys.size() / 2];
        auto t1 = std::chrono::steady_clock::now();
        flow_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    void reset_kf(float cx, float cy) {
        kf.init(4, 2, 0, CV_32F);
        kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
            1,0,1,0,
            0,1,0,1,
            0,0,1,0,
            0,0,0,1);
        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(5e-2));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
        kf.statePost.at<float>(0) = cx;
        kf.statePost.at<float>(1) = cy;
        kf.statePost.at<float>(2) = 0.0f;
        kf.statePost.at<float>(3) = 0.0f;
    }

    static cv::Rect clamp_rect(const cv::Rect &r, const cv::Size &sz) {
        return r & cv::Rect(0, 0, sz.width, sz.height);
    }

    // Global bootstrap: scan all templates once and keep top-K.
    bool bootstrap(const cv::Mat &frame_bgr, const cv::Mat &frame_edges, int frameIdx, const OfflineOptions &opt,
                   double &bootstrap_ms, int &scanned) {
        auto t0 = std::chrono::steady_clock::now();
        scanned = 0;
        activeTemplates.clear();

        struct Cand { float score; float edge; float color; int ti; cv::Point pt; cv::Size wh; };
        std::vector<Cand> cands;
        cands.reserve(g_pixel_templates.size());

        for (int ti = 0; ti < (int)g_pixel_templates.size(); ++ti) {
            const auto &tmpl = g_pixel_templates[(size_t)ti];
            if (tmpl.edges.empty()) continue;
            scanned++;

            double s = 1.0;
            if (g_pixel_scales_calibrated && (size_t)ti < g_pixel_template_scales.size()) {
                s = g_pixel_template_scales[(size_t)ti];
            }
            int th = (int)(tmpl.edges.rows * s);
            int tw = (int)(tmpl.edges.cols * s);
            if (th < 8 || tw < 8) continue;
            if (th >= frame_edges.rows || tw >= frame_edges.cols) continue;

            cv::Mat tmpl_rs;
            cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

            cv::Mat res;
            cv::matchTemplate(frame_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
            if (res.empty()) continue;
            double minV, maxV;
            cv::Point minP, maxP;
            cv::minMaxLoc(res, &minV, &maxV, &minP, &maxP);
            cv::Rect patch(maxP.x, maxP.y, tw, th);
            patch = clamp_rect(patch, frame_edges.size());
            float cscore = 0.0f;
            if (patch.width == tw && patch.height == th && !frame_bgr.empty()) {
                cv::Mat tpl_bgr_rs;
                cv::resize(tmpl.bgr, tpl_bgr_rs, cv::Size(tw, th), 0, 0, cv::INTER_NEAREST);
                cv::Mat roi_bgr = frame_bgr(patch);
                cscore = color_score_mean_bgr(tpl_bgr_rs, roi_bgr);
            }
            float edge = (float)maxV;
            float final = 0.7f * edge + 0.3f * cscore;
            cands.push_back(Cand{final, edge, cscore, ti, maxP, cv::Size(tw, th)});
        }

        std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b){ return a.score > b.score; });
        int topK = opt.pixelTopK;
        if (topK <= 0) topK = (int)cands.size();
        topK = std::max(1, topK);
        if ((int)cands.size() > topK) cands.resize((size_t)topK);
        for (auto &c : cands) activeTemplates.push_back(c.ti);

        // Use best candidate as measurement
        if (cands.empty()) {
            auto t1 = std::chrono::steady_clock::now();
            bootstrap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        auto &best = cands[0];
        lastScore = best.score;
        lastBox = clamp_rect(cv::Rect(best.pt.x, best.pt.y, best.wh.width, best.wh.height), frame_edges.size());
        float cx = lastBox.x + lastBox.width * 0.5f;
        float cy = lastBox.y + lastBox.height * 0.5f;
        reset_kf(cx, cy);

        inited = true;
        lastBootstrapFrame = frameIdx;
        lostCount = 0;

        auto t1 = std::chrono::steady_clock::now();
        bootstrap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    bool roi_match(const cv::Mat &frame_bgr, const cv::Mat &frame_edges, int frameIdx, const OfflineOptions &opt,
                   cv::Rect &outBox, float &outScore, int &outTemplateIdx,
                   std::vector<std::pair<int, float>> &outRanked, // (ti, score) sorted desc
                   double &roi_ms, double &kalman_ms, int &scanned) {
        auto t0 = std::chrono::steady_clock::now();
        kalman_ms = 0.0;
        scanned = 0;

        if (!inited || lastBox.area() <= 0) {
            auto t1 = std::chrono::steady_clock::now();
            roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        float pcx = lastBox.x + lastBox.width * 0.5f;
        float pcy = lastBox.y + lastBox.height * 0.5f;
        if (opt.pixelUseKalman) {
            auto t_k0 = std::chrono::steady_clock::now();
            cv::Mat pred = kf.predict();
            pcx = pred.at<float>(0);
            pcy = pred.at<float>(1);
            auto t_k1 = std::chrono::steady_clock::now();
            kalman_ms += std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();
        }
        int pad = std::max(0, opt.pixelRoiPad);
        cv::Rect roi((int)(pcx - lastBox.width * 0.5f) - pad,
                     (int)(pcy - lastBox.height * 0.5f) - pad,
                     lastBox.width + 2 * pad,
                     lastBox.height + 2 * pad);
        roi = clamp_rect(roi, frame_edges.size());
        if (roi.width < 8 || roi.height < 8) {
            auto t1 = std::chrono::steady_clock::now();
            roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }
        cv::Mat roi_edges = frame_edges(roi);

        outRanked.clear();
        struct Cand { float score; int ti; cv::Point pt; cv::Size wh; };
        std::vector<Cand> cands;
        cands.reserve(activeTemplates.size());

        for (int ti : activeTemplates) {
            if (ti < 0 || ti >= (int)g_pixel_templates.size()) continue;
            const auto &tmpl = g_pixel_templates[(size_t)ti];
            if (tmpl.edges.empty()) continue;
            scanned++;

            double s = 1.0;
            if (g_pixel_scales_calibrated && (size_t)ti < g_pixel_template_scales.size()) {
                s = g_pixel_template_scales[(size_t)ti];
            }
            int th = (int)(tmpl.edges.rows * s);
            int tw = (int)(tmpl.edges.cols * s);
            if (th < 8 || tw < 8) continue;
            if (th >= roi_edges.rows || tw >= roi_edges.cols) continue;

            cv::Mat tmpl_rs;
            cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
            cv::Mat res;
            cv::matchTemplate(roi_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
            if (res.empty()) continue;
            double minV, maxV;
            cv::Point minP, maxP;
            cv::minMaxLoc(res, &minV, &maxV, &minP, &maxP);
            // Color check at best location inside ROI
            cv::Rect patch(roi.x + maxP.x, roi.y + maxP.y, tw, th);
            patch = clamp_rect(patch, frame_edges.size());
            float cscore = 0.0f;
            if (patch.width == tw && patch.height == th && !frame_bgr.empty()) {
                cv::Mat tpl_bgr_rs;
                cv::resize(tmpl.bgr, tpl_bgr_rs, cv::Size(tw, th), 0, 0, cv::INTER_NEAREST);
                cv::Mat roi_bgr = frame_bgr(patch);
                cscore = color_score_mean_bgr(tpl_bgr_rs, roi_bgr);
            }
            float edge = (float)maxV;
            float final = 0.7f * edge + 0.3f * cscore;
            cands.push_back(Cand{final, ti, maxP, cv::Size(tw, th)});
        }

        if (cands.empty()) {
            auto t1 = std::chrono::steady_clock::now();
            roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b){ return a.score > b.score; });
        for (const auto &c : cands) outRanked.push_back({c.ti, c.score});

        const auto &best = cands[0];
        outScore = best.score;
        outTemplateIdx = best.ti;

        cv::Rect b(roi.x + best.pt.x, roi.y + best.pt.y, best.wh.width, best.wh.height);
        b = clamp_rect(b, frame_edges.size());
        outBox = b;

        if (opt.pixelUseKalman) {
            auto t_k2 = std::chrono::steady_clock::now();
            cv::Mat meas(2, 1, CV_32F);
            meas.at<float>(0) = b.x + b.width * 0.5f;
            meas.at<float>(1) = b.y + b.height * 0.5f;
            kf.correct(meas);
            auto t_k3 = std::chrono::steady_clock::now();
            kalman_ms += std::chrono::duration<double, std::milli>(t_k3 - t_k2).count();
        }

        lastBox = b;
        lastScore = best.score;

        auto t1 = std::chrono::steady_clock::now();
        roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    // Multi-peak scan: given precomputed bands and optional dy shift, scan for one template and return many peaks.
    bool band_scan_multi_precomputed(const cv::Mat &frame_bgr, const cv::Mat &frame_edges, int ti,
                         const std::vector<int> &bandYs, float dy,
                         const OfflineOptions &opt,
                         std::vector<std::pair<cv::Rect, float>> &out,
                         double &band_ms, int &scanned) {
        auto t0 = std::chrono::steady_clock::now();
        out.clear();
        scanned = 0;
        if (ti < 0 || ti >= (int)g_pixel_templates.size()) {
            auto t1 = std::chrono::steady_clock::now();
            band_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }
        const auto &tmpl = g_pixel_templates[(size_t)ti];
        if (tmpl.edges.empty()) {
            auto t1 = std::chrono::steady_clock::now();
            band_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        // scale
        double s = 1.0;
        if (g_pixel_scales_calibrated && (size_t)ti < g_pixel_template_scales.size()) {
            s = g_pixel_template_scales[(size_t)ti];
        }
        int th = (int)(tmpl.edges.rows * s);
        int tw = (int)(tmpl.edges.cols * s);
        if (th < 8 || tw < 8 || th >= frame_edges.rows || tw >= frame_edges.cols) {
            auto t1 = std::chrono::steady_clock::now();
            band_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        int padY = std::max(0, opt.pixelBandPadY);
        int maxPeaks = std::max(1, opt.pixelMaxPeaks);
        int nms = std::max(1, opt.pixelPeakNms);
        // For repetitive tiles, suppressing only a few pixels around the peak causes many overlapping
        // detections to cluster (e.g., left side), exhausting maxPeaks and leaving other areas (e.g., right side)
        // unmasked. Ensure the suppression window is at least proportional to the template size so picked peaks
        // correspond to mostly non-overlapping boxes.
        // Use a suppression window close to the template size to strongly discourage overlaps.
        // This helps ensure we cover the whole row (including right side) instead of repeatedly
        // selecting adjacent maxima around the same few tiles.
        int nms_x = std::max(nms, std::max(2, (int)std::round((double)tw * 0.8)));
        int nms_y = std::max(nms, std::max(2, (int)std::round((double)th * 0.8)));
        float thr = pixel_template_accept_thr(ti, opt);

        cv::Mat tmpl_rs;
        cv::resize(tmpl.edges, tmpl_rs, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

        for (int by : bandYs) {
            int cy = (int)std::round((double)by + (double)dy);
            int y0 = cy - th / 2 - padY;
            int y1 = cy + th / 2 + padY;
            y0 = std::max(0, y0);
            y1 = std::min(frame_edges.rows, y1);
            cv::Rect band(0, y0, frame_edges.cols, std::max(0, y1 - y0));
            if (band.height < th + 2) continue;

            cv::Mat band_edges = frame_edges(band);
            cv::Mat res;
            cv::matchTemplate(band_edges, tmpl_rs, res, cv::TM_CCOEFF_NORMED);
            if (res.empty()) continue;
            scanned += 1;

            cv::Mat work = res.clone();
            for (int i = 0; i < maxPeaks; ++i) {
                double minV, maxV;
                cv::Point minP, maxP;
                cv::minMaxLoc(work, &minV, &maxV, &minP, &maxP);
                float edge = (float)maxV;
                if (edge < thr * 0.5f) break;

                cv::Rect patch(band.x + maxP.x, band.y + maxP.y, tw, th);
                patch = clamp_rect(patch, frame_edges.size());
                if (patch.width == tw && patch.height == th) {
                    float cscore = 0.0f;
                    if (!frame_bgr.empty()) {
                        cv::Mat tpl_bgr_rs;
                        cv::resize(tmpl.bgr, tpl_bgr_rs, cv::Size(tw, th), 0, 0, cv::INTER_NEAREST);
                        cscore = color_score_mean_bgr(tpl_bgr_rs, frame_bgr(patch));
                    }
                    float final = 0.7f * edge + 0.3f * cscore;
                    if (final >= thr) out.push_back({patch, final});
                }

                int sx0 = std::max(0, maxP.x - nms_x);
                int sy0 = std::max(0, maxP.y - nms_y);
                int sx1 = std::min(work.cols, maxP.x + nms_x + 1);
                int sy1 = std::min(work.rows, maxP.y + nms_y + 1);
                work(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0)).setTo(0.0f);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        band_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return !out.empty();
    }
};


OfflineProcessor::OfflineProcessor(const std::string& outDir, const std::string& dictDir) 
    : outDir(outDir), dictDir(dictDir) {
    fs::create_directories(outDir);
    // Only create dictionary directory if a non-empty path is provided.
    if (!dictDir.empty()) {
        fs::create_directories(dictDir);
    }
 
    // Create debug directory
    debugDir = fs::path(outDir) / "debug_masks";
    fs::create_directories(debugDir);
    std::srand(std::time(nullptr));
}

bool OfflineProcessor::loadDictionary() {
    fs::path index_path = fs::path(dictDir) / "index.json";
    std::ifstream f(index_path);
    if (!f.is_open()) return false;
    json j; f >> j;
    if (!j.contains("items")) return false;
    for (const auto &entry : j["items"]) {
        DictItem it;
        it.hash = entry.value("hash", "");
        it.path = entry.value("path", "");
        it.width = entry.value("width", 0);
        it.height = entry.value("height", 0);
        it.cls = entry.value("class", 0);
        it.file_size = entry.value("file_size", 0ULL);
        it.count = entry.value("count", 0ULL);
        it.phash64 = entry.value("phash64", 0ULL);
        if (!it.hash.empty()) dict[it.hash] = it;
    }
    return true;
}

void OfflineProcessor::saveDictionary() {
    json j;
    j["items"] = json::array();
    for (const auto &kv : dict) {
        const auto &it = kv.second;
        j["items"].push_back({
            {"hash", it.hash},
            {"path", it.path},
            {"width", it.width},
            {"height", it.height},
            {"class", it.cls},
            {"file_size", it.file_size},
            {"count", it.count},
            {"phash64", it.phash64}
        });
    }
    fs::path index_path = fs::path(dictDir) / "index.json";
    std::ofstream f(index_path);
    f << j.dump(2);
}

bool OfflineProcessor::addToDictionary(const std::string& hash, const cv::Mat& mask, const cv::Rect& box, int cls, const cv::Mat& frame) {
    // Randomly save 20 debug images
    if (debugSavedCount < 20) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        // Increase probability to ensure we capture frames. Since we run once, 
        // and detection might be sparse or frequent, let's just save first 20 valid ones to be safe, 
        // or sample. Let's sample with 10% probability.
        if (r < 0.1) { 
            std::string fname = "debug_" + std::to_string(debugSavedCount) + "_" + hash.substr(0,8) + ".png";
            fs::path debugPath = fs::path(debugDir) / fname;
            
            // Extract frame ROI for the box
            cv::Rect safeBox = box & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safeBox.area() > 0) {
                cv::Mat roi = frame(safeBox).clone();
                
                // The mask passed here is already cropped to the box size (by inference.cpp logic)
                // OR is it? Let's check inference.cpp. 
                // In inference.cpp: result.boxMask = finalMask; where finalMask is full image size but zeroed outside box.
                // Wait, in my update to inference.cpp:
                // result.boxMask = result.boxMask > 0.5;
                // result.boxMask = finalMask; (Full image size)
                
                // So 'mask' here is FULL IMAGE SIZE.
                // We need to crop the mask to the box to overlay on ROI.
                
                cv::Mat maskROI = mask(safeBox).clone();
                
                // Overlay Red on ROI where mask is active
                for(int y=0; y<roi.rows; y++) {
                    for(int x=0; x<roi.cols; x++) {
                        if(maskROI.at<uchar>(y, x) > 0) {
                            // Blend red
                            cv::Vec3b& pixel = roi.at<cv::Vec3b>(y, x);
                            pixel[2] = std::min(255, pixel[2] + 100); // Boost Red
                        }
                    }
                }
                
                cv::imwrite(debugPath.string(), roi);
                debugSavedCount++;
            }
        }
    }

    if (dict.find(hash) != dict.end()) {
        dict[hash].count++;
        return false; // Already exists
    }
    
    cv::Mat rgba = extract_object_rgba(frame, mask, box);
    fs::path out_png = fs::path(dictDir) / (hash + ".png");
    cv::imwrite(out_png.string(), rgba);
    
    uint64_t fsize = 0; 
    try { fsize = (uint64_t)fs::file_size(out_png); } catch (...) {}
    
    DictItem item;
    item.hash = hash;
    item.path = out_png.string();
    item.width = rgba.cols;
    item.height = rgba.rows;
    item.cls = cls;
    item.file_size = fsize;
    item.count = 1;
    // Store pHash of the segmented object mask ROI (position-invariant keying).
    cv::Rect safeBox = box & cv::Rect(0, 0, mask.cols, mask.rows);
    if (safeBox.area() > 0) {
        try {
            cv::Mat maskROI = mask(safeBox);
            item.phash64 = mask_phash64(maskROI);
        } catch (...) {
            item.phash64 = mask_phash64(mask);
        }
    } else {
        item.phash64 = mask_phash64(mask);
    }
    
    dict[hash] = item;
    return true; // New item added
}

std::string OfflineProcessor::sha256_norm(const cv::Mat &mat) {
    if (mat.empty()) return "";
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return "";
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }
    // dimensions and aspect ratio for normalization
    std::string dims = std::to_string(mat.cols) + "x" + std::to_string(mat.rows);
    EVP_DigestUpdate(ctx, dims.data(), dims.size());
    double aspect = (mat.rows>0) ? static_cast<double>(mat.cols) / static_cast<double>(mat.rows) : 0.0;
    std::string aspectStr = std::to_string(aspect);
    EVP_DigestUpdate(ctx, aspectStr.data(), aspectStr.size());

    if (mat.isContinuous()) {
        EVP_DigestUpdate(ctx, mat.data, static_cast<size_t>(mat.total()*mat.elemSize()));
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            EVP_DigestUpdate(ctx, mat.ptr(i), static_cast<size_t>(mat.cols*mat.elemSize()));
        }
    }
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }
    EVP_MD_CTX_free(ctx);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < hash_len; ++i) {
        oss << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

uint64_t OfflineProcessor::mask_phash64(const cv::Mat &mask) {
    if (mask.empty()) return 0ULL;
    cv::Mat gray;
    if (mask.type() == CV_8UC1) {
        gray = mask;
    } else if (mask.channels() == 3) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else {
        mask.convertTo(gray, CV_8U);
    }
    cv::Mat norm; cv::resize(gray, norm, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(norm, norm, cv::Size(3,3), 0.8, 0.8);
    cv::Mat small; cv::resize(norm, small, cv::Size(8,8), 0, 0, cv::INTER_AREA);
    cv::Scalar m = cv::mean(small);
    double meanv = m[0];
    uint64_t bits = 0ULL;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t v = small.at<uint8_t>(y, x);
            bits = (bits << 1) | (uint64_t)(v >= meanv);
        }
    }
    return bits;
}

int OfflineProcessor::hamming64(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

cv::Vec3b OfflineProcessor::deterministic_color_from_hash(const std::string &hash) {
    uint32_t h = 0;
    for (size_t i = 0; i < std::min<size_t>(8, hash.size()); ++i) {
        char c = hash[i];
        uint8_t v = (c >= '0' && c <= '9') ? (c - '0')
                    : (c >= 'a' && c <= 'f') ? (10 + c - 'a')
                    : (c >= 'A' && c <= 'F') ? (10 + c - 'A') : 0;
        h = (h << 4) ^ v;
    }
    uchar b = static_cast<uchar>((h      ) & 0xFF);
    uchar g = static_cast<uchar>((h >>  8) & 0xFF);
    uchar r = static_cast<uchar>((h >> 16) & 0xFF);
    auto bump = [](uchar c){ return static_cast<uchar>(std::max<int>(64, c)); };
    return cv::Vec3b(bump(b), bump(g), bump(r));
}

void OfflineProcessor::paint_mask_color(cv::Mat &dst_bgr, const cv::Mat &mask, const cv::Rect &bbox, const cv::Vec3b &color, float alpha) {
    if (dst_bgr.empty() || mask.empty()) return;
    
    int left = bbox.x; int top = bbox.y;
    
    // Mask is full size, so we can just iterate over the bounding box area to save time
    int right = std::min(left + bbox.width, dst_bgr.cols);
    int bottom = std::min(top + bbox.height, dst_bgr.rows);
    
    // Ensure bounds
    left = std::max(0, left);
    top = std::max(0, top);
    
    // Clamp alpha for safety.
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    const bool solid = (alpha >= 0.999f);

    for (int y = top; y < bottom; ++y) {
        const uchar *mrow = mask.ptr<uchar>(y);
        cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
        for (int x = left; x < right; ++x) {
            if (mrow[x] == 0) continue;
            cv::Vec3b &dpx = drow[x];
            if (solid) {
                // Solid fill: overwrite pixel with constant color
                dpx[0] = color[0];
                dpx[1] = color[1];
                dpx[2] = color[2];
            } else {
                // Alpha blend: d = (1-a)*d + a*c
                for (int k = 0; k < 3; ++k) {
                    float v = (1.0f - alpha) * (float)dpx[k] + alpha * (float)color[k];
                    int iv = (int)std::lround(v);
                    dpx[k] = (uchar)std::max(0, std::min(255, iv));
                }
            }
        }
    }
}

cv::Mat OfflineProcessor::extract_object_rgba(const cv::Mat &frame_bgr, const cv::Mat &mask, const cv::Rect &bbox) {
    // Returns a crop of the object in RGBA
    cv::Mat rgba(bbox.size(), CV_8UC4, cv::Scalar(0,0,0,0));
    
    int left = bbox.x; int top = bbox.y;
    
    // Iterate over the bbox area in the mask (which is full size)
    for (int y = 0; y < bbox.height; ++y) {
        int fy = top + y;
        if (fy < 0 || fy >= frame_bgr.rows) continue;
        
        const uchar* mrow = mask.ptr<uchar>(fy);
        const cv::Vec3b* frow = frame_bgr.ptr<cv::Vec3b>(fy);
        cv::Vec4b* rrow = rgba.ptr<cv::Vec4b>(y);
        
        for (int x = 0; x < bbox.width; ++x) {
            int fx = left + x;
            if (fx < 0 || fx >= frame_bgr.cols) continue;
            
            if (mrow[fx] != 0) {
                const cv::Vec3b& bgr = frow[fx];
                cv::Vec4b& px = rrow[x];
                px[0] = bgr[0];
                px[1] = bgr[1];
                px[2] = bgr[2];
                px[3] = 255;
            }
        }
    }
    return rgba;
}

// --- Helper: overlay RGBA template back onto BGR frame (client-style recovery) ----

static void overlay_template_rgba(cv::Mat &dst_bgr, const cv::Mat &rgba, const cv::Rect &bbox) {
    if (dst_bgr.empty() || rgba.empty()) return;
    CV_Assert(dst_bgr.type() == CV_8UC3);
    CV_Assert(rgba.type() == CV_8UC4);

    int left = bbox.x;
    int top = bbox.y;
    for (int y = 0; y < rgba.rows; ++y) {
        int dy = top + y;
        if (dy < 0 || dy >= dst_bgr.rows) continue;
        const cv::Vec4b* srcRow = rgba.ptr<cv::Vec4b>(y);
        cv::Vec3b* dstRow = dst_bgr.ptr<cv::Vec3b>(dy);
        for (int x = 0; x < rgba.cols; ++x) {
            int dx = left + x;
            if (dx < 0 || dx >= dst_bgr.cols) continue;
            const cv::Vec4b& sp = srcRow[x];
            uchar a = sp[3];
            if (!a) continue;
            cv::Vec3b& dp = dstRow[dx];
            float alpha = a / 255.0f;
            // Alpha blend template over current pixel
            for (int c = 0; c < 3; ++c) {
                dp[c] = static_cast<uchar>(sp[c] * alpha + dp[c] * (1.0f - alpha));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// YOLO reliability: template-mask agreement check
// -----------------------------------------------------------------------------
// If we paint pixels according to the current segmentation mask, the client can
// only heal them perfectly if the template alpha (after resize) matches that
// mask inside the bbox. Otherwise we see residual mask-color "edges" (or
// overwrite unmasked background pixels).
//
// Returns true if template alpha is a near-perfect match for current mask.
static bool template_alpha_matches_mask(
    const cv::Mat &tpl_rgba_resized,
    const cv::Mat &full_mask_u8,
    const cv::Rect &bbox,
    float iou_thr,
    float extra_thr
) {
    if (tpl_rgba_resized.empty() || full_mask_u8.empty()) return false;
    if (tpl_rgba_resized.type() != CV_8UC4) return false;

    cv::Rect safe = bbox & cv::Rect(0, 0, full_mask_u8.cols, full_mask_u8.rows);
    if (safe.area() <= 0) return false;
    if (tpl_rgba_resized.cols != bbox.width || tpl_rgba_resized.rows != bbox.height) return false;

    uint64_t inter = 0, uni = 0, extra = 0, miss = 0;

    // Iterate only on visible part; map dst coords to template coords.
    int dx0 = safe.x - bbox.x;
    int dy0 = safe.y - bbox.y;

    for (int y = 0; y < safe.height; ++y) {
        int my = safe.y + y;
        const uchar *mrow = full_mask_u8.ptr<uchar>(my);
        const cv::Vec4b *trow = tpl_rgba_resized.ptr<cv::Vec4b>(dy0 + y);
        for (int x = 0; x < safe.width; ++x) {
            int mx = safe.x + x;
            bool m = (mrow[mx] != 0);
            bool t = (trow[dx0 + x][3] != 0);
            if (m && t) inter++;
            if (m || t) uni++;
            if (t && !m) extra++;
            if (m && !t) miss++;
        }
    }

    if (uni == 0) return false;
    double iou = (double)inter / (double)uni;
    // extra ratio relative to template alpha area; if template is empty, reject
    double t_area = (double)(inter + extra);
    if (t_area <= 0.0) return false;
    double extra_ratio = (double)extra / t_area;
    // miss ratio relative to mask area; if mask empty, reject
    double m_area = (double)(inter + miss);
    if (m_area <= 0.0) return false;
    double miss_ratio = (double)miss / m_area;

    // In heal-only mode we primarily need to ensure we don't paint outside the current mask.
    // Perfect equality is often unrealistic due to segmentation jitter, so use a softer rule:
    // - template alpha must not spill outside current mask too much (extra_ratio)
    // - optional IoU lower bound if caller wants it
    // - miss_ratio is not critical if we paint using template alpha (not full mask)
    bool ok_extra = (extra_ratio <= extra_thr);
    bool ok_iou = (iou_thr <= 0.0f) ? true : (iou >= iou_thr);
    (void)miss_ratio;
    return ok_extra && ok_iou;
}

// Paint a bbox region with a solid color wherever the RGBA template has alpha>0.
static void paint_bbox_alpha_color(cv::Mat &dst_bgr, const cv::Mat &rgba, const cv::Rect &bbox, const cv::Vec3b &color) {
    if (dst_bgr.empty() || rgba.empty()) return;
    if (dst_bgr.type() != CV_8UC3 || rgba.type() != CV_8UC4) return;
    if (rgba.cols != bbox.width || rgba.rows != bbox.height) return;
    cv::Rect safe = bbox & cv::Rect(0, 0, dst_bgr.cols, dst_bgr.rows);
    if (safe.area() <= 0) return;
    int dx0 = safe.x - bbox.x;
    int dy0 = safe.y - bbox.y;
    for (int y = 0; y < safe.height; ++y) {
        int dy = safe.y + y;
        const cv::Vec4b *srow = rgba.ptr<cv::Vec4b>(dy0 + y);
        cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(dy);
        for (int x = 0; x < safe.width; ++x) {
            if (srow[dx0 + x][3] == 0) continue;
            cv::Vec3b &dp = drow[safe.x + x];
            dp = color;
        }
    }
}

// Find a small integer offset (dx,dy) that best aligns template alpha to the current mask within bbox.
// We optimize for minimal alpha spill outside mask (extra_ratio), and maximal intersection.
static bool best_alpha_mask_offset(
    const cv::Mat &tpl_rgba_resized,
    const cv::Mat &full_mask_u8,
    const cv::Rect &bbox,
    int max_shift,
    int &best_dx,
    int &best_dy,
    double &best_extra_ratio_out
) {
    best_dx = 0; best_dy = 0; best_extra_ratio_out = 1e9;
    if (tpl_rgba_resized.empty() || full_mask_u8.empty()) return false;
    if (tpl_rgba_resized.type() != CV_8UC4) return false;
    if (tpl_rgba_resized.cols != bbox.width || tpl_rgba_resized.rows != bbox.height) return false;
    cv::Rect safe = bbox & cv::Rect(0, 0, full_mask_u8.cols, full_mask_u8.rows);
    if (safe.area() <= 0) return false;

    // Evaluate shifts
    uint64_t best_inter = 0;
    for (int dy = -max_shift; dy <= max_shift; ++dy) {
        for (int dx = -max_shift; dx <= max_shift; ++dx) {
            uint64_t inter = 0, extra = 0;
            // Iterate on visible part; map mask pixel -> template pixel with shift
            for (int y = 0; y < safe.height; ++y) {
                int my = safe.y + y;
                const uchar *mrow = full_mask_u8.ptr<uchar>(my);
                int ty = (my - bbox.y) - dy;
                if (ty < 0 || ty >= tpl_rgba_resized.rows) continue;
                const cv::Vec4b *trow = tpl_rgba_resized.ptr<cv::Vec4b>(ty);
                for (int x = 0; x < safe.width; ++x) {
                    int mx = safe.x + x;
                    int tx = (mx - bbox.x) - dx;
                    if (tx < 0 || tx >= tpl_rgba_resized.cols) continue;
                    bool m = (mrow[mx] != 0);
                    bool t = (trow[tx][3] != 0);
                    if (!t) continue;
                    if (m) inter++;
                    else extra++;
                }
            }
            double t_area = (double)(inter + extra);
            if (t_area <= 1.0) continue;
            double extra_ratio = (double)extra / t_area;
            // Primary objective: minimize spill, secondary: maximize intersection
            if (extra_ratio < best_extra_ratio_out - 1e-9 ||
                (std::abs(extra_ratio - best_extra_ratio_out) <= 1e-9 && inter > best_inter)) {
                best_extra_ratio_out = extra_ratio;
                best_inter = inter;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    return best_extra_ratio_out < 1e9;
}

static uint64_t count_alpha_nonzero_in_safe(const cv::Mat &rgba, const cv::Rect &bbox, const cv::Rect &safe) {
    if (rgba.empty() || rgba.type() != CV_8UC4) return 0;
    if (safe.area() <= 0) return 0;
    int ax0 = safe.x - bbox.x;
    int ay0 = safe.y - bbox.y;
    if (ax0 < 0 || ay0 < 0) return 0;
    if (ax0 + safe.width > rgba.cols) return 0;
    if (ay0 + safe.height > rgba.rows) return 0;
    cv::Mat a;
    cv::extractChannel(rgba, a, 3);
    cv::Mat aroi = a(cv::Rect(ax0, ay0, safe.width, safe.height));
    return (uint64_t)cv::countNonZero(aroi);
}

// Find a small integer offset (dx,dy) that best aligns template RGB (under alpha) to the current frame ROI.
// This is more robust than relying on segmentation mask geometry when the mask jitters or is imperfect.
static bool best_alpha_rgb_offset_mse(
    const cv::Mat &tpl_rgba_resized,
    const cv::Mat &frame_bgr,
    const cv::Rect &bbox,
    int max_shift,
    int &best_dx,
    int &best_dy,
    double &best_mse_out
) {
    best_dx = 0; best_dy = 0; best_mse_out = 1e18;
    if (tpl_rgba_resized.empty() || frame_bgr.empty()) return false;
    if (tpl_rgba_resized.type() != CV_8UC4) return false;
    if (frame_bgr.type() != CV_8UC3) return false;
    if (tpl_rgba_resized.cols != bbox.width || tpl_rgba_resized.rows != bbox.height) return false;
    cv::Rect safe = bbox & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
    if (safe.area() <= 0) return false;

    // Evaluate shifts: minimize masked RGB MSE under template alpha.
    //
    // Performance notes:
    // - For large bboxes (e.g., racing HUD regions), an exhaustive search is extremely expensive.
    // - We down-sample by striding, and also cap the shift search for large regions.
    const int64_t area = (int64_t)safe.width * (int64_t)safe.height;
    int stride = 4;
    if (area > 600000) stride = 16;
    else if (area > 250000) stride = 8;
    // Cap shift search aggressively for large regions.
    int shift_cap = max_shift;
    if (area > 250000) shift_cap = std::min(shift_cap, 2);
    else shift_cap = std::min(shift_cap, 4);

    for (int dy = -shift_cap; dy <= shift_cap; ++dy) {
        for (int dx = -shift_cap; dx <= shift_cap; ++dx) {
            double sse = 0.0;
            uint64_t cnt = 0;

            for (int y = 0; y < safe.height; ++y) {
                if ((y % stride) != 0) continue;
                int fy = safe.y + y;
                int ty = (fy - bbox.y) - dy;
                if (ty < 0 || ty >= tpl_rgba_resized.rows) continue;
                const cv::Vec3b *frow = frame_bgr.ptr<cv::Vec3b>(fy);
                const cv::Vec4b *trow = tpl_rgba_resized.ptr<cv::Vec4b>(ty);
                for (int x = 0; x < safe.width; ++x) {
                    if ((x % stride) != 0) continue;
                    int fx = safe.x + x;
                    int tx = (fx - bbox.x) - dx;
                    if (tx < 0 || tx >= tpl_rgba_resized.cols) continue;
                    const cv::Vec4b &tp = trow[tx];
                    if (tp[3] == 0) continue;
                    const cv::Vec3b &fp = frow[fx];
                    for (int c = 0; c < 3; ++c) {
                        double d = (double)tp[c] - (double)fp[c];
                        sse += d * d;
                    }
                    cnt++;
                }
            }
            if (cnt < 32) continue; // too few sampled alpha pixels, reject
            double mse = sse / (double)(cnt * 3);
            if (mse < best_mse_out) {
                best_mse_out = mse;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    return best_mse_out < 1e18;
}

static double template_alpha_mask_coverage(
    const cv::Mat &tpl_rgba_resized,
    const cv::Mat &full_mask_u8,
    const cv::Rect &bbox,
    int dx,
    int dy
) {
    if (tpl_rgba_resized.empty() || full_mask_u8.empty()) return 0.0;
    if (tpl_rgba_resized.type() != CV_8UC4) return 0.0;
    cv::Rect safe = bbox & cv::Rect(0, 0, full_mask_u8.cols, full_mask_u8.rows);
    if (safe.area() <= 0) return 0.0;
    uint64_t alpha_cnt = 0;
    uint64_t inter = 0;
    for (int y = 0; y < safe.height; ++y) {
        int my = safe.y + y;
        int ty = (my - bbox.y) - dy;
        if (ty < 0 || ty >= tpl_rgba_resized.rows) continue;
        const uchar *mrow = full_mask_u8.ptr<uchar>(my);
        const cv::Vec4b *trow = tpl_rgba_resized.ptr<cv::Vec4b>(ty);
        for (int x = 0; x < safe.width; ++x) {
            int mx = safe.x + x;
            int tx = (mx - bbox.x) - dx;
            if (tx < 0 || tx >= tpl_rgba_resized.cols) continue;
            if (trow[tx][3] == 0) continue;
            alpha_cnt++;
            if (mrow[mx] != 0) inter++;
        }
    }
    if (alpha_cnt == 0) return 0.0;
    return (double)inter / (double)alpha_cnt;
}

// --- Quality metrics (PSNR / SSIM) -----------------------------------------

// Compute PSNR between two images
static double computePSNR(const cv::Mat &I1, const cv::Mat &I2) {
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    cv::Scalar s = cv::sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10) return 100.0;
    double mse = sse / (double)(I1.channels() * I1.total());
    double psnr = 10.0 * log10((255 * 255) / mse);
    return psnr;
}

// Compute SSIM between two images
static double computeSSIM(const cv::Mat &i1, const cv::Mat &i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);

    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_I2 = I1.mul(I2);       // I1 * I2

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);     // t1 = ((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);
    return (mssim[0] + mssim[1] + mssim[2]) / 3.0;
}

static cv::Vec3b choose_mask_color_bgr(const OfflineOptions &opt, int classId, const cv::Vec3b &dominant_bgr) {
    if (opt.maskColor == "dominant") return dominant_bgr;
    if (opt.maskColor == "black") return cv::Vec3b(0, 0, 0);
    if (opt.maskColor == "brown") return cv::Vec3b(42, 42, 165); // BGR for RGB(165,42,42)
    if (opt.maskColor == "rgb") return cv::Vec3b((uchar)opt.maskColorB, (uchar)opt.maskColorG, (uchar)opt.maskColorR);
    if (opt.maskColor == "class") return yolo_class_color(classId);
    if (opt.yoloClassConsistentColor) return yolo_class_color(classId);
    return cv::Vec3b(0, 255, 0);
}

struct PixelGridRuntimeGroup {
    PixelGridGroup meta;
    std::vector<cv::Rect> cellBoxes;
};

static cv::Size pixel_template_size_for_path(const std::string &path, const cv::Size &fallback) {
    auto it = g_pixel_relpath_to_index.find(path);
    if (it == g_pixel_relpath_to_index.end()) return fallback;
    int idx = it->second;
    if (idx < 0 || idx >= static_cast<int>(g_pixel_templates.size())) return fallback;
    const auto &tmpl = g_pixel_templates[(size_t)idx];
    if (tmpl.bgr.empty()) return fallback;
    return tmpl.bgr.size();
}

static cv::Mat get_pixel_template_rgba_by_path(const std::string &path) {
    auto it = g_pixel_relpath_to_index.find(path);
    if (it == g_pixel_relpath_to_index.end()) return cv::Mat();
    int idx = it->second;
    if (idx < 0 || idx >= static_cast<int>(g_pixel_templates.size())) return cv::Mat();
    const auto &tmpl = g_pixel_templates[(size_t)idx];
    if (tmpl.bgr.empty()) return cv::Mat();
    cv::Mat rgba;
    cv::cvtColor(tmpl.bgr, rgba, cv::COLOR_BGR2BGRA);
    return rgba;
}

static void paint_pixel_rect_color(cv::Mat &dst_bgr,
                                   cv::Mat &mask_union_u8,
                                   const cv::Rect &rect,
                                   const cv::Vec3b &color,
                                   float alpha) {
    if (dst_bgr.empty()) return;
    cv::Rect safe = rect & cv::Rect(0, 0, dst_bgr.cols, dst_bgr.rows);
    if (safe.area() <= 0) return;
    if (!mask_union_u8.empty()) {
        cv::rectangle(mask_union_u8, safe, cv::Scalar(255), cv::FILLED);
    }
    alpha = std::max(0.0f, std::min(1.0f, alpha));
    const bool solid = (alpha >= 0.999f);
    for (int y = safe.y; y < safe.y + safe.height; ++y) {
        cv::Vec3b *row = dst_bgr.ptr<cv::Vec3b>(y);
        for (int x = safe.x; x < safe.x + safe.width; ++x) {
            cv::Vec3b &px = row[x];
            if (solid) {
                px = color;
            } else {
                for (int c = 0; c < 3; ++c) {
                    float v = (1.0f - alpha) * (float)px[c] + alpha * (float)color[c];
                    px[c] = (uchar)std::max(0, std::min(255, (int)std::lround(v)));
                }
            }
        }
    }
}

static int build_pixel_grid_groups_and_paint(const std::vector<DL_RESULT> &dets,
                                             const OfflineOptions &opt,
                                             const cv::Size &frameSize,
                                             const cv::Vec3b &dominant_bgr,
                                             cv::Mat &processed,
                                             cv::Mat &mask_union_u8,
                                             std::vector<SEIRegion> &legacy_regions,
                                             std::vector<PixelGridGroup> &grid_groups,
                                             std::vector<PixelGridRuntimeGroup> &runtime_groups) {
    struct Snap {
        size_t idx{0};
        int row{0};
        int col{0};
        cv::Rect box;
    };

    std::unordered_map<std::string, std::vector<size_t>> byPath;
    byPath.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto &d = dets[i];
        if (d.confidence < opt.confThreshold || d.box.area() <= 0 || d.seiPath.empty()) continue;
        byPath[d.seiPath].push_back(i);
    }

    std::vector<uint8_t> grouped(dets.size(), 0);
    int refCells = 0;
    const cv::Rect bounds(0, 0, frameSize.width, frameSize.height);
    const int minRun = std::max(1, opt.pixelGridMinRun);
    const int snapTol = std::max(0, opt.pixelGridSnapTolPx);

    for (auto &kv : byPath) {
        const std::string &path = kv.first;
        auto &idxs = kv.second;
        if (idxs.empty()) continue;
        std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
            const cv::Rect &ra = dets[a].box;
            const cv::Rect &rb = dets[b].box;
            if (ra.y != rb.y) return ra.y < rb.y;
            return ra.x < rb.x;
        });

        cv::Size tileSize = pixel_template_size_for_path(path, dets[idxs.front()].box.size());
        if (tileSize.width <= 0 || tileSize.height <= 0 ||
            tileSize.width > 65535 || tileSize.height > 65535) {
            continue;
        }

        std::unordered_map<int, std::vector<Snap>> snapsByRow;
        std::vector<int> rowReps;
        for (size_t idx : idxs) {
            const cv::Rect box = dets[idx].box & bounds;
            if (box.area() <= 0) continue;
            int rowY = box.y;
            bool rowMatched = false;
            for (int &rep : rowReps) {
                if (std::abs(box.y - rep) <= snapTol) {
                    rowY = rep;
                    rowMatched = true;
                    break;
                }
            }
            if (!rowMatched) rowReps.push_back(rowY);
            snapsByRow[rowY].push_back(Snap{idx, 0, 0, cv::Rect(box.x, rowY, tileSize.width, tileSize.height) & bounds});
        }
        if (snapsByRow.empty()) continue;
        std::vector<int> rowKeys = rowReps;
        std::sort(rowKeys.begin(), rowKeys.end());

        int classId = dets[idxs.front()].classId;
        for (int rowY : rowKeys) {
            auto &rowSnaps = snapsByRow[rowY];
            std::sort(rowSnaps.begin(), rowSnaps.end(), [](const Snap &a, const Snap &b) {
                return a.box.x < b.box.x;
            });
            for (size_t i = 0; i < rowSnaps.size();) {
                size_t j = i + 1;
                while (j < rowSnaps.size() &&
                       std::abs(rowSnaps[j].box.x - (rowSnaps[j - 1].box.x + tileSize.width)) <= snapTol) {
                    ++j;
                }
                int len = (int)(j - i);
                if (len >= minRun && len <= 65535) {
                    const int originX = rowSnaps[i].box.x;
                    const int originY = rowY;
                    PixelGridRuntimeGroup runtime{};
                    runtime.meta.id = (uint32_t)grid_groups.size();
                    runtime.meta.origin_x = (uint32_t)std::max(0, originX);
                    runtime.meta.origin_y = (uint32_t)std::max(0, originY);
                    runtime.meta.tile_w = (uint16_t)tileSize.width;
                    runtime.meta.tile_h = (uint16_t)tileSize.height;
                    runtime.meta.step_x = (int16_t)tileSize.width;
                    runtime.meta.step_y = 1;
                    runtime.meta.paint_pad_x = (uint8_t)std::min(255, std::max(0, opt.pixelMaskPadPx));
                    runtime.meta.paint_pad_y = runtime.meta.paint_pad_x;
                    runtime.meta.flags = 1;
                    runtime.meta.class_id = (uint8_t)std::max(0, std::min(255, classId));
                    runtime.meta.path = path;
                    PixelGridRun run{};
                    run.row = 0;
                    run.col0 = 0;
                    run.count = (uint16_t)len;
                    runtime.meta.runs.push_back(run);
                    int x0 = originX;
                    int y0 = originY;
                    cv::Rect paintRect(x0, y0, len * tileSize.width, tileSize.height);
                    paintRect = expand_pixel_mask_box(paintRect, opt.pixelMaskPadPx, frameSize);
                    cv::Vec3b maskColor = kind_color_from_kind_id(classId);
                    if (opt.maskColor == "dominant" || opt.maskColor == "black" || opt.maskColor == "brown" || opt.maskColor == "rgb") {
                        maskColor = choose_mask_color_bgr(opt, classId, dominant_bgr);
                    }
                    paint_pixel_rect_color(processed, mask_union_u8, paintRect, maskColor, opt.paintAlpha);
                    for (int k = 0; k < len; ++k) {
                        grouped[rowSnaps[i + (size_t)k].idx] = 1;
                        runtime.cellBoxes.push_back((cv::Rect(originX + k * tileSize.width,
                                                              originY,
                                                              tileSize.width,
                                                              tileSize.height) & bounds));
                    }
                    refCells += len;
                    grid_groups.push_back(runtime.meta);
                    runtime_groups.push_back(std::move(runtime));
                }
                i = j;
            }
        }
    }

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto &d = dets[i];
        if (d.confidence < opt.confThreshold || d.box.area() <= 0 || d.seiPath.empty()) continue;
        if (i < grouped.size() && grouped[i]) continue;
        cv::Rect box = expand_pixel_mask_box(d.box, opt.pixelMaskPadPx, frameSize);
        cv::Vec3b maskColor = kind_color_from_kind_id(d.classId);
        if (opt.maskColor == "dominant" || opt.maskColor == "black" || opt.maskColor == "brown" || opt.maskColor == "rgb") {
            maskColor = choose_mask_color_bgr(opt, d.classId, dominant_bgr);
        }
        paint_pixel_rect_color(processed, mask_union_u8, box, maskColor, opt.paintAlpha);
        SEIRegion r{};
        r.id = (uint32_t)legacy_regions.size();
        r.x = (uint32_t)std::max(0, box.x);
        r.y = (uint32_t)std::max(0, box.y);
        r.w = (uint32_t)std::max(0, box.width);
        r.h = (uint32_t)std::max(0, box.height);
        r.flags = 1;
        r.class_id = (uint8_t)std::max(0, std::min(255, d.classId));
        r.path = d.seiPath;
        legacy_regions.push_back(std::move(r));
        refCells++;
    }
    return refCells;
}

static cv::Vec3b dominant_color_bgr_hist(const cv::Mat &frame_bgr) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) return cv::Vec3b(0, 0, 0);
    // Downsample for speed.
    cv::Mat small;
    const int target_w = 160;
    int w = frame_bgr.cols;
    int h = frame_bgr.rows;
    if (w <= 0 || h <= 0) return cv::Vec3b(0, 0, 0);
    int tw = std::min(target_w, w);
    int th = std::max(1, (int)((double)h * (double)tw / (double)std::max(1, w)));
    cv::resize(frame_bgr, small, cv::Size(tw, th), 0, 0, cv::INTER_AREA);

    // 4 bits per channel => 16^3 bins = 4096.
    std::array<uint32_t, 4096> hist{};
    hist.fill(0);
    for (int y = 0; y < small.rows; ++y) {
        const cv::Vec3b *row = small.ptr<cv::Vec3b>(y);
        for (int x = 0; x < small.cols; ++x) {
            cv::Vec3b p = row[x];
            int b = p[0] >> 4;
            int g = p[1] >> 4;
            int r = p[2] >> 4;
            int idx = (r << 8) | (g << 4) | b;
            hist[(size_t)idx] += 1;
        }
    }
    uint32_t best = 0;
    int best_idx = 0;
    for (int i = 0; i < 4096; ++i) {
        if (hist[(size_t)i] > best) { best = hist[(size_t)i]; best_idx = i; }
    }
    int b4 = best_idx & 0xF;
    int g4 = (best_idx >> 4) & 0xF;
    int r4 = (best_idx >> 8) & 0xF;
    // Bin center in 8-bit space.
    int b8 = b4 * 16 + 8;
    int g8 = g4 * 16 + 8;
    int r8 = r4 * 16 + 8;
    return cv::Vec3b((uchar)b8, (uchar)g8, (uchar)r8);
}

static cv::Vec3b dominant_color_bgr_hist_local(
    const cv::Mat &frame_bgr,
    const std::vector<DL_RESULT> &dets,
    int pad_px
) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) return cv::Vec3b(0, 0, 0);
    if (dets.empty() || pad_px <= 0) return dominant_color_bgr_hist(frame_bgr);

    std::array<uint32_t, 4096> hist{};
    hist.fill(0);
    uint64_t samples = 0;
    const cv::Rect frame_rect(0, 0, frame_bgr.cols, frame_bgr.rows);

    for (const auto &d : dets) {
        if (d.box.area() <= 0) continue;
        cv::Rect expanded(
            d.box.x - pad_px,
            d.box.y - pad_px,
            d.box.width + 2 * pad_px,
            d.box.height + 2 * pad_px
        );
        expanded &= frame_rect;
        if (expanded.area() <= 0) continue;

        for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
            const cv::Vec3b *row = frame_bgr.ptr<cv::Vec3b>(y);
            const uchar *mask_row = (!d.boxMask.empty() && d.boxMask.type() == CV_8UC1 && d.boxMask.rows == frame_bgr.rows && d.boxMask.cols == frame_bgr.cols)
                ? d.boxMask.ptr<uchar>(y)
                : nullptr;
            for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
                // Sample the neighborhood, not the detected object itself.
                if (mask_row && mask_row[x] != 0) continue;
                cv::Vec3b p = row[x];
                int b = p[0] >> 4;
                int g = p[1] >> 4;
                int r = p[2] >> 4;
                int idx = (r << 8) | (g << 4) | b;
                hist[(size_t)idx] += 1;
                samples++;
            }
        }
    }

    if (samples == 0) return dominant_color_bgr_hist(frame_bgr);
    uint32_t best = 0;
    int best_idx = 0;
    for (int i = 0; i < 4096; ++i) {
        if (hist[(size_t)i] > best) { best = hist[(size_t)i]; best_idx = i; }
    }
    int b4 = best_idx & 0xF;
    int g4 = (best_idx >> 4) & 0xF;
    int r4 = (best_idx >> 8) & 0xF;
    return cv::Vec3b((uchar)(b4 * 16 + 8), (uchar)(g4 * 16 + 8), (uchar)(r4 * 16 + 8));
}

static cv::Vec3b dominant_color_bgr_hist_region(
    const cv::Mat &frame_bgr,
    const DL_RESULT &det,
    int pad_px,
    const cv::Vec3b &fallback_bgr,
    const std::string &stat
) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) return fallback_bgr;
    if (det.box.area() <= 0 || pad_px <= 0) return fallback_bgr;

    std::array<uint32_t, 4096> hist{};
    hist.fill(0);
    std::vector<uchar> bs;
    std::vector<uchar> gs;
    std::vector<uchar> rs;
    bool use_median = (stat == "median");
    bool use_average = (stat == "average");
    uint64_t sum_b = 0, sum_g = 0, sum_r = 0;
    uint64_t samples = 0;
    const cv::Rect frame_rect(0, 0, frame_bgr.cols, frame_bgr.rows);
    cv::Rect expanded(
        det.box.x - pad_px,
        det.box.y - pad_px,
        det.box.width + 2 * pad_px,
        det.box.height + 2 * pad_px
    );
    expanded &= frame_rect;
    if (expanded.area() <= 0) return fallback_bgr;

    for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
        const cv::Vec3b *row = frame_bgr.ptr<cv::Vec3b>(y);
        const uchar *mask_row = (!det.boxMask.empty() && det.boxMask.type() == CV_8UC1 && det.boxMask.rows == frame_bgr.rows && det.boxMask.cols == frame_bgr.cols)
            ? det.boxMask.ptr<uchar>(y)
            : nullptr;
        for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
            // Keep flat fill entropy low, but choose it from this object's immediate environment.
            if (mask_row && mask_row[x] != 0) continue;
            cv::Vec3b p = row[x];
            if (use_median) {
                bs.push_back(p[0]);
                gs.push_back(p[1]);
                rs.push_back(p[2]);
            } else if (use_average) {
                sum_b += p[0];
                sum_g += p[1];
                sum_r += p[2];
            } else {
                int b = p[0] >> 4;
                int g = p[1] >> 4;
                int r = p[2] >> 4;
                int idx = (r << 8) | (g << 4) | b;
                hist[(size_t)idx] += 1;
            }
            samples++;
        }
    }

    if (samples == 0) return fallback_bgr;
    if (use_median) {
        auto median_channel = [](std::vector<uchar> &vals) -> uchar {
            if (vals.empty()) return 0;
            size_t mid = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin() + (long)mid, vals.end());
            return vals[mid];
        };
        return cv::Vec3b(median_channel(bs), median_channel(gs), median_channel(rs));
    }
    if (use_average) {
        auto clamp_u8 = [](int v) -> uchar {
            return (uchar)std::max(0, std::min(255, v));
        };
        return cv::Vec3b(
            clamp_u8((int)std::lround((double)sum_b / (double)samples)),
            clamp_u8((int)std::lround((double)sum_g / (double)samples)),
            clamp_u8((int)std::lround((double)sum_r / (double)samples))
        );
    }

    uint32_t best = 0;
    int best_idx = 0;
    for (int i = 0; i < 4096; ++i) {
        if (hist[(size_t)i] > best) { best = hist[(size_t)i]; best_idx = i; }
    }
    int b4 = best_idx & 0xF;
    int g4 = (best_idx >> 4) & 0xF;
    int r4 = (best_idx >> 8) & 0xF;
    return cv::Vec3b((uchar)(b4 * 16 + 8), (uchar)(g4 * 16 + 8), (uchar)(r4 * 16 + 8));
}

// -----------------------------------------------------------------------------
// Fill + feather compositing helpers (reduce sharp edges from solid paint)
// -----------------------------------------------------------------------------
static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

// Replace pixels under mask within bbox using fill frame, with optional feathering back to original at boundary.
// - mask_full_u8: full-frame CV_8UC1 (0/255 or alpha mask), non-zero means "inside region".
// - fill_bgr: full-frame CV_8UC3 to sample fill pixels from (same size as orig).
static void apply_fill_with_feather_bbox(
    cv::Mat &dst_bgr,
    const cv::Mat &orig_bgr,
    const cv::Mat &fill_bgr,
    const cv::Mat &mask_full_u8,
    const cv::Rect &bbox,
    int feather_px
) {
    if (dst_bgr.empty() || orig_bgr.empty() || fill_bgr.empty() || mask_full_u8.empty()) return;
    if (dst_bgr.type() != CV_8UC3 || orig_bgr.type() != CV_8UC3 || fill_bgr.type() != CV_8UC3) return;
    if (mask_full_u8.type() != CV_8UC1) return;
    if (dst_bgr.size() != orig_bgr.size() || dst_bgr.size() != fill_bgr.size() || dst_bgr.size() != mask_full_u8.size()) return;

    cv::Rect safe = bbox & cv::Rect(0, 0, dst_bgr.cols, dst_bgr.rows);
    if (safe.area() <= 0) return;

    cv::Mat mroi = mask_full_u8(safe);
    if (cv::countNonZero(mroi) == 0) return;

    // No feather: direct copy from fill for masked pixels.
    if (feather_px <= 0) {
        for (int y = safe.y; y < safe.y + safe.height; ++y) {
            const uchar *mrow = mask_full_u8.ptr<uchar>(y);
            const cv::Vec3b *frow = fill_bgr.ptr<cv::Vec3b>(y);
            cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
            for (int x = safe.x; x < safe.x + safe.width; ++x) {
                if (mrow[x] == 0) continue;
                drow[x] = frow[x];
            }
        }
        return;
    }

    // Feather: blend fill->original across a boundary band inside the mask.
    // DistanceTransform expects non-zero = foreground; it returns distance to nearest zero (background).
    cv::Mat bin;
    cv::threshold(mroi, bin, 0, 255, cv::THRESH_BINARY);
    cv::Mat dist;
    cv::distanceTransform(bin, dist, cv::DIST_L2, 3);

    const float inv = 1.0f / (float)std::max(1, feather_px);
    for (int yy = 0; yy < safe.height; ++yy) {
        int y = safe.y + yy;
        const uchar *mrow = mask_full_u8.ptr<uchar>(y);
        const cv::Vec3b *orow = orig_bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b *frow = fill_bgr.ptr<cv::Vec3b>(y);
        cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
        const float *drowf = dist.ptr<float>(yy);
        for (int xx = 0; xx < safe.width; ++xx) {
            int x = safe.x + xx;
            if (mrow[x] == 0) continue;
            float w = drowf[xx] * inv;
            if (w <= 0.0f) {
                drow[x] = orow[x];
            } else if (w >= 1.0f) {
                drow[x] = frow[x];
            } else {
                cv::Vec3b o = orow[x];
                cv::Vec3b f = frow[x];
                cv::Vec3b out;
                for (int c = 0; c < 3; ++c) {
                    float v = (1.0f - w) * (float)o[c] + w * (float)f[c];
                    out[c] = (uchar)clampi((int)std::lround(v), 0, 255);
                }
                drow[x] = out;
            }
        }
    }
}

static void apply_flat_color_with_feather_bbox(
    cv::Mat &dst_bgr,
    const cv::Mat &orig_bgr,
    const cv::Vec3b &fill_color,
    const cv::Mat &mask_full_u8,
    const cv::Rect &bbox,
    int feather_px
) {
    if (dst_bgr.empty() || orig_bgr.empty() || mask_full_u8.empty()) return;
    if (dst_bgr.type() != CV_8UC3 || orig_bgr.type() != CV_8UC3 || mask_full_u8.type() != CV_8UC1) return;
    if (dst_bgr.size() != orig_bgr.size() || dst_bgr.size() != mask_full_u8.size()) return;

    cv::Rect safe = bbox & cv::Rect(0, 0, dst_bgr.cols, dst_bgr.rows);
    if (safe.area() <= 0) return;

    cv::Mat mroi = mask_full_u8(safe);
    if (cv::countNonZero(mroi) == 0) return;

    if (feather_px <= 0) {
        for (int y = safe.y; y < safe.y + safe.height; ++y) {
            const uchar *mrow = mask_full_u8.ptr<uchar>(y);
            cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
            for (int x = safe.x; x < safe.x + safe.width; ++x) {
                if (mrow[x] == 0) continue;
                drow[x] = fill_color;
            }
        }
        return;
    }

    cv::Mat bin;
    cv::threshold(mroi, bin, 0, 255, cv::THRESH_BINARY);
    cv::Mat dist;
    cv::distanceTransform(bin, dist, cv::DIST_L2, 3);

    const float inv = 1.0f / (float)std::max(1, feather_px);
    for (int yy = 0; yy < safe.height; ++yy) {
        int y = safe.y + yy;
        const uchar *mrow = mask_full_u8.ptr<uchar>(y);
        const cv::Vec3b *orow = orig_bgr.ptr<cv::Vec3b>(y);
        cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
        const float *drowf = dist.ptr<float>(yy);
        for (int xx = 0; xx < safe.width; ++xx) {
            int x = safe.x + xx;
            if (mrow[x] == 0) continue;
            float w = drowf[xx] * inv;
            if (w <= 0.0f) {
                drow[x] = orow[x];
            } else if (w >= 1.0f) {
                drow[x] = fill_color;
            } else {
                cv::Vec3b o = orow[x];
                cv::Vec3b out;
                for (int c = 0; c < 3; ++c) {
                    float v = (1.0f - w) * (float)o[c] + w * (float)fill_color[c];
                    out[c] = (uchar)clampi((int)std::lround(v), 0, 255);
                }
                drow[x] = out;
            }
        }
    }
}

// Create a fill frame based on mode (blur/bg_ema/inpaint/solid-as-image).
// bg_ema is stateful across frames; we keep it as CV_32FC3 in BGR.
static cv::Mat make_fill_frame_bgr(
    const cv::Mat &orig_bgr,
    const cv::Mat &mask_union_u8,
    const OfflineOptions &opt,
    const cv::Vec3b &dominant_bgr,
    cv::Mat &bg_ema_f32,
    bool &bg_inited
) {
    if (orig_bgr.empty() || orig_bgr.type() != CV_8UC3) return cv::Mat();
    cv::Mat fill = orig_bgr.clone();

    std::string mode = opt.fillMode;
    for (auto &ch : mode) ch = (char)std::tolower((unsigned char)ch);

    if (mode == "blur") {
        float sigma = std::max(0.0f, opt.fillBlurSigma);
        if (sigma <= 1e-6f) return fill;
        int k = (int)(2 * std::lround(3.0 * sigma) + 1);
        k = std::max(3, k);
        if ((k % 2) == 0) k += 1;
        cv::GaussianBlur(orig_bgr, fill, cv::Size(k, k), sigma, sigma, cv::BORDER_DEFAULT);
        return fill;
    }

    if (mode == "bg_ema") {
        float a = opt.fillBgEmaAlpha;
        if (a < 0.0f) a = 0.0f;
        if (a > 0.9999f) a = 0.9999f;
        if (!bg_inited || bg_ema_f32.empty() || bg_ema_f32.size() != orig_bgr.size()) {
            orig_bgr.convertTo(bg_ema_f32, CV_32FC3);
            bg_inited = true;
        } else {
            // Update EMA only on unmasked pixels (mask=0)
            cv::Mat cur_f32;
            orig_bgr.convertTo(cur_f32, CV_32FC3);
            for (int y = 0; y < orig_bgr.rows; ++y) {
                const uchar *mrow = mask_union_u8.empty() ? nullptr : mask_union_u8.ptr<uchar>(y);
                const cv::Vec3f *crow = cur_f32.ptr<cv::Vec3f>(y);
                cv::Vec3f *brow = bg_ema_f32.ptr<cv::Vec3f>(y);
                for (int x = 0; x < orig_bgr.cols; ++x) {
                    bool masked = (mrow && mrow[x] != 0);
                    if (masked) continue;
                    brow[x] = a * brow[x] + (1.0f - a) * crow[x];
                }
            }
        }
        bg_ema_f32.convertTo(fill, CV_8UC3);
        return fill;
    }

    if (mode == "inpaint") {
        if (mask_union_u8.empty() || cv::countNonZero(mask_union_u8) == 0) return fill;
        int radius = std::max(1, opt.fillInpaintRadius);
        int method = cv::INPAINT_TELEA;
        std::string m = opt.fillInpaintMethod;
        for (auto &ch : m) ch = (char)std::tolower((unsigned char)ch);
        if (m == "ns" || m == "navier" || m == "navier-stokes") method = cv::INPAINT_NS;
        cv::inpaint(orig_bgr, mask_union_u8, fill, (double)radius, method);
        return fill;
    }

    // solid (as an image): build constant-color fill frame so the same compositing path can be used.
    if (mode == "solid" || mode.empty()) {
        cv::Vec3b c = choose_mask_color_bgr(opt, /*classId=*/0, dominant_bgr);
        fill.setTo(cv::Scalar(c[0], c[1], c[2]));
        return fill;
    }

    // Unknown mode: fallback to solid green
    fill.setTo(cv::Scalar(0, 255, 0));
    return fill;
}

int run_offline_evaluation(const OfflineOptions &opt, YOLO_V8& yoloDetector) {
    // Avoid hard-crash (exit code 141) if ffmpeg terminates early and our stdin pipe breaks.
    // We'll detect broken pipes via fwrite return values instead.
    std::signal(SIGPIPE, SIG_IGN);
    const std::string yolo_matcher = resolve_yolo_matcher(opt);
    fs::create_directories(opt.outDir);
    if (!opt.pixelMode) {
        fs::create_directories(opt.dictDir);
    }

    // For pixelMode we pass an empty dictDir to avoid creating / writing dictionaries.
    OfflineProcessor processor(opt.outDir, opt.pixelMode ? "" : opt.dictDir);
    
    // Load existing dictionary only for traditional (YOLO) mode
    if (!opt.pixelMode && yolo_matcher != "latent_key") {
        if (processor.loadDictionary()) {
            std::cout << "Loaded existing dictionary with " << processor.getDict().size() << " items." << std::endl;
        } else {
            std::cout << "No existing dictionary found (or empty). Starting fresh." << std::endl;
        }
    }

    cv::VideoCapture cap;
    cap.open(opt.source, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source: " << opt.source << std::endl;
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    std::string stitchedOut = opt.outDir + "/segmented_output.mp4";
    std::string originalOut = opt.outDir + "/original_output.mp4"; // Baseline video (no masking)
    std::string reportPath = opt.outDir + "/report.json";

    // ---------------------------------------------------------------------
    // Controlled encoder: FFmpeg/libx264 via stdin pipe (BGR rawvideo).
    // This lets us fix GOP/keyint/scenecut/bframes/preset/profile/crf, unlike OpenCV VideoWriter.
    // ---------------------------------------------------------------------
    struct FfmpegPipeWriter {
        FILE *pipe{nullptr};
        std::string cmd;
        std::string stderrLogPath;
        bool warnedWrite{false};
        uint64_t bytesWritten{0};
        int w{0}, h{0};
        double fps{30.0};
        bool open(const std::string &outPath, int width, int height, double fpsIn, const OfflineOptions &opt, const char *tag) {
            w = width; h = height; fps = fpsIn;
            // Normalize codec name
            std::string codec = opt.encCodec.empty() ? std::string("libx264") : opt.encCodec;
            for (auto &ch : codec) ch = (char)std::tolower((unsigned char)ch);
            const int keyint = std::max(1, opt.encGop);
            const int bframes = std::max(0, opt.encBFrames);

            // Codec-specific params (x264/x265) and preset mapping (AV1).
            std::string paramFlag;
            std::string paramStr;
            std::string presetArg;
            std::string extraCodecArgs;

            if (codec == "libx264") {
                std::ostringstream p;
                if (!opt.encOpenGopDefaults) {
                    p << "keyint=" << keyint
                      << ":min-keyint=" << keyint
                      << ":";
                    if (opt.encNoScenecut) p << "scenecut=0:";
                }
                p << "bframes=" << bframes;
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
                // SVT-AV1 uses numeric preset (0=slowest/best, 13=fastest). Map common x264-like presets.
                auto mapPreset = [&](const std::string &s) -> std::string {
                    std::string v = s;
                    for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                    // If user already passed a number, keep it.
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
                // libaom-av1 uses -cpu-used (0=best/slow, 8=fast). Map roughly from x264 presets.
                auto mapCpuUsed = [&](const std::string &s) -> std::string {
                    std::string v = s;
                    for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
                    if (v == "veryslow" || v == "slower" || v == "slow") return "2";
                    if (v == "medium") return "4";
                    if (v == "fast") return "6";
                    if (v == "faster" || v == "veryfast" || v == "superfast" || v == "ultrafast") return "8";
                    return "6";
                };
                presetArg.clear(); // not used
                extraCodecArgs = "-cpu-used " + mapCpuUsed(opt.encPreset) + " -g " + std::to_string(keyint) + " -bf " + std::to_string(bframes) + " -sc_threshold 0";
            } else {
                // Unknown codec: try to run it with generic GOP controls (may be ignored by codec).
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

            if (!presetArg.empty()) {
                oss << "-preset " << presetArg << " ";
            }
            if (!opt.encTune.empty() && opt.encTune != "none") {
                // Tune is generally x264/x265-only; harmless if unsupported (ffmpeg will error).
                if (codec == "libx264" || codec == "libx265") {
                    oss << "-tune " << opt.encTune << " ";
                }
            }
            // Profile/level handling:
            // - libx264: allow baseline/main/high, etc.
            // - libx265: ffmpeg/x265 expects main/main10/etc. Passing baseline/high will fail.
            if (codec == "libx264") {
                std::string profile = opt.encProfile;
                std::string level = opt.encLevel;
                for (auto &ch : profile) ch = (char)std::tolower((unsigned char)ch);
                for (auto &ch : level) ch = (char)std::tolower((unsigned char)ch);
                if (!profile.empty() && profile != "none") {
                    oss << "-profile:v " << opt.encProfile << " ";
                }
                if (!level.empty() && level != "none") {
                    oss << "-level:v " << opt.encLevel << " ";
                }
            }
            if (!extraCodecArgs.empty()) {
                oss << extraCodecArgs << " ";
            }
            oss << "-pix_fmt yuv420p "
                ;
            if (opt.encBitrateMbps > 0.0) {
                double bitrate = std::max(0.1, opt.encBitrateMbps);
                double maxrate = (opt.encMaxrateMbps > 0.0) ? opt.encMaxrateMbps : bitrate;
                double bufsize = (opt.encBufsizeMbits > 0.0) ? opt.encBufsizeMbits : (2.0 * maxrate);
                oss << "-b:v " << bitrate << "M "
                    << "-maxrate " << maxrate << "M "
                    << "-bufsize " << bufsize << "M ";
            } else {
                oss << "-crf " << std::max(0, opt.encCrf) << " ";
            }
            if (!paramFlag.empty() && !paramStr.empty()) {
                oss << paramFlag << " \"" << paramStr << "\" ";
            }
            oss << "\"" << outPath << "\"";
            // Capture ffmpeg stderr to a sidecar log for debugging. This prevents silent failures
            // (e.g., invalid encoder args) from going unnoticed while keeping console output clean.
            stderrLogPath = outPath + ".ffmpeg.stderr.log";
            oss << " 2>\"" << stderrLogPath << "\"";

            cmd = oss.str();
            std::cout << "[FFmpegPipeWriter] start(" << tag << ") cmd: " << cmd << std::endl;
            pipe = popen(cmd.c_str(), "w");
            if (!pipe) {
                std::cerr << "[FFmpegPipeWriter] Failed to start ffmpeg for " << tag << "\n";
                std::cerr << "  cmd: " << cmd << "\n";
                std::cerr << "  stderr: " << stderrLogPath << "\n";
                return false;
            }
            return true;
        }
        bool write(const cv::Mat &bgr) {
            if (!pipe) return false;
            if (bgr.empty()) return false;
            if (bgr.type() != CV_8UC3) {
                if (!warnedWrite) {
                    warnedWrite = true;
                    std::cerr << "[FFmpegPipeWriter] Unexpected frame type=" << bgr.type()
                              << " (expected CV_8UC3). stderr: " << stderrLogPath << "\n";
                    std::cerr << "  cmd: " << cmd << "\n";
                }
                return false;
            }
            if (bgr.cols != w || bgr.rows != h) {
                if (!warnedWrite) {
                    warnedWrite = true;
                    std::cerr << "[FFmpegPipeWriter] Unexpected frame size=" << bgr.cols << "x" << bgr.rows
                              << " (expected " << w << "x" << h << "). stderr: " << stderrLogPath << "\n";
                    std::cerr << "  cmd: " << cmd << "\n";
                }
                return false;
            }
            size_t need = (size_t)w * (size_t)h * 3;
            if (!bgr.isContinuous()) {
                // fallback copy
                cv::Mat tmp = bgr.clone();
                size_t wrote = fwrite(tmp.data, 1, need, pipe);
                if (wrote != need && !warnedWrite) {
                    warnedWrite = true;
                    std::cerr << "[FFmpegPipeWriter] Short write to ffmpeg pipe (need=" << need
                              << " wrote=" << wrote << "). stderr: " << stderrLogPath << "\n";
                    std::cerr << "  cmd: " << cmd << "\n";
                }
                if (wrote == need) bytesWritten += (uint64_t)wrote;
                return wrote == need;
            }
            size_t wrote = fwrite(bgr.data, 1, need, pipe);
            if (wrote != need && !warnedWrite) {
                warnedWrite = true;
                std::cerr << "[FFmpegPipeWriter] Short write to ffmpeg pipe (need=" << need
                          << " wrote=" << wrote << "). stderr: " << stderrLogPath << "\n";
                std::cerr << "  cmd: " << cmd << "\n";
            }
            if (wrote == need) bytesWritten += (uint64_t)wrote;
            return wrote == need;
        }
        void close() {
            if (!pipe) return;
            fflush(pipe);
            int rc = pclose(pipe);
            std::cout << "[FFmpegPipeWriter] close bytesWritten=" << bytesWritten
                      << " stderr: " << stderrLogPath << std::endl;
            if (rc != 0) {
                std::cerr << "[FFmpegPipeWriter] ffmpeg non-zero exit (" << rc << "). stderr: " << stderrLogPath << "\n";
                std::cerr << "  cmd: " << cmd << "\n";
            }
            pipe = nullptr;
        }
        bool isOpen() const { return pipe != nullptr; }
    };

    std::string recoveredOut = opt.outDir + "/recovered_output.mp4";
    FfmpegPipeWriter writer, origWriter, recWriter;
    if (!opt.buildIndexOnly) {
        if (opt.useFfmpegEncoder) {
            (void)origWriter.open(originalOut, width, height, fps, opt, "baseline");
            (void)writer.open(stitchedOut, width, height, fps, opt, "masked");
            (void)recWriter.open(recoveredOut, width, height, fps, opt, "recovered");
        } else {
            // Fallback to OpenCV VideoWriter if requested (legacy behavior).
            cv::VideoWriter vw_mask, vw_base, vw_rec;
            (void)vw_mask; (void)vw_base; (void)vw_rec;
            std::cerr << "Warning: --no-ffmpeg-enc selected but OpenCV VideoWriter fallback is not wired in this build.\n";
        }
    }
    std::string frameCacheDir = (fs::path(opt.outDir) / "frame_cache").string();
    std::string originalFrameCachePath = (fs::path(frameCacheDir) / "original_frames.bgr").string();
    std::string maskedFrameCachePath = (fs::path(frameCacheDir) / "masked_frames.bgr").string();
    std::ofstream originalFrameCache;
    std::ofstream maskedFrameCache;
    if (opt.dumpFrameCache) {
        fs::create_directories(frameCacheDir);
        originalFrameCache.open(originalFrameCachePath, std::ios::binary);
        maskedFrameCache.open(maskedFrameCachePath, std::ios::binary);
        if (!originalFrameCache.is_open() || !maskedFrameCache.is_open()) {
            std::cerr << "Warning: could not open frame cache files under " << frameCacheDir << std::endl;
        }
    }

    uint64_t total_occurrences = 0;
    uint64_t matched_occurrences = 0;

    // Sidecar SEI payload dump (for pixel experiments / inspection)
    std::string seiOutPath = opt.outDir + "/msk1_payloads.bin";
    std::ofstream seiOut(seiOutPath, std::ios::binary);
    if (!seiOut.is_open()) {
        std::cerr << "Warning: could not open SEI payload dump: " << seiOutPath << std::endl;
    }

    // Quality metrics (SSIM / PSNR) for masked vs baseline
    double ssim_sum = 0.0;
    double psnr_sum = 0.0;
    int metric_frames = 0;
    std::vector<int> frame_indices;
    std::vector<double> ssim_values;
    std::vector<double> psnr_values;

    // Quality metrics for recovered vs baseline
    double rec_ssim_sum = 0.0;
    double rec_psnr_sum = 0.0;
    int rec_metric_frames = 0;
    std::vector<double> rec_ssim_values;
    std::vector<double> rec_psnr_values;

    // Timing (optional, stored in report.json; no extra printing)
    std::vector<double> t_infer_ms, t_dict_ms, t_paint_ms, t_recover_ms, t_total_ms;
    std::vector<double> t_pixel_bootstrap_ms, t_pixel_roi_ms, t_pixel_band_ms;
    std::vector<double> t_pixel_row_ms, t_pixel_flow_ms;
    std::vector<int> t_pixel_bands_used;
    std::vector<json> t_pixel_template_counts; // per-frame: {relPath: count}
    std::vector<int> t_pixel_scanned_bootstrap, t_pixel_scanned_roi, t_pixel_scanned_band;

    // Per-frame CSV fields (always recorded into report.json/per_frame for downstream CSV merge)
    std::vector<double> t_preprocess_ms, t_inference_ms, t_postprocess_ms;
    std::vector<double> t_masking_ms, t_encode_baseline_ms, t_encode_masked_ms, t_stitching_ms;
    std::vector<int> t_object_present_model, t_object_successfully_masked, t_frame_flags; // 0=raw, 1=ref
    std::vector<int> t_frame_flags_raw;
    std::vector<int> t_latent_minted_regions, t_latent_reused_regions;
    std::vector<double> t_masked_bbox_pct, t_masked_alpha_pct;
    std::vector<int> t_changed_pixels;
    std::vector<double> t_changed_pixels_pct;

    // Optional latent dump (for threshold sweep experiments)
    std::ofstream latOut;
    if (opt.dumpLatents) {
        std::string path = opt.latentsOutPath.empty()
            ? (fs::path(opt.outDir) / "latents.jsonl").string()
            : opt.latentsOutPath;
        latOut.open(path, std::ios::out);
        if (!latOut.is_open()) {
            std::cerr << "Warning: could not open latents dump: " << path << std::endl;
        }
    }

    // Simple temporal stabilization for YOLO template selection: keep last chosen template per class
    // if the bbox is highly overlapping (reduces flashing).
    struct LastTpl {
        cv::Rect box;
        std::string hash;
        uint64_t phash64{0};
        bool valid{false};
    };
    std::unordered_map<int, LastTpl> last_tpl_by_class;
    std::unordered_map<std::string, cv::Mat> rgb_hist_cache;
    struct LastAssigned {
        cv::Rect box;
        std::string key;
        bool valid{false};
    };
    std::unordered_map<int, std::vector<LastAssigned>> last_assigned_by_class;
    uint64_t matcher_new_templates = 0;
    uint64_t matcher_id_switches = 0;
    auto record_assignment = [&](int cls, const cv::Rect &box, const std::string &key, bool is_new) {
        if (key.empty() || box.area() <= 0) return;
        auto &vec = last_assigned_by_class[cls];
        size_t best_idx = (size_t)-1;
        float best_iou = 0.0f;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (!vec[i].valid) continue;
            float iou = iou_rect(vec[i].box, box);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = i;
            }
        }
        if (best_idx != (size_t)-1 && best_iou >= 0.5f && vec[best_idx].key != key) {
            matcher_id_switches++;
        }
        if (is_new) matcher_new_templates++;
        LastAssigned cur;
        cur.box = box;
        cur.key = key;
        cur.valid = true;
        if (best_idx != (size_t)-1 && best_iou >= 0.3f) {
            vec[best_idx] = std::move(cur);
        } else {
            vec.push_back(std::move(cur));
            if (vec.size() > 8) vec.erase(vec.begin());
        }
    };

    // Latent-key offline simulator state (YOLO only): template_id -> embedding/path
    uint32_t next_template_id = 1;
    struct LatentTpl {
        uint32_t id{0};
        int cls{0};
        cv::Size wh{};
        std::array<float, 32> emb_norm{};
        std::string path; // saved RGBA template path
        uint64_t use_count{0}; // how many times selected for this run (server-side)
        int minted_frame{-1};
        int last_used_frame{-1};
    };
    std::vector<LatentTpl> latent_bank;
    std::unordered_map<uint32_t, size_t> latent_id_to_index;
    // Cache loaded RGBA templates to avoid per-detection disk IO (critical for performance).
    std::unordered_map<uint32_t, cv::Mat> latent_rgba_cache;

    // Optional: load prebuilt latent bank (e.g., from a build-index pass).
    auto try_load_latent_bank = [&](const std::string &path) -> bool {
        try {
            std::ifstream f(path);
            if (!f.is_open()) return false;
            json bank; f >> bank;
            if (!bank.is_array()) return false;
            latent_bank.clear();
            latent_id_to_index.clear();
            uint32_t max_id = 0;
            for (const auto &it : bank) {
                if (!it.is_object()) continue;
                LatentTpl t;
                t.id = (uint32_t)it.value("id", 0);
                t.cls = (int)it.value("cls", 0);
                t.wh = cv::Size((int)it.value("w", 0), (int)it.value("h", 0));
                t.path = it.value("path", "");
                t.use_count = (uint64_t)it.value("use_count", 0ULL);
                t.minted_frame = (int)it.value("minted_frame", -1);
                t.last_used_frame = (int)it.value("last_used_frame", -1);
                // emb
                if (it.contains("emb") && it["emb"].is_array() && it["emb"].size() == 32) {
                    for (int i = 0; i < 32; ++i) t.emb_norm[(size_t)i] = it["emb"][i].get<float>();
                } else {
                    continue; // invalid
                }
                if (t.id == 0 || t.wh.width <= 0 || t.wh.height <= 0 || t.path.empty()) continue;
                latent_id_to_index[t.id] = latent_bank.size();
                latent_bank.push_back(std::move(t));
                if (t.id > max_id) max_id = t.id;
            }
            next_template_id = max_id + 1;
            std::cout << "[LatentBank] Loaded " << latent_bank.size() << " templates from " << path << std::endl;
            return !latent_bank.empty();
        } catch (...) {
            return false;
        }
    };
    if (yolo_matcher == "latent_key" && !opt.pixelMode) {
        std::string p = opt.latentBankLoadPath;
        if (p.empty()) {
            fs::path def = fs::path(opt.dictDir) / "latent_bank.json";
            if (fs::exists(def)) p = def.string();
        }
        if (!p.empty()) {
            (void)try_load_latent_bank(p);
        }
    }
    struct LastLatent {
        cv::Rect box;
        uint32_t id{0};
        std::array<float, 32> emb_norm{};
        bool valid{false};
    };
    // Multi-instance support: maintain multiple "last" entries per class to avoid mixing
    // templates when there are multiple objects of the same class in a frame (e.g., 2 guns).
    std::unordered_map<int, std::vector<LastLatent>> last_latents_by_class;
    std::unordered_map<int, int> latent_boost_until_frame; // per class

    // Client-side fallback: if a template_id is "new" this frame (not yet delivered),
    // reuse the last recoverable template for the same class to avoid flashing.
    struct LastClientTpl {
        cv::Rect box;
        uint32_t id{0};
        bool valid{false};
    };
    // Multi-instance cache per class
    std::unordered_map<int, std::vector<LastClientTpl>> last_client_tpls_by_class;

    // Pixel mode tracker state (single-object tracker; ROI+Kalman)
    PixelTracker pixelTracker;

    // Latent-key stats (server-side selection)
    uint64_t latent_minted = 0;
    uint64_t latent_reused = 0;
    uint64_t latent_forced_mints = 0;
    uint64_t latent_motion_boost_mints = 0;
    uint64_t latent_periodic_skipped = 0;
    uint64_t latent_motion_boost_events = 0;
    uint64_t latent_merged_mints = 0; // number of would-be mints merged into an existing template
    std::vector<float> latent_reuse_cos_sims;

    // Latent matching cost breakdown
    double latent_search_ms_sum = 0.0;   // bank scanning cost
    double latent_mint_io_ms_sum = 0.0;  // imwrite cost for newly minted templates

    // Reliability stats
    uint64_t yolo_heal_skipped = 0; // regions not masked because not perfectly recoverable
    uint64_t yolo_forced_masked = 0; // regions masked in upper-bound experiment mode
    bool frame_mode_state_inited = false;
    int frame_mode_state = 0;
    int frame_mode_candidate = 0;
    int frame_mode_candidate_count = 0;
    
    int frameIdx = 0;
    cv::Mat frame;
    cv::Vec3b dominant_bgr(0, 255, 0);
    cv::Mat bg_ema_f32;
    bool bg_ema_inited = false;
    while (cap.read(frame)) {
        if (opt.maxFrames > 0 && frameIdx >= opt.maxFrames) break;
        cv::Mat processed = frame.clone();  // masked view (what server sends)
        cv::Mat recovered;                 // reconstructed client view from masked + side-channel
        double infer_ms = 0.0, paint_ms = 0.0, recover_ms = 0.0, dict_ms = 0.0, frame_total_ms = 0.0;
        double preprocess_ms = 0.0, inference_ms = 0.0, postprocess_ms = 0.0;
        double masking_ms = 0.0, encode_baseline_ms = 0.0, encode_masked_ms = 0.0, sei_build_ms = 0.0;
        auto t_frame_start = std::chrono::steady_clock::now();

        // Per-frame latent/coverage counters (only meaningful for YOLO mode).
        int frame_latent_minted = 0;
        int frame_latent_reused = 0;
        uint64_t frame_masked_bbox_px = 0;
        uint64_t frame_masked_alpha_px = 0;
        // Union mask of pixels we decide to modify on the server stream (before feather/composite).
        // Used for fill generation (inpaint/bg_ema) and as an approximation for changed_pixels metrics.
        cv::Mat mask_union_u8 = cv::Mat::zeros(frame.size(), CV_8UC1);

        std::vector<DL_RESULT> dets;
        double pixel_bootstrap_ms = 0.0;
        double pixel_roi_ms = 0.0;
        double pixel_kalman_ms = 0.0;
        double pixel_band_ms = 0.0;
        double pixel_row_ms = 0.0;
        double pixel_flow_ms = 0.0;
        int pixel_bands_used = 0;
        int pixel_scanned_bootstrap = 0;
        int pixel_scanned_roi = 0;
        int pixel_scanned_band = 0;
        {
            auto t0 = std::chrono::steady_clock::now();
            if (opt.pixelMode) {
                // Pixel mode: Kalman + ROI template matching, with periodic re-bootstrap.
                if (!load_pixel_templates_once(opt)) {
                    // no templates
                } else {
                    // Frame edges
                    cv::Mat frame_gray, frame_edges;
                    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
                    cv::Canny(frame_gray, frame_edges, 50, 150);
                    if (!g_pixel_scales_calibrated) {
                        calibrate_template_scales(frame_edges, opt);
                    }

                    bool need_bootstrap = (!pixelTracker.inited) ||
                        (opt.pixelBootstrapInterval > 0 && (frameIdx - pixelTracker.lastBootstrapFrame) >= opt.pixelBootstrapInterval) ||
                        (pixelTracker.lostCount >= std::max(1, opt.pixelLostMax));

                    if (need_bootstrap) {
                        bool ok = pixelTracker.bootstrap(frame, frame_edges, frameIdx, opt, pixel_bootstrap_ms, pixel_scanned_bootstrap);
                        (void)ok;
                    }

                    cv::Rect box;
                    float score = 0.0f;
                    int bestTi = -1;
                    std::vector<std::pair<int, float>> ranked; // (ti, score)
                    bool ok2 = pixelTracker.roi_match(frame, frame_edges, frameIdx, opt, box, score, bestTi, ranked,
                                                      pixel_roi_ms, pixel_kalman_ms, pixel_scanned_roi);
                    float bestThr = pixel_template_accept_thr(bestTi, opt);
                    if (ok2 && score >= bestThr && box.area() > 0 && bestTi >= 0) {
                        pixelTracker.lostCount = 0;

                        // Precompute bands around tracked ROI Y (avoid picking HUD edges).
                        std::vector<int> bandYs;
                        int cy = box.y + box.height / 2;
                        int halfW = std::max(opt.pixelBandPadY * 3, box.height * 3);
                        PixelTracker::select_bands_from_edges_window(frame_edges, cy, halfW,
                                                                     opt.pixelNumBands, opt.pixelBandMinSep,
                                                                     bandYs, pixel_row_ms);
                        pixel_bands_used = (int)bandYs.size();

                        // Optical flow dx/dy once
                        float dx = 0.0f, dy = 0.0f;
                        if (opt.pixelUseFlow && !pixelTracker.prev_gray.empty()) {
                            PixelTracker::estimate_flow_dxdy(pixelTracker.prev_gray, frame_gray,
                                                           opt.pixelFlowMaxPts, dx, dy, pixel_flow_ms);
                            pixelTracker.prev_gray = frame_gray;
                        } else if (pixelTracker.prev_gray.empty() && opt.pixelUseFlow) {
                            pixelTracker.prev_gray = frame_gray;
                        }

                        // Reduce flicker: refresh multi-peak detections only on bootstrap frames.
                        // Between bootstraps, reuse the previous peak set shifted by global optical flow.
                        if (need_bootstrap || pixelTracker.persisted_peaks.empty()) {
                            // Band-scan top-K templates (ranked by ROI score) to avoid missing variants (e.g. dark bricks).
                            struct Peak { cv::Rect r; float s; int ti; };
                            std::vector<Peak> all;
                            int bandTopK = opt.pixelBandTopK;
                            if (bandTopK <= 0) bandTopK = (int)ranked.size();
                            if ((int)ranked.size() < bandTopK) bandTopK = (int)ranked.size();

                            double band_ms_sum = 0.0;
                            int scanned_sum = 0;
                            for (int k = 0; k < bandTopK; ++k) {
                                int ti = ranked[(size_t)k].first;
                                std::vector<std::pair<cv::Rect, float>> peaks;
                                double ms_i = 0.0;
                                int scanned_i = 0;
                                pixelTracker.band_scan_multi_precomputed(frame, frame_edges, ti, bandYs, dy, opt,
                                                                         peaks, ms_i, scanned_i);
                                band_ms_sum += ms_i;
                                scanned_sum += scanned_i;
                                for (auto &pp : peaks) all.push_back(Peak{pp.first, pp.second, ti});
                            }
                            pixel_band_ms = band_ms_sum;
                            pixel_scanned_band = scanned_sum;

                            pixelTracker.persisted_peaks.clear();
                            pixelTracker.persisted_peak_ti.clear();
                            pixelTracker.persisted_peaks_frame = frameIdx;

                            // If nothing found, fallback to best single detection.
                            if (all.empty()) {
                                pixelTracker.persisted_peaks.push_back(box);
                                pixelTracker.persisted_peak_ti.push_back(bestTi);
                            } else {
                                // Cross-template NMS to reduce confusion from similar-looking units
                                std::sort(all.begin(), all.end(), [](const Peak &a, const Peak &b){ return a.s > b.s; });
                                auto iou = [](const cv::Rect &a, const cv::Rect &b)->float {
                                    cv::Rect inter = a & b;
                                    float ia = (float)std::max(0, inter.area());
                                    float ua = (float)std::max(1, a.area() + b.area() - inter.area());
                                    return ia / ua;
                                };
                                const float nms_iou = 0.3f;
                                for (const auto &p : all) {
                                    bool ok = true;
                                    for (const auto &q : pixelTracker.persisted_peaks) {
                                        if (iou(p.r, q) > nms_iou) { ok = false; break; }
                                    }
                                    if (ok) {
                                        pixelTracker.persisted_peaks.push_back(p.r);
                                        pixelTracker.persisted_peak_ti.push_back(p.ti);
                                    }
                                    if ((int)pixelTracker.persisted_peaks.size() >= opt.pixelMaxPeaks) break;
                                }
                            }
                        } else {
                            // Reuse previous peaks with flow-based shift.
                            pixel_band_ms = 0.0;
                            pixel_scanned_band = 0;
                            int sdx = (int)std::round((double)dx);
                            int sdy = (int)std::round((double)dy);
                            if (sdx != 0 || sdy != 0) {
                                for (auto &r : pixelTracker.persisted_peaks) {
                                    r.x += sdx;
                                    r.y += sdy;
                                    r = PixelTracker::clamp_rect(r, frame.size());
                                }
                            }
                        }

                        // Emit detections from the persisted peak set (current-frame coordinates).
                        for (size_t pi = 0; pi < pixelTracker.persisted_peaks.size(); ++pi) {
                            cv::Rect r = pixelTracker.persisted_peaks[pi];
                            if (r.area() <= 0) continue;
                            int ti = (pi < pixelTracker.persisted_peak_ti.size()) ? pixelTracker.persisted_peak_ti[pi] : bestTi;
                            ti = std::max(0, std::min(ti, (int)g_pixel_templates.size() - 1));
                            const auto &tmpl = g_pixel_templates[(size_t)ti];
                            DL_RESULT d{};
                            d.confidence = 1.0f;
                            d.box = opt.pixelGridHeader ? (r & cv::Rect(0, 0, frame.cols, frame.rows))
                                                        : expand_pixel_mask_box(r, opt.pixelMaskPadPx, frame.size());
                            if (d.box.area() <= 0) continue;
                            d.boxMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                            cv::rectangle(d.boxMask, d.box, cv::Scalar(255), cv::FILLED);
                            d.classId = tmpl.kindId;
                            d.seiPath = tmpl.relPath;
                            dets.push_back(std::move(d));
                        }

                    } else {
                        // If ROI match fails temporarily, keep the last persisted peak set alive
                        // by shifting with optical flow. This avoids hard on/off flicker.
                        pixelTracker.lostCount++;
                        if (!pixelTracker.persisted_peaks.empty() &&
                            opt.pixelUseFlow &&
                            pixelTracker.lostCount <= std::max(1, opt.pixelLostMax) &&
                            !pixelTracker.prev_gray.empty()) {
                            float dx = 0.0f, dy = 0.0f;
                            PixelTracker::estimate_flow_dxdy(pixelTracker.prev_gray, frame_gray,
                                                             opt.pixelFlowMaxPts, dx, dy, pixel_flow_ms);
                            pixelTracker.prev_gray = frame_gray;
                            int sdx = (int)std::round((double)dx);
                            int sdy = (int)std::round((double)dy);
                            for (auto &r : pixelTracker.persisted_peaks) {
                                r.x += sdx;
                                r.y += sdy;
                                r = PixelTracker::clamp_rect(r, frame.size());
                            }
                            pixel_band_ms = 0.0;
                            pixel_scanned_band = 0;
                            // Emit detections from persisted peaks even on lost frames.
                            for (size_t pi = 0; pi < pixelTracker.persisted_peaks.size(); ++pi) {
                                cv::Rect r = pixelTracker.persisted_peaks[pi];
                                if (r.area() <= 0) continue;
                                int ti = (pi < pixelTracker.persisted_peak_ti.size()) ? pixelTracker.persisted_peak_ti[pi] : 0;
                                ti = std::max(0, std::min(ti, (int)g_pixel_templates.size() - 1));
                                const auto &tmpl = g_pixel_templates[(size_t)ti];
                                DL_RESULT d{};
                                d.confidence = 1.0f;
                                d.box = opt.pixelGridHeader ? (r & cv::Rect(0, 0, frame.cols, frame.rows))
                                                            : expand_pixel_mask_box(r, opt.pixelMaskPadPx, frame.size());
                                if (d.box.area() <= 0) continue;
                                d.boxMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                                cv::rectangle(d.boxMask, d.box, cv::Scalar(255), cv::FILLED);
                                d.classId = tmpl.kindId;
                                d.seiPath = tmpl.relPath;
                                dets.push_back(std::move(d));
                            }
                        } else if (opt.pixelUseFlow && pixelTracker.prev_gray.empty()) {
                            pixelTracker.prev_gray = frame_gray;
                        }
                    }
                }
            } else {
                // Use YOLO model as before
                if (yoloDetector.RunSession(frame, dets) != RET_OK) {
                    std::cerr << "Error running inference on frame " << frameIdx << std::endl;
                    if (writer.isOpen()) (void)writer.write(processed);
                    ++frameIdx; 
                    continue;
                }
                // Timing breakdown from the inference engine (YOLO only)
                preprocess_ms = yoloDetector.LastPreprocessMs();
                inference_ms = yoloDetector.LastInferMs();
                postprocess_ms = yoloDetector.LastPostprocessMs(); // will be augmented by latent/gating work (dict_ms)
            }
            auto t1 = std::chrono::steady_clock::now();
            infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Dump raw per-frame latent embeddings (YOLO only; independent of the reuse threshold).
        // This lets us sweep thresholds offline without rerunning inference.
        if (opt.dumpLatents && latOut.is_open() && !opt.pixelMode) {
            for (const auto &d : dets) {
                if (d.confidence < opt.confThreshold) continue;
                if (d.boxMask.empty()) continue;
                auto emb = l2_normalize_32(d.maskCoeff);
                // JSONL: one detection per line
                latOut << "{\"frame\":" << frameIdx
                       << ",\"cls\":" << d.classId
                       << ",\"conf\":" << d.confidence
                       << ",\"x\":" << d.box.x << ",\"y\":" << d.box.y
                       << ",\"w\":" << d.box.width << ",\"h\":" << d.box.height
                       << ",\"emb\":[";
                for (int i = 0; i < 32; ++i) {
                    if (i) latOut << ",";
                    latOut << emb[(size_t)i];
                }
                latOut << "]}\n";
            }
        }

        // Dominant color update (only if requested).
        if (opt.maskColor == "dominant") {
            int period = std::max(1, opt.maskColorPeriod);
            if (frameIdx == 0 || (frameIdx % period) == 0) {
                if (opt.maskColorLocalPadPx > 0) {
                    dominant_bgr = dominant_color_bgr_hist_local(frame, dets, opt.maskColorLocalPadPx);
                } else {
                    dominant_bgr = dominant_color_bgr_hist(frame);
                }
            }
        }

        std::vector<SEIRegion> sei_regions;
        std::vector<PixelGridGroup> pixel_grid_groups;
        std::vector<PixelGridRuntimeGroup> pixel_grid_runtime_groups;
        int pixel_grid_ref_regions = 0;
        if (seiOut.is_open()) sei_regions.reserve(dets.size());
        // Track which YOLO hashes are NEW in this frame. A real client cannot recover these
        // until the corresponding template is transmitted out-of-band.
        std::unordered_set<std::string> yolo_new_hashes_this_frame;
        std::unordered_set<uint32_t> yolo_new_tplids_this_frame;
        // Cache canonical hashes per detection index for pass-2 recovery.
        std::vector<std::string> det_hashes;
        std::vector<uint8_t> det_is_new;
        std::vector<uint32_t> det_tplids;
        std::vector<uint8_t> det_masked; // whether the server actually masked this region (client should stitch only then)
        std::vector<cv::Rect> det_stitch_boxes; // bbox actually used for masking + stitching (may differ from detection bbox)
        if (!opt.pixelMode) {
            det_hashes.resize(dets.size());
            det_is_new.assign(dets.size(), 0);
            det_tplids.assign(dets.size(), 0);
            det_masked.assign(dets.size(), 0);
            det_stitch_boxes.assign(dets.size(), cv::Rect());
        }

        // Pass 1: paint masked stream (server output) + build canonical template keying
        auto t_paint0 = std::chrono::steady_clock::now();
        for (size_t di = 0; di < dets.size(); ++di) {
            const auto &d = dets[di];
            // For YOLO: we used to assume class 0 = cockpit; for pixel mode we use template index (classId).
            if (d.confidence < opt.confThreshold) continue;
            if (d.boxMask.empty()) continue;

            total_occurrences++;

            if (opt.pixelMode) {
                if (opt.pixelGridHeader) {
                    continue;
                }
                // Pixel mode: paint with kind-consistent color (better motion prediction),
                // while stitching uses template key (seiPath).
                // If user explicitly requests a non-default masking color strategy (e.g. dominant),
                // honor it for pixel mode too; otherwise keep kind-consistent colors.
                cv::Vec3b maskColor = kind_color_from_kind_id(d.classId);
                if (opt.maskColor == "dominant" || opt.maskColor == "black" || opt.maskColor == "brown" || opt.maskColor == "rgb") {
                    maskColor = choose_mask_color_bgr(opt, /*classId=*/d.classId, dominant_bgr);
                }
                OfflineProcessor::paint_mask_color(processed, d.boxMask, d.box, maskColor, opt.paintAlpha);
                // Pixel-mode: keep existing behavior (not part of feather/fill experiments).
                try {
                    cv::Rect safe = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (safe.area() > 0 && !mask_union_u8.empty()) {
                        cv::Mat mroi = d.boxMask(safe);
                        cv::Mat dst = mask_union_u8(safe);
                        cv::bitwise_or(dst, mroi, dst);
                    }
                } catch (...) {}

                if (seiOut.is_open()) {
                    SEIRegion r{};
                    r.id = (uint32_t)sei_regions.size();
                    r.x = (uint32_t)std::max(0, d.box.x);
                    r.y = (uint32_t)std::max(0, d.box.y);
                    r.w = (uint32_t)std::max(0, d.box.width);
                    r.h = (uint32_t)std::max(0, d.box.height);
                    r.flags = 1; // reference
                    r.class_id = (uint8_t)std::max(0, std::min(255, d.classId)); // kind_id
                    r.path = d.seiPath; // template rel path
                    sei_regions.push_back(std::move(r));
                }
            } else {
                cv::Rect safeBox = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (safeBox.area() <= 0) continue;
                cv::Mat maskROI;
                try {
                    maskROI = d.boxMask(safeBox);
                } catch (...) {
                    continue;
                }
                if (maskROI.empty()) continue;

                // Upper-bound experiment: paint ALL detected masks regardless of healability/template match.
                // This estimates the best possible H.264 bitrate reduction if object regions were perfectly
                // removable/compressible. Recovery output will NOT be correct.
                if (opt.yoloForceMaskAll) {
                    // Mark for masking; actual fill is applied after we decide per-frame fill.
                    try {
                        cv::Rect safe = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe.area() > 0) {
                            cv::Mat mroi = d.boxMask(safe);
                            cv::Mat dst = mask_union_u8(safe);
                            cv::bitwise_or(dst, mroi, dst);
                        }
                    } catch (...) {}
                    if (di < det_masked.size()) det_masked[di] = 1;
                    if (di < det_stitch_boxes.size()) det_stitch_boxes[di] = d.box;
                    yolo_forced_masked++;
                    continue;
                }

                // YOLO latent-key mode: assign template_id by maskCoeff embedding similarity.
                if (yolo_matcher == "latent_key") {
                    auto t_dict0 = std::chrono::steady_clock::now();

                    // Normalize embedding
                    std::array<float, 32> emb = l2_normalize_32(d.maskCoeff);

                    // Hybrid sampling: base periodic mint + motion-triggered boost
                    bool force_mint = false;
                    // Build-index: aggressively mint candidates (but allow merge/skip thresholds to compact)
                    if (opt.buildIndexOnly) {
                        force_mint = true;
                    } else if (opt.latentSamplePeriod > 0 && (frameIdx % opt.latentSamplePeriod) == 0) {
                        force_mint = true;
                    }

                    // Motion detection vs last box for this class
                    auto itLastVec0 = last_latents_by_class.find(d.classId);
                    if (itLastVec0 != last_latents_by_class.end()) {
                        // compare against best-overlap instance for this class
                        float best_iou = 0.0f;
                        cv::Rect best_box;
                        bool found = false;
                        for (const auto &ll : itLastVec0->second) {
                            if (!ll.valid) continue;
                            float iou = iou_rect(ll.box, safeBox);
                            if (iou > best_iou) { best_iou = iou; best_box = ll.box; found = true; }
                        }
                        if (found) {
                            float cdist = center_dist_px(best_box, safeBox);
                            float ad = area_ratio_delta(best_box, safeBox);
                            if (best_iou < opt.latentMotionIouThr || cdist > opt.latentMotionCenterPx || ad > opt.latentMotionScaleThr) {
                            latent_boost_until_frame[d.classId] = frameIdx + std::max(0, opt.latentMotionBoostFrames);
                            latent_motion_boost_events++;
                        }
                        }
                    }
                    if (!opt.buildIndexOnly) {
                        auto itBoost = latent_boost_until_frame.find(d.classId);
                        if (itBoost != latent_boost_until_frame.end() && frameIdx < itBoost->second) {
                            // In boost window, mint every frame for fluency
                            force_mint = true;
                        }
                    }

                    // Temporal stability: if last template for this class overlaps strongly, prefer it
                    uint32_t chosen_id = 0;
                    bool is_new = false;
                    auto itLastVec = last_latents_by_class.find(d.classId);
                    if (itLastVec != last_latents_by_class.end()) {
                        float best_iou = 0.0f;
                        size_t best_idx = 0;
                        bool found = false;
                        for (size_t li = 0; li < itLastVec->second.size(); ++li) {
                            const auto &ll = itLastVec->second[li];
                            if (!ll.valid) continue;
                            float iou = iou_rect(ll.box, safeBox);
                            if (iou > best_iou) { best_iou = iou; best_idx = li; found = true; }
                        }
                        if (found && best_iou >= 0.7f) {
                            const auto &ll = itLastVec->second[best_idx];
                            float sim = cosine_sim_32(emb, ll.emb_norm);
                            if (sim >= (opt.latentCosineThreshold - 0.01f)) {
                                chosen_id = ll.id;
                                latent_reuse_cos_sims.push_back(sim);
                            }
                        }
                    }

                    // Global search:
                    // - needed when no chosen_id (normal reuse)
                    // - ALSO needed when force_mint is active so we can decide to skip/merge redundant mints.
                    float bestSim = -1.0f;
                    uint32_t bestId = 0;
                    if (chosen_id == 0 || force_mint) {
                        auto t_search0 = std::chrono::steady_clock::now();
                        for (const auto &tpl : latent_bank) {
                            if (tpl.cls != d.classId) continue;
                            // Heal-only: allow size mismatch (we can resize the template) and rely on spill checks.
                            if (!opt.yoloHealOnly) {
                                if (std::abs(tpl.wh.width - safeBox.width) > 2 || std::abs(tpl.wh.height - safeBox.height) > 2) continue;
                            }
                            float sim = cosine_sim_32(emb, tpl.emb_norm);
                            if (sim > bestSim) {
                                bestSim = sim;
                                bestId = tpl.id;
                            }
                        }
                        auto t_search1 = std::chrono::steady_clock::now();
                        latent_search_ms_sum += std::chrono::duration<double, std::milli>(t_search1 - t_search0).count();
                    }
                    if (chosen_id == 0 && !force_mint) {
                        if (bestId != 0 && bestSim >= opt.latentCosineThreshold) {
                            chosen_id = bestId;
                            latent_reuse_cos_sims.push_back(bestSim);
                        }
                    }
                    // Efficiency: on periodic forced mint, skip mint if bestSim is extremely high.
                    if (force_mint && bestId != 0 && bestSim >= opt.latentPeriodSkipThr) {
                        chosen_id = bestId;
                        force_mint = false;
                        latent_periodic_skipped++;
                        latent_reuse_cos_sims.push_back(bestSim);
                    }

                    // Compaction: if we are about to mint, but bestSim is extremely high, merge into bestId.
                    if ((chosen_id == 0 || force_mint) && bestId != 0 && bestSim >= opt.latentMergeThr) {
                        chosen_id = bestId;
                        force_mint = false;
                        latent_merged_mints++;
                        latent_reuse_cos_sims.push_back(bestSim);
                    }

                    // Heal-only gating: only mask if the server can pick a template the client can
                    // understand (i.e., already in storage AND consistent with current mask).
                    // - New templates are considered not yet available on client.
                    // - Reused templates must have alpha mask matching current segmentation mask.
                    bool healable_now = true;
                    cv::Rect stitch_box = d.box;
                    if (opt.yoloHealOnly) {
                        healable_now = false;
                        const bool is_gop_i = (opt.gopSize > 0) ? ((frameIdx % opt.gopSize) == 0) : false;
                        // Try chosen_id first (if it is a reuse candidate)
                        auto try_id = [&](uint32_t cand_id) -> bool {
                            auto itIdx = latent_id_to_index.find(cand_id);
                            if (itIdx == latent_id_to_index.end()) return false;
                            const auto &tpl = latent_bank[itIdx->second];
                            cv::Mat base;
                            auto itC = latent_rgba_cache.find(cand_id);
                            if (itC != latent_rgba_cache.end()) {
                                base = itC->second;
                            } else {
                                base = cv::imread(tpl.path, cv::IMREAD_UNCHANGED);
                                if (base.empty()) return false;
                                latent_rgba_cache.emplace(cand_id, base);
                            }
                            cv::Mat img = base;
                            if (img.cols != d.box.width || img.rows != d.box.height) {
                                cv::resize(base, img, d.box.size(), 0, 0, cv::INTER_NEAREST);
                            }
                            // Alignment/match: use alpha-masked RGB MSE to handle segmentation jitter.
                            int dx = 0, dy = 0;
                            double mse = 1e18;
                            int max_shift = std::max(0, opt.yoloHealMaxShiftPx);
                            if (!best_alpha_rgb_offset_mse(img, frame, d.box, max_shift, dx, dy, mse)) return false;
                            if (mse > (double)opt.yoloHealAppearanceMseThr) return false;
                            // Also require that template alpha still overlaps the current segmentation mask at least a bit,
                            // to avoid stitching a totally unrelated template (wrong gun from the pool).
                            double cov = template_alpha_mask_coverage(img, d.boxMask, d.box, dx, dy);
                            if (cov < (double)opt.yoloHealMinMaskCoverage) return false;
                            stitch_box = d.box + cv::Point(dx, dy);
                            if (!template_alpha_matches_mask(img, d.boxMask, stitch_box,
                                                             opt.yoloHealMaskIouThr,
                                                             opt.yoloHealMaskExtraThr)) {
                                return false;
                            }
                            return true;
                        };

                        // In heal-only, we accept reuse candidates if they do not spill outside the current mask.
                        // This is sufficient because we paint using template alpha (what we can heal), not full mask.
                        if (chosen_id != 0 && !force_mint && try_id(chosen_id)) {
                            healable_now = true;
                        } else if (!force_mint) {
                            // "Distance in the other order": select the best cosine candidate that ALSO matches mask.
                            // Limit to a few candidates to avoid heavy IO.
                            struct Cand { float sim; uint32_t id; };
                            std::vector<Cand> cands;
                            cands.reserve(8);
                            for (const auto &tpl : latent_bank) {
                                if (tpl.cls != d.classId) continue;
                                if (!opt.yoloHealOnly) {
                                    if (std::abs(tpl.wh.width - safeBox.width) > 2 || std::abs(tpl.wh.height - safeBox.height) > 2) continue;
                                }
                                float sim = cosine_sim_32(emb, tpl.emb_norm);
                                cands.push_back({sim, tpl.id});
                            }
                            std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b){ return a.sim > b.sim; });
                            const size_t topK = std::min<size_t>(5, cands.size());
                            for (size_t k = 0; k < topK; ++k) {
                                if (cands[k].sim < opt.latentCosineThreshold) break;
                                if (try_id(cands[k].id)) {
                                    chosen_id = cands[k].id;
                                    healable_now = true;
                                    break;
                                }
                            }
                        }

                        // Fallback: if still not healable, reuse the most similar latent used within a short window.
                        // This increases masking ratio while keeping temporal coherence (vision persistence).
                        if (!healable_now && opt.yoloHealFallbackWindowFrames > 0) {
                            float bestSim2 = -1.0f;
                            uint32_t bestId2 = 0;
                            const int win = std::max(1, opt.yoloHealFallbackWindowFrames);
                            const float minSim = is_gop_i ? opt.yoloHealFallbackMinSimI : opt.yoloHealFallbackMinSim;
                            for (const auto &tpl : latent_bank) {
                                if (tpl.cls != d.classId) continue;
                                if (tpl.last_used_frame < 0) continue;
                                if ((frameIdx - tpl.last_used_frame) > win) continue;
                                float sim = cosine_sim_32(emb, tpl.emb_norm);
                                if (sim > bestSim2) { bestSim2 = sim; bestId2 = tpl.id; }
                            }
                            if (bestId2 != 0 && bestSim2 >= minSim) {
                                if (try_id(bestId2)) {
                                    chosen_id = bestId2;
                                    healable_now = true;
                                }
                            }
                        }
                    }

                    if (opt.latentNoMint) {
                        // Match-only mode from a prebuilt pool:
                        // - Never mint new templates, even on periodic/motion "force_mint" frames.
                        // - If we can't find a reusable id, skip this region (raw transmit).
                        force_mint = false;
                        if (chosen_id == 0) {
                            // Debug: for early frames, print best similarity so we can tell whether this is
                            // a thresholding problem (bestSim too low) or a class/bank mismatch (bestId==0).
                            if (frameIdx < 3) {
                                std::cout << "[LatentNoMint] skip frame=" << frameIdx
                                          << " det_cls=" << d.classId
                                          << " bestId=" << bestId
                                          << " bestSim=" << bestSim
                                          << " thr=" << opt.latentCosineThreshold
                                          << " bank_size=" << latent_bank.size()
                                          << std::endl;
                            }
                            yolo_heal_skipped++;
                            auto t_dict1 = std::chrono::steady_clock::now();
                            dict_ms += std::chrono::duration<double, std::milli>(t_dict1 - t_dict0).count();
                            continue;
                        }
                    }

                    if (!opt.latentNoMint && (chosen_id == 0 || force_mint)) {
                        // Mint new template_id and save RGBA template
                        chosen_id = next_template_id++;
                        is_new = true;

                        cv::Mat rgba = OfflineProcessor::extract_object_rgba(frame, d.boxMask, safeBox);
                        fs::path out_png = fs::path(opt.dictDir) / (std::to_string(chosen_id) + ".png");
                        {
                            auto t_io0 = std::chrono::steady_clock::now();
                            try { cv::imwrite(out_png.string(), rgba); } catch (...) {}
                            auto t_io1 = std::chrono::steady_clock::now();
                            latent_mint_io_ms_sum += std::chrono::duration<double, std::milli>(t_io1 - t_io0).count();
                        }

                        LatentTpl tpl;
                        tpl.id = chosen_id;
                        tpl.cls = d.classId;
                        tpl.wh = safeBox.size();
                        tpl.emb_norm = emb;
                        tpl.path = out_png.string();
                        tpl.use_count = 1;
                        tpl.minted_frame = frameIdx;
                        tpl.last_used_frame = frameIdx;
                        latent_id_to_index[tpl.id] = latent_bank.size();
                        latent_bank.push_back(std::move(tpl));
                        latent_minted++;
                        frame_latent_minted++;
                        if (force_mint) {
                            latent_forced_mints++;
                            auto itB = latent_boost_until_frame.find(d.classId);
                            if (itB != latent_boost_until_frame.end() && frameIdx < itB->second) {
                                latent_motion_boost_mints++;
                            }
                        }
                    } else {
                        matched_occurrences++;
                        latent_reused++;
                        frame_latent_reused++;
                        auto itIdx2 = latent_id_to_index.find(chosen_id);
                        if (itIdx2 != latent_id_to_index.end()) {
                            latent_bank[itIdx2->second].use_count++;
                            latent_bank[itIdx2->second].last_used_frame = frameIdx;
                        }
                    }

                    det_tplids[di] = chosen_id;
                    det_is_new[di] = is_new ? 1 : 0;
                    det_stitch_boxes[di] = stitch_box;
                    record_assignment(d.classId, safeBox, std::to_string(chosen_id), is_new);
                    if (is_new) {
                        yolo_new_tplids_this_frame.insert(chosen_id);
                    }

                    // Update last selection for this class (multi-instance)
                    {
                        auto &vec = last_latents_by_class[d.classId];
                        size_t best_idx = (size_t)-1;
                        float best_iou = 0.0f;
                        for (size_t li = 0; li < vec.size(); ++li) {
                            if (!vec[li].valid) continue;
                            float iou = iou_rect(vec[li].box, safeBox);
                            if (iou > best_iou) { best_iou = iou; best_idx = li; }
                        }
                        LastLatent ll;
                        ll.box = safeBox;
                        ll.id = chosen_id;
                        ll.emb_norm = emb;
                        ll.valid = true;
                        if (best_idx != (size_t)-1 && best_iou >= 0.3f) {
                            vec[best_idx] = std::move(ll);
                        } else {
                            vec.push_back(std::move(ll));
                            if (vec.size() > 8) vec.erase(vec.begin());
                        }
                    }

                    auto t_dict1 = std::chrono::steady_clock::now();
                    dict_ms += std::chrono::duration<double, std::milli>(t_dict1 - t_dict0).count();

                    // Build-index mode: only prefill the pool; skip painting/SEI (much faster).
                    if (opt.buildIndexOnly) {
                        continue;
                    }

                    // Mark masked pixels (union mask) only when healable (or when heal-only is disabled).
                    if (!opt.yoloHealOnly || healable_now) {
                        if (opt.yoloHealOnly) {
                            // Heal-only: union mask is template alpha (what the client can stitch).
                            auto itIdx = latent_id_to_index.find(chosen_id);
                            if (itIdx != latent_id_to_index.end()) {
                                const auto &tpl = latent_bank[itIdx->second];
                                cv::Mat base;
                                auto itC = latent_rgba_cache.find(chosen_id);
                                if (itC != latent_rgba_cache.end()) {
                                    base = itC->second;
                                } else {
                                    base = cv::imread(tpl.path, cv::IMREAD_UNCHANGED);
                                    if (!base.empty()) latent_rgba_cache.emplace(chosen_id, base);
                                }
                                if (!base.empty()) {
                                    cv::Mat img = base;
                                    if (img.cols != stitch_box.width || img.rows != stitch_box.height) {
                                        cv::resize(base, img, stitch_box.size(), 0, 0, cv::INTER_NEAREST);
                                    }
                                    cv::Rect safe = stitch_box & cv::Rect(0, 0, frame.cols, frame.rows);
                                    frame_masked_bbox_px += (uint64_t)std::max(0, safe.area());
                                    frame_masked_alpha_px += count_alpha_nonzero_in_safe(img, stitch_box, safe);
                                    if (safe.area() > 0) {
                                        cv::Mat alpha;
                                        cv::extractChannel(img, alpha, 3);
                                        int ax0 = safe.x - stitch_box.x;
                                        int ay0 = safe.y - stitch_box.y;
                                        if (ax0 >= 0 && ay0 >= 0 &&
                                            ax0 + safe.width <= alpha.cols &&
                                            ay0 + safe.height <= alpha.rows) {
                                            cv::Mat aroi = alpha(cv::Rect(ax0, ay0, safe.width, safe.height));
                                            cv::Mat dst = mask_union_u8(safe);
                                            cv::bitwise_or(dst, aroi, dst);
                                        }
                                    }
                                }
                            }
                        } else {
                            cv::Rect safe = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                            frame_masked_bbox_px += (uint64_t)std::max(0, safe.area());
                            try {
                                cv::Mat mroi = d.boxMask(safe);
                                frame_masked_alpha_px += (uint64_t)cv::countNonZero(mroi);
                                if (safe.area() > 0) {
                                    cv::Mat dst = mask_union_u8(safe);
                                    cv::bitwise_or(dst, mroi, dst);
                                }
                            } catch (...) {}
                        }

                        det_masked[di] = 1;
                        if (seiOut.is_open()) {
                            SEIRegion r{};
                            r.id = chosen_id; // template_id (stable across frames)
                            r.x = (uint32_t)std::max(0, stitch_box.x);
                            r.y = (uint32_t)std::max(0, stitch_box.y);
                            r.w = (uint32_t)std::max(0, stitch_box.width);
                            r.h = (uint32_t)std::max(0, stitch_box.height);
                            r.flags = is_new ? 0 : 1;
                            r.class_id = (uint8_t)std::max(0, std::min(255, d.classId));
                            r.path.clear(); // no client hashing; id is enough
                            sei_regions.push_back(std::move(r));
                        }
                    } else {
                        yolo_heal_skipped++;
                    }
                    continue;
                }

                // Non-latent matcher family: pHash, IoU-only, or RGB histogram.
                auto t_dict0 = std::chrono::steady_clock::now();
                uint64_t ph = OfflineProcessor::mask_phash64(maskROI);
                std::string matched;
                auto itLast = last_tpl_by_class.find(d.classId);
                if (itLast != last_tpl_by_class.end() && itLast->second.valid) {
                    float iou = iou_rect(itLast->second.box, safeBox);
                    if (iou >= 0.7f) {
                        if (yolo_matcher == "iou_only") {
                            if (processor.getDict().find(itLast->second.hash) != processor.getDict().end()) {
                                matched = itLast->second.hash;
                            }
                        } else if (yolo_matcher == "phash") {
                            int dist = OfflineProcessor::hamming64(ph, itLast->second.phash64);
                            if (dist <= opt.phashThreshold + 2 &&
                                processor.getDict().find(itLast->second.hash) != processor.getDict().end()) {
                                matched = itLast->second.hash;
                            }
                        } else if (yolo_matcher == "rgb_hist") {
                            if (processor.getDict().find(itLast->second.hash) != processor.getDict().end()) {
                                matched = itLast->second.hash;
                            }
                        }
                    }
                }
                if (matched.empty()) {
                    if (yolo_matcher == "phash") {
                        matched = find_best_phash_match(processor.getDict(), ph, d.classId, safeBox.size(), opt.phashThreshold);
                    } else if (yolo_matcher == "rgb_hist") {
                        cv::Mat roi_bgr;
                        try { roi_bgr = frame(safeBox).clone(); } catch (...) { roi_bgr.release(); }
                        if (!roi_bgr.empty()) {
                            matched = find_best_rgb_hist_match(processor.getDict(), roi_bgr, maskROI, d.classId, safeBox.size(), rgb_hist_cache);
                        }
                    }
                }
                bool is_new = false;
                std::string canon;
                if (!matched.empty()) {
                    canon = matched;
                    // bump count
                    auto &dictMut = processor.getDictMutable();
                    auto itc = dictMut.find(canon);
                    if (itc != dictMut.end()) itc->second.count++;
                } else {
                    canon = OfflineProcessor::sha256_norm(maskROI);
                    if (canon.empty()) {
                        auto t_dict1 = std::chrono::steady_clock::now();
                        dict_ms += std::chrono::duration<double, std::milli>(t_dict1 - t_dict0).count();
                        continue;
                    }
                    // Add to dictionary (exact-hash check inside)
                    is_new = processor.addToDictionary(canon, d.boxMask, safeBox, d.classId, frame);
                }
                auto t_dict1 = std::chrono::steady_clock::now();
                dict_ms += std::chrono::duration<double, std::milli>(t_dict1 - t_dict0).count();

                det_hashes[di] = canon;
                det_is_new[di] = is_new ? 1 : 0;
                if (is_new) {
                    yolo_new_hashes_this_frame.insert(canon);
                } else {
                    matched_occurrences++;
                }

                // Update last template selection for this class (even if new, use its phash for stability)
                LastTpl lt;
                lt.box = safeBox;
                lt.hash = canon;
                lt.phash64 = ph;
                lt.valid = true;
                last_tpl_by_class[d.classId] = std::move(lt);
                record_assignment(d.classId, safeBox, canon, is_new);

                // Heal-only gating (legacy pHash mode): only mask if recoverable with an existing template.
                bool healable_now = true;
                cv::Rect stitch_box = d.box;
                if (opt.yoloHealOnly) {
                    healable_now = false;
                    if (!is_new) {
                        const auto &dictRef = processor.getDict();
                        auto it = dictRef.find(canon);
                        if (it != dictRef.end()) {
                            cv::Mat tpl = cv::imread(it->second.path, cv::IMREAD_UNCHANGED);
                            if (!tpl.empty()) {
                                if (tpl.cols != d.box.width || tpl.rows != d.box.height) {
                                    cv::resize(tpl, tpl, d.box.size(), 0, 0, cv::INTER_NEAREST);
                                }
                                int dx = 0, dy = 0;
                                double extra_ratio = 1e9;
                                if (best_alpha_mask_offset(tpl, d.boxMask, d.box, /*max_shift=*/2, dx, dy, extra_ratio) &&
                                    extra_ratio <= (double)opt.yoloHealMaskExtraThr) {
                                    stitch_box = d.box + cv::Point(dx, dy);
                                    healable_now = template_alpha_matches_mask(tpl, d.boxMask, stitch_box,
                                                                              opt.yoloHealMaskIouThr,
                                                                              opt.yoloHealMaskExtraThr);
                                }
                            }
                        }
                    }
                }

                if (!opt.yoloHealOnly || healable_now) {
                    // Legacy pHash path: for fill/feather experiments we treat it like non-heal-only,
                    // using the segmentation mask union.
                    try {
                        cv::Rect safe = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe.area() > 0) {
                            cv::Mat mroi = d.boxMask(safe);
                            cv::Mat dst = mask_union_u8(safe);
                            cv::bitwise_or(dst, mroi, dst);
                        }
                    } catch (...) {}
                    det_masked[di] = 1;
                    det_stitch_boxes[di] = stitch_box;

                    if (seiOut.is_open()) {
                        SEIRegion r{};
                        r.id = (uint32_t)sei_regions.size();
                        r.x = (uint32_t)std::max(0, stitch_box.x);
                        r.y = (uint32_t)std::max(0, stitch_box.y);
                        r.w = (uint32_t)std::max(0, stitch_box.width);
                        r.h = (uint32_t)std::max(0, stitch_box.height);
                        r.flags = is_new ? 0 : 1; // 0=new, 1=reference
                        r.class_id = (uint8_t)std::max(0, std::min(255, d.classId)); // class id
                        r.path = canon; // key for stitching (canonical hash)
                        sei_regions.push_back(std::move(r));
                    }
                } else {
                    yolo_heal_skipped++;
                }
            }
        }
        if (opt.pixelMode && opt.pixelGridHeader) {
            pixel_grid_ref_regions = build_pixel_grid_groups_and_paint(
                dets, opt, frame.size(), dominant_bgr, processed, mask_union_u8,
                sei_regions, pixel_grid_groups, pixel_grid_runtime_groups);
            frame_masked_alpha_px = (uint64_t)cv::countNonZero(mask_union_u8);
            frame_masked_bbox_px = frame_masked_alpha_px;
        }
        // Apply fill + feather (YOLO mode only; pixel mode already painted above).
        if (!opt.pixelMode) {
            // Build fill frame (solid/blur/bg_ema/inpaint) using union mask for inpaint/bg_ema updates.
            bool per_region_flat_color =
                opt.maskColorLocalPerRegion &&
                opt.maskColor == "dominant" &&
                opt.maskColorLocalPadPx > 0 &&
                (opt.fillMode == "solid" || opt.fillMode.empty());
            cv::Mat fill_bgr;
            if (!per_region_flat_color) {
                fill_bgr = make_fill_frame_bgr(frame, mask_union_u8, opt, dominant_bgr, bg_ema_f32, bg_ema_inited);
            }
            if ((per_region_flat_color || !fill_bgr.empty()) && cv::countNonZero(mask_union_u8) > 0) {
                // Composite only within per-detection boxes to keep it fast.
                for (size_t di = 0; di < dets.size(); ++di) {
                    if (di < det_masked.size() && det_masked[di] == 0) continue;
                    cv::Rect box = (di < det_stitch_boxes.size() && det_stitch_boxes[di].area() > 0) ? det_stitch_boxes[di] : dets[di].box;
                    if (per_region_flat_color) {
                        cv::Vec3b region_color = dominant_color_bgr_hist_region(frame, dets[di], opt.maskColorLocalPadPx, dominant_bgr, opt.maskColorLocalStat);
                        apply_flat_color_with_feather_bbox(processed, frame, region_color, mask_union_u8, box, opt.featherPx);
                    } else {
                        apply_fill_with_feather_bbox(processed, frame, fill_bgr, mask_union_u8, box, opt.featherPx);
                    }
                }
            }
        }
        auto t_paint1 = std::chrono::steady_clock::now();
        paint_ms = std::chrono::duration<double, std::milli>(t_paint1 - t_paint0).count();
        masking_ms = paint_ms;
        // Postprocess includes YOLO postprocess + latent selection/gating (dict_ms).
        postprocess_ms += dict_ms;

        // Build-index only: do not generate masked/recovered outputs. We still want to populate
        // templates + latent bank for the next (heal-only) run.
        if (opt.buildIndexOnly) {
            ++frameIdx;
            if (frameIdx % 200 == 0) {
                std::cout << "Indexed " << frameIdx << "/" << cap.get(cv::CAP_PROP_FRAME_COUNT) << " frames." << std::endl;
            }
            continue;
        }

        // Client receives the masked stream, then reconstructs using side channel (templates/dict).
        recovered = processed.clone();

        // Pass 2: apply recovery overlays onto recovered frame (client reconstruction)
        // IMPORTANT: if a region was not masked on the server, the client should not waste
        // work stitching it (and it could even introduce artifacts).
        auto t_rec0 = std::chrono::steady_clock::now();
        if (opt.pixelMode && opt.pixelGridHeader) {
            for (const auto &group : pixel_grid_runtime_groups) {
                cv::Mat tpl_rgba = get_pixel_template_rgba_by_path(group.meta.path);
                if (tpl_rgba.empty()) continue;
                for (const auto &cellBox : group.cellBoxes) {
                    overlay_template_rgba(recovered, tpl_rgba, cellBox);
                    matched_occurrences++;
                }
            }
            for (const auto &r : sei_regions) {
                if (r.path.empty()) continue;
                cv::Mat tpl_rgba = get_pixel_template_rgba_by_path(r.path);
                if (tpl_rgba.empty()) continue;
                cv::Rect box((int)r.x, (int)r.y, (int)r.w, (int)r.h);
                if (tpl_rgba.cols != box.width || tpl_rgba.rows != box.height) {
                    cv::resize(tpl_rgba, tpl_rgba, box.size(), 0, 0, cv::INTER_NEAREST);
                }
                overlay_template_rgba(recovered, tpl_rgba, box);
                matched_occurrences++;
            }
        }
        for (size_t di = 0; di < dets.size(); ++di) {
            const auto &d = dets[di];
            if (d.confidence < opt.confThreshold) continue;
            if (d.boxMask.empty()) continue;
            if (opt.pixelMode && opt.pixelGridHeader) continue;
            if (!opt.pixelMode) {
                if (di < det_masked.size() && det_masked[di] == 0) continue;
            }

            if (opt.pixelMode) {
                cv::Mat tpl_rgba = get_pixel_template_rgba_for_detection(d);
                if (!tpl_rgba.empty()) {
                    overlay_template_rgba(recovered, tpl_rgba, d.box);
                    matched_occurrences++; // treat as matched since we can always recover from template set
                }
            } else {
                if (yolo_matcher == "latent_key") {
                    uint32_t tid = det_tplids[di];
                    if (!tid) continue;
                    bool is_new_for_client = det_is_new[di] || (yolo_new_tplids_this_frame.find(tid) != yolo_new_tplids_this_frame.end());

                    uint32_t use_id = tid;
                    if (is_new_for_client) {
                        // Fallback to last recoverable template for this class if bbox overlaps enough.
                        auto itVec = last_client_tpls_by_class.find(d.classId);
                        if (itVec != last_client_tpls_by_class.end()) {
                            float best_iou = 0.0f;
                            uint32_t best_id = 0;
                            for (const auto &lc : itVec->second) {
                                if (!lc.valid) continue;
                                float iou = iou_rect(lc.box, d.box);
                                if (iou > best_iou) { best_iou = iou; best_id = lc.id; }
                            }
                            if (best_id != 0 && best_iou >= 0.5f) {
                                use_id = best_id;
                                is_new_for_client = false; // recoverable via fallback
                            } else {
                                continue; // no good fallback, keep masked pixels
                            }
                        } else {
                            continue; // no cached fallback
                        }
                    }

                    auto itIdx = latent_id_to_index.find(use_id);
                    if (itIdx == latent_id_to_index.end()) continue;
                    const auto &tpl = latent_bank[itIdx->second];
                    cv::Mat img = cv::imread(tpl.path, cv::IMREAD_UNCHANGED);
                    if (!img.empty()) {
                        cv::Rect box = (di < det_stitch_boxes.size() && det_stitch_boxes[di].area() > 0) ? det_stitch_boxes[di] : d.box;
                        if (img.cols != box.width || img.rows != box.height) {
                            cv::resize(img, img, box.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        overlay_template_rgba(recovered, img, box);

                        // Update last client template for this class (recoverable/cached)
                        auto &vec = last_client_tpls_by_class[d.classId];
                        size_t best_idx = (size_t)-1;
                        float best_iou = 0.0f;
                        for (size_t li = 0; li < vec.size(); ++li) {
                            if (!vec[li].valid) continue;
                            float iou = iou_rect(vec[li].box, box);
                            if (iou > best_iou) { best_iou = iou; best_idx = li; }
                        }
                        LastClientTpl lc;
                        lc.box = box;
                        lc.id = use_id;
                        lc.valid = true;
                        if (best_idx != (size_t)-1 && best_iou >= 0.3f) {
                            vec[best_idx] = std::move(lc);
                        } else {
                            vec.push_back(std::move(lc));
                            if (vec.size() > 8) vec.erase(vec.begin());
                        }
                    }
                    continue;
                }
                const std::string &canon = det_hashes[di];
                if (canon.empty()) continue;
                // Only recover if this object is NOT new in this frame (client already has template).
                if (det_is_new[di] || yolo_new_hashes_this_frame.find(canon) != yolo_new_hashes_this_frame.end()) {
                    continue;
                }
                const auto &dictRef = processor.getDict();
                auto it = dictRef.find(canon);
                if (it != dictRef.end()) {
                    cv::Mat tpl = cv::imread(it->second.path, cv::IMREAD_UNCHANGED);
                    if (!tpl.empty()) {
                        cv::Rect box = (di < det_stitch_boxes.size() && det_stitch_boxes[di].area() > 0) ? det_stitch_boxes[di] : d.box;
                        if (tpl.cols != box.width || tpl.rows != box.height) {
                            cv::resize(tpl, tpl, box.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        overlay_template_rgba(recovered, tpl, box);
                    }
                }
            }
        }
        auto t_rec1 = std::chrono::steady_clock::now();
        recover_ms = std::chrono::duration<double, std::milli>(t_rec1 - t_rec0).count();

        // Determine ref/raw status and "successfully masked" count.
        int ref_regions = 0;
        if (opt.pixelMode) {
            ref_regions = opt.pixelGridHeader ? pixel_grid_ref_regions : (int)sei_regions.size();
        } else {
            // In YOLO mode, sei_regions contains only actually-masked regions (except force-mask-all).
            if (!sei_regions.empty()) {
                ref_regions = (int)sei_regions.size();
            } else if (!det_masked.empty()) {
                for (uint8_t v : det_masked) if (v) ref_regions++;
            }
        }
        int frame_flags_raw_i = ref_regions > 0 ? 1 : 0; // 0=raw, 1=ref
        int frame_flags_i = frame_flags_raw_i;
        if (opt.frameModeHysteresis) {
            if (!frame_mode_state_inited) {
                frame_mode_state_inited = true;
                frame_mode_state = frame_flags_raw_i;
                frame_mode_candidate = frame_mode_state;
                frame_mode_candidate_count = 0;
            } else if (frame_flags_raw_i == frame_mode_state) {
                frame_mode_candidate = frame_mode_state;
                frame_mode_candidate_count = 0;
            } else {
                if (frame_flags_raw_i != frame_mode_candidate) {
                    frame_mode_candidate = frame_flags_raw_i;
                    frame_mode_candidate_count = 1;
                } else {
                    frame_mode_candidate_count++;
                }
                if (frame_mode_candidate_count >= std::max(1, opt.frameModeHysteresisConfirmFrames)) {
                    frame_mode_state = frame_mode_candidate;
                    frame_mode_candidate_count = 0;
                }
            }
            frame_flags_i = frame_mode_state;
        }
        cv::Mat emittedFrame = (frame_flags_i != 0) ? processed : frame;
        cv::Mat deliveredRecovered = (frame_flags_i != 0) ? recovered : frame;
        if (frame_flags_i == 0) {
            ref_regions = 0;
            sei_regions.clear();
            pixel_grid_groups.clear();
            pixel_grid_runtime_groups.clear();
            pixel_grid_ref_regions = 0;
            frame_masked_bbox_px = 0;
            frame_masked_alpha_px = 0;
            mask_union_u8.setTo(cv::Scalar(0));
        }

        if (seiOut.is_open()) {
            auto t_sei0 = std::chrono::steady_clock::now();
            uint64_t pts = (uint64_t)frameIdx;
            uint8_t frame_flags = (uint8_t)frame_flags_i;
            uint16_t msk1_version = (opt.pixelMode && opt.pixelGridHeader) ? 5 : 4;
            std::vector<uint8_t> payload = build_msk1_payload((uint64_t)frameIdx, pts, sei_regions, msk1_version, frame_flags, pixel_grid_groups);
            uint32_t len = (uint32_t)payload.size();
            seiOut.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len) seiOut.write(reinterpret_cast<const char*>(payload.data()), len);
            auto t_sei1 = std::chrono::steady_clock::now();
            sei_build_ms = std::chrono::duration<double, std::milli>(t_sei1 - t_sei0).count();
        }

        // Record quality metrics between baseline and masked frame
        if (frame.size() == emittedFrame.size()) {
            double ssim = computeSSIM(frame, emittedFrame);
            double psnr = computePSNR(frame, emittedFrame);
            ssim_sum += ssim;
            psnr_sum += psnr;
            metric_frames++;
            frame_indices.push_back(frameIdx);
            ssim_values.push_back(ssim);
            psnr_values.push_back(psnr);

            // Pixel-specific breakdown: always record so CSV is meaningful even if --timing is omitted.
            t_pixel_bootstrap_ms.push_back(opt.pixelMode ? pixel_bootstrap_ms : 0.0);
            t_pixel_roi_ms.push_back(opt.pixelMode ? pixel_roi_ms : 0.0);
            t_pixel_scanned_bootstrap.push_back(opt.pixelMode ? pixel_scanned_bootstrap : 0);
            t_pixel_scanned_roi.push_back(opt.pixelMode ? pixel_scanned_roi : 0);
            t_pixel_band_ms.push_back(opt.pixelMode ? pixel_band_ms : 0.0);
            t_pixel_scanned_band.push_back(opt.pixelMode ? pixel_scanned_band : 0);
            t_pixel_row_ms.push_back(opt.pixelMode ? pixel_row_ms : 0.0);
            t_pixel_flow_ms.push_back(opt.pixelMode ? pixel_flow_ms : 0.0);
            t_pixel_bands_used.push_back(opt.pixelMode ? pixel_bands_used : 0);

            if (opt.pixelMode) {
                std::unordered_map<std::string, int> cnt;
                for (const auto &d : dets) {
                    if (d.confidence < opt.confThreshold) continue;
                    if (d.seiPath.empty()) continue;
                    cnt[d.seiPath] += 1;
                }
                json m = json::object();
                for (const auto &kv : cnt) m[kv.first] = kv.second;
                t_pixel_template_counts.push_back(std::move(m));
            } else {
                t_pixel_template_counts.push_back(json::object());
            }
        }

        // Record quality metrics between baseline and recovered frame
        if (frame.size() == deliveredRecovered.size()) {
            double ssim_r = computeSSIM(frame, deliveredRecovered);
            double psnr_r = computePSNR(frame, deliveredRecovered);
            rec_ssim_sum += ssim_r;
            rec_psnr_sum += psnr_r;
            rec_metric_frames++;
            rec_ssim_values.push_back(ssim_r);
            rec_psnr_values.push_back(psnr_r);
        }

        // Encode proxy times (stdin write time + SEI build time on masked stream).
        if (origWriter.isOpen()) {
            auto t0 = std::chrono::steady_clock::now();
            (void)origWriter.write(frame);
            auto t1 = std::chrono::steady_clock::now();
            encode_baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double masked_write_ms = 0.0;
        if (writer.isOpen()) {
            auto t0 = std::chrono::steady_clock::now();
            (void)writer.write(emittedFrame);
            auto t1 = std::chrono::steady_clock::now();
            masked_write_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        encode_masked_ms = masked_write_ms + sei_build_ms;
        if (recWriter.isOpen()) (void)recWriter.write(deliveredRecovered);

        auto t_frame_end = std::chrono::steady_clock::now();
        frame_total_ms = std::chrono::duration<double, std::milli>(t_frame_end - t_frame_start).count();

        // Per-frame CSV fields (always recorded into report.json/per_frame)
        // Model object count: detections passing conf threshold with a non-empty mask.
        int model_cnt = 0;
        for (const auto &d : dets) {
            if (d.confidence < opt.confThreshold) continue;
            if (d.boxMask.empty()) continue;
            model_cnt++;
        }
        t_object_present_model.push_back(model_cnt);
        t_object_successfully_masked.push_back(ref_regions);
        t_frame_flags.push_back(frame_flags_i);
        t_frame_flags_raw.push_back(frame_flags_raw_i);
        t_latent_minted_regions.push_back(frame_latent_minted);
        t_latent_reused_regions.push_back(frame_latent_reused);
        double frame_px = (double)std::max<int64_t>(1, (int64_t)frame.cols * (int64_t)frame.rows);
        t_masked_bbox_pct.push_back(100.0 * (double)frame_masked_bbox_px / frame_px);
        t_masked_alpha_pct.push_back(100.0 * (double)frame_masked_alpha_px / frame_px);
        int changed = 0;
        // Approximation: changed pixels count based on union mask (feather may reduce actual changes slightly).
        try { changed = cv::countNonZero(mask_union_u8); } catch (...) { changed = 0; }
        t_changed_pixels.push_back(changed);
        t_changed_pixels_pct.push_back(100.0 * (double)changed / frame_px);

        // Cross-game timing breakdown
        if (opt.pixelMode) {
            preprocess_ms = 0.0;
            inference_ms = infer_ms;
            postprocess_ms = 0.0;
        } else {
            if (preprocess_ms <= 0.0 && inference_ms <= 0.0 && postprocess_ms <= 0.0) {
                inference_ms = infer_ms;
            }
        }
        t_preprocess_ms.push_back(preprocess_ms);
        t_inference_ms.push_back(inference_ms);
        t_postprocess_ms.push_back(postprocess_ms);
        t_masking_ms.push_back(masking_ms);
        t_encode_baseline_ms.push_back(encode_baseline_ms);
        t_encode_masked_ms.push_back(encode_masked_ms);
        t_stitching_ms.push_back(recover_ms);

        // Pixel counters:
        // - ROI filter (Kalman): kalman predict+correct time inside roi_match()
        // - Template matching: bootstrap + ROI match (minus kalman) + row band selection + band scan
        // - Motion filter: optical flow stabilization
        double pixel_template_matching_ms = 0.0;
        if (opt.pixelMode) {
            pixel_template_matching_ms =
                pixel_bootstrap_ms +
                std::max(0.0, pixel_roi_ms - pixel_kalman_ms) +
                pixel_row_ms + pixel_band_ms;
        }

        if (opt.recordTiming) {
            t_infer_ms.push_back(infer_ms);
            t_dict_ms.push_back(dict_ms);
            t_paint_ms.push_back(paint_ms);
            t_recover_ms.push_back(recover_ms);
            t_total_ms.push_back(frame_total_ms);
        }
        if (opt.dumpFrameCache && originalFrameCache.is_open() && maskedFrameCache.is_open()) {
            cv::Mat frameWrite = frame.isContinuous() ? frame : frame.clone();
            cv::Mat processedWrite = emittedFrame.isContinuous() ? emittedFrame : emittedFrame.clone();
            const size_t frameBytes = (size_t)width * (size_t)height * 3;
            originalFrameCache.write(reinterpret_cast<const char*>(frameWrite.data), (std::streamsize)frameBytes);
            maskedFrameCache.write(reinterpret_cast<const char*>(processedWrite.data), (std::streamsize)frameBytes);
        }

        ++frameIdx;
        if (frameIdx % 200 == 0) {
            std::cout << "Processed " << frameIdx << "/" << cap.get(cv::CAP_PROP_FRAME_COUNT) << " frames." << std::endl;
        }
    }

    // Save updated dictionary index only for traditional (YOLO) mode
    if (!opt.pixelMode) {
        processor.saveDictionary();
        std::cout << "Dictionary size: " << processor.getDict().size() << std::endl;
    }

    // Save metrics report for Python visualization
    double avg_ssim = metric_frames ? (ssim_sum / metric_frames) : 0.0;
    double avg_psnr = metric_frames ? (psnr_sum / metric_frames) : 0.0;
    double avg_rec_ssim = rec_metric_frames ? (rec_ssim_sum / rec_metric_frames) : 0.0;
    double avg_rec_psnr = rec_metric_frames ? (rec_psnr_sum / rec_metric_frames) : 0.0;
    RunLengthStats frame_mode_raw_stats = compute_run_length_stats(t_frame_flags_raw);
    RunLengthStats frame_mode_final_stats = compute_run_length_stats(t_frame_flags);

    json report;
    if (!opt.commandline.empty()) {
        report["commandline"] = opt.commandline;
    }
    report["input_ffprobe"] = ffprobe_input_summary_json(opt.source);

    report["encoder_settings"] = {
        {"use_ffmpeg_encoder", opt.useFfmpegEncoder},
        {"enc_codec", opt.encCodec},
        {"enc_crf", opt.encCrf},
        {"enc_bitrate_mbps", opt.encBitrateMbps},
        {"enc_maxrate_mbps", opt.encMaxrateMbps},
        {"enc_bufsize_mbits", opt.encBufsizeMbits},
        {"enc_gop", opt.encGop},
        {"enc_bframes", opt.encBFrames},
        {"enc_scenecut", opt.encNoScenecut ? 0 : 1},
        {"enc_open_gop_defaults", opt.encOpenGopDefaults},
        {"enc_preset", opt.encPreset},
        {"enc_tune", opt.encTune},
        {"enc_profile", opt.encProfile},
        {"enc_level", opt.encLevel},
        {"enc_aud", opt.encAud ? 1 : 0},
        {"enc_repeat_headers", opt.encRepeatHeaders ? 1 : 0}
    };
    // Record the exact ffmpeg invocations that produced the MP4s.
    report["ffmpeg_commands"] = {
        {"baseline", origWriter.cmd},
        {"masked", writer.cmd},
        {"recovered", recWriter.cmd},
        {"baseline_stderr_log", origWriter.stderrLogPath},
        {"masked_stderr_log", writer.stderrLogPath},
        {"recovered_stderr_log", recWriter.stderrLogPath}
    };
    report["outputs"] = {
        {"original_output_mp4", originalOut},
        {"segmented_output_mp4", stitchedOut},
        {"recovered_output_mp4", recoveredOut},
        {"msk1_payloads_bin", seiOutPath},
        {"dict_dir", opt.dictDir}
    };
    if (opt.dumpFrameCache) {
        report["outputs"]["frame_cache_dir"] = frameCacheDir;
        report["outputs"]["original_frame_cache_bgr"] = originalFrameCachePath;
        report["outputs"]["masked_frame_cache_bgr"] = maskedFrameCachePath;
    }
    report["metric_definitions"] = {
        {"rate_codec_sweep_saving_pct", "Computed from whole-file bitrate: (mp4_size_bits/duration) and includes meta_bps from msk1_payloads.bin. See tools/metrics/codec_sweep_analyze.py"},
        {"rate_per_frame_bytes", "Computed from ffprobe -show_packets packet sizes, aligned by index or pts_time. See tools/metrics/build_per_frame_csv.py and tools/metrics/compare_mp4_packets.py"}
    };
    report["total_frames"] = frameIdx;
    report["total_detections"] = total_occurrences;
    report["matched_detections"] = matched_occurrences;
    report["avg_ssim"] = avg_ssim;
    report["avg_psnr"] = avg_psnr;
    report["avg_recovered_ssim"] = avg_rec_ssim;
    report["avg_recovered_psnr"] = avg_rec_psnr;
    report["yolo_matcher"] = yolo_matcher;
    report["matcher_new_templates"] = matcher_new_templates;
    report["matcher_id_switches"] = matcher_id_switches;
    report["timing_enabled"] = opt.recordTiming;
    report["frame_mode_hysteresis"] = {
        {"enabled", opt.frameModeHysteresis},
        {"confirm_frames", opt.frameModeHysteresisConfirmFrames},
        {"pre_smoothing", {
            {"switch_count", frame_mode_raw_stats.switchCount},
            {"fraction_mode_switches", frame_mode_raw_stats.fractionModeSwitches},
            {"mean_run_length_ref_raw_states", frame_mode_raw_stats.meanRunLength},
            {"mean_masked_run_length", frame_mode_raw_stats.meanMaskedRunLength},
            {"mean_raw_run_length", frame_mode_raw_stats.meanRawRunLength},
            {"max_masked_run_length", frame_mode_raw_stats.maxMaskedRunLength},
            {"max_raw_run_length", frame_mode_raw_stats.maxRawRunLength}
        }},
        {"post_smoothing", {
            {"switch_count", frame_mode_final_stats.switchCount},
            {"fraction_mode_switches", frame_mode_final_stats.fractionModeSwitches},
            {"mean_run_length_ref_raw_states", frame_mode_final_stats.meanRunLength},
            {"mean_masked_run_length", frame_mode_final_stats.meanMaskedRunLength},
            {"mean_raw_run_length", frame_mode_final_stats.meanRawRunLength},
            {"max_masked_run_length", frame_mode_final_stats.maxMaskedRunLength},
            {"max_raw_run_length", frame_mode_final_stats.maxRawRunLength}
        }}
    };
    report["latent_key_enabled"] = (yolo_matcher == "latent_key");
    report["yolo_heal_only"] = opt.yoloHealOnly;
    report["yolo_heal_skipped_regions"] = yolo_heal_skipped;
    report["yolo_force_mask_all"] = opt.yoloForceMaskAll;
    report["yolo_forced_masked_regions"] = yolo_forced_masked;
    if (yolo_matcher == "latent_key") {
        report["latent_bank_size"] = (uint64_t)latent_bank.size();
        report["latent_cosine_threshold"] = opt.latentCosineThreshold;
        report["latent_sample_period"] = opt.latentSamplePeriod;
        report["latent_motion_iou_thr"] = opt.latentMotionIouThr;
        report["latent_motion_center_px"] = opt.latentMotionCenterPx;
        report["latent_motion_scale_thr"] = opt.latentMotionScaleThr;
        report["latent_motion_boost_frames"] = opt.latentMotionBoostFrames;

        report["latent_minted"] = latent_minted;
        report["latent_reused"] = latent_reused;
        report["latent_forced_mints"] = latent_forced_mints;
        report["latent_motion_boost_mints"] = latent_motion_boost_mints;
        report["latent_periodic_skipped"] = latent_periodic_skipped;
        report["latent_motion_boost_events"] = latent_motion_boost_events;
        report["latent_period_skip_thr"] = opt.latentPeriodSkipThr;
        report["latent_merge_thr"] = opt.latentMergeThr;
        report["latent_merged_mints"] = latent_merged_mints;
        report["latent_reuse_ratio"] = (latent_minted + latent_reused) ? (double)latent_reused / (double)(latent_minted + latent_reused) : 0.0;
        report["latent_avg_search_ms_per_det"] = (double)(latent_search_ms_sum / std::max<uint64_t>(1, total_occurrences));
        report["latent_avg_mint_io_ms_per_mint"] = (double)(latent_mint_io_ms_sum / std::max<uint64_t>(1, latent_minted));

        if (!latent_reuse_cos_sims.empty()) {
            std::sort(latent_reuse_cos_sims.begin(), latent_reuse_cos_sims.end());
            auto q = [&](double p){
                size_t n = latent_reuse_cos_sims.size();
                size_t i = (size_t)std::min<double>(n - 1, std::max<double>(0.0, p * (n - 1)));
                return (double)latent_reuse_cos_sims[i];
            };
            report["latent_reuse_cosine_quantiles"] = {
                {"p01", q(0.01)}, {"p05", q(0.05)}, {"p10", q(0.10)},
                {"p50", q(0.50)}, {"p90", q(0.90)}, {"p95", q(0.95)}, {"p99", q(0.99)}
            };
        }

        // Dump full latent bank (for heatmaps/clustering or for re-use in heal-only runs)
        try {
            json bank = json::array();
            for (const auto &t : latent_bank) {
                json e = json::array();
                for (int i = 0; i < 32; ++i) e.push_back(t.emb_norm[(size_t)i]);
                bank.push_back({
                    {"id", t.id},
                    {"cls", t.cls},
                    {"w", t.wh.width},
                    {"h", t.wh.height},
                    {"use_count", t.use_count},
                    {"path", t.path},
                    {"minted_frame", t.minted_frame},
                    {"last_used_frame", t.last_used_frame},
                    {"emb", e}
                });
            }
            fs::path outPath = fs::path(opt.outDir) / "latent_bank.json";
            // In build-index mode, also write into dictDir so later runs can load from there by default.
            if (opt.buildIndexOnly) {
                outPath = fs::path(opt.dictDir) / "latent_bank.json";
            }
            std::ofstream bf(outPath);
            bf << bank.dump(2);
            report["latent_bank_path"] = outPath.string();
        } catch (...) {
            // ignore
        }
    }
    report["latents_dump_enabled"] = opt.dumpLatents;
    if (opt.dumpLatents) {
        report["latents_dump_path"] = opt.latentsOutPath.empty()
            ? (fs::path(opt.outDir) / "latents.jsonl").string()
            : opt.latentsOutPath;
    }
    if (opt.recordTiming && !t_total_ms.empty()) {
        auto avg = [](const std::vector<double> &v){
            double s = 0.0;
            for (double x : v) s += x;
            return v.empty() ? 0.0 : (s / (double)v.size());
        };
        report["timing_avg_ms"] = {
            {"infer",   avg(t_infer_ms)},
            {"dict",    avg(t_dict_ms)},
            {"paint",   avg(t_paint_ms)},
            {"recover", avg(t_recover_ms)},
            {"total",   avg(t_total_ms)}
        };
    }

    json per_frame = json::array();
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        json item = {
            {"frame", frame_indices[i]},
            {"ssim",  ssim_values[i]},
            {"psnr",  psnr_values[i]}
        };
        if (i < rec_ssim_values.size()) {
            item["rec_ssim"] = rec_ssim_values[i];
        }
        if (i < rec_psnr_values.size()) {
            item["rec_psnr"] = rec_psnr_values[i];
        }
        if (opt.recordTiming && i < t_total_ms.size()) {
            item["infer_ms"] = t_infer_ms[i];
            item["dict_ms"] = t_dict_ms[i];
            item["paint_ms"] = t_paint_ms[i];
            item["recover_ms"] = t_recover_ms[i];
            item["total_ms"] = t_total_ms[i];
            if (i < t_pixel_bootstrap_ms.size()) {
                item["pixel_bootstrap_ms"] = t_pixel_bootstrap_ms[i];
                item["pixel_roi_ms"] = t_pixel_roi_ms[i];
                item["pixel_scanned_bootstrap"] = t_pixel_scanned_bootstrap[i];
                item["pixel_scanned_roi"] = t_pixel_scanned_roi[i];
                if (i < t_pixel_band_ms.size()) {
                    item["pixel_band_ms"] = t_pixel_band_ms[i];
                    item["pixel_scanned_band"] = t_pixel_scanned_band[i];
                    item["pixel_row_ms"] = t_pixel_row_ms[i];
                    item["pixel_flow_ms"] = t_pixel_flow_ms[i];
                    item["pixel_bands_used"] = t_pixel_bands_used[i];
                }
                if (i < t_pixel_template_counts.size()) {
                    item["pixel_template_counts"] = t_pixel_template_counts[i];
                }
            }
        }
        // Per-frame CSV fields (always emitted)
        if (i < t_object_present_model.size()) {
            item["object_present_model"] = t_object_present_model[i];
            item["object_successfully_masked"] = t_object_successfully_masked[i];
            item["frame_flags_raw"] = t_frame_flags_raw[i];
            item["frame_flags"] = t_frame_flags[i];
        }
        if (i < t_latent_minted_regions.size()) {
            item["latent_minted_regions"] = t_latent_minted_regions[i];
            item["latent_reused_regions"] = t_latent_reused_regions[i];
            item["masked_bbox_pct"] = t_masked_bbox_pct[i];
            item["masked_alpha_pct"] = t_masked_alpha_pct[i];
        }
        if (i < t_changed_pixels.size()) {
            item["changed_pixels"] = t_changed_pixels[i];
            item["changed_pixels_pct"] = t_changed_pixels_pct[i];
        }
        if (i < t_preprocess_ms.size()) {
            item["preprocess_ms"] = t_preprocess_ms[i];
            item["inference_ms"] = t_inference_ms[i];
            item["postprocess_ms"] = t_postprocess_ms[i];
            item["masking_ms"] = t_masking_ms[i];
            item["encode_baseline_ms"] = t_encode_baseline_ms[i];
            item["encode_masked_ms"] = t_encode_masked_ms[i];
            item["stitching_ms"] = t_stitching_ms[i];
        }
        // Pixel-mode timing keys (align with requested CSV schema)
        if (i < t_pixel_roi_ms.size()) {
            // ROI filter (Kalman), template matching, motion filter (flow)
            // Note: pixel_kalman_ms is recorded as part of the frame loop (not stored in a dedicated vector),
            // so we expose it via the derived fields written by the CSV builder using pixel_* timers.
            item["pixel_bootstrap_ms"] = t_pixel_bootstrap_ms[i];
            item["pixel_roi_ms"] = t_pixel_roi_ms[i];
            item["pixel_row_ms"] = t_pixel_row_ms[i];
            item["pixel_band_ms"] = t_pixel_band_ms[i];
            item["pixel_flow_ms"] = t_pixel_flow_ms[i];
        }
        per_frame.push_back(item);
    }
    report["per_frame"] = per_frame;

    std::ofstream rf(reportPath);
    rf << report.dump(2);
    std::cout << "Saved evaluation report to " << reportPath << std::endl;

    writer.close();
    origWriter.close();
    recWriter.close();
    cap.release();

    return 0;
}

