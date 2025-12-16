#include "offline_processor.h"
#include <fstream>
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

namespace fs = std::filesystem;
using json = nlohmann::json;

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

// Calibrate best scale per template using the first frame's edges.
// We test multiple candidate scales and keep the one with the highest max NCC score.
static void calibrate_template_scales(const cv::Mat &frame_edges) {
    if (g_pixel_scales_calibrated) return;
    if (g_pixel_templates.empty()) return;

    // Finer-grained candidate scales; only used on the first frame.
    const std::vector<double> candidate_scales = {
        0.6, 0.7, 0.8, 0.9,
        1.0,
        1.1, 1.2, 1.3, 1.4
    };

    g_pixel_template_scales.assign(g_pixel_templates.size(), 1.0);

    for (size_t ti = 0; ti < g_pixel_templates.size(); ++ti) {
        const auto &tmpl = g_pixel_templates[ti];
        if (tmpl.edges.empty()) continue;

        double best_scale = 1.0;
        double best_score = -1.0;

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

        g_pixel_template_scales[ti] = best_scale;
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
        calibrate_template_scales(frame_edges);
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
    bool bootstrap(const cv::Mat &frame_edges, int frameIdx, const OfflineOptions &opt,
                   double &bootstrap_ms, int &scanned) {
        auto t0 = std::chrono::steady_clock::now();
        scanned = 0;
        activeTemplates.clear();

        struct Cand { float score; int ti; cv::Point pt; cv::Size wh; };
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
            cands.push_back(Cand{(float)maxV, ti, maxP, cv::Size(tw, th)});
        }

        std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b){ return a.score > b.score; });
        int topK = std::max(1, opt.pixelTopK);
        if ((int)cands.size() > topK) cands.resize(topK);
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

    bool roi_match(const cv::Mat &frame_edges, int frameIdx, const OfflineOptions &opt,
                   cv::Rect &outBox, float &outScore, int &outTemplateIdx,
                   double &roi_ms, int &scanned) {
        auto t0 = std::chrono::steady_clock::now();
        scanned = 0;

        if (!inited || lastBox.area() <= 0) {
            auto t1 = std::chrono::steady_clock::now();
            roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        cv::Mat pred = kf.predict();
        float pcx = pred.at<float>(0);
        float pcy = pred.at<float>(1);
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

        float bestScore = -1.0f;
        cv::Point bestPt(0,0);
        cv::Size bestWH(0,0);
        int bestTi = -1;

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
            if ((float)maxV > bestScore) {
                bestScore = (float)maxV;
                bestPt = maxP;
                bestWH = cv::Size(tw, th);
                bestTi = ti;
            }
        }

        outScore = bestScore;
        outTemplateIdx = bestTi;
        if (bestTi < 0) {
            auto t1 = std::chrono::steady_clock::now();
            roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return false;
        }

        cv::Rect b(roi.x + bestPt.x, roi.y + bestPt.y, bestWH.width, bestWH.height);
        b = clamp_rect(b, frame_edges.size());
        outBox = b;

        // Kalman update
        cv::Mat meas(2, 1, CV_32F);
        meas.at<float>(0) = b.x + b.width * 0.5f;
        meas.at<float>(1) = b.y + b.height * 0.5f;
        kf.correct(meas);

        lastBox = b;
        lastScore = bestScore;

        auto t1 = std::chrono::steady_clock::now();
        roi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
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
    
    for (int y = top; y < bottom; ++y) {
        const uchar *mrow = mask.ptr<uchar>(y);
        cv::Vec3b *drow = dst_bgr.ptr<cv::Vec3b>(y);
        for (int x = left; x < right; ++x) {
            if (mrow[x] == 0) continue;
            // Solid fill: overwrite pixel with constant color (no blending)
            cv::Vec3b &dpx = drow[x];
            dpx[0] = color[0];
            dpx[1] = color[1];
            dpx[2] = color[2];
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

int run_offline_evaluation(const OfflineOptions &opt, YOLO_V8& yoloDetector) {
    fs::create_directories(opt.outDir);
    if (!opt.pixelMode) {
        fs::create_directories(opt.dictDir);
    }

    // For pixelMode we pass an empty dictDir to avoid creating / writing dictionaries.
    OfflineProcessor processor(opt.outDir, opt.pixelMode ? "" : opt.dictDir);
    
    // Load existing dictionary only for traditional (YOLO) mode
    if (!opt.pixelMode && !opt.yoloLatentKey) {
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

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1'); // H.264 codec
    writer.open(stitchedOut, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Warning: could not open stitched output video: " << stitchedOut << std::endl;
    }

    cv::VideoWriter origWriter;
    origWriter.open(originalOut, fourcc, fps, cv::Size(width, height));
    if (!origWriter.isOpened()) {
        std::cerr << "Warning: could not open original output video: " << originalOut << std::endl;
    }

    // Recovered (client-like) video
    std::string recoveredOut = opt.outDir + "/recovered_output.mp4";
    cv::VideoWriter recWriter;
    recWriter.open(recoveredOut, fourcc, fps, cv::Size(width, height));
    if (!recWriter.isOpened()) {
        std::cerr << "Warning: could not open recovered output video: " << recoveredOut << std::endl;
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
    std::vector<double> t_pixel_bootstrap_ms, t_pixel_roi_ms;
    std::vector<int> t_pixel_scanned_bootstrap, t_pixel_scanned_roi;

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

    // Latent-key offline simulator state (YOLO only): template_id -> embedding/path
    uint32_t next_template_id = 1;
    struct LatentTpl {
        uint32_t id{0};
        int cls{0};
        cv::Size wh{};
        std::array<float, 32> emb_norm{};
        std::string path; // saved RGBA template path
        uint64_t use_count{0}; // how many times selected for this run (server-side)
    };
    std::vector<LatentTpl> latent_bank;
    std::unordered_map<uint32_t, size_t> latent_id_to_index;
    struct LastLatent {
        cv::Rect box;
        uint32_t id{0};
        std::array<float, 32> emb_norm{};
        bool valid{false};
    };
    std::unordered_map<int, LastLatent> last_latent_by_class;
    std::unordered_map<int, int> latent_boost_until_frame; // per class

    // Client-side fallback: if a template_id is "new" this frame (not yet delivered),
    // reuse the last recoverable template for the same class to avoid flashing.
    struct LastClientTpl {
        cv::Rect box;
        uint32_t id{0};
        bool valid{false};
    };
    std::unordered_map<int, LastClientTpl> last_client_tpl_by_class;

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
    
    int frameIdx = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (opt.maxFrames > 0 && frameIdx >= opt.maxFrames) break;
        cv::Mat processed = frame.clone();  // masked view (what server sends)
        cv::Mat recovered;                 // reconstructed client view from masked + side-channel
        double infer_ms = 0.0, paint_ms = 0.0, recover_ms = 0.0, dict_ms = 0.0, frame_total_ms = 0.0;
        auto t_frame_start = std::chrono::steady_clock::now();

        std::vector<DL_RESULT> dets;
        double pixel_bootstrap_ms = 0.0;
        double pixel_roi_ms = 0.0;
        int pixel_scanned_bootstrap = 0;
        int pixel_scanned_roi = 0;
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
                        calibrate_template_scales(frame_edges);
                    }

                    bool need_bootstrap = (!pixelTracker.inited) ||
                        (opt.pixelBootstrapInterval > 0 && (frameIdx - pixelTracker.lastBootstrapFrame) >= opt.pixelBootstrapInterval) ||
                        (pixelTracker.lostCount >= std::max(1, opt.pixelLostMax));

                    if (need_bootstrap) {
                        bool ok = pixelTracker.bootstrap(frame_edges, frameIdx, opt, pixel_bootstrap_ms, pixel_scanned_bootstrap);
                        (void)ok;
                    }

                    cv::Rect box;
                    float score = 0.0f;
                    int bestTi = -1;
                    bool ok2 = pixelTracker.roi_match(frame_edges, frameIdx, opt, box, score, bestTi, pixel_roi_ms, pixel_scanned_roi);
                    if (ok2 && score >= opt.pixelMinScore && box.area() > 0 && bestTi >= 0) {
                        pixelTracker.lostCount = 0;

                        DL_RESULT d{};
                        d.confidence = score;
                        d.box = box;
                        d.boxMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                        cv::rectangle(d.boxMask, box, cv::Scalar(255), cv::FILLED);

                        if (bestTi >= 0 && bestTi < (int)g_pixel_templates.size()) {
                            const auto &tmpl = g_pixel_templates[(size_t)bestTi];
                            d.classId = tmpl.kindId;
                            d.seiPath = tmpl.relPath;
                        }

                        dets.push_back(std::move(d));
                    } else {
                        pixelTracker.lostCount++;
                    }
                }
            } else {
                // Use YOLO model as before
                if (yoloDetector.RunSession(frame, dets) != RET_OK) {
                    std::cerr << "Error running inference on frame " << frameIdx << std::endl;
                    if (writer.isOpened()) writer.write(processed);
                    ++frameIdx; 
                    continue;
                }
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

        std::vector<SEIRegion> sei_regions;
        if (seiOut.is_open()) sei_regions.reserve(dets.size());
        // Track which YOLO hashes are NEW in this frame. A real client cannot recover these
        // until the corresponding template is transmitted out-of-band.
        std::unordered_set<std::string> yolo_new_hashes_this_frame;
        std::unordered_set<uint32_t> yolo_new_tplids_this_frame;
        // Cache canonical hashes per detection index for pass-2 recovery.
        std::vector<std::string> det_hashes;
        std::vector<uint8_t> det_is_new;
        std::vector<uint32_t> det_tplids;
        if (!opt.pixelMode) {
            det_hashes.resize(dets.size());
            det_is_new.assign(dets.size(), 0);
            det_tplids.assign(dets.size(), 0);
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
                // Pixel mode: paint with kind-consistent color (better motion prediction),
                // while stitching uses template key (seiPath).
                cv::Vec3b maskColor = kind_color_from_kind_id(d.classId);
                OfflineProcessor::paint_mask_color(processed, d.boxMask, d.box, maskColor, opt.paintAlpha);

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

                // YOLO latent-key mode: assign template_id by maskCoeff embedding similarity.
                if (opt.yoloLatentKey) {
                    auto t_dict0 = std::chrono::steady_clock::now();

                    // Normalize embedding
                    std::array<float, 32> emb = l2_normalize_32(d.maskCoeff);

                    // Hybrid sampling: base periodic mint + motion-triggered boost
                    bool force_mint = false;
                    if (opt.latentSamplePeriod > 0 && (frameIdx % opt.latentSamplePeriod) == 0) {
                        force_mint = true;
                    }

                    // Motion detection vs last box for this class
                    auto itLastBox = last_latent_by_class.find(d.classId);
                    if (itLastBox != last_latent_by_class.end() && itLastBox->second.valid) {
                        float iou = iou_rect(itLastBox->second.box, safeBox);
                        float cdist = center_dist_px(itLastBox->second.box, safeBox);
                        float ad = area_ratio_delta(itLastBox->second.box, safeBox);
                        if (iou < opt.latentMotionIouThr || cdist > opt.latentMotionCenterPx || ad > opt.latentMotionScaleThr) {
                            latent_boost_until_frame[d.classId] = frameIdx + std::max(0, opt.latentMotionBoostFrames);
                            latent_motion_boost_events++;
                        }
                    }
                    auto itBoost = latent_boost_until_frame.find(d.classId);
                    if (itBoost != latent_boost_until_frame.end() && frameIdx < itBoost->second) {
                        // In boost window, mint every frame for fluency
                        force_mint = true;
                    }

                    // Temporal stability: if last template for this class overlaps strongly, prefer it
                    uint32_t chosen_id = 0;
                    bool is_new = false;
                    auto itLast = last_latent_by_class.find(d.classId);
                    if (itLast != last_latent_by_class.end() && itLast->second.valid) {
                        float iou = iou_rect(itLast->second.box, safeBox);
                        if (iou >= 0.7f) {
                            float sim = cosine_sim_32(emb, itLast->second.emb_norm);
                            if (sim >= (opt.latentCosineThreshold - 0.01f)) {
                                chosen_id = itLast->second.id;
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
                            if (std::abs(tpl.wh.width - safeBox.width) > 2 || std::abs(tpl.wh.height - safeBox.height) > 2) continue;
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

                    if (chosen_id == 0 || force_mint) {
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
                        latent_id_to_index[tpl.id] = latent_bank.size();
                        latent_bank.push_back(std::move(tpl));
                        latent_minted++;
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
                        auto itIdx2 = latent_id_to_index.find(chosen_id);
                        if (itIdx2 != latent_id_to_index.end()) {
                            latent_bank[itIdx2->second].use_count++;
                        }
                    }

                    det_tplids[di] = chosen_id;
                    det_is_new[di] = is_new ? 1 : 0;
                    if (is_new) {
                        yolo_new_tplids_this_frame.insert(chosen_id);
                    }

                    // Update last selection for this class
                    LastLatent ll;
                    ll.box = safeBox;
                    ll.id = chosen_id;
                    ll.emb_norm = emb;
                    ll.valid = true;
                    last_latent_by_class[d.classId] = std::move(ll);

                    auto t_dict1 = std::chrono::steady_clock::now();
                    dict_ms += std::chrono::duration<double, std::milli>(t_dict1 - t_dict0).count();

                    // Paint masked stream
                    cv::Vec3b color = opt.yoloClassConsistentColor ? yolo_class_color(d.classId) : cv::Vec3b(0, 255, 0);
                    OfflineProcessor::paint_mask_color(processed, d.boxMask, d.box, color, opt.paintAlpha);

                    if (seiOut.is_open()) {
                        SEIRegion r{};
                        r.id = chosen_id; // template_id (stable across frames)
                        r.x = (uint32_t)std::max(0, d.box.x);
                        r.y = (uint32_t)std::max(0, d.box.y);
                        r.w = (uint32_t)std::max(0, d.box.width);
                        r.h = (uint32_t)std::max(0, d.box.height);
                        r.flags = is_new ? 0 : 1;
                        r.class_id = (uint8_t)std::max(0, std::min(255, d.classId));
                        r.path.clear(); // no client hashing; id is enough
                        sei_regions.push_back(std::move(r));
                    }
                    continue;
                }

                // pHash-keying mode (legacy simulator): position-invariant pHash on cropped ROI.
                auto t_dict0 = std::chrono::steady_clock::now();
                uint64_t ph = OfflineProcessor::mask_phash64(maskROI);
                // Temporal stability: if last template for this class overlaps strongly and is phash-close,
                // prefer it to avoid bouncing between near-duplicates.
                std::string matched;
                auto itLast = last_tpl_by_class.find(d.classId);
                if (itLast != last_tpl_by_class.end() && itLast->second.valid) {
                    float iou = iou_rect(itLast->second.box, safeBox);
                    if (iou >= 0.7f) {
                        int dist = OfflineProcessor::hamming64(ph, itLast->second.phash64);
                        if (dist <= opt.phashThreshold + 2) {
                            // reuse last hash if it's still in dict
                            if (processor.getDict().find(itLast->second.hash) != processor.getDict().end()) {
                                matched = itLast->second.hash;
                            }
                        }
                    }
                }
                if (matched.empty()) {
                    matched = find_best_phash_match(processor.getDict(), ph, d.classId, safeBox.size(), opt.phashThreshold);
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

                // Visualization (server masked stream)
                cv::Vec3b color = opt.yoloClassConsistentColor
                                  ? yolo_class_color(d.classId)
                                  : OfflineProcessor::deterministic_color_from_hash(canon);
                OfflineProcessor::paint_mask_color(processed, d.boxMask, d.box, color, opt.paintAlpha);

                if (seiOut.is_open()) {
                    SEIRegion r{};
                    r.id = (uint32_t)sei_regions.size();
                    r.x = (uint32_t)std::max(0, d.box.x);
                    r.y = (uint32_t)std::max(0, d.box.y);
                    r.w = (uint32_t)std::max(0, d.box.width);
                    r.h = (uint32_t)std::max(0, d.box.height);
                    r.flags = is_new ? 0 : 1; // 0=new, 1=reference
                    r.class_id = (uint8_t)std::max(0, std::min(255, d.classId)); // class id
                    r.path = canon; // key for stitching (canonical hash)
                    sei_regions.push_back(std::move(r));
                }
            }
        }
        auto t_paint1 = std::chrono::steady_clock::now();
        paint_ms = std::chrono::duration<double, std::milli>(t_paint1 - t_paint0).count();

        // Client receives the masked stream, then reconstructs using side channel (templates/dict).
        recovered = processed.clone();

        // Pass 2: apply recovery overlays onto recovered frame (client reconstruction)
        auto t_rec0 = std::chrono::steady_clock::now();
        for (size_t di = 0; di < dets.size(); ++di) {
            const auto &d = dets[di];
            if (d.confidence < opt.confThreshold) continue;
            if (d.boxMask.empty()) continue;

            if (opt.pixelMode) {
                cv::Mat tpl_rgba = get_pixel_template_rgba_for_detection(d);
                if (!tpl_rgba.empty()) {
                    overlay_template_rgba(recovered, tpl_rgba, d.box);
                    matched_occurrences++; // treat as matched since we can always recover from template set
                }
            } else {
                if (opt.yoloLatentKey) {
                    uint32_t tid = det_tplids[di];
                    if (!tid) continue;
                    bool is_new_for_client = det_is_new[di] || (yolo_new_tplids_this_frame.find(tid) != yolo_new_tplids_this_frame.end());

                    uint32_t use_id = tid;
                    if (is_new_for_client) {
                        // Fallback to last recoverable template for this class if bbox overlaps enough.
                        auto itLast = last_client_tpl_by_class.find(d.classId);
                        if (itLast != last_client_tpl_by_class.end() && itLast->second.valid) {
                            float iou = iou_rect(itLast->second.box, d.box);
                            if (iou >= 0.5f) {
                                use_id = itLast->second.id;
                                is_new_for_client = false; // treat as recoverable via fallback
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
                        if (img.cols != d.box.width || img.rows != d.box.height) {
                            cv::resize(img, img, d.box.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        overlay_template_rgba(recovered, img, d.box);

                        // Update last client template for this class (recoverable/cached)
                        LastClientTpl lc;
                        lc.box = d.box;
                        lc.id = use_id;
                        lc.valid = true;
                        last_client_tpl_by_class[d.classId] = std::move(lc);
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
                        if (tpl.cols != d.box.width || tpl.rows != d.box.height) {
                            cv::resize(tpl, tpl, d.box.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        overlay_template_rgba(recovered, tpl, d.box);
                    }
                }
            }
        }
        auto t_rec1 = std::chrono::steady_clock::now();
        recover_ms = std::chrono::duration<double, std::milli>(t_rec1 - t_rec0).count();

        if (seiOut.is_open()) {
            uint64_t pts = (uint64_t)frameIdx;
            std::vector<uint8_t> payload = build_msk1_payload((uint64_t)frameIdx, pts, sei_regions, /*version=*/3);
            uint32_t len = (uint32_t)payload.size();
            seiOut.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len) seiOut.write(reinterpret_cast<const char*>(payload.data()), len);
        }

        // Record quality metrics between baseline and masked frame
        if (frame.size() == processed.size()) {
            double ssim = computeSSIM(frame, processed);
            double psnr = computePSNR(frame, processed);
            ssim_sum += ssim;
            psnr_sum += psnr;
            metric_frames++;
            frame_indices.push_back(frameIdx);
            ssim_values.push_back(ssim);
            psnr_values.push_back(psnr);

            if (opt.recordTiming) {
                // Pixel-specific breakdown (0 if not pixel mode)
                t_pixel_bootstrap_ms.push_back(opt.pixelMode ? pixel_bootstrap_ms : 0.0);
                t_pixel_roi_ms.push_back(opt.pixelMode ? pixel_roi_ms : 0.0);
                t_pixel_scanned_bootstrap.push_back(opt.pixelMode ? pixel_scanned_bootstrap : 0);
                t_pixel_scanned_roi.push_back(opt.pixelMode ? pixel_scanned_roi : 0);
            }
        }

        // Record quality metrics between baseline and recovered frame
        if (frame.size() == recovered.size()) {
            double ssim_r = computeSSIM(frame, recovered);
            double psnr_r = computePSNR(frame, recovered);
            rec_ssim_sum += ssim_r;
            rec_psnr_sum += psnr_r;
            rec_metric_frames++;
            rec_ssim_values.push_back(ssim_r);
            rec_psnr_values.push_back(psnr_r);
        }

        if (origWriter.isOpened()) origWriter.write(frame);
        if (writer.isOpened()) writer.write(processed);
        if (recWriter.isOpened()) recWriter.write(recovered);

        auto t_frame_end = std::chrono::steady_clock::now();
        frame_total_ms = std::chrono::duration<double, std::milli>(t_frame_end - t_frame_start).count();

        if (opt.recordTiming) {
            t_infer_ms.push_back(infer_ms);
            t_dict_ms.push_back(dict_ms);
            t_paint_ms.push_back(paint_ms);
            t_recover_ms.push_back(recover_ms);
            t_total_ms.push_back(frame_total_ms);
        }

        ++frameIdx;
        if (frameIdx % 50 == 0) {
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

    json report;
    report["total_frames"] = frameIdx;
    report["total_detections"] = total_occurrences;
    report["matched_detections"] = matched_occurrences;
    report["avg_ssim"] = avg_ssim;
    report["avg_psnr"] = avg_psnr;
    report["avg_recovered_ssim"] = avg_rec_ssim;
    report["avg_recovered_psnr"] = avg_rec_psnr;
    report["timing_enabled"] = opt.recordTiming;
    report["latent_key_enabled"] = opt.yoloLatentKey;
    if (opt.yoloLatentKey) {
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

        // Dump full latent bank (for heatmaps/clustering)
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
                    {"emb", e}
                });
            }
            std::ofstream bf(fs::path(opt.outDir) / "latent_bank.json");
            bf << bank.dump(2);
            report["latent_bank_path"] = (fs::path(opt.outDir) / "latent_bank.json").string();
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
            }
        }
        per_frame.push_back(item);
    }
    report["per_frame"] = per_frame;

    std::ofstream rf(reportPath);
    rf << report.dump(2);
    std::cout << "Saved evaluation report to " << reportPath << std::endl;

    if (writer.isOpened()) writer.release();
    if (origWriter.isOpened()) origWriter.release();
    if (recWriter.isOpened()) recWriter.release();
    cap.release();

    return 0;
}

