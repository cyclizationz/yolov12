#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "../sei_parser.h"
#include "../template_ipc.h"

namespace fs = std::filesystem;

namespace {

struct PayloadStats {
    size_t frames = 0;
    size_t regions = 0;
    size_t parse_failures = 0;
    uint64_t bytes = 0;
};

struct ThinMetrics {
    size_t frames = 0;
    size_t payload_regions = 0;
    size_t parse_failures = 0;
    size_t template_refs = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t request_failures = 0;
    size_t templates_cached = 0;
    uint64_t payload_bytes = 0;
    uint64_t template_bytes_received = 0;
    double parse_ms = 0.0;
    double lookup_ms = 0.0;
    double request_wait_ms = 0.0;
    double cache_insert_ms = 0.0;
    double total_ms = 0.0;
    std::vector<double> request_latencies_ms;
};

void print_usage(const char *program_name) {
    std::cout
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "\n"
        << "RESPAWN online-client scaffold. This validates the artifact contract\n"
        << "emitted by RespawnOnlineServer and documents the Ref/Raw cache policy\n"
        << "used for the deferred online/Exp5 client.\n"
        << "\n"
        << "Options:\n"
        << "  -i, --input <dir>       Server artifact directory\n"
        << "      --cache-mode <mode> cold|warm|partial-warm\n"
        << "      --template-server host:port\n"
        << "                           Fetch missing templates over localhost instead of reading dict/\n"
        << "      --metrics-out <path> Write thin-client metrics JSON\n"
        << "      --template-delay-ms <n> Add artificial delay before each template request\n"
        << "      --strict            Return non-zero if required artifacts are missing\n"
        << "      --help              Show this help\n";
}

uint32_t read_le32(const std::vector<uint8_t> &data, size_t off) {
    return static_cast<uint32_t>(data[off]) |
           (static_cast<uint32_t>(data[off + 1]) << 8) |
           (static_cast<uint32_t>(data[off + 2]) << 16) |
           (static_cast<uint32_t>(data[off + 3]) << 24);
}

PayloadStats inspect_payloads(const fs::path &path) {
    PayloadStats stats;
    std::ifstream in(path, std::ios::binary);
    if (!in) return stats;
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    stats.bytes = data.size();
    size_t off = 0;
    while (off + 4 <= data.size()) {
        const uint32_t n = read_le32(data, off);
        off += 4;
        if (off + n > data.size()) {
            stats.parse_failures++;
            break;
        }
        ParsedSEI parsed;
        if (parse_msk1_payload(data.data() + off, n, parsed)) {
            stats.frames++;
            stats.regions += parsed.regions.size();
            for (const auto &group : parsed.pixel_grid_groups) {
                stats.regions += group.runs.size();
            }
        } else {
            stats.parse_failures++;
        }
        off += n;
    }
    if (off != data.size()) stats.parse_failures++;
    return stats;
}

size_t count_regular_files(const fs::path &dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return 0;
    size_t count = 0;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) count++;
    }
    return count;
}

std::string template_key(uint32_t id, const std::string &path) {
    if (!path.empty()) return fs::path(path).filename().string();
    return std::to_string(id);
}

std::vector<std::string> template_keys_from_payload(const ParsedSEI &parsed) {
    std::vector<std::string> keys;
    for (const auto &region : parsed.regions) {
        if (region.id != 0 || !region.path.empty()) {
            keys.push_back(template_key(region.id, region.path));
        }
    }
    for (const auto &group : parsed.pixel_grid_groups) {
        if (group.id != 0 || !group.path.empty()) {
            keys.push_back(template_key(group.id, group.path));
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

bool parse_host_port(const std::string &value, std::string &host, uint16_t &port) {
    auto pos = value.rfind(':');
    if (pos == std::string::npos) return false;
    host = value.substr(0, pos);
    port = static_cast<uint16_t>(std::stoi(value.substr(pos + 1)));
    return !host.empty() && port > 0;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(std::ceil(p * values.size()));
    if (idx == 0) idx = 1;
    return values[std::min(values.size() - 1, idx - 1)];
}

void write_metrics_json(const fs::path &path, const ThinMetrics &m, const std::string &input_dir, const std::string &server) {
    if (path.empty()) return;
    std::ofstream out(path);
    out << "{\n";
    out << "  \"input_dir\": \"" << input_dir << "\",\n";
    out << "  \"template_server\": \"" << server << "\",\n";
    out << "  \"frames\": " << m.frames << ",\n";
    out << "  \"payload_regions\": " << m.payload_regions << ",\n";
    out << "  \"parse_failures\": " << m.parse_failures << ",\n";
    out << "  \"template_refs\": " << m.template_refs << ",\n";
    out << "  \"cache_hits\": " << m.cache_hits << ",\n";
    out << "  \"cache_misses\": " << m.cache_misses << ",\n";
    out << "  \"request_failures\": " << m.request_failures << ",\n";
    out << "  \"templates_cached\": " << m.templates_cached << ",\n";
    out << "  \"payload_bytes\": " << m.payload_bytes << ",\n";
    out << "  \"template_bytes_received\": " << m.template_bytes_received << ",\n";
    out << "  \"parse_ms\": " << m.parse_ms << ",\n";
    out << "  \"lookup_ms\": " << m.lookup_ms << ",\n";
    out << "  \"request_wait_ms\": " << m.request_wait_ms << ",\n";
    out << "  \"cache_insert_ms\": " << m.cache_insert_ms << ",\n";
    out << "  \"total_ms\": " << m.total_ms << ",\n";
    out << "  \"request_latency_p50_ms\": " << percentile(m.request_latencies_ms, 0.50) << ",\n";
    out << "  \"request_latency_p95_ms\": " << percentile(m.request_latencies_ms, 0.95) << ",\n";
    out << "  \"request_latency_p99_ms\": " << percentile(m.request_latencies_ms, 0.99) << ",\n";
    out << "  \"request_latency_first_ms\": " << (m.request_latencies_ms.empty() ? 0.0 : m.request_latencies_ms.front()) << "\n";
    out << "}\n";
}

ThinMetrics run_thin_replay(
    const fs::path &payloads,
    const std::string &host,
    uint16_t port,
    int template_delay_ms
) {
    ThinMetrics m;
    std::ifstream in(payloads, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    m.payload_bytes = data.size();
    int fd = connect_template_server(host, port);
    if (fd < 0) {
        throw std::runtime_error("failed to connect to template server");
    }
    std::unordered_map<std::string, std::vector<uint8_t>> cache;
    auto total0 = std::chrono::steady_clock::now();
    size_t off = 0;
    while (off + 4 <= data.size()) {
        const uint32_t n = read_le32(data, off);
        off += 4;
        if (off + n > data.size()) {
            m.parse_failures++;
            break;
        }
        ParsedSEI parsed;
        auto parse0 = std::chrono::steady_clock::now();
        bool ok = parse_msk1_payload(data.data() + off, n, parsed);
        auto parse1 = std::chrono::steady_clock::now();
        m.parse_ms += std::chrono::duration<double, std::milli>(parse1 - parse0).count();
        off += n;
        if (!ok) {
            m.parse_failures++;
            continue;
        }
        m.frames++;
        m.payload_regions += parsed.regions.size();
        for (const auto &group : parsed.pixel_grid_groups) m.payload_regions += group.runs.size();
        auto keys = template_keys_from_payload(parsed);
        for (const auto &key : keys) {
            m.template_refs++;
            auto lookup0 = std::chrono::steady_clock::now();
            auto it = cache.find(key);
            auto lookup1 = std::chrono::steady_clock::now();
            m.lookup_ms += std::chrono::duration<double, std::milli>(lookup1 - lookup0).count();
            if (it != cache.end()) {
                m.cache_hits++;
                continue;
            }
            m.cache_misses++;
            if (template_delay_ms > 0) {
                ::usleep(static_cast<useconds_t>(template_delay_ms) * 1000);
            }
            auto req0 = std::chrono::steady_clock::now();
            TemplateResponse response;
            bool requested = send_template_request(fd, key) && recv_template_response(fd, response);
            auto req1 = std::chrono::steady_clock::now();
            double latency = std::chrono::duration<double, std::milli>(req1 - req0).count();
            m.request_wait_ms += latency;
            m.request_latencies_ms.push_back(latency);
            if (!requested || response.status != 0) {
                m.request_failures++;
                continue;
            }
            auto insert0 = std::chrono::steady_clock::now();
            m.template_bytes_received += response.bytes.size();
            cache.emplace(key, std::move(response.bytes));
            auto insert1 = std::chrono::steady_clock::now();
            m.cache_insert_ms += std::chrono::duration<double, std::milli>(insert1 - insert0).count();
        }
    }
    auto total1 = std::chrono::steady_clock::now();
    m.total_ms = std::chrono::duration<double, std::milli>(total1 - total0).count();
    m.templates_cached = cache.size();
    (void)send_template_request(fd, "__quit__");
    ::close(fd);
    return m;
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
    fs::path input_dir = "../outputs/online_server";
    std::string cache_mode = "cold";
    std::string template_server;
    fs::path metrics_out;
    int template_delay_ms = 0;
    bool strict = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        if (arg == "-i" || arg == "--input") {
            if (!require_value(i, argc, argv, value)) return 2;
            input_dir = value;
        } else if (arg == "--cache-mode") {
            if (!require_value(i, argc, argv, cache_mode)) return 2;
            if (cache_mode != "cold" && cache_mode != "warm" && cache_mode != "partial-warm") {
                std::cerr << "Unsupported cache mode: " << cache_mode << "\n";
                return 2;
            }
        } else if (arg == "--strict") {
            strict = true;
        } else if (arg == "--template-server") {
            if (!require_value(i, argc, argv, template_server)) return 2;
        } else if (arg == "--metrics-out") {
            if (!require_value(i, argc, argv, value)) return 2;
            metrics_out = value;
        } else if (arg == "--template-delay-ms") {
            if (!require_value(i, argc, argv, value)) return 2;
            template_delay_ms = std::stoi(value);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    const fs::path segmented = input_dir / "segmented_output.mp4";
    const fs::path recovered = input_dir / "recovered_output.mp4";
    const fs::path report = input_dir / "report.json";
    const fs::path dict = input_dir / "dict";
    const fs::path payloads = input_dir / "msk1_payloads.bin";

    if (!template_server.empty()) {
        std::string host;
        uint16_t port = 0;
        if (!parse_host_port(template_server, host, port)) {
            std::cerr << "Invalid --template-server, expected host:port\n";
            return 2;
        }
        if (!fs::exists(segmented) || !fs::exists(payloads) || !fs::exists(report)) {
            std::cerr << "Thin-client mode requires segmented_output.mp4, msk1_payloads.bin, and report.json only.\n";
            return 1;
        }
        try {
            ThinMetrics metrics = run_thin_replay(payloads, host, port, template_delay_ms);
            write_metrics_json(metrics_out, metrics, input_dir.string(), template_server);
            std::cout << "[ThinClient] frames=" << metrics.frames
                      << " refs=" << metrics.template_refs
                      << " hits=" << metrics.cache_hits
                      << " misses=" << metrics.cache_misses
                      << " failures=" << metrics.request_failures
                      << " cached=" << metrics.templates_cached
                      << " template_bytes=" << metrics.template_bytes_received
                      << " p95_request_ms=" << percentile(metrics.request_latencies_ms, 0.95)
                      << " total_ms=" << metrics.total_ms
                      << "\n";
            if (strict && (metrics.parse_failures > 0 || metrics.request_failures > 0)) return 1;
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "Thin-client replay failed: " << e.what() << "\n";
            return 1;
        }
    }

    size_t missing = 0;
    auto check = [&](const fs::path &path, const std::string &label) {
        const bool ok = fs::exists(path);
        std::cout << "[OnlineClient] " << label << ": " << (ok ? "found" : "missing") << " (" << path.string() << ")\n";
        if (!ok) missing++;
    };

    std::cout << "[OnlineClient] cache_mode=" << cache_mode << " input_dir=" << input_dir.string() << "\n";
    check(segmented, "masked stream");
    check(payloads, "RSEI/MSK1 sidecar");
    check(dict, "template dictionary");
    check(report, "report");
    check(recovered, "offline recovered stream");

    const size_t template_files = count_regular_files(dict);
    const PayloadStats payload_stats = inspect_payloads(payloads);
    std::cout << "[OnlineClient] templates=" << template_files
              << " payload_frames=" << payload_stats.frames
              << " payload_regions=" << payload_stats.regions
              << " payload_bytes=" << payload_stats.bytes
              << " parse_failures=" << payload_stats.parse_failures
              << "\n";

    std::cout << "[OnlineClient] Ref/Raw policy: use Ref when the referenced template is in the client cache; "
              << "otherwise render Raw and request the missing template asynchronously.\n";
    std::cout << "[OnlineClient] This target validates the current sidecar artifact contract. "
              << "Live decode, SEI extraction, OOB template delivery, and frame-deadline measurement remain the next online step.\n";

    if (strict && (missing > 0 || payload_stats.parse_failures > 0)) {
        return 1;
    }
    return 0;
}
