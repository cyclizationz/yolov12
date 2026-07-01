#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "../template_ipc.h"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
    g_stop = true;
}

struct Stats {
    uint64_t requests{0};
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t bytes_sent{0};
    double service_ms_sum{0.0};
    double service_ms_max{0.0};
};

void usage(const char *program) {
    std::cout
        << "Usage: " << program << " --dict <dir> [--host 127.0.0.1] [--port 19061] [--metrics-out path]\n"
        << "\n"
        << "Serves RESPAWN template PNG files from a server-owned directory over localhost.\n";
}

bool read_file(const fs::path &path, std::vector<uint8_t> &bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

fs::path resolve_template_path(const fs::path &root, const std::string &key) {
    fs::path name = fs::path(key).filename();
    if (name.empty()) return {};
    fs::path direct = root / name;
    if (fs::exists(direct)) return direct;
    if (!name.has_extension()) {
        fs::path png = root / (name.string() + ".png");
        if (fs::exists(png)) return png;
    }
    return direct;
}

void write_metrics(const fs::path &path, const Stats &stats, const fs::path &dict, const std::string &host, uint16_t port) {
    if (path.empty()) return;
    std::ofstream out(path);
    out << "{\n";
    out << "  \"dict_dir\": \"" << dict.string() << "\",\n";
    out << "  \"host\": \"" << host << "\",\n";
    out << "  \"port\": " << port << ",\n";
    out << "  \"requests\": " << stats.requests << ",\n";
    out << "  \"hits\": " << stats.hits << ",\n";
    out << "  \"misses\": " << stats.misses << ",\n";
    out << "  \"bytes_sent\": " << stats.bytes_sent << ",\n";
    out << "  \"avg_service_ms\": " << (stats.requests ? stats.service_ms_sum / stats.requests : 0.0) << ",\n";
    out << "  \"max_service_ms\": " << stats.service_ms_max << "\n";
    out << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    fs::path dict_dir;
    fs::path metrics_out;
    std::string host = "127.0.0.1";
    uint16_t port = 19061;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--dict" || arg == "-d") && i + 1 < argc) {
            dict_dir = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--metrics-out" && i + 1 < argc) {
            metrics_out = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            usage(argv[0]);
            return 2;
        }
    }
    if (dict_dir.empty() || !fs::is_directory(dict_dir)) {
        std::cerr << "Template dictionary directory is required and must exist.\n";
        return 2;
    }

    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    int listen_fd = listen_template_server(host, port);
    if (listen_fd < 0) return 1;

    Stats stats;
    std::cout << "[TemplateServer] serving " << dict_dir << " on " << host << ":" << port << "\n";
    while (!g_stop) {
        int client_fd = ::accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (g_stop) break;
            continue;
        }
        int yes = 1;
        (void)::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        while (!g_stop) {
            std::string key;
            if (!recv_template_request(client_fd, key)) break;
            if (key == "__quit__") {
                g_stop = true;
                break;
            }
            auto t0 = std::chrono::steady_clock::now();
            std::vector<uint8_t> bytes;
            fs::path path = resolve_template_path(dict_dir, key);
            uint8_t status = read_file(path, bytes) ? 0 : 1;
            bool ok = send_template_response(client_fd, status, bytes);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            stats.requests++;
            stats.service_ms_sum += ms;
            stats.service_ms_max = std::max(stats.service_ms_max, ms);
            if (status == 0) {
                stats.hits++;
                stats.bytes_sent += bytes.size();
            } else {
                stats.misses++;
            }
            if (!ok) break;
        }
        ::close(client_fd);
    }
    ::close(listen_fd);
    write_metrics(metrics_out, stats, dict_dir, host, port);
    std::cout << "[TemplateServer] requests=" << stats.requests
              << " hits=" << stats.hits
              << " misses=" << stats.misses
              << " bytes_sent=" << stats.bytes_sent << "\n";
    return 0;
}
