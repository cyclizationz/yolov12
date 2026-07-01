#include "template_ipc.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

bool read_exact(int fd, void *buf, size_t n) {
    auto *p = static_cast<uint8_t *>(buf);
    while (n > 0) {
        ssize_t got = ::recv(fd, p, n, 0);
        if (got == 0) return false;
        if (got < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += got;
        n -= static_cast<size_t>(got);
    }
    return true;
}

bool write_exact(int fd, const void *buf, size_t n) {
    const auto *p = static_cast<const uint8_t *>(buf);
    while (n > 0) {
        ssize_t sent = ::send(fd, p, n, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += sent;
        n -= static_cast<size_t>(sent);
    }
    return true;
}

} // namespace

bool send_template_request(int fd, const std::string &key) {
    if (key.size() > UINT32_MAX) return false;
    uint32_t len = htonl(static_cast<uint32_t>(key.size()));
    return write_exact(fd, &len, sizeof(len)) && write_exact(fd, key.data(), key.size());
}

bool recv_template_request(int fd, std::string &key) {
    uint32_t len_net = 0;
    if (!read_exact(fd, &len_net, sizeof(len_net))) return false;
    uint32_t len = ntohl(len_net);
    if (len > 1024 * 1024) return false;
    key.assign(len, '\0');
    return len == 0 || read_exact(fd, key.data(), len);
}

bool send_template_response(int fd, uint8_t status, const std::vector<uint8_t> &bytes) {
    if (bytes.size() > UINT32_MAX) return false;
    uint32_t len = htonl(static_cast<uint32_t>(bytes.size()));
    return write_exact(fd, &status, sizeof(status)) && write_exact(fd, &len, sizeof(len)) &&
           (bytes.empty() || write_exact(fd, bytes.data(), bytes.size()));
}

bool recv_template_response(int fd, TemplateResponse &response) {
    uint32_t len_net = 0;
    if (!read_exact(fd, &response.status, sizeof(response.status))) return false;
    if (!read_exact(fd, &len_net, sizeof(len_net))) return false;
    uint32_t len = ntohl(len_net);
    response.bytes.assign(len, 0);
    return len == 0 || read_exact(fd, response.bytes.data(), len);
}

int connect_template_server(const std::string &host, uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

int listen_template_server(const std::string &host, uint16_t port, int backlog) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }
    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        std::cerr << "bind failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return -1;
    }
    if (::listen(fd, backlog) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}
