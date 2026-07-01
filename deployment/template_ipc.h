#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct TemplateResponse {
    uint8_t status{0};
    std::vector<uint8_t> bytes;
};

bool send_template_request(int fd, const std::string &key);
bool recv_template_request(int fd, std::string &key);
bool send_template_response(int fd, uint8_t status, const std::vector<uint8_t> &bytes);
bool recv_template_response(int fd, TemplateResponse &response);

int connect_template_server(const std::string &host, uint16_t port);
int listen_template_server(const std::string &host, uint16_t port, int backlog = 8);
