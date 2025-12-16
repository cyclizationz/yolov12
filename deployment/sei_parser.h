#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct SEIRegion {
    uint32_t id;
    uint32_t x, y, w, h;
    uint8_t flags;
    uint8_t class_id;
    std::string path;
};

struct ParsedSEI {
    uint32_t magic;
    uint16_t version;
    uint64_t frame_counter;
    uint64_t pts;
    std::vector<SEIRegion> regions;
};

bool parse_msk1_payload(const uint8_t* data, size_t size, ParsedSEI &out);
std::vector<uint8_t> build_msk1_payload(uint64_t frame_counter, uint64_t pts,
                                        const std::vector<SEIRegion>& regions,
                                        uint16_t version = 3);



