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

struct PixelGridRun {
    int16_t row;
    int16_t col0;
    uint16_t count;
};

struct PixelGridGroup {
    uint32_t id;
    uint32_t origin_x, origin_y;
    uint16_t tile_w, tile_h;
    int16_t step_x, step_y;
    uint8_t paint_pad_x, paint_pad_y;
    uint8_t flags;
    uint8_t class_id;
    std::string path;
    std::vector<PixelGridRun> runs;
};

struct ParsedSEI {
    uint32_t magic;
    uint16_t version;
    uint64_t frame_counter;
    uint64_t pts;
    uint8_t frame_flags; // v4+: per-frame flags (0=raw, 1=ref)
    std::vector<SEIRegion> regions;
    std::vector<PixelGridGroup> pixel_grid_groups; // v5+: compact repeated-template grids
};

bool parse_msk1_payload(const uint8_t* data, size_t size, ParsedSEI &out);
std::vector<uint8_t> build_msk1_payload(uint64_t frame_counter, uint64_t pts,
                                        const std::vector<SEIRegion>& regions,
                                        uint16_t version = 3,
                                        uint8_t frame_flags = 0,
                                        const std::vector<PixelGridGroup>& pixel_grid_groups = {});



