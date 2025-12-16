#include "sei_parser.h"
#include <cstring>
#include <arpa/inet.h>

static inline uint64_t ntohll_u64(uint64_t v){
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (((uint64_t)ntohl((uint32_t)(v & 0xFFFFFFFFULL))) << 32) | ntohl((uint32_t)(v >> 32));
#else
    return v;
#endif
}

bool parse_msk1_payload(const uint8_t* data, size_t size, ParsedSEI &out) {
    if (!data || size < 4+2+8+8+4) return false;
    const uint8_t* p = data;
    uint32_t magic_be; memcpy(&magic_be, p, 4); p += 4; out.magic = ntohl(magic_be);
    uint16_t ver_be; memcpy(&ver_be, p, 2); p += 2; out.version = ntohs(ver_be);
    uint64_t frame_be; memcpy(&frame_be, p, 8); p += 8; out.frame_counter = ntohll_u64(frame_be);
    uint64_t pts_be; memcpy(&pts_be, p, 8); p += 8; out.pts = ntohll_u64(pts_be);
    uint32_t nr_be; memcpy(&nr_be, p, 4); p += 4; uint32_t N = ntohl(nr_be);
    out.regions.clear(); out.regions.reserve(N);
    for (uint32_t i=0;i<N;i++) {
        // v2: id(4) x(4) y(4) w(4) h(4) class_id(1) path_len(1) path(L)
        // v3: id(4) x(4) y(4) w(4) h(4) flags(1) class_id(1) path_len(1) path(L)
        size_t need = 4*5 + 1 + 1; // v2 minimum (class_id + path_len)
        if (out.version >= 3) need += 1; // flags
        if ((size_t)(p - data) + need > size) return false;
        SEIRegion r; uint32_t be32;
        memcpy(&be32, p,4); p+=4; r.id = ntohl(be32);
        memcpy(&be32, p,4); p+=4; r.x = ntohl(be32);
        memcpy(&be32, p,4); p+=4; r.y = ntohl(be32);
        memcpy(&be32, p,4); p+=4; r.w = ntohl(be32);
        memcpy(&be32, p,4); p+=4; r.h = ntohl(be32);
        if (out.version >= 3) {
            r.flags = *p++;
        } else {
            r.flags = 1; // treat all v2 records as reference for backward-compat
        }
        r.class_id = *p++;
        uint8_t L = *p++;
        if ((size_t)(p - data) + L > size) return false;
        r.path.assign(reinterpret_cast<const char*>(p), reinterpret_cast<const char*>(p)+L);
        p += L;
        out.regions.push_back(std::move(r));
    }
    return true;
}

std::vector<uint8_t> build_msk1_payload(uint64_t frame_counter, uint64_t pts,
                                        const std::vector<SEIRegion>& regions,
                                        uint16_t version) {
    uint32_t magic = htonl(0x4D534B31u); // 'MSK1'
    uint16_t ver   = htons(version);
    auto htonll_local = [](uint64_t v){
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (((uint64_t)htonl((uint32_t)(v & 0xFFFFFFFFULL))) << 32) | htonl((uint32_t)(v >> 32));
#else
        return v;
#endif
    };
    uint64_t f_be = htonll_local(frame_counter);
    uint64_t pts_be = htonll_local(pts);
    uint32_t nr_be = htonl((uint32_t)regions.size());
    size_t payload_len = 4+2+8+8+4;
    for (auto &r: regions) {
        size_t L = r.path.size(); if (L>255) L=255;
        if (version >= 3)
            payload_len += 4*5 + 1 + 1 + 1 + L; // id,x,y,w,h,flags,class_id,path_len,path
        else
            payload_len += 4*5 + 1 + 1 + L; // v2
    }
    std::vector<uint8_t> buf(payload_len);
    uint8_t *p = buf.data();
    memcpy(p, &magic,4); p+=4; memcpy(p,&ver,2); p+=2; memcpy(p,&f_be,8); p+=8; memcpy(p,&pts_be,8); p+=8; memcpy(p,&nr_be,4); p+=4;
    for (auto &r: regions) {
        uint32_t be32; size_t L = r.path.size(); if (L>255) L=255;
        be32 = htonl(r.id); memcpy(p,&be32,4); p+=4;
        be32 = htonl(r.x); memcpy(p,&be32,4); p+=4;
        be32 = htonl(r.y); memcpy(p,&be32,4); p+=4;
        be32 = htonl(r.w); memcpy(p,&be32,4); p+=4;
        be32 = htonl(r.h); memcpy(p,&be32,4); p+=4;
        if (version >= 3) { *p++ = r.flags; }
        *p++ = r.class_id;
        *p++ = (uint8_t)L; if (L) { memcpy(p, r.path.data(), L); p+=L; }
    }
    return buf;
}



