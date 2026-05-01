#!/usr/bin/env python3
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    region_id: int
    x: int
    y: int
    w: int
    h: int
    flags: int
    class_id: int
    path: str


@dataclass(frozen=True)
class ParsedPayload:
    magic: int
    version: int
    frame_counter: int
    pts: int
    frame_flags: int
    regions: list[Region]


def read_len_prefixed_payloads(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out: list[bytes] = []
    off = 0
    while off + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        if off + n > len(data):
            break
        out.append(data[off : off + n])
        off += n
    return out


def parse_payload(buf: bytes) -> ParsedPayload | None:
    if len(buf) < 26:
        return None
    off = 0
    magic = int.from_bytes(buf[off : off + 4], "big")
    off += 4
    version = int.from_bytes(buf[off : off + 2], "big")
    off += 2
    frame_counter = int.from_bytes(buf[off : off + 8], "big")
    off += 8
    pts = int.from_bytes(buf[off : off + 8], "big")
    off += 8
    frame_flags = 0
    if version >= 4:
        if off + 1 > len(buf):
            return None
        frame_flags = buf[off]
        off += 1
    if off + 4 > len(buf):
        return None
    region_count = int.from_bytes(buf[off : off + 4], "big")
    off += 4
    regions: list[Region] = []
    for _ in range(region_count):
        need = 20 + 2 if version < 3 else 20 + 3
        if off + need > len(buf):
            return None
        region_id = int.from_bytes(buf[off : off + 4], "big")
        off += 4
        x = int.from_bytes(buf[off : off + 4], "big")
        off += 4
        y = int.from_bytes(buf[off : off + 4], "big")
        off += 4
        w = int.from_bytes(buf[off : off + 4], "big")
        off += 4
        h = int.from_bytes(buf[off : off + 4], "big")
        off += 4
        flags = 1
        if version >= 3:
            flags = buf[off]
            off += 1
        class_id = buf[off]
        off += 1
        path_len = buf[off]
        off += 1
        if off + path_len > len(buf):
            return None
        path = buf[off : off + path_len].decode("utf-8", errors="replace")
        off += path_len
        regions.append(
            Region(
                region_id=region_id,
                x=x,
                y=y,
                w=w,
                h=h,
                flags=flags,
                class_id=class_id,
                path=path,
            )
        )
    return ParsedPayload(
        magic=magic,
        version=version,
        frame_counter=frame_counter,
        pts=pts,
        frame_flags=frame_flags,
        regions=regions,
    )


def load_payloads(path: Path) -> list[ParsedPayload]:
    payloads = []
    for raw in read_len_prefixed_payloads(path):
        parsed = parse_payload(raw) if raw else ParsedPayload(0x4D534B31, 4, len(payloads), len(payloads), 0, [])
        if parsed is not None:
            payloads.append(parsed)
    return payloads
