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
class PixelGridRun:
    row: int
    col0: int
    count: int


@dataclass(frozen=True)
class PixelGridGroup:
    group_id: int
    origin_x: int
    origin_y: int
    tile_w: int
    tile_h: int
    step_x: int
    step_y: int
    paint_pad_x: int
    paint_pad_y: int
    flags: int
    class_id: int
    path: str
    runs: list[PixelGridRun]


@dataclass(frozen=True)
class ParsedPayload:
    magic: int
    version: int
    frame_counter: int
    pts: int
    frame_flags: int
    regions: list[Region]
    pixel_grid_groups: list[PixelGridGroup] | None = None


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
    pixel_grid_groups: list[PixelGridGroup] = []
    if version >= 5 and off < len(buf):
        if off + 2 > len(buf):
            return None
        group_count = int.from_bytes(buf[off : off + 2], "big")
        off += 2
        expanded_seq = len(regions)
        for _ in range(group_count):
            min_group = 4 + 4 + 4 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 2
            if off + min_group > len(buf):
                return None
            group_id = int.from_bytes(buf[off : off + 4], "big")
            off += 4
            origin_x = int.from_bytes(buf[off : off + 4], "big")
            off += 4
            origin_y = int.from_bytes(buf[off : off + 4], "big")
            off += 4
            tile_w = int.from_bytes(buf[off : off + 2], "big")
            off += 2
            tile_h = int.from_bytes(buf[off : off + 2], "big")
            off += 2
            step_x = int.from_bytes(buf[off : off + 2], "big", signed=True)
            off += 2
            step_y = int.from_bytes(buf[off : off + 2], "big", signed=True)
            off += 2
            paint_pad_x = buf[off]
            off += 1
            paint_pad_y = buf[off]
            off += 1
            flags = buf[off]
            off += 1
            class_id = buf[off]
            off += 1
            path_len = buf[off]
            off += 1
            if off + path_len + 2 > len(buf):
                return None
            path = buf[off : off + path_len].decode("utf-8", errors="replace")
            off += path_len
            run_count = int.from_bytes(buf[off : off + 2], "big")
            off += 2
            if off + run_count * 6 > len(buf):
                return None
            runs: list[PixelGridRun] = []
            for _run in range(run_count):
                row = int.from_bytes(buf[off : off + 2], "big", signed=True)
                off += 2
                col0 = int.from_bytes(buf[off : off + 2], "big", signed=True)
                off += 2
                count = int.from_bytes(buf[off : off + 2], "big")
                off += 2
                runs.append(PixelGridRun(row=row, col0=col0, count=count))
                for k in range(count):
                    x = origin_x + (col0 + k) * step_x - paint_pad_x
                    y = origin_y + row * step_y - paint_pad_y
                    w = tile_w + 2 * paint_pad_x
                    h = tile_h + 2 * paint_pad_y
                    regions.append(
                        Region(
                            region_id=expanded_seq,
                            x=max(0, x),
                            y=max(0, y),
                            w=max(0, w),
                            h=max(0, h),
                            flags=flags,
                            class_id=class_id,
                            path=path,
                        )
                    )
                    expanded_seq += 1
            pixel_grid_groups.append(
                PixelGridGroup(
                    group_id=group_id,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    step_x=step_x,
                    step_y=step_y,
                    paint_pad_x=paint_pad_x,
                    paint_pad_y=paint_pad_y,
                    flags=flags,
                    class_id=class_id,
                    path=path,
                    runs=runs,
                )
            )
    return ParsedPayload(
        magic=magic,
        version=version,
        frame_counter=frame_counter,
        pts=pts,
        frame_flags=frame_flags,
        regions=regions,
        pixel_grid_groups=pixel_grid_groups,
    )


def load_payloads(path: Path) -> list[ParsedPayload]:
    payloads = []
    for raw in read_len_prefixed_payloads(path):
        parsed = parse_payload(raw) if raw else ParsedPayload(0x4D534B31, 4, len(payloads), len(payloads), 0, [], [])
        if parsed is not None:
            payloads.append(parsed)
    return payloads
