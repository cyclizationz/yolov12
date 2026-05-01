import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Tuple
from typing import cast
from typing import Optional
from typing import List

import numpy as np


def read_len_prefixed_payloads(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out: list[bytes] = []
    off = 0
    while off + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        if n == 0:
            out.append(b"")
            continue
        if off + n > len(data):
            break
        out.append(data[off : off + n])
        off += n
    return out


def be_u32(b: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(b[off : off + 4], "big"), off + 4


def be_u16(b: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(b[off : off + 2], "big"), off + 2


def be_u64(b: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(b[off : off + 8], "big"), off + 8


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    flags: int
    path: str

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)


@dataclass(frozen=True)
class Frame:
    ver: int
    frame_counter: int
    pts: int
    frame_flags: Optional[int]
    regions: list[Region]


def parse_msk1(payload: bytes) -> Frame:
    if not payload or len(payload) < 4 + 2 + 8 + 8 + 4:
        return Frame(ver=0, frame_counter=0, pts=0, frame_flags=None, regions=[])
    off = 0
    magic, off = be_u32(payload, off)
    if magic != 0x4D534B31:  # 'MSK1'
        return Frame(ver=0, frame_counter=0, pts=0, frame_flags=None, regions=[])
    ver, off = be_u16(payload, off)
    frame_counter, off = be_u64(payload, off)
    pts, off = be_u64(payload, off)
    frame_flags: Optional[int] = None
    if ver >= 4:
        if off >= len(payload):
            return Frame(ver=ver, frame_counter=frame_counter, pts=pts, frame_flags=None, regions=[])
        frame_flags = int(payload[off])
        off += 1
    nreg, off = be_u32(payload, off)
    regions: list[Region] = []
    for _ in range(int(nreg)):
        if off + 4 * 5 + 1 + 1 + 1 > len(payload):
            break
        _, off = be_u32(payload, off)  # id (unused here)
        x, off = be_u32(payload, off)
        y, off = be_u32(payload, off)
        w, off = be_u32(payload, off)
        h, off = be_u32(payload, off)
        flags = 1
        if ver >= 3:
            flags = int(payload[off])
            off += 1
        class_id = int(payload[off])
        off += 1
        L = int(payload[off])
        off += 1
        if off + L > len(payload):
            break
        path = payload[off : off + L].decode("utf-8", errors="replace")
        off += L
        regions.append(Region(x=int(x), y=int(y), w=int(w), h=int(h), class_id=class_id, flags=flags, path=path))
    return Frame(ver=ver, frame_counter=frame_counter, pts=pts, frame_flags=frame_flags, regions=regions)


def iou(a: Region, b: Region) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1, a.area + b.area - inter)
    return float(inter) / float(ua)


def greedy_match(prev: list[Region], cur: list[Region], iou_thr: float) -> tuple[int, list[float]]:
    """Greedy IoU matching count and list of best IoUs for matched pairs."""
    if not prev or not cur:
        return 0, []
    used_prev = [False] * len(prev)
    matched = 0
    ious: list[float] = []
    # Match larger boxes first to reduce fragmentation
    cur_order = sorted(range(len(cur)), key=lambda i: cur[i].area, reverse=True)
    for ci in cur_order:
        best_pi = -1
        best_iou = 0.0
        for pi in range(len(prev)):
            if used_prev[pi]:
                continue
            v = iou(prev[pi], cur[ci])
            if v > best_iou:
                best_iou = v
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_thr:
            used_prev[best_pi] = True
            matched += 1
            ious.append(best_iou)
    return matched, ious


def q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.quantile(np.array(xs, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Quantify temporal stability ('flicker') of MSK1 regions by matching boxes across frames.\n"
            "This is most useful for pixelMode, where unstable per-frame detections can break prediction."
        )
    )
    ap.add_argument("--msk1-bin", required=True, type=Path)
    ap.add_argument("--iou-thr", type=float, default=0.7, help="IoU threshold to count a region as persistent (default: 0.7).")
    ap.add_argument(
        "--group-by",
        choices=["all", "path", "class_id"],
        default="all",
        help="Optionally compute stability separately per template path or class_id.",
    )
    args = ap.parse_args()

    payloads = read_len_prefixed_payloads(args.msk1_bin)
    frames = [parse_msk1(p) for p in payloads]

    # Define key function for grouping
    def key_of(r: Region) -> str:
        if args.group_by == "path":
            return r.path or "<empty>"
        if args.group_by == "class_id":
            return str(r.class_id)
        return "<all>"

    # Collect per-frame sets
    keys = sorted(set(key_of(r) for f in frames for r in f.regions)) or ["<all>"]
    out: dict[str, object] = {
        "msk1_bin": str(args.msk1_bin),
        "frames": len(frames),
        "iou_thr": args.iou_thr,
        "group_by": args.group_by,
        "groups": {},
    }

    for g in keys:
        prev: list[Region] = []
        pers_ratios: list[float] = []
        counts: list[int] = []
        matched_ious: list[float] = []
        for f in frames:
            cur = [r for r in f.regions if key_of(r) == g] if g != "<all>" else list(f.regions)
            counts.append(len(cur))
            if prev:
                m, ious_ = greedy_match(prev, cur, args.iou_thr)
                denom = max(1, min(len(prev), len(cur)))
                pers_ratios.append(float(m) / float(denom))
                matched_ious.extend(ious_)
            prev = cur

        grp = {
            "regions_per_frame": {
                "mean": float(np.mean(np.array(counts, dtype=np.float64))) if counts else 0.0,
                "p50": q([float(x) for x in counts], 0.50),
                "p90": q([float(x) for x in counts], 0.90),
                "max": int(max(counts)) if counts else 0,
            },
            "persistence_ratio_frame_to_frame": {
                "mean": float(np.mean(np.array(pers_ratios, dtype=np.float64))) if pers_ratios else float("nan"),
                "p50": q(pers_ratios, 0.50),
                "p10": q(pers_ratios, 0.10),
            },
            "matched_iou": {
                "mean": float(np.mean(np.array(matched_ious, dtype=np.float64))) if matched_ious else float("nan"),
                "p50": q(matched_ious, 0.50),
                "p10": q(matched_ious, 0.10),
            },
        }
        cast(dict, out["groups"])[g] = grp

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

