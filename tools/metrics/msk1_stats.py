import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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


@dataclass
class FrameStat:
    ver: int
    frame_flags: int | None
    nreg: int
    payload_bytes: int
    path_bytes: int


def parse_msk1(payload: bytes) -> FrameStat:
    if not payload or len(payload) < 4 + 2 + 8 + 8 + 4:
        return FrameStat(ver=0, frame_flags=None, nreg=0, payload_bytes=len(payload), path_bytes=0)
    off = 0
    magic, off = be_u32(payload, off)
    if magic != 0x4D534B31:
        return FrameStat(ver=0, frame_flags=None, nreg=0, payload_bytes=len(payload), path_bytes=0)
    ver, off = be_u16(payload, off)
    _, off = be_u64(payload, off)  # frame_counter
    _, off = be_u64(payload, off)  # pts
    frame_flags: int | None = None
    if ver >= 4:
        frame_flags = int(payload[off])
        off += 1
    nreg, off = be_u32(payload, off)
    path_bytes = 0
    for _ in range(nreg):
        _, off = be_u32(payload, off)  # tid
        _, off = be_u32(payload, off)  # x
        _, off = be_u32(payload, off)  # y
        _, off = be_u32(payload, off)  # w
        _, off = be_u32(payload, off)  # h
        if ver >= 3:
            off += 1  # flags
        off += 1  # class_id
        L = int(payload[off])
        off += 1
        path_bytes += L
        off += L
        if off > len(payload):
            break
    return FrameStat(ver=ver, frame_flags=frame_flags, nreg=int(nreg), payload_bytes=len(payload), path_bytes=path_bytes)


def q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.quantile(np.array(xs, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize MSK1 per-frame flags/regions/overhead from msk1_payloads.bin.")
    ap.add_argument("--msk1-bin", required=True, type=Path)
    ap.add_argument(
        "--ref-frame-policy",
        choices=["frame_flags", "regions_nonzero"],
        default="frame_flags",
        help="How to define ref vs raw frames. 'frame_flags' uses v4 frame_flags if present; "
        "'regions_nonzero' treats frames with nreg>0 as ref.",
    )
    args = ap.parse_args()

    payloads = read_len_prefixed_payloads(args.msk1_bin)
    stats = [parse_msk1(p) for p in payloads]

    nregs = [s.nreg for s in stats]
    payload_bytes = [s.payload_bytes for s in stats]
    path_bytes = [s.path_bytes for s in stats]

    # payload+framing bytes per frame is just file_bytes / frames.
    file_bytes = args.msk1_bin.stat().st_size
    frames = len(payloads)

    # Determine ref/raw
    raw = 0
    ref = 0
    if args.ref_frame_policy == "frame_flags":
        # v4: frame_flags==0 => raw; otherwise ref. If frame_flags missing (ver<4), fall back to nreg>0.
        for s in stats:
            if s.frame_flags is None:
                if s.nreg > 0:
                    ref += 1
                else:
                    raw += 1
            else:
                if s.frame_flags == 0:
                    raw += 1
                else:
                    ref += 1
    else:
        for s in stats:
            if s.nreg > 0:
                ref += 1
            else:
                raw += 1

    out = {
        "frames": frames,
        "msk1_bin_file_bytes": file_bytes,
        "msk1_bin_bytes_per_frame": (file_bytes / frames) if frames else 0.0,  # payload + framing
        "payload_bytes_total": int(sum(payload_bytes)),
        "payload_bytes_per_frame": float(np.mean(np.array(payload_bytes, dtype=np.float64))) if frames else 0.0,
        "path_bytes_total": int(sum(path_bytes)),
        "path_bytes_pct_of_payload": (100.0 * sum(path_bytes) / sum(payload_bytes)) if sum(payload_bytes) else 0.0,
        "regions_per_frame": {
            "mean": float(np.mean(np.array(nregs, dtype=np.float64))) if frames else 0.0,
            "p50": q([float(x) for x in nregs], 0.50),
            "p90": q([float(x) for x in nregs], 0.90),
            "max": int(max(nregs)) if nregs else 0,
        },
        "ref_frames": ref,
        "raw_frames": raw,
        "ref_frame_ratio": (ref / frames) if frames else 0.0,
        "raw_frame_ratio": (raw / frames) if frames else 0.0,
        "ref_frame_policy": args.ref_frame_policy,
        "version_counts": {str(v): int(sum(1 for s in stats if s.ver == v)) for v in sorted(set(s.ver for s in stats))},
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


