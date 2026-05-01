import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class Region:
    tid: int
    x: int
    y: int
    w: int
    h: int
    flags: int
    class_id: int
    path: str


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


def parse_msk1(payload: bytes) -> Tuple[int, int, list[Region]]:
    """
    Parse MSK1 payload (v3/v4). Returns (version, frame_flags, regions).
    frame_flags==0 means "raw frame" in v4.
    """
    if len(payload) < 4 + 2 + 8 + 8 + 4:
        return 0, 0, []
    off = 0
    magic, off = be_u32(payload, off)
    if magic != 0x4D534B31:
        return 0, 0, []
    ver, off = be_u16(payload, off)
    _, off = be_u64(payload, off)  # frame_counter
    _, off = be_u64(payload, off)  # pts
    frame_flags = 0
    if ver >= 4:
        frame_flags = payload[off]
        off += 1
    nreg, off = be_u32(payload, off)
    regs: list[Region] = []
    for _ in range(nreg):
        tid, off = be_u32(payload, off)
        x, off = be_u32(payload, off)
        y, off = be_u32(payload, off)
        w, off = be_u32(payload, off)
        h, off = be_u32(payload, off)
        flags = 1
        if ver >= 3:
            flags = payload[off]
            off += 1
        class_id = payload[off]
        off += 1
        L = payload[off]
        off += 1
        path = payload[off : off + L].decode("utf-8", errors="ignore") if L else ""
        off += L
        regs.append(Region(int(tid), int(x), int(y), int(w), int(h), int(flags), int(class_id), path))
    return ver, frame_flags, regs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify pixel-exactness in healed regions using templates (dict PNG alpha) and MSK1 regions."
    )
    ap.add_argument("--orig", required=True, type=Path, help="original_output.mp4")
    ap.add_argument("--recovered", required=True, type=Path, help="recovered_output.mp4")
    ap.add_argument("--msk1-bin", required=True, type=Path, help="msk1_payloads.bin (len-prefixed payloads)")
    ap.add_argument("--dict-dir", required=True, type=Path, help="dict/ directory with {tid}.png templates (RGBA)")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, only process first N frames (for quick sanity checks).",
    )
    args = ap.parse_args()

    payloads = read_len_prefixed_payloads(args.msk1_bin)
    cap_o = cv2.VideoCapture(str(args.orig))
    cap_r = cv2.VideoCapture(str(args.recovered))
    if not cap_o.isOpened():
        raise SystemExit(f"failed to open orig: {args.orig}")
    if not cap_r.isOpened():
        raise SystemExit(f"failed to open recovered: {args.recovered}")

    frame_w = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_pixels = frame_w * frame_h

    cache: dict[int, np.ndarray] = {}
    idx = 0
    total_alpha_px = 0
    total_mismatch_px = 0
    ref_frames = 0
    raw_frames = 0
    ref_regions = 0

    while True:
        if args.max_frames > 0 and idx >= args.max_frames:
            break
        ok0, f0 = cap_o.read()
        ok1, f1 = cap_r.read()
        if not ok0 or not ok1:
            break
        if idx >= len(payloads):
            idx += 1
            continue

        ver, frame_flags, regs = parse_msk1(payloads[idx])
        # v4: frame_flags==0 => raw frame (no stitching expected)
        if ver >= 4 and frame_flags == 0:
            raw_frames += 1
            idx += 1
            continue

        # If ver<4 we don't have frame_flags; still evaluate regions if any.
        if regs:
            ref_frames += 1
        for r in regs:
            # In our pipeline, flags!=0 means "ref template already available on client"
            if r.flags == 0:
                continue
            ref_regions += 1
            tid = r.tid
            tpl = cache.get(tid)
            if tpl is None:
                p = args.dict_dir / f"{tid}.png"
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0 or img.ndim != 3 or img.shape[2] != 4:
                    cache[tid] = None  # type: ignore[assignment]
                    continue
                cache[tid] = img
                tpl = img
            if tpl is None:
                continue

            if tpl.shape[0] != r.h or tpl.shape[1] != r.w:
                tpl_rs = cv2.resize(tpl, (r.w, r.h), interpolation=cv2.INTER_NEAREST)
            else:
                tpl_rs = tpl

            # clamp ROI to frame
            x0 = max(0, r.x)
            y0 = max(0, r.y)
            x1 = min(frame_w, r.x + r.w)
            y1 = min(frame_h, r.y + r.h)
            if x1 <= x0 or y1 <= y0:
                continue

            rx0 = x0 - r.x
            ry0 = y0 - r.y
            rx1 = rx0 + (x1 - x0)
            ry1 = ry0 + (y1 - y0)

            alpha = tpl_rs[ry0:ry1, rx0:rx1, 3]
            m = alpha > 0
            if not np.any(m):
                continue

            roi0 = f0[y0:y1, x0:x1]
            roi1 = f1[y0:y1, x0:x1]

            # pixel mismatch if any channel differs
            diff = roi0 != roi1
            mismatch = np.any(diff, axis=2) & m

            total_alpha_px += int(np.count_nonzero(m))
            total_mismatch_px += int(np.count_nonzero(mismatch))

        idx += 1

    cap_o.release()
    cap_r.release()

    exact_ratio = 1.0
    if total_alpha_px > 0:
        exact_ratio = 1.0 - (total_mismatch_px / total_alpha_px)

    out = {
        "frames_processed": idx,
        "frame_size": [frame_w, frame_h],
        "frame_pixels": frame_pixels,
        "payload_frames": len(payloads),
        "raw_frames_v4": raw_frames,
        "ref_frames": ref_frames,
        "ref_regions": ref_regions,
        "healed_alpha_pixels": total_alpha_px,
        "healed_alpha_coverage_pct": (100.0 * total_alpha_px / (frame_pixels * max(idx, 1))),
        "mismatch_pixels_in_healed_regions": total_mismatch_px,
        "pixel_exact_ratio_in_healed_regions": exact_ratio,
    }

    print(json.dumps(out, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


