import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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


def read_len_prefixed_payloads(path: Path) -> List[bytes]:
    data = path.read_bytes()
    out: List[bytes] = []
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


def parse_msk1(payload: bytes) -> Tuple[int, int, List[Region]]:
    """
    Parse MSK1 payload (v3/v4). Returns (version, frame_flags, regions).
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
    regs: List[Region] = []
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


def overlay_rgba(dst_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    H, W = dst_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    rx0 = x0 - x
    ry0 = y0 - y
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    src = rgba[ry0:ry1, rx0:rx1].astype(np.float32)
    alpha = (src[:, :, 3:4] / 255.0)
    roi[:] = roi * (1.0 - alpha) + src[:, :, :3] * alpha
    dst_bgr[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path, help="Masked video (client receives this)")
    ap.add_argument("--msk1-bin", required=True, type=Path, help="Len-prefixed MSK1 payloads (one per frame)")
    ap.add_argument("--dict-dir", required=True, type=Path, help="Directory containing template PNGs (e.g., dict/{id}.png)")
    ap.add_argument("--out", required=True, type=Path, help="Recovered output mp4")
    ap.add_argument("--assume-templates-available", action="store_true",
                    help="If set, overlay templates even when region flags indicate 'new'")
    args = ap.parse_args()

    payloads = read_len_prefixed_payloads(args.msk1_bin)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer H.264 if available; fall back to MPEG-4 Part 2 if the build lacks H.264 encoder.
    out = None
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
        if vw.isOpened():
            out = vw
            break
    if out is None:
        raise SystemExit(f"failed to open writer (tried avc1, mp4v): {args.out}")

    cache = {}
    idx = 0
    stitched_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= len(payloads):
            out.write(frame)
            idx += 1
            continue
        ver, frame_flags, regs = parse_msk1(payloads[idx])
        if ver == 0 or frame_flags == 0:
            # raw frame
            out.write(frame)
            idx += 1
            continue
        # apply overlays
        for r in regs:
            is_ref = (r.flags != 0)
            if (not is_ref) and (not args.assume_templates_available):
                continue
            tpl_path = args.dict_dir / f"{r.tid}.png"
            if tpl_path not in cache:
                img = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
                cache[tpl_path] = img
            img = cache.get(tpl_path)
            if img is None or img.size == 0 or img.shape[2] != 4:
                continue
            if img.shape[1] != r.w or img.shape[0] != r.h:
                img_rs = cv2.resize(img, (r.w, r.h), interpolation=cv2.INTER_NEAREST)
            else:
                img_rs = img
            overlay_rgba(frame, img_rs, r.x, r.y)
        stitched_frames += 1
        out.write(frame)
        idx += 1

    out.release()
    cap.release()
    meta = {
        "frames_written": idx,
        "stitched_frames": stitched_frames,
        "assume_templates_available": args.assume_templates_available,
    }
    (args.out.parent / "stitch_report.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote recovered video: {args.out}")


if __name__ == "__main__":
    main()


