#!/usr/bin/env python3
"""
Export top-K frames (by reconstruction quality) into a preview folder:
  - original frame (from original_output.mp4)
  - recovered frame (from recovered_output.mp4)
  - the template PNG used (from dict/{tid}.png), chosen as the largest-area region in MSK1 payload

Output layout:
  <out-root>/<frame_id>/original.png
  <out-root>/<frame_id>/recovered.png
  <out-root>/<frame_id>/template.png
  <out-root>/<frame_id>/meta.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _read_len_prefixed_payloads(path: Path) -> list[bytes]:
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


def _be_u32(b: bytes, off: int) -> tuple[int, int]:
    return int.from_bytes(b[off : off + 4], "big"), off + 4


def _be_u16(b: bytes, off: int) -> tuple[int, int]:
    return int.from_bytes(b[off : off + 2], "big"), off + 2


def _be_u64(b: bytes, off: int) -> tuple[int, int]:
    return int.from_bytes(b[off : off + 8], "big"), off + 8


@dataclass(frozen=True)
class Region:
    tid: int
    x: int
    y: int
    w: int
    h: int
    flags: int
    class_id: int
    path: str


def _parse_msk1(payload: bytes) -> tuple[int, int, list[Region]]:
    """
    Parse MSK1 payload (v3/v4). Returns (version, frame_flags, regions).
    """
    if len(payload) < 4 + 2 + 8 + 8 + 4:
        return 0, 0, []
    off = 0
    magic, off = _be_u32(payload, off)
    if magic != 0x4D534B31:  # "MSK1"
        return 0, 0, []
    ver, off = _be_u16(payload, off)
    _, off = _be_u64(payload, off)  # frame_counter
    _, off = _be_u64(payload, off)  # pts
    frame_flags = 0
    if ver >= 4:
        frame_flags = int(payload[off])
        off += 1
    nreg, off = _be_u32(payload, off)
    regs: list[Region] = []
    for _ in range(int(nreg)):
        tid, off = _be_u32(payload, off)
        x, off = _be_u32(payload, off)
        y, off = _be_u32(payload, off)
        w, off = _be_u32(payload, off)
        h, off = _be_u32(payload, off)
        flags = 1
        if ver >= 3:
            flags = int(payload[off])
            off += 1
        class_id = int(payload[off])
        off += 1
        L = int(payload[off])
        off += 1
        path_s = payload[off : off + L].decode("utf-8", errors="ignore") if L else ""
        off += L
        regs.append(Region(int(tid), int(x), int(y), int(w), int(h), int(flags), int(class_id), path_s))
    return int(ver), int(frame_flags), regs


def _extract_frame_png(mp4: Path, frame0: int, out_png: Path) -> None:
    # frame0 is 0-based.
    vf = f"select=eq(n\\,{int(frame0)})"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vf", vf, "-vframes", "1", str(out_png)]
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--metric", choices=["rec_ssim", "rec_psnr"], default="rec_ssim")
    ap.add_argument("--require-template", action="store_true", help="Only select frames with at least one MSK1 region (so template.png exists)")
    args = ap.parse_args()

    run_dir = args.run_dir
    rep = json.loads((run_dir / "report.json").read_text())
    per = rep.get("per_frame", []) or []

    msk1_bin = run_dir / "msk1_payloads.bin"
    payloads = _read_len_prefixed_payloads(msk1_bin) if msk1_bin.exists() else []

    orig_mp4 = run_dir / "original_output.mp4"
    rec_mp4 = run_dir / "recovered_output.mp4"
    dict_dir = run_dir / "dict"

    # Select top-K frames by metric (tie-break: other metric).
    scored: list[tuple[float, float, int]] = []
    for i, e in enumerate(per):
        a = e.get(args.metric)
        b = e.get("rec_psnr" if args.metric == "rec_ssim" else "rec_ssim")
        if a is None:
            continue
        try:
            fa = float(a)
            fb = float(b) if b is not None else 0.0
        except Exception:
            continue
        if args.require_template and payloads and i < len(payloads):
            ver, frame_flags, regs = _parse_msk1(payloads[i])
            if len(regs) <= 0:
                continue
        scored.append((fa, fb, i))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = scored[: max(0, int(args.topk))]

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for fa, fb, frame0 in top:
        frame_id = frame0 + 1
        out_dir = out_root / str(frame_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        regs: list[Region] = []
        tid: int | None = None
        ver = 0
        frame_flags = 0
        if frame0 < len(payloads):
            ver, frame_flags, regs = _parse_msk1(payloads[frame0])
            if regs:
                best = max(regs, key=lambda r: int(r.w) * int(r.h))
                tid = int(best.tid)

        _extract_frame_png(orig_mp4, frame0, out_dir / "original.png")
        _extract_frame_png(rec_mp4, frame0, out_dir / "recovered.png")

        if tid is not None:
            tpl = dict_dir / f"{tid}.png"
            if tpl.exists():
                shutil.copyfile(tpl, out_dir / "template.png")

        meta = {
            "run_dir": str(run_dir),
            "frame0": frame0,
            "frame_id": frame_id,
            "metric": args.metric,
            "metric_value": fa,
            "tie_value": fb,
            "msk1_version": ver,
            "msk1_frame_flags": frame_flags,
            "template_tid": tid,
            "regions": [
                {
                    "tid": int(r.tid),
                    "x": int(r.x),
                    "y": int(r.y),
                    "w": int(r.w),
                    "h": int(r.h),
                    "flags": int(r.flags),
                    "class_id": int(r.class_id),
                    "path": str(r.path),
                }
                for r in regs
            ],
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(str(out_root))
    print("frames:", [i + 1 for _, _, i in top])


if __name__ == "__main__":
    main()

