#!/usr/bin/env python3
"""
Extract per-frame encoded sizes from a video stream using ffprobe -show_frames.

Why -show_frames instead of -show_packets?
- Packets are not guaranteed 1:1 with frames, especially across different encodes.
- Frame-level timestamps are what we want for aligning baseline vs masked outputs.

We align frames by best_effort_timestamp_time (presentation order) within a tolerance.

This is used to populate baseline_bytes/masked_bytes columns in per_frame_metrics.csv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def ffprobe_frames(video_path: Path) -> list[tuple[float, int]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_size,pict_type",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    frames: list[tuple[float, int]] = []
    for fr in data.get("frames", []) or []:
        # Only keep video frames that have a timestamp and size.
        # best_effort_timestamp_time is robust even when pts is missing.
        pts_t = fr.get("best_effort_timestamp_time")
        sz = fr.get("pkt_size")
        if pts_t is None or sz is None:
            continue
        frames.append((float(pts_t), int(sz)))
    frames.sort(key=lambda x: x[0])
    return frames


def ffprobe_frame_sizes_in_decode_order(video_path: Path) -> list[int]:
    """
    Return pkt_size per decoded frame in presentation/decode order as emitted by ffprobe -show_frames.

    This is preferable when comparing two MP4s produced by the same pipeline (same frame count),
    because it avoids any ambiguity from timestamps (duplicates/missing best_effort_timestamp_time).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pkt_size",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    sizes: list[int] = []
    for fr in data.get("frames", []) or []:
        sz = fr.get("pkt_size")
        if sz is None:
            continue
        sizes.append(int(sz))
    return sizes


def align_by_time(
    a: list[tuple[float, int]], b: list[tuple[float, int]], tol_ms: float
) -> tuple[list[float], list[int], list[int]]:
    tol = tol_ms / 1000.0
    i = 0
    j = 0
    ts: list[float] = []
    asz: list[int] = []
    bsz: list[int] = []
    while i < len(a) and j < len(b):
        t0, s0 = a[i]
        t1, s1 = b[j]
        dt = t0 - t1
        if abs(dt) <= tol:
            ts.append(t0)
            asz.append(s0)
            bsz.append(s1)
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    return ts, asz, bsz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--masked", required=True, type=Path)
    ap.add_argument("--tol-ms", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    p0 = ffprobe_frames(args.baseline)
    p1 = ffprobe_frames(args.masked)
    ts, b0, b1 = align_by_time(p0, p1, tol_ms=args.tol_ms)
    res = {
        "pairs": len(ts),
        "baseline": str(args.baseline),
        "masked": str(args.masked),
        "pts_time": ts,
        "baseline_bytes": b0,
        "masked_bytes": b1,
    }
    txt = json.dumps(res)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()


