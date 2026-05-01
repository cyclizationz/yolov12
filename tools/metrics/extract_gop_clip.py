#!/usr/bin/env python3
"""
Extract a single GOP-aligned clip from baseline+masked MP4s under an output directory.

This is intended for debugging per-frame vs bitstream behavior:
- pick a GOP index (derived from baseline keyframes)
- extract the exact time window [kf_i, kf_{i+1}) into a small mp4
- optionally dump AnnexB elementary streams for bitstream inspection

Usage:
  python tools/metrics/extract_gop_clip.py --out-dir <run_out_dir> --gop-id 10 --out <clip_dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _ffprobe_keyframes(mp4: Path) -> list[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,key_frame,pict_type",
        "-of",
        "json",
        str(mp4),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    times: list[float] = []
    first = True
    for fr in data.get("frames", []) or []:
        t = fr.get("best_effort_timestamp_time")
        if t is None:
            continue
        is_kf = int(fr.get("key_frame", 0) or 0) == 1 or (fr.get("pict_type") == "I")
        if first:
            # always treat frame 0 time as GOP start
            times.append(float(t))
            first = False
            continue
        if is_kf:
            times.append(float(t))
    return times


def _ffmpeg_cut(inp: Path, t0: float, t1: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # We re-encode to avoid non-keyframe copy pitfalls across MP4s.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t0:.6f}",
        "-to",
        f"{t1:.6f}",
        "-i",
        str(inp),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(out),
    ]
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path, help="Run output dir with original_output.mp4 and segmented_output.mp4")
    ap.add_argument("--gop-id", required=True, type=int, help="0-based GOP index (by baseline keyframes)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for the extracted clip")
    args = ap.parse_args()

    base = args.out_dir / "original_output.mp4"
    mask = args.out_dir / "segmented_output.mp4"
    if not base.exists() or not mask.exists():
        raise SystemExit("Missing original_output.mp4 or segmented_output.mp4 under --out-dir")

    kf = _ffprobe_keyframes(base)
    if len(kf) < 2:
        raise SystemExit("Not enough keyframes found to define GOP intervals.")

    gid = int(args.gop_id)
    if gid < 0 or gid >= len(kf) - 1:
        raise SystemExit(f"gop-id out of range: 0..{len(kf)-2}")

    t0 = kf[gid]
    t1 = kf[gid + 1]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps({"gop_id": gid, "t0": t0, "t1": t1}, indent=2))

    _ffmpeg_cut(base, t0, t1, args.out / "baseline_clip.mp4")
    _ffmpeg_cut(mask, t0, t1, args.out / "masked_clip.mp4")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

