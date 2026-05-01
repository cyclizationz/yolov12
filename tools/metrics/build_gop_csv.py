#!/usr/bin/env python3
"""
Build a per-GOP CSV from an experiment output folder.

Inputs (expected in --out-dir):
  - report.json
  - original_output.mp4
  - segmented_output.mp4

We define GOP boundaries using baseline (original_output.mp4) I-frames (pict_type == 'I')
in decode order. For each GOP interval [start, end), we sum baseline/masked pkt_size over
the same frame-index range in both MP4s (requires same frame count; we validate).

We also aggregate changed_pixels_pct (content removed) from report.json/per_frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def ffprobe_pict_types(mp4: Path) -> list[str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pict_type",
        "-of",
        "json",
        str(mp4),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    out: list[str] = []
    for fr in data.get("frames", []) or []:
        pt = fr.get("pict_type")
        if pt is None:
            continue
        out.append(str(pt))
    return out


def ffprobe_pkt_sizes(mp4: Path) -> list[int]:
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
        str(mp4),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    out: list[int] = []
    for fr in data.get("frames", []) or []:
        sz = fr.get("pkt_size")
        if sz is None:
            continue
        out.append(int(sz))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=30, help="Only used to annotate time (seconds) in output.")
    ap.add_argument(
        "--group-mode",
        choices=["iframe", "fixed"],
        default="fixed",
        help="How to define groups. 'iframe' uses baseline I-frames (true GOPs). "
        "'fixed' uses fixed-size frame windows (robust when encoder keyint is unknown).",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=30,
        help="Group size in frames when --group-mode=fixed (default: 30 ~= 1s at 30FPS).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    report_path = out_dir / "report.json"
    base_mp4 = out_dir / "original_output.mp4"
    masked_mp4 = out_dir / "segmented_output.mp4"

    rep: dict[str, Any] = json.loads(report_path.read_text())
    per = rep.get("per_frame", []) or []
    if not isinstance(per, list) or not per:
        raise SystemExit(f"{report_path} has no per_frame entries (need a non-index run).")

    base_sizes = ffprobe_pkt_sizes(base_mp4)
    masked_sizes = ffprobe_pkt_sizes(masked_mp4)
    base_types = ffprobe_pict_types(base_mp4)

    n = min(len(per), len(base_sizes), len(masked_sizes), len(base_types))
    if len(base_sizes) != len(masked_sizes):
        raise SystemExit(f"Frame count mismatch: baseline={len(base_sizes)} masked={len(masked_sizes)}")

    if args.group_mode == "iframe":
        # GOP starts at every I-frame (including frame 0 even if not I).
        starts = [0]
        for i in range(1, n):
            if base_types[i] == "I":
                starts.append(i)
        if starts[-1] != n:
            starts.append(n)
    else:
        g = max(1, int(args.group_size))
        starts = list(range(0, n, g))
        if starts[-1] != n:
            starts.append(n)

    out_rows: list[dict[str, Any]] = []
    for gid in range(len(starts) - 1):
        s = starts[gid]
        e = starts[gid + 1]
        if e <= s:
            continue

        base_sum = int(sum(base_sizes[s:e]))
        mask_sum = int(sum(masked_sizes[s:e]))
        delta = int(base_sum - mask_sum)
        saving_pct = (100.0 * delta / base_sum) if base_sum > 0 else 0.0

        # mean changed_pixels_pct over frames in the GOP
        cps: list[float] = []
        present = 0
        masked_ok = 0
        for i in range(s, e):
            cps.append(float(per[i].get("changed_pixels_pct", 0.0) or 0.0))
            if int(per[i].get("object_present_model", 0) or 0) >= 1:
                present += 1
                if int(per[i].get("object_successfully_masked", 0) or 0) >= 1:
                    masked_ok += 1
        ch_mean = sum(cps) / max(1, len(cps))
        succ = (masked_ok / present) if present > 0 else 0.0

        out_rows.append(
            {
                "group_id": gid,
                "start_frame_id": s + 1,  # 1-based for humans
                "end_frame_id": e,  # inclusive end would be e, but we keep half-open; end_frame_id is last+1
                "frames": e - s,
                "start_sec": (s / max(1, args.fps)),
                "baseline_bytes_sum": base_sum,
                "masked_bytes_sum": mask_sum,
                "bytes_delta_sum": delta,
                "saving_pct": saving_pct,
                "changed_pixels_pct_mean": ch_mean,
                "present_frames": present,
                "masked_frames": masked_ok,
                "success_rate_given_present": succ,
                "group_mode": args.group_mode,
                "group_size_frames": int(args.group_size) if args.group_mode == "fixed" else "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {args.out} ({len(out_rows)} GOPs, {n} frames)")


if __name__ == "__main__":
    main()

