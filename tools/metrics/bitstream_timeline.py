#!/usr/bin/env python3
"""
Visualize where each visual frame lives in the bitstream, and whether there is
periodic behavior within each GOP.

Inputs:
  --orig   baseline/original MP4
  --masked masked/segmented MP4

Outputs (under --out-dir):
  - timeline.csv: per-frame bytes + pkt_pos + timestamps + GOP position + delta
  - delta_by_gop_pos.csv: mean/median delta by GOP position
  - delta_timeseries.png: bytes and delta over time
  - delta_gop_heatmap.png: GOP index vs GOP position heatmap of delta bytes
  - pos_vs_pts.png: pkt_pos vs pts_time scatter (baseline and masked)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameInfo:
    idx: int
    pts_time: float | None
    dts_time: float | None
    best_effort_time: float | None
    pkt_pos: int | None
    pkt_size: int
    pict_type: str | None
    key_frame: int | None


@dataclass(frozen=True)
class PacketInfo:
    idx: int
    pts_time: float | None
    dts_time: float | None
    pos: int | None
    size: int


def _ffprobe_frames_full(mp4: Path) -> list[FrameInfo]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,pkt_pos,pkt_size,pict_type,key_frame",
        "-of",
        "json",
        str(mp4),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    out: list[FrameInfo] = []
    for i, fr in enumerate(data.get("frames", []) or []):
        def _f(key: str) -> float | None:
            v = fr.get(key)
            return None if v is None else float(v)

        def _i(key: str) -> int | None:
            v = fr.get(key)
            return None if v is None else int(v)

        out.append(
            FrameInfo(
                idx=i,
                pts_time=_f("pkt_pts_time"),
                dts_time=_f("pkt_dts_time"),
                best_effort_time=_f("best_effort_timestamp_time"),
                pkt_pos=_i("pkt_pos"),
                pkt_size=int(fr.get("pkt_size", 0) or 0),
                pict_type=(str(fr.get("pict_type")) if fr.get("pict_type") is not None else None),
                key_frame=_i("key_frame"),
            )
        )
    return out


def _ffprobe_packets_full(mp4: Path) -> list[PacketInfo]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,pos,size",
        "-of",
        "json",
        str(mp4),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    out: list[PacketInfo] = []
    for i, p in enumerate(data.get("packets", []) or []):
        pts = p.get("pts_time")
        dts = p.get("dts_time")
        pos = p.get("pos")
        sz = p.get("size")
        out.append(
            PacketInfo(
                idx=i,
                pts_time=None if pts is None else float(pts),
                dts_time=None if dts is None else float(dts),
                pos=None if pos is None else int(pos),
                size=int(sz or 0),
            )
        )
    return out


def _choose_time(fi: FrameInfo) -> float | None:
    # Prefer best-effort (presentation-ish), fallback to pts.
    return fi.best_effort_time if fi.best_effort_time is not None else fi.pts_time


def _align_index(a: list[FrameInfo], b: list[FrameInfo]) -> list[tuple[FrameInfo, FrameInfo]]:
    n = min(len(a), len(b))
    return [(a[i], b[i]) for i in range(n)]


def _gop_positions_from_baseline(base: list[FrameInfo]) -> tuple[list[int], list[int]]:
    """
    Return (gop_id[i], gop_pos[i]) for each frame i, derived from baseline keyframes.
    """
    gop_id: list[int] = []
    gop_pos: list[int] = []
    cur_gid = 0
    cur_pos = 0
    first = True
    for i, fr in enumerate(base):
        is_kf = int(fr.key_frame or 0) == 1 or (fr.pict_type == "I")
        if first:
            # frame 0 starts GOP 0 even if ffprobe doesn't mark it as keyframe
            first = False
            cur_gid = 0
            cur_pos = 0
        elif is_kf:
            cur_gid += 1
            cur_pos = 0
        else:
            cur_pos += 1
        gop_id.append(cur_gid)
        gop_pos.append(cur_pos)
    return gop_id, gop_pos


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _safe_float(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--masked", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--align", choices=["index"], default="index")
    ap.add_argument(
        "--size-source",
        choices=["frames", "packets"],
        default="frames",
        help="Where to get size/pos from. 'frames' uses ffprobe -show_frames pkt_size/pkt_pos. "
        "'packets' uses ffprobe -show_packets size/pos (more reliable for AV1 where frame pkt_size can be missing).",
    )
    args = ap.parse_args()

    # Lazy imports (so the CSV-only path can run without plotting deps).
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe; avoid Qt backend issues
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            f"Missing plotting deps (numpy/matplotlib). Run under the conda env. Error: {e}"
        )

    # We always use show_frames for GOP/keyframe/pict_type metadata.
    base_frames = _ffprobe_frames_full(args.orig)
    mask_frames = _ffprobe_frames_full(args.masked)
    frame_pairs = _align_index(base_frames, mask_frames)
    base_gop_id, base_gop_pos = _gop_positions_from_baseline([a for a, _ in frame_pairs])

    # Size/pos can come from frames (pkt_size/pkt_pos) or packets (pos/size).
    if args.size_source == "packets":
        base_pkts = _ffprobe_packets_full(args.orig)
        mask_pkts = _ffprobe_packets_full(args.masked)
        pkt_pairs = _align_index(base_pkts, mask_pkts)
        n = min(len(frame_pairs), len(pkt_pairs), len(base_gop_id), len(base_gop_pos))
    else:
        pkt_pairs = None
        n = min(len(frame_pairs), len(base_gop_id), len(base_gop_pos))

    rows: list[dict[str, Any]] = []
    for i in range(n):
        a, b = frame_pairs[i]
        if pkt_pairs is not None:
            pa, pb = pkt_pairs[i]
            size_base = pa.size
            size_mask = pb.size
            pos_base = pa.pos
            pos_mask = pb.pos
            t = pa.pts_time if pa.pts_time is not None else _choose_time(a)
            dts_t = pa.dts_time if pa.dts_time is not None else a.dts_time
        else:
            size_base = a.pkt_size
            size_mask = b.pkt_size
            pos_base = a.pkt_pos
            pos_mask = b.pkt_pos
            t = _choose_time(a)
            dts_t = a.dts_time
        rows.append(
            {
                "frame_idx": i,
                "pts_time": _safe_float(t),
                "dts_time": _safe_float(dts_t),
                "pkt_pos_base": pos_base if pos_base is not None else "",
                "pkt_pos_mask": pos_mask if pos_mask is not None else "",
                "pkt_size_base": int(size_base),
                "pkt_size_mask": int(size_mask),
                "delta_bytes": int(size_base) - int(size_mask),
                "pict_type_base": a.pict_type or "",
                "key_frame_base": int(a.key_frame or 0),
                "gop_id": base_gop_id[i],
                "gop_pos": base_gop_pos[i],
                "size_source": args.size_source,
            }
        )

    out_csv = args.out_dir / "timeline.csv"
    _write_csv(out_csv, rows)

    # Aggregate by GOP position
    by_pos: dict[int, list[int]] = {}
    for r in rows:
        pos = int(r["gop_pos"])
        by_pos.setdefault(pos, []).append(int(r["delta_bytes"]))

    agg_rows: list[dict[str, Any]] = []
    for pos, vals in sorted(by_pos.items()):
        v = np.array(vals, dtype=np.int64)
        agg_rows.append(
            {
                "gop_pos": pos,
                "count": int(v.size),
                "delta_mean": float(v.mean()) if v.size else 0.0,
                "delta_median": float(np.median(v)) if v.size else 0.0,
                "delta_p10": float(np.quantile(v, 0.10)) if v.size else 0.0,
                "delta_p90": float(np.quantile(v, 0.90)) if v.size else 0.0,
            }
        )
    _write_csv(args.out_dir / "delta_by_gop_pos.csv", agg_rows)

    # Plots
    t = np.array([r["pts_time"] for r in rows], dtype=np.float64)
    bsz = np.array([r["pkt_size_base"] for r in rows], dtype=np.float64)
    msz = np.array([r["pkt_size_mask"] for r in rows], dtype=np.float64)
    dlt = np.array([r["delta_bytes"] for r in rows], dtype=np.float64)

    # Timeseries
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, bsz, label="baseline pkt_size", linewidth=0.8)
    ax[0].plot(t, msz, label="masked pkt_size", linewidth=0.8)
    ax[0].set_ylabel("bytes")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)
    ax[1].plot(t, dlt, label="delta (base-mask)", linewidth=0.8)
    ax[1].axhline(0, color="k", linewidth=0.7)
    ax[1].set_ylabel("delta bytes")
    ax[1].set_xlabel("time (s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "delta_timeseries.png")
    plt.close(fig)

    # Heatmap GOP_id x GOP_pos
    max_gid = max(base_gop_id) if base_gop_id else 0
    max_pos = max(base_gop_pos) if base_gop_pos else 0
    H = np.full((max_gid + 1, max_pos + 1), np.nan, dtype=np.float64)
    # Average deltas for each (gid,pos)
    buckets: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        key = (int(r["gop_id"]), int(r["gop_pos"]))
        buckets.setdefault(key, []).append(float(r["delta_bytes"]))
    for (gid, pos), vals in buckets.items():
        H[gid, pos] = float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(H, aspect="auto", interpolation="nearest")
    ax.set_title("Delta bytes heatmap (GOP index vs position)")
    ax.set_xlabel("GOP position (frames since keyframe)")
    ax.set_ylabel("GOP index")
    fig.colorbar(im, ax=ax, label="delta bytes (base-mask)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "delta_gop_heatmap.png")
    plt.close(fig)

    # pkt_pos vs pts_time scatter
    def _pos_from_rows(key: str) -> np.ndarray:
        return np.array(
            [float(r[key]) if r[key] != "" else np.nan for r in rows],
            dtype=np.float64,
        )

    base_pos = _pos_from_rows("pkt_pos_base")
    mask_pos = _pos_from_rows("pkt_pos_mask")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(t, base_pos, s=3, alpha=0.5, label="baseline pkt_pos")
    ax.scatter(t, mask_pos, s=3, alpha=0.5, label="masked pkt_pos")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pkt_pos (byte offset in file)")
    ax.set_title("Where frames live in the MP4 bitstream (pkt_pos vs pts_time)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pos_vs_pts.png")
    plt.close(fig)

    print(f"WROTE {out_csv}")
    print(f"WROTE {args.out_dir / 'delta_by_gop_pos.csv'}")
    print(f"WROTE plots under {args.out_dir}")


if __name__ == "__main__":
    main()

