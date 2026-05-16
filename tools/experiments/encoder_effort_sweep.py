#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    codec: str
    crf: int
    preset: str
    gop: int


DEFAULT_CONFIGS = [
    EncoderConfig("svtav1_crf38_p12_g60", "libsvtav1", 38, "12", 60),
    EncoderConfig("svtav1_crf42_p12_g60", "libsvtav1", 42, "12", 60),
    EncoderConfig("svtav1_crf46_p12_g60", "libsvtav1", 46, "12", 60),
    EncoderConfig("svtav1_crf42_p8_g60", "libsvtav1", 42, "8", 60),
    EncoderConfig("x264_crf18_superfast_g60", "libx264", 18, "superfast", 60),
    EncoderConfig("x264_crf23_superfast_g60", "libx264", 23, "superfast", 60),
    EncoderConfig("x264_crf18_medium_g60", "libx264", 18, "medium", 60),
]


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def packet_sizes(path: Path) -> list[int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,size",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    packets = data.get("packets", [])

    def pts(packet: dict[str, str]) -> float:
        try:
            return float(packet.get("pts_time", 0.0))
        except ValueError:
            return 0.0

    return [int(packet["size"]) for packet in sorted(packets, key=pts)]


def encode_sequence(frame_dir: Path, out_video: Path, cfg: EncoderConfig, fps: float, dry_run: bool) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:.6g}",
        "-i",
        str(frame_dir / "frame%05d.png"),
        "-an",
        "-c:v",
        cfg.codec,
        "-preset",
        cfg.preset,
        "-g",
        str(cfg.gop),
        "-bf",
        "0",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(cfg.crf),
        str(out_video),
    ]
    run(cmd, dry_run)


def measure_pair(original_video: Path, masked_video: Path) -> dict[str, float]:
    original = packet_sizes(original_video)
    masked = packet_sizes(masked_video)
    n = min(len(original), len(masked))
    original = original[:n]
    masked = masked[:n]
    original_bytes = sum(original)
    masked_bytes = sum(masked)
    negative = [(m - b) for b, m in zip(original, masked) if m > b]
    positive = [(b - m) for b, m in zip(original, masked) if b >= m]
    negative_overhead = sum(negative)
    positive_saving = sum(positive)
    return {
        "frames": n,
        "original_packet_bytes": original_bytes,
        "masked_packet_bytes": masked_bytes,
        "packet_bsp_pct": (1.0 - masked_bytes / original_bytes) * 100.0 if original_bytes else 0.0,
        "file_bsp_pct": (1.0 - masked_video.stat().st_size / original_video.stat().st_size) * 100.0,
        "negative_frame_pct": (len(negative) / n) * 100.0 if n else 0.0,
        "weighted_negative_contribution_pct": (
            negative_overhead / (negative_overhead + positive_saving) * 100.0
            if (negative_overhead + positive_saving) > 0
            else 0.0
        ),
    }


def stage_frames(original_dir: Path, masked_dir: Path, out_dir: Path, subset: str) -> tuple[Path, Path]:
    frame_root = out_dir / "frames" / subset
    if frame_root.exists():
        shutil.rmtree(frame_root)
    original_stage = frame_root / "original"
    masked_stage = frame_root / "masked"
    original_stage.mkdir(parents=True)
    masked_stage.mkdir(parents=True)
    original_frames = sorted(original_dir.glob("*.png"))
    masked_frames = sorted(masked_dir.glob("*.png"))
    if len(original_frames) != len(masked_frames):
        raise ValueError(f"Frame count mismatch: {len(original_frames)} original vs {len(masked_frames)} masked")
    if subset == "all":
        indices = range(len(original_frames))
    elif subset == "odd":
        indices = range(0, len(original_frames), 2)
    else:
        raise ValueError(f"Unsupported subset: {subset}")
    for out_idx, src_idx in enumerate(indices, start=1):
        (original_stage / f"frame{out_idx:05d}.png").symlink_to(original_frames[src_idx].resolve())
        (masked_stage / f"frame{out_idx:05d}.png").symlink_to(masked_frames[src_idx].resolve())
    return original_stage, masked_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep encoder settings over paired original/masked PNG frame dumps.")
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--masked-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.303)
    parser.add_argument("--subset", choices=["all", "odd"], action="append", default=["all", "odd"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str | float | int]] = []
    for subset in args.subset:
        original_frames, masked_frames = stage_frames(args.original_dir, args.masked_dir, args.out_dir, subset)
        for cfg in DEFAULT_CONFIGS:
            original_video = args.out_dir / "videos" / subset / f"{cfg.name}_original.mp4"
            masked_video = args.out_dir / "videos" / subset / f"{cfg.name}_masked.mp4"
            encode_sequence(original_frames, original_video, cfg, args.fps, args.dry_run)
            encode_sequence(masked_frames, masked_video, cfg, args.fps, args.dry_run)
            if args.dry_run:
                continue
            metrics = measure_pair(original_video, masked_video)
            rows.append(
                {
                    "subset": subset,
                    "config": cfg.name,
                    "codec": cfg.codec,
                    "crf": cfg.crf,
                    "preset": cfg.preset,
                    "gop": cfg.gop,
                    **metrics,
                }
            )

    if rows:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "encoder_effort_sweep.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
