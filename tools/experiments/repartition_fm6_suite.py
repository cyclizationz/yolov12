#!/usr/bin/env python3
"""Merge selected normalized FM6 clips and repartition them into equal clips."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    REPO / "record/RESPAWN2026/manifest/normalized/fm6/fm6_00.mp4",
    REPO / "record/RESPAWN2026/manifest/normalized/fm6/fm6_01.mp4",
    REPO / "record/RESPAWN2026/manifest/normalized/fm6/fm6_02.mp4",
    REPO / "record/RESPAWN2026/manifest/normalized/fm6/fm6_04.mp4",
]


def frame_count(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nokey=1:nw=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(output)


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_manifest(out_root: Path, outputs: list[Path], frames_per_clip: int) -> None:
    duration = frames_per_clip / 60.0
    rows = []
    for idx, output in enumerate(outputs):
        rows.append(
            {
                "clip_id": f"fm6_{idx:02d}",
                "game": "fm6",
                "family": "learned",
                "source_tag": "fm6_repartitioned_without_fm6_03",
                "source_path": str(out_root / "repartition_provenance.json"),
                "clip_path": str(output),
                "normalized_path": str(output),
                "start_s": idx * duration,
                "duration_s": duration,
                "source_width": 1920,
                "source_height": 1080,
                "source_fps": 60.0,
                "target_width": 1920,
                "target_height": 1080,
                "target_fps": 60.0,
                "eligible_duo": True,
                "eligible_rd": True,
                "notes": (
                    "Equal-frame repartition of normalized fm6_00, fm6_01, "
                    "fm6_02, and fm6_04; fm6_03 excluded by requested policy."
                ),
            }
        )
    (out_root / "offline_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out_root / "offline_manifest.csv").open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "record/RESPAWN2026/manifest_fm6_repartitioned",
    )
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    normalized_dir = args.out_root / "normalized/fm6"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    counts = [frame_count(path) for path in args.inputs]
    total = sum(counts)
    if total % args.clips:
        raise SystemExit(
            f"{total} frames cannot be divided exactly into {args.clips} clips"
        )
    frames_per_clip = total // args.clips
    outputs = [normalized_dir / f"fm6_{idx:02d}.mp4" for idx in range(args.clips)]
    if all(path.exists() for path in outputs) and not args.force:
        print("[skip] repartitioned clips already exist")
    else:
        with tempfile.TemporaryDirectory(prefix="fm6-repartition-") as temp_dir:
            concat_list = Path(temp_dir) / "inputs.txt"
            concat_list.write_text(
                "".join(f"file '{path.resolve()}'\n" for path in args.inputs)
            )
            merged = Path(temp_dir) / "merged.mp4"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-an",
                    "-c",
                    "copy",
                    str(merged),
                ]
            )
            for idx, output in enumerate(outputs):
                start = idx * frames_per_clip
                stop = start + frames_per_clip
                run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(merged),
                        "-vf",
                        f"trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "12",
                        "-pix_fmt",
                        "yuv420p",
                        str(output),
                    ]
                )

    actual = [frame_count(path) for path in outputs]
    if actual != [frames_per_clip] * args.clips:
        raise SystemExit(f"Unexpected output frame counts: {actual}")
    provenance = {
        "policy": "discard fm6_03; concatenate remaining four; equal-frame split",
        "inputs": [
            {"path": str(path.resolve()), "frames": count}
            for path, count in zip(args.inputs, counts)
        ],
        "total_frames": total,
        "clips": args.clips,
        "frames_per_clip": frames_per_clip,
        "duration_per_clip_s": frames_per_clip / 60.0,
        "normalization": "decoded merged stream re-encoded libx264 veryfast CRF12 yuv420p",
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (args.out_root / "repartition_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    write_manifest(args.out_root, outputs, frames_per_clip)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
