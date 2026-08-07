#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]
DATASET = Path("/home/tiehangz/proj/datasets/spaceflight")
DEFAULT_MODEL = REPO / "deployment" / "yolov12n_spaceflight_cockpit_spaceship_e300_v1.pt"
DEFAULT_OUT = REPO / "record" / "RESPAWN2026" / "spaceflight_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_video(model: YOLO, path: Path, conf: float, batch_size: int) -> tuple[float, list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    rows: list[dict[str, Any]] = []
    frames: list[Any] = []
    frame_ids: list[int] = []
    index = 0

    def flush() -> None:
        if not frames:
            return
        results = model.predict(frames, imgsz=640, conf=conf, device=0, verbose=False, batch=len(frames))
        for frame_id, result in zip(frame_ids, results):
            classes = [int(v) for v in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
            confidences = [float(v) for v in result.boxes.conf.cpu().tolist()] if result.boxes is not None else []
            mask_pixels = []
            if result.masks is not None:
                mask_pixels = [int(mask.sum().item()) for mask in result.masks.data]
            rows.append(
                {
                    "frame_index": frame_id,
                    "timestamp_s": frame_id / fps,
                    "positive": bool(classes) and (not mask_pixels or max(mask_pixels) > 0),
                    "classes": classes,
                    "confidences": confidences,
                    "mask_pixels": mask_pixels,
                }
            )
        frames.clear()
        frame_ids.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        frame_ids.append(index)
        index += 1
        if len(frames) >= batch_size:
            flush()
    flush()
    cap.release()
    return fps, rows


def positive_runs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, row in enumerate(rows + [{"positive": False}]):
        if row["positive"] and start is None:
            start = idx
        elif not row["positive"] and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def choose_spans(
    scans: dict[str, dict[str, Any]], labels: list[dict[str, Any]], span_frames: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label in labels:
        if label.get("status") != "accepted":
            continue
        stem = Path(label["source_clip"]).stem
        scan = scans[stem]
        anchor = int(round((float(label["timestamp_ms"]) / 1000.0) * scan["fps"]))
        for run_start, run_end in scan["runs"]:
            if not (run_start <= anchor <= run_end) or run_end - run_start + 1 < span_frames:
                continue
            start = max(run_start, min(anchor - span_frames // 2, run_end - span_frames + 1))
            end = start + span_frames - 1
            rows = scan["rows"][start : end + 1]
            label_class = int(label["class_id"])
            if label_class not in rows[anchor - start]["classes"]:
                continue
            candidates.append(
                {
                    "source_stem": stem,
                    "source_path": str(DATASET / label["source_clip"]),
                    "start_frame": start,
                    "end_frame": end,
                    "anchor_timestamp_ms": int(label["timestamp_ms"]),
                    "anchor_class_id": label_class,
                    "anchor_class_name": label["class_name"],
                    "minimum_confidence": min(max(row["confidences"]) for row in rows),
                }
            )
            break

    desired = [
        ("flight_part1", 1),
        ("flight_part2", 0),
        ("flight_part3", 0),
        ("flight_part1", 0),
        ("flight_part3", 1),
    ]
    selected: list[dict[str, Any]] = []
    for stem, class_id in desired:
        choices = sorted(
            (c for c in candidates if c["source_stem"] == stem and c["anchor_class_id"] == class_id),
            key=lambda c: c["minimum_confidence"],
            reverse=True,
        )
        choice = next(
            (
                c
                for c in choices
                if all(
                    c["source_stem"] != prior["source_stem"]
                    or c["end_frame"] < prior["start_frame"]
                    or c["start_frame"] > prior["end_frame"]
                    for prior in selected
                )
            ),
            None,
        )
        if choice is None:
            raise RuntimeError(f"No non-overlapping fully positive span for {stem}, class {class_id}")
        selected.append(choice)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create five frame-verified two-class spaceflight clips.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--span-frames", type=int, default=90)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out_dir / "clips"
    predictions_dir = args.out_dir / "source_predictions"
    clips_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    scans: dict[str, dict[str, Any]] = {}
    for part in (1, 2, 3):
        source = DATASET / f"flight_part{part}.mkv"
        fps, rows = scan_video(model, source, args.conf, args.batch)
        runs = positive_runs(rows)
        scans[source.stem] = {"fps": fps, "rows": rows, "runs": runs, "path": str(source)}
        with (predictions_dir / f"{source.stem}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    labels = json.loads((DATASET / "_pipeline" / "metadata" / "manifest.json").read_text())
    spans = choose_spans(scans, labels, args.span_frames)
    manifest = []
    verification = []
    for index, span in enumerate(spans):
        clip_id = f"spaceflight_{index:02d}"
        output = clips_dir / f"{clip_id}.mp4"
        fps = float(scans[span["source_stem"]]["fps"])
        vf = (
            f"select='between(n\\,{span['start_frame']}\\,{span['end_frame']})',"
            f"setpts=N/({fps:.12f}*TB)"
        )
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", span["source_path"], "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(output),
        ]
        subprocess.run(command, check=True)
        clip_fps, clip_rows = scan_video(model, output, args.conf, args.batch)
        if len(clip_rows) != args.span_frames or not all(row["positive"] for row in clip_rows):
            raise RuntimeError(f"Frame verification failed for {output}")
        source_labels = [
            row for row in labels
            if row.get("status") == "accepted"
            and Path(row["source_clip"]).stem == span["source_stem"]
            and span["start_frame"] <= int(round((row["timestamp_ms"] / 1000.0) * fps)) <= span["end_frame"]
        ]
        manifest.append(
            {
                "clip_id": clip_id,
                "game": "spaceflight",
                "family": "learned",
                "source_tag": span["source_stem"],
                "source_path": span["source_path"],
                "clip_path": str(output),
                "normalized_path": str(output),
                "start_s": span["start_frame"] / fps,
                "duration_s": args.span_frames / fps,
                "source_width": 1280,
                "source_height": 720,
                "source_fps": fps,
                "target_width": 1280,
                "target_height": 720,
                "target_fps": fps,
                "eligible_duo": True,
                "eligible_rd": True,
                "notes": f"frame-exact fully positive span; anchor={span['anchor_class_name']}@{span['anchor_timestamp_ms']}ms",
            }
        )
        verification.append(
            {
                **span,
                "clip_id": clip_id,
                "clip_path": str(output),
                "encoded_frame_count": len(clip_rows),
                "verified_positive_frames": sum(row["positive"] for row in clip_rows),
                "clip_fps": clip_fps,
                "source_labels": source_labels,
                "class_detection_frame_counts": {
                    "cockpit": sum(0 in row["classes"] for row in clip_rows),
                    "spaceship": sum(1 in row["classes"] for row in clip_rows),
                },
                "command": command,
            }
        )

    (args.out_dir / "offline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "confidence_threshold": args.conf,
        "required_clip_count": 5,
        "actual_clip_count": len(manifest),
        "all_frames_verified_positive": all(
            row["encoded_frame_count"] == row["verified_positive_frames"] for row in verification
        ),
        "clips": verification,
    }
    (args.out_dir / "clip_provenance_and_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Spaceflight evaluation clips", "", "Exactly five frame-exact clips were selected from model-positive runs around accepted segmentation labels.", ""]
    for row in verification:
        lines += [
            f"## {row['clip_id']}",
            f"- Source: `{row['source_path']}` frames {row['start_frame']}–{row['end_frame']}",
            f"- Encoded: `{row['clip_path']}` ({row['encoded_frame_count']} frames at {row['clip_fps']:.6f} fps)",
            f"- Verification: {row['verified_positive_frames']}/{row['encoded_frame_count']} frames contain at least one predicted cockpit or spaceship mask",
            f"- Anchor label: {row['anchor_class_name']} at {row['anchor_timestamp_ms']} ms",
            f"- Per-class detected frames: {row['class_detection_frame_counts']}",
            "",
        ]
    (args.out_dir / "clip_provenance_and_verification.md").write_text("\n".join(lines))
    print(json.dumps({"manifest": str(args.out_dir / "offline_manifest.json"), "clips": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
