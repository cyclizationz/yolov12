#!/usr/bin/env python3
"""
Build the paper table row(s) for:
Game | Model | Number of Training Frames | Training Accuracy | Validation Accuracy | Prediction/Masking Accuracy | Parameter tuning

Sources:
- Training logs: runs/segment/<run_name>/{args.yaml,results.csv}
- Deployment/offline masking logs: record/final2/<run_dir>/report.json (+ run.log)

Outputs:
- CSV or Markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_text(p: Path) -> str:
    return p.read_text(errors="replace")


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(_read_text(p))


def _parse_simple_yaml_kv(text: str) -> dict[str, str]:
    """
    Minimal YAML key:value parser sufficient for our args.yaml (flat mapping).
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class TrainBest:
    epoch: int
    Pm: float
    Rm: float
    mAP50m: float
    mAP5095m: float


def _train_best_from_results(results_csv: Path) -> TrainBest:
    rows: list[dict[str, str]] = []
    with results_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return TrainBest(epoch=0, Pm=float("nan"), Rm=float("nan"), mAP50m=float("nan"), mAP5095m=float("nan"))

    def fget(row: dict[str, str], k: str) -> float:
        try:
            return float(row.get(k, "nan"))
        except Exception:
            return float("nan")

    best_row = None
    best_key = None
    for row in rows:
        key = (fget(row, "metrics/mAP50-95(M)"), fget(row, "metrics/mAP50(M)"))
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
    assert best_row is not None
    return TrainBest(
        epoch=int(float(best_row.get("epoch", "0") or "0")),
        Pm=fget(best_row, "metrics/precision(M)"),
        Rm=fget(best_row, "metrics/recall(M)"),
        mAP50m=fget(best_row, "metrics/mAP50(M)"),
        mAP5095m=fget(best_row, "metrics/mAP50-95(M)"),
    )


def _count_images(dirpath: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not dirpath.exists():
        return 0
    n = 0
    for p in dirpath.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def _masking_summary(run_dir: Path) -> dict[str, Any]:
    rep = _read_json(run_dir / "report.json")
    per = rep.get("per_frame", []) or []
    total_frames = int(rep.get("total_frames", len(per)) or len(per))
    total_dets = int(rep.get("total_detections", rep.get("matched_detections", 0)) or 0)
    matched_dets = int(rep.get("matched_detections", 0) or 0)
    ref = 0
    raw = 0
    obj_present = 0
    obj_masked = 0
    for e in per:
        ff = e.get("frame_flags", None)
        if ff is None:
            if int(e.get("object_successfully_masked", 0) or 0) > 0:
                ref += 1
            else:
                raw += 1
        else:
            if int(ff) == 0:
                raw += 1
            else:
                ref += 1
        obj_present += int(e.get("object_present_model", 0) or 0)
        obj_masked += int(e.get("object_successfully_masked", 0) or 0)
    return {
        "ref_frame_ratio": (ref / total_frames) if total_frames else 0.0,
        "match_rate": (matched_dets / total_dets) if total_dets else 0.0,
        "object_masked_over_present": (obj_masked / obj_present) if obj_present else 0.0,
        "avg_recovered_ssim": rep.get("avg_recovered_ssim"),
        "avg_recovered_psnr": rep.get("avg_recovered_psnr"),
        "avg_ssim_masked": rep.get("avg_ssim"),
        "avg_psnr_masked": rep.get("avg_psnr"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="Output path (.csv or .md)")
    ap.add_argument("--racing-run", default="yolov12n_racing", help="runs/segment/<name> folder")
    ap.add_argument("--fc5-run", default="yolov12n_fc5_seg_v1", help="runs/segment/<name> folder")
    ap.add_argument("--racing-final2", default="fm6_final_full_v2", help="record/final2/<dir> for masking acc (FM6 eval)")
    ap.add_argument("--pixel-final2", default="mario_dominant_full", help="record/final2/<dir> for masking acc (Mario eval)")
    ap.add_argument("--fc5-final2", default="fc5_crop_final_full_v2", help="record/final2/<dir> for masking acc (FC5 eval)")
    args = ap.parse_args()

    root = Path("/home/tiehangz/proj/yolov12")

    def row_for_yolo(game: str, model: str, train_img_dir: Path, run_name: str, final2_dir: str) -> dict[str, Any]:
        run_dir = root / "runs" / "segment" / run_name
        args_yaml = run_dir / "args.yaml"
        results_csv = run_dir / "results.csv"
        cfg = _parse_simple_yaml_kv(_read_text(args_yaml)) if args_yaml.exists() else {}
        best = _train_best_from_results(results_csv) if results_csv.exists() else TrainBest(0, float("nan"), float("nan"), float("nan"), float("nan"))

        mask = _masking_summary(root / "record" / "final2" / final2_dir)

        tuning_bits = []
        for k in ["epochs", "batch", "imgsz", "close_mosaic", "copy_paste", "auto_augment", "erasing", "mask_ratio", "overlap_mask", "pretrained", "seed", "deterministic"]:
            if k in cfg:
                tuning_bits.append(f"{k}={cfg[k]}")
        tuning = ", ".join(tuning_bits)

        training_acc = "N/A (Ultralytics logs val metrics; no train-set mAP in results.csv)"
        val_acc = f"best@e{best.epoch}: P={best.Pm:.4f} R={best.Rm:.4f} mAP50={best.mAP50m:.4f} mAP50-95={best.mAP5095m:.4f}"
        masking_acc = (
            f"match_rate={mask['match_rate']:.4f}, ref_frame_ratio={mask['ref_frame_ratio']:.4f}, "
            f"masked/present={mask['object_masked_over_present']:.4f}, "
            f"rec_ssim={float(mask['avg_recovered_ssim'] or 0.0):.4f}"
        )

        return {
            "Game": game,
            "Model": model,
            "Number of Training Frames": _count_images(train_img_dir),
            "Training Accuracy": training_acc,
            "Validation Accuracy": val_acc,
            "Prediction/Masking Accuracy": masking_acc,
            "Parameter tuning": tuning,
        }

    def row_for_pixel(game: str, final2_dir: str) -> dict[str, Any]:
        mask = _masking_summary(root / "record" / "final2" / final2_dir)
        masking_acc = (
            f"match_rate={mask['match_rate']:.4f}, ref_frame_ratio={mask['ref_frame_ratio']:.4f}, "
            f"masked/present={mask['object_masked_over_present']:.4f}, "
            f"rec_ssim={float(mask['avg_recovered_ssim'] or 0.0):.4f}"
        )
        return {
            "Game": game,
            "Model": "pixel_mode (template matching + kalman + flow)",
            "Number of Training Frames": "N/A",
            "Training Accuracy": "N/A",
            "Validation Accuracy": "N/A",
            "Prediction/Masking Accuracy": masking_acc,
            "Parameter tuning": "pixel mode (no segmentation training)",
        }

    rows = [
        row_for_yolo(
            game="Racing (FM6)",
            model=args.racing_run,
            # Prefer split layout if present (train/images); fall back to legacy images/ (older experiments).
            train_img_dir=(
                Path("/home/tiehangz/proj/datasets/racing/train/images")
                if Path("/home/tiehangz/proj/datasets/racing/train/images").exists()
                else Path("/home/tiehangz/proj/datasets/racing/images")
            ),
            run_name=args.racing_run,
            final2_dir=args.racing_final2,
        ),
        row_for_pixel(game="Pixel (Mario)", final2_dir=args.pixel_final2),
        row_for_yolo(
            game="FPS (FC5)",
            model=args.fc5_run,
            train_img_dir=Path("/home/tiehangz/proj/datasets/fps/fc5/train/images"),
            run_name=args.fc5_run,
            final2_dir=args.fc5_final2,
        ),
    ]

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        # markdown
        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for r in rows:
            lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
        out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

