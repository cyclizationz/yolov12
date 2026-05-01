#!/usr/bin/env python3
"""
Create qualitative reconstruction examples by selecting best/worst frames (by PSNR+SSIM)
from report.json and extracting matching frames from baseline and reconstructed videos.

Output is a single grid figure:
  rows = cases (Pixel, Racing, FPS)
  cols = [baseline(best), reconstructed(best), baseline(worst), reconstructed(worst)]

We intentionally include baseline for both best and worst frames, so comparisons are
apples-to-apples within each pair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    report: Path
    baseline_video: Path
    reconstructed_video: Path


def _load_report(report_path: Path) -> List[Dict[str, Any]]:
    rep = json.loads(report_path.read_text())
    per = rep.get("per_frame", [])
    if not isinstance(per, list) or not per:
        raise ValueError(f"{report_path}: missing/empty per_frame")
    return per


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def select_best_worst(per_frame: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Select best/worst frames by a combined (rank-based) score of (rec_psnr, rec_ssim).
    This avoids situations where PSNR and SSIM disagree.
    """
    rows = []
    for r in per_frame:
        if _finite(r.get("rec_psnr")) and _finite(r.get("rec_ssim")) and _finite(r.get("frame")):
            rows.append(r)
    if not rows:
        raise ValueError("No frames with finite rec_psnr + rec_ssim + frame index")

    psnrs = np.array([float(r["rec_psnr"]) for r in rows], dtype=np.float64)
    ssims = np.array([float(r["rec_ssim"]) for r in rows], dtype=np.float64)

    # rank-normalize: higher is better
    psnr_rank = psnrs.argsort().argsort().astype(np.float64) / max(1, (len(rows) - 1))
    ssim_rank = ssims.argsort().argsort().astype(np.float64) / max(1, (len(rows) - 1))
    score = 0.5 * psnr_rank + 0.5 * ssim_rank

    best_i = int(np.argmax(score))
    worst_i = int(np.argmin(score))
    return rows[best_i], rows[worst_i]


def extract_frame_ffmpeg(video: Path, frame_idx: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg select on frame index (n is 0-based frame count in decode order).
    vf = f"select=eq(n\\,{frame_idx})"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-vsync",
        "0",
        str(out_png),
    ]
    subprocess.run(cmd, check=True)


def _read_img(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return img


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    new_w = max(1, int(round(w * (height / h))))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _pad_to_width(img: np.ndarray, width: int, color=(255, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= width:
        return img[:, :width].copy()
    pad = width - w
    return cv2.copyMakeBorder(img, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=color)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    # semi-opaque white box behind text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = 8, 20
    cv2.rectangle(out, (x - 4, y - th - 6), (x + tw + 4, y + 6), (255, 255, 255), -1)
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def build_grid(cases: List[Case], out_png: Path, tile_h: int = 260) -> None:
    tmp = Path("/tmp/recon_examples_frames")
    if tmp.exists():
        # keep deterministic but don’t fail if cannot clean
        try:
            for f in tmp.glob("*.png"):
                f.unlink()
        except Exception:
            pass
    tmp.mkdir(parents=True, exist_ok=True)

    row_imgs: List[np.ndarray] = []
    for case in cases:
        per = _load_report(case.report)
        best, worst = select_best_worst(per)

        best_idx = int(best["frame"])
        worst_idx = int(worst["frame"])

        # Extract 4 panels per case: base(best), rec(best), base(worst), rec(worst)
        paths = {
            "base_best": tmp / f"{case.name}_base_best_{best_idx}.png",
            "rec_best": tmp / f"{case.name}_rec_best_{best_idx}.png",
            "base_worst": tmp / f"{case.name}_base_worst_{worst_idx}.png",
            "rec_worst": tmp / f"{case.name}_rec_worst_{worst_idx}.png",
        }
        extract_frame_ffmpeg(case.baseline_video, best_idx, paths["base_best"])
        extract_frame_ffmpeg(case.reconstructed_video, best_idx, paths["rec_best"])
        extract_frame_ffmpeg(case.baseline_video, worst_idx, paths["base_worst"])
        extract_frame_ffmpeg(case.reconstructed_video, worst_idx, paths["rec_worst"])

        base_best = _resize_to_height(_read_img(paths["base_best"]), tile_h)
        rec_best = _resize_to_height(_read_img(paths["rec_best"]), tile_h)
        base_worst = _resize_to_height(_read_img(paths["base_worst"]), tile_h)
        rec_worst = _resize_to_height(_read_img(paths["rec_worst"]), tile_h)

        # Labels include metrics for the reconstructed frame
        rec_best = _put_label(
            rec_best,
            f"{case.name} best  f={best_idx}  PSNR={float(best['rec_psnr']):.2f}  SSIM={float(best['rec_ssim']):.4f}",
        )
        rec_worst = _put_label(
            rec_worst,
            f"{case.name} worst f={worst_idx} PSNR={float(worst['rec_psnr']):.2f} SSIM={float(worst['rec_ssim']):.4f}",
        )
        base_best = _put_label(base_best, f"{case.name} baseline (best f={best_idx})")
        base_worst = _put_label(base_worst, f"{case.name} baseline (worst f={worst_idx})")

        # Make same width per column inside a row
        col_w = max(base_best.shape[1], rec_best.shape[1], base_worst.shape[1], rec_worst.shape[1])
        base_best = _pad_to_width(base_best, col_w)
        rec_best = _pad_to_width(rec_best, col_w)
        base_worst = _pad_to_width(base_worst, col_w)
        rec_worst = _pad_to_width(rec_worst, col_w)

        gap = 6
        spacer = np.full((tile_h, gap, 3), 255, dtype=np.uint8)
        row = np.concatenate([base_best, spacer, rec_best, spacer, base_worst, spacer, rec_worst], axis=1)
        row_imgs.append(row)

    # unify row widths
    W = max(r.shape[1] for r in row_imgs)
    row_imgs = [_pad_to_width(r, W) for r in row_imgs]

    vgap = 10
    vspacer = np.full((vgap, W, 3), 255, dtype=np.uint8)
    grid = row_imgs[0]
    for r in row_imgs[1:]:
        grid = np.concatenate([grid, vspacer, r], axis=0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/home/tiehangz/proj/yolov12/record/figures/recon_examples_best_worst.png",
        help="Output PNG path",
    )
    ap.add_argument("--tile-height", type=int, default=260)
    args = ap.parse_args()

    cases = [
        Case(
            name="Pixel (Mario)",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/report.json"),
            baseline_video=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/original_output.mp4"),
            reconstructed_video=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/recovered_output.mp4"),
        ),
        Case(
            name="Racing (FM6)",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/report.json"),
            baseline_video=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/original_output.mp4"),
            reconstructed_video=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/recovered_output.mp4"),
        ),
        Case(
            name="FPS (FC5 gun-heavy)",
            report=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/report.json"),
            baseline_video=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4"),
            reconstructed_video=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/recovered_output.mp4"),
        ),
    ]

    build_grid(cases, Path(args.out), tile_h=args.tile_height)
    print(f"Wrote figure: {args.out}")


if __name__ == "__main__":
    main()


