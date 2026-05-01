#!/usr/bin/env python3
"""
Export top-K frames by masked SSIM (segmented_output vs original_output) as side-by-side PNGs:
  [original | masked]

This is purely to visualize the masking effect (NO recovery / stitching).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _extract_frame(video: Path, frame_idx: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_idx})"
    subprocess.run(
        [
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
        ],
        check=True,
    )


def _read_img(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return img


def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == h:
        return img
    nw = max(1, int(round(iw * (h / ih))))
    return cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = 8, 22
    cv2.rectangle(out, (x - 4, y - th - 6), (x + tw + 4, y + 6), (255, 255, 255), -1)
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def _load_per_frame(report: Path) -> List[Dict[str, Any]]:
    rep = json.loads(report.read_text())
    per = rep.get("per_frame", [])
    if not isinstance(per, list) or not per:
        raise ValueError(f"{report}: missing/empty per_frame")
    return per


def _topk_by_ssim(per: List[Dict[str, Any]], k: int) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    for r in per:
        if "frame" not in r or "ssim" not in r:
            continue
        try:
            fi = int(r["frame"])
            ssim = float(r["ssim"])
        except Exception:
            continue
        rows.append((fi, ssim))
    if not rows:
        raise ValueError("No (frame, ssim) rows found")
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:k]

def _topk_by_ssim_filtered(
    per: List[Dict[str, Any]],
    k: int,
    exclude_ssim_ge: float | None,
    exclude_psnr_ge: float | None,
) -> List[Tuple[int, float]]:
    """
    Like _topk_by_ssim, but excludes frames that look like "no-op" masking:
    - ssim >= exclude_ssim_ge (very close to 1.0)
    - psnr >= exclude_psnr_ge (often 100.0 due to clamp in our PSNR impl)
    """
    rows: List[Tuple[int, float, float]] = []
    for r in per:
        if "frame" not in r or "ssim" not in r:
            continue
        try:
            fi = int(r["frame"])
            ssim = float(r["ssim"])
            psnr = float(r.get("psnr", float("nan")))
        except Exception:
            continue
        if exclude_ssim_ge is not None and ssim >= exclude_ssim_ge:
            continue
        if exclude_psnr_ge is not None and np.isfinite(psnr) and psnr >= exclude_psnr_ge:
            continue
        rows.append((fi, ssim, psnr))
    if not rows:
        raise ValueError("No frames left after filtering (try relaxing exclude thresholds)")
    rows.sort(key=lambda x: x[1], reverse=True)
    return [(fi, ssim) for (fi, ssim, _psnr) in rows[:k]]


def export_case(case_name: str, report: Path, orig: Path, masked: Path, out_dir: Path, topk: int) -> None:
    per = _load_per_frame(report)
    picks = _topk_by_ssim(per, topk)

    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path("/tmp/top_masked_ssim_frames") / case_name.replace(" ", "_")
    tmp.mkdir(parents=True, exist_ok=True)

    for rank, (fi, ssim) in enumerate(picks, start=1):
        o_png = tmp / f"orig_f{fi}.png"
        m_png = tmp / f"masked_f{fi}.png"
        _extract_frame(orig, fi, o_png)
        _extract_frame(masked, fi, m_png)

        o = _read_img(o_png)
        m = _read_img(m_png)
        h = min(o.shape[0], m.shape[0], 360)
        o = _resize_to_h(o, h)
        m = _resize_to_h(m, h)

        o = _put_label(o, f"{case_name}  f={fi}  original")
        m = _put_label(m, f"masked  SSIM={ssim:.4f}")

        gap = 8
        spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
        pair = np.concatenate([o, spacer, m], axis=1)

        out_png = case_dir / f"top{rank:02d}_f{fi}_ssim{ssim:.4f}.png"
        cv2.imwrite(str(out_png), pair)

def export_case_filtered(
    case_name: str,
    report: Path,
    orig: Path,
    masked: Path,
    out_dir: Path,
    topk: int,
    exclude_ssim_ge: float | None,
    exclude_psnr_ge: float | None,
) -> None:
    per = _load_per_frame(report)
    picks = _topk_by_ssim_filtered(per, topk, exclude_ssim_ge=exclude_ssim_ge, exclude_psnr_ge=exclude_psnr_ge)

    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path("/tmp/top_masked_ssim_frames") / case_name.replace(" ", "_")
    tmp.mkdir(parents=True, exist_ok=True)

    for rank, (fi, ssim) in enumerate(picks, start=1):
        o_png = tmp / f"orig_f{fi}.png"
        m_png = tmp / f"masked_f{fi}.png"
        _extract_frame(orig, fi, o_png)
        _extract_frame(masked, fi, m_png)

        o = _read_img(o_png)
        m = _read_img(m_png)
        h = min(o.shape[0], m.shape[0], 360)
        o = _resize_to_h(o, h)
        m = _resize_to_h(m, h)

        o = _put_label(o, f"{case_name}  f={fi}  original")
        m = _put_label(m, f"masked  SSIM={ssim:.4f}")

        gap = 8
        spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
        pair = np.concatenate([o, spacer, m], axis=1)

        out_png = case_dir / f"top{rank:02d}_f{fi}_ssim{ssim:.4f}.png"
        cv2.imwrite(str(out_png), pair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/tiehangz/proj/yolov12/record/temp", help="Output directory")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument(
        "--cases",
        default="pixel,racing_fm6,fps_fc5_gunheavy",
        help="Comma-separated case keys: pixel,racing_fm6,fps_fc5_gunheavy",
    )
    ap.add_argument(
        "--exclude-ssim-ge",
        type=float,
        default=None,
        help="Exclude frames with masked SSIM >= this (e.g. 0.99999 to drop no-op frames)",
    )
    ap.add_argument(
        "--exclude-psnr-ge",
        type=float,
        default=None,
        help="Exclude frames with masked PSNR >= this (e.g. 99.9 to drop 100dB-clamped no-op frames)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    want = {c.strip() for c in args.cases.split(",") if c.strip()}

    if "pixel" in want:
        export_case(
            "pixel",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/segmented_output.mp4"),
            out_dir=out_dir,
            topk=args.topk,
        )

    if "racing_fm6" in want:
        export_case(
            "racing_fm6",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/segmented_output.mp4"),
            out_dir=out_dir,
            topk=args.topk,
        )

    if "fps_fc5_gunheavy" in want:
        export_case_filtered(
            "fps_fc5_gunheavy",
            report=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/segmented_output.mp4"),
            out_dir=out_dir,
            topk=args.topk,
            exclude_ssim_ge=args.exclude_ssim_ge,
            exclude_psnr_ge=args.exclude_psnr_ge,
        )

    print(f"Wrote top-{args.topk} SSIM pairs into: {out_dir}")


if __name__ == "__main__":
    main()


