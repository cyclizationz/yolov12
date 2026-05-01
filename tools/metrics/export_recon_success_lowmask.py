#!/usr/bin/env python3
"""
Select frames where masking changes the image a lot, but client reconstruction succeeds.

Criterion (per frame):
  - Primary: high reconstruction SSIM (rec_ssim, reconstructed vs original)
  - Secondary: low masking SSIM (ssim, masked/segmented vs original)

Implementation:
  - Take the top `--rec-top-frac` fraction of frames by rec_ssim
  - Within that subset, pick the lowest `ssim` frames (tie-break by higher rec_ssim)

Exports for each selected frame a triplet PNG:
  [original | masked | reconstructed]
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


@dataclass(frozen=True)
class Case:
    key: str
    report: Path
    orig: Path
    masked: Path
    recon: Path


def _extract_frame(video: Path, frame_idx: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_idx})"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", vf, "-frames:v", "1", "-vsync", "0", str(out_png)],
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


def _pad_to_w(img: np.ndarray, w: int) -> np.ndarray:
    h, iw = img.shape[:2]
    if iw >= w:
        return img[:, :w].copy()
    return cv2.copyMakeBorder(img, 0, 0, 0, w - iw, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))


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


def select_frames(
    per: List[Dict[str, Any]],
    topk: int,
    rec_top_frac: float,
    rec_min: float | None,
    masked_ssim_max: float | None,
) -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    for r in per:
        if "frame" not in r or "ssim" not in r or "rec_ssim" not in r:
            continue
        try:
            fi = int(r["frame"])
            ssim = float(r["ssim"])
            rec_ssim = float(r["rec_ssim"])
        except Exception:
            continue
        # Exclude "no-op" masking frames where masked output is effectively identical to original.
        if masked_ssim_max is not None and ssim >= masked_ssim_max:
            continue
        if rec_min is not None and rec_ssim < rec_min:
            continue
        rows.append((fi, ssim, rec_ssim))
    if not rows:
        raise ValueError("No frames with (frame, ssim, rec_ssim) after filtering")

    # top fraction by rec_ssim
    rows.sort(key=lambda x: x[2], reverse=True)
    keep_n = max(1, int(round(len(rows) * rec_top_frac)))
    subset = rows[:keep_n]

    # within subset: minimize masking ssim, tie-break by higher rec_ssim
    subset.sort(key=lambda x: (x[1], -x[2]))
    return subset[:topk]


def export_case(
    case: Case,
    out_root: Path,
    topk: int,
    rec_top_frac: float,
    rec_min: float | None,
    masked_ssim_max: float | None,
    tile_h: int,
) -> None:
    per = _load_per_frame(case.report)
    picks = select_frames(per, topk=topk, rec_top_frac=rec_top_frac, rec_min=rec_min, masked_ssim_max=masked_ssim_max)

    case_dir = out_root / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/recon_success_lowmask") / case.key
    tmp.mkdir(parents=True, exist_ok=True)

    for rank, (fi, ssim, rec_ssim) in enumerate(picks, start=1):
        o_png = tmp / f"orig_f{fi}.png"
        m_png = tmp / f"masked_f{fi}.png"
        r_png = tmp / f"recon_f{fi}.png"
        _extract_frame(case.orig, fi, o_png)
        _extract_frame(case.masked, fi, m_png)
        _extract_frame(case.recon, fi, r_png)

        o = _resize_to_h(_read_img(o_png), tile_h)
        m = _resize_to_h(_read_img(m_png), tile_h)
        r = _resize_to_h(_read_img(r_png), tile_h)

        o = _put_label(o, f"{case.key} f={fi} original")
        m = _put_label(m, f"masked ssim={ssim:.4f}")
        r = _put_label(r, f"recon rec_ssim={rec_ssim:.4f}")

        col_w = max(o.shape[1], m.shape[1], r.shape[1])
        o = _pad_to_w(o, col_w)
        m = _pad_to_w(m, col_w)
        r = _pad_to_w(r, col_w)

        gap = 8
        spacer = np.full((tile_h, gap, 3), 255, dtype=np.uint8)
        trip = np.concatenate([o, spacer, m, spacer, r], axis=1)

        out_png = case_dir / f"pick{rank:02d}_f{fi}_masked{ssim:.4f}_recon{rec_ssim:.4f}.png"
        cv2.imwrite(str(out_png), trip)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/tiehangz/proj/yolov12/record/temp/recon_success_lowmask")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rec-top-frac", type=float, default=0.20, help="Take top fraction by rec_ssim before minimizing masked ssim")
    ap.add_argument("--rec-min", type=float, default=None, help="Optional absolute minimum rec_ssim")
    ap.add_argument(
        "--masked-ssim-max",
        type=float,
        default=0.99999,
        help="Exclude frames with masked SSIM >= this (default drops no-op/skip-masking frames). Use 1.0 to disable.",
    )
    ap.add_argument("--tile-height", type=int, default=320)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(
            key="pixel",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/recovered_output.mp4"),
        ),
        Case(
            key="racing_fm6",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/recovered_output.mp4"),
        ),
        Case(
            key="fps_fc5_gunheavy",
            report=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/report.json"),
            orig=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/recovered_output.mp4"),
        ),
    ]

    for c in cases:
        export_case(
            c,
            out_root=out_dir,
            topk=args.topk,
            rec_top_frac=args.rec_top_frac,
            rec_min=args.rec_min,
            masked_ssim_max=args.masked_ssim_max,
            tile_h=args.tile_height,
        )

    print(f"Wrote triplets into: {out_dir}")


if __name__ == "__main__":
    main()


