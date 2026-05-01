#!/usr/bin/env python3
"""
For each case, pick frames where reconstruction succeeds (recon_ssim >= threshold)
while masking changes the frame a lot (masked_ssim is low).

We compute SSIM directly from videos:
  masked_ssim = SSIM(orig, masked)
  recon_ssim  = SSIM(orig, recon)

Then export ONLY pairs (no triplet) for qualitative masking effect:
  [original | masked]

Filename encodes both SSIMs so you can select examples.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Case:
    key: str
    orig: Path
    masked: Path
    recon: Path


def ssim_bgr(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = 6.5025
    C2 = 58.5225
    I1 = img1.astype(np.float32)
    I2 = img2.astype(np.float32)

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2

    t1 = 2.0 * mu1_mu2 + C1
    t2 = 2.0 * sigma12 + C2
    t3 = t1 * t2

    t1 = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    ssim_map = t3 / (t1 + 1e-12)
    m = cv2.mean(ssim_map)
    return float((m[0] + m[1] + m[2]) / 3.0)


def compute_ssims(orig_mp4: Path, masked_mp4: Path, recon_mp4: Path) -> List[Tuple[int, float, float]]:
    cap_o = cv2.VideoCapture(str(orig_mp4))
    cap_m = cv2.VideoCapture(str(masked_mp4))
    cap_r = cv2.VideoCapture(str(recon_mp4))
    if not cap_o.isOpened() or not cap_m.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Failed to open one of the videos for SSIM computation")

    out: List[Tuple[int, float, float]] = []
    fi = 0
    while True:
        ok_o, fo = cap_o.read()
        ok_m, fm = cap_m.read()
        ok_r, fr = cap_r.read()
        if not (ok_o and ok_m and ok_r):
            break
        if fo.shape != fm.shape or fo.shape != fr.shape:
            raise RuntimeError(f"Frame shape mismatch at {fi}: {fo.shape} vs {fm.shape} vs {fr.shape}")
        out.append((fi, ssim_bgr(fo, fm), ssim_bgr(fo, fr)))
        fi += 1

    cap_o.release()
    cap_m.release()
    cap_r.release()
    return out


def extract_frame_ffmpeg(video: Path, frame_idx: int, out_png: Path) -> None:
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


def export_case(case: Case, out_dir: Path, topk: int, recon_ssim_min: float, masked_ssim_max: float, tile_h: int) -> None:
    rows = compute_ssims(case.orig, case.masked, case.recon)
    # keep recon-good and non-noop masking
    rows = [r for r in rows if r[2] >= recon_ssim_min and r[1] < masked_ssim_max]
    if not rows:
        raise RuntimeError(f"{case.key}: no frames after filtering (recon_ssim_min={recon_ssim_min}, masked_ssim_max={masked_ssim_max})")

    # select: lowest masked first, tie-break higher recon
    rows.sort(key=lambda x: (x[1], -x[2]))
    picks = rows[:topk]

    case_dir = out_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/recon_success_lowmask_pairs") / case.key
    tmp.mkdir(parents=True, exist_ok=True)

    for rank, (fi, mssim, rssim) in enumerate(picks, start=1):
        o_png = tmp / f"orig_f{fi}.png"
        m_png = tmp / f"masked_f{fi}.png"
        extract_frame_ffmpeg(case.orig, fi, o_png)
        extract_frame_ffmpeg(case.masked, fi, m_png)

        o = _resize_to_h(_read_img(o_png), tile_h)
        m = _resize_to_h(_read_img(m_png), tile_h)
        o = _put_label(o, f"{case.key} f={fi} original")
        m = _put_label(m, f"masked ssim={mssim:.4f}  recon_ssim={rssim:.4f}")

        gap = 8
        spacer = np.full((tile_h, gap, 3), 255, dtype=np.uint8)
        pair = np.concatenate([o, spacer, m], axis=1)

        out_png = case_dir / f"pick{rank:02d}_f{fi}_masked{mssim:.4f}_recon{rssim:.4f}.png"
        cv2.imwrite(str(out_png), pair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/tiehangz/proj/yolov12/record/temp/recon_success_lowmask_pairs")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--recon-ssim-min", type=float, default=0.9)
    ap.add_argument("--masked-ssim-max", type=float, default=0.99999)
    ap.add_argument("--tile-height", type=int, default=320)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(
            key="pixel",
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/recovered_output.mp4"),
        ),
        Case(
            key="racing_fm6",
            orig=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/recovered_output.mp4"),
        ),
        Case(
            key="fps_fc5_gunheavy_inband",
            orig=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/segmented_output.mp4"),
            recon=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/recovered_from_sei.mp4"),
        ),
    ]

    for c in cases:
        export_case(
            c,
            out_dir=out_dir,
            topk=args.topk,
            recon_ssim_min=args.recon_ssim_min,
            masked_ssim_max=args.masked_ssim_max,
            tile_h=args.tile_height,
        )

    print(f"Wrote pairs into: {out_dir}")


if __name__ == "__main__":
    main()


