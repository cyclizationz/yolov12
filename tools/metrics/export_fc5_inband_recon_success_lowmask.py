#!/usr/bin/env python3
"""
FC5 (gun-heavy) qualitative selector using *in-band SEI stitched* reconstruction.

We compute per-frame SSIM directly from videos:
  - masked_ssim: segmented_output.mp4 vs original_output.mp4
  - recon_ssim:  recovered_from_sei.mp4 vs original_output.mp4

Selection goal:
  - recon_ssim >= --recon-ssim-min (default 0.9)
  - among those, pick frames with the lowest masked_ssim (mask changed a lot)

Exports triplets:
  [original | masked | reconstructed(in-band SEI)]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def ssim_bgr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Match the OpenCV SSIM implementation style used in offline_processor.cpp (channel-mean SSIM)."""
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

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2

    ssim_map = t3 / (t1 + 1e-12)
    m = cv2.mean(ssim_map)
    return float((m[0] + m[1] + m[2]) / 3.0)


def extract_frame_ffmpeg(video: Path, frame_idx: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_idx})"
    # Use cv2.VideoCapture for extraction would drift on some codecs; ffmpeg is deterministic.
    import subprocess

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
        masked_ssim = ssim_bgr(fo, fm)
        recon_ssim = ssim_bgr(fo, fr)
        out.append((fi, masked_ssim, recon_ssim))
        fi += 1

    cap_o.release()
    cap_m.release()
    cap_r.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/tiehangz/proj/yolov12/record/temp/recon_success_lowmask_inband/fps_fc5_gunheavy")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--recon-ssim-min", type=float, default=0.9)
    ap.add_argument("--tile-height", type=int, default=320)
    args = ap.parse_args()

    orig = Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4")
    masked = Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/segmented_output.mp4")
    recon = Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/recovered_from_sei.mp4")

    rows = compute_ssims(orig, masked, recon)
    # filter: recon good
    rows = [r for r in rows if r[2] >= args.recon_ssim_min]
    if not rows:
        raise SystemExit(f"No frames with recon_ssim >= {args.recon_ssim_min}")

    # sort: lowest masked_ssim first; tie-break higher recon_ssim
    rows.sort(key=lambda x: (x[1], -x[2]))
    picks = rows[: args.topk]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/fc5_inband_triplets")
    tmp.mkdir(parents=True, exist_ok=True)

    for rank, (fi, mssim, rssim) in enumerate(picks, start=1):
        o_png = tmp / f"orig_f{fi}.png"
        m_png = tmp / f"masked_f{fi}.png"
        r_png = tmp / f"recon_f{fi}.png"
        extract_frame_ffmpeg(orig, fi, o_png)
        extract_frame_ffmpeg(masked, fi, m_png)
        extract_frame_ffmpeg(recon, fi, r_png)

        o = _resize_to_h(_read_img(o_png), args.tile_height)
        m = _resize_to_h(_read_img(m_png), args.tile_height)
        r = _resize_to_h(_read_img(r_png), args.tile_height)

        o = _put_label(o, f"FC5 gun-heavy f={fi} original")
        m = _put_label(m, f"masked ssim={mssim:.4f}")
        r = _put_label(r, f"inband recon ssim={rssim:.4f}")

        col_w = max(o.shape[1], m.shape[1], r.shape[1])
        o = _pad_to_w(o, col_w)
        m = _pad_to_w(m, col_w)
        r = _pad_to_w(r, col_w)

        gap = 8
        spacer = np.full((args.tile_height, gap, 3), 255, dtype=np.uint8)
        trip = np.concatenate([o, spacer, m, spacer, r], axis=1)

        out_png = out_dir / f"pick{rank:02d}_f{fi}_masked{mssim:.4f}_recon{rssim:.4f}.png"
        cv2.imwrite(str(out_png), trip)

    print(f"Wrote {len(picks)} triplets to {out_dir}")


if __name__ == "__main__":
    main()


