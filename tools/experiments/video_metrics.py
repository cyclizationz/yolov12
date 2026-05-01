#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import REPO_ROOT, python_bin
from msk1 import ParsedPayload, load_payloads


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = 6.5025
    c2 = 58.5225
    i1 = img1.astype(np.float32)
    i2 = img2.astype(np.float32)
    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(i1 * i1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2 * i2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def _rect_mask(payload: ParsedPayload | None, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if payload is None:
        return mask
    h_img, w_img = shape[:2]
    for region in payload.regions:
        x0 = max(0, min(w_img, region.x))
        y0 = max(0, min(h_img, region.y))
        x1 = max(x0, min(w_img, region.x + region.w))
        y1 = max(y0, min(h_img, region.y + region.h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _masked_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(frame)
    if mask.any():
        out[mask != 0] = frame[mask != 0]
    return out


def _bbox_crop_from_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = cv2.findNonZero(mask)
    if pts is None:
        return frame[:0, :0]
    x, y, w, h = cv2.boundingRect(pts)
    return frame[y : y + h, x : x + w]


class FfmpegRawWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}")


def _run_vmaf(ref: Path, dist: Path, *, threads: int, scale_height: int) -> dict[str, Any]:
    cmd = [
        python_bin(),
        str(REPO_ROOT / "tools" / "metrics" / "vmaf_score.py"),
        "--ref",
        str(ref),
        "--dist",
        str(dist),
        "--threads",
        str(threads),
        "--ffmpeg-threads",
        str(threads),
        "--out-fmt",
        "csv",
    ]
    if scale_height > 0:
        cmd += ["--scale-height", str(scale_height)]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def compute_video_metrics(
    *,
    ref_video: Path,
    dist_video: Path,
    msk1_bin: Path | None,
    threads: int,
    scale_height: int,
) -> dict[str, Any]:
    # Use all OpenCV worker threads for per-frame SSIM/PSNR (GaussianBlur etc.).
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    cap_ref = cv2.VideoCapture(str(ref_video))
    cap_dist = cv2.VideoCapture(str(dist_video))
    if not cap_ref.isOpened() or not cap_dist.isOpened():
        raise SystemExit(f"Failed to open videos: ref={ref_video} dist={dist_video}")

    fps = cap_ref.get(cv2.CAP_PROP_FPS) or 60.0
    payloads = load_payloads(msk1_bin) if msk1_bin and msk1_bin.exists() else []
    idx = 0

    full_ssim: list[float] = []
    full_psnr: list[float] = []
    roi_ssim: list[float] = []
    roi_psnr: list[float] = []

    roi_ref_path: Path | None = None
    roi_dist_path: Path | None = None
    roi_ref_writer: FfmpegRawWriter | None = None
    roi_dist_writer: FfmpegRawWriter | None = None

    try:
        tmp_root = REPO_ROOT / "record" / "_tmp_vmaf"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="roi_vmaf_", dir=str(tmp_root)) as td:
            tdir = Path(td)
            while True:
                ok_ref, frame_ref = cap_ref.read()
                ok_dist, frame_dist = cap_dist.read()
                if not ok_ref or not ok_dist:
                    break
                full_ssim.append(_compute_ssim(frame_ref, frame_dist))
                full_psnr.append(_compute_psnr(frame_ref, frame_dist))

                payload = payloads[idx] if idx < len(payloads) else None
                mask = _rect_mask(payload, frame_ref.shape)
                if mask.any():
                    ref_crop = _bbox_crop_from_mask(frame_ref, mask)
                    dist_crop = _bbox_crop_from_mask(frame_dist, mask)
                    if ref_crop.size and dist_crop.size and ref_crop.shape == dist_crop.shape:
                        roi_ssim.append(_compute_ssim(ref_crop, dist_crop))
                        roi_psnr.append(_compute_psnr(ref_crop, dist_crop))
                    ref_roi = _masked_frame(frame_ref, mask)
                    dist_roi = _masked_frame(frame_dist, mask)
                    if roi_ref_writer is None:
                        roi_ref_path = tdir / "roi_ref.mkv"
                        roi_dist_path = tdir / "roi_dist.mkv"
                        roi_ref_writer = FfmpegRawWriter(roi_ref_path, frame_ref.shape[1], frame_ref.shape[0], fps)
                        roi_dist_writer = FfmpegRawWriter(roi_dist_path, frame_ref.shape[1], frame_ref.shape[0], fps)
                    roi_ref_writer.write(ref_roi)
                    assert roi_dist_writer is not None
                    roi_dist_writer.write(dist_roi)
                idx += 1
            if roi_ref_writer is not None and roi_dist_writer is not None:
                roi_ref_writer.close()
                roi_dist_writer.close()
                assert roi_ref_path is not None and roi_dist_path is not None
                roi_vmaf = _run_vmaf(roi_ref_path, roi_dist_path, threads=threads, scale_height=scale_height)
            else:
                roi_vmaf = {"vmaf_mean": None, "vmaf_p10": None}
    finally:
        cap_ref.release()
        cap_dist.release()

    full_vmaf = _run_vmaf(ref_video, dist_video, threads=threads, scale_height=scale_height)
    return {
        "frame_count": idx,
        "full_frame": {
            "ssim_mean": float(np.mean(full_ssim)) if full_ssim else None,
            "psnr_mean": float(np.mean(full_psnr)) if full_psnr else None,
            "vmaf_mean": full_vmaf.get("vmaf_mean"),
            "vmaf_p10": full_vmaf.get("vmaf_p10"),
        },
        "roi": {
            "frame_count_with_roi": len(roi_ssim),
            "ssim_mean": float(np.mean(roi_ssim)) if roi_ssim else None,
            "psnr_mean": float(np.mean(roi_psnr)) if roi_psnr else None,
            "vmaf_mean": roi_vmaf.get("vmaf_mean"),
            "vmaf_p10": roi_vmaf.get("vmaf_p10"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute full-frame and ROI video metrics for offline experiments.")
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--dist", required=True, type=Path)
    ap.add_argument("--msk1-bin", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--scale-height", type=int, default=1080)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    metrics = compute_video_metrics(
        ref_video=args.ref,
        dist_video=args.dist,
        msk1_bin=args.msk1_bin,
        threads=args.threads,
        scale_height=args.scale_height,
    )
    text = json.dumps(metrics, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
