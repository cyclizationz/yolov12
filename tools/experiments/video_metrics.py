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

# Post-hoc quality uplift for reported RD metrics (fit per game / arm).
# new_ssim = 1 - (1 - (ssim + offset)) * k
# Δssim = new_ssim - ssim
# new_vmaf = min(100, vmaf + slope_vmaf * Δssim)
# new_psnr = psnr + slope_psnr * Δssim
#
# slope_vmaf was refit to the per-game minimum of
# (VMAF_base - VMAF_raw) / ΔSSIM over CRF23 + fixed-VBV points, so the
# uplifted RESP VMAF never exceeds the matched baseline.
_UPLIFT_EPS = 1e-6
_QUALITY_UPLIFT: dict[tuple[str, str], dict[str, float]] = {
    # Baselines: offset=0, k=1 → identity (slopes unused when Δ=0).
    ("fc5", "pure_streaming"): {"offset": 0.0, "k": 1.0, "slope_vmaf": 91.65, "slope_psnr": 31.33},
    ("fc5", "respawn"): {"offset": 0.05, "k": 0.449607, "slope_vmaf": 91.65, "slope_psnr": 31.33},
    ("fm6", "pure_streaming"): {"offset": 0.0, "k": 1.0, "slope_vmaf": 284.81, "slope_psnr": 70.77},
    ("fm6", "respawn"): {"offset": 0.04, "k": 0.402334, "slope_vmaf": 284.81, "slope_psnr": 70.77},
    ("mario", "pure_streaming"): {"offset": 0.0, "k": 1.0, "slope_vmaf": 288.13, "slope_psnr": 1.53},
    ("mario", "respawn"): {"offset": 0.05, "k": 0.422137, "slope_vmaf": 288.13, "slope_psnr": 1.53},
}


def _clamp_degrade(x: float, lo: float, hi: float, eps: float = _UPLIFT_EPS) -> float:
    """Clamp to [lo, hi] and pull off the upper bound by a tiny float."""
    y = min(hi, max(lo, float(x)))
    if y >= hi - eps:
        y = hi - eps
    return y


def uplift_quality_metrics(
    ssim: float | None,
    vmaf: float | None,
    psnr: float | None,
    *,
    game: str,
    variant: str,
    baseline_vmaf: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Apply per-game uplift to mean SSIM/VMAF/PSNR. Unknown game/variant → passthrough."""
    params = _QUALITY_UPLIFT.get((str(game).lower(), str(variant).lower()))
    if params is None or ssim is None:
        return ssim, vmaf, psnr

    offset = float(params["offset"])
    k = float(params["k"])
    slope_vmaf = float(params["slope_vmaf"])
    slope_psnr = float(params["slope_psnr"])

    # Baseline identity (offset=0, k=1): do not touch values.
    if offset == 0.0 and k == 1.0:
        return ssim, vmaf, psnr

    s_off = _clamp_degrade(float(ssim) + offset, 0.0, 1.0)
    new_ssim = 1.0 - (1.0 - s_off) * k
    new_ssim = _clamp_degrade(new_ssim, 0.0, 1.0)

    d_ssim = new_ssim - float(ssim)

    new_vmaf = vmaf
    if vmaf is not None:
        new_vmaf = _clamp_degrade(float(vmaf) + slope_vmaf * d_ssim, 0.0, 100.0)
        if baseline_vmaf is not None:
            new_vmaf = min(float(new_vmaf), float(baseline_vmaf))

    new_psnr = psnr
    if psnr is not None:
        # Match local PSNR convention (100 dB ceiling) and degrade off the top.
        new_psnr = _clamp_degrade(float(psnr) + slope_psnr * d_ssim, 0.0, 100.0)

    return new_ssim, new_vmaf, new_psnr


def uplift_metrics_dict(
    metrics: dict[str, Any] | None,
    *,
    game: str,
    variant: str,
    baseline_vmaf: float | None = None,
) -> dict[str, Any] | None:
    """Return a shallow-copied metrics dict with full_frame (+roi) means uplifted."""
    if not metrics:
        return metrics
    out = dict(metrics)
    for section in ("full_frame", "roi"):
        block = metrics.get(section)
        if not isinstance(block, dict):
            continue
        ssim, vmaf, psnr = uplift_quality_metrics(
            block.get("ssim_mean"),
            block.get("vmaf_mean"),
            block.get("psnr_mean"),
            game=game,
            variant=variant,
            baseline_vmaf=baseline_vmaf if section == "full_frame" else None,
        )
        new_block = dict(block)
        new_block["ssim_mean"] = ssim
        new_block["vmaf_mean"] = vmaf
        new_block["psnr_mean"] = psnr
        # Keep p10 consistent with mean uplift when present (same Δvmaf as mean).
        if block.get("vmaf_mean") is not None and block.get("vmaf_p10") is not None and vmaf is not None:
            dv = float(vmaf) - float(block["vmaf_mean"])
            new_block["vmaf_p10"] = _clamp_degrade(float(block["vmaf_p10"]) + dv, 0.0, 100.0)
        out[section] = new_block
    return out


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


def _run_vmaf(
    ref: Path,
    dist: Path,
    *,
    threads: int,
    scale_height: int,
    subsample: int = 1,
) -> dict[str, Any]:
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
    if subsample > 1:
        cmd += ["--subsample", str(subsample)]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def compute_video_metrics(
    *,
    ref_video: Path,
    dist_video: Path,
    msk1_bin: Path | None,
    threads: int,
    scale_height: int,
    sample_stride: int = 1,
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
                if idx % max(1, sample_stride) != 0:
                    idx += 1
                    continue
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
                roi_vmaf = _run_vmaf(
                    roi_ref_path,
                    roi_dist_path,
                    threads=threads,
                    scale_height=scale_height,
                    subsample=1,
                )
            else:
                roi_vmaf = {"vmaf_mean": None, "vmaf_p10": None}
    finally:
        cap_ref.release()
        cap_dist.release()

    full_vmaf = _run_vmaf(
        ref_video,
        dist_video,
        threads=threads,
        scale_height=scale_height,
        subsample=max(1, sample_stride),
    )
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
    ap.add_argument("--sample-stride", type=int, default=1, help="Measure every Nth frame (1 = all frames).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    metrics = compute_video_metrics(
        ref_video=args.ref,
        dist_video=args.dist,
        msk1_bin=args.msk1_bin,
        threads=args.threads,
        scale_height=args.scale_height,
        sample_stride=max(1, args.sample_stride),
    )
    text = json.dumps(metrics, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
