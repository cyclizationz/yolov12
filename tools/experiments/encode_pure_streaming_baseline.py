#!/usr/bin/env python3
"""
Encode normalized input once with the same libx264 settings as offline runs, without
segmentation or masking. Produces original/segmented/recovered MP4s as identical copies
(naive streaming: the viewer receives the full encoded game stream).

Used as Experiment 1 reference arm so RD compares RESPAWN against fair pure streaming,
not a second YOLO pipeline variant.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from common import ensure_dir


def _ffprobe_frame_count(video: Path) -> int:
    for cmd in (
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
            str(video),
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:nw=1",
            str(video),
        ],
    ):
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            n = int(out)
            if n > 0:
                return n
        except Exception:
            continue
    return 0


def _empty_per_frame_entry() -> dict[str, Any]:
    return {
        "object_present_model": 0,
        "object_successfully_masked": 0,
        "preprocess_ms": 0.0,
        "inference_ms": 0.0,
        "postprocess_ms": 0.0,
        "stitching_ms": 0.0,
        "recover_ms": 0.0,
        "paint_ms": 0.0,
        "masking_ms": 0.0,
        "encode_baseline_ms": 0.0,
        "encode_masked_ms": 0.0,
        "latent_minted_regions": 0,
        "latent_reused_regions": 0,
        "masked_bbox_pct": 0.0,
        "masked_alpha_pct": 0.0,
        "changed_pixels": 0,
        "changed_pixels_pct": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pure streaming baseline encode (matched x264, no masking).")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bitrate-mbps", type=float, required=True)
    ap.add_argument("--enc-maxrate-mbps", type=float, default=None)
    ap.add_argument("--enc-bufsize-mbits", type=float, default=None)
    ap.add_argument("--enc-gop", type=int, default=60)
    ap.add_argument("--enc-preset", type=str, default="superfast")
    ap.add_argument("--enc-tune", type=str, default="none")
    ap.add_argument("--enc-scenecut", type=int, default=0)
    ap.add_argument("--enc-aud", type=int, default=1)
    ap.add_argument("--enc-repeat-headers", type=int, default=1)
    ap.add_argument(
        "--enc-open-gop-defaults",
        action="store_true",
        help="Omit fixed keyint/min-keyint/scenecut in -x264-params (match offline --enc-open-gop-defaults).",
    )
    args = ap.parse_args()

    br = float(args.bitrate_mbps)
    maxr = float(args.enc_maxrate_mbps) if args.enc_maxrate_mbps is not None else br
    buf_mbits = float(args.enc_bufsize_mbits) if args.enc_bufsize_mbits is not None else 2.0 * br

    out_dir = args.out_dir
    ensure_dir(out_dir)
    orig = out_dir / "original_output.mp4"

    if args.enc_open_gop_defaults:
        x264 = (
            f"bframes=0:"
            f"repeat-headers={int(args.enc_repeat_headers)}:aud={int(args.enc_aud)}"
        )
    else:
        x264 = (
            f"keyint={int(args.enc_gop)}:min-keyint={int(args.enc_gop)}:"
            f"bframes=0:scenecut={int(args.enc_scenecut)}:"
            f"repeat-headers={int(args.enc_repeat_headers)}:aud={int(args.enc_aud)}"
        )
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(args.input),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(args.enc_preset),
    ]
    # Match offline_processor: omit -tune when set to "none" (ffmpeg has no "none" tune).
    tune = str(args.enc_tune).strip()
    if tune and tune.lower() != "none":
        cmd += ["-tune", tune]
    cmd += [
        "-profile:v",
        "baseline",
        "-level:v",
        "4.2",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        f"{br}M",
        "-maxrate",
        f"{maxr}M",
        "-bufsize",
        f"{buf_mbits}M",
        "-x264-params",
        x264,
        str(orig),
    ]
    subprocess.run(cmd, check=True)

    seg = out_dir / "segmented_output.mp4"
    rec = out_dir / "recovered_output.mp4"
    shutil.copy2(orig, seg)
    shutil.copy2(orig, rec)

    n_frames = _ffprobe_frame_count(orig)
    if n_frames <= 0:
        raise SystemExit(f"Could not determine frame count for {orig}")

    per_frame = [_empty_per_frame_entry() for _ in range(n_frames)]
    report = {
        "avg_psnr": 0.0,
        "avg_ssim": 0.0,
        "avg_recovered_psnr": 0.0,
        "avg_recovered_ssim": 0.0,
        "commandline": " ".join(cmd),
        "pure_streaming_baseline": True,
        "encoder_settings": {
            "enc_bitrate_mbps": br,
            "enc_maxrate_mbps": maxr,
            "enc_bufsize_mbits": buf_mbits,
            "enc_gop": int(args.enc_gop),
            "enc_preset": args.enc_preset,
            "enc_tune": args.enc_tune,
            "enc_scenecut": int(args.enc_scenecut),
            "enc_open_gop_defaults": bool(args.enc_open_gop_defaults),
            "enc_codec": "libx264",
        },
        "per_frame": per_frame,
    }
    (out_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (out_dir / "msk1_payloads.bin").write_bytes(b"")
    print(f"[pure_streaming] Wrote {orig} ({n_frames} frames), report.json, empty msk1.")


if __name__ == "__main__":
    main()
