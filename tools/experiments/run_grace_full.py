#!/usr/bin/env python3
"""Run a full-clip, zero-loss GRACE encode outside the GRACE checkout.

The released GRACE evaluator is loaded at runtime from an external checkout so
its academic-licensed source is not redistributed here. Unlike grace-gpu.py's
16-frame sweep, this adapter processes the requested frame range, writes a
framed payload containing the actual entropy/BPG byte strings, and exports the
in-memory decoded frames as lossless FFV1 for source-referenced measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor


MODEL_SCALE = {
    "64": 0.25,
    "128": 0.5,
    "256": 0.5,
    "512": 0.5,
    "1024": 0.5,
    "2048": 1.0,
    "4096": 1.0,
    "6144": 1.0,
    "8192": 1.0,
    "12288": 1.0,
    "16384": 1.0,
}


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_upstream_namespace(grace_root: Path) -> dict[str, Any]:
    """Load helper classes without running the upstream 16-frame main call."""
    evaluator = require_file(grace_root / "grace-gpu.py", "GRACE evaluator")
    source = evaluator.read_text(encoding="utf-8")
    model_init = "models = init_ae_model()"
    main_call = 'run_one_file("INDEX.txt", "results/grace")'
    if source.count(model_init) != 1 or source.count(main_call) != 1:
        raise RuntimeError(
            "Pinned GRACE evaluator layout changed; refusing unsafe runtime patch"
        )
    source = source.replace(model_init, "models = {}", 1)
    source = source.replace(main_call, "", 1)

    namespace: dict[str, Any] = {
        "__file__": str(evaluator),
        "__name__": "_respawn_grace_runtime",
    }
    sys.path.insert(0, str(grace_root))
    exec(compile(source, str(evaluator), "exec"), namespace)
    return namespace


def probe_video(path: Path) -> tuple[int, int, float, int | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}")
    return width, height, fps, frame_count if frame_count > 0 else None


def pad_frame(rgb: np.ndarray, multiple: int = 128) -> Image.Image:
    height, width = rgb.shape[:2]
    padded_width = math.ceil(width / multiple) * multiple
    padded_height = math.ceil(height / multiple) * multiple
    padded = np.pad(
        rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded)


def tensor_rgb8(frame: Any, width: int, height: int) -> np.ndarray:
    cropped = frame[:, :height, :width].detach().float().clamp(0, 1)
    return (
        cropped.mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def psnr(reference: Any, decoded: Any, width: int, height: int) -> float:
    ref = reference[:, :height, :width].float()
    dec = decoded[:, :height, :width].float().to(ref.device)
    mse = float((ref - dec).square().mean().item())
    return 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(namespace: dict[str, Any], reference: Any, decoded: Any, width: int, height: int) -> float:
    ref = reference[:, :height, :width]
    dec = decoded[:, :height, :width]
    return float(namespace["SSIM"](ref, dec))


def encode_p_streams(eframe: Any, entropy_coder: Any) -> list[bytes]:
    code = eframe.code.float()
    z = eframe.z.float()
    mv_size = int(np.prod(eframe.shapex))
    mv = torch_reshape(code[:mv_size], eframe.shapex)
    residual = torch_reshape(code[mv_size:], eframe.shapey)
    sigma = entropy_coder.model.respriorDecoder(z)
    residual_stream = compress_res_modern_torchac(
        residual, sigma, entropy_coder
    )
    motion_stream, _ = entropy_coder.compress_mv(mv)
    z_stream, _ = entropy_coder.compress_z(z)
    return [residual_stream, motion_stream, z_stream, eframe.ipart.code]


def compress_res_modern_torchac(
    residual: Any, sigma: Any, entropy_coder: Any
) -> bytes:
    """Mirror upstream residual coding with the CDF moved to CPU.

    GRACE's pinned torchac accepted a CUDA CDF. Modern torchac requires both
    the normalized CDF and symbols on CPU; this changes placement, not coding.
    """
    residual = residual.clamp(-16, 16)
    sigma = sigma.clamp(1e-5, 1e10)
    distribution = entropy_coder.compress_res.__globals__["torch"].distributions.laplace.Laplace(
        sigma.new_zeros(sigma.shape), sigma
    )
    max_range = int(
        min(entropy_coder.model.mxrange, residual.abs().max().item())
    )
    shifted = residual + max_range
    cdfs = [
        distribution.cdf(sigma.new_tensor(index - 0.5)).unsqueeze(-1)
        for index in range(-max_range, max_range + 2)
    ]
    cdf = entropy_coder.compress_res.__globals__["torch"].cat(
        cdfs, dim=-1
    ).detach().cpu()
    symbols = shifted.detach().cpu().to(
        entropy_coder.compress_res.__globals__["torch"].int16
    )
    torchac = entropy_coder.compress_res.__globals__["torchac"]
    return torchac.encode_float_cdf(
        cdf, symbols, check_input_bounds=False
    )


def torch_reshape(value: Any, shape: Any) -> Any:
    # Keeping torch imported through the external namespace avoids requiring
    # this adapter's host interpreter to match the dedicated GRACE environment.
    return value.reshape(tuple(int(item) for item in shape))


def write_payload_frame(
    output: BinaryIO,
    *,
    frame_index: int,
    frame_type: str,
    streams: list[bytes],
    metadata: dict[str, Any],
) -> int:
    header = json.dumps(
        {
            "frame_index": frame_index,
            "frame_type": frame_type,
            "stream_lengths": [len(stream) for stream in streams],
            **metadata,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    before = output.tell()
    output.write(struct.pack(">I", len(header)))
    output.write(header)
    for stream in streams:
        output.write(stream)
    return output.tell() - before


def start_ffv1(path: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def run(args: argparse.Namespace) -> None:
    grace_root = args.grace_root.expanduser().resolve()
    video = require_file(args.input, "input video")
    model_path = require_file(
        grace_root / "models/grace" / f"{args.model_id}_freeze.model",
        f"GRACE {args.model_id} checkpoint",
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(grace_root)
    try:
        namespace = load_upstream_namespace(grace_root)
        grace_coder = namespace["GraceInterface"](
            {"path": str(model_path)},
            scale_factor=MODEL_SCALE[args.model_id],
        )
        model = namespace["AEModel"](None, grace_coder)

        width, height, fps, reported_frames = probe_video(video)
        cap = cv2.VideoCapture(str(video))
        decoded_path = output_dir / "decoded.mkv"
        payload_path = output_dir / "payload.grf"
        writer = (
            None
            if args.no_decoded_video
            else start_ffv1(decoded_path, width, height, fps)
        )
        if writer is not None and writer.stdin is None:
            raise RuntimeError("Failed to open FFmpeg input pipe")

        rows: list[dict[str, Any]] = []
        reference_frame = None
        started = time.monotonic()
        with payload_path.open("wb") as payload:
            payload.write(b"GRF1")
            frame_index = 0
            while args.max_frames is None or frame_index < args.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_frame = pad_frame(rgb)
                original_tensor = to_tensor(pil_frame)

                is_iframe = frame_index == 0
                if reference_frame is not None:
                    model.update_reference(reference_frame)
                eframe, estimated_size = model.encode_frame(
                    pil_frame, isIframe=is_iframe
                )
                decoded = model.decode_frame(eframe)

                if is_iframe:
                    streams = [eframe.code]
                    metadata = {
                        "original_size": [width, height],
                        "padded_size": list(pil_frame.size),
                        "shape": [int(eframe.shapex), int(eframe.shapey)],
                    }
                else:
                    streams = encode_p_streams(eframe, grace_coder.ecmodel)
                    metadata = {
                        "original_size": [width, height],
                        "padded_size": list(pil_frame.size),
                        "motion_shape": list(eframe.shapex),
                        "residual_shape": list(eframe.shapey),
                        "z_shape": list(eframe.z.shape),
                        "patch_offset": [
                            int(eframe.ipart.offset_width),
                            int(eframe.ipart.offset_height),
                        ],
                        "patch_shape": [
                            int(eframe.ipart.shapey),
                            int(eframe.ipart.shapex),
                        ],
                    }
                actual_size = write_payload_frame(
                    payload,
                    frame_index=frame_index,
                    frame_type="I" if is_iframe else "P",
                    streams=streams,
                    metadata=metadata,
                )
                frame_psnr = psnr(
                    original_tensor, decoded, width=width, height=height
                )
                frame_ssim = ssim(
                    namespace,
                    original_tensor,
                    decoded,
                    width=width,
                    height=height,
                )
                if writer is not None:
                    assert writer.stdin is not None
                    writer.stdin.write(
                        tensor_rgb8(decoded, width, height).tobytes()
                    )
                rows.append(
                    {
                        "frame_index": frame_index,
                        "frame_type": "I" if is_iframe else "P",
                        "actual_payload_bytes": actual_size,
                        "upstream_estimated_bytes": float(estimated_size),
                        "psnr": frame_psnr,
                        "ssim": frame_ssim,
                    }
                )
                reference_frame = decoded
                frame_index += 1
                if frame_index % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "frames": frame_index,
                                "payload_bytes": payload.tell() - 4,
                                "elapsed_s": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
        cap.release()
        if writer is not None:
            assert writer.stdin is not None
            writer.stdin.close()
            return_code = writer.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg FFV1 writer exited with {return_code}"
                )
        if not rows:
            raise RuntimeError(f"No frames decoded from {video}")

        with (output_dir / "per_frame.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            csv_writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)

        summary = {
            "status": "full_zero_loss",
            "input": str(video),
            "model_id": args.model_id,
            "frames": len(rows),
            "reported_input_frames": reported_frames,
            "resolution": [width, height],
            "fps": fps,
            "padding": "edge-pad bottom/right to next multiple of 128; crop after decode",
            "actual_payload_bytes": payload_path.stat().st_size - 4,
            "container_bytes": payload_path.stat().st_size,
            "upstream_estimated_bytes": sum(
                row["upstream_estimated_bytes"] for row in rows
            ),
            "psnr_mean": sum(row["psnr"] for row in rows) / len(rows),
            "ssim_mean": sum(row["ssim"] for row in rows) / len(rows),
            "decoded_video": None if writer is None else str(decoded_path),
            "payload_file": str(payload_path),
            "elapsed_s": time.monotonic() - started,
            "semantics": (
                "Actual entropy/BPG payloads plus per-frame framing metadata; "
                "decoded frames are reconstructed in memory from the same "
                "latent codes because upstream entropy_decode is unimplemented."
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    finally:
        os.chdir(original_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grace-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=list(MODEL_SCALE), required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-decoded-video", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
