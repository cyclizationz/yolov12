#!/usr/bin/env python3
"""
Build per-frame CSV for a given output folder.

Inputs (expected in --out-dir):
  - report.json
  - original_output.mp4
  - segmented_output.mp4
Optional:
  - object_gt.csv (from GUI) with columns: frame_id, object_present_gt

Output:
  - per_frame_metrics.csv (frame_id is 1-based)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from per_frame_bytes_mp4 import align_by_time, ffprobe_frame_sizes_in_decode_order, ffprobe_frames


def load_object_gt(p: Path) -> dict[int, int]:
    if not p.exists():
        return {}
    out: dict[int, int] = {}
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fid = int(row["frame_id"])
                gt = int(row["object_present_gt"])
            except Exception:
                continue
            out[fid] = gt
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tol-ms", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=None, help="Optional output path (defaults to <out-dir>/per_frame_metrics.csv)")
    args = ap.parse_args()

    out_dir = args.out_dir
    report_path = out_dir / "report.json"
    base_mp4 = out_dir / "original_output.mp4"
    masked_mp4 = out_dir / "segmented_output.mp4"
    if not report_path.exists():
        raise SystemExit(f"Missing {report_path}")
    if not base_mp4.exists():
        raise SystemExit(f"Missing {base_mp4}")
    if not masked_mp4.exists():
        raise SystemExit(f"Missing {masked_mp4}")

    rep: dict[str, Any] = json.loads(report_path.read_text())
    per = rep.get("per_frame", []) or []
    if not isinstance(per, list) or not per:
        raise SystemExit(f"{report_path} has no per_frame entries")

    # bytes/frame
    # Prefer strict index alignment when both MP4s have the same frame count.
    b0_idx = ffprobe_frame_sizes_in_decode_order(base_mp4)
    b1_idx = ffprobe_frame_sizes_in_decode_order(masked_mp4)
    if len(b0_idx) == len(b1_idx) and len(b0_idx) > 0:
        b0 = b0_idx
        b1 = b1_idx
    else:
        # Fallback: timestamp alignment (best-effort timestamp time)
        p0 = ffprobe_frames(base_mp4)
        p1 = ffprobe_frames(masked_mp4)
        _ts, b0, b1 = align_by_time(p0, p1, tol_ms=args.tol_ms)
    n = min(len(per), len(b0), len(b1))

    gt_map = load_object_gt(out_dir / "object_gt.csv")

    out_csv = args.out or (out_dir / "per_frame_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Column order requested for plotting (keep exact spelling/order).
    cols = [
        "frame_id",
        "baseline_bytes",
        "masked_bytes",
        "object_present_model",
        "object_present_gt",
        "object_successfully_masked",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "stitching_ms",
        "roi_filter_ms",
        "template_matching_ms",
        "motion_filter_ms",
        "masking_ms",
        "encode_baseline_ms",
        "encode_masked_ms",
        "bytes_delta (baseline-masked; >0 saving\t<0 overhead)",
        "latent_minted_regions",
        "latent_reused_regions",
        "latent_reuse_ratio",
        "masked_bbox_pct (sum bbox area / frame area; may double-count overlaps)",
        "masked_alpha_pct (sum alpha/mask pixels / frame area; may double-count overlaps)",
        "changed_pixels (unique painted/masked pixels in output frame)",
        "changed_pixels_pct (changed_pixels / frame area)",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n):
            item = per[i]
            frame_id = i + 1  # 1-based
            # Pixel-mode timing fields (only present when --timing is enabled).
            pixel_bootstrap_ms = float(item.get("pixel_bootstrap_ms", 0.0) or 0.0)
            pixel_roi_ms = float(item.get("pixel_roi_ms", 0.0) or 0.0)
            pixel_band_ms = float(item.get("pixel_band_ms", 0.0) or 0.0)
            pixel_row_ms = float(item.get("pixel_row_ms", 0.0) or 0.0)
            pixel_flow_ms = float(item.get("pixel_flow_ms", 0.0) or 0.0)
            latent_minted = int(item.get("latent_minted_regions", 0) or 0)
            latent_reused = int(item.get("latent_reused_regions", 0) or 0)
            latent_reuse_ratio = latent_reused / max(1, latent_minted + latent_reused)
            baseline_bytes = int(b0[i])
            masked_bytes = int(b1[i])

            row = {
                "frame_id": frame_id,
                "object_present_model": int(item.get("object_present_model", 0) or 0),
                "object_present_gt": gt_map.get(frame_id, ""),
                "object_successfully_masked": int(item.get("object_successfully_masked", 0) or 0),
                # YOLO-mode fields (will be 0 for pixel mode)
                "preprocess_ms": float(item.get("preprocess_ms", 0.0) or 0.0),
                "inference_ms": float(item.get("inference_ms", 0.0) or 0.0),
                "postprocess_ms": float(item.get("postprocess_ms", 0.0) or 0.0),
                "stitching_ms": float(item.get("stitching_ms", item.get("recover_ms", 0.0)) or 0.0),
                # Pixel-mode fields (requested)
                # ROI stage time (includes ROI matching overhead)
                "roi_filter_ms": pixel_roi_ms,
                # Template matching: bootstrap + band selection + band scan
                # (kept separate from roi_filter_ms to avoid double-counting)
                "template_matching_ms": pixel_bootstrap_ms + pixel_row_ms + pixel_band_ms,
                # Motion filter: optical flow stabilization
                "motion_filter_ms": pixel_flow_ms,
                # Common
                "masking_ms": float(item.get("masking_ms", item.get("paint_ms", 0.0)) or 0.0),
                "encode_baseline_ms": float(item.get("encode_baseline_ms", 0.0) or 0.0),
                "encode_masked_ms": float(item.get("encode_masked_ms", 0.0) or 0.0),
                "baseline_bytes": baseline_bytes,
                "masked_bytes": masked_bytes,
                "bytes_delta (baseline-masked; >0 saving\t<0 overhead)": baseline_bytes - masked_bytes,
                "latent_minted_regions": latent_minted,
                "latent_reused_regions": latent_reused,
                "latent_reuse_ratio": latent_reuse_ratio,
                "masked_bbox_pct (sum bbox area / frame area; may double-count overlaps)": float(item.get("masked_bbox_pct", 0.0) or 0.0),
                "masked_alpha_pct (sum alpha/mask pixels / frame area; may double-count overlaps)": float(item.get("masked_alpha_pct", 0.0) or 0.0),
                "changed_pixels (unique painted/masked pixels in output frame)": int(item.get("changed_pixels", 0) or 0),
                "changed_pixels_pct (changed_pixels / frame area)": float(item.get("changed_pixels_pct", 0.0) or 0.0),
            }
            w.writerow(row)

    print(f"Wrote {out_csv} ({n} frames)")


if __name__ == "__main__":
    main()


