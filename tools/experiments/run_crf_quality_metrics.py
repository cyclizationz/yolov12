#!/usr/bin/env python3
"""Measure and populate direct CRF23 quality metrics without post-hoc uplift."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "record" / "RESPAWN2026"
POINTS = ROOT / "exp35_crf" / "exp35_points.csv"
MANIFEST = ROOT / "manifest" / "offline_manifest.json"


def measure(
    row: dict[str, str],
    sources: dict[str, Path],
    *,
    threads: int,
    scale_height: int,
    sample_stride: int,
    force: bool,
) -> tuple[str, str, dict[str, Any]]:
    run = Path(row["run_dir"])
    out = run / "source_quality_metrics.json"
    if out.exists() and not force:
        return row["clip_id"], row["variant"], json.loads(out.read_text())
    dist = run / ("recovered_output.mp4" if row["variant"] == "respawn" else "original_output.mp4")
    cmd = [
        sys.executable,
        str(REPO / "tools" / "experiments" / "video_metrics.py"),
        "--ref", str(sources[row["clip_id"]]),
        "--dist", str(dist),
        "--threads", str(threads),
        "--scale-height", str(scale_height),
        "--sample-stride", str(max(1, sample_stride)),
        "--out", str(out),
    ]
    msk1 = run / "msk1_payloads.bin"
    if row["variant"] == "respawn" and msk1.exists():
        cmd += ["--msk1-bin", str(msk1)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return row["clip_id"], row["variant"], json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--scale-height", type=int, default=720)
    ap.add_argument("--sample-stride", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--points", type=Path, default=POINTS)
    ap.add_argument("--manifests", nargs="+", type=Path, default=[MANIFEST])
    args = ap.parse_args()

    sources: dict[str, Path] = {}
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text())
        sources.update(
            {item["clip_id"]: Path(item["normalized_path"]) for item in manifest}
        )
    with args.points.open(newline="") as src:
        rows = list(csv.DictReader(src))
        fieldnames = list(rows[0])

    measured: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(
                measure, row, sources,
                threads=max(1, args.threads_per_job),
                scale_height=args.scale_height,
                sample_stride=args.sample_stride,
                force=args.force,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            clip_id, variant, metrics = future.result()
            measured[(clip_id, variant)] = metrics
            print(f"measured {clip_id} {variant}", flush=True)

    metric_fields = [
        "vmaf_mean", "vmaf_p10", "ssim_mean", "psnr_mean",
        "roi_vmaf_mean", "roi_ssim_mean", "roi_psnr_mean", "roi_frame_count",
    ]
    for name in metric_fields:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        metrics = measured[(row["clip_id"], row["variant"])]
        full = metrics["full_frame"]
        roi = metrics["roi"]
        row["vmaf_mean"] = full["vmaf_mean"]
        row["vmaf_p10"] = full["vmaf_p10"]
        row["ssim_mean"] = full["ssim_mean"]
        row["psnr_mean"] = full["psnr_mean"]
        row["roi_vmaf_mean"] = roi["vmaf_mean"]
        row["roi_ssim_mean"] = roi["ssim_mean"]
        row["roi_psnr_mean"] = roi["psnr_mean"]
        row["roi_frame_count"] = roi["frame_count_with_roi"]

    by_clip = {(row["clip_id"], row["variant"]): row for row in rows}
    summary_fields = [
        "baseline_vmaf_mean", "respawn_vmaf_mean",
        "baseline_ssim_mean", "respawn_ssim_mean",
        "baseline_psnr_mean", "respawn_psnr_mean",
    ]
    for name in summary_fields:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        base = by_clip[(row["clip_id"], "pure_streaming")]
        resp = by_clip[(row["clip_id"], "respawn")]
        row["baseline_vmaf_mean"] = base["vmaf_mean"]
        row["respawn_vmaf_mean"] = resp["vmaf_mean"]
        row["baseline_ssim_mean"] = base["ssim_mean"]
        row["respawn_ssim_mean"] = resp["ssim_mean"]
        row["baseline_psnr_mean"] = base["psnr_mean"]
        row["respawn_psnr_mean"] = resp["psnr_mean"]

    tmp = args.points.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.points)
    print(json.dumps({"points": str(args.points), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
