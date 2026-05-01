#!/usr/bin/env python3
"""
Run the original BSP plotting script on a single run directory.

This uses `video/figures/plotting.py` (the original BSP/BW plots code) and feeds it a
single CSV generated from `report.json` + MP4 packet sizes.

Outputs PDF figures (BSP overall, BSP success-only, etc.) into --out-dir.

Example:
  MPLBACKEND=Agg conda run -n yolov12 python tools/metrics/plot_bsp_from_run.py \
    --run-dir record/final3/e4_fc5_x264_crf18_dominant_max2000 \
    --label fc5_x264_crf18_max2000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_per_frame_csv(run_dir: Path, out_csv: Path) -> None:
    script = _repo_root() / "tools" / "metrics" / "build_per_frame_csv.py"
    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(run_dir),
        "--out",
        str(out_csv),
    ]
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--label", type=str, default=None, help="Label used in the plots (defaults to run-dir name)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write PDFs (defaults to <run-dir>/bsp_figs)",
    )
    ap.add_argument("--reuse-csv", action="store_true", help="If set, skip rebuilding per_frame_metrics.csv if it exists")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Missing run dir: {run_dir}")

    out_dir = args.out_dir or (run_dir / "bsp_figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "per_frame_metrics.csv"
    if args.reuse_csv and out_csv.exists():
        pass
    else:
        _build_per_frame_csv(run_dir, out_csv)

    label = (args.label or run_dir.name).strip()
    if not label:
        label = run_dir.name

    # Use the original plotting script implementation.
    # We override its CSV_FILES and OUTDIR globals to point at our single-run CSV.
    sys.path.insert(0, str(_repo_root()))
    from video.figures import plotting as vp  # noqa: E402

    vp.OUTDIR = out_dir
    vp.OUTDIR.mkdir(parents=True, exist_ok=True)
    vp.CSV_FILES = {label: str(out_csv)}

    # Run the plots the user asked for: BSP + BSP(success-only) and related BW summaries.
    vp.plot5_box_baseline_vs_masked_all()
    vp.plot6_box_baseline_vs_masked_success_only()
    vp.plot7_box_bsp_all()
    vp.plot8_box_bsp_success_only()
    vp.plot9_masking_success_percentage()

    vp.plot_bsp_bar_mean_std_all_csvs()
    vp.plot_bsp_bar_mean_std_success_only_all_csvs()
    vp.plot_bsp_bar_overall_with_per_second_std_all_csvs()
    vp.plot_bsp_bar_overall_with_per_second_std_success_only_all_csvs()

    print(str(out_dir))


if __name__ == "__main__":
    main()

