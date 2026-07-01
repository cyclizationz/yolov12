#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import RESPAWN2026_DIR, python_bin


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the offline-first RESPAWN experiment suite end-to-end.")
    ap.add_argument("--root-out-dir", type=Path, default=RESPAWN2026_DIR)
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--skip-duo", action="store_true")
    ap.add_argument("--skip-rd", action="store_true")
    ap.add_argument("--skip-overhead", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-generality", action="store_true")
    ap.add_argument("--limit-clips", type=int, default=0)
    args, unknown = ap.parse_known_args()

    scripts_dir = Path("/home/tiehangz/proj/yolov12/tools/experiments")
    manifest_json = args.root_out_dir / "manifest" / "offline_manifest.json"

    if not args.skip_manifest:
        subprocess.run([python_bin(), str(scripts_dir / "build_manifest.py"), "--out-dir", str(args.root_out_dir / "manifest")], check=True)
    if not args.skip_duo:
        cmd = [python_bin(), str(scripts_dir / "run_duo_screen.py"), "--manifest", str(manifest_json), "--out-dir", str(args.root_out_dir / "duo_screen")]
        if args.limit_clips > 0:
            cmd += ["--limit", str(args.limit_clips)]
        subprocess.run(cmd, check=True)
    if not args.skip_rd:
        cmd = [python_bin(), str(scripts_dir / "run_rd_suite.py"), "--manifest", str(manifest_json), "--out-dir", str(args.root_out_dir / "exp1")]
        if args.limit_clips > 0:
            cmd += ["--limit-clips", str(args.limit_clips)]
        subprocess.run(cmd + unknown, check=True)
    if not args.skip_overhead:
        subprocess.run(
            [
                python_bin(),
                str(scripts_dir / "analyze_overhead.py"),
                "--manifest",
                str(manifest_json),
                "--rd-dir",
                str(args.root_out_dir / "exp35_crf"),
                "--rd-points",
                str(args.root_out_dir / "exp35_crf" / "exp35_points.csv"),
                "--out-dir",
                str(args.root_out_dir / "exp3"),
                "--synthetic-template-model",
            ],
            check=True,
        )
    if not args.skip_ablation:
        subprocess.run(
            [
                python_bin(),
                str(scripts_dir / "run_ablation_suite.py"),
                "--manifest",
                str(manifest_json),
                "--out-dir",
                str(args.root_out_dir / "exp4"),
            ],
            check=True,
        )
    if not args.skip_generality:
        subprocess.run(
            [
                python_bin(),
                str(scripts_dir / "analyze_generality.py"),
                "--manifest",
                str(manifest_json),
                "--rd-points",
                str(args.root_out_dir / "exp35_crf" / "exp35_points.csv"),
                "--out-dir",
                str(args.root_out_dir / "exp5"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
