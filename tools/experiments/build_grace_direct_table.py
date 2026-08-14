#!/usr/bin/env python3
"""Build the matched-quality GRACE versus RESPAWN accounting table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLIPS = ("fc5_00", "fm6_00", "spaceflight_00")
LABELS = {
    "fc5_00": "FC5",
    "fm6_00": "FM6",
    "spaceflight_00": "SC",
}


def read_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(source))
    return rows


def quality_string(vmaf: float | None, ssim: float) -> str:
    vmaf_text = "--" if vmaf is None else f"{vmaf:.2f}"
    return f"{vmaf_text}/{ssim:.4f}"


def select_grace(
    summaries: list[dict[str, Any]], target_vmaf: float
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No completed GRACE model summaries")
    measured = [row for row in summaries if row["vmaf_mean"] is not None]
    if not measured:
        raise ValueError("No GRACE summaries have source-referenced VMAF")
    return min(
        measured,
        key=lambda row: (
            abs(float(row["vmaf_mean"]) - target_vmaf),
            int(row["actual_payload_bytes"]),
        ),
    )


def load_grace_summaries(root: Path, clip_id: str) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((root / clip_id).glob("model_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        quality_path = path.parent / "quality.json"
        summary["vmaf_mean"] = None
        summary["output_ssim_mean"] = summary["ssim_mean"]
        if quality_path.exists():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            summary["vmaf_mean"] = quality["full_frame"]["vmaf_mean"]
            summary["output_ssim_mean"] = quality["full_frame"]["ssim_mean"]
        summaries.append(summary)
    return summaries


def build(args: argparse.Namespace) -> None:
    points = read_csvs(args.points)
    overhead = read_csvs(args.overhead)
    respawn = {
        row["clip_id"]: row
        for row in points
        if row["variant"] == "respawn" and row["clip_id"] in CLIPS
    }
    resident = {
        row["clip_id"]: row
        for row in overhead
        if row["cache_mode"] == "partial_warm"
        and float(row["preload_fraction"]) == 1.0
        and row["clip_id"] in CLIPS
    }

    output_rows: list[dict[str, Any]] = []
    for clip_id in CLIPS:
        if clip_id not in respawn or clip_id not in resident:
            raise ValueError(f"Missing RESPAWN accounting for {clip_id}")
        accounting = resident[clip_id]
        respawn_quality_path = args.respawn_quality / f"{clip_id}.json"
        respawn_quality = json.loads(
            respawn_quality_path.read_text(encoding="utf-8")
        )["full_frame"]
        target_ssim = float(respawn_quality["ssim_mean"])
        target_vmaf = float(respawn_quality["vmaf_mean"])
        grace = select_grace(
            load_grace_summaries(args.grace_output, clip_id),
            target_vmaf,
        )
        baseline_bytes = float(accounting["baseline_total_bytes"])
        respawn_core_bytes = (
            baseline_bytes - float(accounting["final_net_saved_bytes"])
        )
        runtime_template_bytes = float(accounting["template_bytes_sent"])
        respawn_all_in_bytes = respawn_core_bytes + runtime_template_bytes
        grace_bytes = int(grace["actual_payload_bytes"])
        output_rows.append(
            {
                "title": LABELS[clip_id],
                "clip_id": clip_id,
                "grace_model": grace["model_id"],
                "grace_payload_mb": grace_bytes / 1e6,
                "grace_bsp_percent": 100.0
                * (1.0 - grace_bytes / baseline_bytes),
                "grace_vmaf_mean": grace["vmaf_mean"],
                "grace_ssim_mean": float(grace["output_ssim_mean"]),
                "respawn_video_rmd_mb": respawn_core_bytes / 1e6,
                "respawn_core_bsp_percent": 100.0
                * (1.0 - respawn_core_bytes / baseline_bytes),
                "runtime_template_mb": runtime_template_bytes / 1e6,
                "respawn_all_in_mb": respawn_all_in_bytes / 1e6,
                "respawn_all_in_bsp_percent": 100.0
                * (1.0 - respawn_all_in_bytes / baseline_bytes),
                "respawn_vmaf_mean": float(
                    target_vmaf
                ),
                "respawn_ssim_mean": target_ssim,
                "cache_start": "resident (preload excluded)",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "grace_direct_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Title} & \textbf{GRACE} & \multicolumn{3}{c}{\textbf{GRACE}} & \multicolumn{5}{c}{\textbf{\sysname}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-10}",
        r" & \textbf{model} & \textbf{Payload} & \textbf{BSP} & \textbf{VMAF/SSIM} & \textbf{Video+RMD} & \textbf{BSP} & \textbf{Tpl.} & \textbf{All-in} & \textbf{VMAF/SSIM} \\",
        r" & & \textbf{(MB)} & \textbf{(\%)} & & \textbf{(MB)} & \textbf{(\%)} & \textbf{(MB)} & \textbf{(MB)} & \\",
        r"\midrule",
    ]
    for row in output_rows:
        lines.append(
            f"{row['title']} & {row['grace_model']} & "
            f"{row['grace_payload_mb']:.2f} & "
            f"{row['grace_bsp_percent']:+.2f} & "
            f"{quality_string(row['grace_vmaf_mean'], row['grace_ssim_mean'])} & "
            f"{row['respawn_video_rmd_mb']:.2f} & "
            f"{row['respawn_core_bsp_percent']:+.2f} & "
            f"{row['runtime_template_mb']:.2f} & "
            f"{row['respawn_all_in_mb']:.2f} & "
            f"{quality_string(row['respawn_vmaf_mean'], row['respawn_ssim_mean'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = args.output_dir / "grace_direct_comparison.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "tex": str(tex_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    root = Path("record/RESPAWN2026")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grace-output",
        type=Path,
        default=root / "grace_comparison/full",
    )
    parser.add_argument(
        "--points",
        type=Path,
        nargs="+",
        default=[
            root / "exp35_crf_fm6_repartitioned/exp35_points.csv",
            root / "spaceflight_v2/exp35_crf/exp35_points.csv",
        ],
    )
    parser.add_argument(
        "--overhead",
        type=Path,
        nargs="+",
        default=[
            root
            / "exp2/measured_template_fm6_repartitioned/overhead_summary.csv",
            root / "spaceflight_v2/overhead_crf23/overhead_summary.csv",
        ],
    )
    parser.add_argument(
        "--respawn-quality",
        type=Path,
        default=root / "grace_comparison/full/respawn_quality",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "grace_comparison/full",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
