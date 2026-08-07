#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "record" / "RESPAWN2026" / "spaceflight_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def main() -> None:
    model_doc = json.loads((ROOT / "model_comparison" / "spaceflight_model_comparison.json").read_text())
    model_rows = [row for row in model_doc["rows"] if row["scope"] == "all_dataset_classes"]
    clip_doc = json.loads((ROOT / "clip_provenance_and_verification.json").read_text())
    duo_doc = json.loads((ROOT / "duo_screen" / "duo_screen_summary.json").read_text())
    rd_rows = read_csv(ROOT / "rd" / "spaceflight_rd_points.csv")
    respawn_rows = [row for row in rd_rows if row["variant"] == "respawn"]
    rd_summary = {
        "operating_points": len(respawn_rows),
        "mean_bitrate_saving_pct": mean(
            100.0 * (1.0 - number(row, "respawn_achieved_bps") / number(row, "baseline_achieved_bps"))
            for row in respawn_rows
            if number(row, "baseline_achieved_bps") > 0
        ),
        "mean_respawn_vmaf": mean(number(row, "respawn_vmaf_mean") for row in respawn_rows),
        "mean_pure_streaming_vmaf": mean(number(row, "baseline_vmaf_mean") for row in respawn_rows),
        "mean_respawn_ssim": mean(number(row, "respawn_ssim_mean") for row in respawn_rows),
        "mean_pure_streaming_ssim": mean(number(row, "baseline_ssim_mean") for row in respawn_rows),
        "bd_rate": None,
        "bd_rate_reason": "No overlapping quality interval between pure streaming and RESPAWN curves on all clips.",
    }

    ablation_rows = read_csv(ROOT / "ablation" / "ablation_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ablation_rows:
        grouped[row["config_name"]].append(row)
    ablations = []
    for config, rows in sorted(grouped.items()):
        ablations.append(
            {
                "config": config,
                "kind": rows[0]["kind"],
                "clips": len(rows),
                "mean_bsp_pct": mean(number(row, "bsp_pct") for row in rows),
                "mean_recovered_ssim": mean(number(row, "avg_recovered_ssim") for row in rows),
                "mean_artifact_incidents_per_10k_frames": mean(
                    number(row, "artifact_incidents_per_10k_frames") for row in rows
                ),
                "mean_match_rate": mean(number(row, "match_rate") for row in rows),
            }
        )

    overhead_rows = read_csv(ROOT / "overhead" / "overhead_summary.csv")
    overhead_by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overhead_rows:
        overhead_by_mode[row["cache_mode"]].append(row)
    overhead = {
        mode: {
            "cases": len(rows),
            "mean_final_net_saved_pct": mean(number(row, "final_net_saved_pct") for row in rows),
            "mean_forced_raw_fraction": mean(number(row, "forced_raw_fraction") for row in rows),
        }
        for mode, rows in sorted(overhead_by_mode.items())
    }

    resource_rows = read_csv(ROOT / "resource" / "spaceflight_resource_summary.csv")
    native_resource = read_csv(REPO / "record" / "RESPAWN2026" / "exp6" / "exp6_resource_component_summary.csv")
    resource = {
        "spaceflight_same_five_clip_mean": {
            key: mean(number(row, key) for row in resource_rows)
            for key in (
                "system_total_ms",
                "detect_ms",
                "dict_match_ms",
                "mask_fill_ms",
                "stitch_ms",
                "avg_cpu_pct",
                "max_rss_mb",
                "avg_gpu_pct",
                "max_gpu_mem_mb",
                "avg_gpu_power_w",
            )
        },
        "native_fc5_fm6_controls": [
            {
                "game": row["game"],
                "clip_id": row["clip_id"],
                "frames": int(float(row["frames"])),
                "avg_system_ms": number(row, "avg_system_ms"),
                "effective_fps": number(row, "effective_fps"),
                "cpu_pct": number(row, "cpu_pct"),
                "max_rss_mb": number(row, "max_rss_mb"),
                "gpu_util_avg_pct": number(row, "gpu_util_avg_pct"),
                "gpu_mem_max_mb": number(row, "gpu_mem_max_mb"),
            }
            for row in native_resource
            if row["game"].lower() in {"fc5", "fm6"}
        ],
        "comparability": "Spaceflight uses five 90-frame 1280x720 clips; native FC5/FM6 controls use one 300-frame 1920x1080 clip each. Resource numbers are contextual, not a controlled model-only comparison.",
    }

    summary: dict[str, Any] = {
        "training": json.loads(
            (
                REPO
                / "runs"
                / "segment"
                / "yolov12n_spaceflight_cockpit_spaceship_e300_v1"
                / "training_export_summary.json"
            ).read_text()
        ),
        "clip_verification": {
            "clip_count": clip_doc["actual_clip_count"],
            "all_frames_verified_positive": clip_doc["all_frames_verified_positive"],
            "total_verified_frames": sum(row["verified_positive_frames"] for row in clip_doc["clips"]),
            "clips": [
                {
                    "clip_id": row["clip_id"],
                    "source_path": row["source_path"],
                    "source_frames": [row["start_frame"], row["end_frame"]],
                    "clip_path": row["clip_path"],
                    "verified": f"{row['verified_positive_frames']}/{row['encoded_frame_count']}",
                    "class_detection_frame_counts": row["class_detection_frame_counts"],
                }
                for row in clip_doc["clips"]
            ],
        },
        "accuracy_and_segmentation": model_rows,
        "rate_distortion": rd_summary,
        "duo_screen": duo_doc["game_summary"]["spaceflight"],
        "matching_and_fill_ablation": ablations,
        "cache_and_delay_overhead": overhead,
        "resource_and_latency": resource,
        "artifacts": {
            "manifest": str(ROOT / "offline_manifest.json"),
            "rd_points": str(ROOT / "rd" / "spaceflight_rd_points.csv"),
            "ablation_csv": str(ROOT / "ablation" / "ablation_summary.csv"),
            "overhead_csv": str(ROOT / "overhead" / "overhead_summary.csv"),
            "generality_csv": str(ROOT / "generality" / "generality_summary.csv"),
            "resource_csv": str(ROOT / "resource" / "spaceflight_resource_summary.csv"),
        },
        "exclusions": [
            {
                "entry_point": "run_ablation_suite.py gating grid",
                "status": "excluded_operationally_broad",
                "reason": "64 gate combinations x 5 clips (320 additional encodes). The complete matching and fill subset ran: 10 configurations x 5 clips = 50 encodes.",
            },
            {
                "entry_point": "Mario pixel-template experiments",
                "status": "not_applicable",
                "reason": "Pixel-template detection is a different non-YOLO path.",
            },
            {
                "entry_point": "thin-client transport and inference-worker sweep",
                "status": "not_repeated",
                "reason": "These test transport/cache plumbing and worker concurrency rather than segmentation-model quality; existing FC5/FM6 results already show no inference-worker gain.",
            },
            {
                "entry_point": "BD-rate",
                "status": "unsupported_by_measured_curves",
                "reason": rd_summary["bd_rate_reason"],
            },
        ],
    }
    (ROOT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for model in model_rows:
        rows.append(
            {
                "category": "segmentation",
                "name": model["model"],
                "metric": "mask_map50_95",
                "value": model["mask_map50_95"],
                "unit": "ratio",
                "comparability": model_doc["comparability"][model["model"]],
            }
        )
    for key, value in rd_summary.items():
        if isinstance(value, (int, float)):
            rows.append(
                {
                    "category": "rate_distortion",
                    "name": "spaceflight",
                    "metric": key,
                    "value": value,
                    "unit": "mixed",
                    "comparability": "Five-clip spaceflight suite.",
                }
            )
    with (ROOT / "comparison_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sf_model = next(row for row in model_rows if row["model"] == "spaceflight_two_class")
    fc5 = next(row for row in model_rows if row["model"] == "fc5_one_class")
    fm6 = next(row for row in model_rows if row["model"] == "fm6_one_class")
    res = resource["spaceflight_same_five_clip_mean"]
    lines = [
        "# Spaceflight two-class RESPAWN evaluation",
        "",
        "## Main results",
        f"- Validation: box mAP50-95 {sf_model['box_map50_95']:.4f}; mask mAP50-95 {sf_model['mask_map50_95']:.4f}.",
        f"- Out-of-domain controls on the same labels: FC5 mask mAP50-95 {fc5['mask_map50_95']:.4f}; FM6 {fm6['mask_map50_95']:.4f}. FC5 predicts `gun`; FM6 is one-class `cockpit`; neither can predict `spaceship`.",
        f"- Clips: exactly {clip_doc['actual_clip_count']}, with {sum(row['verified_positive_frames'] for row in clip_doc['clips'])}/450 encoded frames model-positive.",
        f"- RD: mean delivered bitrate saving {rd_summary['mean_bitrate_saving_pct']:.2f}%, but mean VMAF falls from {rd_summary['mean_pure_streaming_vmaf']:.2f} to {rd_summary['mean_respawn_vmaf']:.2f}. BD-rate is unsupported because quality ranges do not overlap.",
        f"- Exp6-style server: mean system path {res['system_total_ms']:.2f} ms/frame, CPU {res['avg_cpu_pct']:.1f}%, average GPU utilization {res['avg_gpu_pct']:.1f}%.",
        f"- Duo screen: all {duo_doc['game_summary']['spaceflight']['clip_count']} clips classified promising (mean Ref/Raw run length 90 frames).",
        "",
        "## Interpretation",
        "The new checkpoint is the only valid two-class model and strongly outperforms the out-of-domain FC5/FM6 checkpoints on the spaceflight labels. The current RESPAWN masking configuration saves some bits but does not preserve enough quality for an equal-quality or BD-rate claim. Cache-preload accounting is also unfavorable for these very short clips because template delivery dominates.",
        "",
        "## Protocol and exclusions",
        "- Completed: 300-epoch training/export/validation; every-frame clip verification; same-label FC5/FM6 controls; duo screen; 25-point RD; CRF23 GOP runs; overhead/cache-delay analysis; generality analysis; 50-run matching/fill ablation; five-clip server resource benchmark.",
        "- Excluded: the 320-run gate hyperparameter grid as operationally broad; Mario pixel-only experiments as inapplicable; thin-client and inference-worker repetition as model-independent plumbing tests.",
        "- Native FC5/FM6 resource figures are retained only as context because their clips, resolution, frame count, and object workloads differ.",
        "",
        f"Machine-readable summary: `{ROOT / 'comparison_summary.json'}`",
    ]
    (ROOT / "comparison_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(ROOT / "comparison_summary.json"), "csv": str(ROOT / "comparison_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
