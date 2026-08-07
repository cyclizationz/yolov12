#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]
MODELS = {
    "spaceflight_two_class": REPO / "deployment" / "yolov12n_spaceflight_cockpit_spaceship_e300_v1.pt",
    "fc5_one_class": REPO / "runs" / "segment" / "yolov12n_fc5_seg_v1_e300_p0" / "weights" / "best.pt",
    "fm6_one_class": REPO / "runs" / "segment" / "yolov12n_racing_e300_split1" / "weights" / "best.pt",
}


def metric_rows(name: str, path: Path, metrics: Any, model_names: dict[int, str]) -> list[dict[str, Any]]:
    box_by_class = {
        int(class_id): float(ap_values.mean())
        for class_id, ap_values in zip(metrics.box.ap_class_index, metrics.box.ap)
    }
    mask_by_class = {
        int(class_id): float(ap_values.mean())
        for class_id, ap_values in zip(metrics.seg.ap_class_index, metrics.seg.ap)
    }
    rows = [
        {
            "model": name,
            "model_path": str(path),
            "scope": "all_dataset_classes",
            "class_id": "",
            "class_name": "",
            "model_native_names": json.dumps(model_names, sort_keys=True),
            "box_map50": float(metrics.box.map50),
            "box_map50_95": float(metrics.box.map),
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
            "preprocess_ms_per_image": float(metrics.speed.get("preprocess", 0.0)),
            "inference_ms_per_image": float(metrics.speed.get("inference", 0.0)),
            "postprocess_ms_per_image": float(metrics.speed.get("postprocess", 0.0)),
        }
    ]
    for class_id, class_name in enumerate(("cockpit", "spaceship")):
        rows.append(
            {
                **rows[0],
                "scope": "dataset_class",
                "class_id": class_id,
                "class_name": class_name,
                "box_map50": "",
                "box_map50_95": box_by_class.get(class_id, 0.0),
                "mask_map50": "",
                "mask_map50_95": mask_by_class.get(class_id, 0.0),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate spaceflight, FC5, and FM6 checkpoints on spaceflight labels.")
    parser.add_argument("--data", type=Path, default=REPO / "spaceflight_cockpit.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "record" / "RESPAWN2026" / "spaceflight_v1" / "model_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison checkpoint: {path}")
        model = YOLO(str(path))
        metrics = model.val(
            data=str(args.data),
            imgsz=640,
            batch=16,
            device=0,
            plots=False,
            project=str(args.out_dir),
            name=name,
            exist_ok=True,
        )
        rows.extend(metric_rows(name, path, metrics, dict(model.names)))

    csv_path = args.out_dir / "spaceflight_model_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dataset": str(args.data),
        "rows": rows,
        "comparability": {
            "spaceflight_two_class": "Direct two-class evaluation.",
            "fc5_one_class": "Out-of-domain one-class checkpoint; class 0 is scored against dataset class 0 by index. It has no spaceship output.",
            "fm6_one_class": "Out-of-domain one-class checkpoint; class 0 is scored against dataset class 0 by index. It has no spaceship output.",
        },
    }
    (args.out_dir / "spaceflight_model_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Spaceflight model comparison",
        "",
        "The new model is directly comparable on both classes. FC5 and FM6 are one-class, out-of-domain controls: only their output index 0 can be scored against cockpit, and they cannot predict spaceship.",
        "",
        "| Model | Box mAP50-95 | Mask mAP50-95 | Inference ms/image |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in (row for row in rows if row["scope"] == "all_dataset_classes"):
        lines.append(
            f"| {row['model']} | {row['box_map50_95']:.4f} | {row['mask_map50_95']:.4f} | {row['inference_ms_per_image']:.3f} |"
        )
    (args.out_dir / "spaceflight_model_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
