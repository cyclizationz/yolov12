import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


os.environ["DISABLE_FLASH_ATTN"] = "1"


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    run_name = "yolov12n_spaceflight_cockpit_spaceship_e300_v1"
    base_pt = repo_root / "modification" / "model" / "yolov12n-seg.pt"
    data_yaml = repo_root / "spaceflight_cockpit.yaml"
    run_dir = repo_root / "runs" / "segment" / run_name

    if not base_pt.exists():
        raise FileNotFoundError(f"Missing base weights: {base_pt}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")

    model = YOLO(str(base_pt))
    model.train(
        data=str(data_yaml),
        epochs=300,
        imgsz=640,
        batch=16,
        patience=0,
        seed=1337,
        deterministic=True,
        workers=8,
        device=0,
        project=str(repo_root / "runs" / "segment"),
        name=run_name,
        exist_ok=False,
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_pt}")

    best = YOLO(str(best_pt))
    metrics = best.val(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        project=str(run_dir),
        name="best_validation",
        exist_ok=True,
    )
    onnx_path = Path(
        best.export(
            format="onnx",
            simplify=True,
            opset=18,
            imgsz=640,
            dynamic=False,
        )
    )

    deploy_pt = repo_root / "deployment" / f"{run_name}.pt"
    deploy_onnx = repo_root / "deployment" / f"{run_name}.onnx"
    shutil.copy2(best_pt, deploy_pt)
    shutil.copy2(onnx_path, deploy_onnx)

    summary = {
        "run_name": run_name,
        "base_checkpoint": str(base_pt),
        "data_yaml": str(data_yaml),
        "best_checkpoint": str(best_pt),
        "deployment_pt": str(deploy_pt),
        "deployment_onnx": str(deploy_onnx),
        "validation": {
            "box_map50": float(metrics.box.map50),
            "box_map50_95": float(metrics.box.map),
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
        },
    }
    (run_dir / "training_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
