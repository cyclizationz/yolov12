import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# Disable FlashAttention to avoid potential CUDA issues on some environments
os.environ["DISABLE_FLASH_ATTN"] = "1"


def main():
    repo_root = Path(__file__).resolve().parent
    run_name = "yolov12n_fc5_seg_v1_e300_p0"

    # Base model
    # In this repo, the pinned weights live under modification/model/
    base_pt = repo_root / "modification" / "model" / "yolov12n-seg.pt"
    if not base_pt.exists():
        raise FileNotFoundError(f"Missing base weights: {base_pt}")

    # Train
    model = YOLO(str(base_pt))
    model.train(
        data=str(repo_root / "fc5.yaml"),
        epochs=300,
        imgsz=640,
        batch=16,
        patience=0,  # disable early stopping to guarantee full epoch budget
        project=str(repo_root / "runs" / "segment"),
        name=run_name,
        exist_ok=True,
    )

    # Export best to ONNX
    best_pt = repo_root / "runs" / "segment" / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_pt}")

    best = YOLO(str(best_pt))
    onnx_path = best.export(
        format="onnx",
        simplify=True,
        opset=18,
        imgsz=640,
        dynamic=False,
    )

    # Copy to deployment/ for C++ tool default usage
    deploy_onnx = repo_root / "deployment" / f"{run_name}.onnx"
    deploy_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(onnx_path), str(deploy_onnx))
    print(f"Saved ONNX to: {deploy_onnx}")


if __name__ == "__main__":
    main()


