import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# Disable FlashAttention to avoid potential CUDA issues on some environments
os.environ['DISABLE_FLASH_ATTN'] = '1'

def main():
    repo_root = Path(__file__).resolve().parent
    run_name = "yolov12n_racing_e300_split1_p0"

    # Base model weights (keep consistent with other training scripts)
    base_pt = repo_root / "modification" / "model" / "yolov12n-seg.pt"
    if not base_pt.exists():
        # Fallback to legacy path if present
        base_pt = repo_root / "model" / "yolov12n-seg.pt"
    if not base_pt.exists():
        raise FileNotFoundError(f"Missing base weights: {base_pt}")

    print(f"Loading base weights: {base_pt}")
    model = YOLO(str(base_pt))

    # Train the model
    print("Starting training...")
    model.train(
        data=str(repo_root / "racing_split.yaml"),
        epochs=300,
        imgsz=640,
        batch=16,
        patience=0,  # disable early stopping to guarantee full epoch budget
        project=str(repo_root / "runs" / "segment"),
        name=run_name,
        exist_ok=True,
    )
    print("Training completed.")

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

    # Copy to deployment/ for offline pipeline usage
    deploy_onnx = repo_root / "deployment" / f"{run_name}.onnx"
    deploy_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(onnx_path), str(deploy_onnx))
    print(f"Saved ONNX to: {deploy_onnx}")

if __name__ == '__main__':
    main()

