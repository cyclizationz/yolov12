import os
import torch
from ultralytics import YOLO

# Disable FlashAttention to avoid potential CUDA issues on some environments
os.environ['DISABLE_FLASH_ATTN'] = '1'

def main():
    # Load the model
    print("Loading YOLOv12n-seg model...")
    model = YOLO('model/yolov12n-seg.pt')

    # Train the model
    print("Starting training...")
    results = model.train(
        data='racing.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='runs/segment',
        name='yolov12n_racing',
        exist_ok=True
    )
    print("Training completed.")

    # Export the model to ONNX
    print("Exporting model to ONNX...")
    success = model.export(format='onnx', simplify=True, dynamic=True)
    print(f"Export success: {success}")

if __name__ == '__main__':
    main()

