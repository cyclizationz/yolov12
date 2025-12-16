import os
import cv2
import numpy as np
import glob
from pathlib import Path
import time
from ultralytics import YOLO

def run_inference_onnx(model_path, image_dir, output_dir, conf_thres=0.25):
    print(f"Loading ONNX model from {model_path} using Ultralytics YOLO...")
    model = YOLO(model_path, task='segment')
    
    # Get images
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting inference...")
    start_time = time.time()
    
    # Since static shape export expects batch size of 1 (or whatever it was exported with),
    # and Autobackend/ONNXRuntime complains when passing a list (which Ultralytics tries to batch),
    # we will iterate one by one or let Ultralytics handle stream with batch=1.
    
    # The error `index: 0 Got: 922 Expected: 1` means it tried to pass 922 images as a batch but the model expects batch size 1 (static).
    # We can force batch=1 in predict.
    
    for i, img_path in enumerate(image_paths):
        results = model.predict(
            source=img_path, 
            imgsz=640, 
            conf=conf_thres,
            save=True,
            project=str(Path(output_dir).parent),
            name=Path(output_dir).name,
            exist_ok=True,
            verbose=False
        )
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(image_paths)} images...")
            
    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    model_path = "runs/segment/yolov12n_racing/weights/best.onnx"
    image_dir = "../datasets/racing/images_900"
    output_dir = "runs/segment/predict_racing_onnx"
    
    run_inference_onnx(model_path, image_dir, output_dir)
