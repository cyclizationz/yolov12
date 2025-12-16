
import sys
import torch
import pandas as pd
import time
import numpy as np
import gc
from ultralytics import YOLO
from pe_ablation_utils import replace_aattn_modules

# Configuration
MODEL_PATH = '/home/tiehangz/proj/yolov12/modification/model/yolov12n-seg.pt'
DATA_YAML = '/home/tiehangz/proj/yolov12/modification/coco_1k.yaml' # 1000 images
KERNEL_SIZES = [3, 5, 7, 9, 0] # 0 = NoPE
OUTPUT_FILE = 'modification/outputs/pe_ablation_results_new.csv'

def benchmark(model, device='cuda'):
    # Latency
    dummy = torch.randn(1, 3, 640, 640, device=device)
    # Warmup
    for _ in range(10): model(dummy, verbose=False)
    
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        if device=='cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return 1.0 / np.mean(times), np.mean(times) * 1000

def main():
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for k in KERNEL_SIZES:
        print(f"\n--- Testing PPC Kernel Size: {k} (0=NoPE) ---")
        
        # Reload model fresh
        gc.collect()
        torch.cuda.empty_cache()
        model = YOLO(MODEL_PATH)
        
        # Replace modules
        n = replace_aattn_modules(model, kernel_size=k)
        if n == 0:
            print("WARNING: No modules replaced. Skipping.")
            continue
            
        # Benchmark Speed
        fps, lat = benchmark(model, device)
        print(f"FPS: {fps:.2f}, Latency: {lat:.2f}ms")
        
        # Benchmark Accuracy on 1k subset
        try:
            # batch=4, cache=False
            metrics = model.val(data=DATA_YAML, imgsz=640, batch=4, device=device, verbose=False, plots=False, cache=False)
            map50 = metrics.seg.map50
            map5095 = metrics.seg.map
        except Exception as e:
            print(f"Val failed: {e}")
            map50, map5095 = 0, 0
            
        print(f"mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}")
        
        results.append({
            'variant': f'PPC {k}x{k}' if k > 0 else 'NoPE',
            'kernel_size': k,
            'fps': fps,
            'latency_ms': lat,
            'map50': map50,
            'map50_95': map5095
        })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(df)

if __name__ == "__main__":
    main()
