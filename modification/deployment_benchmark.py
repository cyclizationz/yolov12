#!/usr/bin/env python3
"""
Deployment Benchmarking Script

Comprehensive benchmarking of all model variants (baseline and optimal) in different formats:
- PyTorch (.pt)
- ONNX (.onnx)
- OpenVINO INT8 (CPU)
- TensorRT FP16/INT8 (GPU)

Measures: latency, throughput (FPS), memory usage, accuracy (mAP)
"""

import sys
import time
import gc
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from ultralytics import YOLO

def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if model_path and Path(model_path).exists():
        return Path(model_path).stat().st_size / (1024 * 1024)
    return 0.0

def benchmark_pytorch_model(model_path, test_data='coco128-seg.yaml', imgsz=640, 
                           warmup=10, iterations=100, device='cuda'):
    """Benchmark PyTorch model."""
    print(f"  Loading PyTorch model: {model_path}")
    model = YOLO(str(model_path))
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    for _ in range(warmup):
        _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark latency
    print(f"  Benchmarking latency ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    # Memory usage
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        max_memory = 0.0
    
    # Accuracy (mAP)
    print(f"  Evaluating accuracy on {test_data}...")
    try:
        metrics = model.val(data=test_data, imgsz=imgsz, verbose=False)
        map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
        map50_95 = metrics.seg.map50_95 if hasattr(metrics.seg, 'map50_95') else 0.0
    except Exception as e:
        print(f"  Warning: Accuracy evaluation failed: {e}")
        map50 = 0.0
        map50_95 = 0.0
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'fps': fps,
        'max_memory_mb': max_memory,
        'map50': map50,
        'map50_95': map50_95
    }

def benchmark_onnx_model(model_path, test_data='coco128-seg.yaml', imgsz=640,
                         warmup=10, iterations=100, device='cuda'):
    """Benchmark ONNX model using ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ✗ ONNX Runtime not available. Skipping ONNX benchmark.")
        return None
    
    print(f"  Loading ONNX model: {model_path}")
    
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"  ✗ Failed to load ONNX model: {e}")
        return None
    
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    print(f"  Benchmarking latency ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    # Memory (approximate)
    max_memory = 0.0  # ONNX Runtime doesn't expose memory stats easily
    
    # Accuracy - use YOLO wrapper
    print(f"  Evaluating accuracy on {test_data}...")
    try:
        model = YOLO(str(model_path))
        metrics = model.val(data=test_data, imgsz=imgsz, verbose=False)
        map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
        map50_95 = metrics.seg.map50_95 if hasattr(metrics.seg, 'map50_95') else 0.0
    except Exception as e:
        print(f"  Warning: Accuracy evaluation failed: {e}")
        map50 = 0.0
        map50_95 = 0.0
    
    del session
    gc.collect()
    
    return {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'fps': fps,
        'max_memory_mb': max_memory,
        'map50': map50,
        'map50_95': map50_95
    }

def benchmark_openvino_model(model_path, test_data='coco128-seg.yaml', imgsz=640,
                            warmup=10, iterations=100):
    """Benchmark OpenVINO model (CPU only)."""
    try:
        import openvino as ov
    except ImportError:
        print("  ✗ OpenVINO not available. Skipping OpenVINO benchmark.")
        return None
    
    print(f"  Loading OpenVINO model: {model_path}")
    
    try:
        core = ov.Core()
        model = core.read_model(str(model_path))
        compiled_model = core.compile_model(model, "CPU")
    except Exception as e:
        print(f"  ✗ Failed to load OpenVINO model: {e}")
        return None
    
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = compiled_model([dummy_input])
    
    # Benchmark
    print(f"  Benchmarking latency ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = compiled_model([dummy_input])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    # Memory (approximate)
    max_memory = 0.0
    
    # Accuracy - use YOLO wrapper
    print(f"  Evaluating accuracy on {test_data}...")
    try:
        yolo_model = YOLO(str(model_path))
        metrics = yolo_model.val(data=test_data, imgsz=imgsz, verbose=False)
        map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
        map50_95 = metrics.seg.map50_95 if hasattr(metrics.seg, 'map50_95') else 0.0
    except Exception as e:
        print(f"  Warning: Accuracy evaluation failed: {e}")
        map50 = 0.0
        map50_95 = 0.0
    
    del compiled_model, model
    gc.collect()
    
    return {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'fps': fps,
        'max_memory_mb': max_memory,
        'map50': map50,
        'map50_95': map50_95
    }

def benchmark_tensorrt_model(model_path, test_data='coco128-seg.yaml', imgsz=640,
                             warmup=10, iterations=100):
    """Benchmark TensorRT model (GPU only)."""
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available. Skipping TensorRT benchmark.")
        return None
    
    print(f"  Loading TensorRT model: {model_path}")
    
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"  ✗ Failed to load TensorRT model: {e}")
        return None
    
    dummy_input = torch.randn(1, 3, imgsz, imgsz).cuda()
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
    
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking latency ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    
    # Accuracy
    print(f"  Evaluating accuracy on {test_data}...")
    try:
        metrics = model.val(data=test_data, imgsz=imgsz, verbose=False)
        map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
        map50_95 = metrics.seg.map50_95 if hasattr(metrics.seg, 'map50_95') else 0.0
    except Exception as e:
        print(f"  Warning: Accuracy evaluation failed: {e}")
        map50 = 0.0
        map50_95 = 0.0
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'fps': fps,
        'max_memory_mb': max_memory,
        'map50': map50,
        'map50_95': map50_95
    }

def run_deployment_benchmark():
    """Run comprehensive deployment benchmark."""
    print("=" * 80)
    print("DEPLOYMENT BENCHMARKING")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    baseline_pt = base_dir.parent / 'model' / 'yolov12n-seg.pt'
    optimal_pt = base_dir / 'yolov12n-seg-ppc5x5.pt'
    
    # Load export results
    export_results_file = base_dir / 'deployment_models' / 'export_results.json'
    if export_results_file.exists():
        with open(export_results_file, 'r') as f:
            export_results = json.load(f)
    else:
        print("⚠ Warning: export_results.json not found. Using default paths.")
        export_results = {
            'baseline': {
                'onnx': None,
                'openvino_int8': None,
                'tensorrt_fp16': None,
                'tensorrt_int8': None
            },
            'optimal': {
                'onnx': None,
                'openvino_int8': None,
                'tensorrt_fp16': None,
                'tensorrt_int8': None
            }
        }
    
    test_data = 'coco128-seg.yaml'
    imgsz = 640
    warmup = 10
    iterations = 100
    
    results = []
    
    # Benchmark configurations
    benchmarks = [
        # PyTorch models
        ('baseline', 'pytorch', baseline_pt, benchmark_pytorch_model, 'cuda'),
        ('optimal', 'pytorch', optimal_pt, benchmark_pytorch_model, 'cuda'),
        
        # ONNX models
        ('baseline', 'onnx', export_results['baseline'].get('onnx'), benchmark_onnx_model, 'cuda'),
        ('optimal', 'onnx', export_results['optimal'].get('onnx'), benchmark_onnx_model, 'cuda'),
        
        # OpenVINO INT8 (CPU)
        ('baseline', 'openvino_int8', export_results['baseline'].get('openvino_int8'), benchmark_openvino_model, 'cpu'),
        ('optimal', 'openvino_int8', export_results['optimal'].get('openvino_int8'), benchmark_openvino_model, 'cpu'),
        
        # TensorRT FP16 (GPU)
        ('baseline', 'tensorrt_fp16', export_results['baseline'].get('tensorrt_fp16'), benchmark_tensorrt_model, 'cuda'),
        ('optimal', 'tensorrt_fp16', export_results['optimal'].get('tensorrt_fp16'), benchmark_tensorrt_model, 'cuda'),
        
        # TensorRT INT8 (GPU)
        ('baseline', 'tensorrt_int8', export_results['baseline'].get('tensorrt_int8'), benchmark_tensorrt_model, 'cuda'),
        ('optimal', 'tensorrt_int8', export_results['optimal'].get('tensorrt_int8'), benchmark_tensorrt_model, 'cuda'),
    ]
    
    for model_name, format_name, model_path, benchmark_func, device in benchmarks:
        if model_path is None or not Path(model_path).exists():
            print(f"\n⚠ Skipping {model_name} {format_name}: model not found")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Benchmarking: {model_name.upper()} - {format_name.upper()}")
        print("=" * 80)
        
        try:
            if format_name == 'openvino_int8':
                result = benchmark_func(model_path, test_data, imgsz, warmup, iterations)
            else:
                result = benchmark_func(model_path, test_data, imgsz, warmup, iterations, device)
            
            if result:
                result['model_name'] = model_name
                result['format'] = format_name
                result['device'] = device
                result['model_path'] = str(model_path)
                result['model_size_mb'] = get_model_size_mb(model_path)
                results.append(result)
                
                print(f"\n  Results:")
                print(f"    Latency: {result['avg_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")
                print(f"    FPS: {result['fps']:.2f}")
                print(f"    Memory: {result['max_memory_mb']:.2f} MB")
                print(f"    mAP50: {result['map50']:.4f}")
                print(f"    mAP50-95: {result['map50_95']:.4f}")
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_dir = base_dir / 'outputs'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results_file = output_dir / 'deployment_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_file = output_dir / 'deployment_benchmark_results.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Results saved to CSV: {csv_file}")
    except ImportError:
        print("\n⚠ pandas not available. Skipping CSV export.")
    
    print(f"\n✓ Results saved to JSON: {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['model_name'].upper()} - {result['format'].upper()} ({result['device'].upper()}):")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Latency: {result['avg_latency_ms']:.2f} ms")
        print(f"  mAP50-95: {result['map50_95']:.4f}")
    
    return results

if __name__ == '__main__':
    try:
        results = run_deployment_benchmark()
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

