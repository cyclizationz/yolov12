#!/usr/bin/env python3
"""
Deployment Export Script

Exports baseline and optimal models to optimized formats:
- ONNX (for cross-platform deployment)
- OpenVINO INT8 (for CPU optimization)
- TensorRT FP16/INT8 (for GPU optimization)
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

def export_models():
    """Export baseline and optimal models to various formats."""
    print("=" * 80)
    print("DEPLOYMENT MODEL EXPORT")
    print("=" * 80)
    
    # Paths
    base_dir = Path(__file__).parent
    baseline_path = base_dir.parent / 'model' / 'yolov12n-seg.pt'
    optimal_path = base_dir / 'yolov12n-seg-ppc5x5.pt'
    output_dir = base_dir / 'deployment_models'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calibration dataset (for INT8 quantization)
    calibration_data = 'coco128-seg.yaml'
    
    models_to_export = [
        ('baseline', baseline_path),
        ('optimal', optimal_path)
    ]
    
    export_results = {}
    
    for model_name, model_path in models_to_export:
        if not model_path.exists():
            print(f"\n⚠ Warning: {model_name} model not found at {model_path}")
            print("Skipping exports for this model.")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Processing {model_name.upper()} model: {model_path.name}")
        print("=" * 80)
        
        model = YOLO(str(model_path))
        model_results = {}
        
        # 1. Export to ONNX
        print(f"\n[1/4] Exporting {model_name} to ONNX...")
        try:
            onnx_path = model.export(
                format='onnx',
                simplify=True,
                opset=17,
                imgsz=640,
                dynamic=False
            )
            model_results['onnx'] = str(onnx_path)
            print(f"✓ ONNX export successful: {onnx_path}")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            model_results['onnx'] = None
        
        # 2. Export to OpenVINO INT8 (CPU optimization)
        print(f"\n[2/4] Exporting {model_name} to OpenVINO INT8...")
        try:
            openvino_path = model.export(
                format='openvino',
                int8=True,
                data=calibration_data,
                imgsz=640,
                dynamic=False
            )
            model_results['openvino_int8'] = str(openvino_path)
            print(f"✓ OpenVINO INT8 export successful: {openvino_path}")
        except Exception as e:
            print(f"✗ OpenVINO INT8 export failed: {e}")
            print("  Note: This requires openvino>=2024.5.0 and nncf>=2.14.0")
            model_results['openvino_int8'] = None
        
        # 3. Export to TensorRT FP16 (GPU optimization)
        print(f"\n[3/4] Exporting {model_name} to TensorRT FP16...")
        try:
            trt_fp16_path = model.export(
                format='engine',
                half=True,
                imgsz=640,
                workspace=4  # 4 GB workspace
            )
            model_results['tensorrt_fp16'] = str(trt_fp16_path)
            print(f"✓ TensorRT FP16 export successful: {trt_fp16_path}")
        except Exception as e:
            print(f"✗ TensorRT FP16 export failed: {e}")
            print("  Note: This requires TensorRT and CUDA")
            model_results['tensorrt_fp16'] = None
        
        # 4. Export to TensorRT INT8 (GPU optimization)
        print(f"\n[4/4] Exporting {model_name} to TensorRT INT8...")
        try:
            trt_int8_path = model.export(
                format='engine',
                int8=True,
                data=calibration_data,
                imgsz=640,
                workspace=4  # 4 GB workspace
            )
            model_results['tensorrt_int8'] = str(trt_int8_path)
            print(f"✓ TensorRT INT8 export successful: {trt_int8_path}")
        except Exception as e:
            print(f"✗ TensorRT INT8 export failed: {e}")
            print("  Note: This requires TensorRT, CUDA, and calibration data")
            model_results['tensorrt_int8'] = None
        
        export_results[model_name] = model_results
    
    # Save export results to JSON
    results_file = output_dir / 'export_results.json'
    with open(results_file, 'w') as f:
        json.dump(export_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    
    for model_name, results in export_results.items():
        print(f"\n{model_name.upper()}:")
        for format_name, path in results.items():
            status = "✓" if path else "✗"
            print(f"  {status} {format_name}: {path if path else 'Failed'}")
    
    print(f"\n✓ Export results saved to: {results_file}")
    
    return export_results

if __name__ == '__main__':
    try:
        results = export_models()
        print("\n" + "=" * 80)
        print("EXPORT COMPLETE")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

