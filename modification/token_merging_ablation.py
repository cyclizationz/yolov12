#!/usr/bin/env python3
"""
Token Merging (ToMe) Ablation Study for YOLOv12 Segmentation

Implements Token Merging from "Token Merging: Your ViT But Faster" (https://arxiv.org/abs/2210.09461)
for YOLOv12 segmentation models.

Compares:
- Baseline (PPC 7×7, no ToMe)
- Optimal (PPC 5×5, no ToMe)  
- Optimal + ToMe with different merge ratios
"""

import sys
import os
import glob
import time
import gc
import json
from pathlib import Path

# Add mamba environment to Python path
mamba_env_path = '/home/tiehangz/micromamba/envs/yolov12'
if os.path.exists(mamba_env_path):
    python_lib_pattern = f'{mamba_env_path}/lib/python*/site-packages'
    python_lib_dirs = glob.glob(python_lib_pattern)
    if python_lib_dirs:
        mamba_site_packages = python_lib_dirs[0]
        if mamba_site_packages not in sys.path:
            sys.path.insert(0, mamba_site_packages)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from ultralytics import YOLO
from ultralytics.nn.modules.block import (
    Attention,
    PSABlock, PSA, C2PSA, C2fPSA
)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 11
sns.set_palette("husl")

# Output directory
output_dir = Path('/home/tiehangz/proj/yolov12/modification/outputs')
output_dir.mkdir(exist_ok=True, parents=True)


class TokenMergingWrapper(nn.Module):
    """
    Wrapper to apply Token Merging to attention modules.
    
    Simplified implementation: reduces spatial resolution by merging tokens
    before attention computation, then upsamples back.
    """
    def __init__(self, attention_module, merge_ratio=0.25):
        super().__init__()
        self.attention = attention_module
        self.merge_ratio = merge_ratio
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.merge_ratio > 0 and self.training == False:
            # Apply spatial downsampling (simplified token merging)
            scale_factor = 1.0 - self.merge_ratio
            H_new = max(1, int(H * scale_factor))
            W_new = max(1, int(W * scale_factor))
            
            # Downsample
            x_down = torch.nn.functional.interpolate(
                x, size=(H_new, W_new), mode='bilinear', align_corners=False
            )
            
            # Apply attention on downsampled features
            x_attn = self.attention(x_down)
            
            # Upsample back to original size
            x_out = torch.nn.functional.interpolate(
                x_attn, size=(H, W), mode='bilinear', align_corners=False
            )
            
            return x_out
        else:
            # No merging
            return self.attention(x)


def apply_tome_to_model(model, merge_ratio=0.25):
    """
    Apply Token Merging to all attention modules in a model.
    
    Args:
        model: YOLO model
        merge_ratio: Ratio of tokens to merge (0.0-1.0)
    """
    replaced_count = 0
    
    def _replace_module(module, name=''):
        nonlocal replaced_count
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is a base Attention module
            if isinstance(child_module, Attention):
                # Wrap with Token Merging
                wrapped = TokenMergingWrapper(child_module, merge_ratio=merge_ratio)
                setattr(module, child_name, wrapped)
                replaced_count += 1
                print(f"  Applied ToMe (merge_ratio={merge_ratio}) to {full_name}")
            
            # Recursively check nested modules
            elif isinstance(child_module, (PSABlock, PSA, C2PSA, C2fPSA, nn.Sequential, nn.ModuleList)):
                _replace_module(child_module, full_name)
            else:
                if len(list(child_module.children())) > 0:
                    _replace_module(child_module, full_name)
    
    if hasattr(model, 'model'):
        _replace_module(model.model)
    else:
        _replace_module(model)
    
    return replaced_count


def benchmark_model(model, test_data='coco128-seg.yaml', imgsz=640, 
                   warmup=10, iterations=100, device='cuda'):
    """Benchmark model performance."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark latency
    latencies = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.predict(dummy_input, verbose=False, imgsz=imgsz)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    # Memory usage
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
    else:
        max_memory = 0.0
    
    # Accuracy
    try:
        metrics = model.val(data=test_data, imgsz=imgsz, verbose=False)
        # Extract mAP50
        map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
        
        # Try multiple ways to get mAP50-95
        map50_95 = None
        if hasattr(metrics.seg, 'map'):
            map50_95 = metrics.seg.map
        elif hasattr(metrics.seg, 'maps') and len(metrics.seg.maps) > 0:
            # maps is a list, take the mean if available
            map50_95 = np.mean(metrics.seg.maps) if isinstance(metrics.seg.maps, (list, np.ndarray)) else None
        elif hasattr(metrics, 'results_dict'):
            # Try accessing from results_dict
            results_dict = metrics.results_dict
            if isinstance(results_dict, dict):
                # Look for mAP50-95(M) key
                for key in ['metrics/mAP50-95(M)', 'mAP50-95(M)', 'map50_95']:
                    if key in results_dict:
                        map50_95 = results_dict[key]
                        break
        
        # If still None, try to compute from all_ap
        if map50_95 is None and hasattr(metrics.seg, 'all_ap'):
            all_ap = metrics.seg.all_ap
            if isinstance(all_ap, np.ndarray) and all_ap.size > 0:
                # all_ap shape is (nc, 10) where 10 is IoU thresholds from 0.5 to 0.95
                map50_95 = all_ap.mean()  # Mean across all classes and IoU thresholds
        
        # Default to 0 if still None
        if map50_95 is None:
            map50_95 = 0.0
            print(f"  ⚠️  Warning: Could not extract mAP50-95, using 0.0")
            if hasattr(metrics.seg, '__dict__'):
                print(f"     Available seg attributes: {[attr for attr in dir(metrics.seg) if not attr.startswith('_')]}")
        
        print(f"  Accuracy: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
    except Exception as e:
        print(f"  Warning: Accuracy evaluation failed: {e}")
        import traceback
        traceback.print_exc()
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


def main():
    """Main function to run Token Merging ablation study."""
    # Configuration
    baseline_path = Path('/home/tiehangz/proj/yolov12/model/yolov12n-seg.pt')
    optimal_path = Path('/home/tiehangz/proj/yolov12/modification/yolov12n-seg-ppc5x5.pt')
    test_data = 'coco.yaml'
    imgsz = 640
    warmup = 10
    iterations = 100
    
    # Token merging ratios to test
    MERGE_RATIOS = [0.0, 0.1, 0.2, 0.25, 0.3]
    
    print("=" * 80)
    print("TOKEN MERGING ABLATION STUDY CONFIGURATION")
    print("=" * 80)
    print(f"Baseline model: {baseline_path}")
    print(f"Optimal model (PPC 5×5): {optimal_path}")
    print(f"Test data: {test_data}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Merge ratios to test: {MERGE_RATIOS}")
    print("=" * 80)
    
    # Storage for results
    results = []
    
    print("\n" + "=" * 80)
    print("TOKEN MERGING ABLATION STUDY")
    print("=" * 80)
    
    # 1. Baseline (PPC 7×7, no ToMe)
    print("\n" + "=" * 80)
    print("Testing BASELINE (PPC 7×7, no ToMe)")
    print("=" * 80)
    if baseline_path.exists():
        model = YOLO(str(baseline_path))
        bench_results = benchmark_model(model, test_data, imgsz, warmup, iterations, 'cuda')
        results.append({
            'model': 'baseline',
            'variant': 'PPC 7×7',
            'merge_ratio': 0.0,
            'model_path': str(baseline_path),
            **bench_results
        })
        print(f"  Results: FPS={bench_results['fps']:.2f}, Latency={bench_results['avg_latency_ms']:.2f}ms, mAP50-95={bench_results['map50_95']:.4f}")
    else:
        print(f"  ⚠ Baseline model not found at {baseline_path}")
    
    # 2. Optimal (PPC 5×5, no ToMe)
    print("\n" + "=" * 80)
    print("Testing OPTIMAL (PPC 5×5, no ToMe)")
    print("=" * 80)
    if optimal_path.exists():
        model = YOLO(str(optimal_path))
        bench_results = benchmark_model(model, test_data, imgsz, warmup, iterations, 'cuda')
        results.append({
            'model': 'optimal',
            'variant': 'PPC 5×5',
            'merge_ratio': 0.0,
            'model_path': str(optimal_path),
            **bench_results
        })
        print(f"  Results: FPS={bench_results['fps']:.2f}, Latency={bench_results['avg_latency_ms']:.2f}ms, mAP50-95={bench_results['map50_95']:.4f}")
    else:
        print(f"  ⚠ Optimal model not found at {optimal_path}")
    
    # 3. Optimal with different ToMe merge ratios
    if optimal_path.exists():
        for merge_ratio in MERGE_RATIOS:
            if merge_ratio == 0.0:
                continue  # Already tested above
            
            print("\n" + "=" * 80)
            print(f"Testing OPTIMAL + ToMe (merge_ratio={merge_ratio})")
            print("=" * 80)
            
            # Load model
            model = YOLO(str(optimal_path))
            
            # Apply Token Merging
            replaced_count = apply_tome_to_model(model, merge_ratio=merge_ratio)
            print(f"  Applied ToMe to {replaced_count} attention modules")
            
            # Benchmark
            bench_results = benchmark_model(model, test_data, imgsz, warmup, iterations, 'cuda')
            results.append({
                'model': 'optimal+tome',
                'variant': f'PPC 5×5 + ToMe({merge_ratio})',
                'merge_ratio': merge_ratio,
                'model_path': str(optimal_path),
                **bench_results
            })
            print(f"  Results: FPS={bench_results['fps']:.2f}, Latency={bench_results['avg_latency_ms']:.2f}ms, mAP50-95={bench_results['map50_95']:.4f}")
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    
    # Save results
    df_results = pd.DataFrame(results)
    
    # Ensure map50_95 is numeric and handle any potential issues
    if 'map50_95' in df_results.columns:
        df_results['map50_95'] = pd.to_numeric(df_results['map50_95'], errors='coerce').fillna(0.0)
        print(f"\n✓ Verified map50_95 values: {df_results['map50_95'].tolist()}")
    
    csv_file = output_dir / 'token_merging_ablation_results.csv'
    df_results.to_csv(csv_file, index=False)
    
    json_file = output_dir / 'token_merging_ablation_results.json'
    # Ensure all map50_95 values are floats in the results list
    for result in results:
        if 'map50_95' in result:
            result['map50_95'] = float(result['map50_95']) if result['map50_95'] is not None else 0.0
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {csv_file}")
    print(f"✓ Results saved to: {json_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df_results[['model', 'variant', 'merge_ratio', 'fps', 'avg_latency_ms', 'map50_95']].to_string(index=False))
    
    # Create visualization
    if len(df_results) > 0:
        # Verify data before visualization
        if 'map50_95' in df_results.columns:
            print(f"\n✓ Visualization data - map50_95 values: {df_results['map50_95'].tolist()}")
            print(f"✓ Visualization data - map50_95 min: {df_results['map50_95'].min()}, max: {df_results['map50_95'].max()}")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 1. FPS Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(df_results)), df_results['fps'], 
                        color=colors[:len(df_results)], alpha=0.8)
        ax1.set_xticks(range(len(df_results)))
        ax1.set_xticklabels(df_results['variant'], rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('FPS', fontsize=10)
        ax1.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
        ax1.axhline(y=30, color='r', linestyle='--', linewidth=2, label='30 FPS Target', alpha=0.7)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars1, df_results['fps']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Latency Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(df_results)), df_results['avg_latency_ms'],
                        color=colors[:len(df_results)], alpha=0.8)
        ax2.set_xticks(range(len(df_results)))
        ax2.set_xticklabels(df_results['variant'], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Latency (ms)', fontsize=10)
        ax2.set_title('Latency Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, df_results['avg_latency_ms']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Accuracy Comparison (both mAP50 and mAP50-95)
        ax3 = axes[1, 0]
        x = np.arange(len(df_results))
        width = 0.35
        
        # Ensure map50 and map50_95 are numeric arrays
        map50_values = pd.to_numeric(df_results['map50'], errors='coerce').fillna(0.0).values
        map50_95_values = pd.to_numeric(df_results['map50_95'], errors='coerce').fillna(0.0).values
        
        bars3a = ax3.bar(x - width/2, map50_values, width,
                         label='mAP50', color='#1f77b4', alpha=0.8)
        bars3b = ax3.bar(x + width/2, map50_95_values, width,
                         label='mAP50-95', color='#ff7f0e', alpha=0.8)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_results['variant'], rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('mAP', fontsize=10)
        ax3.set_title('Accuracy Comparison (mAP50 & mAP50-95)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars3a, bars3b]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Trade-off Plot (both mAP50 and mAP50-95)
        ax4 = axes[1, 1]
        colors_map = {'baseline': '#1f77b4', 'optimal': '#ff7f0e', 'optimal+tome': '#2ca02c'}
        
        # Plot mAP50
        for model_type in df_results['model'].unique():
            model_data = df_results[df_results['model'] == model_type]
            ax4.scatter(model_data['fps'], model_data['map50'],
                       label=f'{model_type.replace("+", " + ").title()} (mAP50)',
                       alpha=0.7, s=150, marker='o',
                       color=colors_map.get(model_type, '#808080'),
                       edgecolors='black', linewidths=1.5)
        
        # Plot mAP50-95
        for model_type in df_results['model'].unique():
            model_data = df_results[df_results['model'] == model_type]
            map50_95_vals = pd.to_numeric(model_data['map50_95'], errors='coerce').fillna(0.0).values
            fps_vals = pd.to_numeric(model_data['fps'], errors='coerce').fillna(0.0).values
            ax4.scatter(fps_vals, map50_95_vals,
                       label=f'{model_type.replace("+", " + ").title()} (mAP50-95)',
                       alpha=0.7, s=150, marker='^',
                       color=colors_map.get(model_type, '#808080'),
                       edgecolors='black', linewidths=1.5)
        
        ax4.set_xlabel('FPS (Throughput)', fontsize=10)
        ax4.set_ylabel('mAP (Accuracy)', fontsize=10)
        ax4.set_title('Accuracy vs Speed Trade-off (mAP50 & mAP50-95)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8, ncol=2, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=30, color='r', linestyle='--', linewidth=2, alpha=0.7, label='30 FPS Target')
        
        plt.suptitle('Token Merging (ToMe) Ablation Study: Baseline vs Optimal vs Optimal+ToMe', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        viz_file = output_dir / 'token_merging_ablation_comparison.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Visualization saved to: {viz_file}")
        plt.close()


if __name__ == '__main__':
    main()

