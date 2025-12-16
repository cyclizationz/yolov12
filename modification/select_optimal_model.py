#!/usr/bin/env python3
"""
Model Selection Analysis Script

Analyzes ablation study results to identify the optimal model variant for deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_ablation_results(csv_path='outputs/comprehensive_comprehensive_pe_ablation_results.csv'):
    """
    Analyze ablation results and identify optimal model variant.
    
    Args:
        csv_path: Path to comprehensive ablation results CSV
        
    Returns:
        dict: Analysis results with optimal model info
    """
    csv_file = Path(__file__).parent / csv_path
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print("COMPREHENSIVE ABLATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Filter GPU results (most relevant for deployment)
    gpu_results = df[df['device'] == 'cuda'].copy()
    
    if len(gpu_results) == 0:
        print("Warning: No GPU results found. Using all results.")
        gpu_results = df.copy()
    
    print("\nAll Variants (sorted by FPS):")
    display_cols = ['variant_label', 'fps', 'avg_latency_ms', 'map50', 'map50_95', 'max_memory_mb']
    available_cols = [col for col in display_cols if col in gpu_results.columns]
    print(gpu_results[available_cols].sort_values('fps', ascending=False).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OPTIMAL MODEL SELECTION")
    print("=" * 80)
    
    # Find baseline
    baseline = gpu_results[gpu_results['variant_type'] == 'baseline']
    if len(baseline) == 0:
        # Fallback: find PPC 7x7
        baseline = gpu_results[gpu_results['variant_value'] == 'ppc_7x7']
    
    if len(baseline) == 0:
        # Use first row as baseline
        baseline = gpu_results.iloc[[0]]
        print("Warning: Baseline not found, using first variant as baseline")
    
    baseline = baseline.iloc[0]
    
    # Find best variant (highest FPS while maintaining accuracy)
    # Prioritize FPS, then latency, then check accuracy is maintained
    best_idx = gpu_results['fps'].idxmax()
    best = gpu_results.loc[best_idx]
    
    print(f"\nBaseline Model: {baseline['variant_label']}")
    print(f"  FPS: {baseline['fps']:.2f}")
    print(f"  Latency: {baseline['avg_latency_ms']:.2f} ms")
    print(f"  mAP50: {baseline['map50']:.4f}")
    print(f"  mAP50-95: {baseline['map50_95']:.4f}")
    print(f"  Memory: {baseline['max_memory_mb']:.2f} MB")
    
    print(f"\nOptimal Variant: {best['variant_label']}")
    print(f"  Variant Type: {best['variant_type']}")
    print(f"  Variant Value: {best['variant_value']}")
    print(f"  FPS: {best['fps']:.2f}")
    print(f"  Latency: {best['avg_latency_ms']:.2f} ms")
    print(f"  mAP50: {best['map50']:.4f}")
    print(f"  mAP50-95: {best['map50_95']:.4f}")
    print(f"  Memory: {best['max_memory_mb']:.2f} MB")
    
    print(f"\nImprovement over Baseline:")
    fps_improvement = best['fps'] - baseline['fps']
    fps_improvement_pct = ((best['fps'] / baseline['fps'] - 1) * 100)
    latency_improvement = baseline['avg_latency_ms'] - best['avg_latency_ms']
    latency_improvement_pct = ((1 - best['avg_latency_ms'] / baseline['avg_latency_ms']) * 100)
    
    print(f"  FPS: +{fps_improvement:.2f} ({fps_improvement_pct:+.1f}%)")
    print(f"  Latency: {latency_improvement:.2f} ms faster ({latency_improvement_pct:+.1f}%)")
    
    # Check accuracy preservation
    map50_diff = best['map50'] - baseline['map50']
    map50_95_diff = best['map50_95'] - baseline['map50_95']
    
    print(f"  mAP50: {map50_diff:+.4f} ({'✓ Maintained' if abs(map50_diff) < 0.001 else '⚠ Changed'})")
    print(f"  mAP50-95: {map50_95_diff:+.4f} ({'✓ Maintained' if abs(map50_95_diff) < 0.001 else '⚠ Changed'})")
    
    # Save selection to JSON
    selection = {
        'baseline': {
            'variant_label': baseline['variant_label'],
            'variant_type': baseline.get('variant_type', 'baseline'),
            'variant_value': baseline.get('variant_value', 'ppc_7x7'),
            'model_path': baseline['model_path'],
            'fps': float(baseline['fps']),
            'latency_ms': float(baseline['avg_latency_ms']),
            'map50': float(baseline['map50']),
            'map50_95': float(baseline['map50_95']),
            'memory_mb': float(baseline['max_memory_mb'])
        },
        'optimal': {
            'variant_label': best['variant_label'],
            'variant_type': best['variant_type'],
            'variant_value': best['variant_value'],
            'model_path': best['model_path'],
            'fps': float(best['fps']),
            'latency_ms': float(best['avg_latency_ms']),
            'map50': float(best['map50']),
            'map50_95': float(best['map50_95']),
            'memory_mb': float(best['max_memory_mb'])
        },
        'improvements': {
            'fps_improvement': float(fps_improvement),
            'fps_improvement_pct': float(fps_improvement_pct),
            'latency_improvement_ms': float(latency_improvement),
            'latency_improvement_pct': float(latency_improvement_pct),
            'map50_diff': float(map50_diff),
            'map50_95_diff': float(map50_95_diff)
        }
    }
    
    output_file = Path(__file__).parent / 'outputs' / 'optimal_model_selection.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(selection, f, indent=2)
    
    print(f"\n✓ Selection saved to: {output_file}")
    
    return selection

if __name__ == '__main__':
    try:
        selection = analyze_ablation_results()
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nOptimal model selected: {selection['optimal']['variant_label']}")
        print(f"Variant type: {selection['optimal']['variant_type']}")
        print(f"Variant value: {selection['optimal']['variant_value']}")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

