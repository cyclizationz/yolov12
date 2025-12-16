#!/usr/bin/env python3
"""
Deployment Comparison Visualization

Creates comprehensive visualizations comparing deployment model variants.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
except ImportError as e:
    print(f"Error: Required packages not available: {e}")
    print("Please install: pandas, matplotlib, seaborn")
    sys.exit(1)

def load_benchmark_results(csv_path=None, json_path=None):
    """Load benchmark results from CSV or JSON."""
    base_dir = Path(__file__).parent
    
    if csv_path is None:
        csv_path = base_dir / 'outputs' / 'deployment_benchmark_results.csv'
    if json_path is None:
        json_path = base_dir / 'outputs' / 'deployment_benchmark_results.json'
    
    # Try CSV first
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        return df
    
    # Fallback to JSON
    if Path(json_path).exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    raise FileNotFoundError(f"Benchmark results not found at {csv_path} or {json_path}")

def create_comparison_visualizations(df, output_path=None):
    """Create comprehensive comparison visualizations."""
    if output_path is None:
        output_path = Path(__file__).parent / 'outputs' / 'deployment_comparison.png'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Configure plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Prepare data
    df['model_format'] = df['model_name'] + ' - ' + df['format']
    
    # Color mapping
    format_colors = {
        'pytorch': '#1f77b4',
        'onnx': '#ff7f0e',
        'openvino_int8': '#2ca02c',
        'tensorrt_fp16': '#d62728',
        'tensorrt_int8': '#9467bd'
    }
    
    colors = [format_colors.get(fmt, '#808080') for fmt in df['format']]
    
    # 1. FPS Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.barh(range(len(df)), df['fps'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['model_format'], fontsize=8)
    ax1.set_xlabel('FPS (Throughput)', fontsize=10)
    ax1.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
    ax1.axvline(x=30, color='r', linestyle='--', linewidth=2, label='30 FPS Target', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df['fps'])):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', ha='left', va='center', fontsize=8)
    
    # 2. Latency Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.barh(range(len(df)), df['avg_latency_ms'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(df['model_format'], fontsize=8)
    ax2.set_xlabel('Latency (ms)', fontsize=10)
    ax2.set_title('Latency Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars2, df['avg_latency_ms'])):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center', fontsize=8)
    
    # 3. Memory Usage Comparison
    ax3 = plt.subplot(2, 3, 3)
    memory_data = df['max_memory_mb'].fillna(0)
    bars3 = ax3.barh(range(len(df)), memory_data, color=colors, alpha=0.8)
    ax3.set_yticks(range(len(df)))
    ax3.set_yticklabels(df['model_format'], fontsize=8)
    ax3.set_xlabel('Memory Usage (MB)', fontsize=10)
    ax3.set_title('Memory Usage Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars3, memory_data)):
        if val > 0:
            ax3.text(val + 5, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', ha='left', va='center', fontsize=8)
    
    # 4. mAP50 Comparison
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.barh(range(len(df)), df['map50'], color=colors, alpha=0.8)
    ax4.set_yticks(range(len(df)))
    ax4.set_yticklabels(df['model_format'], fontsize=8)
    ax4.set_xlabel('mAP50', fontsize=10)
    ax4.set_title('mAP50 Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars4, df['map50'])):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=8)
    
    # 5. mAP50-95 Comparison
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.barh(range(len(df)), df['map50_95'], color=colors, alpha=0.8)
    ax5.set_yticks(range(len(df)))
    ax5.set_yticklabels(df['model_format'], fontsize=8)
    ax5.set_xlabel('mAP50-95', fontsize=10)
    ax5.set_title('mAP50-95 Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars5, df['map50_95'])):
        ax5.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=8)
    
    # 6. Accuracy vs Speed Trade-off
    ax6 = plt.subplot(2, 3, 6)
    for fmt in df['format'].unique():
        fmt_data = df[df['format'] == fmt]
        ax6.scatter(fmt_data['fps'], fmt_data['map50_95'],
                   label=fmt.replace('_', ' ').title(),
                   alpha=0.7, s=150,
                   color=format_colors.get(fmt, '#808080'),
                   edgecolors='black', linewidths=1.5)
    
    ax6.set_xlabel('FPS (Throughput)', fontsize=10)
    ax6.set_ylabel('mAP50-95 (Accuracy)', fontsize=10)
    ax6.set_title('Accuracy vs Speed Trade-off', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=30, color='r', linestyle='--', linewidth=2, alpha=0.7, label='30 FPS Target')
    
    plt.suptitle('Deployment Model Comparison: Baseline vs Optimal (PPC 5×5)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()
    
    # Create speedup comparison
    create_speedup_comparison(df, output_path.parent / 'deployment_speedup_comparison.png')

def create_speedup_comparison(df, output_path):
    """Create speedup comparison chart."""
    # Group by format and calculate speedup
    baseline_pytorch = df[(df['model_name'] == 'baseline') & (df['format'] == 'pytorch')]
    if len(baseline_pytorch) == 0:
        print("⚠ Baseline PyTorch not found. Skipping speedup comparison.")
        return
    
    baseline_fps = baseline_pytorch.iloc[0]['fps']
    
    # Calculate speedup for each variant
    df['speedup'] = df['fps'] / baseline_fps
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Speedup by format
    format_speedup = df.groupby('format')['speedup'].mean().sort_values(ascending=False)
    bars1 = ax1.bar(range(len(format_speedup)), format_speedup.values, 
                    color=[format_colors.get(fmt, '#808080') for fmt in format_speedup.index],
                    alpha=0.8)
    ax1.set_xticks(range(len(format_speedup)))
    ax1.set_xticklabels([fmt.replace('_', ' ').title() for fmt in format_speedup.index], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Speedup (×)', fontsize=10)
    ax1.set_title('Average Speedup by Format (vs Baseline PyTorch)', fontsize=12, fontweight='bold')
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, format_speedup.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}×', ha='center', va='bottom', fontsize=9)
    
    # Speedup comparison: baseline vs optimal
    comparison_data = []
    for fmt in df['format'].unique():
        baseline_data = df[(df['model_name'] == 'baseline') & (df['format'] == fmt)]
        optimal_data = df[(df['model_name'] == 'optimal') & (df['format'] == fmt)]
        
        if len(baseline_data) > 0 and len(optimal_data) > 0:
            comparison_data.append({
                'format': fmt,
                'baseline_speedup': baseline_data.iloc[0]['speedup'],
                'optimal_speedup': optimal_data.iloc[0]['speedup']
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x = np.arange(len(comp_df))
        width = 0.35
        
        bars2a = ax2.bar(x - width/2, comp_df['baseline_speedup'], width,
                        label='Baseline', color='#1f77b4', alpha=0.8)
        bars2b = ax2.bar(x + width/2, comp_df['optimal_speedup'], width,
                        label='Optimal (PPC 5×5)', color='#ff7f0e', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([fmt.replace('_', ' ').title() for fmt in comp_df['format']],
                            rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Speedup (×)', fontsize=10)
        ax2.set_title('Speedup: Baseline vs Optimal by Format', fontsize=12, fontweight='bold')
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars2a, bars2b]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}×', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Deployment Speedup Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Speedup comparison saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    try:
        print("=" * 80)
        print("DEPLOYMENT COMPARISON VISUALIZATION")
        print("=" * 80)
        
        df = load_benchmark_results()
        print(f"\nLoaded {len(df)} benchmark results")
        
        create_comparison_visualizations(df)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

