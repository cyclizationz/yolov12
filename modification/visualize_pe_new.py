
# Comprehensive Visualization: All Variants Together
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Load NEW results
results_file = 'modification/outputs/pe_ablation_results_new.csv'
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
else:
    df = pd.read_csv(results_file)
    
    # Filter GPU results (assuming 'cuda' device)
    # If device column not present or needed, just use df
    gpu_results = df.copy()
    
    if len(gpu_results) == 0:
        print("No GPU results available for visualization.")
    else:
        # Create comprehensive figure with 3 subplots (mAP, FPS, Latency)
        fig = plt.figure(figsize=(18, 6))
        
        variants_list = gpu_results['variant'].tolist()
        
        # Color scheme
        colors = {}
        for v in variants_list:
            if 'PPC' in v:
                if '3x3' in v: colors[v] = '#90ee90'
                elif '5x5' in v: colors[v] = '#2ca02c' # Baseline
                elif '7x7' in v: colors[v] = '#32cd32'
                elif '9x9' in v: colors[v] = '#228b22'
            elif 'NoPE' in v: colors[v] = '#ff7f0e'
            else: colors[v] = '#808080'
            
        # 1. mAP50 Comparison
        ax1 = plt.subplot(1, 3, 1)
        map50_values = gpu_results['map50'].tolist()
        map5095_values = gpu_results['map50_95'].tolist()
        x = np.arange(len(variants_list))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, map50_values, width, label='mAP50', color=[colors.get(v, 'gray') for v in variants_list], alpha=0.8)
        bars2 = ax1.bar(x + width/2, map5095_values, width, label='mAP50-95', color=[colors.get(v, 'gray') for v in variants_list], alpha=0.5, hatch='//')
        
        ax1.set_ylabel('mAP Score')
        ax1.set_title('Accuracy (mAP) on COCO 1k Subset')
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. FPS Comparison
        ax2 = plt.subplot(1, 3, 2)
        fps_values = gpu_results['fps'].tolist()
        bars = ax2.bar(x, fps_values, color=[colors.get(v, 'gray') for v in variants_list], alpha=0.8)
        ax2.axhline(y=30, color='r', linestyle='--', label='30 FPS Target')
        ax2.set_ylabel('FPS')
        ax2.set_title('Throughput (FPS)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(variants_list, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, fps_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        # 3. Trade-off
        ax3 = plt.subplot(1, 3, 3)
        for i, v in enumerate(variants_list):
            ax3.scatter(fps_values[i], map50_values[i], label=v, s=150, color=colors.get(v, 'gray'), edgecolors='black')
        
        ax3.set_xlabel('FPS')
        ax3.set_ylabel('mAP50')
        ax3.set_title('Accuracy vs Speed Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle('PE Ablation Study (New): PPC Kernel Size on YOLOv12n-seg', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path('modification/outputs/pe_ablation_comparison_new.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to: {output_path}")
        plt.show()
