
# ToMe Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Load NEW results
results_file = 'modification/outputs/tome_ablation_results_new.csv'
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
else:
    df = pd.read_csv(results_file)
    
    # Filter out failed runs (mAP=0)
    # valid_results = df[df['map50'] > 0].copy()
    # Actually, show all to demonstrate impact
    valid_results = df.copy()
    
    if len(valid_results) == 0:
        print("No results available.")
    else:
        fig = plt.figure(figsize=(12, 6))
        placements = valid_results['placement'].tolist()
        
        # 1. FPS
        ax1 = plt.subplot(1, 2, 1)
        fps_values = valid_results['fps'].tolist()
        x = np.arange(len(placements))
        bars = ax1.bar(x, fps_values, color='#1f77b4', alpha=0.8)
        ax1.set_ylabel('FPS')
        ax1.set_title('ToMe Throughput (r=0.25)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(placements, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
        # 2. mAP
        ax2 = plt.subplot(1, 2, 2)
        map_values = valid_results['map50'].tolist()
        bars2 = ax2.bar(x, map_values, color='#ff7f0e', alpha=0.8)
        ax2.set_ylabel('mAP50')
        ax2.set_title('ToMe Accuracy (r=0.25)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(placements, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, map_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Token Merging Placement Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path('modification/outputs/tome_ablation_comparison_new.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to: {output_path}")
        plt.show()

