
import os
import random
import yaml
from pathlib import Path

# Paths
# Based on 'find' result: /home/tiehangz/proj/datasets/..datasets/coco/val2017.txt
COCO_ROOT = Path('/home/tiehangz/proj/datasets/..datasets/coco')
VAL_TXT = COCO_ROOT / 'val2017.txt'
SUBSET_TXT = COCO_ROOT / 'val2017_1k.txt'
NEW_YAML = Path('/home/tiehangz/proj/yolov12/modification/coco_1k.yaml')
ORIG_YAML = Path('/home/tiehangz/proj/yolov12/ultralytics/cfg/datasets/coco.yaml')

def main():
    print(f"Checking for {VAL_TXT}...")
    if not VAL_TXT.exists():
        print(f"Error: {VAL_TXT} not found.")
        # Try fallback?
        return

    # Read lines
    with open(VAL_TXT, 'r') as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    
    print(f"Found {len(lines)} images in val set.")
    
    # Sample
    if len(lines) > 1000:
        subset = random.sample(lines, 1000)
        print(f"Sampled 1000 images.")
    else:
        subset = lines
        print(f"Using all {len(lines)} images.")
        
    # Write subset txt
    with open(SUBSET_TXT, 'w') as f:
        f.write('\n'.join(subset))
    print(f"Wrote subset list to {SUBSET_TXT}")
    
    # Create new YAML
    # Load original to get names
    with open(ORIG_YAML, 'r') as f:
        orig_cfg = yaml.safe_load(f)
    
    new_cfg = {
        'path': str(COCO_ROOT),
        'train': 'train2017.txt', # Keep original
        'val': 'val2017_1k.txt',  # Use subset
        'names': orig_cfg.get('names', {})
    }
    
    with open(NEW_YAML, 'w') as f:
        yaml.dump(new_cfg, f)
    
    print(f"Created new config: {NEW_YAML}")

if __name__ == "__main__":
    main()

