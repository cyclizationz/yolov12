# Training YOLOv12-seg Models with PE Variants

## Problem
YOLOv12-seg models don't contain `Attention` modules directly - they use `A2C2f` blocks instead of `C2PSA` blocks. To test PE variants, we need to train models that use `C2PSA` blocks (which contain `Attention` modules).

## Solution

### Option 1: Use YOLO11-seg Config (Recommended)
YOLO11-seg models use `C2PSA` blocks which contain `Attention` modules. You can train YOLO11-seg models with PE variants.

### Option 2: Create Modified Config
Create a modified YOLOv12-seg config that uses `C2PSA` blocks instead of `A2C2f`.

## Training Steps

### 1. Patch PSABlock to Use PE Variants

The notebook includes a function `set_pe_variant_for_training()` that patches `PSABlock.__init__` to use the specified PE variant.

### 2. Train Models

```python
# Train with NoPE (baseline)
train_model_with_pe_variant(
    model_config='yolo11-seg.yaml',
    pe_variant='none',
    data='coco8-seg.yaml',  # or 'coco.yaml' for full dataset
    epochs=10,
    imgsz=640,
    batch=16
)

# Train with RPE
train_model_with_pe_variant(
    model_config='yolo11-seg.yaml',
    pe_variant='rpe',
    data='coco8-seg.yaml',
    epochs=10,
    imgsz=640,
    batch=16
)

# Train with RoPE
train_model_with_pe_variant(
    model_config='yolo11-seg.yaml',
    pe_variant='rope',
    data='coco8-seg.yaml',
    epochs=10,
    imgsz=640,
    batch=16
)
```

### 3. Evaluate Trained Models

After training, load the trained models and run the ablation study:

```python
# Load trained models
model_none = YOLO('runs/segment/yolo11-seg_none_pe/weights/best.pt')
model_rpe = YOLO('runs/segment/yolo11-seg_rpe_pe/weights/best.pt')
model_rope = YOLO('runs/segment/yolo11-seg_rope_pe/weights/best.pt')

# Run benchmarks
results_none = benchmark_model(model_none, 'coco8-seg.yaml')
results_rpe = benchmark_model(model_rpe, 'coco8-seg.yaml')
results_rope = benchmark_model(model_rope, 'coco8-seg.yaml')
```

## Important Notes

1. **Training Time**: Training from scratch takes significant time. For quick testing:
   - Use `coco8-seg.yaml` (small dataset)
   - Reduce epochs (e.g., 5-10)
   - Use smaller batch size

2. **Model Config**: Use `yolo11-seg.yaml` which already has `C2PSA` blocks. If you want YOLOv12 architecture, you'll need to create a modified config.

3. **PE Variant Selection**: The patching happens at model creation time, so you need to:
   - Set PE variant BEFORE loading the model
   - Train separate models for each PE variant

4. **Evaluation**: After training, you can use the standard ablation notebook to evaluate all three variants.

## Quick Start

1. Open `PE_ablation.ipynb`
2. Set `TRAIN_MODELS = True` in the training cell
3. Adjust `EPOCHS`, `DATA_CONFIG`, etc. as needed
4. Run the training cell
5. After training completes, use the trained models in the ablation study

