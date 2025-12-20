## Three-game comparison (pixel + racing + FC5)

This document compares the current **best pipeline per game type**:

- **Pixel game (Super Mario clip)**: pixel mode **Kalman + band selection + optical flow** (stable multi-object masking with low overhead).
- **Racing game (FM6)**: YOLO segmentation **latent-key** recovery (stable template IDs via maskCoeff embedding).
- **FPS game (FC5, roi_720p)**: YOLO segmentation **latent-key** recovery (same settings as racing-best).

### Artifacts (where to look)

- **Pixel best output**: `deployment/outputs/pixel_kalman_band2_flow_v1/`
- **Racing best output**: `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/`
- **FC5 output**: `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/`

### Summary table (key numbers)

Bandwidth is computed from **per-frame `pkt_size` (ffprobe)** comparing:
`original_output.mp4` vs `segmented_output.mp4`.

| Game | Method | Frames | Bandwidth saving % | Worse% (masked>base) | Avg SSIM (masked) | Avg SSIM (recovered) | Avg PSNR (masked) | Avg PSNR (recovered) | Avg total ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pixel (Mario) | Pixel Kalman+Flow | 187 | 12.37 | 44.39 | 0.9096 | 0.9746 | 17.89 | 31.27 | 110.00 |
| Racing (FM6) | YOLO latent-key | 910 | 13.66 | 24.73 | 0.6019 | 0.9392 | 9.29 | 29.41 | 238.55 |
| FPS (FC5) | YOLO latent-key | 1366 | 0.17 | 56.70 | 0.9113 | 0.9224 | 29.91 | 34.84 | 222.36 |

Notes:
- The FPS clip has **very small net bandwidth gain** despite high visual quality, suggesting the masked regions are not large enough (or not compressible enough) to dominate bitrate.
- “Worse%” is the fraction of frames where the masked bitstream packet is larger than baseline; it’s a useful indicator that masking sometimes hurts encoder prediction for that sequence.

### Reliability mode: “only send what the client can understand”

We added an optional **YOLO heal-only** policy that makes masking conservative:

- The server **only masks** a region if it can be reconstructed from an **already-known template** (client storage).
- Additionally, the server checks that the **template alpha mask matches the current segmentation mask** inside the bbox (near-perfect IoU, low spill). This prevents green-edge artifacts and background overwrite.
- If the check fails, the server keeps **original pixels** for that region (equivalent to “send original frame content” for what cannot be healed).

CLI flags:
- `--yolo-heal-only`
- `--yolo-heal-iou` (default 0.995)
- `--yolo-heal-extra` (default 0.005)

Tradeoff: this can reduce bitrate savings (sometimes even negative) because many regions are deemed not safely healable.

### Pixel game (best): Kalman + band selection + optical flow

**Config**: `deployment/outputs/pixel_kalman_band2_flow_v1/`

Plots:
- `deployment/outputs/pixel_kalman_band2_flow_v1/bandwidth_pixel_kalman_band2_flow_v1.png`
- `deployment/outputs/pixel_kalman_band2_flow_v1/quality_pixel_kalman_band2_flow_v1.png`

Why this is best for pixel games:
- Avoids scanning all templates every frame (ROI + periodic bootstrap).
- Band selection + sparse optical flow reduces noise while capturing multiple rows/instances.

### Racing game (best): YOLO latent-key (FM6)

**Config**: `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/`

Plots:
- `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/bandwidth_yolo_fm6_latent_hybrid_v2_reuse_stats.png`
- `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/quality_yolo_fm6_latent_hybrid_v2_reuse_stats.png`
- `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/latent_top_cosine_heatmap.png`
- `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/latent_top_usage.png`

Latent-bank stats (from `report.json`):
- `latent_bank_size`: 455
- `latent_reuse_ratio`: 0.50

Interpretation:
- About half of detections reuse a previous template ID at cosine >= 0.95, improving temporal stability without pHash brittleness.

### FPS game (FC5): YOLO latent-key (roi_720p)

**Model**:
- ONNX used by deployment: `deployment/yolov12n_fc5_seg_v1.onnx` (IR downgraded to 9 for ORT 1.16.x compatibility)

**Output**: `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/`

Plots:
- `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/bandwidth_yolo_fc5_latent_hybrid_v2_reuse_stats.png`
- `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/quality_yolo_fc5_latent_hybrid_v2_reuse_stats.png`
- `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/latent_top_cosine_heatmap.png`
- `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/latent_top_usage.png`

Latent-bank stats (from `report.json`):
- `latent_bank_size`: 979
- `latent_reuse_ratio`: 0.173
- `latent_reuse_cosine_quantiles.p50`: ~0.9995

Interpretation:
- FC5 produces a larger bank and a much lower reuse ratio than racing, consistent with higher appearance variability (or a less stable embedding under the FC5 data distribution).

### Repro commands (exact)

#### FC5 training + export (conda env)

```bash
cd /home/tiehangz/proj/yolov12
source /home/tiehangz/miniforge3/etc/profile.d/conda.sh
conda activate yolov12

python train_fc5.py
python deployment/downgrade_ir.py deployment/yolov12n_fc5_seg_v1.onnx
```

#### FC5 deployment (latent-key)

```bash
cd /home/tiehangz/proj/yolov12/deployment
./build/Yolov12Deployment \
  -i /home/tiehangz/proj/datasets/fps/roi_720p.mp4 \
  -o /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats \
  -m /home/tiehangz/proj/yolov12/deployment/yolov12n_fc5_seg_v1.onnx \
  --timing \
  --yolo-class-color \
  --latent-key \
  --latent-thr 0.95 \
  --latent-period 2 \
  --latent-motion-iou 0.6 \
  --latent-motion-center 20 \
  --latent-motion-scale 0.25 \
  --latent-motion-boost 6
```

#### FC5 analysis plots

```bash
cd /home/tiehangz/proj/yolov12/deployment
source /home/tiehangz/miniforge3/etc/profile.d/conda.sh
conda activate yolov12

python plot_bandwidth_ssim.py \
  --out-dir /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats \
  --tag yolo_fc5_latent_hybrid_v2_reuse_stats

python analyze_latent_bank.py \
  --out-dir /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats \
  --top 40
```


