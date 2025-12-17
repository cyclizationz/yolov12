# Pixel mode: Kalman + Flow (recommended)

For pixel-style videos, the most robust configuration we found is:

- **Kalman + ROI matching**: keep the tracker locked to the object family.
- **Row-energy band selection**: scan only a few horizontal bands (cheap).
- **Sparse optical flow (LK)**: stabilize band centers under camera motion.
- **Band multi-peak**: detect multiple instances in those bands (multi-object).

This avoids the “template switching / flashing” behavior that breaks inter-frame prediction and hurts bitrate.

## Recommended command (multi-object)

```bash
cd /home/tiehangz/proj/yolov12/deployment

./build/Yolov12Deployment -p \
  -i /home/tiehangz/proj/datasets/pixel/pix_test_multi.mkv \
  -T /home/tiehangz/proj/datasets/pixel/templates_rescale \
  -o /home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_flow_multiobj \
  --timing \
  --pixel-bootstrap 30 \
  --pixel-topk 6 \
  --pixel-roi-pad 60 \
  --pixel-bands 2 \
  --pixel-band-sep 26 \
  --pixel-band-pad-y 24 \
  --pixel-flow 1 \
  --pixel-flow-pts 140 \
  --pixel-band-topk 2 \
  --pixel-max-peaks 80 \
  --pixel-peak-nms 8 \
  --pixel-min-score 0.45 \
  --pixel-adaptive-thr 1 \
  --pixel-thr-k 0.85 \
  --pixel-thr-lo 0.30 \
  --pixel-thr-hi 0.65
```

## Reading bandwidth saving

We compute bandwidth by comparing **per-frame packet sizes** (`pkt_size`) of:

- `original_output.mp4` (baseline)
- `segmented_output.mp4` (masked)

You can use `ffprobe` (or the helper script in `deployment/plot_bandwidth_ssim.py` if your python has matplotlib).


