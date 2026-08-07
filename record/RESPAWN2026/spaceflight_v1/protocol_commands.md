# Spaceflight evaluation protocol and commands

## Training and export

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/tiehangz/miniforge3/envs/yolov12/bin/python train_spaceflight.py
```

Pinned setup: `modification/model/yolov12n-seg.pt`, 300 epochs, 640 px,
batch 16, patience 0, seed 1337, deterministic mode, CUDA device 0. The
deployment ONNX copy was made compatible with the repository's ONNX Runtime:

```bash
/home/tiehangz/miniforge3/envs/yolov12/bin/python deployment/downgrade_ir.py \
  deployment/yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx
```

## Clips and model comparison

```bash
/home/tiehangz/miniforge3/envs/yolov12/bin/python \
  tools/experiments/prepare_spaceflight_eval.py
/home/tiehangz/miniforge3/envs/yolov12/bin/python \
  tools/experiments/compare_spaceflight_models.py
```

The clip tool scanned every decoded source frame at confidence 0.25, selected
90-frame contiguous spans around accepted labels, encoded exactly five clips,
then decoded and re-ran inference on every encoded frame.

## RESPAWN index and inference

```bash
deployment/build/Yolov12Deployment -d \
  -i record/RESPAWN2026/spaceflight_v1/clips/all_five.mp4 \
  -o experiments/encoder_eval/spaceflight_index_v1 \
  -m deployment/yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx \
  --build-index --latent-key
```

## Experiment entry points

```bash
.venv-experiments/bin/python tools/experiments/run_duo_screen.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --out-dir record/RESPAWN2026/spaceflight_v1/duo_screen \
  --model deployment/yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx

.venv-experiments/bin/python tools/experiments/run_rd_suite.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --out-dir record/RESPAWN2026/spaceflight_v1/rd --threads 4 \
  --figure-dir record/RESPAWN2026/spaceflight_v1/figures \
  --summary-stem spaceflight_rd

.venv-experiments/bin/python tools/experiments/run_gop_analysis_crf23.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --out-root record/RESPAWN2026/spaceflight_v1/gop_analysis \
  --game spaceflight \
  --clip-ids spaceflight_00 spaceflight_01 spaceflight_02 spaceflight_03 spaceflight_04

.venv-experiments/bin/python tools/experiments/analyze_overhead.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --rd-dir record/RESPAWN2026/spaceflight_v1/rd \
  --rd-points record/RESPAWN2026/spaceflight_v1/rd/spaceflight_rd_points.csv \
  --out-dir record/RESPAWN2026/spaceflight_v1/overhead

.venv-experiments/bin/python tools/experiments/analyze_generality.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --rd-points record/RESPAWN2026/spaceflight_v1/rd/spaceflight_rd_points.csv \
  --out-dir record/RESPAWN2026/spaceflight_v1/generality

.venv-experiments/bin/python tools/experiments/run_ablation_suite.py \
  --manifest record/RESPAWN2026/spaceflight_v1/offline_manifest.json \
  --out-dir record/RESPAWN2026/spaceflight_v1/ablation \
  --model deployment/yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx \
  --limit-clips-per-game 5 \
  --clip-ids spaceflight_00 spaceflight_01 spaceflight_02 spaceflight_03 spaceflight_04 \
  --config-kinds matching fill --exp35-encoder --use-normalized-input

/home/tiehangz/miniforge3/envs/yolov12/bin/python \
  tools/experiments/run_spaceflight_resource_benchmark.py
```

The RD suite used five target rates (8, 12, 16, 20, and 24 Mbps), matched
maxrate, 2x VBV buffer, libx264 medium, open-GOP defaults, and GOP 60.
