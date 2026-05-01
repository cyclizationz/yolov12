# Reconstructed commands for codec sweep (FC5 crop)

This file documents how to reproduce the **FC5 crop** codec-sweep points under `record/codec/`, focusing on the commonly referenced **~7% saving** runs.

## Important: what “saving%” means in `codec_rd_points.csv`

`record/codec/codec_rd_points.csv` is produced by `tools/metrics/codec_sweep_analyze.py`. Its `saving_pct` is computed from **file bitrate**:

- baseline_bps = `(original_output.mp4 bytes * 8) / duration_s`
- masked_bps = `(segmented_output.mp4 bytes * 8) / duration_s`
- meta_bps = `(msk1_payloads.bin bytes * 8) / duration_s`
- saving_pct = `100 * (1 - (masked_bps + meta_bps) / baseline_bps)`

So it is a **whole-file** metric (not per-frame packet-sum).

## The “~7% saving” FC5 crop run (x264 CRF18)

Target run directory (from `codec_rd_points.csv`):

- `record/codec/fc5_crop_codec-libx264_crf18_mask-dominant_p200_fill-solid_feather4_k60_b0_sc0_tunenone_max2000`

Evidence sources:

- Folder name encodes: `encCodec=libx264`, `encCrf=18`, `encGop=60`, `bframes=0`, `scenecut=0`, `tune=none`, `mask-color=dominant period=200`, `fill=solid`, `feather=4`, `maxFrames=2000`.
- `report.json` encodes latent-key knobs that are **not** in the folder name:
  - `latent_key_enabled=true`
  - `latent_cosine_threshold=0.86`
  - `latent_sample_period=2`
  - motion gating: `latent_motion_iou=0.6`, `latent_motion_center=20`, `latent_motion_scale=0.25`, `latent_motion_boost=6`

Reconstructed command (paths match this repo layout):

```bash
/home/tiehangz/proj/yolov12/deployment/build/Yolov12Deployment \
  -i /home/tiehangz/proj/yolov12/video/fc5_crop.mkv \
  -o /home/tiehangz/proj/yolov12/record/codec/fc5_crop_codec-libx264_crf18_mask-dominant_p200_fill-solid_feather4_k60_b0_sc0_tunenone_max2000 \
  -m /home/tiehangz/proj/yolov12/deployment/yolov12n_fc5_seg_v1.onnx \
  --timing \
  --latent-key \
  --latent-bank /home/tiehangz/proj/yolov12/experiments/encoder_eval/fc5_crop_index_full_v1/dict/latent_bank.json \
  --latent-thr 0.86 \
  --latent-motion-iou 0.6 \
  --latent-motion-center 20 \
  --latent-motion-scale 0.25 \
  --latent-motion-boost 6 \
  --mask-color dominant \
  --mask-color-period 200 \
  --fill-mode solid \
  --feather-px 4 \
  --enc-codec libx264 \
  --enc-crf 18 \
  --enc-gop 60 \
  --enc-scenecut 0 \
  --enc-tune none \
  --max-frames 2000
```

Notes:

- The historical run’s `run.log` did **not** print `[Config]`/`[Command]` at the time, so the above combines:
  - **deterministic fields** from the directory name and `report.json`
  - **assumed defaults** for any flags not represented (e.g., whether `--yolo-class-color` was used is irrelevant to bitrate).

