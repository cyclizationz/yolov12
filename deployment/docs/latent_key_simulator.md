# Latent-Key Offline Simulator (YOLOv12 Seg)

This repo contains an offline simulator that mimics a **server + client** pipeline for bandwidth saving:

- **Server** paints detected regions to a fixed color (masked stream).
- **Server** sends metadata in **MSK1 v3 SEI** (or a sidecar dump `msk1_payloads.bin` in offline mode).
- **Client** reconstructs the original frame by overlaying cached RGBA templates.

## Why latent-key

Hash / pHash based reuse can **flash** (template switches) because small boundary jitter can change the mask ROI enough to select different near-matches.

YOLOv12-seg already provides a compact 32-D latent vector per detection:

- **mask coefficients** (the 32 floats used with proto masks to generate the instance mask)

We can treat this vector as an **embedding** and reuse templates by **cosine similarity**, which better reflects semantic similarity than pHash of a binary mask.

## Mode overview

### Legacy (pHash ROI) mode

- Template reuse key: pHash(mask ROI) + tolerance.
- SEI carries: `path = canonical_hash_string` (string key)
- Client: uses key to fetch template.

### Latent-key mode (new)

- Template reuse key: **template_id** assigned by comparing **maskCoeff embedding** (cosine similarity).
- SEI carries:
  - `region.id = template_id`
  - `region.class_id = classId`
  - `region.flags = 0(new) / 1(reference)`
  - `region.path = ""` (unused)
- Client: **no hashing**, only `template_id -> RGBA` lookup.

## How template_id is assigned

For each detection:

1. Take `DL_RESULT.maskCoeff` (32 floats) from the model output.
2. L2-normalize it.
3. Find the best cached template (same class, similar bbox size) by cosine similarity.
4. If best similarity ≥ `--latent-thr`, reuse its `template_id`.
5. Otherwise, mint a new `template_id` and save the RGBA crop to `dictDir/<template_id>.png`.

We also apply a small temporal bias: if the bbox overlaps strongly with the previous frame’s bbox for that class, prefer reusing the last `template_id` to avoid flicker.

## How to run

### Build

```bash
cd deployment/build
cmake --build . -j
```

### Run racing video with latent-key

```bash
cd deployment
./build/Yolov12Deployment \
  -i /home/tiehangz/proj/datasets/racing/fm6.mkv \
  -m /home/tiehangz/proj/yolov12/deployment/yolov12n-seg.onnx \
  -o /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent \
  --yolo-class-color \
  --latent-key \
  --latent-thr 0.95 \
  --timing
```

Outputs:

- `original_output.mp4`
- `segmented_output.mp4` (masked)
- `recovered_output.mp4` (client reconstruction)
- `dict/` (templates saved as `<template_id>.png`)
- `msk1_payloads.bin` (MSK1 v3 payload dump)
- `report.json` (quality + timing + latent bank stats)

## Notes / knobs

- `--latent-thr`:
  - Higher (e.g. 0.97): fewer reuses, more templates, less risk of mismatch.
  - Lower (e.g. 0.92): more reuse, smaller template bank, potentially more boundary mismatch.
- **Hybrid sampling (recommended for rendered games)**:
  - `--latent-period 2` mints a new template about **every other frame** (helps recover quality/fluency).
  - Motion-triggered boost mints **more aggressively** during rapid motion:
    - `--latent-motion-iou` (default 0.6)
    - `--latent-motion-center` (default 20 px)
    - `--latent-motion-scale` (default 0.25 area ratio delta)
    - `--latent-motion-boost` (default 6 frames)
- **Perceptual note (paper discussion point)**:
  - We tested server-side compaction/reuse (`--latent-period-skip` / `--latent-merge`) and found that in high-motion,
    rendered content it can **reintroduce visible flashing/mismatch**, even if cosine similarity is high.
  - The safer approach is to keep **sampling** (periodic + motion boost) and use only the **client last-frame fallback**
    (reuse the last recoverable template when the current one hasn’t arrived yet).
- Bbox jitter still matters; recovery resizes template to bbox each frame.
- This is an **offline simulator**: “new template delivery” is approximated by skipping reconstruction for templates minted in the same frame.


