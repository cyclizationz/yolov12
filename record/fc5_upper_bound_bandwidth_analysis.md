# FC5 Upper-Bound Bandwidth Saving (YOLOv12-seg) — Measurement + Theory + Test TODOs

This note answers: “If we **paint/mask all possible guns** (ignore healability/templates), how much H.264 bitrate can we save?” and “What is a reasonable **theoretical upper bound**?”

## What we measured (empirical “upper bound”)

### Setup

- **Video**: `/home/tiehangz/proj/datasets/fps/roi_720p.mp4`
- **Frames**: first **40** (`--max-frames 40`)
- **Model**: `deployment/yolov12n_fc5_seg_v1.onnx`
- **Mode**: **mask everything YOLO finds** (upper-bound experiment)
  - CLI flag: `--yolo-force-mask-all`
  - Notes:
    - This intentionally **breaks recovery** (client cannot reconstruct).
    - It is only meant to estimate “best possible encoder saving if object pixels became easy-to-code.”

### Command used

```bash
cd /home/tiehangz/proj/yolov12/deployment
OUT=/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_upperbound_maskall_40
rm -rf "$OUT" && mkdir -p "$OUT"

./build/Yolov12Deployment \
  -i /home/tiehangz/proj/datasets/fps/roi_720p.mp4 \
  -o "$OUT" \
  -m /home/tiehangz/proj/yolov12/deployment/yolov12n_fc5_seg_v1.onnx \
  -c 0.05 \
  -s 0.10 \
  --yolo-force-mask-all \
  --yolo-class-color \
  --max-frames 40 \
  --timing \
  > "$OUT/run.log" 2>&1
```

### Result summary (from `report.json` + per-frame pkt sizes)

- **Detections**: 40
- **Forced-masked regions**: 40
- **Average H.264 packet size**:
  - original: **11868.0 bytes/frame**
  - masked: **11567.3 bytes/frame**
- **Measured saving**: **2.53%**

Interpretation: even if we “mask all guns” (as YOLO sees them), the H.264 stream shrinks only ~2–3% on this short clip under the current encoding pipeline.

## Why “object is 50% pixels” does NOT imply “50% bitrate saving”

H.264 is not “bits-per-pixel constant.” It is mostly **inter-prediction**:

- Motion compensation can explain large parts of the image cheaply.
- Bits are spent on **residuals** (hard-to-predict details), edges, noise, and motion/scene changes.

### Simple bitrate decomposition model

Let the per-frame bitrate (or average bytes/frame) be:

```text
R = R_bg + R_obj + R_oh
```

- **`R_bg`**: background contribution
- **`R_obj`**: object (gun) contribution
- **`R_oh`**: overhead (headers, motion vectors, etc.)

If we could make the object region perfectly predictable (near-zero residual), then:

```text
R' ≈ R_bg + R_oh
```

So the fractional saving is:

```text
S = (R - R') / R  ≈  R_obj / R  ≡  k
```

where \(k\) is “what fraction of bits were actually spent on the object.”

### Connecting to the “50% pixels” assumption

If you assume bitrate is proportional to pixel area, then \(k \approx a\) where \(a\) is the object pixel fraction.

- With \(a=0.5\), you’d predict \(S \approx 50\%\).

But our measurement suggests:
\[
k \approx 0.025
\]
meaning **only ~2.5% of the coded bits** were attributable (in this clip/setting) to the gun region *as defined by the segmentation masks + painting method*.

This can happen even when the gun occupies a lot of pixels, because:

- The gun is often predictable via motion compensation.
- The background (camera motion, textures, HUD, lighting flicker) may dominate residual bitrate.
- Mask painting can *increase* edges/residuals if it introduces sharp boundaries or color changes.

## Exported calculation (how “saving%” was computed)

We compute saving over \(N\) frames using average frame packet sizes from `ffprobe`:

```text
saving% = (B̄_orig - B̄_masked) / B̄_orig * 100
```

Where \(\bar{B}\) is the mean of `pkt_size` over frames:

- `ffprobe -show_entries frame=pkt_size`
- take the first \(N\) frames (min length of both videos)
- compute mean pkt size for original and masked

## Test plan TODOs (configs to run next)

Goal: estimate a realistic **upper bound** and understand which encoder knobs change it. Checklist below is meant to be copy/paste friendly and unambiguous.

### A) YOLO detection coverage (paint-all mode)

- **Figure**: saving% vs `(-c, -s)` thresholds (mask-all mode)

![A) Upper-bound saving% vs YOLO thresholds](../experiments/fc5_upperbound/figures/A_threshold_heatmap.png)

- [ ] **Sweep detection thresholds** in `--yolo-force-mask-all` mode (it still depends on YOLO thresholds to decide which masks exist):
  - `-c`: `0.01, 0.03, 0.05, 0.10, 0.20`
  - `-s`: `0.05, 0.10, 0.20, 0.30, 0.50`
  - Record: saving%, avg pkt sizes, number of detections, and “worse%” (fraction of frames where masked frame is larger).
- [ ] **Try ROI crop variants** where gun is always present (first 40 frames), because background complexity changes bitrate a lot:
  - no crop
  - crop y to bottom 2/3
  - crop x to include more left/right context

### B) Painting style (affects residual bitrate)

Forced mode uses `paint_mask_color()` with `opt.paintAlpha` (default 0.6). This can create edges/residuals.

- **Figure**: average saving% vs `--paint-alpha` (mask-all mode)

![B) Avg saving% vs paint alpha](../experiments/fc5_upperbound/figures/B_paint_alpha.png)

- [ ] Sweep `paintAlpha`: `0.3, 0.6, 1.0` (if we don’t have a CLI flag yet, add `--paint-alpha`).
- [ ] Compare **solid fill** (alpha=1.0) vs **soft edges** (mask blur / feather) to see how much boundaries cost.

### C) Encoder settings (ffmpeg/x264) — YES, parameters matter

Changing x264 settings can materially change “saving%” because it changes prediction strategy and bit allocation.

- **Figure**: encoder sweep (CRF × preset) using the **best (conf, mask_thr)** from Fig A at `paint_alpha=1.0`

![C) Best saving% in sweep](../experiments/fc5_upperbound/figures/C_best_in_sweep.png)

#### Why `bframes` / `keyint` / `tune=zerolatency` can change *saving%*

Our metric is **relative saving** between two separately encoded videos:

```text
saving% = (B̄_orig - B̄_masked) / B̄_orig * 100
```

So an encoder knob can improve compression for **both** streams but still **reduce** saving%, if it helps the original *more* than the masked stream.

- **B-frames (`bframes`)**:
  - Often improves absolute compression for both streams.
  - Can *reduce saving%* if it makes the **original** much easier to predict (drops `B̄_orig`) while the masked stream benefits less.
  
- **GOP length (`keyint`)**:
  - Changes where I-frames appear and how far prediction reaches.
  - Can change the relative gap between original vs masked depending on motion/scene structure.

- **Tune `zerolatency` vs none**:
  - `-tune zerolatency` is designed for low-latency; it typically reduces temporal lookahead/decisions.
  - In our FC5 tests it often made the masked stream relatively worse, so saving% dropped or even flipped negative.

##### Default encoder setting for this project (recommended)

For bandwidth-saving evaluation (not strict real-time), we use:

- **tune**: **none**
- **bframes**: **0**
- **keyint**: **60** (unless otherwise noted)

We also produced an explicit knobs sweep under:

- `experiments/fc5_upperbound/results_encoder_knobs.csv`
- `experiments/fc5_upperbound/figures/C_encoder_knobs.png`

#### Recommended fixed encode settings for fair comparison

For consistent measurement, re-encode BOTH original and masked with the same `ffmpeg` parameters (don’t rely on OpenCV’s writer defaults).

```bash
# Example: constant-quality (CRF) encode, fixed GOP, default evaluation settings (tune none, bframes=0)
ffmpeg -y -v error -i original_output.mp4 \
  -c:v libx264 -preset veryfast -crf 23 \
  -x264-params "keyint=60:min-keyint=60:scenecut=0:bframes=0" \
  -an original_x264.mp4

ffmpeg -y -v error -i segmented_output.mp4 \
  -c:v libx264 -preset veryfast -crf 23 \
  -x264-params "keyint=60:min-keyint=60:scenecut=0:bframes=0" \
  -an masked_x264.mp4
```

- [ ] Sweep **CRF**: `18, 23, 28, 33`
- [ ] Sweep **preset**: `ultrafast, veryfast, medium, slow`
- [ ] Try **tune**: `zerolatency` vs (none)
- [ ] Try enabling **B-frames**: `bframes=0` vs `bframes=2`
- [ ] Try different **GOP**: `keyint=30` vs `60` vs `120`

For each setting, record:

- total bytes, bytes/frame, saving%, worse%

### D) Synthetic “50% pixels” sanity check (calibrate area → bits)

To connect “50% area” to bitrate, run a synthetic replacement experiment:

- **Figure**: synthetic 50% rectangle replacement vs best mask-all saving%

![D) Synthetic vs mask-all](../experiments/fc5_upperbound/figures/D_synthetic_vs_maskall.png)

- [ ] Replace a fixed **50% rectangular area** with a constant color (or blur) on the same 40 frames.
- [ ] Encode with the same x264 params as above and compute saving%.
- [ ] Compare:
  - 50% rectangle replacement saving
  - gun-mask replacement saving

## Notes / file locations

- Code adding forced mode:
  - `deployment/offline_processor.h`: `OfflineOptions::yoloForceMaskAll`
  - `deployment/main.cpp`: `--yolo-force-mask-all`
  - `deployment/offline_processor.cpp`: short-circuit to paint all masks
- Upper-bound output folder:
  - `/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_upperbound_maskall_40`

## In-band SEI end-to-end (inject → extract → stitch) on gun-heavy crop

We verified an end-to-end “in-band SEI” flow by:

- encoding masked video to **AnnexB H.264** with **AUD enabled**
- injecting MSK1 payloads into **SEI (nal_type=6, user_data_unregistered)**
- extracting the MSK1 payloads back from the `.h264`
- stitching a recovered video using the **extracted** payloads + the template `dict/`

### Inputs / outputs

- **Gun-heavy crop video (4s)**: `/home/tiehangz/proj/datasets/fps/roi_720p_gunheavy_4s.mp4`
- **Run folder**: `experiments/fc5_inband_flow_gunheavy/`
  - `server/segmented_output.mp4`: masked stream (server)
  - `masked_inband_sei.h264` + `masked_inband_sei.mp4`: masked stream with **in-band SEI**
  - `msk1_extracted.bin`: MSK1 payloads extracted from the injected `.h264`
  - `recovered_from_sei.mp4`: stitched using `msk1_extracted.bin`

### Final ratio (default encoder: tune none, bframes=0)

With the default evaluation encoder (`preset=veryfast, crf=23, keyint=60, bframes=0, tune=none`), on 120 frames:

- **saving%**: **2.67%** (orig_avg_pkt 8356.7 → masked_avg_pkt 8133.3)
- **worse%**: 35.8%

With the same settings but **CRF 18** (both original and masked re-encoded to CRF 18), on 120 frames:

- **saving%**: **26.29%** (orig_avg_pkt 13948.1 → masked_avg_pkt 10281.7)
- **worse%**: 9.2%

This is the “full flow” measurement on the crop where the gun occupies most pixels, using in-band SEI transport.
