# Three-game comparison (pixel + racing + FC5)

This document compares the current **best pipeline per game type**:

- **Pixel game (Super Mario clip)**: pixel mode **Kalman + band selection + optical flow** (stable multi-object masking with low overhead).
- **Racing game (FM6)**: YOLO segmentation **latent-key** recovery (stable template IDs via maskCoeff embedding).
- **FPS game (FC5, roi_720p)**: YOLO segmentation **latent-key** recovery (same settings as racing-best).

## Artifacts (where to look)

- **Pixel best output**: `deployment/outputs/pixel_kalman_band2_flow_v1/`
- **Racing best output**: `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/`
- **FC5 output**: `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/`

## Summary (key numbers)

Bandwidth is computed from **per-frame `pkt_size` (ffprobe)** comparing `original_output.mp4` vs `segmented_output.mp4`.

Update (more exact per-frame method):

- We now compute bandwidth from **video packet sizes** (`ffprobe -show_packets`) and align in **presentation time order** (sort by `pts_time`).
- This avoids bad alignment when B-frames make PTS non-monotonic in file order.

- **Pixel (Mario)**:
  - **Method**: Pixel Kalman+Flow
  - **Frames**: 187
  - **Bandwidth saving %**: 12.37
  - **Worse% (masked>base)**: -44.39
  - **Avg SSIM (masked / recovered)**: 0.9096 / 0.9746
  - **Avg PSNR (masked / recovered)**: 17.89 / 31.27
  - **Avg total ms/frame**: 110.00

- **Racing (FM6)**:
  - **Method**: YOLO latent-key
  - **Frames**: 910
  - **Bandwidth saving %**: 13.66
  - **Worse% (masked>base)**: -24.73
  - **Avg SSIM (masked / recovered)**: 0.6019 / 0.9392
  - **Avg PSNR (masked / recovered)**: 9.29 / 29.41
  - **Avg total ms/frame**: 238.55

- **FPS (FC5)**:
  - **Method**: YOLO latent-key
  - **Frames**: 1366
  - **Bandwidth saving %**: 0.17
  - **Worse% (masked>base)**: -56.70
  - **Avg SSIM (masked / recovered)**: 0.9113 / 0.9224
  - **Avg PSNR (masked / recovered)**: 29.91 / 34.84
  - **Avg total ms/frame**: 222.36

Notes:

- The FPS clip has **very small net bandwidth gain** despite high visual quality, suggesting the masked regions are not large enough (or not compressible enough) to dominate bitrate.
- “Worse%” is the fraction of frames where the masked bitstream packet is larger than baseline; it’s a useful indicator that masking sometimes hurts encoder prediction for that sequence.

## Offline evaluation spec (what to measure, and how)

This project is motivated by cloud gaming, but the current milestone is an **offline, reproducible evaluation harness** for:

- **Bandwidth / cost**: how many bytes would we send if we ship masked video + metadata instead of vanilla video.
- **Quality**: how close the client-recovered frames are to the original.
- **Reliability**: whether we *only* send what the client can correctly recover (no green edges / no background overwrite), and how often we fall back.
- **Compute**: server-side + client-side processing time per frame.

### Bandwidth metrics (offline proxy for “bytes sent”)

We currently measure bandwidth as **coded video payload bytes in MP4 samples**:

- **Per-frame bytes**: `ffprobe` packet `size` for the video stream (often 1 MP4 sample ≈ 1 encoded picture).
- **Total bytes**: sum of those packet sizes across the clip.
- **Saving%**: \(100 * (1 - masked\_bytes / orig\_bytes)\).
- **Worse%**: fraction of aligned packets where `masked_pkt_size > orig_pkt_size`.

How to compute (uses the same output MP4s already referenced in this doc):

```bash
python /home/tiehangz/proj/yolov12/tools/metrics/compare_mp4_packets.py \
  --orig  /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/original_output.mp4 \
  --masked /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/segmented_output.mp4 \
  --align time --time-tol-ms 6
```

**Do we need to redesign per-frame measurement for MP4?**

- **Usually no**: for H.264/H.265 in MP4, `ffprobe -show_packets` yields one `AVPacket` per sample, and one sample is typically one coded frame (access unit).
- **But be careful with reordering**: with B-frames, decode/display order differs. If you want per-frame curves, align by **PTS** (`--align pts`) rather than naive index.
- **What it excludes**: container overhead (moov/mdat structure) and all transport overhead (RTP/UDP/IP/etc.). It’s a codec-payload metric.

### Metadata overhead (MSK1 / SEI)

For YOLO latent-key runs, each output directory contains `msk1_payloads.bin` (len-prefixed MSK1 payload per frame).

Compute payload overhead:

```bash
python /home/tiehangz/proj/yolov12/tools/metrics/msk1_overhead.py \
  --msk1-bin /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/msk1_payloads.bin
```

Notes:

- This reports **payload bytes** (and the `.bin` framing overhead). Actual in-band SEI NAL overhead adds a UUID wrapper + RBSP fields.
- For cloud cost accounting, the right “bytes sent” definition is: **(video payload bytes) + (metadata bytes)**, then add transport overhead (next section).

### Quality metrics

We already compute these in `report.json`:

- **SSIM / PSNR**: for `segmented_output.mp4` vs `original_output.mp4`, and `recovered_output.mp4` vs `original_output.mp4`.

Optional (not currently enabled here):

- **VMAF**: recommended for QoE, but this requires `ffmpeg` built with `libvmaf` (this machine currently does not have that filter enabled).

### Reliability metrics (client correctness, “no green edges” story)

What we want to claim offline:

- The server **only masks what the client can reconstruct**, otherwise it should keep original pixels (heal-only mode).
- In recovered regions, visual artifacts like “green edges” should not appear.

What we can measure **with existing artifacts** (no pipeline rewrite):

- **Template reuse stability**: `latent_reuse_ratio`, `latent_bank_size` from `report.json`.
- **Masked coverage (proxy)**: sum of template alpha pixels over stitched regions (derived from `dict/{tid}.png` alpha + MSK1 regions).
- **Healed-region pixel exactness (diagnostic)**: compare `recovered_output.mp4` vs `original_output.mp4` only inside template alpha pixels.
  - Warning: because outputs are independently encoded, “exact match == 100%” is generally **not achievable** unless we compare in a lossless domain. This metric is still useful as a **regression detector** for scaling / bbox alignment issues (green-edge type problems usually spike errors).

Command:

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/metrics/heal_exactness.py \
  --orig /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/original_output.mp4 \
  --recovered /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/recovered_output.mp4 \
  --msk1-bin /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/msk1_payloads.bin \
  --dict-dir /home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/dict
```

### Compute metrics (why we’re not real-time yet, but still valuable)

From `report.json` we already have:

- **Per-frame total_ms** (end-to-end for the offline pipeline).
- We should add a future breakdown (decode / YOLO / stitch / encode) if we want actionable profiling.

### On-wire bandwidth (WebRTC) with minimal pipeline change

Goal: keep the **same produced videos** (`original_output.mp4`, `segmented_output.mp4`) and measure **bytes on the wire** when sending via WebRTC.

Two practical options:

- **Option A (recommended, real WebRTC accounting)**: run a WebRTC sender/receiver and use `getStats()`:
  - Use `outbound-rtp.bytesSent` for media bytes and `candidate-pair.bytesSent` for total transport bytes on the selected path.
  - This includes SRTP/RTP/IP overhead and reflects pacing/retransmissions (depending on stat used).
  - This does **not** require changing the segmentation/recovery pipeline; it’s a separate transport harness that streams the existing outputs.

- **Option B (pcap ground truth)**: run the same WebRTC session, capture packets with `tcpdump`, and sum bytes for the flow:
  - Most accurate, but slightly more setup (session + capture).

Implementation note:

- Because `gst-inspect-1.0 webrtcbin` is available on this machine, we can stream **already-encoded H.264** from MP4 (demux → parse → RTP payloader → WebRTC) without re-encoding. This keeps the codec payload comparable between “original” and “masked” cases.
- We are not including on-wire numbers in the main comparison yet (keeping the offline story focused on codec-payload + metadata).

## Encoder-aligned evaluation (optimal encoder settings)

To fairly compare games under an encoder regime aligned with the FC5 “optimal settings”, we re-encode both `original_output.mp4` and `segmented_output.mp4` with identical x264 parameters:

- `preset=veryfast, crf=18, keyint=60, bframes=0, tune=none`

Results (fair re-encode comparison, CRF18):

- **Pixel (Mario)**:
  - **Config**: `deployment/outputs/pixel_kalman_band2_flow_v1/`
  - **Frames**: 187
  - **Saving% @ CRF18**: 18.32
  - **Worse%**: -6.95
  - **Artifacts**: `experiments/encoder_eval/pixel_kalman_band2_flow_v1/`

![Pixel (Mario) encoder-aligned bandwidth @ CRF18](../experiments/encoder_eval/pixel_kalman_band2_flow_v1/bandwidth_crf18.png)

*Figure (Pixel/Mario, CRF18 encoder-aligned bandwidth). Blue = `orig_x264_crf18.mp4`; orange = `masked_x264_crf18.mp4`. Packet sizes are PTS-sorted and time-aligned (tolerance 6ms), then averaged over 10-frame windows.*

- **Racing (FM6)**:
  - **Config**: `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/`
  - **Frames**: 910
  - **Saving% @ CRF18**: 17.25
  - **Worse%**: -3.52
  - **Artifacts**: `experiments/encoder_eval/yolo_fm6_latent_hybrid_v2_reuse_stats/`

![Racing (FM6) encoder-aligned bandwidth @ CRF18](../experiments/encoder_eval/yolo_fm6_latent_hybrid_v2_reuse_stats/bandwidth_crf18.png)

*Figure (FM6, CRF18 encoder-aligned bandwidth). Blue = `orig_x264_crf18.mp4`; orange = `masked_x264_crf18.mp4`. Packet sizes are PTS-sorted and time-aligned (tolerance 6ms), then averaged over 10-frame windows.*

- **FPS (FC5, gun-heavy crop 4s)**:
  - **Config**: `experiments/fc5_inband_flow_gunheavy/`
  - **Frames**: 120
  - **Saving% @ CRF18**: 26.29
  - **Worse%**: -9.2
  - **Artifacts**: `experiments/fc5_inband_flow_gunheavy/` (CRF18 re-encodes: `orig_x264_crf18.mp4`, `masked_x264_crf18.mp4`)

![FC5 gun-heavy crop (4s) bandwidth @ CRF18: baseline vs masked bytes per frame (10-frame window averages)](../experiments/fc5_inband_flow_gunheavy/bandwidth_fc5_gunheavy_crf18.png)

*Figure (FC5 gun-heavy, CRF18 encoder-aligned bandwidth). Blue = `orig_x264_crf18.mp4`; orange = `masked_x264_crf18.mp4`. Packet sizes are PTS-sorted and time-aligned (tolerance 6ms), then averaged over 10-frame windows. Black markers highlight top savings windows.*

- **FPS (FC5, full-length 1366 frames, same ROI as gun-heavy crop)**:
  - **Config**: `deployment/outputs/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/` (defaults: `c=0.2`, `s=0.6`, `paint_alpha=1.0`)
  - **Frames**: 1366
  - **ROI**: `crop=960:520:x=320:y=200` (matches `/home/tiehangz/proj/datasets/fps/roi_720p_gunheavy_4s.mp4`)
  - **Saving% @ CRF18**: -0.12
  - **Artifacts**: `experiments/encoder_eval/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/` (`orig_x264_crf18.mp4`, `masked_x264_crf18.mp4`)

## Additional measurements (quality / reliability / compute)

All values below are computed from existing artifacts under each output folder:

- `report.json` (per-frame SSIM/PSNR and `total_ms`)
- `msk1_payloads.bin` (metadata bytes)
- YOLO only: `dict/{tid}.png` + MSK1 regions for “healed alpha coverage”

Helper scripts:

- `tools/metrics/report_summary.py`
- `tools/metrics/msk1_overhead.py`
- `tools/metrics/heal_exactness.py` (YOLO only diagnostic)
- `tools/metrics/recon_frame_examples.py` (qualitative examples: best/worst reconstructed frames)
- `tools/metrics/build_per_frame_csv.py` (merge per-frame bytes + per-frame timings/counts into `per_frame_metrics.csv`)
- `tools/gui/review_frames.py` (OpenCV GUI to review original|masked and optionally label gt object counts)

### Qualitative reconstruction examples (best vs worst frames)

We generate a single comparison figure that picks, for each game case in Table~\ref{tab:eval-quality}, the **best** and **worst** reconstructed frames by a combined score of (PSNR, SSIM) from `report.json`, then extracts baseline and reconstructed frames for those indices:

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/metrics/recon_frame_examples.py \
  --out /home/tiehangz/proj/yolov12/record/figures/recon_examples_best_worst.png
```

Interpretation notes:

- **Racing (FM6)**: worst frames often coincide with **dimmed scenes**, where appearance changes reduce detector confidence/template selection stability, causing visibly darker/mismatched overlays.
- **FPS (FC5)**: worst frames often reflect **mis-segmentation or mask/bbox misalignment**, which can mask/stitch the wrong pixels and break reconstruction quality.

### Per-frame metrics CSV (bytes/timing/counts)

Each output folder now records per-frame timing/count fields into `report.json` under `per_frame[]`.
To build a CSV that merges those fields with **baseline/masked bytes per frame** (from `ffprobe -show_packets`), run:

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/metrics/build_per_frame_csv.py \
  --out-dir /path/to/output_folder
```

This writes `/path/to/output_folder/per_frame_metrics.csv` with `frame_id` starting at **1**.

Optional: annotate ground-truth object counts (object_present_gt) using the GUI, which writes `object_gt.csv`:

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/gui/review_frames.py \
  --out-dir /path/to/output_folder
```

### Pixel (Mario) – `pixel_kalman_band2_flow_v1`

- **Compute (ms/frame)**: mean=110.00, p50=96.28, p95=111.75, p99=239.40
- **Quality (masked vs orig)**:
  - SSIM: mean=0.9096, p50=0.9080, p10=0.9014
  - PSNR: mean=17.89, p50=15.69, p10=15.30
- **Quality (recovered vs orig)**:
  - SSIM: mean=0.9746, p50=0.9736, p10=0.9716
  - PSNR: mean=31.27, p50=29.13, p10=28.85
- **Metadata overhead (MSK1 payload, relative to segmented video payload bytes)**:
  - avg payload = 1885.8 B/frame
  - payload/video = 37.29%  (note: Pixel mode packs much denser metadata than YOLO)
- **Template-heal diagnostics**: N/A (Pixel pipeline does not use YOLO `dict/{tid}.png` templates)
  - **MSK1 structure (paper-critical)**:
    - ref-frame ratio (frames with any regions): 97.33%
    - regions/frame: mean=42.27, p50=45, p90=47, max=48
    - path-string share of MSK1 payload: 47.07%

### Racing (FM6) – `yolo_fm6_latent_hybrid_v2_reuse_stats`

- **Compute (ms/frame)**: mean=238.55, p50=237.08, p95=259.08, p99=269.67
- **Quality (masked vs orig)**:
  - SSIM: mean=0.6019, p50=0.5986, p10=0.5936
  - PSNR: mean=9.29, p50=9.29, p10=9.14
- **Quality (recovered vs orig)**:
  - SSIM: mean=0.9392, p50=0.9406, p10=0.8938
  - PSNR: mean=29.41, p50=27.63, p10=24.69
- **Reliability**:
  - latent bank size = 455
  - latent reuse ratio = 0.50
  - MSK1 payload overhead = 49.0 B/frame (payload/video = 0.331%)
  - healed alpha coverage = 21.19% of pixels (YOLO templates’ alpha, over all frames)
  - ref-frame ratio (frames with any regions): 100.0%
  - regions/frame: mean=1.00, p50=1, p90=1, max=1

### FPS (FC5) – `yolo_fc5_latent_hybrid_v2_reuse_stats`

- **Compute (ms/frame)**: mean=222.36, p50=221.57, p95=238.95, p99=252.84
- **Quality (masked vs orig)**:
  - SSIM: mean=0.9113, p50=0.9126, p10=0.8244
  - PSNR: mean=29.91, p50=16.00, p10=12.67
- **Quality (recovered vs orig)**:
  - SSIM: mean=0.9224, p50=0.9322, p10=0.8306
  - PSNR: mean=34.84, p50=22.52, p10=14.59
- **Reliability**:
  - latent bank size = 979
  - latent reuse ratio = 0.173
  - MSK1 payload overhead = 45.94 B/frame (payload/video = 0.336%)
  - healed alpha coverage = 1.81% of pixels (YOLO templates’ alpha, over all frames)
  - note: ref/raw ratios and regions/frame should be reported for the cropped FC5 case used in the paper table (see `experiments/fc5_inband_flow_gunheavy/`)

## Reliability mode: “only send what the client can understand”

We added an optional **YOLO heal-only** policy that makes masking conservative:

- The server **only masks** a region if it can be reconstructed from an **already-known template** (client storage).
- Additionally, the server checks that the **template alpha mask matches the current segmentation mask** inside the bbox (near-perfect IoU, low spill). This prevents green-edge artifacts and background overwrite.
- If the check fails, the server keeps **original pixels** for that region (equivalent to “send original frame content” for what cannot be healed).

CLI flags:

- `--yolo-heal-only`
- `--yolo-heal-iou` (default 0.995)
- `--yolo-heal-extra` (default 0.005)

Tradeoff: this can reduce bitrate savings (sometimes even negative) because many regions are deemed not safely healable.

## Pixel game (best): Kalman + band selection + optical flow

**Config**: `deployment/outputs/pixel_kalman_band2_flow_v1/`

Plots:

![Pixel (Mario) bandwidth: baseline vs masked bytes per frame (10-frame window averages)](../deployment/outputs/pixel_kalman_band2_flow_v1/bandwidth_pixel_kalman_band2_flow_v1.png)

*Figure (Pixel/Mario bandwidth). Blue = baseline `original_output.mp4` bytes/frame; orange = masked `segmented_output.mp4` bytes/frame, using PTS-aligned packet sizes and 10-frame averaging. Black markers highlight top savings windows.*

![Pixel (Mario) quality: SSIM/PSNR for masked vs recovered](../deployment/outputs/pixel_kalman_band2_flow_v1/quality_pixel_kalman_band2_flow_v1.png)

*Figure (Pixel/Mario quality). Bars show SSIM and PSNR for masked vs recovered outputs, aggregated over 10-frame windows. “Recovered” is the client-stitched reconstruction compared to `original_output.mp4`.*

Why this is best for pixel games:

- Avoids scanning all templates every frame (ROI + periodic bootstrap).
- Band selection + sparse optical flow reduces noise while capturing multiple rows/instances.

## Racing game (best): YOLO latent-key (FM6)

**Config**: `deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/`

Plots:

![Racing (FM6) bandwidth: baseline vs masked bytes per frame (10-frame window averages)](../deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/bandwidth_yolo_fm6_latent_hybrid_v2_reuse_stats.png)

*Figure (FM6 bandwidth). Blue = baseline `original_output.mp4` bytes/frame; orange = masked `segmented_output.mp4` bytes/frame, using PTS-aligned packet sizes and 10-frame averaging. Black markers highlight top savings windows.*

![Racing (FM6) quality: SSIM/PSNR for masked vs recovered](../deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/quality_yolo_fm6_latent_hybrid_v2_reuse_stats.png)

*Figure (FM6 quality). Bars show SSIM and PSNR for masked vs recovered outputs, aggregated over 10-frame windows. Recovered quality is the key “cloud gaming” proxy (client experience).*

![Racing (FM6) latent cosine heatmap](../deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/latent_top_cosine_heatmap.png)

*Figure (FM6 latent cosine heatmap). Similarity between frequent latent templates; tighter clusters indicate stable embedding reuse and easier template ID assignment.*

![Racing (FM6) latent top usage](../deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/latent_top_usage.png)

*Figure (FM6 latent usage). Template usage frequency (top-K) across the clip; a long tail usually means higher visual diversity and less reuse.*

Latent-bank stats (from `report.json`):

- `latent_bank_size`: 455
- `latent_reuse_ratio`: 0.50

Interpretation:

- About half of detections reuse a previous template ID at cosine >= 0.95, improving temporal stability without pHash brittleness.

## FPS game (FC5): YOLO latent-key (roi_720p)

**Model**:

- ONNX used by deployment: `deployment/yolov12n_fc5_seg_v1.onnx` (IR downgraded to 9 for ORT 1.16.x compatibility)

**Output**: `deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/`

Plots:

![FPS (FC5) bandwidth: baseline vs masked bytes per frame (10-frame window averages)](../deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/bandwidth_yolo_fc5_latent_hybrid_v2_reuse_stats.png)

*Figure (FC5 bandwidth). Blue = baseline `original_output.mp4` bytes/frame; orange = masked `segmented_output.mp4` bytes/frame, using PTS-aligned packet sizes and 10-frame averaging. Black markers highlight top savings windows.*

![FPS (FC5) quality: SSIM/PSNR for masked vs recovered](../deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/quality_yolo_fc5_latent_hybrid_v2_reuse_stats.png)

*Figure (FC5 quality). Bars show SSIM and PSNR for masked vs recovered outputs, aggregated over 10-frame windows.*

![FPS (FC5) latent cosine heatmap](../deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/latent_top_cosine_heatmap.png)

*Figure (FC5 latent cosine heatmap). Compared to racing, the structure is less clustered, consistent with lower reuse in FC5.*

![FPS (FC5) latent top usage](../deployment/outputs/yolo_fc5_latent_hybrid_v2_reuse_stats/latent_top_usage.png)

*Figure (FC5 latent usage). Top-K template usage. A broader distribution implies more unique templates and a lower reuse ratio.*

Latent-bank stats (from `report.json`):

- `latent_bank_size`: 979
- `latent_reuse_ratio`: 0.173
- `latent_reuse_cosine_quantiles.p50`: ~0.9995

Interpretation:

- FC5 produces a larger bank and a much lower reuse ratio than racing, consistent with higher appearance variability (or a less stable embedding under the FC5 data distribution).

## Repro commands (exact)

### FC5 training + export (conda env)

```bash
cd /home/tiehangz/proj/yolov12
source /home/tiehangz/miniforge3/etc/profile.d/conda.sh
conda activate yolov12

python train_fc5.py
python deployment/downgrade_ir.py deployment/yolov12n_fc5_seg_v1.onnx
```

### FC5 deployment (latent-key)

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

### FC5 analysis plots

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
