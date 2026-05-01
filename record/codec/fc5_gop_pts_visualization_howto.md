# FC5 crop: GOP/PTS/bitstream visualization (how to run)

This note explains how to visualize:
- where each *visual* frame maps into the MP4 bitstream (`pkt_pos` vs `pts_time`)
- whether there is periodic behavior *inside* each GOP (delta by GOP position)

## 1) Produce timeline + plots for a run

Given a run output folder with:
- `original_output.mp4`
- `segmented_output.mp4`

Run:

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/metrics/bitstream_timeline.py \
  --orig   /path/to/out/original_output.mp4 \
  --masked /path/to/out/segmented_output.mp4 \
  --out-dir /path/to/out/bitstream_viz
```

Artifacts:
- `timeline.csv`: per-frame pkt_size + pkt_pos + timestamps + GOP position + delta bytes
- `delta_by_gop_pos.csv`: mean/median/p10/p90 delta by GOP position
- `delta_timeseries.png`: pkt_size(base/masked) and delta over time
- `delta_gop_heatmap.png`: GOP index vs GOP position heatmap of delta bytes
- `pos_vs_pts.png`: scatter of pkt_pos vs pts_time (baseline + masked)

## 2) Extract a single GOP clip (baseline + masked)

```bash
conda run -n yolov12 python /home/tiehangz/proj/yolov12/tools/metrics/extract_gop_clip.py \
  --out-dir /path/to/out \
  --gop-id 10 \
  --out /path/to/out/gop10_clip
```

This re-encodes a short debug clip:
- `gop10_clip/baseline_clip.mp4`
- `gop10_clip/masked_clip.mp4`

You can then rerun `bitstream_timeline.py` on the clips for a zoomed-in view.

