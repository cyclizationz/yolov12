# FC5_00 Offline Processor Open x264 Test

## What Was Run

This run uses the offline processor directly on `fc5_00`, rather than re-encoding the previously dumped PNGs.

Output folder:

`record/RESPAWN2026/gop_analysis/fc5_00_offline_open_x264_crf23`

Processor command used x264 CRF-style settings close to the mentor script:

- `--enc-codec libx264`
- `--enc-crf 23`
- `--enc-preset medium`
- `--enc-tune none`
- `--enc-profile none`
- `--enc-level none`
- `--enc-open-gop-defaults`
- `--enc-aud 1`
- `--enc-repeat-headers 0`
- no bitrate/maxrate/bufsize

The resulting FFmpeg command inside the processor is:

```bash
ffmpeg ... -c:v libx264 -preset medium -pix_fmt yuv420p -crf 23 -x264-params "bframes=0:aud=1" ...
```

So the processor no longer forces x264 `keyint`, `min-keyint`, `scenecut=0`, profile, level, or VBV for this run.

## Whole-Clip Results

| Run | Video BSP (%) | Net BSP incl. MSK1 (%) | File BSP (%) | Negative frame (%) | Weighted negative contribution (%) |
|---|---:|---:|---:|---:|---:|
| offline processor open x264 CRF23 | 9.31 | 8.90 | 9.30 | 49.56 | 15.52 |
| Exp1 fixed 20 Mbps x264/VBV | 0.85 | 0.73 | 0.85 | 50.28 | 46.80 |

## Keyframe / GOP Behavior

| Run | Original keyframes | Masked keyframes | Original gap range | Masked gap range | Avg original gap | Avg masked gap |
|---|---:|---:|---|---|---:|---:|
| offline processor open x264 CRF23 | 30 | 40 | 1-167 | 1-167 | 59.93 | 45.13 |
| Exp1 fixed 20 Mbps x264/VBV | 31 | 31 | 60-60 | 60-60 | 60.00 | 60.00 |

Open x264 makes GOP boundaries content-dependent. Original and masked streams no longer share the same GOP indices: the open run has `30` original keyframes and `40` masked keyframes, with gaps ranging from `1-167` and `1-167`. The fixed Exp1 run has identical 60-frame keyframe spacing.

## Interpretation

Running the settings inside the offline processor confirms the encoder-protocol diagnosis. FC5_00 improves from about `0.73%` net savings under fixed 20 Mbps x264/VBV to `8.90%` net savings under open x264 CRF23.

This is close to, but lower than, the mentor-style PNG re-encode result (`~10.23%`). The likely reason is that the PNG re-encode starts from frames decoded from an already encoded output video, while this processor run encodes directly from the source frames generated during processing. The two inputs are visually similar but not codec-identical, and CRF encoders are sensitive to that difference.

For local GOP analysis, the open run should not be compared using fixed GOP IDs. The encoder chooses different scenecut/keyframe points for original and masked streams, so any local analysis must use encoder-reported keyframe intervals rather than `frame // 60` bins.

Raw data: `record/RESPAWN2026/gop_analysis/fc5_00_offline_open_x264_analysis.csv`.
