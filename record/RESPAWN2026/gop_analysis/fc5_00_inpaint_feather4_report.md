# FC5_00 Neighbor Fill + Feather Test

This test checks whether using the same local neighborhood idea as mask feathering gives a more natural masked stream than a single dominant color. The run uses OpenCV inpaint as the fill source and keeps the previously preferred `--feather-px 4`.

Output video:

`record/RESPAWN2026/gop_analysis/fc5_00_open_x264_inpaint_feather4/segmented_output.mp4`

## Configuration

- Encoder: open x264 CRF23 (`libx264`, `preset=medium`, no forced GOP/scenecut/profile/level)
- Mask fill: `--fill-mode inpaint --fill-inpaint-radius 5 --fill-inpaint-method telea`
- Edge smoothing: `--feather-px 4`
- Matcher/path: same FC5 latent-key settings as the open x264 color-period tests

## Results

| Run | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Avg masking ms | Avg no-encode total ms |
|---|---:|---:|---:|---:|---:|---:|
| period 200, global dominant color | 9.31 | 8.90 | 0.8940 | 33.31 | 10.271 | 32.547 |
| period 2, global dominant color | 8.78 | 8.38 | 0.8955 | 33.41 | 10.077 | 32.151 |
| period 2, local dominant color, pad 100 | 8.65 | 8.25 | 0.8954 | 33.26 | 10.136 | 31.535 |
| period 1, local dominant color, pad 100 | 9.01 | 8.61 | 0.8953 | 33.26 | 10.687 | 33.574 |
| inpaint fill + feather 4 | -2.13 | -2.53 | 0.8989 | 33.85 | 227.055 | 248.531 |

## Interpretation

Using a neighboring region is feasible, and inpaint is the strongest version of that idea because it fills the masked pixels directly from surrounding texture rather than reducing the neighborhood to one color. Visually, it should merge better with the environment than the solid-color variants.

However, it is a poor bandwidth default for FC5 under this encoder. The segmented stream becomes larger than the original stream (`-2.13%` video-only BSP), and masking time rises from about `10 ms/frame` to `227 ms/frame`. The recovered quality improves slightly, but the cost is too high for the current compression claim.

This suggests the issue is not just the sampling window size. A single dominant color is codec-friendly but visually flat; full inpaint is visually better but introduces high-frequency texture that x264 spends bits preserving. A useful next candidate is a middle ground: sample local context, then use a low-frequency local fill such as ROI-local blur/median color, still with `--feather-px 4`.

Raw CSV:

`record/RESPAWN2026/gop_analysis/fc5_00_inpaint_feather4_summary.csv`
