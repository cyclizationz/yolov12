# FC5_00 Short Pad50 Mode vs Median Local Color

Both runs use per-region flat color, local pad `50px`, solid fill, `--feather-px 4`, open x264 CRF23, and the same 300-frame prefix. The mode run uses the most common quantized BGR histogram bin; the median run uses per-channel medians over the same local samples.

| Run | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Avg masking ms | Output |
|---|---:|---:|---:|---:|---:|---|
| pad50 most-common/mode | 3.94 | 3.56 | 0.9103 | 47.64 | 8.313 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad50_feather4/segmented_output.mp4` |
| pad50 median | 3.36 | 2.99 | 0.9125 | 47.93 | 9.321 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad50_median_feather4/segmented_output.mp4` |

## Notes

The current default local sampler was already most-common/mode, not average. Median is less likely to be dominated by one repeated texture/color bin, but it may create a blended-looking flat color that is not actually present in the local background. The metric difference is small, so visual inspection should decide whether median better matches the boundary-crossing gun frames.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_short_pad50_mode_vs_median_summary.csv`
