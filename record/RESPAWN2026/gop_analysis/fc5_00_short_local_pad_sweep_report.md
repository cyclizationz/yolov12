# FC5_00 Short Local Flat-Color Pad Sweep

This sweep keeps flat-color masking and `--feather-px 4`, then changes the local sampling pad used for per-region dominant color. `pad16` matches the H.264 macroblock size.

| Run | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Avg masking ms | Output |
|---|---:|---:|---:|---:|---:|---|
| single local color per frame, pad100 | 4.22 | 3.85 | 0.9100 | 47.73 | 7.988 | `record/RESPAWN2026/gop_analysis/fc5_00_short_frame_flat_pad100_feather4/segmented_output.mp4` |
| per-region local color, pad16 | 3.77 | 3.40 | 0.9103 | 47.57 | 8.270 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad16_feather4/segmented_output.mp4` |
| per-region local color, pad50 | 3.94 | 3.56 | 0.9103 | 47.64 | 8.313 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad50_feather4/segmented_output.mp4` |
| per-region local color, pad100 | 4.21 | 3.84 | 0.9102 | 47.72 | 8.303 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad100_feather4/segmented_output.mp4` |

## Notes

The byte/quality metrics are nearly identical across pad sizes on this 300-frame prefix, so the best pad should be selected by visual naturalness in the problematic frames. Smaller pads avoid mixing unrelated background surfaces; `pad16` is the most encoder-aligned choice for H.264, while `pad50` is a compromise if `pad16` is too noisy or samples too few pixels.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_short_local_pad_sweep_summary.csv`
