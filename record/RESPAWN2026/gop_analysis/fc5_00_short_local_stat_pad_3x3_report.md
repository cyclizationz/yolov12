# FC5_00 Short Local Color Statistic/Pad Sweep

All runs use the same 300-frame FC5 prefix, per-region flat-color fill, `--feather-px 4`, and open x264 CRF23. Pads are `16`, `32`, and `64` pixels: 1x/2x/4x the H.264 macroblock size.

| Statistic | Pad px | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Masking ms | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| mode | 16 | 3.77 | 3.40 | 0.9103 | 47.57 | 8.270 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad16_feather4/segmented_output.mp4` |
| mode | 32 | 3.65 | 3.28 | 0.9102 | 47.50 | 8.210 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad32_mode_feather4/segmented_output.mp4` |
| mode | 64 | 4.08 | 3.70 | 0.9101 | 47.64 | 8.276 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad64_mode_feather4/segmented_output.mp4` |
| median | 16 | 3.41 | 3.03 | 0.9125 | 47.92 | 9.230 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad16_median_feather4/segmented_output.mp4` |
| median | 32 | 3.44 | 3.07 | 0.9125 | 47.93 | 9.297 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad32_median_feather4/segmented_output.mp4` |
| median | 64 | 3.48 | 3.11 | 0.9125 | 47.94 | 9.492 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad64_median_feather4/segmented_output.mp4` |
| average | 16 | 3.15 | 2.77 | 0.9125 | 47.94 | 8.020 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad16_average_feather4/segmented_output.mp4` |
| average | 32 | 3.55 | 3.18 | 0.9125 | 47.96 | 8.045 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad32_average_feather4/segmented_output.mp4` |
| average | 64 | 3.55 | 3.18 | 0.9125 | 47.99 | 8.111 | `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad64_average_feather4/segmented_output.mp4` |

## BSP Matrix

| Statistic | 16px | 32px | 64px |
|---|---:|---:|---:|
| mode | 3.77% | 3.65% | 4.08% |
| median | 3.41% | 3.44% | 3.48% |
| average | 3.15% | 3.55% | 3.55% |

## Notes

Best video-only BSP in this short sweep is `mode` at `64px`: `4.08%`. Best net BSP is `mode` at `64px`: `3.70%`.

Mode preserves the strongest compression because it snaps to a repeated local color bin. Median/average are smoother estimates and may look less wrong in boundary-crossing frames, but they generally reduce BSP because the chosen color can vary more continuously over time and may not align to a repeated quantized surface.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_short_local_stat_pad_3x3_summary.csv`
