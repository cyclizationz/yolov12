# FC5_00 Full Local Color Statistic Test (Pad 64)

All runs use the full `fc5_00` clip (1802 frames), per-region flat-color fill, `--mask-color-local-pad 64`, `--feather-px 4`, and open x264 CRF23.

| Statistic | Video BSP (%) | Net BSP (%) | Original bytes | Segmented bytes | MSK1 bytes | Recovered SSIM | Recovered PSNR | Masking ms | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mode | 8.91 | 8.50 | 23125638 | 21065985 | 92961 | 0.8955 | 33.28 | 10.535 | `record/RESPAWN2026/gop_analysis/fc5_00_full_per_region_flat_pad64_mode_feather4/segmented_output.mp4` |
| median | 8.02 | 7.62 | 23125638 | 21271040 | 92961 | 0.8967 | 33.52 | 12.270 | `record/RESPAWN2026/gop_analysis/fc5_00_full_per_region_flat_pad64_median_feather4/segmented_output.mp4` |
| average | 8.18 | 7.77 | 23125638 | 21235079 | 92961 | 0.8967 | 33.54 | 10.238 | `record/RESPAWN2026/gop_analysis/fc5_00_full_per_region_flat_pad64_average_feather4/segmented_output.mp4` |

## Notes

Best full-clip video BSP is `mode` at `8.91%` video-only and `8.50%` including MSK1.

Compared with the 300-frame prefix, the full clip is much more favorable for all three statistics, which means the short prefix was not representative of the whole clip's savings. Mode still gives the strongest BSP because it chooses a repeated quantized color bin, while median/average give slightly higher recovered quality but lower byte savings.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_full_local_stat_pad64_summary.csv`
