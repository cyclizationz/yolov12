# FC5_00 Local Dominant Color Test

Implemented `--mask-color-local-pad <px>` for `mask-color=dominant`.

When enabled, the dominant color is sampled from expanded detection neighborhoods instead of the whole frame. For this test, `--mask-color-local-pad 100` samples each detected box expanded by 100 px and excludes detected mask pixels from the histogram. If no local samples exist, it falls back to the whole-frame dominant color.

## Output To Inspect

`record/RESPAWN2026/gop_analysis/fc5_00_open_x264_color_period_2_localpad100/segmented_output.mp4`

## Results

| Run | Video BSP (%) | Net BSP (%) | Avg masking ms | Avg no-encode total ms | Avg masked SSIM | Avg recovered SSIM |
|---|---:|---:|---:|---:|---:|---:|
| period 200, global dominant | 9.31 | 8.90 | 10.271 | 32.547 | 0.8923 | 0.8940 |
| period 2, global dominant | 8.79 | 8.39 | 10.077 | 32.151 | 0.8948 | 0.8955 |
| period 2, local pad 100 | 8.66 | 8.25 | 10.136 | 31.535 | 0.8981 | 0.8954 |

## Interpretation

Local neighborhood sampling changes the mask color in the expected direction visually: the masked-frame SSIM improves from `0.8948` to `0.8981` relative to period-2 global dominant color. However, it does not improve bandwidth saving on this clip; net BSP drops from `8.39%` to `8.25%`, and remains below the period-200 global baseline (`8.90%`).

This suggests local color may help perceptual/color matching in problematic frames, but the encoder does not reward it here. The next visual check should inspect whether the frames you noticed are fixed; if yes, this might be a quality tradeoff rather than a bandwidth optimization.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_local_color_pad100_summary.csv`
