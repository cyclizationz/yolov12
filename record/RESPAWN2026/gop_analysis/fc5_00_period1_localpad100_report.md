# FC5_00 Period-1 Local Color Test

This run uses local dominant color sampling with `--mask-color-period 1 --mask-color-local-pad 100` under the open x264 CRF23 exploratory encoder.

Output video:

`record/RESPAWN2026/gop_analysis/fc5_00_open_x264_color_period_1_localpad100/segmented_output.mp4`

## Results

| Run | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Avg masking ms | Avg no-encode total ms |
|---|---:|---:|---:|---:|---:|---:|
| period 200, global | 9.31 | 8.90 | 0.8940 | 33.31 | 10.271 | 32.547 |
| period 2, global | 8.79 | 8.39 | 0.8955 | 33.41 | 10.077 | 32.151 |
| period 2, local pad 100 | 8.66 | 8.25 | 0.8954 | 33.26 | 10.136 | 31.535 |
| period 1, local pad 100 | 9.01 | 8.61 | 0.8953 | 33.26 | 10.687 | 33.574 |

## Interpretation

Period 1 with local pad 100 improves bandwidth versus period 2 local pad 100 (`8.61%` vs `8.25%` net), but it still does not beat the global period-200 baseline (`8.90%` net). Recovered quality is essentially flat across the local-pad runs and slightly higher than period 200 by SSIM, but lower than period 2 global by PSNR.

Timing cost becomes visible at period 1: masking time rises to `10.687 ms/frame`, about `+0.415 ms/frame` over period 200 and `+0.550 ms/frame` over period 2 local pad 100.

Since we only care about reconstruction quality, the current evidence is: period 1 local color is not clearly better than period 2/global for recovery, and it costs more time while still saving less bandwidth than period 200 global.

Raw CSV: `record/RESPAWN2026/gop_analysis/fc5_00_period1_localpad100_summary.csv`
