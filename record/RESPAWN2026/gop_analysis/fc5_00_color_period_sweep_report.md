# FC5_00 Adaptive Color Period Sweep

This tests whether recomputing the dominant mask color more frequently helps when using the open exploratory encoder setting:

```bash
--enc-codec libx264 --enc-crf 23 --enc-preset medium --enc-tune none \
--enc-profile none --enc-level none --enc-open-gop-defaults \
--enc-aud 1 --enc-repeat-headers 0
```

All runs use the same `fc5_00` offline processor path and differ only in `--mask-color-period`.

## Results

| Mask color period | Video BSP (%) | Net BSP (%) | Delta net BSP vs 200 | Avg masking ms/frame | Delta masking ms | Avg total no-encode ms | Delta total no-encode ms | Avg recovered SSIM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 9.31 | 8.90 | +0.00 | 10.271 | +0.000 | 32.547 | +0.000 | 0.8940 |
| 2 | 8.79 | 8.39 | -0.52 | 10.077 | -0.195 | 32.151 | -0.396 | 0.8955 |
| 5 | 8.94 | 8.54 | -0.36 | 10.131 | -0.140 | 32.216 | -0.330 | 0.8955 |
| 10 | 9.06 | 8.66 | -0.25 | 10.118 | -0.154 | 31.759 | -0.788 | 0.8956 |

## Interpretation

- Recomputing dominant color more frequently did **not** increase bandwidth saving on this FC5_00 clip.
- The best result remains the existing `mask-color-period=200` baseline: `9.31%` video BSP and `8.90%` net BSP.
- Period 10 is closest but still lower by `0.25` percentage points net; periods 5 and 2 are lower by `0.36` and `0.52` points.
- Timing does not show an extra cost from smaller periods in this run. The measured masking and total no-encode times are slightly lower for smaller periods, which is more likely run-to-run noise than a true speedup.
- Quality is slightly higher for smaller periods (`avg_recovered_ssim` around `0.8955` vs `0.8940`), so faster color adaptation may improve local visual matching a little, but it does not translate into more byte savings here.

## Raw Data

- CSV: `record/RESPAWN2026/gop_analysis/fc5_00_color_period_sweep.csv`
- Output folders:
  - period 200: `record/RESPAWN2026/gop_analysis/fc5_00_offline_open_x264_crf23`
  - period 2: `record/RESPAWN2026/gop_analysis/fc5_00_open_x264_color_period_2`
  - period 5: `record/RESPAWN2026/gop_analysis/fc5_00_open_x264_color_period_5`
  - period 10: `record/RESPAWN2026/gop_analysis/fc5_00_open_x264_color_period_10`
