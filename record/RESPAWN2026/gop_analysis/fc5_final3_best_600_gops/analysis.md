# FC5 Final3 Best First-600 GOP Analysis

- Source run: `record/RESPAWN2026/gop_analysis/fc5_final3_best_600`
- Parameters: final3 best FC5 `libsvtav1`, CRF 42, GOP 60, no B-frames, no scenecut, latent threshold 0.86, dominant fill, feather 4.
- Frames analyzed: `600`
- Whole 600-frame video-only BSP: `5.09%`
- Whole 600-frame net BSP incl. MSK1: `4.67%`
- Best local GOP net BSP: GOP `0` = `8.71%`
- Worst local GOP net BSP: GOP `6` = `1.47%`

## GOP Table

| GOP | Frames | Video BSP (%) | Net BSP (%) | Avg Changed (%) | Masked/Present (%) |
|---:|---|---:|---:|---:|---:|
| 0 | 1-60 | 9.28 | 8.71 | 21.02 | 100.00 |
| 1 | 61-120 | 6.34 | 6.00 | 7.06 | 93.48 |
| 2 | 121-180 | 5.40 | 5.04 | 9.88 | 100.00 |
| 3 | 181-240 | 3.45 | 3.01 | 5.96 | 83.87 |
| 4 | 241-300 | 3.89 | 3.28 | 14.01 | 100.00 |
| 5 | 301-360 | 3.74 | 3.17 | 19.69 | 100.00 |
| 6 | 361-420 | 1.93 | 1.47 | 12.80 | 90.74 |
| 7 | 421-480 | 5.98 | 5.69 | 9.28 | 86.84 |
| 8 | 481-540 | 4.56 | 4.20 | 15.07 | 94.44 |
| 9 | 541-600 | 5.58 | 5.19 | 29.76 | 100.00 |

## Prefix / Part-Video Table

| Through GOP | Frames | Video BSP (%) | Net BSP (%) |
|---:|---|---:|---:|
| 0 | 1-60 | 9.28 | 8.71 |
| 1 | 1-120 | 7.52 | 7.08 |
| 2 | 1-180 | 6.75 | 6.34 |
| 3 | 1-240 | 6.07 | 5.66 |
| 4 | 1-300 | 5.72 | 5.28 |
| 5 | 1-360 | 5.43 | 4.97 |
| 6 | 1-420 | 4.93 | 4.46 |
| 7 | 1-480 | 5.10 | 4.67 |
| 8 | 1-540 | 5.03 | 4.60 |
| 9 | 1-600 | 5.09 | 4.67 |

## Interpretation

The individual GOPs are still local packet windows, so they swing more than the part-video/whole-video numbers. In this final3 CRF run, however, even every local GOP is positive; as prefixes accumulate, the estimate becomes steadier and ends at the whole 600-frame result.
