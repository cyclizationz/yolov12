# FC5 Reencode Verification

This verifies the mentor-style reencode using the current frame dump in `record/RESPAWN2026/gop_analysis/original` and `masked` with final3 FC5 AV1 settings: `libsvtav1`, CRF 42, GOP 60, no B-frames, no scenecut, `yuv420p`.

## Whole-Video Results

| Subset | Frames | Original packet bytes | Masked packet bytes | Video BSP (%) | Negative frame count (%) | Weighted negative contribution (%) |
|---|---:|---:|---:|---:|---:|---:|
| all | 1802 | 11461750 | 10627199 | 7.28 | 35.52 | 18.92 |
| odd | 901 | 9445805 | 8684225 | 8.06 | 24.86 | 14.09 |

## Comparison To Mentor Numbers

- Mentor all-frame whole-video BSP: `10.24%`; this reproduction: `7.28%` packet bytes (`7.27%` file size).
- Mentor odd-frame whole-video BSP: `10.29%`; this reproduction: `8.06%` packet bytes (`8.06%` file size).
- Mentor CDF crossing values (`17.7%`, `15.7%`) are much closer to a **byte-weighted negative contribution share** than to raw negative-frame count. In this reproduction, weighted negative shares are `18.91%` for all frames and `14.10%` for odd frames.
- Raw negative-frame count is larger: `35.52%` for all frames and `24.86%` for odd frames. Many negative frames have tiny original packets, so counting frames and weighting bytes tell different stories.

## Most Negative Frames

The most negative frame by percentage can be misleading when the original packet is tiny. The tables below should be read together with `*_most_negative_frames.csv` under `fc5_reencode_verify/metrics/`.

### all

| Encoded frame | Source frame | Saving (%) | Baseline bytes | Masked bytes | SSIM | MAD |
|---:|---:|---:|---:|---:|---:|---:|
| 216 | 216 | -6871.08 | 83 | 5786 | 0.712685 | 17.829 |
| 1020 | 1020 | -4576.74 | 43 | 2011 | 0.961870 | 2.789 |
| 36 | 36 | -3853.01 | 83 | 3281 | 0.999738 | 1.042 |
| 224 | 224 | -3498.59 | 71 | 2555 | 0.685068 | 18.366 |
| 1252 | 1252 | -2865.26 | 95 | 2817 | 0.999630 | 0.737 |

### odd

| Encoded frame | Source frame | Saving (%) | Baseline bytes | Masked bytes | SSIM | MAD |
|---:|---:|---:|---:|---:|---:|---:|
| 396 | 791 | -7108.33 | 96 | 6920 | 0.999771 | 0.973 |
| 400 | 799 | -3052.43 | 103 | 3247 | 0.250508 | 39.692 |
| 356 | 711 | -1159.34 | 91 | 1146 | 0.999729 | 1.095 |
| 20 | 39 | -965.98 | 97 | 1034 | 0.741447 | 20.363 |
| 816 | 1631 | -870.00 | 90 | 873 | 0.813011 | 14.091 |

## Interpretation

- The current reproduction confirms the direction of the mentor result: re-encoding the frame dump with AV1/CRF gives much better savings than the fixed-bitrate Exp1 encoded clip.
- It does **not** exactly reproduce the mentor whole-video BSP (`10.24/10.29%`). The remaining gap likely comes from encoder invocation details, input frame set, or accounting definition.
- Odd-frame savings being close to all-frame savings supports the earlier conclusion that simple repeated-frame duplication is not the main cause.
- This strengthens the encoder-protocol diagnosis: RESPAWN looks much better under quality-targeted CRF/AV1 than under fixed-target x264/VBV because the encoder is allowed to turn easier masked content into fewer bytes.
