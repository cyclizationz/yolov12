# Mario CRF23 path ablation (5-clip means)

| Strategy | Clips | BSP (%) | Recovered SSIM | Recovered PSNR | Artifact frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full pipeline | 5 | 30.2 | 0.918 | 20.57 | 1095 |
| No optical flow | 5 | 25.6 | 0.928 | 33.59 | 936 |
| No Kalman | 5 | 30.2 | 0.918 | 20.58 | 1094 |
| Template only | 5 | 30.9 | 0.921 | 26.47 | 1033 |
