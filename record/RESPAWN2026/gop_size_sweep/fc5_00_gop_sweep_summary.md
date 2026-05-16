# FC5_00 GOP Size Sweep

20 Mbps, Exp1 settings, only `--enc-gop` changed. Video metrics skipped; table is byte/per-frame accounting only.

| enc_gop | baseline MB | respawn video MB | MSK1 MB | respawn total MB | total BSP % | negative 60f bins | worst 60f BSP % | p10 60f BSP % | median 60f BSP % | best 60f BSP % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 77.64 | 77.54 | 0.093 | 77.63 | 0.01 | 16/31 | -13.52 | -9.71 | -0.00 | 10.89 |
| 30 | 77.40 | 77.15 | 0.093 | 77.24 | 0.21 | 17/31 | -13.76 | -10.17 | -0.72 | 13.48 |
| 60 | 77.21 | 76.55 | 0.093 | 76.65 | 0.73 | 15/31 | -16.87 | -11.33 | 1.31 | 14.21 |
| 120 | 77.00 | 76.19 | 0.093 | 76.28 | 0.94 | 13/31 | -15.00 | -12.37 | 1.60 | 14.14 |
| 240 | 76.76 | 75.90 | 0.093 | 75.99 | 1.01 | 11/31 | -15.72 | -12.49 | 2.53 | 14.07 |
