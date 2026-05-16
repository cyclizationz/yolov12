# GOP Tail Analysis

GOP BSP is computed as `1 - (RESPAWN video bytes + MSK1 bytes) / baseline video bytes` over 60-frame bins.

| Selection | Game | Clip | Rate (Mbps) | GOP ID | Local GOP | Frames | Baseline bytes | RESPAWN+MSK1 bytes | GOP BSP (%) | Dumped frames |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| most_negative | fc5 | fc5_00 | 20.0 | 104 | 11 | 661-720 | 2257995 | 2683118 | -18.83 | 60 |
| negative | fc5 | fc5_01 | 12.0 | 211 | 25 | 1501-1560 | 1362768 | 1591058 | -16.75 | 60 |
| best_positive | fc5 | fc5_04 | 8.0 | 631 | 11 | 661-720 | 921734 | 772037 | 16.24 | 60 |
| most_negative | fm6 | fm6_03 | 24.0 | 397 | 97 | 5821-5880 | 1160832 | 1749341 | -50.70 | 60 |
| negative | fm6 | fm6_04 | 8.0 | 411 | 11 | 661-720 | 997003 | 1151606 | -15.51 | 60 |
| best_positive | fm6 | fm6_04 | 20.0 | 497 | 4 | 241-300 | 2632734 | 2094029 | 20.46 | 60 |
| most_negative | mario | mario_01 | 16.0 | 149 | 16 | 961-1020 | 57377 | 111358 | -94.08 | 60 |
| negative | mario | mario_04 | 8.0 | 383 | 3 | 181-240 | 415663 | 605900 | -45.77 | 60 |
| best_positive | mario | mario_00 | 24.0 | 76 | 0 | 1-60 | 3179104 | 2519660 | 20.74 | 60 |
