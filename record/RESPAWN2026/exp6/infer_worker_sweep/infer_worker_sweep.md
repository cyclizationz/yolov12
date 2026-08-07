# Exp6 Inference Worker Sweep

This sweep increases concurrent YOLO inference sessions in the server pipeline and measures component-summed system time from detection through client stitching.

![Inference worker sweep](infer_worker_sweep.svg)

| Game | Workers | Avg system ms | P95 system ms | Avg infer ms | Wall ms/frame | Avg GPU % | Max GPU % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 1 | 28.146 | 51.931 | 4.952 | 201.600 | 4.3 | 8.0 |
| FC5 | 2 | 27.992 | 52.482 | 5.034 | 204.000 | 4.9 | 8.0 |
| FC5 | 3 | 28.299 | 52.417 | 5.036 | 203.700 | 4.8 | 58.0 |
| FC5 | 4 | 28.115 | 53.028 | 5.117 | 205.567 | 4.8 | 12.0 |
| FM6 | 1 | 54.079 | 80.778 | 4.973 | 223.733 | 4.7 | 8.0 |
| FM6 | 2 | 54.233 | 82.646 | 4.964 | 224.633 | 4.1 | 8.0 |
| FM6 | 3 | 54.451 | 81.134 | 5.019 | 224.833 | 4.3 | 8.0 |
| FM6 | 4 | 54.295 | 80.159 | 5.061 | 225.067 | 4.5 | 8.0 |

## Critical Conclusion

Increasing concurrent YOLO CUDA sessions from 1 to 4 does not reduce end-to-end system latency on these clips. FC5 stays near 28 ms/frame and FM6 stays near 54 ms/frame, while model inference itself remains about 5 ms/frame; the remaining bottleneck is therefore downstream server/client processing, memory movement, template matching/masking/stitching, and benchmark-side wall work rather than raw GPU inference occupancy.

## Notes

- `workers` is the number of concurrent YOLO CUDA sessions feeding an ordered result queue.
- `Avg system ms` excludes MP4 write/proxy encode time and quality-metric bookkeeping.
- CSV: `record/RESPAWN2026/exp6/infer_worker_sweep/infer_worker_sweep.csv`
