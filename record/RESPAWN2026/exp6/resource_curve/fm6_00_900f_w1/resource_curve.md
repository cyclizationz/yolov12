# Exp6 Longer-Clip Resource Curve

This run samples host process CPU/RSS and GPU telemetry over elapsed time while processing a longer FM6 clip segment.

![Resource usage curve](resource_curve.svg)

## Run

- Clip: `fm6_00`
- Frames: `900`
- Wall time: `199.48` s
- Server pipeline depth: `4`
- Inference workers: `1`

## Average Resource Usage

| CPU % | RSS MB | GPU % | GPU Mem MB | GPU W |
| ---: | ---: | ---: | ---: | ---: |
| 142.1 | 2203.8 | 4.3 | 3834.2 | 46.6 |

## Conclusion

The longer run keeps CPU activity sustained while GPU utilization remains low and bursty, which supports the earlier conclusion that the current bottleneck is not raw YOLO GPU occupancy but downstream CPU/memory and benchmark-side work.

- CSV: `record/RESPAWN2026/exp6/resource_curve/fm6_00_900f_w1/resource_curve.csv`
- Report: `record/RESPAWN2026/exp6/resource_curve/fm6_00_900f_w1/report.json`
