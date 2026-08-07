# Exp6 Resource and Component Timing Summary

These numbers come from rerunning the same 300-frame `RespawnOnlineServer` samples under `/usr/bin/time -v` while sampling `nvidia-smi`.

Driver/extraction fix: resource runs are now collected in CUDA mode (no `--cpu`) with live `nvidia-smi` sampling.

## Host Parameters

| CPU model | Logical cores | GPU | Driver | CUDA | GPU memory MiB | GPU power limit W |
| --- | ---: | --- | --- | --- | ---: | ---: |
| AMD Ryzen 7 9800X3D 8-Core Processor | 16 | NVIDIA GeForce RTX 5070 Ti | 595.71.05 | 13.2 | 16303 | 300.00 |

## Run Parameters

- `max_frames`: 300
- `encoder`: `libx264` / CRF `23` / preset `medium` / tune `none`
- `open_gop_defaults`: `True`
- `server_pipeline`: `1`
- `server_pipeline_depth`: `4`
- `server_post_parallel`: `1`
- `gpu_sampling`: `nvidia-smi` every ~0.5 s

## Resource Usage

| Game | CPU % | Max RAM MB | GPU telemetry | Avg GPU % | Max GPU % | Avg GPU Mem MB | Max GPU Mem MB | Avg GPU W | Wall time |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 187 | 1440.6 | ok | 4.5 | 8.0 | 3554.7 | 3854.0 | 47.0 | 1:01.10 |
| FM6 | 206 | 2088.0 | ok | 5.0 | 9.0 | 3626.0 | 3831.0 | 49.1 | 1:06.79 |
| Mario | 213 | 367.6 | ok | 3.3 | 5.0 | 2994.0 | 3173.0 | 37.5 | 0:22.90 |

## GPU Usage

| Game | Samples | Avg GPU % | Max GPU % | Avg GPU Mem MB | Max GPU Mem MB | Avg GPU W | Max GPU W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 133 | 4.5 | 8.0 | 3554.7 | 3854.0 | 47.0 | 49.4 |
| FM6 | 134 | 5.0 | 9.0 | 3626.0 | 3831.0 | 49.1 | 51.0 |
| Mario | 46 | 3.3 | 5.0 | 2994.0 | 3173.0 | 37.5 | 48.0 |

## System Component Time Cost

| Game | Avg system ms | P95 system ms | Eff. FPS | Detect ms | Detect prep ms | Detector core ms | Detector post ms | Dict/match ms | Mask/fill ms | Client stitch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 30.040 | 55.641 | 33.29 | 7.700 | 1.633 | 4.990 | 1.076 | 7.741 | 9.838 | 4.761 |
| FM6 | 55.652 | 85.665 | 17.97 | 7.620 | 1.375 | 4.978 | 1.267 | 13.348 | 23.444 | 11.239 |
| Mario | 10.322 | 32.560 | 96.88 | 10.168 | 0.000 | 10.168 | 0.000 | 0.000 | 0.041 | 0.114 |

## Critical Conclusion

The latency problem after queue-depth and CPU post-processing parallelization is not raw GPU inference. FC5/FM6 detector core time is only about 5 ms/frame, and the inference-worker sweep shows that 2-4 concurrent CUDA sessions do not improve system or wall time; future optimization should target dictionary/template matching, mask/fill, client stitching, memory movement, and removal of benchmark-only encode/quality overhead from the online path.

## Notes

- System time is the sum of measured pipeline components from server-side detection through client-side stitching; it excludes quality metrics, MP4 write/encode proxy time, report bookkeeping, and other benchmark-only wall-clock work.
- `Detector post ms` is detector post-processing only; dictionary/template matching is split into `Dict/match ms` to avoid double-counting.
- CPU `%` is process CPU from `/usr/bin/time -v`; values above 100% mean multiple cores were active.
- Max RAM is maximum resident set size from `/usr/bin/time -v`.
- GPU metrics come from periodic `nvidia-smi` sampling during each run.
- CSV: `/home/tiehangz/proj/yolov12/record/RESPAWN2026/exp6/exp6_resource_component_summary.csv`
