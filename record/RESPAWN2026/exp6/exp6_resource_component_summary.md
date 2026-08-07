# Exp6 Resource and Component Timing Summary

These numbers come from rerunning the same 300-frame `RespawnOnlineServer` samples under `/usr/bin/time -v` while sampling `nvidia-smi`.

Driver/extraction fix: resource runs are now collected in CUDA mode (no `--cpu`) with live `nvidia-smi` sampling.

## Host Parameters

| CPU model | Logical cores | GPU | Driver | CUDA | GPU memory MiB | GPU power limit W |
| --- | ---: | --- | --- | --- | ---: | ---: |
| AMD Ryzen 7 9800X3D 8-Core Processor | 16 | unknown | unknown | unknown | unknown | unknown |

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

## RESPAWN-Exclusive Processing Cost

| Game | Avg path ms | P95 path ms | Eff. FPS | Detect ms | Detect prep ms | Detector core ms | Detector post ms | Dict/match ms | Mask/fill ms | Stitch mean ms | Stitch p50 ms | Stitch p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 30.040 | 55.641 | 33.29 | 7.700 | 1.633 | 4.990 | 1.076 | 7.741 | 9.838 | 4.761 | 5.743 | 10.398 |
| FM6 | 55.652 | 85.665 | 17.97 | 7.620 | 1.375 | 4.978 | 1.267 | 13.348 | 23.444 | 11.239 | 11.413 | 11.791 |
| Mario | 10.322 | 32.560 | 96.88 | 10.168 | 0.000 | 10.168 | 0.000 | 0.000 | 0.041 | 0.114 | 0.113 | 0.126 |

## Critical Conclusion

The latency problem after queue-depth and CPU post-processing parallelization is not raw GPU inference. FC5/FM6 detector core time is only about 5 ms/frame, and the inference-worker sweep shows that 2-4 concurrent CUDA sessions do not improve system or wall time; future optimization should target dictionary/template matching, mask/fill, client stitching, memory movement, and removal of benchmark-only encode/quality overhead from the online path.

## Longer-Clip Resource Curve

A 900-frame FM6 run (`record/RESPAWN2026/exp6/resource_curve/fm6_00_900f_w1/resource_curve.md`) records CPU/RSS and GPU telemetry over elapsed time. The average CPU load is 142.1%, while GPU utilization averages 4.3%, reinforcing that the optimized scaffold is CPU/memory and benchmark-overhead limited rather than GPU-inference limited.

## Notes

- `Avg path` is the sum of RESPAWN-specific measured components from server-side detection through client-side stitching; it is not an end-to-end pipeline time.
- Common baseline work (video decode, display/frame presentation, and ordinary encode/transport) was not measured in this trace and is reported as N/A rather than folded into RESPAWN modules.
- `Stitch` is template lookup plus alpha compositing only; it is not total client display time.
- `Detector post ms` is detector post-processing only; dictionary/template matching is split into `Dict/match ms` to avoid double-counting.
- CPU `%` is process CPU from `/usr/bin/time -v`; values above 100% mean multiple cores were active.
- Max RAM is maximum resident set size from `/usr/bin/time -v`.
- GPU metrics come from periodic `nvidia-smi` sampling during each run.
- CSV: `/home/tiehangz/proj/yolov12/record/RESPAWN2026/exp6/exp6_resource_component_summary.csv`
