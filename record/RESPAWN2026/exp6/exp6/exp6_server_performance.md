# Exp6 Server Performance Smoke Test

This run uses the `RespawnOnlineServer` scaffold on one representative normalized clip per game. Each run is capped at 300 frames and uses CRF23/open-GOP-style encoder settings (`libx264`, `medium`, `tune=none`, profile/level unset). The latest rerun enables `--server-pipeline-depth 4` and `--server-post-parallel 1` for the YOLO path.

The timing columns below are system component sums from server-side frame detection through client-side stitching. They exclude benchmark-only quality metrics, MP4 write/encode proxy time, and report bookkeeping.

## Summary

| Game | Clip | Frames | Avg system ms | P95 system ms | Effective FPS | Server BSP | RSEI bytes | Templates | Recovered SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | `fc5_00` | 300 | 30.040 | 55.641 | 33.29 | 4.41% | 14176 | 211 | 0.9106 |
| FM6 | `fm6_00` | 300 | 55.652 | 85.665 | 17.97 | 15.13% | 16085 | 150 | 0.9515 |
| Mario | `mario_00` | 300 | 10.322 | 32.560 | 96.88 | 39.33% | 48900 | 0 | 0.9113 |

## Component Split

| Game | Detect ms | Detect prep ms | Detector core ms | Detector post ms | Dict/match ms | Mask/fill ms | Client stitch ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 7.700 | 1.633 | 4.990 | 1.076 | 7.741 | 9.838 | 4.761 |
| FM6 | 7.620 | 1.375 | 4.978 | 1.267 | 13.348 | 23.444 | 11.239 |
| Mario | 10.168 | 0.000 | 10.168 | 0.000 | 0.000 | 0.041 | 0.114 |

## Interpretation

- FC5 measures 30.040 ms/frame (33.29 effective FPS) for the detection-to-stitching system path; old benchmark wall total was 198.077 ms/frame.
- FM6 measures 55.652 ms/frame (17.97 effective FPS) for the detection-to-stitching system path; old benchmark wall total was 217.264 ms/frame.
- Mario measures 10.322 ms/frame (96.88 effective FPS) for the detection-to-stitching system path; old benchmark wall total was 73.607 ms/frame.

## Critical Conclusion

The YOLO detector is no longer the dominant latency limiter in the optimized scaffold. Detector core time is about 5 ms/frame for FC5/FM6, and a separate 1-4 worker CUDA-session sweep shows no meaningful improvement from higher inference concurrency; the practical bottleneck is the downstream system path, especially dictionary/template matching, mask/fill work, client stitching, and non-system benchmark wall overhead.

## Generated Artifacts

- CSV: `record/RESPAWN2026/exp6/exp6_server_performance.csv`
- Resource/component summary: `record/RESPAWN2026/exp6/exp6_resource_component_summary.md`
- Resource/component CSV: `record/RESPAWN2026/exp6/exp6_resource_component_summary.csv`
- Inference worker sweep: `record/RESPAWN2026/exp6/infer_worker_sweep/infer_worker_sweep.md`
- Runs: `record/RESPAWN2026/exp6/resource_runs/{fc5_00,fm6_00,mario_00}/`
- Each run contains `original_output.mp4`, `segmented_output.mp4`, `recovered_output.mp4`, `msk1_payloads.bin`, `dict/`, and `report.json`.

## Client Contract Check

`RespawnOnlineClient --strict` passed on all three refreshed run directories with zero RSEI/MSK1 parse failures.
