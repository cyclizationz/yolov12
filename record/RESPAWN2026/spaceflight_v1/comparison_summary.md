# Spaceflight two-class RESPAWN evaluation

## Main results
- Validation: box mAP50-95 0.8515; mask mAP50-95 0.7880.
- Out-of-domain controls on the same labels: FC5 mask mAP50-95 0.0000; FM6 0.0007. FC5 predicts `gun`; FM6 is one-class `cockpit`; neither can predict `spaceship`.
- Clips: exactly 5, with 450/450 encoded frames model-positive.
- RD: mean delivered bitrate saving 5.79%, but mean VMAF falls from 98.68 to 73.45. BD-rate is unsupported because quality ranges do not overlap.
- Exp6-style server: mean system path 103.63 ms/frame, CPU 190.3%, average GPU utilization 3.6%.
- Duo screen: all 5 clips classified promising (mean Ref/Raw run length 90 frames).

## Interpretation
The new checkpoint is the only valid two-class model and strongly outperforms the out-of-domain FC5/FM6 checkpoints on the spaceflight labels. The current RESPAWN masking configuration saves some bits but does not preserve enough quality for an equal-quality or BD-rate claim. Cache-preload accounting is also unfavorable for these very short clips because template delivery dominates.

## Protocol and exclusions
- Completed: 300-epoch training/export/validation; every-frame clip verification; same-label FC5/FM6 controls; duo screen; 25-point RD; CRF23 GOP runs; overhead/cache-delay analysis; generality analysis; 50-run matching/fill ablation; five-clip server resource benchmark.
- Excluded: the 320-run gate hyperparameter grid as operationally broad; Mario pixel-only experiments as inapplicable; thin-client and inference-worker repetition as model-independent plumbing tests.
- Native FC5/FM6 resource figures are retained only as context because their clips, resolution, frame count, and object workloads differ.

Machine-readable summary: `/home/tiehangz/proj/yolov12/record/RESPAWN2026/spaceflight_v1/comparison_summary.json`
