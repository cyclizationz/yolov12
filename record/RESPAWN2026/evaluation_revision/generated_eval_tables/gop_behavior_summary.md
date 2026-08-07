# Actual H.264 GOP behavior

GOP boundaries are baseline-stream keyframes reported by ffprobe; they are not arbitrary 60-frame windows.
RESPAWN bytes are aligned to those frame intervals and include RMD.

- FC5: 89 GOPs, mean +5.86% BSP, 12.4% negative.
- FM6: 120 GOPs, mean +12.47% BSP, 8.3% negative.
- Mario: 31 GOPs, mean +18.43% BSP, 9.7% negative.

Spearman correlations with GOP BSP:
- rec_fraction: rho=+0.595
- reuse_fraction: rho=+0.339
- mask_coverage_pct: rho=+0.493
- frames: rho=+0.515
- unique_templates: rho=+0.223

Interpretation rule: only discuss features with a visible monotonic effect and |rho| >= 0.2. If H.264 keyframe/scenecut placement dominates and no feature meets that threshold, the paper should call the ordering codec-dependent rather than inventing a causal GOP taxonomy. A documented FC5-only SVT-AV1 CRF42/GOP60 diagnostic exists, but there is no matched 15-clip AV1 suite; it can be used as a supplement, not as a replacement for a matched codec-generality experiment.
