# Pixel + encoder sweep summary (Mario)

- Anchor bitrate: 16.0 Mbps (maxrate=buf matched to Exp1 style)
- Encoder: `--enc-preset medium --enc-tune none`
- VMAF floor used for gating: 55.0
- VMAF (`--vmaf-mode segmented_vs_original`): `segmented_output.mp4` vs same-run `original_output.mp4` (masking effect at matched encode)

- **Delivered bitrate (bandwidth):** `ffprobe` bitrate of `segmented_output.mp4` plus `msk1_payloads.bin` bits/frame. **Pure baseline** uses `encode_pure_streaming_baseline.py`: copies the full encode to `segmented_output.mp4` (same bytes as `original_output.mp4`) with an empty `msk1_payloads.bin`. **Savings vs pure:** `(1 - delivered_respawn / delivered_pure) × 100%`.

## Stage 0 (mario_00, all configs)
- CSV: `/home/tiehangz/proj/yolov12/record/RESPAWN2026/pixel_enc_sweep/leaderboard_stage0.csv`

## Stage 1 (VMAF on top-K by coverage)
- See `leaderboard_stage1.csv`

## Stage 2 (winner configs × 5 clips)
- See `leaderboard_stage2.csv` (also copied to `leaderboard.csv`).

## Recommended winners
- No config passed VMAF min in stage1 top-K; inspect CSVs or relax floor.

## Best-effort leader (Stage1 top-K, **does not** meet VMAF floor)
- **cfg_id:** `p90_b15_f1_k0.80_og1`
- mean changed_pixels_pct: 10.072
- saving % vs pure: 19.74
- vmaf_mean: 48.3840464052631 (VMAF_mean ≥ 55.0 required)

## Stage 2 (cross-clip; worst clip per cfg)
- `p120_b15_f1_k0.80_og1`: worst vmaf_mean=48.384, worst saving %=-0.21 (5 clips)
- `p60_b15_f1_k0.80_og1`: worst vmaf_mean=48.384, worst saving %=-0.18 (5 clips)
- `p90_b15_f1_k0.80_og1`: worst vmaf_mean=48.384, worst saving %=-0.18 (5 clips)
