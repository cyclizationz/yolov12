# Exp3 ablation raw workbook (Table 10-13)

`exp3_ablation_tables_10_13.xlsx` stores raw frame-level data from run artifacts (`report.json` per-frame arrays + ffmpeg stderr logs).

## Raw source mapping

- `table10`: no-heal feather sweep at `record/RESPAWN2026/exp3/feather_sweep_crf23_open_noheal/{fc5_00,fm6_00}/feather_{0,4,8,16}`.
- `table11`: no-heal fill variants at `record/RESPAWN2026/exp3/fill_fc5_fm6_crf23_open_noheal/{fc5_00,fm6_00}/{black,dominant,inpaint}`, plus Mario aggregate rows from `record/RESPAWN2026/exp3/fill_mario_crf23_open/mario_agg/{dominant,reference_color}`; `blur` is marked missing.
- `table12`: no-heal matching ablation from `record/RESPAWN2026/exp3/exp3_artifact_churn_crf23_open_noheal/{fc5_00,fm6_00}/{latent_key,phash,iou_only,rgb_hist}`.
- `table13`: Mario CRF23 ablation from `record/RESPAWN2026/exp3/mario_crf23_ablation/*/*`.

## Column meanings

- `source_status`: whether per-frame data was extracted (`ok`) or is missing.
- `source_note`: provenance note or missing-data explanation for the row.
- `run_dir`, `report_path`, `log_original`, `log_recovered`, `log_segmented`: exact artifact paths used for extraction.
- `per_frame_source`: source of per-frame rows (currently `report.json:per_frame`).
- `frame`: frame index from the run report.
- `masked_alpha_pct`, `masked_bbox_pct`: masked area coverage percentages for that frame.
- `object_present_model`: detector/object presence count for that frame.
- `frame_flags`, `frame_flags_raw`: frame mode/state flags from the pipeline.
- `latent_minted_regions`, `latent_reused_regions`: newly minted vs reused template/latent region counts.
- `changed_pixels_pct`: percent of changed pixels relative to frame area.
- `ssim`, `rec_ssim`, `psnr`, `rec_psnr`: frame-level quality metrics before/after recovery.
- `encode_masked_ms`, `encode_baseline_ms`, `inference_ms`, `masking_ms`, `stitching_ms`, `total_ms`: frame-level timing components in milliseconds.
