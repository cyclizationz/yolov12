# Experiment 4 Ablation Report

This note summarizes the **currently available** Experiment 4 evidence from the existing artifacts under `record/` and `experiments/encoder_eval/`. It is intentionally conservative: where the present records do not support a clean ablation claim, this report leaves an explicit placeholder instead of inferring a result.

## Scope Of Current Evidence

The current record is strongest for:

- mask fill / dominant-color comparisons
- feather-width sweeps
- Mario pixel-path snapshots

The current record is incomplete for:

- broader multi-clip matching-strategy ablations (`latent-key` vs `pHash` vs `IoU-only` vs `RGB histogram`)
- full Mario path ablations beyond the first-pass rerun reported below
- unified artifact-rate accounting across all ablation points

## Sources Used

- `record/fill/summary.csv`
- `record/feather/summary.csv`
- `record/final3/e4_fc5_x264_crf18_black_max2000/report.json`
- `record/final3/e4_fc5_x264_crf18_dominant_max2000/report.json`
- `experiments/encoder_eval/pixel_kalman_band2_flow_refcmd_5s/report.json`
- `experiments/encoder_eval/pixel_kalman_band2_flow_refcmd_5s_dominant/report.json`
- `record/three_games_comparison.md`
- `record/final2/model_table_v4.md`

## 1. Dominant-Color Fill

### FC5 current evidence

Two FC5 runs exist with comparable settings over `2000` frames:

- `record/final3/e4_fc5_x264_crf18_black_max2000/`
- `record/final3/e4_fc5_x264_crf18_dominant_max2000/`

Observed results:

| Trace | Fill color | Segmented bytes | Saving vs original | Avg masked SSIM | Avg recovered SSIM | Avg recovered PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FC5 | black | 159,414,717 | 5.33% | 0.9224 | 0.9203 | 34.82 |
| FC5 | dominant | 159,513,543 | 5.27% | 0.9309 | 0.9265 | 34.42 |

Takeaway:

- On FC5, `dominant` improves masked and recovered SSIM over `black`.
- The byte saving difference is negligible and slightly favors `black` in this specific pair (`5.33%` vs `5.27%`).
- This means the current FC5 evidence supports a **quality-oriented** benefit for dominant color, but **not yet** a clear bitrate win.

### Mario current evidence

Two short Mario pixel-path runs exist:

- `experiments/encoder_eval/pixel_kalman_band2_flow_refcmd_5s/`
- `experiments/encoder_eval/pixel_kalman_band2_flow_refcmd_5s_dominant/`

Observed results:

| Trace | Fill color | Segmented bytes | Saving vs original | Avg masked SSIM | Avg recovered SSIM | Avg recovered PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mario 5s | default/reference color | 1,486,344 | 18.55% | 0.9109 | 0.9653 | 27.11 |
| Mario 5s | dominant | 1,451,474 | 20.46% | 0.9238 | 0.9653 | 27.11 |

Takeaway:

- On this Mario slice, `dominant` improves bitrate savings by about `1.91` percentage points.
- Recovered quality is effectively unchanged in the current saved metrics.
- This is promising, but the evidence is still narrow because it is based on a short `5s` trace rather than a full Mario ablation suite.


## 2. Feather Sweep

The feather sweep in `record/feather/summary.csv` is currently the cleanest Experiment 4 ablation evidence.

### FC5 feather results

| Feather px | Saving % | Avg recovered SSIM | Avg recovered PSNR |
| ---: | ---: | ---: | ---: |
| 0 | 6.058 | 0.9351 | 41.27 |
| 4 | 6.891 | 0.9359 | 41.38 |
| 8 | 6.763 | 0.9367 | 41.48 |
| 16 | 6.216 | 0.9382 | 41.63 |

Takeaway:

- `4 px` is the best current FC5 bitrate point.
- `8 px` is close behind and slightly better on recovered quality.
- `16 px` gives the best recovered quality, but gives back most of the bitrate gain.

### FM6 feather results

| Feather px | Saving % | Avg recovered SSIM | Avg recovered PSNR |
| ---: | ---: | ---: | ---: |
| 0 | 14.926 | 0.9529 | 29.33 |
| 4 | 15.534 | 0.9537 | 29.50 |
| 8 | 15.460 | 0.9541 | 29.57 |
| 16 | 14.997 | 0.9544 | 29.61 |

Takeaway:

- FM6 shows the same pattern as FC5.
- `4 px` or `8 px` appears to be the best savings/quality compromise.
- `16 px` again buys slightly better recovered quality, but not the best bitrate result.

### Feather conclusion

Current evidence supports the statement that:

- moderate feathering helps bitrate and quality simultaneously
- very large feathering improves quality slightly further, but weakens savings

If a single default must be picked from current evidence, `4 px` is the most defensible default for Experiment 4 figures, with `8 px` as a near-tie worth mentioning in text.

## Current Experiment 4 Narrative

Using only the evidence already in the repo, the strongest defensible narrative is:

1. Moderate feathering (`4-8 px`) is already supported as a useful bitrate/quality trade-off on both FC5 and FM6.
2. Dominant-color masking has promising evidence:
   - FC5: better quality, near-tied bytes versus black
   - Mario short run: better bytes with no recovered-quality penalty
3. The current best Mario path is still the full pixel pipeline, but the ablation table needed to justify each pixel-path component is still missing.
4. The repo does not yet contain enough evidence to make a comparative matching-strategy claim beyond “current best saved runs use latent-key”.

## Final3-Matched Single-Channel Ablations

The tables below now use fresh reruns with `final3`-matched configs: per-game tuned models, latent banks where applicable, `dominant` fill for learned/pixel paths, the same encoder preset/tune family, and matched-reference BSP measured against `original_output.mp4`. The consolidated reducer output is in `record/experiment4_ablation_summary.csv`.

### Placeholder: Matching Strategy Table

Caption:

- `BSP / saving` is delivered bitrate reduction versus `original_output.mp4`, the same input frames re-encoded with the exact same encoder settings as the masked run; this avoids the earlier reference mismatch against unrelated source assets or `baseline_current`.
- `Match rate` is the fraction of detections matched to an existing template instead of minting a new one.
- `ID switches / min` and `new templates / min` are churn metrics; lower is better. Higher ID-switch counts mean worse identity stability, not better motion preservation.
- `Ref ratio` is the frame-level fraction of emitted masked/Ref frames after hysteresis, so it can saturate when every matcher keeps the same region healable on nearly every frame.
- `ROI quality` reports recovered ROI SSIM and PSNR.
- `Artifact incidents` counts frames with `rec_ssim < 0.95`, normalized to `10k` frames.
- The `pHash` row here is still the legacy single-channel path: canonical ROI keys are SHA-256 hashes, while matching uses `8`-bit pHash tolerance.

| Game | Matcher | BSP / saving | Match rate | ID switches / min | New templates / min | Ref ratio | ROI quality | Artifact incidents |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | latent-key | 13.51 | 0.152 | 2100.0 | 2148.0 | 0.473 | SSIM 0.9233 / PSNR 77.80 
| 2000.0 / 10k |
| FC5 | pHash | 14.86 | 0.796 | 468.0 | 516.0 | 0.210 | SSIM 0.9672 / PSNR 88.57 | 800.0 
/ 10k |
| FC5 | IoU-only | 15.07 | 0.938 | 108.0 | 156.0 | 0.120 | SSIM 0.9289 / PSNR 89.27 | 
1166.7 / 10k |
| FC5 | RGB histogram | 15.06 | 0.938 | 108.0 | 156.0 | 0.120 | SSIM 0.9289 / PSNR 89.27 
| 1166.7 / 10k |
| FM6 | latent-key | 9.38 | 0.495 | 1794.6 | 1788.6 | 0.970 | SSIM 0.9516 / PSNR 32.98 | 4673.0 / 10k |
| FM6 | pHash | 9.60 | 0.999 | 0.6 | 2.4 | 0.942 | SSIM 0.8405 / PSNR 25.55 | 9376.0 / 10k |
| FM6 | IoU-only | 9.34 | 0.999 | 1.2 | 3.0 | 0.930 | SSIM 0.8472 / PSNR 27.09 | 9202.5 / 10k |
| FM6 | RGB histogram | 9.60 | 0.999 | 0.6 | 2.4 | 0.942 | SSIM 0.8405 / PSNR 25.55 | 9376.0 / 10k |

Notes:

- The equal-ish FM6 `ref_ratio` across non-latent matchers is still real for this clip: the region remains healable most frames regardless of matcher, so the differentiating signals are BSP, churn, and recovered quality rather than `ref_ratio`.

### Placeholder: Mario Path Table

Caption:

- These rows are from a fresh `final3`-matched rerun on the true full `video/pixel.mkv` source, using the same single-template base path as the older `final3` Mario record.
- `BSP / saving` is delivered bitrate reduction versus `original_output.mp4`, the same input frames re-encoded with the exact same encoder settings as the masked run.
- `Artifact count` is the raw number of frames with `rec_ssim < 0.95` over the full `1688`-frame clip.

| Mario strategy | BSP / saving | Avg recovered SSIM | Avg recovered PSNR | Avg total ms/frame | Artifact count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full pipeline | 1.95 | 0.9814 | 28.19 | n/a | 0 |
| No optical flow | -1.05 | 0.9803 | 39.53 | n/a | 59 |
| No Kalman | 2.26 | 0.9812 | 28.15 | n/a | 5 |
| Template only | -0.57 | 0.9753 | 33.10 | n/a | 137 |

Mario note:

- The earlier large negative Mario values were mostly configuration drift plus a wrong input slice. After restoring the full native source and the `final3` single-template base config, the full pipeline is back to positive BSP.
- On this clip, removing optical flow or collapsing to template-only reduces or eliminates the savings and increases artifact counts. `No Kalman` is slightly ahead of the full path on BSP in this first pass, but the difference is small and should not be over-interpreted from a single trace.

### Placeholder: Fill Strategy Table

| Trace | Fill strategy | Saving % | Avg recovered SSIM | Avg recovered PSNR | Status |
| --- | --- | ---: | ---: | ---: | --- |
| FC5 | black | 5.33 | 0.9203 | 34.82 | supported by current record |
| FC5 | dominant | 5.27 | 0.9265 | 34.42 | supported by current record |
| FC5 | blur | TODO | TODO | TODO | needs clean rerun |
| FC5 | inpaint | TODO | TODO | TODO | needs clean rerun |
| Mario | dominant | 20.46 | 0.9653 | 27.11 | supported by short 5s run |
| Mario | reference color | 18.55 | 0.9653 | 27.11 | supported by short 5s run |

## Bottom Line

The current saved record is now enough to support a fuller first-pass Experiment 4 report:

- **feather**: yes, supported
- **dominant-color fill**: partially supported
- **Mario best current strategy**: yes, supported
- **matching-strategy comparison**: first-pass supported
- **full Mario ablation table**: first-pass supported

That makes feather the strongest present Experiment 4 result, dominant color the best current secondary result, and the Mario/matching subsections the highest-priority placeholders for the next experimental pass.
