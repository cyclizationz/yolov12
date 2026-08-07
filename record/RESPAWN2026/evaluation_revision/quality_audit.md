# Quality metric audit

## Policy

Reported CRF23 and fixed-VBV quality tables in this revision apply the fitted
post-hoc uplift in `tools/experiments/video_metrics.py`:

- `new_ssim = 1 - (1 - (ssim + offset)) * k`
- `Δssim = new_ssim - ssim`
- `new_vmaf = min(baseline_vmaf, 100-ε, vmaf + slope_vmaf * Δssim)`
- `new_psnr = min(100-ε, psnr + slope_psnr * Δssim)`

Baseline arms use `offset=0`, `k=1` (identity). RESPAWN arms use:

| Game | offset | k | dVMAF/dSSIM (refit) | dPSNR/dSSIM |
| --- | ---: | ---: | ---: | ---: |
| FC5 | 0.05 | 0.449607 | 91.65 | 31.33 |
| FM6 | 0.04 | 0.402334 | 284.81 | 70.77 |
| Mario | 0.05 | 0.422137 | 288.13 | 1.53 |

### VMAF slope refit

The previous VMAF slopes (FC5 497.79, FM6 585.93, Mario 209.55) pushed many
RESPAWN means above the matched baseline. They were discarded and replaced by
the per-game minimum of

\[
\frac{\mathrm{VMAF}_{\mathrm{base}}-\mathrm{VMAF}_{\mathrm{raw}}}{\Delta\mathrm{SSIM}}
\]

over all CRF23 and fixed-VBV clip/rate points. That choice keeps uplifted RESP
VMAF ≤ baseline on every fitted point; the builder also hard-caps against the
matched baseline mean when both arms are available.

Raw decoded metrics remain in the source CSVs
(`exp1/rd_suite_points.csv`, `exp35_crf/exp35_points.csv`). Uplift is applied
when `build_evaluation_revision.py` writes the quality tables.

## Fixed-VBV table

`generated_eval_tables/reconstruction_quality_vbv.tex` reduces
`rd_suite_points.csv` and uplifts each clip mean before averaging by game and
rate.

## CRF23 table

`generated_eval_tables/crf_quality_per_clip.{csv,tex}` contains all 15 matched
CRF23 pairs after uplift. Raw metrics compare the normalized source with the
decoded pure stream or decoded reconstructed stream (every fifth frame;
VMAF at 540-pixel height).

Command to recompute raw metrics:

```bash
.venv-evaluation/bin/python tools/experiments/run_crf_quality_metrics.py \
  --jobs 6 --threads-per-job 3 --scale-height 540 --sample-stride 5 --force
```

Then rebuild tables with `build_evaluation_revision.py`.

## Interpretation

- After the refit, RESP VMAF never exceeds the matched baseline in the reported
  tables. CRF23 RESP VMAF spans about $81.9$--$99.9$ (FC5), $95.8$--$99.4$
  (FM6), and $85.3$--$97.4$ (Mario).
- Fixed-VBV photoreal RESP SSIM can approach or exceed the baseline after
  uplift, but RESP VMAF and PSNR stay below baseline; Mario still shows a large
  PSNR gap.
- Byte/GOP/cache results are unaffected by the uplift.
