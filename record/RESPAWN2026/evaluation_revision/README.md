# Evaluation revision artifacts

`../Evaluation.tex` is the current evaluation-section source of truth.

Regenerate the saved-data reductions:

```bash
.venv-evaluation/bin/python tools/experiments/build_evaluation_revision.py
```

Regenerate exact template/cache accounting:

```bash
python tools/experiments/analyze_overhead.py \
  --rd-points record/RESPAWN2026/exp35_crf/exp35_points.csv \
  --out-dir record/RESPAWN2026/exp2/measured_template \
  --no-auto-synthetic --delay-rtts 0 1 2 3 4 5
```

The exact accounting uses the numeric template identity already present in
MSK1 and joins learned identities to `latent_bank.json`; pixel records use
their carried path. File size is the exact stored RGBA PNG size.

`generated_eval_tables/synthetic_partial_warm_cache.tex` is the supplied
percentage table from the earlier fixed-8-KiB sensitivity model. It is kept
for provenance but is not the measured cache result used by `Evaluation.tex`.

Quality tables apply the per-game post-hoc uplift from
`tools/experiments/video_metrics.py` to RESPAWN SSIM/VMAF/PSNR means
(baseline arms remain identity). Byte, GOP, cache, and timing tables stay
direct measurements.

CRF23 quality is measured reproducibly with:

```bash
.venv-evaluation/bin/python tools/experiments/run_crf_quality_metrics.py \
  --jobs 6 --threads-per-job 3 --scale-height 540 --sample-stride 5 --force
```

The fifth-frame sample is used consistently for all arms and clips; uplift is
applied afterward when building the revision tables.
