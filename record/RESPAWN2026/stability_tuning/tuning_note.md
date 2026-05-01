# FC5 and Mario Stability Tuning Note

This note records the first-pass 5-second tuning sweep for the FC5 flicker and Mario template-scale issues seen in Experiment 1.

## Inputs and scope

- Tested only the first `300` frames (`5 s @ 60 fps`) of `fc5_00` and `mario_00` for the sweep.
- Validated the selected parameters on `fc5_00`-`fc5_04` and `mario_00`-`mario_04`, again using only the first `300` frames.
- Outputs are under `record/RESPAWN2026/stability_tuning/`.

## Diagnostics added

The reusable summarizer is `tools/experiments/summarize_stability.py`. It reports:

- mask/state transition count and run length
- post-hysteresis switch fraction when present in `report.json`
- changed-pixel mean/std and large jumps
- object-count stability
- latent mint/reuse ratio
- recovered SSIM/PSNR from `report.json`

## FC5 finding

The original Exp1-like setting was unstable on `fc5_00`:

| Setting | Post-hysteresis switch fraction | Mean post-hysteresis run | Recovered SSIM | Notes |
| --- | ---: | ---: | ---: | --- |
| old default: `latent-thr=0.86`, no explicit `yolo-heal-iou` | `0.0803` | `12.0` | `0.9480` | visible flicker risk |
| selected: `latent-thr=0.85`, `yolo-heal-iou=0.999`, `yolo-heal-extra=0.01` | `0.0134` | `60.0` | `0.9987` | stable first-5s run |

The important correction is using the **heal gate** (`--yolo-heal-iou`) rather than only changing `--latent-motion-iou`. The earlier ablation matrix's good cells used `--yolo-heal-iou 0.999`; matching that on the Exp1 input/model fixed the short-run flicker.

Selected FC5 defaults:

```text
--latent-thr 0.85
--latent-motion-iou 0.6
--latent-motion-center 20
--latent-motion-scale 0.25
--latent-motion-boost 6
--yolo-heal-iou 0.999
--yolo-heal-extra 0.01
--mask-color dominant
--mask-color-period 200
--fill-mode solid
--feather-px 4
```

## Mario finding

Mario did not have a ref/raw flicker problem in the short runs. The issue was the template footprint. The normalized Exp1 input auto-calibrated the single template near `0.9`, which produced a larger, wrong-looking brick footprint and high side-channel overhead.

The best tradeoff is forcing native template scale:

| Setting | Recovered SSIM | Changed-pixel mean | Object-count mean | BSP vs matched original | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| auto scale (`~0.9`) | `0.9800` | `2.536%` | `30.0` | `-11.14%` | too many regions / wrong footprint |
| force scale `0.5` | `0.9936` | `0.604%` | `23.2` | `-9.87%` | higher quality but likely under-sized and still many regions |
| selected force scale `1.0` | `0.9899` | `1.114%` | `10.5` | `-4.02%` | closest to final3-style brick count, lower metadata overhead |

Selected Mario defaults:

```text
--pixel
-T experiments/encoder_eval/_pixel_single_template
--pixel-force-scale 1.0
--mask-color dominant
--mask-color-period 200
--fill-mode solid
--feather-px 0
```

## 10-clip first-5s validation

Validation with selected settings:

| Game | Clips | Mean post-hysteresis switch fraction | Max post-hysteresis switch fraction | Mean recovered SSIM | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FC5 | 5 | `0.0067` | `0.0134` | `0.9995` | stable enough for full reruns |
| Mario | 5 | `0.0000` | `0.0000` | `0.9930` | no mode flicker; scale fixed to native |

Detailed CSV/JSON outputs:

- `record/RESPAWN2026/stability_tuning/sweep_diagnostics.csv`
- `record/RESPAWN2026/stability_tuning/validation_diagnostics.csv`
- `record/RESPAWN2026/stability_tuning/contact_sheets/fc5_current_vs_healiou.png`
- `record/RESPAWN2026/stability_tuning/contact_sheets/mario_scale_compare.png`

## Propagated defaults

The selected defaults were applied to:

- `tools/experiments/run_rd_suite.py`
- `tools/experiments/run_ablation_suite.py`
- `record/offline_experiment_suite.md`

## Remaining caveat

This is a 5-second stability fix, not a full RD re-evaluation. The next full Exp1 / Exp3 / Exp5 reruns should use these defaults, but the final paper numbers should still come from full-length runs.
