# Offline-First RESPAWN Evaluation Suite

This runbook covers the offline tranche of the experiment plan:

- duo-stream viability screen
- Experiment 1: bitrate-quality frontier
- Experiment 3: template/control overhead
- Experiment 4: ablations
- Experiment 5: generality / failure modes

## Scope assumptions

- Learned-path offline experiments are normalized to `1920x1080` and `60 fps`.
- Mario / pixel-path clips keep their source aspect ratio, source resolution, and source FPS because the brick templates and stitching parameters are calibrated on the original pixel-video geometry.
- FC5 uses the cropped / zoomed-in path only in this phase.
- Full-frame FC5 is intentionally deferred.
- Experiment 1 is currently treated as a **theoretical bandwidth-saving upper bound**: the RESPAWN arm may mask from newly minted templates and does not require the current strict client-heal gate for FC5. Experiments 3 and 5 remain client-strategy-sensitive and must separately evaluate cache/pool/heal availability.
- `VMAF`, `SSIM`, and `PSNR` are computed whenever feasible.
- ROI quality uses the RESPAWN ROI sidecar (`msk1_payloads.bin`) as the region definition.
- ROI SSIM / PSNR are computed on the ROI crop. ROI VMAF uses a masked-video fallback so the metric remains available, but should be interpreted with more caution than full-frame VMAF.

## Common configurations

These defaults are shared across the offline writeup unless an experiment explicitly says otherwise.

### Input preparation

- FC5 Experiment 1 uses five shorter, non-overlapping `30s` chunks from `video/fc5_crop.mkv`, normalized to `1920x1080@60`. The source is only `3m39s`, so the earlier `~100s` windows were too long and risked overlap / duplicated content across FC5 records.
- FM6 Experiment 1 / 3 / 5 clips use the manifest-normalized `1920x1080@60` assets.
- Mario Experiment 1 / 3 / 5 clips use manifest-normalized source-geometry assets (`4:3`, source FPS) rather than stretched `1920x1080@60`; stretching to `16:9` distorted the brick template geometry and made the pixel path unreliable.
- Mario Experiment 1 now uses a merged target-block source from `/home/tiehangz/proj/datasets/pixel/supermario_720.mkv` intervals `00:12-00:17`, `00:25-00:40`, `01:15-01:55`, and `02:10-04:20`, split into five non-overlapping `~38s` clips. This replaces the earlier long/overlapping Mario windows that did not contain target blocks continuously.
- FC5 stays on the cropped / zoomed-in dataset for this phase.
- ROI metrics are always derived from the emitted `msk1_payloads.bin` sidecar of the run being evaluated.

### Learned-path defaults

- FM6 model: `deployment/yolov12n_racing_e300_split1.onnx`
- FC5 model: `deployment/yolov12n_fc5_seg_v1.onnx`
- FM6 latent bank: `experiments/encoder_eval/fm6_index_full_v1/dict/latent_bank.json`
- FC5 latent bank: `experiments/encoder_eval/fc5_crop_index_full_v1/dict/latent_bank.json`
- Matching mode: latent-key. For FC5 Experiment 1, use non-heal mode to measure an upper bound on content-removal bandwidth savings; for FM6 and client-reconstruction experiments, keep `--yolo-heal-only` until a separate strategy is selected.
- Shared motion thresholds: `latent-motion-iou=0.6`, `latent-motion-center=20`, `latent-motion-scale=0.25`, `latent-motion-boost=6`
- FC5 Experiment 1 upper-bound setting: `latent-thr=0.85`, no `--yolo-heal-only`.
- The previous strict-heal FC5 sweep (`yolo-heal-iou=0.999`, `yolo-heal-extra=0.01`) is retained as diagnostic evidence that the current heal-only implementation/parameters are too conservative for Exp1; it should be revisited under Experiment 3 / 5 client-cache work.
- FM6 keeps the prior default `latent-thr=0.86` until a separate FM6 stability sweep is run.
- Mask fill: `dominant` color, `fill-mode=solid`, `mask-color-period=200`, `feather-px=4`

### Pixel-path defaults

- Mario uses the single-template path in `experiments/encoder_eval/_pixel_single_template`
- Mario forces native template scale with `pixel-force-scale=1.0`; the earlier stretched `1920x1080@60` Exp1 input auto-calibrated around `0.9`, which produced a visually wrong brick footprint and higher side-channel overhead on `mario_00`.
- Mario uses a wider native-geometry band scan: `pixel-band-pad-y=80`, `pixel-bands=4`, `pixel-max-peaks=60`. The default `2` narrow bands only detected a few castle bricks on `mario_00`, while this setting restored bottom-row coverage without the overmasking seen with looser adaptive thresholds.
- Mask fill: `dominant` color, `fill-mode=solid`, `mask-color-period=200`, `feather-px=0`

### Encoder defaults

- GOP length: `60`
- Preset: `superfast`
- Tune: `none`
- Scenecut: `0`
- AUD: `1`
- Repeat headers: `1`
- Experiment 1 rate points: `8`, `12`, `16`, `20`, `24 Mbps`
- For Experiment 1, the **pure streaming** reference and **RESPAWN** use the same rate control: `maxrate = target bitrate`, `bufsize = 2x target bitrate` (so gains are not from a looser cap on the reference).

## Main entry points

- Manifest builder: `tools/experiments/build_manifest.py`
- Duo screen: `tools/experiments/run_duo_screen.py`
- RD suite: `tools/experiments/run_rd_suite.py`
- Overhead analysis: `tools/experiments/analyze_overhead.py`
- Ablation suite: `tools/experiments/run_ablation_suite.py`
- Generality analysis: `tools/experiments/analyze_generality.py`
- End-to-end offline orchestrator: `tools/experiments/run_offline_suite.py`

## Typical workflow

```bash
python3 -m venv .venv-experiments
./.venv-experiments/bin/pip install opencv-python matplotlib numpy
./.venv-experiments/bin/python tools/experiments/build_manifest.py
./.venv-experiments/bin/python tools/experiments/run_duo_screen.py
./.venv-experiments/bin/python tools/experiments/run_rd_suite.py
./.venv-experiments/bin/python tools/experiments/analyze_overhead.py
./.venv-experiments/bin/python tools/experiments/run_ablation_suite.py
./.venv-experiments/bin/python tools/experiments/analyze_generality.py
```

Or run the offline tranche end-to-end:

```bash
./.venv-experiments/bin/python tools/experiments/run_offline_suite.py
```

## Expected outputs

- `record/RESPAWN2026/manifest/`
- `record/RESPAWN2026/duo_screen/`
- `record/RESPAWN2026/exp1/`
- `record/RESPAWN2026/exp3/`
- `record/RESPAWN2026/exp4/`
- `record/RESPAWN2026/exp5/`

## Related work and expectations

This section is meant to calibrate expectations for the offline suite. The papers below are useful, but most of them optimize a different layer from RESPAWN, so they should be used as directional context rather than one-to-one numeric targets.

### Closest conceptual inspiration

- [Region-based Content Enhancement for Efficient Video Analytics at the Edge (NSDI 2025)](https://www.usenix.org/conference/nsdi25/presentation/wang-weijun) validates the broader idea of region-selective processing. RegenHance reports `10-19%` end-task accuracy improvement and `2-3x` throughput over frame-based enhancement, while avoiding most of the compute spent on full-frame enhancement. This is an important conceptual match for RESPAWN because it shows that selective treatment of important regions can be worthwhile at the systems level. However, it is **not** a fair direct comparator for Experiment 1 because it targets video analytics accuracy and throughput, not bitrate-distortion curves against a cloud-gaming baseline.

### Closest systems comparators

- [Tooth: Toward Optimal Balance of Video QoE and Redundancy Cost by Fine-Grained FEC in Cloud Gaming Streaming (NSDI 2025)](https://www.usenix.org/conference/nsdi25/presentation/an) is the closest cloud-gaming paper for Experiment 2 / 3 framing. Tooth reports stall-rate reductions of `40.2-85.2%`, video bitrate improvements of `11.4-29.2%`, and bandwidth-cost reductions of `54.9-75.0%` by adapting redundancy at the frame level. It does not report BD-rate, VMAF, SSIM, or PSNR, but it strongly supports measuring wire-efficiency together with QoE rather than bytes alone.
- [Dissecting and Streamlining the Interactive Loop of Mobile Cloud Gaming (NSDI 2025)](https://www.usenix.org/conference/nsdi25/presentation/li-yang) is the best latency reference for Experiment 2. LoopTailor shows that even under good network conditions mobile cloud gaming can suffer `112-403 ms` input-to-display latency, and reduces it by about `34%` to stably below `100 ms`. This is not a compression paper, but it is a strong reference for why queueing and end-to-end latency need to be reported explicitly.
- [Pudica: Toward Near-Zero Queuing Delay in Congestion Control for Cloud Gaming (NSDI 2024)](https://www.usenix.org/conference/nsdi24/presentation/wang-shibo) is useful for Experiment 2 because it quantifies the network-control layer. Pudica reduces average and tail frame delay by `3.1x` and `4.9x`, cuts stall rate by `10.3x`, and still increases frame bitrate by `12.1%`. This gives a realistic sense of how much room still exists in queueing and pacing, independent of content-aware delivery.
- [Enabling High Quality Real-Time Communications with Adaptive Frame-Rate (NSDI 2023)](https://www.usenix.org/conference/nsdi23/presentation/meng) is relevant to Experiment 2 for deadline-miss and stutter framing. AFR reduces tail queuing delay by up to `7.4x` and reduces stuttering events by `34%` in production deployment. Again, this is not a direct RESPAWN comparator, but it is strong evidence that the queueing path must be measured jointly with bitrate and quality.
- [GRACE: Loss-Resilient Real-Time Video through Neural Codecs (NSDI 2024)](https://www.usenix.org/conference/nsdi24/presentation/cheng) is an indirect but useful comparator for lossy-network scenarios. GRACE reports `95%` fewer undecodable frames, `90%` shorter stall duration, and `38%` higher MOS than its resilience baselines. It is not a region-masking system, so it should be used mainly to motivate loss robustness metrics rather than bitrate savings targets.
- [ARTEMIS: Adaptive Bitrate Ladder Optimization for Live Video Streaming (NSDI 2024)](https://www.usenix.org/conference/nsdi24/presentation/tashtarian) is not cloud gaming, but it is useful background for content-adaptive streaming. ARTEMIS reports `25%` lower encoding computation, `18%` lower end-to-end latency, and `11%` higher QoE than static ladders. This supports the broader claim that content-aware transport/encoding decisions can matter materially even when the codec itself is unchanged.

### How to use these papers in this suite

- **Experiment 1 (bitrate-quality frontier):** there is no close published NSDI paper that gives a clean expected `BD-rate`, `VMAF`, `SSIM`, or `PSNR` target for RESPAWN-style region/template delivery. The fair expectation is therefore **content dependence**, not a single headline number: clips with large reusable regions should show clearer gains, while clips with weak reuse or highly literal pixel constraints may be neutral or mixed. If `VMAF` improves while `SSIM` / `PSNR` lag, interpret that as a signal to inspect boundary artifacts and measurement protocol rather than as an automatic failure of the idea.
- **Experiment 2 (online pipeline):** Tooth, LoopTailor, Pudica, and AFR all suggest that latency, queue buildup, and deadline misses are first-class outcomes. A useful win condition for RESPAWN is not just fewer bytes, but maintaining or improving delivered quality without worsening misses, stalls, or queueing.
- **Experiment 3 (net gain after overheads):** Tooth is the strongest reminder that control and redundancy overhead can erase nominal savings if they are not adapted. For RESPAWN, cold-start template shipping, cache misses, and side-channel bytes must therefore be reported explicitly, including break-even time.
- **Experiment 5 (generality / failure modes):** Wang et al. supports the expectation that region selectivity helps most when the important content is sparse and predictable. For RESPAWN, that translates into stronger expected gains when object regions are large enough to matter, recur often enough to reuse templates, and remain stable enough that recovery does not introduce excessive artifact cost.

### Cautions for comparison

- Most NSDI papers above optimize a different layer than RESPAWN: analytics enhancement, FEC, congestion control, frame-rate control, or bitrate-ladder design.
- Their reported gains should therefore be treated as **directional systems context**, not as a numeric benchmark that RESPAWN must match in Experiment 1.
- In particular, published cloud-gaming papers rarely report `BD-rate`, full-frame `VMAF`, ROI `VMAF`, `SSIM`, or `PSNR` under a matched encoder protocol, so our own RD results should be interpreted against our documented protocol first.

## Notes

- The duo-screen should be treated as a gating step. If `mean_run_length_ref_raw_states` is close to `1` or `2`, the dual-substream idea is likely not worth deeper investment.

### Experiment 1 reference arm (important)

- **RD / Exp1 compares RESPAWN against pure streaming, not a second YOLO variant.** Under `record/.../exp1/<clip>/<rate>/pure_streaming/`, the reference is **only** a **matched libx264 encode** of the normalized source (same target bitrate, GOP, preset, tune, scenecut, AUD, repeat-headers, maxrate, bufsize as the offline sweep). There is **no** segmentation, masking, template reuse, or side channel—`original_output.mp4`, `segmented_output.mp4`, and `recovered_output.mp4` are identical copies of that encode. This isolates gains from **content-based delivery** vs **naive streaming at the same encoder configuration**.
- **`respawn_hysteresis/`** is the RESPAWN offline path with the matched encoder as in `run_rd_suite.py`. For the current Exp1 rerun, FC5 uses non-heal latent-key mode to estimate an upper bound on bandwidth saving; this is not yet the final client-deployable heal policy.
- To reproduce the **old** A/B (YOLO current-baseline binary vs RESPAWN), run `tools/experiments/run_rd_suite.py` with **`--legacy-yolo-baseline`** (outputs under `baseline_current/` and CSV `variant` = `baseline`).

### Experiment 3 / 5 client-heal note

- Exp3 and Exp5 should not inherit the Exp1 non-heal assumption blindly. Their central question is whether a thin client with a finite template/cache pool can reconstruct masked regions on time.
- The strict `--yolo-heal-only` trials are therefore still useful: they show that the current client-availability gate can collapse FC5 masking to raw mode and needs better pool warming, template shipping, fallback, or gating before being used as the main deployable policy.

- Mario source material is currently lower-resolution / lower-fps than the learned games. The manifest layer records that normalization explicitly so plots and later writeups can disclose it cleanly.
