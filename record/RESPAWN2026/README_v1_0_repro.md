# RESPAWN 2026 v1.0 Reproduction Notes

This branch is a runnable experiment snapshot. It keeps source code, experiment drivers,
manifests, protocols, and lightweight notes while excluding generated videos, binary
payloads, resource run directories, and build outputs.

## Host And Build

- Repository root: `/home/tiehangz/proj/yolov12`
- Python runner used in experiments: `.venv-experiments/bin/python`
- Deployment build directory: `deployment/build` (local build artifact, not committed)
- Online/server binaries expected after build:
  - `deployment/build/Yolov12Deployment`
  - `deployment/build/RespawnOnlineServer`
  - `deployment/build/RespawnOnlineClient`
  - `deployment/build/RespawnTemplateServer`
- Exp6 measured host: AMD Ryzen 7 9800X3D, 16 logical cores, NVIDIA GeForce RTX 5070 Ti,
  driver 595.71.05, CUDA 13.2, 16303 MiB VRAM, 300 W power limit.

## Shared Inputs

- Manifest: `record/RESPAWN2026/manifest/offline_manifest.json`
- Representative clips:
  - FC5: `fc5_00`
  - FM6: `fm6_00`
  - Mario: `mario_00`
- Normalized learned-game clips are 1920x1080 at 60 FPS.
- Large source videos, normalized MP4s, ONNX files, and generated run directories are local
  artifacts and are intentionally not part of the curated backup.

## Shared Encoder Settings

The current CRF/open-GOP experiment stack uses:

```bash
--enc-codec libx264 \
--enc-crf 23 \
--enc-preset medium \
--enc-tune none \
--enc-profile none \
--enc-level none \
--enc-open-gop-defaults \
--enc-aud 1 \
--enc-repeat-headers 0
```

Exp1 RD sweeps use fixed bitrate anchors instead of CRF:

```bash
8 12 16 20 24 Mbps, maxrate = bitrate, bufsize = 2x bitrate
```

## Experiment Entry Points

### Exp1: Rate-Distortion

- Script: `tools/experiments/run_rd_suite.py`
- Protocol note: `record/RESPAWN2026/exp1/rd_suite_protocol.md`
- Current protocol:
  - pure streaming reference arm
  - achieved RESPAWN bitrate = `segmented_output.mp4` plus `msk1_payloads.bin`
  - recovered quality measured against manifest-normalized input
  - FC5 uses latent-key masking without heal-only for the current upper-bound setting
  - Mario uses pixel-mode template settings from the pixel encoder sweep

Typical command shape:

```bash
.venv-experiments/bin/python tools/experiments/run_rd_suite.py \
  --manifest record/RESPAWN2026/manifest/offline_manifest.json \
  --out-dir record/RESPAWN2026/exp1/<run_name>
```

### Exp2: Template And Control Overhead

- Script: `tools/experiments/analyze_overhead.py`
- Output area: `record/RESPAWN2026/exp2/`
- Reports overhead as percentages of baseline file size for the refreshed table/figure path.
- Offline simulation replays `msk1_payloads.bin`, applies cache/template delivery state, and
  accounts for template sends and delayed availability.

Typical command shape:

```bash
.venv-experiments/bin/python tools/experiments/analyze_overhead.py
```

### Exp3: Ablations

- Script: `tools/experiments/run_ablation_suite.py`
- Raw workbook builder/source notes: `record/RESPAWN2026/exp3/README.md`
- Current no-heal ablation runs use CRF23/open-GOP settings.
- Tables:
  - Table 10: feather sweep, `feather = 0/4/8/16`
  - Table 11: fill strategy
  - Table 12: matching strategy
  - Table 13: Mario path ablation

Typical no-heal command shape:

```bash
.venv-experiments/bin/python tools/experiments/run_ablation_suite.py \
  --manifest record/RESPAWN2026/manifest/offline_manifest.json \
  --out-dir record/RESPAWN2026/exp3/<run_name> \
  --clip-ids fc5_00 fm6_00 \
  --config-kinds matching \
  --exp35-encoder \
  --use-normalized-input
```

### Exp4: Generality

- Script: `tools/experiments/analyze_generality.py`
- Notes: `record/RESPAWN2026/exp4_generality/generality_notes.md`
- Uses CRF23 as the encoder quality setting. Labels such as `rate_point_mbps` are reporting
  labels, not fixed bitrate controls, unless the command explicitly uses bitrate mode.

Typical command shape:

```bash
.venv-experiments/bin/python tools/experiments/analyze_generality.py
```

### Exp5: GOP/CDF And Generality Ratios

- Scripts:
  - `tools/experiments/run_gop_analysis_crf23.py`
  - `tools/experiments/build_exp35_points_csv.py`
  - `tools/experiments/plot_mario_exp5_cdf.py`
- Notes:
  - `record/RESPAWN2026/exp5/generality_notes.md`
  - `record/RESPAWN2026/exp5/bsp_vs_ref_ratio.md`

Typical command shape:

```bash
.venv-experiments/bin/python tools/experiments/run_gop_analysis_crf23.py
```

### Exp6: Online Server, Resource, Thin Client

- Server scaffold: `deployment/server/online_server.cpp`
- Client scaffold: `deployment/client/online_client.cpp`
- Template IPC: `deployment/template_ipc.{h,cpp}`
- Thin-client runner: `tools/experiments/run_exp6_thin_client.py`
- Resource summary builder: `tools/experiments/build_exp6_resource_summary.py`

Resource runs are 300-frame CUDA-mode samples using the shared CRF23/open-GOP stack and
`nvidia-smi` sampling every about 0.5 seconds.

Typical server command shape:

```bash
deployment/build/RespawnOnlineServer \
  --input record/RESPAWN2026/manifest/normalized/fc5/fc5_00.mp4 \
  --model deployment/yolov12n_fc5_seg_v1.onnx \
  --output record/RESPAWN2026/exp6/resource_runs/fc5_00 \
  --latent-key \
  --cache-mode partial-warm \
  --feather-px 4 \
  --enc-crf 23 \
  --enc-preset medium \
  --enc-tune none \
  --enc-profile none \
  --enc-level none \
  --enc-open-gop-defaults \
  --enc-aud 1 \
  --enc-repeat-headers 0 \
  --mask-color dominant \
  --mask-color-period 200 \
  --fill-mode solid \
  --max-frames 300 \
  --cuda
```

Thin-client delivery:

```bash
.venv-experiments/bin/python tools/experiments/run_exp6_thin_client.py \
  --out-dir record/RESPAWN2026/exp6/thin_client \
  --template-delay-ms 0
```

Regenerate resource summary after resource runs:

```bash
.venv-experiments/bin/python tools/experiments/build_exp6_resource_summary.py
```

## What This Branch Intentionally Excludes

- `deployment/build/`
- `record/**/server_runs/`
- `record/**/resource_runs/`
- generated videos (`*.mp4`, `*.mkv`, etc.)
- generated CSV/XLSX data packages
- binary metadata payloads such as `msk1_payloads.bin`
- downloaded model files (`*.onnx`, model bundles)
