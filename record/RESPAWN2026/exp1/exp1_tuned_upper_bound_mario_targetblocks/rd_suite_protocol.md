# RD Suite Protocol

This file records the exact Experiment 1 protocol used by `run_rd_suite.py`.

## Current protocol
- Reference arm: `Pure streaming` (`variant=pure_streaming`)
- Input assets: manifest-normalized clips (`1920x1080@60` for learned games; source geometry/FPS for Mario pixel clips)
- Rate control: fixed target bitrate sweep at `8/12/16/20/24 Mbps` with matched `maxrate` and `bufsize=2x bitrate`
- Encoder defaults: `libx264`, `superfast`, `gop=60`, `scenecut=0`, `tune=none`, `aud=1`, `repeat-headers=1`
- Current FC5 Exp1 setting is an upper-bound non-heal mode: latent-key masking without `--yolo-heal-only`
- Mario pixel mode uses native template scale and widened band scan: `pixel-force-scale=1.0`, `pixel-band-pad-y=80`, `pixel-bands=4`, `pixel-max-peaks=60`
- RESPAWN quality is measured on `recovered_output.mp4` against the normalized input clip
- Achieved bitrate is measured from `segmented_output.mp4` plus `msk1_payloads.bin` sidecar bitrate
- BD-rate is averaged per clip first, then averaged across clips in the same game

## Important consistency note
- This protocol is **not** the same as the `record/final3` codec-exact CRF18 runs.
- `final3` uses native source cadence/resolution and compares `segmented_output.mp4` against a matched `original_output.mp4` from the same run.
- Therefore, `final3` source-vs-segmented bitrate savings should not be interpreted as direct RD-suite BD-rate expectations unless the RD suite is run with the same native-source CRF protocol.

