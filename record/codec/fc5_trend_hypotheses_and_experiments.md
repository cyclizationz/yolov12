# FC5 crop: explaining codec saving-vs-CRF trends (AV1 vs x264/x265)

This note focuses on the **FC5 crop** trace only.

## Observed trend (from `record/codec/codec_rd_points.csv`)

- **AV1 (SVT-AV1)**: saving% is higher around CRF42 (and generally improves as CRF increases in these points).
- **x264/x265**: saving% is higher at lower CRF and **decreases** as CRF increases (trend reverses vs AV1).

Important: `saving_pct` here is **whole-file bitrate saving** including MSK1 overhead, as computed by `tools/metrics/codec_sweep_analyze.py`.

## Why this can happen (hypotheses)

### H1: “Inter-prediction disruption” behaves differently per codec
Masking changes local textures and edges. That can:
- Reduce residual energy (good) if the fill becomes smoother/more predictable.
- Hurt motion compensation / reference prediction (bad) if the masked region changes correlation with references.

Different codecs have different motion tools, block structures, and RDO decisions. AV1 (SVT-AV1) may gain more from “smoother masked fill” at higher CRF (where more detail is discarded), while x264/x265 may already be encoding so coarsely at high CRF that the masked-vs-baseline difference becomes dominated by prediction penalties or overhead.

### H2: Rate-control + AQ/psy defaults differ strongly across codecs
CRF is not comparable across codecs. Each encoder’s defaults (AQ mode strength, psy tuning, lookahead, temporal filtering) can change:
- Where bits are spent (detail vs motion vs noise).
- How much “simplification” from masking translates into fewer bits.

SVT-AV1 also has different internal structures (mini-GOP, alt-ref behavior in some configs) that can shift savings with CRF in ways that don’t match x264/x265.

### H3: GOP/scene-cut and keyframe placement interactions
Even when we set `keyint=60` and `scenecut=0`, the practical distribution of coded bits within a GOP can differ across codecs.
If a codec tends to allocate more bits to certain frames (e.g., keyframes or near-keyframes), masking may help or hurt disproportionately depending on CRF.

### H4: Measurement artifact (file-bitrate vs packet-sum; duration mismatch; container overhead)
`codec_rd_points.csv` uses file_size/duration for rate. Per-frame packet-sum can differ slightly.
If outputs have slightly different durations/timebases, bitrate comparisons can shift.

## Verification experiments (FC5 crop only)

All experiments should run baseline+masked+recovered with **identical encoding knobs** for baseline and masked streams and use **both** rate definitions:
- **R1**: file bitrate (codec_sweep style): `(mp4_size_bits/duration)` + meta_bps
- **R2**: packet-sum (per-frame): `ffprobe -show_packets` sizes aligned by PTS

### E1: Remove inter prediction (all-intra)
**Goal**: see if the trend reversal is primarily inter-prediction related.

- Set `enc-gop=1` (keyint=1; every frame is an I-frame).
- Compare saving-vs-CRF for x264/x265/SVT-AV1.

Expected outcomes:
- If trends converge across codecs → inter-prediction is the dominant mechanism.
- If AV1 still behaves differently → defaults/RC/AQ are likely dominant.

### E2: B-frames on/off for x264/x265
**Goal**: see whether temporal reordering/pred structures cause variance and trend flips.

- Keep `enc-gop=60`, `scenecut=0`.
- Compare `bframes=0` vs `bframes=2` (x264 and x265).
- Plot per-frame delta and GOP-position heatmaps (see GOP analysis plan).

### E3: Psy/AQ sensitivity
**Goal**: see whether “human-visual” bit allocation changes the savings response.

For x264:
- Compare `--enc-tune psnr` vs `--enc-tune none` (or `zerolatency` baseline).

For x265:
- If accessible via params, sweep AQ/psy settings (otherwise note “encoder default”).

### E4: Fill color and edge behavior (dominant vs black)
**Goal**: isolate the effect of the masked pixel distribution on compression efficiency.

- Hold mask geometry constant (same latent/detection settings).
- Compare `--mask-color dominant` vs `--mask-color black`.
- Repeat at CRF18 / CRF28 / CRF42 to see if CRF interacts with fill choice.

### E5: Constant-QP sanity check (optional)
If available for a codec, use a constant-QP mode to reduce RC complexity and see whether the trend persists.

## What to log for every run (required for interpretation)

In each run’s `report.json`, record:
- input video ffprobe summary (`input_ffprobe`)
- exact ffmpeg commands for baseline/masked/recovered
- encoder_settings (structured)
- latent/mask knobs (already present)

