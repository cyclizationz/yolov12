# Experiment 3 Interpretation Note

## Intake (May 2026 CRF refresh)

Exp3 reads the **exploratory intake** under [`exp35_crf/exp35_points.csv`](../exp35_crf/exp35_points.csv), not the Exp1 fixed-VBV RD sweep:

| Game | Source | Encoder | Clips | Exp3 scope |
| --- | --- | --- | --- | --- |
| FC5 | `gop_analysis/fc5_XX_offline_open_x264_crf23` | CRF 23, open GOP, `medium` | all 5 | **included** (cache/RTT curves) |
| FM6 | `gop_analysis/fm6_XX_offline_open_x264_crf23` | CRF 23, open GOP, heal-only | all 5 | **included** (cache/RTT curves) |
| Mario | `gop_analysis/mario_XX_offline_open_x264_crf23` | CRF 23, open GOP, pixel pipeline | all 5 | **Exp5 only** — templates assumed pre-known |

All 15 clip pairs use the same CRF23 + open-GOP encoder family. Exp1 fixed-VBV numbers remain authoritative for RD claims.

## Scope: photoreal template-delivery simulation

Exp3 cumulative figures and the cache/RTT narrative focus on **FC5 and FM6**.
For pixel games we assume template libraries are **completely known before session start**; Mario is therefore omitted from Exp3 slides and from template-delay interpretation (Ref ratio saturates at 1.0).
Mario CRF23 BSP (+30.2% mean) is reported under Experiment 5 generality instead.

Example figures: `fc5_00_23p0_cumulative_net_bytes.png`, `fm6_04_23p0_cumulative_net_bytes.png`.

## Accounting model

```text
baseline video bytes - RESPAWN segmented video bytes - RSEI side-channel bytes - simulated template/control bytes
```

Under CRF intake, both arms follow encoder quality rather than a shared VBV peg.

## Observed net BSP (delivered video + RSEI, no synthetic penalty)

| Game | Mean net BSP | Range | Exp3 figures |
| --- | ---: | --- | --- |
| FM6 (heal-only, CRF23) | **+14.1%** | +12.8% to +16.3% | yes |
| FC5 (CRF23) | **+6.2%** | +4.0% to +8.9% | yes |
| Mario (pixel, CRF23) | **+30.2%** | +25.5% to +36.5% | Exp5 only |

Mario CRF23 runs deliver much lower absolute bitrates (~1–1.6 Mbps pure) than the earlier 16 Mbps VBV anchor; cite Exp1 for rate-point sensitivity, CRF intake for fair cross-game generality.

## Cache/RTT curves (synthetic template model)

RSEI payloads lack non-empty `region.path` template identifiers. `analyze_overhead.py` **auto-enables** the synthetic template model when paths are empty.

Default sensitivity grid: RTT `{20, 80, 150}` ms, delay `{0, 2, 5}` RTT, template size 8192 B, cache-hit prob 0.75.

Cumulative PNGs plot **cold/warm RTT grids only** (partial-cache and partial-warm rows stay in CSV). Label as **sensitivity bounds on photoreal clips**, not measured client cache behavior.

Partial-warm rows preload a popularity-ranked fraction of observed templates before session start, use Ref mode on cache hits, and force Raw while missing templates are requested and delivered.
The focused FM6 01--04 summary is in `partial_warm_fm6_summary.csv`, with the paper table in `partial_warm_fm6_table.tex`.
In that subset, a 10% hot preload reduces forced Raw frames from 37.1% to 9.3%, and 50% preload reaches +12.83 MB mean net saved.

## Figure styling

- No figure titles on cumulative PNGs.
- Unique color per `(RTT, delay)` pair; solid = cold start, dashed = warm start.
