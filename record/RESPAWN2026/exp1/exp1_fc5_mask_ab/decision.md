# FC5 mask profile A/B decision

**Status:** Complete (May 2026). Exp1 encoder: `medium` + open GOP + fixed VBV (`8–24 Mbps`).

## Mask profile selection

Compared on FC5 shortclips (`fc5_00`–`fc5_04`) at all five rate points under the refreshed Exp1 protocol.

| Profile | Mean net delivered saving (25 points) |
| --- | ---: |
| `global_period200` | **−0.02%** |
| `local_pad64_mode` | (not rerun; superseded A/B showed −0.03% vs +0.03% under old `superfast`/locked-GOP encoder) |

**Selected profile:** `global_period200` (global dominant color, `mask-color-period=200`, `feather-px=4`).

Mask geometry difference is negligible under fixed VBV; the dominant effect is encoder rate control, not fill mode.

## Cross-check: full Exp1 FC5 (5 clips × 5 rates)

Full 15-clip Exp1 refresh (`record/RESPAWN2026/exp1/`) confirms the A/B finding:

| Rate | A/B mean saving | Full Exp1 mean saving |
| --- | ---: | ---: |
| 8 Mbps | −0.04% | −0.21% |
| 16 Mbps | +0.08% | −0.05% |
| 20 Mbps | −0.01% | +0.09% |
| All 25 points | +0.01% | **−0.01%** |

Video-only savings are slightly positive (+0.08% to +0.20% by rate); MSK1 (~20–25 kbps) erases the gain at the delivered total.

## Implication

FC5 crop upper-bound (non-heal, `latent-thr=0.85`) does **not** produce meaningful net bandwidth reduction under matched fixed-VBV Exp1. Larger savings reported in `gop_analysis` used CRF/open-bitrate encoding, not this VBV protocol.
