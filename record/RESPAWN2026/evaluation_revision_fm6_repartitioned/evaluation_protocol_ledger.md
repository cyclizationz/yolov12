# Evaluation protocol ledger

| Artifact | Encoder regime | Clips | Policy | Byte/timing scope |
| --- | --- | --- | --- | --- |
| Headline per-clip/GOP BSP | CRF23 open GOP, x264 medium, B=0 | 5/title | FC5 latent-key no-heal; FM6 latent-key heal-only; Mario pixel | video + measured RMD; warm cache |
| Fixed-cap BSP/quality | VBV 8/12/16/20/24 Mbps, x264 medium | 5/title | Exp1 selected profile | video + measured RMD |
| Cache/delay | CRF23 open GOP | FC5/FM6 saved pairs | same as headline | video + RMD + exact referenced PNG bytes |
| Feather/fill/matcher | CRF23 open GOP | fc5_00/fm6_00 | no-heal, dominant, feather=4 except varied factor | within-table only |
| Client cost | CRF23, 300 frames | 00/title | online-server scaffold | detect-to-stitch components; common decode/display baseline not measured |

Cross-table rule: compare BSP values only when regime, clip intake, and policy match. Ablation values are within-table effects, not alternate headline configurations.
