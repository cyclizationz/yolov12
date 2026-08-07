# Experiment 5 Generality Notes

This summary is intentionally conservative. It is meant to support the claim that RESPAWN helps most when recurring objects are both large and reusable, and that it degrades gracefully when those conditions fail.

- Rows analyzed: 25
- Content styles represented: photoreal
- Masked-area buckets represented: large, medium, small
- Presence buckets represented: persistent
- Appearance buckets represented: highly_varying, stable

Recommended text direction:

- "RESPAWN is strongest when recurring objects are large and reusable; when these conditions fail, the system gracefully falls back and rarely harms quality."

Interpretation guidance:

- Use `avg_masked_area_pct` and `ref_ratio` as the first-order predictors of savings.
- Use `appearance_variability_roi` and `template_reuse_rate` to explain when reuse becomes unstable or sparse.
- Use `motion_magnitude_roi` against ROI VMAF to explain when motion makes reconstruction harder.
- Use `edge_density`, `sobel_energy`, `gray_variance`, and `gray_entropy` as texture/complexity proxies rather than as hard causal claims.
- Keep the photoreal vs pixel-art comparison descriptive unless enough clips exist in both families to support stronger statistical statements.
- For photoreal clips, `avg_masked_area_pct` is measured from alpha/mask pixels. For pixel-art clips, it is measured from the emitted template region boxes (`sum w*h / frame area`) because the pixel template path does not populate alpha-mask pixels.
- The scatter figure `bsp_vs_masked_area.png` shows styled points only (yellow pixel art, blue photoreal).
- `bsp_vs_ref_ratio.tex` / `.md` tabulate photoreal game averages only; Mario omitted (Ref ratio = 1.0).
- The per-GOP savings CDF drops GOP windows outside [-50%, +50%] and clamps the x-axis to that range.
