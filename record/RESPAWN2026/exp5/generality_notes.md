# Experiment 5 Generality Notes

This summary is intentionally conservative. It is meant to support the claim that RESPAWN helps most when recurring objects are both large and reusable, and that it degrades gracefully when those conditions fail.

- Rows analyzed: 75
- Content styles represented: photoreal, pixel_art
- Masked-area buckets represented: large, medium, small
- Presence buckets represented: mixed, persistent
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
- The scatter figures filter out negative-BSP points and show only upper/lower envelope curves; the transparent band is the observed positive-saving region, while middle points are intentionally hidden to reduce visual clutter. Envelope points are colored by content style so pixel-art and photoreal clips remain distinguishable.
- The per-GOP savings CDF is plotted per game in one figure, not averaged across clips, so workload-specific tails remain visible.
