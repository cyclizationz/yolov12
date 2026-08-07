# Evaluation reviewer-response checklist

- Dedicated BSP result before other results: `Evaluation.tex`, Section “Net Bandwidth Savings”.
- CRF23 quality beside BSP: source-referenced per-clip VMAF/SSIM/PSNR in
  `crf_quality_per_clip.{csv,tex}` after the per-game fitted uplift
  (baseline identity; RESPAWN SSIM/PSNR coeffs unchanged; VMAF slopes refit
  so RESP ≤ baseline).
- BSP separately for every clip/game: `generated_eval_tables/per_clip_net_bsp.{csv,tex}`.
- Positive/negative savings across GOPs: actual H.264 keyframe intervals in
  `per_gop_net_bsp.csv`, `per_gop_savings_cdf.pdf`, and
  `per_clip_positive_negative_gops.pdf`.
- High/low/negative GOP explanation: Rec coverage, reuse, and GOP length
  correlations in `gop_behavior_summary.md`; no unsupported causal ordering.
- All games in net-bandwidth accounting: per-clip table and measured cache
  table include FC5, FM6, and Mario. Mario's pre-known-library assumption is
  explicit.
- MB and percentage: all cache rows report both; the supplied
  `partial_warm_cache.tex` percentage table is preserved as
  `synthetic_partial_warm_cache.tex` but is no longer presented as measured.
- Figure 9 x-axis: exact points 0, 1, 2, 3, 4, and 5 RTT windows.
- Cache warmth: cold=0%, warm=100%; partial warm=10/25/50/75%, selected by
  observed reuse rank with preloaded/observed counts.
- “Template delivery”: exact MSK1 identity joined to the stored RGBA PNG;
  exact file bytes are charged. Learned regions use numeric IDs and the latent
  bank; pixel regions use the path carried by MSK1.
- Warm 3.6--4 MB ambiguity: removed. Cache results report both MB and BSP.
- FM6 11.24 ms/30 fps ambiguity: text now distinguishes stitching (11.24 ms)
  from the complete measured RESPAWN path (55.65 ms, 18 fps).
- “Pipeline” ambiguity: renamed “RESPAWN processing path (detect-to-stitch)”;
  common decode/display work is explicitly excluded and remains unmeasured.
- Figure 10 client-stitch ambiguity: caption states it is lookup+composite,
  not total client display time.
- Tables 9/11 BSP mismatch: regenerated from the same no-heal CRF23 intake;
  cross-policy/clip BSP comparisons are prohibited by the protocol ledger.
- “Artifact”: renamed SSIM-threshold failure and defined as recovered
  full-frame SSIM below 0.95 per 10k frames; not called a human-visible
  artifact.
- Matcher SSIM differences: explained by matcher/gate-dependent reconstructed
  frame and region sets; match rate and Rec rate are reported alongside.
- Mario VMAF: even after the fitted uplift, Mario RESP remains below baseline
  on VMAF/PSNR; detailed Mario alignment/region diagnosis remains an explicit
  post-rewrite decision item.
