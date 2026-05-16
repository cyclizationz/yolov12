# FC5_00 Short Per-Region Flat Color Test

This test keeps the compression-friendly flat-color masking strategy, but makes color selection more sensitive to local environment changes. Instead of using one dominant color for all masks in a frame, it computes one flat color per detected region from that region's `bbox + 100px` neighborhood, excluding the detected mask.

## Outputs

- Previous short control: `record/RESPAWN2026/gop_analysis/fc5_00_short_frame_flat_pad100_feather4/segmented_output.mp4`
- Per-region local flat color: `record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_pad100_feather4/segmented_output.mp4`

Both runs use the same 300-frame prefix of `fc5_00`, open x264 CRF23, solid fill, and `--feather-px 4`.

## Results

| Run | Video BSP (%) | Net BSP (%) | Recovered SSIM | Recovered PSNR | Avg masking ms | Avg no-encode total ms |
|---|---:|---:|---:|---:|---:|---:|
| single local color per frame | 4.22 | 3.85 | 0.9100 | 47.73 | 7.988 | 26.882 |
| local color per region | 4.21 | 3.84 | 0.9102 | 47.72 | 8.303 | 27.208 |

## Interpretation

The per-region variant has almost identical byte savings to the single-color short control, so it preserves the low-entropy property of flat fill. It adds only about `0.32 ms/frame` in this short test.

The quantitative metrics are nearly unchanged, so the decision should be visual: if the per-region video better follows local environment changes in the problematic frames, this is a cheap improvement. If it still selects an unnatural color, the next refinement should keep the same flat-fill structure but change the sampler from "largest histogram bin in bbox+pad" to "boundary-ring color near the mask edge" or "median color in the ring" to avoid the dominant background object overwhelming the local surface color.

Raw CSV:

`record/RESPAWN2026/gop_analysis/fc5_00_short_per_region_flat_summary.csv`
