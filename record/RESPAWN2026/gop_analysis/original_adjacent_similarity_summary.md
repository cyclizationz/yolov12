# Adjacent Frame Similarity

- Folder: `record/RESPAWN2026/gop_analysis/original`
- Frames: `1802`
- Adjacent pairs: `1801`
- Exact repeated pairs: `0` (`0.00%`)
- Near-repeated pairs: `0` (`0.00%`)
- Pairs with SSIM >= 0.999: `1124` (`62.41%`)
- Pairs with SSIM >= 0.99: `1128` (`62.63%`)
- Mean adjacent MAD: `10.1763`
- Median adjacent MAD: `0.3596`
- Mean adjacent SSIM: `0.877944`
- Median adjacent SSIM: `0.999886`
- Full CSV: `original_adjacent_similarity.csv`

Near-repeated means `MAD < 0.10`, changed pixels `< 0.05%`, and grayscale SSIM `> 0.9999`.

## Most Similar Adjacent Pairs

| Frames | SSIM | MAD | Changed pixels (%) | Max diff |
|---|---:|---:|---:|---:|
| 1741-1742 | 0.99999931 | 0.007772 | 0.932002 | 4 |
| 961-962 | 0.99999890 | 0.016278 | 1.878279 | 7 |
| 781-782 | 0.99999887 | 0.023700 | 2.506944 | 8 |
| 361-362 | 0.99999877 | 0.025137 | 2.868827 | 7 |
| 421-422 | 0.99999810 | 0.022414 | 2.415654 | 12 |
| 1141-1142 | 0.99999779 | 0.026133 | 2.945795 | 7 |
| 1381-1382 | 0.99999773 | 0.022158 | 2.472946 | 6 |
| 1201-1202 | 0.99999750 | 0.041989 | 4.856819 | 7 |
| 130-131 | 0.99999748 | 0.030195 | 3.593123 | 9 |
| 181-182 | 0.99999725 | 0.038988 | 4.345486 | 12 |

## Lowest-Difference Adjacent Pairs

| Frames | SSIM | MAD | Changed pixels (%) | Max diff |
|---|---:|---:|---:|---:|
| 1741-1742 | 0.99999931 | 0.007772 | 0.932002 | 4 |
| 961-962 | 0.99999890 | 0.016278 | 1.878279 | 7 |
| 1381-1382 | 0.99999773 | 0.022158 | 2.472946 | 6 |
| 421-422 | 0.99999810 | 0.022414 | 2.415654 | 12 |
| 781-782 | 0.99999887 | 0.023700 | 2.506944 | 8 |
| 361-362 | 0.99999877 | 0.025137 | 2.868827 | 7 |
| 1141-1142 | 0.99999779 | 0.026133 | 2.945795 | 7 |
| 130-131 | 0.99999748 | 0.030195 | 3.593123 | 9 |
| 1561-1562 | 0.99999594 | 0.034364 | 3.758102 | 7 |
| 1492-1493 | 0.99999402 | 0.038296 | 4.115596 | 7 |

## Visual Repeat Check

Because these PNGs are decoded from a lossy video, repeated source frames may not be pixel-identical after encode/decode. A more practical visual-repeat criterion is therefore based on very low mean absolute difference plus high SSIM.

- Strict visual-repeat candidates (`MAD < 0.1`, `SSIM >= 0.9999`): `71` (`3.94%`)
- Looser near-static adjacent pairs (`MAD < 0.5`, `SSIM >= 0.999`): `1082` (`60.08%`)
- Strict candidates by pair-start parity: odd `42`, even `29`
- Loose candidates by pair-start parity: odd `550`, even `532`

The visual-repeat candidates are not concentrated on only odd or even pair starts, so this does not look like a simple deterministic `A,A,B,B,...` frame duplication pattern. It looks more like many naturally near-static adjacent frames, with some very low-motion pairs.
