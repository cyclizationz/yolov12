# Mentor Encoder Command vs Our Reencode

## Result

Using the mentor scripts' effective FFmpeg command on our same `gop_analysis` PNG frames reproduces his numbers almost exactly.

| Subset | Stream | Original bytes | Masked bytes | BSP (%) | Raw negative-frame crossing (%) |
|---|---|---:|---:|---:|---:|
| all | raw H.264 access units | 29,906,637 | 26,847,002 | 10.23 | 17.70 |
| all | MP4 file size | 29,916,159 | 26,856,320 | 10.23 | n/a |
| odd | raw H.264 access units | 23,460,247 | 21,047,168 | 10.29 | 15.76 |
| odd | MP4 file size | 23,465,404 | 21,052,197 | 10.28 | n/a |

This confirms the mentor's reported `10.24%`, `10.29%`, `17.7%`, and `15.7%` are consistent with his script and our frame data.

## Effective Mentor Command

The scripts in `/home/tiehangz/Downloads` encode with:

```bash
ffmpeg -framerate 30 -i frame_%06d.png   -frames:v N   -c:v libx264   -bf 0   -x264-params aud=1   -pix_fmt yuv420p   -vf scale=trunc(iw/2)*2:trunc(ih/2)*2   [-movflags +faststart | -f h264] output
```

Important defaults are left open: x264 CRF defaults to `23`, preset defaults to `medium`, keyint/GOP defaults to x264's default rather than 60, scenecut remains enabled, and VBV/maxrate/bufsize are not set.

## Difference From Our Prior Verification

Our prior final3-style reencode used `libsvtav1`, CRF 42, GOP 60, no B-frames, no scenecut. It produced lower savings on the same frame dump:

| Command family | All-frame BSP (%) | Odd-frame BSP (%) |
|---|---:|---:|
| mentor-like x264 defaults | 10.23 | 10.29 |
| our final3 AV1/GOP60 reproduction | 7.28 | 8.06 |

The earlier Exp1 encoded clip was even smaller (`~0.8%`) because it used fixed target bitrate / VBV (`8/12/16/20/24 Mbps`) rather than quality-targeted CRF.

## Why Mentor Gets Higher Savings

The main cause is encoder protocol, not a new masking effect.

- His command is quality-targeted x264 CRF, so easier masked content is allowed to become fewer bytes.
- He does not force `-g 60` or `-sc_threshold 0`; x264 can use its default longer GOP/keyint behavior and scenecut decisions. This reduces forced keyframe/key-refresh overhead compared with our fixed GOP-60 experiments.
- He uses x264 `medium` by default, while Exp1 used `superfast`; more encoder effort can exploit easier masked regions better.
- He has no VBV/maxrate/bufsize constraints, while Exp1 fixed-bitrate mode intentionally spends the target budget and therefore suppresses byte savings.
- His CDF crossing is raw negative-frame fraction from Annex-B H.264 access units, not byte-weighted contribution and not AV1/MP4 packet accounting.

## Takeaway

The mentor's result strengthens the hypothesis that our RD-suite protocol is too constrained for measuring raw byte reduction. For bitrate-saving claims, we should include a CRF/quality-targeted encoder-effort experiment alongside the fixed-budget RD curves. Fixed-budget RD is still useful for quality-at-budget, but it is not the best way to show the method's byte-saving potential.

## FC5_00 x264 Control Runs

Follow-up controls on the same `fc5_00` frame dump separate preset/CRF from GOP/scenecut:

| Config | File BSP (%) | Original keyframes | Masked keyframes | Original keyframe gap | Masked keyframe gap |
|---|---:|---:|---:|---|---|
| mentor-like x264 defaults (`medium`, CRF 23, open keyint/scenecut) | 10.23 | 31 | 41 | 1-172, avg 57.93 | 1-167, avg 44.00 |
| x264 `medium`, CRF 23, fixed GOP 60, scenecut 0 | 10.98 | 31 | 31 | fixed 60 | fixed 60 |
| x264 `superfast`, CRF 23, fixed GOP 60, scenecut 0 | 8.22 | 31 | 31 | fixed 60 | fixed 60 |
| Exp1 x264 `superfast`, fixed bitrate 20 Mbps, GOP 60 | 0.85 | 31 | 31 | fixed 60 | fixed 60 |

The mentor command's GOP indices do vary from ours. With open x264 defaults, original and masked streams choose different scenecut/keyframe positions; the masked stream has more keyframes (`41` vs `31`). Our fixed-GOP runs force identical keyframe indices (`1, 61, 121, ...`).

For this FC5_00 frame dump, open scenecut/GOP is **not** the reason the savings are higher: fixed GOP60 with x264 `medium` and CRF 23 saves even more (`10.98%`). The main drivers are:

- quality-targeted CRF rather than fixed bitrate/VBV;
- x264 `medium` encoder effort rather than `superfast`;
- no maxrate/bufsize forcing the encoder to spend a target budget.

The GOP-index mismatch still matters for local GOP analysis: mentor-like open x264 cannot be compared GOP-by-GOP against our fixed-GOP outputs using the same GOP IDs, because scenecut changes the GOP boundaries differently for original and masked videos.

## Working Default For Exploratory Tests

Unless a specific experiment requires fixed bitrate, fixed GOP, or low-latency encoder controls, use the open x264 CRF setup for exploratory bandwidth-saving tests:

```bash
--enc-codec libx264 --enc-crf 23 --enc-preset medium --enc-tune none \
--enc-profile none --enc-level none --enc-open-gop-defaults \
--enc-aud 1 --enc-repeat-headers 0
```

This setting better exposes whether RESPAWN makes the content easier to encode. Fixed bitrate/VBV remains appropriate for RD quality-at-budget comparisons, but it suppresses raw byte-saving measurements.
