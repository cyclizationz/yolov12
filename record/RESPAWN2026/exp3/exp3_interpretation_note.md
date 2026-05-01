# Experiment 3 Interpretation Note

## Why low- and high-bitrate cumulative curves can have opposite signs

The cumulative net-savings curve is computed as:

```text
baseline video bytes - RESPAWN segmented video bytes - MSK1 sidecar bytes - simulated template/control bytes
```

For the current tuned upper-bound FC5/Mario runs, template shipping is mostly not what drives the sign. The dominant effect is whether masking reduces encoded video bytes enough to pay for the sidecar.

Mario shows the clearest bitrate-regime effect. Its MSK1 sidecar is roughly constant at about `0.275 Mbps`. At low bitrate, the pure-streaming encoder already compresses the simple pixel scene heavily, so masking does not reduce video bytes enough to offset the sidecar. At higher bitrate, the pure-streaming baseline spends more bits preserving brick detail, while the masked stream remains cheaper; the fixed sidecar is then amortized and cumulative savings becomes positive.

Observed average Mario accounting from the current points:

```text
8 Mbps target:
  baseline video:  ~6.58 Mbps
  RESPAWN video:   ~6.60 Mbps
  MSK1 sidecar:    ~0.28 Mbps
  RESPAWN total:   ~6.88 Mbps  => negative net savings

20 Mbps target:
  baseline video: ~13.93 Mbps
  RESPAWN video:  ~12.73 Mbps
  MSK1 sidecar:   ~0.28 Mbps
  RESPAWN total:  ~13.01 Mbps  => positive net savings
```

FC5 is more unstable because the video-byte delta is very small and clip-dependent. The FC5 sidecar is only about `0.023 Mbps`, but the video savings are also often near zero, so small changes in mask geometry, motion, boundaries, and encoder decisions can flip the sign.

FM6 appeared more consistently positive in older plots because its scenes are more stable and the masked regions were more encoder-friendly. However, any stale FM6 plots in this folder should be interpreted carefully unless Exp3 is rerun from a 150-row combined FC5/FM6/Mario points CSV.

## Why the cache/delay curves are visually inseparable

This is currently an implementation/data-path limitation, not evidence that RTT/cache settings are intrinsically irrelevant.

The Exp3 simulator varies cold start, warm start, partial cache size, RTT, and template-delivery delay. But the current MSK1 payloads used by these Exp1 runs do not carry non-empty `region.path` template identifiers for the simulator to charge and schedule. As a result:

- `active_templates = [region.path for region in payload.regions if region.path]` is empty for the current runs.
- `template_bytes_sent` remains `0`.
- `forced_raw_frames` remains `0`.
- `final_net_saved_bytes` is identical across cold/warm/partial-cache and RTT/delay variants.

The only differences left in the simulation are bookkeeping fields such as warm-start cache footprint, not cumulative delivered bytes. Therefore all cumulative curves collapse onto the same line.

This means the current Exp3 figures should be described as net video+MSK1 accounting curves, not as a valid measurement of client template delivery/cache strategy. To make Exp3 meaningful for cache policy, the pipeline must expose stable template IDs/paths or template-byte records in the sidecar/report, and the simulator must charge those template deliveries and force raw frames while templates are unavailable.

## Synthetic template-delay sensitivity model

To study cache/delay sensitivity before real template IDs are available, `tools/experiments/analyze_overhead.py` now supports a simulation-only synthetic template model with `--synthetic-template-model`.

The synthetic model assigns template IDs from quantized MSK1 region boxes, then applies seeded probabilities for template reuse, cache hit, and delivery loss. A missing template is delayed by the configured RTT window; while unavailable, the simulator forces the frame toward Raw by adding `max(0, baseline_frame_bytes - respawn_frame_bytes)` once per affected frame. This produces sensitivity curves for how much RTT/cache miss/template delivery overhead the current byte savings can tolerate.

Default validation run:

```text
--synthetic-template-size-bytes 4096
--synthetic-reuse-prob 0.9
--synthetic-cache-hit-prob 0.85
--synthetic-loss-prob 0.05
--synthetic-bbox-quant 32
--synthetic-seed 7
```

The synthetic outputs are written separately under `record/RESPAWN2026/exp3_synthetic/` so the original ideal upper-bound Exp3 curves remain intact. The validation run produced 1,500 simulation rows with nonzero template delivery bytes and nonzero forced-Raw fractions, so the synthetic cache/delay curves no longer collapse.

Observed validation ranges:

```text
game   baseline total MB   synthetic RESPAWN total MB   final net MB
FC5    45.87 - 301.48      46.21 - 311.09               -9.62 to +0.04
FM6    30.66 - 302.33      30.56 - 309.62               -10.96 to +5.07
Mario  41.39 - 255.10      56.03 - 427.18               -172.08 to -14.64
```

These numbers should be treated as sensitivity bounds, not measured client behavior. The model is useful for answering how robust the current upper-bound savings are to missing templates, RTT delay, and template delivery cost; it does not replace the future client-side implementation with real template identifiers.
