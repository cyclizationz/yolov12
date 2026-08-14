# Quick GRACE comparison

This harness evaluates whether RESPAWN's masked stream remains smaller after
GRACE coding and whether GRACE's loss response changes on masked content.
GRACE stays in a separate checkout: its academic license permits academic use
but prohibits transferring the software or derivatives.

## Scope

The quick experiment feeds two videos to the unmodified upstream evaluator:

1. `source.mp4`: the normalized full-frame clip.
2. `respawn_masked.mp4`: the same clip after RESPAWN masking, before stitching.

The reducer pairs identical GRACE model/loss/burst conditions and reports:

- GRACE's modeled entropy-size estimate for each arm;
- prorated in-band RESPAWN metadata bytes;
- a size-savings proxy after GRACE coding;
- GRACE PSNR/SSIM against each arm's own input.

This is a mechanism smoke test, not the final four-arm evaluation. In
particular, upstream `grace-gpu.py` does not emit decoded videos, so the quick
result cannot measure source-to-final quality after RESPAWN stitching.

Additional upstream limitations must remain visible in any reported result:

- only the first 16 frames are evaluated;
- frames are resized to the next multiple of 128 rather than padded;
- `size` is an entropy/BPG estimate, not an emitted network bitstream;
- “loss” zeros latent coefficients and is not packet loss;
- the default `segmented_output.mp4` input adds GRACE after x264, stacking two
  lossy codecs.

The publishable follow-up should use `--dump-frame-cache`, convert
`frame_cache/masked_frames.bgr` to a lossless GRACE input, decode GRACE output,
then stitch with the existing MSK1/template path. GRACE does not currently
expose the required file-bitstream/decode interface, so that path needs an
external academic-use-only adapter.

## External setup

Use GRACE commit `14f1721663acd5d16ca776ce61b4a5860167d24d` or record the
actual commit in the generated manifest:

```bash
git clone https://github.com/UChi-JCL/Grace /path/to/Grace
cd /path/to/Grace
conda env create -f env.yml
```

Download `grace_models.tar` from the URL in GRACE's README and extract it so
that `/path/to/Grace/models/grace/*_freeze.model` exists. The published
environment uses Python 3.8, PyTorch 1.13.1, CUDA 11.7, and `torchac 0.9.3`.
Modern GPUs may require a compatibility update; record any deviation.

## Commands

First generate and inspect the exact manifest without invoking GRACE:

```bash
python tools/experiments/run_grace_comparison.py \
  --grace-root /path/to/Grace \
  --python /path/to/grace-python \
  --dry-run --force
```

Run the upstream evaluator and collect the paired proxy:

```bash
python tools/experiments/run_grace_comparison.py \
  --grace-root /path/to/Grace \
  --python /path/to/grace-python \
  --force
```

If GRACE was run separately, reduce an existing `all.csv`:

```bash
python tools/experiments/run_grace_comparison.py \
  --grace-root /path/to/Grace \
  --collect-only /path/to/all.csv \
  --collect-manifest /path/to/run_manifest.json
```

The manifest is required so the collector cannot silently combine a CSV from
one clip with metadata and provenance from another.

Outputs are written under `fc5_00_quick/`:

- `run_manifest.json`: exact inputs, command, and GRACE revision;
- `comparison.csv`: one row per model/loss/burst condition;
- `summary.json`: machine-readable results and interpretation warning.

## Full zero-loss run

`run_grace_full.py` provides the separate-system comparison missing from the
quick compatibility check. It processes the requested full clip with one
released model, edge-pads rather than stretches frames to a multiple of 128,
and crops decoded frames back to the source dimensions. It writes the actual
torchac/BPG payloads with per-frame framing metadata and reports PSNR/SSIM
against the original input:

```bash
LD_LIBRARY_PATH=/path/to/Grace/libs \
TORCH_EXTENSIONS_DIR=/writable/torch_extensions \
/path/to/grace-python tools/experiments/run_grace_full.py \
  --grace-root /path/to/Grace \
  --input /path/to/clip.mp4 \
  --output-dir /path/to/output/model_2048 \
  --model-id 2048
```

Use `--no-decoded-video` for the complete eleven-model rate sweep, then rerun
the selected matched operating point without that flag to retain lossless FFV1
output for VMAF. The payload is real entropy-coded data, but it is a research
container rather than an interoperable GRACE wire format. GRACE does not
implement entropy decoding; decoded frames are therefore reconstructed in
memory from the same quantized latent codes whose byte streams are written.

## Interpretation

`proxy_bsp_percent > 0` means GRACE estimated fewer entropy/BPG bytes for the
RESPAWN-masked input plus prorated RMD than for the source input under the same
condition. It does **not** represent measured wire bytes and does not include
template delivery.

PSNR/SSIM compare GRACE output with each arm's own input. They answer whether
masking changes GRACE's codec robustness, not whether the final stitched frame
matches the original source. A publishable complementarity claim still needs
decoded GRACE frames, RESPAWN stitching, source-referenced VMAF/LPIPS, and
capture-to-display latency/stall measurements.
