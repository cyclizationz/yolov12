# RESPAWN Online Server Scaffold

This directory contains the server-side entry point for the deferred online evaluation. It is intentionally a scaffold over the current offline evaluator: it runs the existing masking, template minting, encoding, RSEI/MSK1 sidecar dump, and report generation path, but names the artifacts as the server contract that a future live client will consume.

The current target does not implement live network transport, WebRTC/GStreamer streaming, or in-band SEI muxing. Those remain the next step for Experiment 5. Today, RSEI/MSK1 metadata is written as `msk1_payloads.bin`.

For local template-delivery experiments, `RespawnTemplateServer` can serve the server-owned `dict/` directory over localhost so the client does not read template files directly.

## Build

```bash
cd deployment/build
cmake ..
cmake --build . --target RespawnOnlineServer -j
cmake --build . --target RespawnTemplateServer -j
```

## Run

```bash
./RespawnOnlineServer \
  --input /path/to/input.mp4 \
  --model /home/tiehangz/proj/yolov12/deployment/yolov12n-seg.onnx \
  --output /home/tiehangz/proj/yolov12/deployment/outputs/online_server \
  --latent-key \
  --cache-mode partial-warm \
  --enc-crf 23
```

For pixel-game runs, use `--pixel --templates <dir>` instead of the YOLO model path. For a prebuilt photoreal pool, use `--latent-bank <path> --latent-no-mint --heal-only`.

## Outputs

The server artifact directory is expected to contain:

- `segmented_output.mp4`: masked stream emitted by the server path.
- `msk1_payloads.bin`: RSEI/MSK1 sidecar, length-prefixed per frame.
- `dict/`: template dictionary for client Ref reconstruction.
- `report.json`: run configuration, quality, timing, and byte accounting.
- `recovered_output.mp4`: current offline evaluator's in-process reconstruction, used as a reference until the standalone live client owns decode and composite.

## Local Template Server

After a server scaffold run, serve the generated template pool with:

```bash
./RespawnTemplateServer \
  --dict /home/tiehangz/proj/yolov12/deployment/outputs/online_server/dict \
  --host 127.0.0.1 \
  --port 19061 \
  --metrics-out /tmp/template_server_metrics.json
```

The template server accepts length-prefixed template-ID requests and returns PNG bytes. This enables Exp6 thin-client measurements where the client receives templates over local communication and has no direct `dict/` access.

## Online State Machine

The intended online policy is:

1. Detect or match a reusable object.
2. If the client is known to have the referenced template, send Ref metadata and mask the region.
3. If the template is missing or in flight, send Raw for that region/frame and request template delivery.
4. Add the template to the client pool after delivery/ack.

The `--cache-mode` flag records the intended cache policy:

- `cold`: client starts with no templates.
- `warm`: required templates are already available, modeling prior-session reuse.
- `partial-warm`: client starts with a popularity-ranked subset and requests misses.

The current server executable records/logs that policy but still uses the offline evaluator for artifact generation. The local template server covers template-byte delivery only; it is not a full live video transport.
