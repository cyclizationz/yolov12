# RESPAWN Online Client Scaffold

This directory contains the client-side scaffold for the deferred online evaluation. The executable validates and summarizes the artifact contract emitted by `RespawnOnlineServer`: masked stream, RSEI/MSK1 sidecar, template dictionary, and report.

It also has an optional thin-client replay mode that fetches missing templates from `RespawnTemplateServer` over localhost instead of reading `dict/` directly. It still does not perform live video decode, in-band SEI extraction from a network stream, or frame-deadline measurement. The existing offline evaluator still produces `recovered_output.mp4`; this client target makes the boundary explicit so the next implementation step can replace the offline recovery path with a live client.

## Environment Setup

Run the local setup check:

```bash
cd deployment/client
bash setup_env.sh
```

The script checks for `cmake`, `g++`, `ffmpeg`, `pkg-config`, OpenCV discovery, and the expected ONNX Runtime bundle. It creates `deployment/build` if needed but does not install packages or change global configuration.

## Build

```bash
cd deployment/build
cmake ..
cmake --build . --target RespawnOnlineClient -j
cmake --build . --target RespawnTemplateServer -j
```

## Run

After generating server artifacts:

```bash
./RespawnOnlineClient \
  --input /home/tiehangz/proj/yolov12/deployment/outputs/online_server \
  --cache-mode partial-warm \
  --strict
```

The client reports:

- whether required artifacts are present;
- number of template files in `dict/`;
- number of parsed RSEI/MSK1 payload frames;
- number of referenced regions/grid runs;
- sidecar byte count and parse failures.

## Thin-Client Template Delivery

To hide the template pool from the client, run a local template server over the server-owned dictionary:

```bash
./RespawnTemplateServer \
  --dict /home/tiehangz/proj/yolov12/deployment/outputs/online_server/dict \
  --host 127.0.0.1 \
  --port 19061 \
  --metrics-out /tmp/template_server_metrics.json
```

Then run the client with only the masked stream, RSEI/MSK1 sidecar, and report in its input directory:

```bash
./RespawnOnlineClient \
  --input /path/to/client_input_no_dict \
  --cache-mode partial-warm \
  --template-server 127.0.0.1:19061 \
  --metrics-out /tmp/thin_client_metrics.json \
  --strict
```

In this mode, the client keeps templates only in memory and requests missing template IDs over the socket.

## Cache Policy

The client policy for Experiment 5 is:

- Use Ref mode only when the referenced template exists in the client cache.
- Use Raw mode when the template is missing.
- Request missing templates asynchronously and add them to the cache after delivery.

Supported cache-policy labels:

- `cold`: empty cache at session start.
- `warm`: templates available from prior session reuse.
- `partial-warm`: popularity-ranked subset preloaded before play.

These labels currently drive validation/logging and thin-client replay accounting. Full live cache mutation tied to video decode deadlines and transport acknowledgements is still future work.
