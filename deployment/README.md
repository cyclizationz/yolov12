# RESPAWN Deployment

The current production-quality implementation is the offline deployment evaluator in this directory. It simulates server and client roles in one process:

- server-side detection, masking, template minting, encoding, and RSEI/MSK1 sidecar generation;
- client-side template lookup and reconstruction;
- reports for byte accounting, quality, and timing.

The new online split is a scaffold for the deferred Experiment 5 live evaluation:

- `server/README.md`: `RespawnOnlineServer`, which emits the server artifact contract using the current evaluator.
- `client/README.md`: `RespawnOnlineClient`, which validates the artifact contract and states the Ref/Raw cache policy.
- `RespawnTemplateServer`: a localhost template-byte server used by Exp6 thin-client experiments to hide the template pool from the client.
- `client/setup_env.sh`: local environment and build checks for the client path.

Full live video transport is not implemented yet. The next step is to replace the sidecar/offline recovery boundary with runtime decode, in-band SEI extraction, cache acknowledgements tied to frame deadlines, and latency/stall measurement.

## Existing Offline Targets

- `Yolov12Deployment`: main RESPAWN offline evaluator.
- `Yolov12DeploymentCurrentBaseline`: current baseline encoder/evaluator.
- `Yolov12DeploymentDuo`: duo-channel comparison evaluator.

## Online Scaffold Targets

```bash
cd deployment/build
cmake ..
cmake --build . --target RespawnOnlineServer RespawnOnlineClient RespawnTemplateServer -j
```

Run the server scaffold first, then point the client scaffold at the generated artifact directory. For thin-client template-delivery experiments, start `RespawnTemplateServer` on the generated `dict/` and run `RespawnOnlineClient --template-server 127.0.0.1:<port>` against a client input directory that does not contain `dict/`.
