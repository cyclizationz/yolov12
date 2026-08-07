# Exp6 Thin-Client Template Delivery

This experiment hides the server template pool from the client. The client input directory contains only `segmented_output.mp4`, `msk1_payloads.bin`, and `report.json`; missing templates are fetched over a localhost TCP template server and cached in client memory.

## Thinness and Delivery

| Game | Templates cached | Template bytes | Client CPU % | Client RAM MB | Wall time | Cache misses | Request failures | P95 latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 200 | 87210095 | 21 | 87.3 | 0:00.08 | 200 | 0 | 0.7311 |
| FM6 | 149 | 132394528 | 19 | 130.2 | 0:00.13 | 149 | 0 | 1.0019 |
| Mario | 1 | 3771 | 0 | 4.5 | 0:00.00 | 1 | 0 | 0.0661 |

## Client Component Cost

| Game | Total ms | Parse ms | Lookup ms | Request wait ms | Cache insert ms | Server avg service ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FC5 | 82.1190 | 0.0190 | 0.0127 | 82.0003 | 0.0357 | 0.3502 |
| FM6 | 131.8110 | 0.0260 | 0.0179 | 131.6540 | 0.0482 | 0.7638 |
| Mario | 0.1334 | 0.0169 | 0.0056 | 0.0661 | 0.0010 | 0.0466 |

## Interpretation

- The client is thin in this experiment: it does not read `dict/` directly and only retains templates requested over the local channel in memory.
- The measured delay is localhost request/response latency, not wide-area network latency. Use `--template-delay-ms` in the runner for controlled artificial delivery delay.
- Mario uses the server-owned pixel template directory as its hidden template pool because the pixel path does not write a per-run `dict/` directory.

- CSV: `/home/tiehangz/proj/yolov12/record/RESPAWN2026/exp6/thin_client/thin_client_summary.csv`
