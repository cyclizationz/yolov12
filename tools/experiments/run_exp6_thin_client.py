#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXP6 = REPO / "record" / "RESPAWN2026" / "exp6"
SERVER_RUNS = EXP6 / "resource_runs"


RUNS = [
    {"game": "FC5", "clip_id": "fc5_00", "cache_mode": "partial-warm", "dict_source": "dict", "port": 19061},
    {"game": "FM6", "clip_id": "fm6_00", "cache_mode": "partial-warm", "dict_source": "dict", "port": 19062},
    {
        "game": "Mario",
        "clip_id": "mario_00",
        "cache_mode": "warm",
        "dict_source": "experiments/encoder_eval/_pixel_single_template",
        "port": 19063,
    },
]


def parse_time(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Elapsed (wall clock) time"):
            out["wall_time"] = line.rsplit(": ", 1)[-1]
        elif line.startswith("User time (seconds):"):
            out["user_s"] = line.split(":", 1)[1].strip()
        elif line.startswith("System time (seconds):"):
            out["system_s"] = line.split(":", 1)[1].strip()
        elif line.startswith("Percent of CPU this job got:"):
            out["cpu_pct"] = line.split(":", 1)[1].strip().rstrip("%")
        elif line.startswith("Maximum resident set size (kbytes):"):
            kb = float(line.split(":", 1)[1].strip())
            out["max_rss_mb"] = f"{kb / 1024.0:.1f}"
    return out


def prepare_client_input(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("segmented_output.mp4", "msk1_payloads.bin", "report.json"):
        shutil.copy2(source / name, target / name)


def run_case(case: dict[str, object], out_root: Path, delay_ms: int) -> dict[str, object]:
    clip_id = str(case["clip_id"])
    source_run = SERVER_RUNS / clip_id
    case_out = out_root / clip_id
    case_out.mkdir(parents=True, exist_ok=True)
    client_input = case_out / "client_input_no_dict"
    prepare_client_input(source_run, client_input)

    dict_source = case["dict_source"]
    if dict_source == "dict":
        dict_dir = source_run / "dict"
    else:
        dict_dir = REPO / str(dict_source)
    if not dict_dir.is_dir():
        raise RuntimeError(f"missing template server directory: {dict_dir}")

    port = int(case["port"])
    server_metrics = case_out / "template_server_metrics.json"
    server_log = case_out / "template_server.log"
    client_metrics = case_out / "thin_client_metrics.json"
    client_time = case_out / "thin_client_time.txt"
    client_log = case_out / "thin_client.log"

    template_server = subprocess.Popen(
        [
            str(REPO / "deployment" / "build" / "RespawnTemplateServer"),
            "--dict",
            str(dict_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--metrics-out",
            str(server_metrics),
        ],
        stdout=server_log.open("w"),
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    time.sleep(0.5)
    try:
        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(client_time),
            str(REPO / "deployment" / "build" / "RespawnOnlineClient"),
            "--input",
            str(client_input),
            "--cache-mode",
            str(case["cache_mode"]),
            "--template-server",
            f"127.0.0.1:{port}",
            "--template-delay-ms",
            str(delay_ms),
            "--metrics-out",
            str(client_metrics),
            "--strict",
        ]
        with client_log.open("w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        if result.returncode != 0:
            raise RuntimeError(f"client failed for {clip_id}; see {client_log}")
    finally:
        if template_server.poll() is None:
            template_server.terminate()
            try:
                template_server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                template_server.kill()

    client = json.loads(client_metrics.read_text())
    server = json.loads(server_metrics.read_text()) if server_metrics.exists() else {}
    time_stats = parse_time(client_time)
    return {
        "game": case["game"],
        "clip_id": clip_id,
        "cache_mode": case["cache_mode"],
        "client_input": str(client_input),
        "dict_hidden_from_client": "yes",
        "template_server_dict": str(dict_dir),
        "frames": client.get("frames", 0),
        "template_refs": client.get("template_refs", 0),
        "cache_hits": client.get("cache_hits", 0),
        "cache_misses": client.get("cache_misses", 0),
        "request_failures": client.get("request_failures", 0),
        "templates_cached": client.get("templates_cached", 0),
        "template_bytes_received": client.get("template_bytes_received", 0),
        "request_latency_p50_ms": f"{float(client.get('request_latency_p50_ms', 0.0)):.4f}",
        "request_latency_p95_ms": f"{float(client.get('request_latency_p95_ms', 0.0)):.4f}",
        "request_latency_p99_ms": f"{float(client.get('request_latency_p99_ms', 0.0)):.4f}",
        "first_template_latency_ms": f"{float(client.get('request_latency_first_ms', 0.0)):.4f}",
        "parse_ms": f"{float(client.get('parse_ms', 0.0)):.4f}",
        "lookup_ms": f"{float(client.get('lookup_ms', 0.0)):.4f}",
        "request_wait_ms": f"{float(client.get('request_wait_ms', 0.0)):.4f}",
        "cache_insert_ms": f"{float(client.get('cache_insert_ms', 0.0)):.4f}",
        "client_total_ms": f"{float(client.get('total_ms', 0.0)):.4f}",
        "client_wall_time": time_stats.get("wall_time", ""),
        "client_cpu_pct": time_stats.get("cpu_pct", ""),
        "client_user_s": time_stats.get("user_s", ""),
        "client_system_s": time_stats.get("system_s", ""),
        "client_max_rss_mb": time_stats.get("max_rss_mb", ""),
        "server_requests": server.get("requests", 0),
        "server_hits": server.get("hits", 0),
        "server_misses": server.get("misses", 0),
        "server_bytes_sent": server.get("bytes_sent", 0),
        "server_avg_service_ms": f"{float(server.get('avg_service_ms', 0.0)):.4f}",
        "server_max_service_ms": f"{float(server.get('max_service_ms', 0.0)):.4f}",
    }


def write_markdown(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    lines = [
        "# Exp6 Thin-Client Template Delivery",
        "",
        "This experiment hides the server template pool from the client. The client input directory contains only `segmented_output.mp4`, `msk1_payloads.bin`, and `report.json`; missing templates are fetched over a localhost TCP template server and cached in client memory.",
        "",
        "## Thinness and Delivery",
        "",
        "| Game | Templates cached | Template bytes | Client CPU % | Client RAM MB | Wall time | Cache misses | Request failures | P95 latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['game']} | {row['templates_cached']} | {row['template_bytes_received']} | {row['client_cpu_pct']} | {row['client_max_rss_mb']} | {row['client_wall_time']} | {row['cache_misses']} | {row['request_failures']} | {row['request_latency_p95_ms']} |"
        )
    lines += [
        "",
        "## Client Component Cost",
        "",
        "| Game | Total ms | Parse ms | Lookup ms | Request wait ms | Cache insert ms | Server avg service ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['game']} | {row['client_total_ms']} | {row['parse_ms']} | {row['lookup_ms']} | {row['request_wait_ms']} | {row['cache_insert_ms']} | {row['server_avg_service_ms']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The client is thin in this experiment: it does not read `dict/` directly and only retains templates requested over the local channel in memory.",
        "- The measured delay is localhost request/response latency, not wide-area network latency. Use `--template-delay-ms` in the runner for controlled artificial delivery delay.",
        "- Mario uses the server-owned pixel template directory as its hidden template pool because the pixel path does not write a per-run `dict/` directory.",
        "",
        f"- CSV: `{csv_path}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=EXP6 / "thin_client")
    ap.add_argument("--template-delay-ms", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, args.out_dir, args.template_delay_ms) for case in RUNS]
    csv_path = args.out_dir / "thin_client_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out_dir / "thin_client_summary.md", csv_path)
    print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
