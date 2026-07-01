#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[2]
EXP6 = REPO / "record" / "RESPAWN2026" / "exp6"
RESOURCE_RUNS = EXP6 / "resource_runs"
OUT_CSV = EXP6 / "exp6_resource_component_summary.csv"
OUT_MD = EXP6 / "exp6_resource_component_summary.md"

RUN_ORDER = [
    ("FC5", "fc5_00", "partial-warm"),
    ("FM6", "fm6_00", "partial-warm"),
    ("Mario", "mario_00", "warm"),
]


def parse_time_stats(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Command being timed:"):
            out["command"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("Elapsed (wall clock) time"):
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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return float(values[idx])


def avg_key(rows: list[dict], key: str) -> float:
    vals = [float(item.get(key, 0.0) or 0.0) for item in rows]
    return mean(vals) if vals else 0.0


def parse_gpu_usage(path: Path, cpu_mode: bool) -> dict[str, str]:
    raw = path.read_text(errors="replace").strip()
    default_status = "cpu-only" if cpu_mode else "unavailable"
    default_note = "CPU mode run: GPU disabled" if cpu_mode else "gpu_usage.csv empty"
    if not raw:
        return {
            "gpu_status": default_status,
            "gpu_note": default_note,
            "gpu_util_avg_pct": "0.0" if cpu_mode else "N/A",
            "gpu_util_max_pct": "0.0" if cpu_mode else "N/A",
            "gpu_mem_avg_mb": "0.0" if cpu_mode else "N/A",
            "gpu_mem_max_mb": "0.0" if cpu_mode else "N/A",
            "gpu_power_avg_w": "0.0" if cpu_mode else "N/A",
            "gpu_power_max_w": "0.0" if cpu_mode else "N/A",
            "gpu_samples": "0",
        }

    if "couldn't communicate with the NVIDIA driver" in raw:
        return {
            "gpu_status": "cpu-only" if cpu_mode else "driver-unavailable",
            "gpu_note": (
                "CPU mode run: GPU disabled; nvidia-smi driver query failed"
                if cpu_mode
                else "nvidia-smi failed to communicate with driver"
            ),
            "gpu_util_avg_pct": "0.0" if cpu_mode else "N/A",
            "gpu_util_max_pct": "0.0" if cpu_mode else "N/A",
            "gpu_mem_avg_mb": "0.0" if cpu_mode else "N/A",
            "gpu_mem_max_mb": "0.0" if cpu_mode else "N/A",
            "gpu_power_avg_w": "0.0" if cpu_mode else "N/A",
            "gpu_power_max_w": "0.0" if cpu_mode else "N/A",
            "gpu_samples": "0",
        }

    util_vals: list[float] = []
    mem_vals: list[float] = []
    power_vals: list[float] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("utilization.gpu"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util_vals.append(float(parts[0]))
            mem_vals.append(float(parts[1]))
            power_vals.append(float(parts[2]))
        except ValueError:
            continue

    if not util_vals:
        return {
            "gpu_status": default_status,
            "gpu_note": "no parseable samples in gpu_usage.csv",
            "gpu_util_avg_pct": "0.0" if cpu_mode else "N/A",
            "gpu_util_max_pct": "0.0" if cpu_mode else "N/A",
            "gpu_mem_avg_mb": "0.0" if cpu_mode else "N/A",
            "gpu_mem_max_mb": "0.0" if cpu_mode else "N/A",
            "gpu_power_avg_w": "0.0" if cpu_mode else "N/A",
            "gpu_power_max_w": "0.0" if cpu_mode else "N/A",
            "gpu_samples": "0",
        }

    return {
        "gpu_status": "ok",
        "gpu_note": "nvidia-smi sampling",
        "gpu_util_avg_pct": f"{mean(util_vals):.1f}",
        "gpu_util_max_pct": f"{max(util_vals):.1f}",
        "gpu_mem_avg_mb": f"{mean(mem_vals):.1f}",
        "gpu_mem_max_mb": f"{max(mem_vals):.1f}",
        "gpu_power_avg_w": f"{mean(power_vals):.1f}",
        "gpu_power_max_w": f"{max(power_vals):.1f}",
        "gpu_samples": str(len(util_vals)),
    }


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for game, clip_id, cache_mode in RUN_ORDER:
        run_dir = RESOURCE_RUNS / clip_id
        report = json.loads((run_dir / "report.json").read_text())
        per_frame = report.get("per_frame", []) or []
        tavg = report.get("timing_avg_ms", {}) or {}
        time_stats = parse_time_stats(run_dir / "resource_time.txt")
        commandline = report.get("commandline", "")
        cpu_mode = "--cpu" in commandline or "--cpu" in time_stats.get("command", "")
        gpu = parse_gpu_usage(run_dir / "gpu_usage.csv", cpu_mode=cpu_mode)

        avg_total = float(tavg.get("total", avg_key(per_frame, "total_ms")))
        p95_total = percentile([float(item.get("total_ms", 0.0) or 0.0) for item in per_frame], 0.95)
        eff_fps = 1000.0 / avg_total if avg_total > 1e-9 else 0.0

        rows.append(
            {
                "game": game,
                "clip_id": clip_id,
                "cache_mode": cache_mode,
                "frames": str(len(per_frame)),
                "wall_time": time_stats.get("wall_time", ""),
                "cpu_pct": time_stats.get("cpu_pct", ""),
                "user_s": time_stats.get("user_s", ""),
                "system_s": time_stats.get("system_s", ""),
                "max_rss_mb": time_stats.get("max_rss_mb", ""),
                **gpu,
                "avg_total_ms": f"{avg_total:.3f}",
                "p95_total_ms": f"{p95_total:.3f}",
                "effective_fps": f"{eff_fps:.2f}",
                "avg_infer_ms": f"{float(tavg.get('infer', avg_key(per_frame, 'infer_ms'))):.3f}",
                "avg_dict_ms": f"{float(tavg.get('dict', avg_key(per_frame, 'dict_ms'))):.3f}",
                "avg_paint_ms": f"{float(tavg.get('paint', avg_key(per_frame, 'paint_ms'))):.3f}",
                "avg_recover_ms": f"{float(tavg.get('recover', avg_key(per_frame, 'recover_ms'))):.3f}",
                "avg_encode_baseline_ms": f"{avg_key(per_frame, 'encode_baseline_ms'):.3f}",
                "avg_encode_masked_ms": f"{avg_key(per_frame, 'encode_masked_ms'):.3f}",
                "avg_stitching_ms": f"{avg_key(per_frame, 'stitching_ms'):.3f}",
                "total_detections": str(int(report.get("total_detections", 0) or 0)),
                "matched_detections": str(int(report.get("matched_detections", 0) or 0)),
                "avg_ssim_recovered": f"{float(report.get('avg_recovered_ssim', 0.0) or 0.0):.4f}",
                "run_dir": str(run_dir.relative_to(REPO)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_host_params() -> dict[str, str]:
    params: dict[str, str] = {
        "cpu_model": "unknown",
        "cpu_logical_cores": "unknown",
        "gpu_name": "unknown",
        "gpu_driver": "unknown",
        "gpu_cuda": "unknown",
        "gpu_total_mem_mib": "unknown",
        "gpu_power_limit_w": "unknown",
    }
    try:
        lscpu = subprocess.check_output(["lscpu"], text=True, errors="replace")
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                params["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                params["cpu_logical_cores"] = line.split(":", 1)[1].strip()
    except Exception:
        pass

    try:
        smi_query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            errors="replace",
        ).strip()
        if smi_query:
            parts = [p.strip() for p in smi_query.splitlines()[0].split(",")]
            if len(parts) >= 4:
                params["gpu_name"] = parts[0]
                params["gpu_driver"] = parts[1]
                params["gpu_total_mem_mib"] = parts[2]
                params["gpu_power_limit_w"] = parts[3]
    except Exception:
        pass

    try:
        smi_full = subprocess.check_output(["nvidia-smi"], text=True, errors="replace")
        m = re.search(r"CUDA Version:\s*([0-9.]+)", smi_full)
        if m:
            params["gpu_cuda"] = m.group(1)
    except Exception:
        pass
    return params


def write_md(rows: list[dict[str, str]], path: Path, csv_path: Path) -> None:
    host = collect_host_params()
    enc = json.loads((RESOURCE_RUNS / RUN_ORDER[0][1] / "report.json").read_text()).get("encoder_settings", {})
    lines = [
        "# Exp6 Resource and Component Timing Summary",
        "",
        "These numbers come from rerunning the same 300-frame `RespawnOnlineServer` samples under `/usr/bin/time -v` while sampling `nvidia-smi`.",
        "",
        "Driver/extraction fix: resource runs are now collected in CUDA mode (no `--cpu`) with live `nvidia-smi` sampling.",
        "",
        "## Host Parameters",
        "",
        "| CPU model | Logical cores | GPU | Driver | CUDA | GPU memory MiB | GPU power limit W |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
        f"| {host['cpu_model']} | {host['cpu_logical_cores']} | {host['gpu_name']} | {host['gpu_driver']} | {host['gpu_cuda']} | {host['gpu_total_mem_mib']} | {host['gpu_power_limit_w']} |",
        "",
        "## Run Parameters",
        "",
        f"- `max_frames`: 300",
        f"- `encoder`: `{enc.get('enc_codec', 'libx264')}` / CRF `{enc.get('enc_crf', '23')}` / preset `{enc.get('enc_preset', 'medium')}` / tune `{enc.get('enc_tune', 'none')}`",
        f"- `open_gop_defaults`: `{enc.get('enc_open_gop_defaults', True)}`",
        f"- `gpu_sampling`: `nvidia-smi` every ~0.5 s",
        "",
        "## Resource Usage",
        "",
        "| Game | CPU % | Max RAM MB | GPU telemetry | Avg GPU % | Max GPU % | Avg GPU Mem MB | Max GPU Mem MB | Avg GPU W | Wall time |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['cpu_pct']} | {r['max_rss_mb']} | {r['gpu_status']} | {r['gpu_util_avg_pct']} | {r['gpu_util_max_pct']} | {r['gpu_mem_avg_mb']} | {r['gpu_mem_max_mb']} | {r['gpu_power_avg_w']} | {r['wall_time']} |"
        )

    lines += [
        "",
        "## GPU Usage",
        "",
        "| Game | Samples | Avg GPU % | Max GPU % | Avg GPU Mem MB | Max GPU Mem MB | Avg GPU W | Max GPU W |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['gpu_samples']} | {r['gpu_util_avg_pct']} | {r['gpu_util_max_pct']} | {r['gpu_mem_avg_mb']} | {r['gpu_mem_max_mb']} | {r['gpu_power_avg_w']} | {r['gpu_power_max_w']} |"
        )

    lines += [
        "",
        "## Component Time Cost",
        "",
        "| Game | Avg total ms | P95 total ms | Eff. FPS | Infer ms | Dict ms | Paint ms | Recover ms | Enc baseline ms | Enc masked ms | Stitch ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['avg_total_ms']} | {r['p95_total_ms']} | {r['effective_fps']} | {r['avg_infer_ms']} | {r['avg_dict_ms']} | {r['avg_paint_ms']} | {r['avg_recover_ms']} | {r['avg_encode_baseline_ms']} | {r['avg_encode_masked_ms']} | {r['avg_stitching_ms']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- CPU `%` is process CPU from `/usr/bin/time -v`; values above 100% mean multiple cores were active.",
        "- Max RAM is maximum resident set size from `/usr/bin/time -v`.",
        "- GPU metrics come from periodic `nvidia-smi` sampling during each run.",
        f"- CSV: `{csv_path}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_md(rows, OUT_MD, OUT_CSV)
    print(json.dumps({"csv": str(OUT_CSV), "md": str(OUT_MD), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
