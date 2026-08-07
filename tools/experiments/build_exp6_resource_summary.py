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


def frame_component_ms(item: dict) -> dict[str, float]:
    """System path only: server detection/postprocess through client stitching."""
    dict_ms = float(item.get("dict_ms", 0.0) or 0.0)
    postprocess_ms = float(item.get("postprocess_ms", 0.0) or 0.0)
    model_post_ms = max(0.0, postprocess_ms - dict_ms)

    pixel_detect_ms = (
        float(item.get("pixel_bootstrap_ms", 0.0) or 0.0)
        + float(item.get("pixel_roi_ms", 0.0) or 0.0)
        + float(item.get("pixel_band_ms", 0.0) or 0.0)
        + float(item.get("pixel_row_ms", 0.0) or 0.0)
        + float(item.get("pixel_flow_ms", 0.0) or 0.0)
    )
    learned_detect_ms = (
        float(item.get("preprocess_ms", 0.0) or 0.0)
        + float(item.get("inference_ms", 0.0) or 0.0)
        + model_post_ms
    )
    detect_ms = learned_detect_ms if learned_detect_ms > 0.0 else pixel_detect_ms
    masking_ms = float(item.get("masking_ms", item.get("paint_ms", 0.0)) or 0.0)
    stitching_ms = float(item.get("stitching_ms", item.get("recover_ms", 0.0)) or 0.0)
    system_ms = detect_ms + dict_ms + masking_ms + stitching_ms
    return {
        "system_ms": system_ms,
        "detect_ms": detect_ms,
        "preprocess_ms": float(item.get("preprocess_ms", 0.0) or 0.0),
        "inference_ms": float(item.get("inference_ms", 0.0) or 0.0),
        "model_post_ms": model_post_ms,
        "dict_ms": dict_ms,
        "masking_ms": masking_ms,
        "stitching_ms": stitching_ms,
    }


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

        components = [frame_component_ms(item) for item in per_frame]
        avg_system = avg_key(components, "system_ms")
        p95_system = percentile([item["system_ms"] for item in components], 0.95)
        eff_fps = 1000.0 / avg_system if avg_system > 1e-9 else 0.0

        rows.append(
            {
                "game": game,
                "clip_id": clip_id,
                "cache_mode": cache_mode,
                "server_pipeline": "1" if report.get("server_pipeline_enabled", False) or "--server-pipeline 1" in commandline else "0",
                "server_pipeline_depth": str(int(report.get("server_pipeline_depth", 1) or 1)),
                "server_post_parallel": "1" if report.get("server_post_parallel_enabled", False) else "0",
                "frames": str(len(per_frame)),
                "wall_time": time_stats.get("wall_time", ""),
                "cpu_pct": time_stats.get("cpu_pct", ""),
                "user_s": time_stats.get("user_s", ""),
                "system_s": time_stats.get("system_s", ""),
                "max_rss_mb": time_stats.get("max_rss_mb", ""),
                **gpu,
                "avg_system_ms": f"{avg_system:.3f}",
                "p95_system_ms": f"{p95_system:.3f}",
                "effective_fps": f"{eff_fps:.2f}",
                "path_scope": "RESPAWN-exclusive detect-to-stitch components",
                "baseline_decode_display_ms": "N/A",
                "avg_detect_ms": f"{avg_key(components, 'detect_ms'):.3f}",
                "avg_preprocess_ms": f"{avg_key(components, 'preprocess_ms'):.3f}",
                "avg_inference_ms": f"{avg_key(components, 'inference_ms'):.3f}",
                "avg_model_post_ms": f"{avg_key(components, 'model_post_ms'):.3f}",
                "avg_dict_ms": f"{avg_key(components, 'dict_ms'):.3f}",
                "avg_masking_ms": f"{avg_key(components, 'masking_ms'):.3f}",
                "avg_stitching_ms": f"{avg_key(components, 'stitching_ms'):.3f}",
                "p50_stitching_ms": f"{percentile([item['stitching_ms'] for item in components], 0.50):.3f}",
                "p95_stitching_ms": f"{percentile([item['stitching_ms'] for item in components], 0.95):.3f}",
                "avg_wall_total_ms": f"{float(tavg.get('total', avg_key(per_frame, 'total_ms'))):.3f}",
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
    first_report = json.loads((RESOURCE_RUNS / RUN_ORDER[0][1] / "report.json").read_text())
    enc = first_report.get("encoder_settings", {})
    pipeline_flag = "1" if first_report.get("server_pipeline_enabled", False) else "0"
    pipeline_depth = int(first_report.get("server_pipeline_depth", 1) or 1)
    post_parallel = "1" if first_report.get("server_post_parallel_enabled", False) else "0"
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
        f"- `server_pipeline`: `{pipeline_flag}`",
        f"- `server_pipeline_depth`: `{pipeline_depth}`",
        f"- `server_post_parallel`: `{post_parallel}`",
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
        "## RESPAWN-Exclusive Processing Cost",
        "",
        "| Game | Avg path ms | P95 path ms | Eff. FPS | Detect ms | Detect prep ms | Detector core ms | Detector post ms | Dict/match ms | Mask/fill ms | Stitch mean ms | Stitch p50 ms | Stitch p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['avg_system_ms']} | {r['p95_system_ms']} | {r['effective_fps']} | {r['avg_detect_ms']} | {r['avg_preprocess_ms']} | {r['avg_inference_ms']} | {r['avg_model_post_ms']} | {r['avg_dict_ms']} | {r['avg_masking_ms']} | {r['avg_stitching_ms']} | {r['p50_stitching_ms']} | {r['p95_stitching_ms']} |"
        )

    lines += [
        "",
        "## Critical Conclusion",
        "",
        "The latency problem after queue-depth and CPU post-processing parallelization is not raw GPU inference. FC5/FM6 detector core time is only about 5 ms/frame, and the inference-worker sweep shows that 2-4 concurrent CUDA sessions do not improve system or wall time; future optimization should target dictionary/template matching, mask/fill, client stitching, memory movement, and removal of benchmark-only encode/quality overhead from the online path.",
        "",
        "## Longer-Clip Resource Curve",
        "",
        "A 900-frame FM6 run (`record/RESPAWN2026/exp6/resource_curve/fm6_00_900f_w1/resource_curve.md`) records CPU/RSS and GPU telemetry over elapsed time. The average CPU load is 142.1%, while GPU utilization averages 4.3%, reinforcing that the optimized scaffold is CPU/memory and benchmark-overhead limited rather than GPU-inference limited.",
        "",
        "## Notes",
        "",
        "- `Avg path` is the sum of RESPAWN-specific measured components from server-side detection through client-side stitching; it is not an end-to-end pipeline time.",
        "- Common baseline work (video decode, display/frame presentation, and ordinary encode/transport) was not measured in this trace and is reported as N/A rather than folded into RESPAWN modules.",
        "- `Stitch` is template lookup plus alpha compositing only; it is not total client display time.",
        "- `Detector post ms` is detector post-processing only; dictionary/template matching is split into `Dict/match ms` to avoid double-counting.",
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
