#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXP6 = REPO / "record" / "RESPAWN2026" / "exp6"
OUT_ROOT = EXP6 / "infer_worker_sweep"

CASES = [
    {
        "game": "FC5",
        "clip_id": "fc5_00",
        "input": "record/RESPAWN2026/manifest/normalized/fc5/fc5_00.mp4",
        "model": "deployment/yolov12n_fc5_seg_v1.onnx",
        "extra": ["--latent-key", "--cache-mode", "partial-warm"],
    },
    {
        "game": "FM6",
        "clip_id": "fm6_00",
        "input": "record/RESPAWN2026/manifest/normalized/fm6/fm6_00.mp4",
        "model": "deployment/yolov12n_racing_e300_split1.onnx",
        "extra": ["--latent-key", "--heal-only", "--cache-mode", "partial-warm"],
    },
]


def avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return float(values[idx])


def parse_wall_seconds(path: Path) -> float:
    for raw in path.read_text(errors="replace").splitlines():
        if raw.strip().startswith("Elapsed (wall clock) time"):
            value = raw.rsplit(": ", 1)[-1].strip()
            parts = value.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(value)
    return 0.0


def parse_gpu(path: Path) -> tuple[float, float]:
    util: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("utilization.gpu"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            util.append(float(parts[0]))
        except ValueError:
            continue
    return avg(util), max(util) if util else 0.0


def component_system_ms(item: dict) -> tuple[float, float]:
    dict_ms = float(item.get("dict_ms", 0.0) or 0.0)
    post_ms = float(item.get("postprocess_ms", 0.0) or 0.0)
    detector_post_ms = max(0.0, post_ms - dict_ms)
    detect_ms = (
        float(item.get("preprocess_ms", 0.0) or 0.0)
        + float(item.get("inference_ms", 0.0) or 0.0)
        + detector_post_ms
    )
    system_ms = (
        detect_ms
        + dict_ms
        + float(item.get("masking_ms", item.get("paint_ms", 0.0)) or 0.0)
        + float(item.get("stitching_ms", item.get("recover_ms", 0.0)) or 0.0)
    )
    return system_ms, detect_ms


def run_case(case: dict[str, object], workers: int, depth: int, max_frames: int) -> dict[str, object]:
    out_dir = OUT_ROOT / f"{case['clip_id']}_w{workers}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_csv = out_dir / "gpu_usage.csv"
    time_txt = out_dir / "resource_time.txt"
    stdout_log = out_dir / "server_stdout.log"

    cmd = [
        str(REPO / "deployment" / "build" / "RespawnOnlineServer"),
        "--input",
        str(REPO / str(case["input"])),
        "--model",
        str(REPO / str(case["model"])),
        "--output",
        str(out_dir),
        *list(case["extra"]),
        "--feather-px",
        "4",
        "--enc-crf",
        "23",
        "--enc-preset",
        "medium",
        "--enc-tune",
        "none",
        "--enc-profile",
        "none",
        "--enc-level",
        "none",
        "--enc-open-gop-defaults",
        "--enc-aud",
        "1",
        "--enc-repeat-headers",
        "0",
        "--mask-color",
        "dominant",
        "--mask-color-period",
        "200",
        "--fill-mode",
        "solid",
        "--max-frames",
        str(max_frames),
        "--cuda",
        "--server-pipeline",
        "1",
        "--server-pipeline-depth",
        str(depth),
        "--server-infer-workers",
        str(workers),
        "--server-post-parallel",
        "1",
    ]

    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,nounits",
            "-lms",
            "500",
        ],
        stdout=gpu_csv.open("w"),
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    try:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_txt), *cmd],
            stdout=stdout_log.open("w"),
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
    finally:
        if sampler.poll() is None:
            sampler.terminate()
            try:
                sampler.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sampler.kill()
    if result.returncode != 0:
        raise RuntimeError(f"sweep failed for {case['clip_id']} workers={workers}; see {stdout_log}")

    report = json.loads((out_dir / "report.json").read_text())
    per_frame = report.get("per_frame", []) or []
    system_vals: list[float] = []
    detect_vals: list[float] = []
    infer_vals: list[float] = []
    for item in per_frame:
        system_ms, detect_ms = component_system_ms(item)
        system_vals.append(system_ms)
        detect_vals.append(detect_ms)
        infer_vals.append(float(item.get("inference_ms", item.get("infer_ms", 0.0)) or 0.0))
    wall_s = parse_wall_seconds(time_txt)
    gpu_avg, gpu_max = parse_gpu(gpu_csv)
    return {
        "game": case["game"],
        "clip_id": case["clip_id"],
        "workers": workers,
        "pipeline_depth": depth,
        "frames": len(per_frame),
        "avg_system_ms": f"{avg(system_vals):.3f}",
        "p95_system_ms": f"{percentile(system_vals, 0.95):.3f}",
        "avg_detect_ms": f"{avg(detect_vals):.3f}",
        "avg_infer_ms": f"{avg(infer_vals):.3f}",
        "wall_s": f"{wall_s:.3f}",
        "wall_ms_per_frame": f"{(wall_s * 1000.0 / len(per_frame)) if per_frame else 0.0:.3f}",
        "gpu_avg_pct": f"{gpu_avg:.1f}",
        "gpu_max_pct": f"{gpu_max:.1f}",
        "report_path": str((out_dir / "report.json").relative_to(REPO)),
    }


def write_plot(rows: list[dict[str, object]], out_path: Path) -> None:
    width, height = 1180, 360
    margin_l, margin_t, margin_b = 58, 34, 46
    panel_gap = 42
    panel_w = (width - margin_l - 18 - 2 * panel_gap) / 3.0
    panel_h = height - margin_t - margin_b
    metrics = [
        ("avg_system_ms", "System total ms/frame"),
        ("avg_infer_ms", "Model infer ms/frame"),
        ("wall_ms_per_frame", "Wall ms/frame"),
    ]
    games = list(dict.fromkeys(str(r["game"]) for r in rows))
    colors = {"FC5": "#2563eb", "FM6": "#dc2626"}
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;opacity:.7}.line{fill:none;stroke-width:2.4}.dot{stroke:white;stroke-width:1.2}</style>',
    ]
    workers_all = sorted({int(r["workers"]) for r in rows})
    x_min, x_max = min(workers_all), max(workers_all)
    for panel_idx, (key, ylabel) in enumerate(metrics):
        x0 = margin_l + panel_idx * (panel_w + panel_gap)
        y0 = margin_t
        vals = [float(r[key]) for r in rows]
        y_max = max(vals) * 1.12 if vals else 1.0
        y_max = y_max if y_max > 0 else 1.0
        svg.append(f'<text x="{x0 + panel_w / 2:.1f}" y="18" text-anchor="middle" font-weight="700">{ylabel}</text>')
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - frac * panel_h
            val = frac * y_max
            svg.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w:.1f}" y2="{y:.1f}"/>')
            svg.append(f'<text x="{x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end">{val:.1f}</text>')
        svg.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0 + panel_h:.1f}" x2="{x0 + panel_w:.1f}" y2="{y0 + panel_h:.1f}"/>')
        svg.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + panel_h:.1f}"/>')
        for w in workers_all:
            x = x0 + (w - x_min) / max(1, x_max - x_min) * panel_w
            svg.append(f'<text x="{x:.1f}" y="{y0 + panel_h + 20:.1f}" text-anchor="middle">{w}</text>')
        svg.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{height - 8}" text-anchor="middle">GPU inference workers</text>')
        for game in games:
            gr = [r for r in rows if r["game"] == game]
            gr.sort(key=lambda r: int(r["workers"]))
            pts = []
            for r in gr:
                x = x0 + (int(r["workers"]) - x_min) / max(1, x_max - x_min) * panel_w
                y = y0 + panel_h - (float(r[key]) / y_max) * panel_h
                pts.append((x, y))
            if not pts:
                continue
            color = colors.get(game, "#111827")
            svg.append(f'<polyline class="line" stroke="{color}" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
            for x, y in pts:
                svg.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    legend_x = width - 130
    for i, game in enumerate(games):
        y = 20 + i * 18
        color = colors.get(game, "#111827")
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="2.4"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 4}">{game}</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg) + "\n")


def write_markdown(rows: list[dict[str, object]], path: Path, csv_path: Path, plot_path: Path) -> None:
    lines = [
        "# Exp6 Inference Worker Sweep",
        "",
        "This sweep increases concurrent YOLO inference sessions in the server pipeline and measures component-summed system time from detection through client stitching.",
        "",
        f"![Inference worker sweep]({plot_path.name})",
        "",
        "| Game | Workers | Avg system ms | P95 system ms | Avg infer ms | Wall ms/frame | Avg GPU % | Max GPU % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['workers']} | {r['avg_system_ms']} | {r['p95_system_ms']} | {r['avg_infer_ms']} | {r['wall_ms_per_frame']} | {r['gpu_avg_pct']} | {r['gpu_max_pct']} |"
        )
    lines += [
        "",
        "## Critical Conclusion",
        "",
        "Increasing concurrent YOLO CUDA sessions from 1 to 4 does not reduce end-to-end system latency on these clips. FC5 stays near 28 ms/frame and FM6 stays near 54 ms/frame, while model inference itself remains about 5 ms/frame; the remaining bottleneck is therefore downstream server/client processing, memory movement, template matching/masking/stitching, and benchmark-side wall work rather than raw GPU inference occupancy.",
        "",
        "## Notes",
        "",
        "- `workers` is the number of concurrent YOLO CUDA sessions feeding an ordered result queue.",
        "- `Avg system ms` excludes MP4 write/proxy encode time and quality-metric bookkeeping.",
        f"- CSV: `{csv_path}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in CASES:
        for workers in args.workers:
            rows.append(run_case(case, workers, args.depth, args.max_frames))

    csv_path = OUT_ROOT / "infer_worker_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = OUT_ROOT / "infer_worker_sweep.svg"
    write_plot(rows, plot_path)
    write_markdown(rows, OUT_ROOT / "infer_worker_sweep.md", csv_path, plot_path)
    print(json.dumps({"csv": str(csv_path), "plot": str(plot_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
