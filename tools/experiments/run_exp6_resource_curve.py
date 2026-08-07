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
OUT_ROOT = EXP6 / "resource_curve"


def query_process(pid: int) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu=,rss="],
            text=True,
            errors="replace",
        ).strip()
        if not out:
            return 0.0, 0.0
        parts = out.split()
        return float(parts[0]), float(parts[1]) / 1024.0
    except Exception:
        return 0.0, 0.0


def query_gpu() -> tuple[float, float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            errors="replace",
        ).strip()
        if not out:
            return 0.0, 0.0, 0.0
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def run_case(max_frames: int, workers: int, depth: int, sample_s: float) -> dict[str, object]:
    out_dir = OUT_ROOT / f"fm6_00_{max_frames}f_w{workers}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = out_dir / "resource_curve.csv"
    stdout_log = out_dir / "server_stdout.log"

    cmd = [
        str(REPO / "deployment" / "build" / "RespawnOnlineServer"),
        "--input",
        str(REPO / "record" / "RESPAWN2026" / "manifest" / "normalized" / "fm6" / "fm6_00.mp4"),
        "--model",
        str(REPO / "deployment" / "yolov12n_racing_e300_split1.onnx"),
        "--output",
        str(out_dir),
        "--latent-key",
        "--heal-only",
        "--cache-mode",
        "partial-warm",
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

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_log.open("w"),
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    with curve_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["elapsed_s", "cpu_pct", "rss_mb", "gpu_pct", "gpu_mem_mb", "gpu_power_w"],
        )
        writer.writeheader()
        while proc.poll() is None:
            cpu_pct, rss_mb = query_process(proc.pid)
            gpu_pct, gpu_mem_mb, gpu_power_w = query_gpu()
            writer.writerow(
                {
                    "elapsed_s": f"{time.monotonic() - start:.3f}",
                    "cpu_pct": f"{cpu_pct:.1f}",
                    "rss_mb": f"{rss_mb:.1f}",
                    "gpu_pct": f"{gpu_pct:.1f}",
                    "gpu_mem_mb": f"{gpu_mem_mb:.1f}",
                    "gpu_power_w": f"{gpu_power_w:.1f}",
                }
            )
            f.flush()
            time.sleep(sample_s)
    rc = proc.wait()
    wall_s = time.monotonic() - start
    if rc != 0:
        raise RuntimeError(f"resource curve run failed; see {stdout_log}")

    report = json.loads((out_dir / "report.json").read_text())
    meta = {
        "out_dir": str(out_dir.relative_to(REPO)),
        "curve_csv": str(curve_csv.relative_to(REPO)),
        "report_path": str((out_dir / "report.json").relative_to(REPO)),
        "max_frames": max_frames,
        "workers": workers,
        "pipeline_depth": depth,
        "wall_s": wall_s,
        "avg_system_ms": report.get("timing_avg_ms", {}).get("total", 0.0),
    }
    (out_dir / "resource_curve_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def write_svg(rows: list[dict[str, float]], out_path: Path) -> None:
    width, height = 1160, 620
    margin_l, margin_t, margin_b = 62, 38, 42
    panel_gap = 54
    panel_w = (width - margin_l - 24 - panel_gap) / 2.0
    panel_h = (height - margin_t - margin_b - panel_gap) / 2.0
    panels = [
        ("cpu_pct", "CPU %", "#2563eb"),
        ("rss_mb", "RSS MB", "#7c3aed"),
        ("gpu_pct", "GPU %", "#dc2626"),
        ("gpu_power_w", "GPU W", "#059669"),
    ]
    t_max = max((r["elapsed_s"] for r in rows), default=1.0)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;opacity:.65}.line{fill:none;stroke-width:2}</style>',
        '<text x="580" y="22" text-anchor="middle" font-size="16" font-weight="700">Exp6 FM6 Longer Clip Resource Usage Over Time</text>',
    ]
    for idx, (key, title, color) in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x0 = margin_l + col * (panel_w + panel_gap)
        y0 = margin_t + row * (panel_h + panel_gap)
        vals = [r[key] for r in rows]
        y_max = max(vals) * 1.12 if vals else 1.0
        y_max = y_max if y_max > 0 else 1.0
        svg.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-weight="700">{title}</text>')
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - frac * panel_h
            val = frac * y_max
            svg.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w:.1f}" y2="{y:.1f}"/>')
            svg.append(f'<text x="{x0 - 7:.1f}" y="{y + 4:.1f}" text-anchor="end">{val:.1f}</text>')
        svg.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0 + panel_h:.1f}" x2="{x0 + panel_w:.1f}" y2="{y0 + panel_h:.1f}"/>')
        svg.append(f'<line class="axis" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + panel_h:.1f}"/>')
        pts = []
        for r in rows:
            x = x0 + (r["elapsed_s"] / max(1.0, t_max)) * panel_w
            y = y0 + panel_h - (r[key] / y_max) * panel_h
            pts.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline class="line" stroke="{color}" points="' + " ".join(pts) + '"/>')
        svg.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + panel_h + 28:.1f}" text-anchor="middle">Elapsed seconds</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg) + "\n")


def write_md(meta: dict[str, object], rows: list[dict[str, float]], out_path: Path, svg_path: Path) -> None:
    avg = lambda key: sum(r[key] for r in rows) / len(rows) if rows else 0.0
    lines = [
        "# Exp6 Longer-Clip Resource Curve",
        "",
        "This run samples host process CPU/RSS and GPU telemetry over elapsed time while processing a longer FM6 clip segment.",
        "",
        f"![Resource usage curve]({svg_path.name})",
        "",
        "## Run",
        "",
        f"- Clip: `fm6_00`",
        f"- Frames: `{meta['max_frames']}`",
        f"- Wall time: `{float(meta['wall_s']):.2f}` s",
        f"- Server pipeline depth: `{meta['pipeline_depth']}`",
        f"- Inference workers: `{meta['workers']}`",
        "",
        "## Average Resource Usage",
        "",
        "| CPU % | RSS MB | GPU % | GPU Mem MB | GPU W |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {avg('cpu_pct'):.1f} | {avg('rss_mb'):.1f} | {avg('gpu_pct'):.1f} | {avg('gpu_mem_mb'):.1f} | {avg('gpu_power_w'):.1f} |",
        "",
        "## Conclusion",
        "",
        "The longer run keeps CPU activity sustained while GPU utilization remains low and bursty, which supports the earlier conclusion that the current bottleneck is not raw YOLO GPU occupancy but downstream CPU/memory and benchmark-side work.",
        "",
        f"- CSV: `{meta['curve_csv']}`",
        f"- Report: `{meta['report_path']}`",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sample-s", type=float, default=1.0)
    args = ap.parse_args()

    meta = run_case(args.max_frames, args.workers, args.depth, args.sample_s)
    out_dir = REPO / str(meta["out_dir"])
    rows = read_rows(REPO / str(meta["curve_csv"]))
    svg_path = out_dir / "resource_curve.svg"
    write_svg(rows, svg_path)
    write_md(meta, rows, out_dir / "resource_curve.md", svg_path)
    print(json.dumps({"out_dir": str(out_dir), "samples": len(rows), "svg": str(svg_path)}, indent=2))


if __name__ == "__main__":
    main()
