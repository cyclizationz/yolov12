#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "record" / "RESPAWN2026" / "spaceflight_v1"
MODEL = REPO / "deployment" / "yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx"
BANK = REPO / "experiments" / "encoder_eval" / "spaceflight_index_v1" / "dict" / "latent_bank.json"
SERVER = REPO / "deployment" / "build" / "RespawnOnlineServer"


def process_sample(pid: int) -> tuple[float, float]:
    out = subprocess.check_output(["ps", "-p", str(pid), "-o", "%cpu=,rss="], text=True).strip()
    if not out:
        return 0.0, 0.0
    cpu, rss = out.split()
    return float(cpu), float(rss) / 1024.0


def gpu_sample() -> tuple[float, float, float]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    gpu, memory, power = [float(value.strip()) for value in out.splitlines()[0].split(",")]
    return gpu, memory, power


def main() -> None:
    manifest = json.loads((ROOT / "offline_manifest.json").read_text())
    out_root = ROOT / "resource"
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip in manifest:
        run_dir = out_root / clip["clip_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(SERVER),
            "--input", clip["normalized_path"],
            "--model", str(MODEL),
            "--output", str(run_dir),
            "--latent-key",
            "--heal-only",
            "--latent-bank", str(BANK),
            "--cache-mode", "partial-warm",
            "--feather-px", "4",
            "--enc-crf", "23",
            "--enc-preset", "medium",
            "--enc-tune", "none",
            "--enc-profile", "none",
            "--enc-level", "none",
            "--enc-open-gop-defaults",
            "--enc-aud", "1",
            "--enc-repeat-headers", "0",
            "--mask-color", "dominant",
            "--mask-color-period", "200",
            "--fill-mode", "solid",
            "--max-frames", "90",
            "--cuda",
            "--server-pipeline", "1",
            "--server-pipeline-depth", "4",
            "--server-infer-workers", "1",
            "--server-post-parallel", "1",
        ]
        log_path = run_dir / "server_stdout.log"
        started = time.monotonic()
        with log_path.open("w") as log:
            proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            samples = []
            while proc.poll() is None:
                try:
                    cpu, rss = process_sample(proc.pid)
                    gpu, gpu_mem, power = gpu_sample()
                    samples.append(
                        {
                            "elapsed_s": time.monotonic() - started,
                            "cpu_pct": cpu,
                            "rss_mb": rss,
                            "gpu_pct": gpu,
                            "gpu_mem_mb": gpu_mem,
                            "gpu_power_w": power,
                        }
                    )
                except Exception:
                    pass
                time.sleep(0.5)
        if proc.returncode != 0:
            raise RuntimeError(f"Server failed for {clip['clip_id']}; see {log_path}")
        report = json.loads((run_dir / "report.json").read_text())
        with (run_dir / "resource_samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]) if samples else ["elapsed_s"])
            writer.writeheader()
            writer.writerows(samples)
        avg = lambda key: sum(row[key] for row in samples) / len(samples) if samples else 0.0
        maximum = lambda key: max((row[key] for row in samples), default=0.0)
        timing = report.get("timing_avg_ms", {})
        summaries.append(
            {
                "clip_id": clip["clip_id"],
                "frames": len(report.get("per_frame", [])),
                "wall_s": time.monotonic() - started,
                "system_total_ms": float(timing.get("total", 0.0) or 0.0),
                "detect_ms": float(timing.get("infer", 0.0) or 0.0),
                "dict_match_ms": float(timing.get("dict", 0.0) or 0.0),
                "mask_fill_ms": float(timing.get("paint", 0.0) or 0.0),
                "stitch_ms": float(timing.get("recover", 0.0) or 0.0),
                "avg_cpu_pct": avg("cpu_pct"),
                "max_rss_mb": maximum("rss_mb"),
                "avg_gpu_pct": avg("gpu_pct"),
                "max_gpu_mem_mb": maximum("gpu_mem_mb"),
                "avg_gpu_power_w": avg("gpu_power_w"),
                "command": command,
            }
        )

    csv_path = out_root / "spaceflight_resource_summary.csv"
    scalar_fields = [key for key in summaries[0] if key != "command"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in summaries)
    (out_root / "spaceflight_resource_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Spaceflight resource benchmark",
        "",
        "Five 90-frame clips were run sequentially with one CUDA inference worker and the Exp6 server settings.",
        "",
        "| Clip | System ms | CPU % | Max RSS MB | GPU % | Max GPU MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['clip_id']} | {row['system_total_ms']:.3f} | {row['avg_cpu_pct']:.1f} | "
            f"{row['max_rss_mb']:.1f} | {row['avg_gpu_pct']:.1f} | {row['max_gpu_mem_mb']:.1f} |"
        )
    (out_root / "spaceflight_resource_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary_csv": str(csv_path), "clips": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
