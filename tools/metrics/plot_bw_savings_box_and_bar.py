#!/usr/bin/env python3
"""
Create summary plots for bandwidth savings across multiple game scenarios.

Outputs:
  1) Box plot of per-frame byte savings: (baseline_bytes - masked_bytes) [bytes/frame]
     - X axis: game scenarios
     - Y axis: bytes/frame saved (positive means smaller masked stream)
     - Overlays: mean and stddev for each scenario
  2) Bar plot of total saving percent: 100 * (1 - sum(masked_bytes)/sum(base_bytes))

Packet sizes are extracted via ffprobe -show_packets and aligned by PTS time (sorted)
with a tolerance, matching our other bandwidth tools.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    baseline: Path
    masked: Path


def ffprobe_packets(video_path: Path) -> list[tuple[float, int]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,size",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    packets: list[tuple[float, int]] = []
    for p in data.get("packets", []) or []:
        pts_t = p.get("pts_time")
        sz = p.get("size")
        if pts_t is None or sz is None:
            continue
        packets.append((float(pts_t), int(sz)))
    packets.sort(key=lambda x: x[0])
    return packets


def align_by_time(
    a: list[tuple[float, int]], b: list[tuple[float, int]], tol_ms: float
) -> tuple[list[float], list[int], list[int]]:
    tol = tol_ms / 1000.0
    i = 0
    j = 0
    ts: list[float] = []
    asz: list[int] = []
    bsz: list[int] = []
    while i < len(a) and j < len(b):
        t0, s0 = a[i]
        t1, s1 = b[j]
        dt = t0 - t1
        if abs(dt) <= tol:
            ts.append(t0)
            asz.append(s0)
            bsz.append(s1)
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    return ts, asz, bsz


def main() -> None:
    out_dir = Path("/home/tiehangz/proj/yolov12/record/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_box = out_dir / "bw_bytes_per_frame_mean_std_baseline_vs_masked.png"
    out_bar = out_dir / "bw_saving_percent_bar.png"

    tol_ms = 6.0

    scenarios = [
        Scenario(
            key="pixel",
            label="Pixel (Mario)",
            baseline=Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/pixel_kalman_band2_flow_v1/orig_x264_crf18.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/pixel_kalman_band2_flow_v1/masked_x264_crf18.mp4"),
        ),
        Scenario(
            key="fm6",
            label="Racing (FM6)",
            baseline=Path(
                "/home/tiehangz/proj/yolov12/experiments/encoder_eval/yolo_fm6_latent_hybrid_v2_reuse_stats/orig_x264_crf18.mp4"
            ),
            masked=Path(
                "/home/tiehangz/proj/yolov12/experiments/encoder_eval/yolo_fm6_latent_hybrid_v2_reuse_stats/masked_x264_crf18.mp4"
            ),
        ),
        Scenario(
            key="fc5_gunheavy",
            label="FPS (FC5 gun-heavy)",
            baseline=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/orig_x264_crf18.mp4"),
            masked=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/masked_x264_crf18.mp4"),
        ),
        Scenario(
            key="fc5_full_roi",
            label="FPS (FC5 full ROI)",
            baseline=Path(
                "/home/tiehangz/proj/yolov12/experiments/encoder_eval/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/orig_x264_crf18.mp4"
            ),
            masked=Path(
                "/home/tiehangz/proj/yolov12/experiments/encoder_eval/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/masked_x264_crf18.mp4"
            ),
        ),
    ]

    per_frame_base: list[np.ndarray] = []
    per_frame_mask: list[np.ndarray] = []
    saving_pct: list[float] = []
    base_means: list[float] = []
    base_stds: list[float] = []
    mask_means: list[float] = []
    mask_stds: list[float] = []
    ns: list[int] = []

    for sc in scenarios:
        p0 = ffprobe_packets(sc.baseline)
        p1 = ffprobe_packets(sc.masked)
        _ts, b0, b1 = align_by_time(p0, p1, tol_ms=tol_ms)
        if not b0 or not b1:
            raise SystemExit(f"Failed to align packets for {sc.key}")
        base = np.array(b0, dtype=np.float64)
        mask = np.array(b1, dtype=np.float64)
        per_frame_base.append(base)
        per_frame_mask.append(mask)
        ns.append(int(len(base)))
        base_means.append(float(base.mean()))
        base_stds.append(float(base.std(ddof=0)))
        mask_means.append(float(mask.mean()))
        mask_stds.append(float(mask.std(ddof=0)))
        saving_pct.append(float(100.0 * (1.0 - (mask.sum() / max(base.sum(), 1e-9)))))

    # --- (1) Mean bars with stddev errorbars: baseline vs masked bytes/frame ---
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    g = np.arange(len(scenarios))
    w = 0.32

    bars0 = ax.bar(
        g - w / 2.0,
        base_means,
        width=w,
        yerr=base_stds,
        capsize=4,
        color="tab:blue",
        alpha=0.85,
        label="baseline (mean ± stddev)",
    )
    bars1 = ax.bar(
        g + w / 2.0,
        mask_means,
        width=w,
        yerr=mask_stds,
        capsize=4,
        color="tab:orange",
        alpha=0.85,
        label="masked (mean ± stddev)",
    )

    xt = [f"{sc.label}\n(n={n})" for sc, n in zip(scenarios, ns)]
    ax.set_xticks(g)
    ax.set_xticklabels(xt, rotation=0)
    ax.set_ylabel("bytes/frame (packet size)")
    ax.set_title("Per-frame bandwidth (mean ± stddev; CRF18 encoder-aligned)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_box, dpi=180)
    plt.close(fig)

    # --- (2) Bar plot: saving percent ---
    deep_blue = "#0B3D91"
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    positions = np.arange(len(scenarios))
    bars = ax.bar(positions, saving_pct, color=deep_blue, alpha=0.90, width=0.55)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([sc.label for sc in scenarios], rotation=0)
    ax.set_ylabel("Saving (%)")
    ax.set_title("Total bandwidth saving % (PTS-aligned; CRF18 encoder-aligned)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for rect, v in zip(bars, saving_pct):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + (0.5 if v >= 0 else -0.8),
            f"{v:.2f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)

    print(f"Wrote: {out_box}")
    print(f"Wrote: {out_bar}")


if __name__ == "__main__":
    main()


