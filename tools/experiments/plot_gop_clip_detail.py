#!/usr/bin/env python3
"""Visualize one full clip's GOP bandwidth and its extreme I-frames."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    REPO
    / "record"
    / "RESPAWN2026"
    / "evaluation_revision"
    / "generated_eval_tables"
    / "per_gop_net_bsp.csv"
)


def read_clip_rows(path: Path, clip_id: str) -> list[dict[str, str]]:
    with path.open(newline="") as src:
        rows = [row for row in csv.DictReader(src) if row["clip_id"] == clip_id]
    return sorted(rows, key=lambda row: int(row["gop_index"]))


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_frame(frame: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="fm6_03")
    ap.add_argument("--gop-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--baseline-video", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "record" / "RESPAWN2026" / "evaluation_revision" / "generated_eval_figures",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_clip_rows(args.gop_csv, args.clip_id)
    if not rows:
        raise SystemExit(f"No GOP rows found for {args.clip_id}")

    usage = np.asarray([100.0 - float(row["bsp_pct"]) for row in rows], dtype=float)
    best_idx = int(np.argmin(usage))
    worst_idx = int(np.argmax(usage))
    best = rows[best_idx]
    worst = rows[worst_idx]
    best_frame_idx = int(best["start_frame"])
    worst_frame_idx = int(worst["start_frame"])
    best_frame = read_frame(args.baseline_video, best_frame_idx)
    worst_frame = read_frame(args.baseline_video, worst_frame_idx)

    best_title = (
        f"Most saving: GOP ID {best['gop_index']} (frame {best_frame_idx}), "
        f"{usage[best_idx]:.1f}% of baseline = {100.0 - usage[best_idx]:.1f}% saved"
    )
    worst_title = (
        f"Most costly: GOP ID {worst['gop_index']} (frame {worst_frame_idx}), "
        f"{usage[worst_idx]:.1f}% of baseline = {usage[worst_idx] - 100.0:.1f}% overhead"
    )
    save_frame(
        best_frame,
        args.out_dir / f"{args.clip_id}_gop{best['gop_index']}_most_saving_iframe.png",
        best_title,
    )
    save_frame(
        worst_frame,
        args.out_dir / f"{args.clip_id}_gop{worst['gop_index']}_most_costly_iframe.png",
        worst_title,
    )

    low = min(float(usage.min()), 100.0)
    high = max(float(usage.max()), 100.0)
    norm = TwoSlopeNorm(vmin=low, vcenter=100.0, vmax=high)
    fig = plt.figure(figsize=(14.5, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.35, 3.0], hspace=0.18, wspace=0.06)
    ax_heat = fig.add_subplot(grid[0, :])
    image = ax_heat.imshow(usage[np.newaxis, :], cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_yticks([0], [args.clip_id])
    ax_heat.set_xticks(np.arange(len(rows)), [row["gop_index"] for row in rows])
    ax_heat.set_xlabel("GOP ID (zero-based)")
    for idx, value in enumerate(usage):
        ax_heat.text(idx, 0, f"{value:.0f}", ha="center", va="center", fontsize=7)
    ax_heat.scatter([best_idx, worst_idx], [0, 0], marker="s", s=520, facecolors="none", edgecolors=["lime", "yellow"], linewidths=2.4)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.012)
    colorbar.set_label("Delivered bytes (% baseline)")

    ax_best = fig.add_subplot(grid[1, 0])
    ax_best.imshow(best_frame)
    ax_best.set_title(best_title, fontsize=10)
    ax_best.axis("off")
    ax_worst = fig.add_subplot(grid[1, 1])
    ax_worst.imshow(worst_frame)
    ax_worst.set_title(worst_title, fontsize=10)
    ax_worst.axis("off")
    fig.suptitle(
        f"{args.clip_id}: full-clip GOP bandwidth and baseline I-frames at the extremes\n"
        "Cell numbers are delivered bytes as % of baseline—not GOP IDs.",
        fontsize=14,
    )
    out = args.out_dir / f"{args.clip_id}_gop_bandwidth_with_extreme_iframes"
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(
        {
            "clip_id": args.clip_id,
            "best_gop_id": int(best["gop_index"]),
            "best_start_frame": best_frame_idx,
            "best_usage_pct": float(usage[best_idx]),
            "worst_gop_id": int(worst["gop_index"]),
            "worst_start_frame": worst_frame_idx,
            "worst_usage_pct": float(usage[worst_idx]),
            "figure": str(out.with_suffix(".png")),
        }
    )


if __name__ == "__main__":
    main()
