#!/usr/bin/env python3
"""Empirical CDFs for Mario RD (Exp1 shortclips): delivered BW saving vs pure streaming + meta sidecar share.

Reads former points from `rd_suite_tuned_upper_bound_shortclips_points.csv` and grid protocol from
`rd_suite_points.csv` (after merge), clips mario_00..mario_04, respawn variant only.
Writes figure under record/RESPAWN2026/exp5/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP1 = REPO / "record" / "RESPAWN2026" / "exp1_tuned_upper_bound_shortclips"
FORMER_CSV = EXP1 / "rd_suite_tuned_upper_bound_shortclips_points.csv"
GRID_CSV = EXP1 / "rd_suite_points.csv"
OUT_DIR = REPO / "record" / "RESPAWN2026" / "exp5"
CLIP_IDS = tuple(f"mario_0{i}" for i in range(5))


def load_respawn_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keep = {c for c in CLIP_IDS}
    return [r for r in rows if r.get("clip_id") in keep and r.get("variant") == "respawn"]


def fnum(x: str) -> float:
    if x is None or x == "":
        return float("nan")
    return float(x)


def delivered_saving_pct(row: dict[str, str]) -> float:
    base = fnum(row.get("baseline_achieved_bps", ""))
    ach = fnum(row.get("achieved_bps", ""))
    if not (base > 0 and ach == ach):
        return float("nan")
    return 100.0 * (1.0 - ach / base)


def meta_share_pct(row: dict[str, str]) -> float:
    tot = fnum(row.get("achieved_bps", ""))
    meta = fnum(row.get("meta_bps", ""))
    if not (tot > 0 and meta == meta):
        return float("nan")
    return 100.0 * meta / tot


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(f"matplotlib required: {e}") from e

    if not FORMER_CSV.exists():
        raise SystemExit(f"missing {FORMER_CSV}")
    if not GRID_CSV.exists():
        raise SystemExit(f"missing {GRID_CSV}")

    former = load_respawn_rows(FORMER_CSV)
    grid = load_respawn_rows(GRID_CSV)
    if len(former) != 25 or len(grid) != 25:
        print(
            f"warning: expected 25 respawn rows per arm, got former={len(former)} grid={len(grid)}",
            flush=True,
        )

    sf = np.array([delivered_saving_pct(r) for r in former])
    sg = np.array([delivered_saving_pct(r) for r in grid])
    mf = np.array([meta_share_pct(r) for r in former])
    mg = np.array([meta_share_pct(r) for r in grid])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, xf, xg, xlab in (
        (axes[0], sf, sg, "Delivered bitrate saving vs pure streaming (%)"),
        (axes[1], mf, mg, "Sidecar share of delivered bitrate (%)"),
    ):
        xf1, yf = ecdf(xf)
        xg1, yg = ecdf(xg)
        if xf1.size:
            ax.step(np.concatenate([[xf1[0]], xf1]), np.concatenate([[0.0], yf]), where="post", label="Former protocol", color="#1f77b4")
        if xg1.size:
            ax.step(np.concatenate([[xg1[0]], xg1]), np.concatenate([[0.0], yg]), where="post", label="Grid header (v5)", color="#ff7f0e")
        ax.set_xlabel(xlab)
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Mario clips mario_00..mario_04 — Exp1 rate points 8–24 Mbps (25 samples / curve)")
    fig.tight_layout()

    out_png = OUT_DIR / "mario_exp1_former_vs_grid_cdf.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)

    # Text sidecar for captions
    caption = OUT_DIR / "mario_exp1_former_vs_grid_cdf_caption.txt"

    def qs(x: np.ndarray) -> str:
        z = x[np.isfinite(x)]
        if z.size == 0:
            return "n/a"
        return f"median={np.median(z):.2f} p10={np.percentile(z,10):.2f} p90={np.percentile(z,90):.2f}"

    lines = [
        "Mario Exp1 CDF (respawn vs matched pure-streaming baseline)",
        "",
        f"Delivered saving former: {qs(sf)}",
        f"Delivered saving grid:   {qs(sg)}",
        "",
        f"Meta share former: {qs(mf)}",
        f"Meta share grid:   {qs(mg)}",
        "",
        f"Mean VMAF former: {np.nanmean([fnum(r.get('vmaf_mean','')) for r in former]):.2f}",
        f"Mean VMAF grid:   {np.nanmean([fnum(r.get('vmaf_mean','')) for r in grid]):.2f}",
        "",
        f"Mean SSIM former: {np.nanmean([fnum(r.get('ssim_mean','')) for r in former]):.4f}",
        f"Mean SSIM grid:   {np.nanmean([fnum(r.get('ssim_mean','')) for r in grid]):.4f}",
    ]
    caption.write_text("\n".join(lines) + "\n")
    print(f"wrote {caption}", flush=True)


if __name__ == "__main__":
    main()
