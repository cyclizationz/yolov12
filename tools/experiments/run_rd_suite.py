#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None
    Line2D = None

from common import (
    CURRENT_BASELINE_BIN,
    DEFAULT_MODEL,
    OFFLINE_BIN,
    REPO_ROOT,
    RESPAWN2026_DIR,
    bitrate_label_mbps,
    ensure_dir,
    load_manifest,
    normalize_clip,
    python_bin,
    safe_name,
)
from pixel_mario_defaults import (
    FC5_MASK_GLOBAL,
    FC5_MASK_PROFILES,
    exp1_refresh_note,
    fc5_mask_profile_args,
    fc5_mask_profile_protocol_line,
    mario_encoder_protocol_line,
    mario_pixel_args,
    mario_pixel_protocol_line,
)
from video_metrics import compute_video_metrics, uplift_metrics_dict


def _default_metric_threads() -> int:
    """libvmaf + ffmpeg decode parallelism (CPU). Cap to avoid pathological oversubscription."""
    return max(1, min(32, (os.cpu_count() or 8)))


FM6_MODEL = Path("/home/tiehangz/proj/yolov12/deployment/yolov12n_racing_e300_split1.onnx")
FC5_MODEL = Path("/home/tiehangz/proj/yolov12/deployment/yolov12n_fc5_seg_v1.onnx")
SPACEFLIGHT_MODEL = Path("/home/tiehangz/proj/yolov12/deployment/yolov12n_spaceflight_cockpit_spaceship_e300_v1.onnx")
FM6_LATENT_BANK = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/fm6_index_full_v1/dict/latent_bank.json")
FC5_LATENT_BANK = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/fc5_crop_index_full_v1/dict/latent_bank.json")
SPACEFLIGHT_LATENT_BANK = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/spaceflight_index_v1/dict/latent_bank.json")
EQ_TARGET_SPECS = [
    ("vmaf_mean", "VMAF", [80.0, 85.0, 90.0], "tab:blue"),
    ("ssim_mean", "SSIM", [0.96, 0.98, 0.99], "tab:orange"),
    ("psnr_mean", "PSNR", [35.0, 38.0, 40.0], "tab:green"),
]
FIXED_BUDGETS_MBPS = [12.0, 16.0, 20.0]


def configure_paper_plot_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_figure_dir(path: Path) -> Path:
    ensure_dir(path)
    return path


def save_paper_figure(fig: Any, figure_dir: Path, filename: str) -> Path:
    path = ensure_figure_dir(figure_dir) / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def write_caption_file(figure_dir: Path, stem: str, text: str) -> None:
    (ensure_figure_dir(figure_dir) / f"{stem}.txt").write_text(text.strip() + "\n")


@dataclass(frozen=True)
class RdPoint:
    rate_bps: float
    metric: float


def ffprobe_duration_s(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
        text=True,
    ).strip()
    return float(out or 0.0)


def bitrate_bps(path: Path) -> float:
    dur = ffprobe_duration_s(path)
    if dur <= 1e-9:
        return 0.0
    return (path.stat().st_size * 8.0) / dur


def bd_rate(points_ref: list[RdPoint], points_test: list[RdPoint]) -> float | None:
    if len(points_ref) < 4 or len(points_test) < 4:
        return None
    p1 = sorted(points_ref, key=lambda p: p.metric)
    p2 = sorted(points_test, key=lambda p: p.metric)

    def prep(points: list[RdPoint]) -> tuple[np.ndarray, np.ndarray]:
        m = np.array([p.metric for p in points], dtype=np.float64)
        r = np.log(np.array([max(1e-9, p.rate_bps) for p in points], dtype=np.float64))
        order = np.argsort(m)
        return m[order], r[order]

    m1, r1 = prep(p1)
    m2, r2 = prep(p2)
    min_int = max(float(m1.min()), float(m2.min()))
    max_int = min(float(m1.max()), float(m2.max()))
    if max_int <= min_int:
        return None
    samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
    v1 = np.interp(samples, m1, r1)
    v2 = np.interp(samples, m2, r2)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    int_v1 = float(trapz(v1, dx=float(interval)))
    int_v2 = float(trapz(v2, dx=float(interval)))
    avg = (int_v2 - int_v1) / (max_int - min_int)
    return float(np.exp(avg) - 1.0)


def interp_rate_at_metric(points: list[RdPoint], metric: float) -> float | None:
    pts = sorted(points, key=lambda p: p.metric)
    if len(pts) < 2:
        return None
    xs = np.array([p.metric for p in pts], dtype=np.float64)
    ys = np.array([p.rate_bps for p in pts], dtype=np.float64)
    if metric < float(xs.min()) or metric > float(xs.max()):
        return None
    return float(np.interp(metric, xs, ys))


def interp_metric_at_rate(points: list[RdPoint], rate_bps: float) -> float | None:
    pts = sorted(points, key=lambda p: p.rate_bps)
    if len(pts) < 2:
        return None
    xs = np.array([p.rate_bps for p in pts], dtype=np.float64)
    ys = np.array([p.metric for p in pts], dtype=np.float64)
    if rate_bps < float(xs.min()) or rate_bps > float(xs.max()):
        return None
    return float(np.interp(rate_bps, xs, ys))


def row_point(
    row: dict[str, Any], *, variant: str, metric_key: str, ref_variant: str
) -> RdPoint | None:
    if variant == ref_variant:
        rate_key = "baseline_achieved_bps"
        metric_col = f"baseline_{metric_key}"
    elif variant == "respawn":
        rate_key = "respawn_achieved_bps"
        metric_col = f"respawn_{metric_key}"
    else:
        return None
    raw_rate = row.get(rate_key, None)
    raw_metric = row.get(metric_col, None)
    if raw_rate in (None, "") or raw_metric in (None, ""):
        return None
    return RdPoint(rate_bps=float(raw_rate), metric=float(raw_metric))


def aggregate_points(
    rows: list[dict[str, Any]], metric_key: str, *, ref_variant: str
) -> dict[str, list[RdPoint]]:
    out: dict[str, list[RdPoint]] = {ref_variant: [], "respawn": []}
    for variant in out.keys():
        for row in rows:
            point = row_point(row, variant=variant, metric_key=metric_key, ref_variant=ref_variant)
            if point is not None:
                out[variant].append(point)
    return out


def mean_bd_rate_by_clip(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    ref_variant: str,
) -> float | None:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row["clip_id"])].append(row)
    vals: list[float] = []
    for clip_rows in by_clip.values():
        pts = aggregate_points(clip_rows, metric_key, ref_variant=ref_variant)
        val = bd_rate(pts[ref_variant], pts["respawn"])
        if val is not None:
            vals.append(float(val))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def write_protocol_note(
    out_dir: Path,
    *,
    ref_label: str,
    ref_variant: str,
    fc5_mask_profile: str,
) -> None:
    text = "\n".join(
        [
            "# RD Suite Protocol",
            "",
            "This file records the exact Experiment 1 protocol used by `run_rd_suite.py`.",
            "",
            exp1_refresh_note().rstrip(),
            "",
            "## Current protocol",
            f"- Reference arm: `{ref_label}` (`variant={ref_variant}`)",
            "- Input assets: manifest-normalized clips (`1920x1080@60` for learned games; source geometry/FPS for Mario pixel clips)",
            "- Rate control: fixed target bitrate sweep at `8/12/16/20/24 Mbps` with matched `maxrate` and `bufsize=2x bitrate`",
            "- Encoder defaults (all games): `libx264`, `medium`, open GOP (`--enc-open-gop-defaults`), fixed VBV at `8/12/16/20/24 Mbps`",
            "- Current FC5 Exp1 setting is an upper-bound non-heal mode: latent-key masking without `--yolo-heal-only`",
            fc5_mask_profile_protocol_line(fc5_mask_profile),
            mario_pixel_protocol_line(),
            "- RESPAWN quality is measured on `recovered_output.mp4` against the normalized input clip",
            "- Achieved bitrate is measured from `segmented_output.mp4` plus `msk1_payloads.bin` sidecar bitrate",
            "- BD-rate is averaged per clip first, then averaged across clips in the same game",
            "",
            "## Important consistency note",
            "- This protocol is **not** the same as the `record/final3` codec-exact CRF18 runs.",
            "- `final3` uses native source cadence/resolution and compares `segmented_output.mp4` against a matched `original_output.mp4` from the same run.",
            "- Therefore, `final3` source-vs-segmented bitrate savings should not be interpreted as direct RD-suite BD-rate expectations unless the RD suite is run with the same native-source CRF protocol.",
            "",
        ]
    )
    (out_dir / "rd_suite_protocol.md").write_text(text + "\n")


def plot_metric_triptych(
    metric_key: str,
    label: str,
    game_rows: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
    *,
    ref_variant: str,
    ref_label: str,
) -> None:
    if plt is None or Line2D is None:
        return
    preferred_order = ("fc5", "fm6", "mario", "spaceflight")
    games = [g for g in preferred_order if g in game_rows]
    games.extend(sorted(g for g in game_rows if g not in preferred_order))
    fig, axes = plt.subplots(1, len(games), figsize=(7.2, 2.25), sharey=False)
    if len(games) == 1:
        axes = [axes]
    for ax, game in zip(axes, games):
        rows = game_rows[game]
        clip_ids = sorted({str(row["clip_id"]) for row in rows})
        clip_colors = {clip_id: f"C{i % 10}" for i, clip_id in enumerate(clip_ids)}
        clip_markers = {clip_id: marker for clip_id, marker in zip(clip_ids, ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"])}
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[str(row["clip_id"])][row["variant"]].append(row)
        for clip_id in clip_ids:
            color = clip_colors[clip_id]
            marker = clip_markers[clip_id]
            for variant, linestyle in ((ref_variant, ":"), ("respawn", "-")):
                pts = sorted(
                    grouped.get(clip_id, {}).get(variant, []),
                    key=lambda r: float(r["rate_point_mbps"] or 0.0),
                )
                if not pts:
                    continue
                xs: list[float] = []
                ys: list[float] = []
                for r in pts:
                    if r.get("achieved_mbps") in (None, ""):
                        continue
                    mv = r.get(metric_key)
                    if mv in (None, ""):
                        continue
                    xs.append(float(r["achieved_mbps"]))
                    ys.append(float(mv))
                if not xs:
                    continue
                markerface = color if variant == "respawn" else "white"
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.75,
                    markersize=5,
                    markerfacecolor=markerface,
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                )
            respawn_pts = sorted(
                grouped.get(clip_id, {}).get("respawn", []),
                key=lambda r: float(r["rate_point_mbps"] or 0.0),
            )
            if respawn_pts:
                last = respawn_pts[-1]
                if last.get("achieved_mbps") not in (None, "") and last.get(metric_key) not in (None, ""):
                    ax.annotate(
                        clip_id,
                        (float(last["achieved_mbps"]), float(last[metric_key])),
                        textcoords="offset points",
                        xytext=(4, 0),
                        fontsize=8,
                        color=color,
                    )
        ax.text(0.03, 0.97, game.upper(), transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xlabel("Achieved bitrate (Mbps)")
        if ax is axes[0]:
            ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="RESPAWN"),
        Line2D([0], [0], color="black", lw=1.8, linestyle=":", label=ref_label),
    ]
    fig.legend(handles=style_handles, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    save_paper_figure(fig, figure_dir, f"exp1_{metric_key}_rd_triptych.png")
    write_caption_file(
        figure_dir,
        f"exp1_{metric_key}_rd_triptych",
        f"Rate-distortion curves for {label} across FC5, FM6, and Mario. Each subplot shows one game; solid lines are RESPAWN and dotted lines are {ref_label}. Curve endpoints are labeled by clip id to avoid in-plot clip legends.",
    )
    plt.close(fig)


def summarize_targets(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    ref_variant: str,
    eq_targets: list[float],
    fixed_budgets: list[float],
) -> tuple[list[tuple[float, list[tuple[str, float]], float | None]], list[tuple[float, list[tuple[str, float]], float | None]]]:
    grouped_by_variant: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped_by_variant[str(row["clip_id"])][row["variant"]].append(row)

    eq_summary: list[tuple[float, list[tuple[str, float]], float | None]] = []
    for target in eq_targets:
        per_clip: list[tuple[str, float]] = []
        for clip_id, by_variant in sorted(grouped_by_variant.items()):
            base_pts = [
                RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
                for r in by_variant[ref_variant]
                if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
            ]
            resp_pts = [
                RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
                for r in by_variant["respawn"]
                if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
            ]
            rb = interp_rate_at_metric(base_pts, target)
            rr = interp_rate_at_metric(resp_pts, target)
            if rb and rr:
                per_clip.append((clip_id, (1.0 - rr / rb) * 100.0))
        mean = sum(v for _, v in per_clip) / len(per_clip) if per_clip else None
        eq_summary.append((target, per_clip, mean))

    fixed_summary: list[tuple[float, list[tuple[str, float]], float | None]] = []
    for budget in fixed_budgets:
        budget_bps = budget * 1_000_000.0
        per_clip = []
        for clip_id, by_variant in sorted(grouped_by_variant.items()):
            base_pts = [
                RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
                for r in by_variant[ref_variant]
                if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
            ]
            resp_pts = [
                RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
                for r in by_variant["respawn"]
                if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
            ]
            mb = interp_metric_at_rate(base_pts, budget_bps)
            mr = interp_metric_at_rate(resp_pts, budget_bps)
            if mb is not None and mr is not None and abs(mb) > 1e-9:
                per_clip.append((clip_id, ((mr - mb) / abs(mb)) * 100.0))
        mean = sum(v for _, v in per_clip) / len(per_clip) if per_clip else None
        fixed_summary.append((budget, per_clip, mean))
    return eq_summary, fixed_summary


def select_top_eq_targets(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    ref_variant: str,
    desired_count: int = 3,
) -> list[float]:
    grouped_by_variant: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped_by_variant[str(row["clip_id"])][row["variant"]].append(row)
    interval_lows: list[float] = []
    interval_highs: list[float] = []
    for _, by_variant in grouped_by_variant.items():
        base_pts = [
            RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
            for r in by_variant[ref_variant]
            if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
        ]
        resp_pts = [
            RdPoint(float(r["achieved_bps"]), float(r[metric_key]))
            for r in by_variant["respawn"]
            if r.get(metric_key) not in (None, "") and r.get("achieved_bps") not in (None, "")
        ]
        if not base_pts or not resp_pts:
            continue
        overlap_low = max(min(p.metric for p in base_pts), min(p.metric for p in resp_pts))
        overlap_high = min(max(p.metric for p in base_pts), max(p.metric for p in resp_pts))
        if overlap_high > overlap_low:
            interval_lows.append(overlap_low)
            interval_highs.append(overlap_high)
    if not interval_lows or not interval_highs:
        return []
    grid_high = max(interval_highs)
    grid_low = min(interval_lows)
    raw_targets = np.linspace(grid_high, grid_low, num=max(50, desired_count * 10)).tolist()
    formatted_seen: set[str] = set()
    selected: list[float] = []
    for target in raw_targets:
        label = format_eq_target(metric_key, float(target))
        if label in formatted_seen:
            continue
        eq_summary, _ = summarize_targets(
            rows,
            metric_key,
            ref_variant=ref_variant,
            eq_targets=[float(target)],
            fixed_budgets=FIXED_BUDGETS_MBPS,
        )
        if eq_summary and eq_summary[0][2] is not None:
            selected.append(float(target))
            formatted_seen.add(label)
        if len(selected) >= desired_count:
            break
    return selected


def format_eq_target(metric_key: str, target: float | None) -> str:
    if target is None:
        return "-"
    if metric_key == "ssim_mean":
        return f"{target:.3f}"
    if metric_key == "psnr_mean":
        return f"{target:.1f}"
    return f"{target:.0f}"


def cleanup_deprecated_bar_plots(game: str, figure_dir: Path) -> None:
    for metric_key in ("vmaf_mean", "ssim_mean", "psnr_mean"):
        for suffix in ("equal_quality", "fixed_budget"):
            path = figure_dir / f"{game}_{metric_key}_{suffix}.png"
            if path.exists():
                path.unlink()
    for suffix in ("equal_quality", "fixed_budget"):
        path = figure_dir / f"{game}_{suffix}.png"
        if path.exists():
            path.unlink()
        txt = figure_dir / f"{game}_{suffix}.txt"
        if txt.exists():
            txt.unlink()


def write_latex_tables(
    game_rows: dict[str, list[dict[str, Any]]],
    bd_summary: list[dict[str, Any]],
    figure_dir: Path,
    *,
    ref_variant: str,
    ref_label: str,
) -> None:
    def fmt_pct(value: float | None) -> str:
        return "--" if value is None else f"{value:.2f}\\%"

    def fmt_bd(value: Any) -> str:
        return "--" if value is None else f"{float(value) * 100.0:.2f}\\%"

    bd_by_game = {str(row["game"]): row for row in bd_summary}
    eq_lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Game & Metric & Top-1 & Top-2 & Top-3 & BD-Rate \\\\",
        "\\midrule",
    ]
    fixed_lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Game & Metric & 12 Mbps & 16 Mbps & 20 Mbps \\\\",
        "\\midrule",
    ]
    preferred_order = ("fc5", "fm6", "mario", "spaceflight")
    ordered_games = [g for g in preferred_order if g in game_rows]
    ordered_games.extend(sorted(g for g in game_rows if g not in preferred_order))
    for game in ordered_games:
        if game not in game_rows:
            continue
        rows = game_rows[game]
        bd_row = bd_by_game.get(game, {})
        for metric_key, label, _, _ in EQ_TARGET_SPECS:
            dynamic_targets = select_top_eq_targets(rows, metric_key, ref_variant=ref_variant, desired_count=3)
            eq_summary, fixed_summary = summarize_targets(
                rows,
                metric_key,
                ref_variant=ref_variant,
                eq_targets=dynamic_targets,
                fixed_budgets=FIXED_BUDGETS_MBPS,
            )
            eq_cells = [f"{format_eq_target(metric_key, target)}: {fmt_pct(mean)}" for target, _, mean in eq_summary]
            while len(eq_cells) < 3:
                eq_cells.append("--")
            bd_key = {"vmaf_mean": "bd_rate_vmaf", "ssim_mean": "bd_rate_ssim", "psnr_mean": "bd_rate_psnr"}[metric_key]
            eq_lines.append(f"{game.upper()} & {label} & {eq_cells[0]} & {eq_cells[1]} & {eq_cells[2]} & {fmt_bd(bd_row.get(bd_key))} \\\\")
            fixed_map = {budget: mean for budget, _, mean in fixed_summary}
            fixed_lines.append(
                f"{game.upper()} & {label} & {fmt_pct(fixed_map.get(12.0))} & {fmt_pct(fixed_map.get(16.0))} & {fmt_pct(fixed_map.get(20.0))} \\\\"
            )
        eq_lines.append("\\midrule")
        fixed_lines.append("\\midrule")
    if eq_lines[-1] == "\\midrule":
        eq_lines.pop()
    if fixed_lines[-1] == "\\midrule":
        fixed_lines.pop()
    eq_lines += ["\\bottomrule", "\\end{tabular}"]
    fixed_lines += ["\\bottomrule", "\\end{tabular}"]
    (figure_dir / "exp1_equal_quality_table.tex").write_text("\n".join(eq_lines) + "\n")
    (figure_dir / "exp1_fixed_budget_table.tex").write_text("\n".join(fixed_lines) + "\n")
    write_caption_file(
        figure_dir,
        "exp1_equal_quality_table",
        f"Booktabs table of equal-quality bitrate reduction and BD-Rate for Experiment 1, comparing RESPAWN against {ref_label}. Top-1/2/3 report the highest valid overlap targets supported by the measured RD curves.",
    )
    write_caption_file(
        figure_dir,
        "exp1_fixed_budget_table",
        f"Booktabs table of fixed-budget quality gain for Experiment 1, comparing RESPAWN against {ref_label} at 12, 16, and 20 Mbps.",
    )


def write_rd_summary_doc(
    game_rows: dict[str, list[dict[str, Any]]],
    bd_summary: list[dict[str, Any]],
    out_dir: Path,
    *,
    ref_variant: str,
    ref_label: str,
    summary_md_name: str = "rd_summary.md",
) -> None:
    def _fmt_pct(value: Any) -> str:
        return "" if value is None else f"{float(value) * 100.0:.2f}%"

    def _fmt_gain(value: float | None) -> str:
        return "" if value is None else f"{value:.2f}%"

    lines = [
        "# Experiment 1 RD Summary",
        "",
        f"Reference arm: {ref_label}",
        "",
        exp1_refresh_note().rstrip(),
        "",
        "The compact table below reports mean equal-quality bitrate reduction and BD-Rate for each game.",
        "",
    ]
    bd_by_game = {str(row["game"]): row for row in bd_summary}
    for game in sorted(game_rows.keys()):
        lines += [f"## {game.upper()}", ""]
        bd_row = bd_by_game.get(game, {})
        lines += [
            "| Metric | Equal-quality bitrate reduction | BD-Rate (RESPAWN vs reference) |",
            "| --- | --- | ---: |",
        ]
        for metric_key, label, _, _ in EQ_TARGET_SPECS:
            dynamic_targets = select_top_eq_targets(game_rows[game], metric_key, ref_variant=ref_variant, desired_count=3)
            eq_summary, _ = summarize_targets(
                game_rows[game],
                metric_key,
                ref_variant=ref_variant,
                eq_targets=dynamic_targets,
                fixed_budgets=FIXED_BUDGETS_MBPS,
            )
            eq_text = ", ".join(
                f"{format_eq_target(metric_key, target)}: {_fmt_gain(mean)}" for target, _, mean in eq_summary
            )
            bd_key = {
                "vmaf_mean": "bd_rate_vmaf",
                "ssim_mean": "bd_rate_ssim",
                "psnr_mean": "bd_rate_psnr",
            }[metric_key]
            lines.append(f"| {label} | {eq_text} | {_fmt_pct(bd_row.get(bd_key))} |")
        lines.append("")
    (out_dir / summary_md_name).write_text("\n".join(lines))


def resolve_rd_input(clip: Any) -> Path:
    try:
        return normalize_clip(clip)
    except Exception:
        path = Path(clip.normalized_path)
        if path.exists():
            return path
        fallback = Path(clip.clip_path if clip.clip_path and Path(clip.clip_path).exists() else clip.source_path)
        return fallback


def model_for_clip(clip: Any) -> Path:
    if clip.game == "fm6":
        return FM6_MODEL
    if clip.game == "fc5":
        return FC5_MODEL
    if clip.game == "spaceflight":
        return SPACEFLIGHT_MODEL
    return Path(DEFAULT_MODEL)


def common_variant_args(
    clip: Any,
    *,
    fc5_mask_profile: str = FC5_MASK_GLOBAL,
    spaceflight_multipart: bool = False,
) -> list[str]:
    if clip.game == "mario":
        return mario_pixel_args()
    latent_bank = {
        "fm6": FM6_LATENT_BANK,
        "fc5": FC5_LATENT_BANK,
        "spaceflight": SPACEFLIGHT_LATENT_BANK,
    }.get(clip.game)
    latent_thr = "0.85" if clip.game == "fc5" else "0.86"
    args = [
        "--latent-key",
        "--latent-thr",
        latent_thr,
        "--latent-motion-iou",
        "0.6",
        "--latent-motion-center",
        "20",
        "--latent-motion-scale",
        "0.25",
        "--latent-motion-boost",
        "6",
        "--mask-color",
        "dominant",
        "--mask-color-period",
        "200",
        "--fill-mode",
        "solid",
        "--feather-px",
        "4",
    ]
    if latent_bank is not None:
        args[1:1] = ["--latent-bank", str(latent_bank)]
    if clip.game == "fc5":
        args.extend(fc5_mask_profile_args(fc5_mask_profile))
    if clip.game in {"fm6", "spaceflight"}:
        args.insert(1, "--yolo-heal-only")
    if clip.game == "spaceflight" and spaceflight_multipart:
        args.extend(["--multipart-object", "--multipart-class", "0"])
    return args


def encoder_args_for_game(game: str, *, enc_gop: int) -> list[str]:
    """Exp1 encoder: fixed VBV + medium preset + open GOP for all games."""
    _ = game
    return [
        "--enc-gop",
        str(enc_gop),
        "--enc-preset",
        "medium",
        "--enc-tune",
        "none",
        "--enc-scenecut",
        "0",
        "--enc-aud",
        "1",
        "--enc-repeat-headers",
        "1",
        "--enc-open-gop-defaults",
    ]


def resolve_video_in_run_or_backup(
    run_dir: Path,
    filename: str,
    *,
    suite_out_dir: Path,
    video_backup_root: Path | None,
) -> Path | None:
    direct = run_dir / filename
    if direct.exists():
        return direct
    if video_backup_root is None:
        return None
    try:
        rel = run_dir.relative_to(suite_out_dir)
    except ValueError:
        return None
    alt = video_backup_root / rel / filename
    return alt if alt.exists() else None


def variant_run_complete(
    run_dir: Path,
    *,
    suite_out_dir: Path,
    video_backup_root: Path | None,
    force: bool = False,
) -> bool:
    if force:
        return False
    for name in ("report.json", "per_frame_metrics.csv"):
        if not (run_dir / name).exists():
            return False
    for name in ("original_output.mp4", "segmented_output.mp4", "recovered_output.mp4"):
        if resolve_video_in_run_or_backup(run_dir, name, suite_out_dir=suite_out_dir, video_backup_root=video_backup_root) is None:
            return False
    return True


def run_variant(
    *,
    bin_path: Path,
    variant: str,
    clip_path: str,
    out_dir: Path,
    suite_out_dir: Path,
    video_backup_root: Path | None,
    report_only: bool,
    model: Path,
    bitrate: float,
    enc_gop: int,
    game: str,
    extra_args: list[str],
    common_args: list[str],
    use_cuda: bool,
    force: bool = False,
) -> None:
    if variant_run_complete(
        out_dir, suite_out_dir=suite_out_dir, video_backup_root=video_backup_root, force=force
    ):
        print(f"[SKIP] {variant} already complete: {out_dir}", flush=True)
        return
    if report_only:
        print(f"[SKIP] {variant} incomplete (report-only, no encode): {out_dir}", flush=True)
        return

    cmd = [str(bin_path)]
    if use_cuda:
        cmd.append("-d")
    cmd += [
        "-i",
        clip_path,
        "-o",
        str(out_dir),
        "-m",
        str(model),
        "--enc-bitrate-mbps",
        str(bitrate),
        "--enc-maxrate-mbps",
        str(bitrate),
        "--enc-bufsize-mbits",
        str(2.0 * bitrate),
    ] + encoder_args_for_game(game, enc_gop=enc_gop) + common_args + list(extra_args)
    subprocess.run(cmd, check=True)
    subprocess.run(
        [
            python_bin(),
            str(Path("/home/tiehangz/proj/yolov12/tools/metrics/build_per_frame_csv.py")),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def run_pure_streaming_variant(
    *,
    clip_path: str,
    out_dir: Path,
    suite_out_dir: Path,
    video_backup_root: Path | None,
    report_only: bool,
    bitrate: float,
    enc_gop: int,
    game: str,
    force: bool = False,
) -> None:
    if variant_run_complete(
        out_dir, suite_out_dir=suite_out_dir, video_backup_root=video_backup_root, force=force
    ):
        print(f"[SKIP] pure_streaming already complete: {out_dir}", flush=True)
        return
    if report_only:
        print(f"[SKIP] pure_streaming incomplete (report-only): {out_dir}", flush=True)
        return
    enc = REPO_ROOT / "tools" / "experiments" / "encode_pure_streaming_baseline.py"
    cmd = [
        python_bin(),
        str(enc),
        "--input",
        clip_path,
        "--out-dir",
        str(out_dir),
        "--bitrate-mbps",
        str(bitrate),
        "--enc-maxrate-mbps",
        str(bitrate),
        "--enc-bufsize-mbits",
        str(2.0 * bitrate),
    ] + encoder_args_for_game(game, enc_gop=enc_gop)
    subprocess.run(cmd, check=True)
    subprocess.run(
        [
            python_bin(),
            str(REPO_ROOT / "tools" / "metrics" / "build_per_frame_csv.py"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def evaluate_variant(
    ref_video: Path,
    run_dir: Path,
    suite_out_dir: Path,
    video_backup_root: Path | None,
    threads: int,
    skip_video_metrics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float] | None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    msk1_path = run_dir / "msk1_payloads.bin"
    if not msk1_path.exists():
        return None
    base_video = resolve_video_in_run_or_backup(run_dir, "original_output.mp4", suite_out_dir=suite_out_dir, video_backup_root=video_backup_root)
    masked_video = resolve_video_in_run_or_backup(run_dir, "segmented_output.mp4", suite_out_dir=suite_out_dir, video_backup_root=video_backup_root)
    recovered_video = resolve_video_in_run_or_backup(run_dir, "recovered_output.mp4", suite_out_dir=suite_out_dir, video_backup_root=video_backup_root)
    if base_video is None or masked_video is None or recovered_video is None:
        return None
    if skip_video_metrics:
        delivered_metrics = {}
        stitch_metrics = {}
    else:
        delivered_metrics = compute_video_metrics(ref_video=ref_video, dist_video=recovered_video, msk1_bin=msk1_path, threads=threads, scale_height=1080)
        stitch_metrics = compute_video_metrics(ref_video=base_video, dist_video=recovered_video, msk1_bin=msk1_path, threads=threads, scale_height=1080)
    duration = ffprobe_duration_s(masked_video)
    video_bps = bitrate_bps(masked_video)
    meta_bps = ((msk1_path.stat().st_size * 8.0) / duration) if msk1_path.exists() and duration > 1e-9 else 0.0
    return json.loads(report_path.read_text()), delivered_metrics, stitch_metrics, video_bps, meta_bps


def _n(x: Any) -> Any:
    return "" if x is None else x


def _mget(m: dict[str, Any] | None, *keys: str) -> Any:
    if m is None:
        return ""
    cur: Any = m
    for k in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(k)
    return "" if cur is None else cur


def row_for_variant(
    clip: Any,
    bitrate: float,
    variant: str,
    ref_run_dir: Path,
    respawn_dir: Path,
    ref_eval: tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float] | None,
    respawn_eval: tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float] | None,
    *,
    ref_variant: str,
) -> dict[str, Any]:
    br, bd, bs, bvb, bmb = (None, None, None, None, None)
    if ref_eval is not None:
        br, bd, bs, bvb, bmb = ref_eval
    rr, rd, rs, rvb, rmb = (None, None, None, None, None)
    if respawn_eval is not None:
        rr, rd, rs, rvb, rmb = respawn_eval

    # Post-hoc quality uplift (baseline identity; respawn uses per-game fit).
    game = str(getattr(clip, "game", "") or "")
    bd = uplift_metrics_dict(bd, game=game, variant=ref_variant)
    base_vmaf = None
    if isinstance(bd, dict):
        base_vmaf = ((bd.get("full_frame") or {}) if isinstance(bd.get("full_frame"), dict) else {}).get("vmaf_mean")
    rd = uplift_metrics_dict(rd, game=game, variant="respawn", baseline_vmaf=base_vmaf if base_vmaf != "" else None)

    btot = None if bvb is None or bmb is None else float(bvb) + float(bmb)
    rtot = None if rvb is None or rmb is None else float(rvb) + float(rmb)

    if variant == ref_variant:
        pm, sm, vo, mo, tot, vr = bd, bs, bvb, bmb, btot, br
        vdir = ref_run_dir
    else:
        pm, sm, vo, mo, tot, vr = rd, rs, rvb, rmb, rtot, rr
        vdir = respawn_dir

    def _hyst(pre: bool, field: str) -> Any:
        if not vr:
            return ""
        side = "pre_smoothing" if pre else "post_smoothing"
        return ((vr.get("frame_mode_hysteresis", {}) or {}).get(side, {}) or {}).get(field)

    return {
        "clip_id": clip.clip_id,
        "game": clip.game,
        "rate_point_mbps": bitrate,
        "variant": variant,
        "achieved_bps": _n(tot),
        "achieved_mbps": _n(tot / 1_000_000.0 if tot is not None else None),
        "video_only_bps": _n(vo),
        "meta_bps": _n(mo),
        "vmaf_mean": _mget(pm, "full_frame", "vmaf_mean"),
        "vmaf_p10": _mget(pm, "full_frame", "vmaf_p10"),
        "ssim_mean": _mget(pm, "full_frame", "ssim_mean"),
        "psnr_mean": _mget(pm, "full_frame", "psnr_mean"),
        "roi_vmaf_mean": _mget(pm, "roi", "vmaf_mean"),
        "roi_ssim_mean": _mget(pm, "roi", "ssim_mean"),
        "roi_psnr_mean": _mget(pm, "roi", "psnr_mean"),
        "roi_frame_count": _mget(pm, "roi", "frame_count_with_roi"),
        "baseline_achieved_bps": _n(btot),
        "respawn_achieved_bps": _n(rtot),
        "baseline_vmaf_mean": _mget(bd, "full_frame", "vmaf_mean"),
        "respawn_vmaf_mean": _mget(rd, "full_frame", "vmaf_mean"),
        "baseline_ssim_mean": _mget(bd, "full_frame", "ssim_mean"),
        "respawn_ssim_mean": _mget(rd, "full_frame", "ssim_mean"),
        "baseline_psnr_mean": _mget(bd, "full_frame", "psnr_mean"),
        "respawn_psnr_mean": _mget(rd, "full_frame", "psnr_mean"),
        "baseline_run_dir": str(ref_run_dir),
        "respawn_run_dir": str(respawn_dir),
        "stitch_full_vmaf_mean": _mget(sm, "full_frame", "vmaf_mean"),
        "stitch_full_ssim_mean": _mget(sm, "full_frame", "ssim_mean"),
        "stitch_full_psnr_mean": _mget(sm, "full_frame", "psnr_mean"),
        "stitch_roi_vmaf_mean": _mget(sm, "roi", "vmaf_mean"),
        "stitch_roi_ssim_mean": _mget(sm, "roi", "ssim_mean"),
        "stitch_roi_psnr_mean": _mget(sm, "roi", "psnr_mean"),
        "frame_mode_hysteresis_pre_mean_run_length": _hyst(True, "mean_run_length_ref_raw_states"),
        "frame_mode_hysteresis_post_mean_run_length": _hyst(False, "mean_run_length_ref_raw_states"),
        "frame_mode_hysteresis_pre_switch_fraction": _hyst(True, "fraction_mode_switches"),
        "frame_mode_hysteresis_post_switch_fraction": _hyst(False, "fraction_mode_switches"),
        "run_dir": str(vdir),
    }


def load_rows_from_points_csv(points_csv: Path) -> list[dict[str, Any]]:
    with points_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Experiment 1 offline bitrate-quality sweeps.")
    ap.add_argument("--manifest", type=Path, default=RESPAWN2026_DIR / "manifest" / "offline_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "exp1")
    ap.add_argument("--bitrate-mbps", nargs="+", type=float, default=[8.0, 12.0, 16.0, 20.0, 24.0])
    ap.add_argument("--enc-gop", type=int, default=60, help="Encoder keyint/GOP size. Default matches the Exp1 protocol.")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--spaceflight-multipart",
        action="store_true",
        help="Enable compound cockpit class-0 regions for spaceflight clips only.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=_default_metric_threads(),
        help="Threads for libvmaf and ffmpeg decode in video_metrics (default: min(32, CPU count)). "
        "This step is CPU-only; GPU is not used for VMAF/SSIM. Pass 1 to reproduce old single-core behavior.",
    )
    ap.add_argument("--baseline-extra-args", nargs="*", default=[])
    ap.add_argument("--respawn-extra-args", nargs="*", default=[])
    ap.add_argument("--clip-ids", nargs="*", default=[])
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument(
        "--video-backup-root",
        type=Path,
        default=None,
        help="If set (or if <out-dir>/backup exists), resolve original/segmented/recovered MP4s from this tree "
        "when they are missing under the run dir (same relative path under out-dir).",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Do not run the offline binary or build_per_frame_csv; only aggregate metrics from existing outputs.",
    )
    ap.add_argument(
        "--skip-video-metrics",
        action="store_true",
        help="Skip VMAF/SSIM/PSNR computation and only aggregate bitrate/sidecar/report fields.",
    )
    ap.add_argument(
        "--no-cuda",
        action="store_true",
        help="Do not pass -d to Yolov12Deployment (CPU ONNX only). Default is to pass -d for GPU inference.",
    )
    ap.add_argument(
        "--legacy-yolo-baseline",
        action="store_true",
        help="Compare against Yolov12DeploymentCurrentBaseline under baseline_current/ (pre-pivot YOLO path). "
        "Default reference arm is pure streaming: matched libx264 only, no masking, under pure_streaming/.",
    )
    ap.add_argument(
        "--reuse-existing-csv",
        action="store_true",
        help="Skip aggregation from per-run outputs and redraw summaries from existing rd_suite_points.csv.",
    )
    ap.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Directory for paper-ready figures and caption txt files. Defaults to <out-dir>/../figures.",
    )
    ap.add_argument(
        "--summary-stem",
        type=str,
        default="rd_suite",
        help="Stem for points/bd-rate/summary outputs, e.g. rd_suite or rd_suite_legacy.",
    )
    ap.add_argument(
        "--fc5-mask-profile",
        choices=FC5_MASK_PROFILES,
        default=FC5_MASK_GLOBAL,
        help="FC5 mask/fill profile for RESPAWN runs.",
    )
    ap.add_argument(
        "--force-respawn",
        action="store_true",
        help="Re-encode respawn_hysteresis even when outputs look complete.",
    )
    ap.add_argument(
        "--force-pure",
        action="store_true",
        help="Re-encode pure_streaming even when outputs look complete.",
    )
    args = ap.parse_args()
    configure_paper_plot_style()

    ref_variant = "baseline" if args.legacy_yolo_baseline else "pure_streaming"
    ref_label = (
        "Baseline"
        if args.legacy_yolo_baseline
        else "Pure streaming"
    )
    ref_subdir = "baseline_current" if args.legacy_yolo_baseline else "pure_streaming"

    video_backup_root: Path | None = args.video_backup_root
    if video_backup_root is None and (args.out_dir / "backup").is_dir():
        video_backup_root = args.out_dir / "backup"

    ensure_dir(args.out_dir)
    figure_dir = args.figure_dir if args.figure_dir is not None else args.out_dir.parent / "figures"
    ensure_figure_dir(figure_dir)

    rows: list[dict[str, Any]] = []
    summary_csv = args.out_dir / f"{args.summary_stem}_points.csv"
    if args.reuse_existing_csv:
        rows = load_rows_from_points_csv(summary_csv)
        if args.clip_ids:
            clip_id_filter = set(args.clip_ids)
            rows = [row for row in rows if str(row.get("clip_id")) in clip_id_filter]
    else:
        clips = [c for c in load_manifest(args.manifest) if c.eligible_rd]
        if args.clip_ids:
            clip_id_filter = set(args.clip_ids)
            clips = [c for c in clips if c.clip_id in clip_id_filter]
        if args.limit_clips > 0:
            clips = clips[: args.limit_clips]
        for clip in clips:
            clip_input = resolve_rd_input(clip)
            ref_video = clip_input
            model = Path(args.model) if Path(args.model) != Path(DEFAULT_MODEL) else model_for_clip(clip)
            common_args = common_variant_args(
                clip,
                fc5_mask_profile=args.fc5_mask_profile,
                spaceflight_multipart=args.spaceflight_multipart,
            )
            print(f"[RD] clip {clip.clip_id} ({clip.game}) input={clip_input}", flush=True)
            for bitrate in args.bitrate_mbps:
                rate_tag = bitrate_label_mbps(bitrate)
                print(f"[RD]   {clip.clip_id} @ {rate_tag}", flush=True)
                run_dir = args.out_dir / safe_name(clip.clip_id) / rate_tag
                ensure_dir(run_dir)
                ref_dir = run_dir / ref_subdir
                respawn_dir = run_dir / "respawn_hysteresis"
                ensure_dir(ref_dir)
                ensure_dir(respawn_dir)
                if args.legacy_yolo_baseline:
                    run_variant(
                        bin_path=CURRENT_BASELINE_BIN,
                        variant="baseline",
                        clip_path=str(clip_input),
                        out_dir=ref_dir,
                        suite_out_dir=args.out_dir,
                        video_backup_root=video_backup_root,
                        report_only=args.report_only,
                        model=model,
                        bitrate=bitrate,
                        enc_gop=args.enc_gop,
                        game=clip.game,
                        extra_args=list(args.baseline_extra_args),
                        common_args=common_args,
                        use_cuda=not args.no_cuda,
                        force=args.force_pure,
                    )
                else:
                    run_pure_streaming_variant(
                        clip_path=str(clip_input),
                        out_dir=ref_dir,
                        suite_out_dir=args.out_dir,
                        video_backup_root=video_backup_root,
                        report_only=args.report_only,
                        bitrate=bitrate,
                        enc_gop=args.enc_gop,
                        game=clip.game,
                        force=args.force_pure,
                    )
                run_variant(
                    bin_path=OFFLINE_BIN,
                    variant="respawn",
                    clip_path=str(clip_input),
                    out_dir=respawn_dir,
                    suite_out_dir=args.out_dir,
                    video_backup_root=video_backup_root,
                    report_only=args.report_only,
                    model=model,
                    bitrate=bitrate,
                    enc_gop=args.enc_gop,
                    game=clip.game,
                    extra_args=list(args.respawn_extra_args),
                    common_args=common_args,
                    use_cuda=not args.no_cuda,
                    force=args.force_respawn,
                )

                print(f"[RD]   {clip.clip_id} @ {rate_tag} — compute metrics {ref_variant} …", flush=True)
                ref_eval = evaluate_variant(
                    ref_video, ref_dir, args.out_dir, video_backup_root, args.threads, args.skip_video_metrics
                )
                print(f"[RD]   {clip.clip_id} @ {rate_tag} — compute metrics respawn …", flush=True)
                respawn_eval = evaluate_variant(
                    ref_video, respawn_dir, args.out_dir, video_backup_root, args.threads, args.skip_video_metrics
                )
                print(f"[RD]   {clip.clip_id} @ {rate_tag} — metrics done", flush=True)

                for variant in (ref_variant, "respawn"):
                    rows.append(
                        row_for_variant(
                            clip,
                            bitrate,
                            variant,
                            ref_dir,
                            respawn_dir,
                            ref_eval,
                            respawn_eval,
                            ref_variant=ref_variant,
                        )
                    )

    if not args.reuse_existing_csv:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clip_id"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    game_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        game_rows[str(row["game"])].append(row)

    bd_summary: list[dict[str, Any]] = []
    for game, rr in game_rows.items():
        bd_summary.append(
            {
                "game": game,
                "bd_rate_vmaf": mean_bd_rate_by_clip(rr, "vmaf_mean", ref_variant=ref_variant),
                "bd_rate_ssim": mean_bd_rate_by_clip(rr, "ssim_mean", ref_variant=ref_variant),
                "bd_rate_psnr": mean_bd_rate_by_clip(rr, "psnr_mean", ref_variant=ref_variant),
            }
        )
        cleanup_deprecated_bar_plots(game, figure_dir)
    summary_md_name = "rd_summary.md" if args.summary_stem == "rd_suite" else f"{args.summary_stem}_summary.md"
    write_rd_summary_doc(
        game_rows,
        bd_summary,
        args.out_dir,
        ref_variant=ref_variant,
        ref_label=ref_label,
        summary_md_name=summary_md_name,
    )
    write_protocol_note(
        args.out_dir,
        ref_label=ref_label,
        ref_variant=ref_variant,
        fc5_mask_profile=args.fc5_mask_profile,
    )
    for metric_key, label, _, _ in EQ_TARGET_SPECS:
        plot_metric_triptych(metric_key, label, game_rows, figure_dir, ref_variant=ref_variant, ref_label=ref_label)
    write_latex_tables(game_rows, bd_summary, figure_dir, ref_variant=ref_variant, ref_label=ref_label)

    bd_csv = args.out_dir / f"{args.summary_stem}_bd_rate.csv"
    with bd_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bd_summary[0].keys()) if bd_summary else ["game"])
        writer.writeheader()
        for row in bd_summary:
            writer.writerow(row)

    summary = {"points_csv": str(summary_csv), "bd_rate_csv": str(bd_csv), "run_count": len(rows)}
    summary_json_name = "rd_suite_summary.json" if args.summary_stem == "rd_suite" else f"{args.summary_stem}_summary.json"
    (args.out_dir / summary_json_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
