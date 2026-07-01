#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from common import RESPAWN2026_DIR, ensure_dir, load_manifest
from msk1 import load_payloads, read_len_prefixed_payloads


def gray_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = float(hist.sum())
    if total <= 1e-12:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def roi_histogram(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
    if hist is None:
        return None
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s <= 1e-12:
        return None
    hist /= s
    return hist


def payload_mask(shape: tuple[int, int], payload: Any | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if payload is None:
        return mask
    for region in payload.regions:
        x0 = max(0, int(region.x))
        y0 = max(0, int(region.y))
        x1 = min(shape[1], int(region.x + region.w))
        y1 = min(shape[0], int(region.y + region.h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def mean_payload_region_area_pct(payloads: list[Any], frame_width: int, frame_height: int) -> float:
    frame_area = max(1, int(frame_width) * int(frame_height))
    values: list[float] = []
    for payload in payloads:
        area = 0
        for region in getattr(payload, "regions", []):
            area += max(0, int(region.w)) * max(0, int(region.h))
        values.append((float(area) / float(frame_area)) * 100.0)
    return float(np.mean(values)) if values else 0.0


def clip_attributes(video_path: Path, payloads: list[Any], *, sample_stride: int = 5) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "motion_magnitude_roi": 0.0,
            "edge_density": 0.0,
            "sobel_energy": 0.0,
            "gray_variance": 0.0,
            "gray_entropy": 0.0,
            "appearance_variability_roi": 0.0,
        }
    prev_gray = None
    prev_hist = None
    idx = 0
    motions: list[float] = []
    edges: list[float] = []
    sobels: list[float] = []
    variances: list[float] = []
    entropies: list[float] = []
    appearances: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_stride != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
        edges.append(edge_density)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobels.append(float(np.mean(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y))))
        variances.append(float(np.var(gray)))
        entropies.append(gray_entropy(gray))
        payload = payloads[idx] if idx < len(payloads) else None
        mask = payload_mask(gray.shape, payload)
        if prev_gray is not None and mask.any():
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(float(np.mean(diff[mask != 0])))
        hist = roi_histogram(frame, mask)
        if hist is not None and prev_hist is not None:
            appearances.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
        if hist is not None:
            prev_hist = hist
        prev_gray = gray
        idx += 1
    cap.release()
    return {
        "motion_magnitude_roi": float(np.mean(motions)) if motions else 0.0,
        "edge_density": float(np.mean(edges)) if edges else 0.0,
        "sobel_energy": float(np.mean(sobels)) if sobels else 0.0,
        "gray_variance": float(np.mean(variances)) if variances else 0.0,
        "gray_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "appearance_variability_roi": float(np.mean(appearances)) if appearances else 0.0,
    }


def cdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([]), np.array([])
    xs = np.sort(np.array(values, dtype=np.float64))
    ys = np.linspace(0.0, 1.0, num=len(xs), endpoint=True)
    return xs, ys


def len_prefixed_sizes(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [4 + len(buf) for buf in read_len_prefixed_payloads(path)]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def quantile_bucket(values: list[float], value: float, labels: tuple[str, str, str]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 3:
        return labels[1]
    q1, q2 = np.quantile(np.array(finite, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


def presence_bucket(value: float) -> str:
    if value >= 0.8:
        return "persistent"
    if value >= 0.3:
        return "mixed"
    return "intermittent"


def appearance_bucket(values: list[float], value: float) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return "stable"
    med = float(np.median(np.array(finite, dtype=np.float64)))
    return "highly_varying" if value > med else "stable"


def scatter_by_style(
    rows: list[dict[str, Any]],
    xkey: str,
    ykey: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    styles = {"photoreal": "tab:blue", "pixel_art": "tab:orange"}
    xs = [safe_float(r[xkey]) for r in rows]
    ys = [safe_float(r[ykey]) for r in rows]
    for style, color in styles.items():
        subset = [r for r in rows if r.get("content_style") == style]
        if not subset:
            continue
        ax.scatter(
            [safe_float(r[xkey]) for r in subset],
            [safe_float(r[ykey]) for r in subset],
            label=style,
            color=color,
            alpha=0.85,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(xs) >= 2 and max(xs) > min(xs):
        z = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 64)
        ax.plot(xx, z[0] * xx + z[1], color="black", linewidth=1.2, label="fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


STYLE_COLORS = {"photoreal": "tab:blue", "pixel_art": "#E6B800"}
GAME_MARKERS = {"fc5": "o", "fm6": "s", "mario": "o"}
GAME_LABELS = {"fc5": "FC5", "fm6": "FM6", "mario": "Mario"}


def generality_scatter_plot(
    rows: list[dict[str, Any]],
    xkey: str,
    ykey: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    *,
    photoreal_only: bool = False,
) -> None:
    if plt is None or not rows:
        return
    plot_rows = [r for r in rows if r.get("content_style") != "pixel_art"] if photoreal_only else list(rows)
    if not plot_rows:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    seen: set[tuple[str, str]] = set()
    for row in plot_rows:
        style = str(row.get("content_style", ""))
        game = str(row.get("game", ""))
        color = STYLE_COLORS.get(style, "tab:gray")
        marker = GAME_MARKERS.get(game, "o")
        label = None
        key = (style, game)
        if key not in seen:
            if photoreal_only:
                label = GAME_LABELS.get(game, game)
            else:
                label = "pixel art" if style == "pixel_art" else GAME_LABELS.get(game, game)
            seen.add(key)
        ax.scatter(
            safe_float(row[xkey]),
            safe_float(row[ykey]),
            color=color,
            marker=marker,
            s=52,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.35,
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_ref_ratio_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Photoreal game averages — too few points for a readable scatter."""
    photo = [r for r in rows if r.get("content_style") == "photoreal"]
    if not photo:
        return
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in photo:
        by_game.setdefault(str(row.get("game", "")), []).append(row)

    md_lines = [
        "# BSP vs Ref ratio (photoreal game averages, CRF23 intake)",
        "",
        "Mario pixel clips omitted: Ref ratio saturates at 1.0 (templates assumed pre-known).",
        "",
        "| Game | Ref ratio | BSP (%) | Masked area (%) | Template reuse |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Game} & \textbf{Ref ratio} & \textbf{BSP} (\%) & \textbf{Masked area} (\%) & \textbf{Template reuse} \\",
        r"\midrule",
    ]
    for game_key in sorted(by_game):
        game_rows = by_game[game_key]
        game = GAME_LABELS.get(game_key, game_key.upper())
        ref_r = float(np.mean([safe_float(row.get("ref_ratio", 0.0)) for row in game_rows]))
        bsp = float(np.mean([safe_float(row.get("bsp_pct", 0.0)) for row in game_rows]))
        area = float(np.mean([safe_float(row.get("avg_masked_area_pct", 0.0)) for row in game_rows]))
        reuse = float(np.mean([safe_float(row.get("template_reuse_rate", 0.0)) for row in game_rows]))
        md_lines.append(
            f"| {game} | {ref_r:.3f} | {bsp:.1f} | {area:.1f} | {reuse:.3f} |"
        )
        tex_lines.append(
            f"{game} & {ref_r:.3f} & {bsp:.1f} & {area:.1f} & {reuse:.3f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    md_lines.append("")

    (out_dir / "bsp_vs_ref_ratio.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_dir / "bsp_vs_ref_ratio.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def envelope_band_plot(
    rows: list[dict[str, Any]],
    xkey: str,
    ykey: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    bins: int = 5,
) -> None:
    if plt is None or not rows:
        return
    positive_rows = [r for r in rows if safe_float(r.get("bsp_pct", 0.0)) > 0.0]
    all_points = [
        (safe_float(r[xkey]), safe_float(r[ykey]), str(r.get("content_style", "")))
        for r in rows
        if math.isfinite(safe_float(r[xkey])) and math.isfinite(safe_float(r[ykey]))
    ]
    points = [
        (safe_float(r[xkey]), safe_float(r[ykey]), str(r.get("content_style", "")))
        for r in positive_rows
        if math.isfinite(safe_float(r[xkey])) and math.isfinite(safe_float(r[ykey]))
    ]
    if len(points) < 2:
        return

    points.sort(key=lambda item: item[0])
    bin_count = min(max(2, bins), len(points))
    chunks = np.array_split(np.array(points, dtype=object), bin_count)
    lower: list[tuple[float, float, str]] = []
    upper: list[tuple[float, float, str]] = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        xs = np.array([float(item[0]) for item in chunk], dtype=np.float64)
        ys = np.array([float(item[1]) for item in chunk], dtype=np.float64)
        x_mid = float(np.mean(xs))
        low_idx = int(np.argmin(ys))
        high_idx = int(np.argmax(ys))
        lower.append((x_mid, float(ys[low_idx]), str(chunk[low_idx][2])))
        upper.append((x_mid, float(ys[high_idx]), str(chunk[high_idx][2])))

    if len(lower) < 2 or len(upper) < 2:
        return

    lower_arr = np.array([(x, y) for x, y, _ in lower], dtype=np.float64)
    upper_arr = np.array([(x, y) for x, y, _ in upper], dtype=np.float64)
    x_min = min(float(lower_arr[:, 0].min()), float(upper_arr[:, 0].min()))
    x_max = max(float(lower_arr[:, 0].max()), float(upper_arr[:, 0].max()))
    xx = np.linspace(x_min, x_max, 128)
    # Piecewise interpolation preserves the observed envelope order. Independent
    # polynomial fits can overshoot and visually cross, especially when many
    # points share similar x-values (e.g., Ref ratio near one).
    lower_curve = np.interp(xx, lower_arr[:, 0], lower_arr[:, 1])
    upper_curve = np.interp(xx, upper_arr[:, 0], upper_arr[:, 1])
    band_low = np.minimum(lower_curve, upper_curve)
    band_high = np.maximum(lower_curve, upper_curve)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    style_colors = {"photoreal": "tab:blue", "pixel_art": "tab:orange"}
    for x, y, style in all_points:
        ax.scatter(
            x,
            y,
            color=style_colors.get(style, "tab:gray"),
            alpha=0.35,
            s=18,
            edgecolors="none",
        )
    ax.fill_between(xx, band_low, band_high, color="tab:blue", alpha=0.16, label="middle region")
    ax.plot(xx, band_high, color="tab:green", linewidth=2.0, label="upper envelope")
    ax.plot(xx, band_low, color="tab:red", linewidth=2.0, label="lower envelope")
    used_labels: set[str] = set()
    for x, y, style in upper:
        label = f"{style} high" if style not in used_labels else None
        ax.scatter(x, y, color=style_colors.get(style, "tab:gray"), marker="^", s=52, edgecolor="black", linewidth=0.4, label=label)
        used_labels.add(style)
    for x, y, style in lower:
        label = f"{style} low" if f"{style}_low" not in used_labels else None
        ax.scatter(x, y, color=style_colors.get(style, "tab:gray"), marker="v", s=52, edgecolor="black", linewidth=0.4, label=label)
        used_labels.add(f"{style}_low")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze generality/failure-mode correlations for offline runs.")
    ap.add_argument("--manifest", type=Path, default=RESPAWN2026_DIR / "manifest" / "offline_manifest.json")
    ap.add_argument("--rd-points", type=Path, default=RESPAWN2026_DIR / "exp1" / "rd_suite_points.csv")
    ap.add_argument("--extra-rd-points", type=Path, action="append", default=[], help="Additional RD CSVs; only games missing from the primary CSV are imported.")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "exp5")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest = {clip.clip_id: clip for clip in load_manifest(args.manifest)}
    rows: list[dict[str, Any]] = []
    rd_rows: list[dict[str, str]] = []
    primary_games: set[str] = set()
    attr_cache: dict[str, dict[str, float]] = {}
    pixel_area_cache: dict[str, float] = {}

    with args.rd_points.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rd_rows.append(row)
            if row.get("variant") == "respawn":
                primary_games.add(str(row.get("game", "")))

    for extra_path in args.extra_rd_points:
        with extra_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("game", "")) not in primary_games:
                    rd_rows.append(row)

    for row in rd_rows:
            if row.get("variant") != "respawn":
                continue
            clip = manifest.get(row["clip_id"])
            if clip is None:
                continue
            run_dir = Path(row["run_dir"])
            report = json.loads((run_dir / "report.json").read_text())
            payloads: list[Any] | None = None
            needs_payloads = clip.clip_id not in attr_cache or (clip.family == "pixel" and clip.clip_id not in pixel_area_cache)
            if needs_payloads:
                payloads = load_payloads(run_dir / "msk1_payloads.bin")
            if clip.clip_id not in attr_cache:
                attr_cache[clip.clip_id] = clip_attributes(Path(clip.normalized_path), payloads or [])
            attr = attr_cache[clip.clip_id]
            per = report.get("per_frame", []) or []
            if clip.family == "pixel":
                if clip.clip_id not in pixel_area_cache:
                    if payloads is None:
                        payloads = load_payloads(run_dir / "msk1_payloads.bin")
                    pixel_area_cache[clip.clip_id] = mean_payload_region_area_pct(payloads, clip.target_width, clip.target_height)
                avg_masked_area = pixel_area_cache[clip.clip_id]
            else:
                avg_masked_area = float(np.mean([float(item.get("masked_alpha_pct", 0.0) or 0.0) for item in per])) if per else 0.0
            object_present_frac = float(np.mean([1.0 if int(item.get("object_present_model", 0) or 0) > 0 else 0.0 for item in per])) if per else 0.0
            ref_ratio = float(np.mean([1.0 if int(item.get("frame_flags", 0) or 0) > 0 else 0.0 for item in per])) if per else 0.0
            reuse_rate = safe_float(report.get("latent_reuse_ratio", None), default=-1.0)
            if reuse_rate < 0.0:
                total_det = safe_float(report.get("total_detections", 0.0), 0.0)
                matched = safe_float(report.get("matched_detections", 0.0), 0.0)
                reuse_rate = (matched / total_det) if total_det > 1e-9 else 0.0
            baseline_bps = float(row.get("baseline_achieved_bps", 0.0) or 0.0)
            respawn_bps = float(row.get("respawn_achieved_bps", 0.0) or 0.0)
            bsp = (1.0 - respawn_bps / baseline_bps) * 100.0 if baseline_bps > 1e-9 else 0.0
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "game": clip.game,
                    "family": clip.family,
                    "content_style": "pixel_art" if clip.family == "pixel" else "photoreal",
                    "source_tag": clip.source_tag,
                    "rate_point_mbps": float(row.get("rate_point_mbps", 0.0) or 0.0),
                    "avg_masked_area_pct": avg_masked_area,
                    "object_present_fraction": object_present_frac,
                    "ref_ratio": ref_ratio,
                    "template_reuse_rate": reuse_rate,
                    "motion_magnitude_roi": attr["motion_magnitude_roi"],
                    "edge_density": attr["edge_density"],
                    "sobel_energy": attr["sobel_energy"],
                    "gray_variance": attr["gray_variance"],
                    "gray_entropy": attr["gray_entropy"],
                    "appearance_variability_roi": attr["appearance_variability_roi"],
                    "bsp_pct": bsp,
                    "roi_vmaf_mean": float(row.get("roi_vmaf_mean", 0.0) or 0.0),
                    "respawn_vmaf_mean": float(row.get("vmaf_mean", 0.0) or 0.0),
                    "roi_ssim_mean": safe_float(row.get("roi_ssim_mean", 0.0), 0.0),
                    "roi_psnr_mean": safe_float(row.get("roi_psnr_mean", 0.0), 0.0),
                    "fullframe_ssim_mean": safe_float(row.get("ssim_mean", 0.0), 0.0),
                    "fullframe_psnr_mean": safe_float(row.get("psnr_mean", 0.0), 0.0),
                    "latency_total_ms": safe_float(((report.get("timing_avg_ms", {}) or {}).get("total")), 0.0),
                    "run_dir": str(run_dir),
                    "baseline_run_dir": str(row.get("baseline_run_dir", "")),
                }
            )

    if rows:
        masked_values = [safe_float(r["avg_masked_area_pct"]) for r in rows]
        appearance_values = [safe_float(r["appearance_variability_roi"]) for r in rows]
        for row in rows:
            row["masked_area_bucket"] = quantile_bucket(masked_values, safe_float(row["avg_masked_area_pct"]), ("small", "medium", "large"))
            row["presence_bucket"] = presence_bucket(safe_float(row["object_present_fraction"]))
            row["appearance_bucket"] = appearance_bucket(appearance_values, safe_float(row["appearance_variability_roi"]))

    out_csv = args.out_dir / "generality_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clip_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if rows:
        generality_scatter_plot(
            rows,
            "avg_masked_area_pct",
            "bsp_pct",
            args.out_dir / "bsp_vs_masked_area.png",
            "Average masked area (%)",
            "BSP (%)",
        )
        write_ref_ratio_table(rows, args.out_dir)
        ref_png = args.out_dir / "bsp_vs_ref_ratio.png"
        if ref_png.exists():
            ref_png.unlink()

        gop_savings_by_game: dict[str, list[float]] = {}
        for row in rows:
            run_dir = Path(row["run_dir"])
            baseline_run_dir = Path(row.get("baseline_run_dir", "")) if row.get("baseline_run_dir") else None
            per_csv = run_dir / "per_frame_metrics.csv"
            baseline_csv = (baseline_run_dir / "per_frame_metrics.csv") if baseline_run_dir else None
            if not per_csv.exists() or baseline_csv is None or not baseline_csv.exists():
                continue
            msk1_sizes = len_prefixed_sizes(run_dir / "msk1_payloads.bin")
            with baseline_csv.open("r", newline="") as fb, per_csv.open("r", newline="") as fr:
                rb = list(csv.DictReader(fb))
                rr = list(csv.DictReader(fr))
                n = min(len(rb), len(rr))
                for start in range(0, n, 60):
                    stop = min(n, start + 60)
                    base_sum = 0.0
                    resp_sum = 0.0
                    for idx in range(start, stop):
                        base_sum += safe_float(rb[idx].get("masked_bytes", 0.0), 0.0)
                        resp_sum += safe_float(rr[idx].get("masked_bytes", 0.0), 0.0)
                        if idx < len(msk1_sizes):
                            resp_sum += float(msk1_sizes[idx])
                    if base_sum > 1e-9:
                        saving_pct = (1.0 - (resp_sum / base_sum)) * 100.0
                        if -50.0 <= saving_pct <= 50.0:
                            game = str(row.get("game", "unknown"))
                            gop_savings_by_game.setdefault(game, []).append(saving_pct)
        if plt is not None and any(values for values in gop_savings_by_game.values()):
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            game_colors = {"fc5": "tab:blue", "fm6": "tab:green", "mario": "#E6B800"}
            for game in sorted(gop_savings_by_game):
                xs, ys = cdf(gop_savings_by_game[game])
                if xs.size == 0:
                    continue
                ax.plot(
                    xs,
                    ys,
                    linewidth=2.0,
                    color=game_colors.get(game, None),
                    label=GAME_LABELS.get(game, game.upper()),
                )
            ax.set_xlim(-50.0, 50.0)
            ax.set_xlabel("Per-GOP savings (%)")
            ax.set_ylabel("CDF")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out_dir / "per_gop_savings_cdf.png")
            plt.close(fig)

        summary_lines = [
            "# Experiment 5 Generality Notes",
            "",
            "This summary is intentionally conservative. It is meant to support the claim that RESPAWN helps most when recurring objects are both large and reusable, and that it degrades gracefully when those conditions fail.",
            "",
            f"- Rows analyzed: {len(rows)}",
            f"- Content styles represented: {', '.join(sorted({str(r['content_style']) for r in rows}))}",
            f"- Masked-area buckets represented: {', '.join(sorted({str(r['masked_area_bucket']) for r in rows}))}",
            f"- Presence buckets represented: {', '.join(sorted({str(r['presence_bucket']) for r in rows}))}",
            f"- Appearance buckets represented: {', '.join(sorted({str(r['appearance_bucket']) for r in rows}))}",
            "",
            "Recommended text direction:",
            "",
            '- "RESPAWN is strongest when recurring objects are large and reusable; when these conditions fail, the system gracefully falls back and rarely harms quality."',
            "",
            "Interpretation guidance:",
            "",
            "- Use `avg_masked_area_pct` and `ref_ratio` as the first-order predictors of savings.",
            "- Use `appearance_variability_roi` and `template_reuse_rate` to explain when reuse becomes unstable or sparse.",
            "- Use `motion_magnitude_roi` against ROI VMAF to explain when motion makes reconstruction harder.",
            "- Use `edge_density`, `sobel_energy`, `gray_variance`, and `gray_entropy` as texture/complexity proxies rather than as hard causal claims.",
            "- Keep the photoreal vs pixel-art comparison descriptive unless enough clips exist in both families to support stronger statistical statements.",
            "- For photoreal clips, `avg_masked_area_pct` is measured from alpha/mask pixels. For pixel-art clips, it is measured from the emitted template region boxes (`sum w*h / frame area`) because the pixel template path does not populate alpha-mask pixels.",
            "- The scatter figure `bsp_vs_masked_area.png` shows styled points only (yellow pixel art, blue photoreal).",
            "- `bsp_vs_ref_ratio.tex` / `.md` tabulate photoreal game averages only; Mario omitted (Ref ratio = 1.0).",
            "- The per-GOP savings CDF drops GOP windows outside [-50%, +50%] and clamps the x-axis to that range.",
        ]
        (args.out_dir / "generality_notes.md").write_text("\n".join(summary_lines) + "\n")

    print(json.dumps({"summary_csv": str(out_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
