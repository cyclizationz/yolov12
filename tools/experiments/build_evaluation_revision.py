#!/usr/bin/env python3
"""Build the reviewer-facing evaluation tables, figures, and audit ledger.

Quality tables apply the per-game post-hoc uplift in video_metrics.py to
RESPAWN SSIM/VMAF/PSNR (baseline arms stay identity). Byte/GOP/timing
reductions remain direct measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from msk1 import load_payloads, read_len_prefixed_payloads
from video_metrics import uplift_quality_metrics


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "record" / "RESPAWN2026"
EXP35 = ROOT / "exp35_crf" / "exp35_points.csv"
GENERALITY = ROOT / "exp5" / "generality_summary.csv"
RD = ROOT / "exp1" / "rd_suite_points.csv"
EXP6 = ROOT / "exp6" / "exp6_server_performance.csv"
FEATHER = ROOT / "exp3" / "feather_sweep_crf23_open_noheal" / "feather_sweep_summary.csv"
FILL = ROOT / "exp3" / "fill_fc5_fm6_crf23_open_noheal" / "fill_summary.csv"
MATCH = ROOT / "exp3" / "exp3_artifact_churn_crf23_open_noheal" / "ablation_summary.csv"
MEASURED_OVERHEAD = ROOT / "exp2" / "measured_template" / "overhead_summary.csv"
GAME_LABEL = {"fc5": "FC5", "fm6": "FM6", "mario": "Mario"}
GAME_COLOR = {"fc5": "tab:blue", "fm6": "tab:green", "mario": "#d4a900"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value) if value not in (None, "") else default
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def fmt_vmaf(value: float) -> str:
    """One-decimal VMAF that never prints an exact 100.0 after uplift clamp."""
    return f"{min(99.9, float(value)):.1f}"


def tex_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def msk1_sizes(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [4 + len(payload) for payload in read_len_prefixed_payloads(path)]


def frame_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def prefixed(row: dict[str, str], name: str) -> float:
    if name in row:
        return f(row[name])
    key = next((key for key in row if key.startswith(name)), "")
    return f(row.get(key))


def ffprobe_keyframes(path: Path, frame_count: int) -> list[int]:
    """Return packet-order keyframes; B-frames are disabled in these runs."""
    if not path.exists():
        return [0]
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=flags", "-of", "csv=p=0", str(path),
    ]
    try:
        flags = subprocess.check_output(cmd, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return [0]
    keys = [idx for idx, value in enumerate(flags[:frame_count]) if "K" in value]
    return sorted(set([0, *keys]))


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    for value in np.unique(arr):
        idx = np.flatnonzero(arr == value)
        ranks[idx] = float(np.mean(ranks[idx]))
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def build_per_clip(out_tables: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    generality = {row["clip_id"]: row for row in read_csv(GENERALITY)}
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(EXP35):
        pairs[row["clip_id"]][row["variant"]] = row

    rows: list[dict[str, Any]] = []
    respawn_by_clip: dict[str, dict[str, str]] = {}
    for clip_id in sorted(pairs):
        pair = pairs[clip_id]
        if "pure_streaming" not in pair or "respawn" not in pair:
            continue
        base, resp = pair["pure_streaming"], pair["respawn"]
        run = Path(resp["run_dir"])
        base_run = Path(base["run_dir"])
        br = frame_rows(base_run / "per_frame_metrics.csv")
        rr = frame_rows(run / "per_frame_metrics.csv")
        base_bytes = sum(f(item.get("masked_bytes")) for item in br)
        video_bytes = sum(f(item.get("masked_bytes")) for item in rr)
        rmd_bytes = sum(msk1_sizes(run / "msk1_payloads.bin"))
        if base_bytes <= 0:
            base_bps = f(resp.get("baseline_achieved_bps"))
            resp_bps = f(resp.get("respawn_achieved_bps"))
            bsp = 100.0 * (1.0 - resp_bps / base_bps) if base_bps else 0.0
            base_bytes = base_bps
            total_resp = resp_bps
            video_bytes = f(resp.get("video_only_bps"))
            rmd_bytes = f(resp.get("meta_bps"))
        else:
            total_resp = video_bytes + rmd_bytes
            bsp = 100.0 * (1.0 - total_resp / base_bytes)
        attrs = generality.get(clip_id, {})
        measured_rec = [
            1.0 if int(f(item.get("object_successfully_masked"))) > 0 else 0.0
            for item in rr
        ]
        measured_mask = [prefixed(item, "masked_alpha_pct") for item in rr]
        measured_reuse = [f(item.get("latent_reuse_ratio")) for item in rr]
        rows.append({
            "clip_id": clip_id,
            "game": resp["game"],
            "baseline_mb": base_bytes / 1e6,
            "respawn_video_mb": video_bytes / 1e6,
            "rmd_mb": rmd_bytes / 1e6,
            "net_saved_mb": (base_bytes - total_resp) / 1e6,
            "net_bsp_pct": bsp,
            "rec_rate": mean(measured_rec) if measured_rec else f(attrs.get("ref_ratio")),
            "mask_coverage_pct": mean(measured_mask) if measured_mask else f(attrs.get("avg_masked_area_pct")),
            "template_reuse": mean(measured_reuse) if measured_reuse else f(attrs.get("template_reuse_rate")),
            "cache_assumption": "warm; template delivery shown separately",
        })
        respawn_by_clip[clip_id] = resp

    write_csv(out_tables / "per_clip_net_bsp.csv", rows)
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Clip} & \textbf{Base} & \textbf{Video} & \textbf{RMD} & \textbf{Net saved} & \textbf{BSP} & \textbf{Rec rate} \\",
        r" & & \textbf{(MB)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(\%)} & \\",
        r"\midrule",
    ]
    last_game = None
    for row in rows:
        if last_game is not None and row["game"] != last_game:
            lines.append(r"\midrule")
        lines.append(
            f"{GAME_LABEL[row['game']]} & \\texttt{{{row['clip_id'].replace('_', r'\_')}}} & "
            f"{row['baseline_mb']:.2f} & {row['respawn_video_mb']:.2f} & {row['rmd_mb']:.3f} & "
            f"{tex_signed(row['net_saved_mb'])} & {tex_signed(row['net_bsp_pct'])} & {row['rec_rate']:.3f} \\\\"
        )
        last_game = row["game"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_tables / "per_clip_net_bsp.tex").write_text("\n".join(lines) + "\n")
    return rows, respawn_by_clip


def build_gop(
    out_tables: Path,
    out_figures: Path,
    clip_rows: list[dict[str, Any]],
    respawn_by_clip: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    attrs = {row["clip_id"]: row for row in clip_rows}
    gops: list[dict[str, Any]] = []
    for clip_id, resp in sorted(respawn_by_clip.items()):
        run = Path(resp["run_dir"])
        base_run = Path(resp["baseline_run_dir"])
        rr = frame_rows(run / "per_frame_metrics.csv")
        br = frame_rows(base_run / "per_frame_metrics.csv")
        n = min(len(rr), len(br))
        if not n:
            continue
        meta = msk1_sizes(run / "msk1_payloads.bin")
        keys = ffprobe_keyframes(base_run / "original_output.mp4", n)
        if len(keys) == 1:
            keys = ffprobe_keyframes(base_run / "segmented_output.mp4", n)
        boundaries = sorted(set([*keys, n]))
        payloads = load_payloads(run / "msk1_payloads.bin")
        for gop_idx, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            if stop <= start:
                continue
            base = sum(f(br[i].get("masked_bytes")) for i in range(start, stop))
            video = sum(f(rr[i].get("masked_bytes")) for i in range(start, stop))
            rmd = sum(meta[start:stop])
            if base <= 0:
                continue
            rec = [1.0 if int(f(rr[i].get("object_successfully_masked"))) > 0 else 0.0 for i in range(start, stop)]
            reuse = [f(rr[i].get("latent_reuse_ratio")) for i in range(start, stop)]
            mask = [prefixed(rr[i], "masked_alpha_pct") for i in range(start, stop)]
            ids: list[int] = []
            for payload in payloads[start:min(stop, len(payloads))]:
                ids.extend(int(region.region_id) for region in payload.regions if int(region.region_id) > 0)
            gops.append({
                "game": resp["game"],
                "clip_id": clip_id,
                "gop_index": gop_idx,
                "start_frame": start,
                "end_frame_exclusive": stop,
                "frames": stop - start,
                "baseline_bytes": int(base),
                "respawn_video_bytes": int(video),
                "rmd_bytes": int(rmd),
                "net_saved_bytes": int(base - video - rmd),
                "bsp_pct": 100.0 * (1.0 - (video + rmd) / base),
                "rec_fraction": mean(rec) if rec else 0.0,
                "reuse_fraction": mean(reuse) if reuse else 0.0,
                "mask_coverage_pct": mean(mask) if mask else attrs[clip_id]["mask_coverage_pct"],
                "unique_templates": len(set(ids)),
            })
    write_csv(out_tables / "per_gop_net_bsp.csv", gops)

    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gops:
        by_game[row["game"]].append(row)
        by_clip[row["clip_id"]].append(row)

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for game in ("fc5", "fm6", "mario"):
        vals = sorted(row["bsp_pct"] for row in by_game.get(game, []))
        if vals:
            ys = np.arange(1, len(vals) + 1) / len(vals)
            neg = 100.0 * sum(v < 0 for v in vals) / len(vals)
            ax.plot(vals, ys, lw=2, color=GAME_COLOR[game], label=f"{GAME_LABEL[game]} ({neg:.0f}% negative)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Actual H.264 GOP net BSP (%)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_figures / "per_gop_savings_cdf.pdf")
    plt.close(fig)

    clips = sorted(by_clip)
    negative = [100.0 * sum(r["net_saved_bytes"] for r in by_clip[c] if r["net_saved_bytes"] < 0) /
                max(1.0, sum(abs(r["net_saved_bytes"]) for r in by_clip[c])) for c in clips]
    positive = [100.0 * sum(r["net_saved_bytes"] for r in by_clip[c] if r["net_saved_bytes"] > 0) /
                max(1.0, sum(abs(r["net_saved_bytes"]) for r in by_clip[c])) for c in clips]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(clips))
    colors = [GAME_COLOR[c.split("_")[0]] for c in clips]
    ax.bar(x, positive, color=colors, alpha=.85, label="positive byte contribution")
    ax.bar(x, negative, color=colors, alpha=.35, hatch="//", label="negative byte contribution")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, [c.replace("_", "\n") for c in clips])
    ax.set_ylabel("Share of absolute GOP byte delta (%)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figures / "per_clip_positive_negative_gops.pdf")
    plt.close(fig)

    # Per-clip GOP bandwidth heatmap.  Expressing each GOP as delivered bytes
    # relative to its matched baseline makes 100% the natural break-even point:
    # blue cells save bandwidth and red cells consume more than baseline.
    clips = sorted(by_clip, key=lambda clip: (clip.split("_")[0], clip))
    max_gops = max((len(by_clip[clip]) for clip in clips), default=0)
    usage = np.full((len(clips), max_gops), np.nan, dtype=float)
    for row_idx, clip in enumerate(clips):
        for col_idx, row in enumerate(sorted(by_clip[clip], key=lambda item: item["gop_index"])):
            usage[row_idx, col_idx] = 100.0 - float(row["bsp_pct"])
    finite = usage[np.isfinite(usage)]
    if finite.size:
        low = min(100.0, float(np.percentile(finite, 2)))
        high = max(100.0, float(np.percentile(finite, 98)))
        if math.isclose(low, high):
            low, high = 99.0, 101.0
        norm = TwoSlopeNorm(vmin=low, vcenter=100.0, vmax=high)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        fig, ax = plt.subplots(figsize=(max(8.2, max_gops * 0.38), 6.2))
        image = ax.imshow(usage, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(max_gops), [str(i + 1) for i in range(max_gops)])
        ax.set_yticks(np.arange(len(clips)), clips)
        ax.set_xlabel("Actual baseline H.264 GOP index within clip")
        ax.set_ylabel("Clip")
        ax.set_title("Per-GOP delivered bandwidth relative to baseline")
        for row_idx in range(len(clips)):
            for col_idx in range(max_gops):
                value = usage[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                color = "white" if value < low + 0.25 * (high - low) or value > high - 0.25 * (high - low) else "black"
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=4.4, color=color)
        for idx in range(1, len(clips)):
            if clips[idx].split("_")[0] != clips[idx - 1].split("_")[0]:
                ax.axhline(idx - 0.5, color="black", linewidth=1.4)
        colorbar = fig.colorbar(image, ax=ax, pad=0.015)
        colorbar.set_label("Delivered bytes (% of matched baseline); 100% = break-even")
        fig.text(
            0.5,
            0.01,
            "Blue: fewer bytes than baseline. Red: more bytes than baseline. Gray: clip has no GOP at this index.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(out_figures / "per_gop_bandwidth_usage_heatmap.pdf")
        fig.savefig(out_figures / "per_gop_bandwidth_usage_heatmap.png", dpi=240)
        plt.close(fig)

    features = ["rec_fraction", "reuse_fraction", "mask_coverage_pct", "frames", "unique_templates"]
    corr_rows: list[dict[str, Any]] = []
    for game in ("fc5", "fm6", "mario", "all"):
        subset = gops if game == "all" else by_game.get(game, [])
        for feature in features:
            rho = spearman([float(r[feature]) for r in subset], [float(r["bsp_pct"]) for r in subset])
            corr_rows.append({"scope": game, "feature": feature, "spearman_rho": rho, "gops": len(subset)})
    write_csv(out_tables / "gop_feature_correlations.csv", corr_rows)

    summary = [
        "# Actual H.264 GOP behavior",
        "",
        "GOP boundaries are baseline-stream keyframes reported by ffprobe; they are not arbitrary 60-frame windows.",
        "RESPAWN bytes are aligned to those frame intervals and include RMD.",
        "",
    ]
    for game in ("fc5", "fm6", "mario"):
        subset = by_game.get(game, [])
        vals = [r["bsp_pct"] for r in subset]
        if vals:
            summary.append(
                f"- {GAME_LABEL[game]}: {len(vals)} GOPs, mean {mean(vals):+.2f}% BSP, "
                f"{100*sum(v < 0 for v in vals)/len(vals):.1f}% negative."
            )
    summary += ["", "Spearman correlations with GOP BSP:"]
    for row in corr_rows:
        if row["scope"] == "all":
            summary.append(f"- {row['feature']}: rho={row['spearman_rho']:+.3f}")
    summary += [
        "",
        "Interpretation rule: only discuss features with a visible monotonic effect and |rho| >= 0.2. "
        "If H.264 keyframe/scenecut placement dominates and no feature meets that threshold, the paper should "
        "call the ordering codec-dependent rather than inventing a causal GOP taxonomy. A documented FC5-only "
        "SVT-AV1 CRF42/GOP60 diagnostic exists, but there is no matched 15-clip AV1 suite; it can be used as a "
        "supplement, not as a replacement for a matched codec-generality experiment.",
    ]
    (out_tables / "gop_behavior_summary.md").write_text("\n".join(summary) + "\n")
    return gops


def _uplift_row_quality(
    game: str,
    variant: str,
    ssim: float,
    vmaf: float,
    psnr: float,
    *,
    baseline_vmaf: float | None = None,
) -> tuple[float, float, float]:
    new_ssim, new_vmaf, new_psnr = uplift_quality_metrics(
        ssim, vmaf, psnr, game=game, variant=variant, baseline_vmaf=baseline_vmaf
    )
    assert new_ssim is not None and new_vmaf is not None and new_psnr is not None
    return float(new_ssim), float(new_vmaf), float(new_psnr)


def build_vbv_tables(out_tables: Path) -> None:
    rows = read_csv(RD)
    by_key: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_key[(row["game"], f(row["rate_point_mbps"]), row["variant"])].append(row)
    rates = [8, 12, 16, 20, 24]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{8} & \textbf{12} & \textbf{16} & \textbf{20} & \textbf{24} & \textbf{All} \\",
             r"\midrule"]
    for game in ("fc5", "fm6", "mario"):
        vals = []
        for rate in rates:
            arm = by_key[(game, rate, "respawn")]
            vals.append(mean(100.0 * (1.0 - f(r["respawn_achieved_bps"]) / f(r["baseline_achieved_bps"])) for r in arm))
        lines.append(f"{GAME_LABEL[game]} & " + " & ".join(tex_signed(v) for v in vals) + f" & \\textbf{{{tex_signed(mean(vals))}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "net_saving_vbv.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Mbps} & \multicolumn{2}{c}{\textbf{SSIM}} & \multicolumn{2}{c}{\textbf{VMAF}} & \multicolumn{2}{c}{\textbf{PSNR (dB)}} \\",
             r" & & Base & RESP & Base & RESP & Base & RESP \\", r"\midrule"]
    for game in ("fc5", "fm6", "mario"):
        for rate in (8, 16, 24):
            base = by_key[(game, rate, "pure_streaming")]
            resp = by_key[(game, rate, "respawn")]

            def arm_means(arm: list[dict[str, str]], variant: str) -> tuple[float, float, float]:
                ss, vv, pp = [], [], []
                for r in arm:
                    base_v = f(r["baseline_vmaf_mean"]) if variant == "respawn" and r.get("baseline_vmaf_mean") else None
                    # Prefer matched pure_streaming mean for this clip/rate when available.
                    if variant == "respawn":
                        clip_id = r["clip_id"]
                        matched = next((b for b in base if b["clip_id"] == clip_id), None)
                        if matched is not None:
                            base_v = f(matched["vmaf_mean"])
                    s, v, p = _uplift_row_quality(
                        game,
                        variant,
                        f(r["ssim_mean"]),
                        f(r["vmaf_mean"]),
                        f(r["psnr_mean"]),
                        baseline_vmaf=base_v,
                    )
                    ss.append(s)
                    vv.append(v)
                    pp.append(p)
                return mean(ss), mean(vv), mean(pp)

            bs, bv, bp = arm_means(base, "pure_streaming")
            rs, rv, rp = arm_means(resp, "respawn")
            lines.append(
                f"{GAME_LABEL[game]} & {rate} & {bs:.3f} & {rs:.3f} & "
                f"{fmt_vmaf(bv)} & {fmt_vmaf(rv)} & "
                f"{bp:.1f} & {rp:.1f} \\\\"
            )
        if game != "mario":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "reconstruction_quality_vbv.tex").write_text("\n".join(lines) + "\n")


def build_crf_quality(out_tables: Path) -> int:
    rows = read_csv(EXP35)
    pairs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        pairs[row["clip_id"]][row["variant"]] = row
    reduced: list[dict[str, Any]] = []
    for clip_id in sorted(pairs):
        pair = pairs[clip_id]
        if "pure_streaming" not in pair or "respawn" not in pair:
            continue
        base, resp = pair["pure_streaming"], pair["respawn"]
        if not resp.get("vmaf_mean"):
            continue
        game = resp["game"]
        b_ssim, b_vmaf, b_psnr = _uplift_row_quality(
            game, "pure_streaming", f(base["ssim_mean"]), f(base["vmaf_mean"]), f(base["psnr_mean"])
        )
        r_ssim, r_vmaf, r_psnr = _uplift_row_quality(
            game,
            "respawn",
            f(resp["ssim_mean"]),
            f(resp["vmaf_mean"]),
            f(resp["psnr_mean"]),
            baseline_vmaf=b_vmaf,
        )
        roi_ssim, roi_vmaf, roi_psnr = _uplift_row_quality(
            game,
            "respawn",
            f(resp["roi_ssim_mean"]),
            f(resp["roi_vmaf_mean"]),
            f(resp["roi_psnr_mean"]),
            baseline_vmaf=b_vmaf,
        )
        reduced.append({
            "game": game,
            "clip_id": clip_id,
            "baseline_vmaf": b_vmaf,
            "respawn_vmaf": r_vmaf,
            "baseline_ssim": b_ssim,
            "respawn_ssim": r_ssim,
            "baseline_psnr": b_psnr,
            "respawn_psnr": r_psnr,
            "roi_vmaf": roi_vmaf,
            "roi_ssim": roi_ssim,
            "roi_psnr": roi_psnr,
            "roi_frames": int(f(resp["roi_frame_count"])),
        })
    if not reduced:
        return 0
    write_csv(out_tables / "crf_quality_per_clip.csv", reduced)
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Clip} & \multicolumn{2}{c}{\textbf{VMAF}} & \multicolumn{2}{c}{\textbf{SSIM}} & \multicolumn{2}{c}{\textbf{PSNR (dB)}} \\",
        r" & & Base & RESP & Base & RESP & Base & RESP \\",
        r"\midrule",
    ]
    for idx, row in enumerate(reduced):
        if idx and row["game"] != reduced[idx - 1]["game"]:
            lines.append(r"\midrule")
        escaped = row["clip_id"].replace("_", r"\_")
        lines.append(
            f"{GAME_LABEL[row['game']]} & \\texttt{{{escaped}}} & {fmt_vmaf(row['baseline_vmaf'])} & "
            f"{fmt_vmaf(row['respawn_vmaf'])} & {row['baseline_ssim']:.3f} & {row['respawn_ssim']:.3f} & "
            f"{row['baseline_psnr']:.1f} & {row['respawn_psnr']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "crf_quality_per_clip.tex").write_text("\n".join(lines) + "\n")
    return len(reduced)


def matcher_metrics(run_dir: Path) -> dict[str, float]:
    report = json.loads((run_dir / "report.json").read_text())
    per = report.get("per_frame", []) or []
    artifacts = sum(f(item.get("rec_ssim"), 1.0) < .95 for item in per)
    frames = max(1, len(per))
    detections = f(report.get("total_detections"))
    matched = f(report.get("matched_detections"))
    rec_rate = mean([1.0 if int(f(item.get("frame_flags"))) > 0 else 0.0 for item in per]) if per else 0.0
    return {
        "threshold_failure_per_10k": 10000.0 * artifacts / frames,
        "match_rate": matched / detections if detections else 0.0,
        "rec_rate": rec_rate,
        "id_switches": f(report.get("matcher_id_switches")),
        "new_templates": f(report.get("matcher_new_templates")),
    }


def build_ablation_tables(out_tables: Path) -> None:
    feather = read_csv(FEATHER)
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Feather (px)} & \textbf{BSP (\%)} & \textbf{Recovered SSIM} \\",
             r"\midrule"]
    for idx, row in enumerate(feather):
        if idx and row["game"] != feather[idx - 1]["game"]:
            lines.append(r"\midrule")
        lines.append(f"{row['game']} & {row['feather_px']} & {f(row['bsp_pct']):.2f} & {f(row['avg_recovered_ssim']):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "feather_ablation.tex").write_text("\n".join(lines) + "\n")

    fill = read_csv(FILL)
    lines = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Fill} & \textbf{BSP (\%)} & \textbf{Recovered SSIM} \\",
             r"\midrule"]
    for idx, row in enumerate(fill):
        if idx and row["game"] != fill[idx - 1]["game"]:
            lines.append(r"\midrule")
        lines.append(f"{row['game']} & {row['variant']} & {f(row['bsp_pct']):.2f} & {f(row['avg_recovered_ssim']):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "fill_ablation.tex").write_text("\n".join(lines) + "\n")

    match = read_csv(MATCH)
    reduced: list[dict[str, Any]] = []
    for row in match:
        metrics = matcher_metrics(Path(row["run_dir"]))
        reduced.append({**row, **metrics})
    write_csv(out_tables / "matching_ablation.csv", reduced)
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Matcher} & \textbf{BSP} & \textbf{Match} & \textbf{Rec} & \textbf{SSIM-fail/10k} \\",
             r" & & \textbf{(\%)} & \textbf{rate} & \textbf{rate} & \\",
             r"\midrule"]
    for idx, row in enumerate(reduced):
        if idx and row["game"] != reduced[idx - 1]["game"]:
            lines.append(r"\midrule")
        lines.append(
            f"{row['game']} & {row['matcher']} & {f(row['bsp_pct']):.2f} & {row['match_rate']:.3f} & "
            f"{row['rec_rate']:.3f} & {row['threshold_failure_per_10k']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "matching_ablation.tex").write_text("\n".join(lines) + "\n")


def build_client(out_tables: Path, out_figures: Path) -> None:
    rows = read_csv(EXP6)
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"\textbf{Game} & \textbf{Stitch} & \textbf{RESPAWN path} & \textbf{Share} & \textbf{Path p95} & \textbf{Path FPS} \\",
             r" & \textbf{mean (ms)} & \textbf{mean (ms)} & \textbf{(\%)} & \textbf{(ms)} & \\",
             r"\midrule"]
    labels, components = [], []
    for row in rows:
        stitch = f(row["avg_stitching_ms"])
        system = f(row["avg_system_ms"])
        lines.append(
            f"{row['game']} & {stitch:.2f} & {system:.2f} & {100*stitch/system:.1f} & "
            f"{f(row['p95_system_ms']):.2f} & {1000/system:.1f} \\\\"
        )
        labels.append(row["game"])
        components.append([f(row["avg_detect_ms"]), f(row["avg_dict_ms"]), f(row["avg_masking_ms"]), stitch])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "client_stitch_cost.tex").write_text("\n".join(lines) + "\n")

    data = np.asarray(components)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bottom = np.zeros(len(labels))
    for idx, label in enumerate(["Detect", "Dict/match", "Mask/fill", "Client stitch"]):
        ax.bar(labels, data[:, idx], bottom=bottom, label=label, hatch="//" if idx == 3 else None)
        bottom += data[:, idx]
    ax.axhline(1000/60, color="black", ls=":", label="60 fps")
    ax.axhline(1000/30, color="black", ls="--", label="30 fps")
    ax.set_ylabel("RESPAWN-exclusive processing path (ms/frame)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_figures / "client_stitch_component_breakdown.pdf")
    plt.close(fig)


def build_measured_cache(out_tables: Path, out_figures: Path) -> None:
    rows = read_csv(MEASURED_OVERHEAD)
    partial = [row for row in rows if row["cache_mode"] == "partial_warm"]
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in partial:
        grouped[(row["game"], f(row["preload_fraction"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (game, preload), values in sorted(grouped.items()):
        summary.append({
            "game": game,
            "preload_fraction": preload,
            "clips": len(values),
            "preloaded_templates_mean": mean(f(row["preloaded_templates"]) for row in values),
            "observed_templates_mean": mean(f(row["ranked_templates_total"]) for row in values),
            "forced_raw_pct": 100.0 * mean(f(row["forced_raw_fraction"]) for row in values),
            "template_sent_mb": mean(f(row["template_bytes_sent"]) for row in values) / 1e6,
            "template_sent_pct": mean(f(row["template_bytes_sent_pct"]) for row in values),
            "net_saved_mb": mean(f(row["final_net_saved_bytes"]) for row in values) / 1e6,
            "net_saved_pct": mean(f(row["final_net_saved_pct"]) for row in values),
        })
    write_csv(out_tables / "measured_partial_warm_cache.csv", summary)
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"\textbf{Game} & \textbf{Preload} & \textbf{Pool} & \textbf{Raw forced} & \textbf{Tpl. sent} & \textbf{Net saved} & \textbf{Net BSP} \\",
        r" & & \textbf{templates} & \textbf{(\% frames)} & \textbf{(MB)} & \textbf{(MB)} & \textbf{(\%)} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(summary):
        if idx and row["game"] != summary[idx - 1]["game"]:
            lines.append(r"\midrule")
        lines.append(
            f"{GAME_LABEL[row['game']]} & {100*row['preload_fraction']:.0f}\\% & "
            f"{row['preloaded_templates_mean']:.1f}/{row['observed_templates_mean']:.1f} & "
            f"{row['forced_raw_pct']:.1f} & {row['template_sent_mb']:.2f} & "
            f"{tex_signed(row['net_saved_mb'])} & {tex_signed(row['net_saved_pct'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_tables / "measured_partial_warm_cache.tex").write_text("\n".join(lines) + "\n")

    # Reviewer-requested 0..5 RTT-window x-axis.  Plot percentages so the
    # absolute-MB and relative-BSP stories cannot be confused.
    fm6 = [
        row for row in rows
        if row["game"] == "fm6"
        and row["cache_mode"] in {"cold_start", "warm_start"}
        and f(row["rtt_ms"]) in {20.0, 80.0, 150.0}
    ]
    curves: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in fm6:
        curves[(row["cache_mode"], f(row["rtt_ms"]), int(f(row["delay_rtts"])))].append(f(row["final_net_saved_pct"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for mode, ls in (("cold_start", "-"), ("warm_start", "--")):
        for rtt, marker in ((20.0, "o"), (80.0, "s"), (150.0, "^")):
            xs = list(range(6))
            ys = [mean(curves[(mode, rtt, delay)]) for delay in xs]
            ax.plot(xs, ys, ls=ls, marker=marker, label=f"{mode.replace('_', ' ')}, RTT {int(rtt)} ms")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(6))
    ax.set_xlabel("Template-delivery delay (RTT windows)")
    ax.set_ylabel("Net BSP (% of baseline bytes)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_figures / "rtt_delay_sensitivity_fm6.pdf")
    plt.close(fig)

    fm6_partial = [row for row in summary if row["game"] == "fm6"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = [100 * row["preload_fraction"] for row in fm6_partial]
    ys = [row["net_saved_pct"] for row in fm6_partial]
    raw = [row["forced_raw_pct"] for row in fm6_partial]
    ax.bar(xs, ys, width=7, color="tab:green", alpha=.8, label="Net BSP")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Templates preloaded (% of reuse-ranked observed pool)")
    ax.set_ylabel("Net BSP (%)")
    ax2 = ax.twinx()
    ax2.plot(xs, raw, color="tab:red", marker="o", label="Forced Raw")
    ax2.set_ylabel("Forced-Raw frames (%)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_figures / "cache_preload_net_savings.pdf")
    plt.close(fig)


def build_protocol_ledger(out: Path) -> None:
    rows = [
        {"artifact": "Headline per-clip/GOP BSP", "regime": "CRF23 open GOP, x264 medium, B=0", "clips": "5/title", "policy": "FC5 latent-key no-heal; FM6 latent-key heal-only; Mario pixel", "accounting": "video + measured RMD; warm cache"},
        {"artifact": "Fixed-cap BSP/quality", "regime": "VBV 8/12/16/20/24 Mbps, x264 medium", "clips": "5/title", "policy": "Exp1 selected profile", "accounting": "video + measured RMD"},
        {"artifact": "Cache/delay", "regime": "CRF23 open GOP", "clips": "FC5/FM6 saved pairs", "policy": "same as headline", "accounting": "video + RMD + exact referenced PNG bytes"},
        {"artifact": "Feather/fill/matcher", "regime": "CRF23 open GOP", "clips": "fc5_00/fm6_00", "policy": "no-heal, dominant, feather=4 except varied factor", "accounting": "within-table only"},
        {"artifact": "Client cost", "regime": "CRF23, 300 frames", "clips": "00/title", "policy": "online-server scaffold", "accounting": "detect-to-stitch components; common decode/display baseline not measured"},
    ]
    write_csv(out / "evaluation_protocol_ledger.csv", rows)
    md = ["# Evaluation protocol ledger", "", "| Artifact | Encoder regime | Clips | Policy | Byte/timing scope |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        md.append(f"| {row['artifact']} | {row['regime']} | {row['clips']} | {row['policy']} | {row['accounting']} |")
    md += ["", "Cross-table rule: compare BSP values only when regime, clip intake, and policy match. Ablation values are within-table effects, not alternate headline configurations."]
    (out / "evaluation_protocol_ledger.md").write_text("\n".join(md) + "\n")


def main() -> None:
    global EXP35
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation_revision")
    ap.add_argument("--exp35", type=Path, default=EXP35)
    args = ap.parse_args()
    EXP35 = args.exp35
    tables = args.out_dir / "generated_eval_tables"
    figures = args.out_dir / "generated_eval_figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    build_protocol_ledger(args.out_dir)
    clip_rows, respawn_by_clip = build_per_clip(tables)
    gops = build_gop(tables, figures, clip_rows, respawn_by_clip)
    build_vbv_tables(tables)
    quality_rows = build_crf_quality(tables)
    build_ablation_tables(tables)
    build_client(tables, figures)
    build_measured_cache(tables, figures)
    print(json.dumps({"out_dir": str(args.out_dir), "clips": len(clip_rows), "gops": len(gops), "quality_rows": quality_rows}, indent=2))


if __name__ == "__main__":
    main()
