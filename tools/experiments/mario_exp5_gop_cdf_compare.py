#!/usr/bin/env python3
"""Capture Mario per-GOP BSP under former on-disk protocol (clips 01–04 only) for CDF fairness,
then plot empirical CDF vs grid protocol using rd_suite CSV run_dir refs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_gop_tail import collect_gops, load_respawn_rows

from common import RESPAWN2026_DIR

CLIP_ORDER = ["mario_00", "mario_01", "mario_02", "mario_03", "mario_04"]
# Former protocol on disk before re-run reliably exists for mario_01–mario_04 (mario_00 was regenerated earlier).
CLIP_SUBSET_MATCHED = {"mario_01", "mario_02", "mario_03", "mario_04"}


def write_bsp_tsv(values: list[float], path: Path, *, meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# " + f"{k}={v}" for k, v in meta.items()]
    lines.append("gop_bsp_pct")
    lines.extend(f"{v:.6f}" for v in sorted(values))
    path.write_text("\n".join(lines) + "\n")


def clip_level_table(
    *,
    former_csv: Path,
    grid_csv: Path,
    out_md: Path,
) -> None:
    def load(cp: Path) -> list[dict[str, str]]:
        with cp.open(newline="") as f:
            return list(csv.DictReader(f))

    frows = load(former_csv)
    grows = load(grid_csv)
    by_old: dict[tuple[str, float], dict[str, str]] = {}
    for r in frows:
        if r["clip_id"] not in CLIP_ORDER or r["variant"] != "respawn":
            continue
        by_old[(r["clip_id"], float(r["rate_point_mbps"]))] = r
    pieces: list[str] = [
        "# Mario pixel protocol: former vs compact grid headers",
        "",
        "Delivered bitrate uses `segmented_output.mp4` + `msk1_payloads.bin` (RD suite convention). ",
        "Quality: VMAF / SSIM / PSNR from `evaluate_variant()` on recovered vs reference.",
        "",
        "| clip | Mb/s | Δ meta bps % | Δ delivered bps % | ΔVMAF | ΔSSIM | ΔPSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in CLIP_ORDER:
        for rp in [8.0, 12.0, 16.0, 20.0, 24.0]:
            ko = by_old.get((cid, rp))
            kg = next(
                (
                    z
                    for z in grows
                    if z["clip_id"] == cid and z["variant"] == "respawn" and abs(float(z["rate_point_mbps"]) - rp) < 1e-6
                ),
                None,
            )
            if not ko or not kg:
                continue

            def fnum(x: dict[str, str], k: str) -> float | None:
                v = x.get(k, "").strip()
                if v == "":
                    return None
                return float(v)

            mo, mn = fnum(ko, "meta_bps"), fnum(kg, "meta_bps")
            ao, ag = fnum(ko, "achieved_bps"), fnum(kg, "achieved_bps")
            dmeta = 100.0 * (mn - mo) / mo if mo not in (None, 0) else float("nan")
            ddel = 100.0 * (ag - ao) / ao if ao not in (None, 0) else float("nan")
            vmaf_o, vmaf_g = fnum(ko, "vmaf_mean"), fnum(kg, "vmaf_mean")
            s_o, s_g = fnum(ko, "ssim_mean"), fnum(kg, "ssim_mean")
            p_o, p_g = fnum(ko, "psnr_mean"), fnum(kg, "psnr_mean")
            dv = vmaf_g - vmaf_o if vmaf_o is not None and vmaf_g is not None else float("nan")
            ds = s_g - s_o if s_o is not None and s_g is not None else float("nan")
            dp = p_g - p_o if p_o is not None and p_g is not None else float("nan")
            pieces.append(
                f"| {cid} | {rp:g} | {dmeta:.2f} | {ddel:.3f} | {dv:+.3f} | {ds:+.5f} | {dp:+.3f} |"
            )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(pieces) + "\n")


def plot_dual_cdf(*, former_bsps: list[float], grid_0104: list[float], grid_all: list[float], png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    def ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
        return xs, ys

    fig, ax = plt.subplots(figsize=(5.75, 3.85))
    for label, data, lw, ls in (
        ("Former (GOP BSP, clips mario_01–mario_04)", np.asarray(former_bsps, dtype=np.float64), 2.0, "-"),
        ("Grid (GOP BSP, clips mario_01–mario_04)", np.asarray(grid_0104, dtype=np.float64), 2.0, "--"),
        ("Grid (GOP BSP, clips mario_00–mario_04)", np.asarray(grid_all, dtype=np.float64), 1.65, ":"),
    ):
        if len(data) == 0:
            continue
        x, y = ecdf(data)
        ax.step(x, y, where="post", label=f"{label} (n={len(data)})", linewidth=lw, linestyle=ls)
    ax.set_xlabel("GOP net savings BSP (%)  [60-frame bins]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    plt.close(fig)


def main() -> None:
    suite = RESPAWN2026_DIR / "exp1_tuned_upper_bound_shortclips"
    exp5 = RESPAWN2026_DIR / "exp5"
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("capture-former-subset-bsp")
    snap.add_argument(
        "--former-points",
        type=Path,
        default=suite / "rd_suite_tuned_upper_bound_shortclips_points.csv",
    )

    plt_p = sub.add_parser("plot-cdf-after-grid")
    plt_p.add_argument("--former-bsp-tsv", type=Path, default=exp5 / "mario_former_subset_gop_bsp.tsv")
    plt_p.add_argument(
        "--grid-points",
        type=Path,
        default=suite / "rd_suite_points.csv",
    )
    plt_p.add_argument("--out-png", type=Path, default=exp5 / "mario_gop_cdf_former_vs_grid.png")
    plt_p.add_argument(
        "--former-points-csv-for-table",
        type=Path,
        default=suite / "rd_suite_tuned_upper_bound_shortclips_points.csv",
    )

    tbl = sub.add_parser("clip-table-only")
    tbl.add_argument("--former-points", type=Path, default=suite / "rd_suite_tuned_upper_bound_shortclips_points.csv")
    tbl.add_argument("--grid-points", type=Path, default=suite / "rd_suite_points.csv")
    tbl.add_argument("--out-md", type=Path, default=exp5 / "mario_former_vs_grid_clip_metrics.md")

    args = ap.parse_args()

    if args.cmd == "capture-former-subset-bsp":
        rows = [
            r
            for r in load_respawn_rows(args.former_points, [])
            if r["clip_id"] in CLIP_SUBSET_MATCHED
        ]
        recs = collect_gops(rows, 60)
        vals = [r.gop_bsp_pct for r in recs]
        out_tsv = exp5 / "mario_former_subset_gop_bsp.tsv"
        write_bsp_tsv(
            vals,
            out_tsv,
            meta={
                "clips": ",".join(sorted(CLIP_SUBSET_MATCHED)),
                "source_rd_csv": str(args.former_points),
                "bins": "60_frames",
                "note": (
                    "Per-GOP BSP = 100*(1-(respawn_video_bytes+msk1_byte_per_frame_over_gop_sum)/baseline_video_bytes)."
                ),
            },
        )
        print(f"wrote {out_tsv} ({len(vals)} GOPs)")

    elif args.cmd == "plot-cdf-after-grid":
        former_lines = [ln for ln in args.former_bsp_tsv.read_text().splitlines() if ln and not ln.startswith("#")]
        if former_lines[0].strip().lower() == "gop_bsp_pct":
            former_lines = former_lines[1:]
        former_bsps = [float(ln) for ln in former_lines]

        grows = [r for r in load_respawn_rows(args.grid_points, []) if r["clip_id"] in CLIP_ORDER]
        grows_0104 = [r for r in grows if r["clip_id"] in CLIP_SUBSET_MATCHED]
        rec_grid_all = collect_gops(grows, 60)
        rec_grid_0104 = collect_gops(grows_0104, 60)
        grid_all_bsps = [r.gop_bsp_pct for r in rec_grid_all]
        grid_0104_bsps = [r.gop_bsp_pct for r in rec_grid_0104]
        plot_dual_cdf(
            former_bsps=former_bsps,
            grid_0104=grid_0104_bsps,
            grid_all=grid_all_bsps,
            png=args.out_png,
        )
        tbl_path = exp5 / "mario_former_vs_grid_clip_metrics.md"
        clip_level_table(
            former_csv=args.former_points_csv_for_table,
            grid_csv=args.grid_points,
            out_md=tbl_path,
        )
        print(f"wrote {args.out_png} and {tbl_path}")

    elif args.cmd == "clip-table-only":
        clip_level_table(former_csv=args.former_points, grid_csv=args.grid_points, out_md=args.out_md)
        print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
