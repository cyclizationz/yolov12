#!/usr/bin/env python3
"""
Plotting entrypoints.

Subcommands:
- `bitstream`: generate `timeline.csv` + bitstream distribution plots for a run dir
              (wrapper around `tools/metrics/bitstream_timeline.py`)
- `ods-suite`: read an `.ods` spreadsheet containing per-frame CSV tables, export
               selected sheets to CSV, then generate the BW/BSP/delay figures
               (using `video/figures/plotting.py`) into a chosen output folder.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Iterable

import zipfile
import xml.etree.ElementTree as ET
import importlib.util


def _run_bitstream_timeline(*, orig: Path, masked: Path, out_dir: Path, size_source: str) -> None:
    script = Path(__file__).resolve().parent / "bitstream_timeline.py"
    cmd = [
        "python",
        str(script),
        "--orig",
        str(orig),
        "--masked",
        str(masked),
        "--out-dir",
        str(out_dir),
        "--size-source",
        size_source,
    ]
    subprocess.check_call(cmd)


def _repo_root() -> Path:
    # tools/metrics/plotting.py -> tools/metrics -> tools -> repo root
    return Path(__file__).resolve().parents[2]


def _ods_sheets(ods_path: Path) -> dict[str, list[list[str]]]:
    """
    Return mapping: sheet_name -> rows (each row is list of cell text).
    Handles `table:number-columns-repeated`.
    """
    xml = None
    with zipfile.ZipFile(ods_path, "r") as z:
        xml = z.read("content.xml")
    ns = {
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    }
    root = ET.fromstring(xml)

    def cell_text(cell: ET.Element) -> str:
        ps = cell.findall(".//text:p", ns)
        s = "".join("".join(p.itertext()) for p in ps).strip()
        return s

    sheets: dict[str, list[list[str]]] = {}
    for table in root.findall(".//table:table", ns):
        name = table.get(f"{{{ns['table']}}}name") or ""
        if not name:
            continue
        rows_out: list[list[str]] = []
        for row in table.findall("table:table-row", ns):
            cells: list[str] = []
            for cell in row.findall("table:table-cell", ns):
                rep = int(cell.get(f"{{{ns['table']}}}number-columns-repeated", "1"))
                t = cell_text(cell)
                for _ in range(rep):
                    cells.append(t)
            while cells and cells[-1] == "":
                cells.pop()
            rows_out.append(cells)
        sheets[name] = rows_out
    return sheets


def _write_csv_from_rows(rows: list[list[str]], out_csv: Path) -> None:
    if not rows:
        raise SystemExit(f"ODS sheet is empty; cannot write {out_csv}")
    header = rows[0]
    data = rows[1:]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in data:
            # pad/truncate to header length
            rr = list(r[: len(header)])
            if len(rr) < len(header):
                rr.extend([""] * (len(header) - len(rr)))
            w.writerow(rr)


def _overall_bsp_bar(csv_files: dict[str, Path], out_pdf: Path, success_only: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402

    labels = list(csv_files.keys())
    vals: list[float] = []
    for k in labels:
        df = pd.read_csv(csv_files[k], sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        if success_only and "object_successfully_masked" in df.columns:
            df = df[df["object_successfully_masked"].fillna(0) >= 1]
        base = pd.to_numeric(df.get("baseline_bytes"), errors="coerce")
        mask = pd.to_numeric(df.get("masked_bytes"), errors="coerce")
        sum_base = float(np.nansum(base.to_numpy()))
        sum_mask = float(np.nansum(mask.to_numpy()))
        v = 0.0 if sum_base <= 0 else (sum_base - sum_mask) / sum_base * 100.0
        vals.append(float(v))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#0B3D91", alpha=0.90)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BSP (%)")
    ax.set_title("Overall BSP (success-only)" if success_only else "Overall BSP")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.6 if v >= 0 else -0.6), f"{v:.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

def _group_sum_n(df, n: int, cols: list[str]):
    import numpy as np
    import pandas as pd

    if len(df) == 0:
        out = {c: [] for c in cols}
        out["group"] = []
        return pd.DataFrame(out)
    # Grouping by a numpy array triggers a pandas FutureWarning; use a real column.
    df2 = df.loc[:, cols].copy()
    df2["_group"] = (np.arange(len(df2)) // int(n)).astype(int)
    agg = df2.groupby("_group", as_index=False)[cols].sum(numeric_only=True)
    agg.insert(0, "group", np.arange(len(agg)))
    return agg


def _boxplot_baseline_vs_masked(
    *,
    labels: list[str],
    baseline_data: list,
    masked_data: list,
    title: str,
    ylabel: str,
    out_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402

    c_base = "#1f77b4"
    c_mask = "#ff7f0e"

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    centers = np.arange(len(labels))
    offset = 0.18
    pos_b = centers - offset
    pos_m = centers + offset

    bp_b = ax.boxplot(
        baseline_data,
        positions=pos_b,
        widths=0.28,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
    )
    bp_m = ax.boxplot(
        masked_data,
        positions=pos_m,
        widths=0.28,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
    )

    for b in bp_b["boxes"]:
        b.set_facecolor(c_base)
        b.set_alpha(0.25)
        b.set_edgecolor(c_base)
    for m in bp_b["medians"]:
        m.set_color(c_base)
        m.set_linewidth(1.4)
    for mean in bp_b["means"]:
        mean.set_color(c_base)
        mean.set_linestyle(":")
        mean.set_linewidth(1.4)

    for b in bp_m["boxes"]:
        b.set_facecolor(c_mask)
        b.set_alpha(0.25)
        b.set_edgecolor(c_mask)
    for m in bp_m["medians"]:
        m.set_color(c_mask)
        m.set_linewidth(1.4)
    for mean in bp_m["means"]:
        mean.set_color(c_mask)
        mean.set_linestyle(":")
        mean.set_linewidth(1.4)

    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=c_base, lw=2, label="Baseline"),
        plt.Line2D([0], [0], color=c_mask, lw=2, label="Masked"),
        plt.Line2D([0], [0], color="black", lw=1.4, label="Median"),
        plt.Line2D([0], [0], color="black", lw=1.4, linestyle=":", label="Mean"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _bar_with_optional_error(*, labels: list[str], means: list[float], stds: list[float] | None, title: str, ylabel: str, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(labels))
    if stds is None:
        ax.bar(x, means, color="#0B3D91", alpha=0.90)
    else:
        ax.bar(x, means, yerr=stds, capsize=4, color="#0B3D91", alpha=0.90)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(means):
        y = v + (0.8 if v >= 0 else -0.8)
        ax.text(i, y, f"{v:.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _plot_new_suite(csv_files: dict[str, Path], out_dir: Path) -> None:
    import pandas as pd
    import numpy as np

    def _get_col(df: pd.DataFrame, key: str):
        if key in df.columns:
            return df[key]
        # ODS-exported CSVs often use the long header: e.g. "changed_pixels_pct (changed_pixels / frame area)"
        for c in df.columns:
            if c.startswith(key + " "):
                return df[c]
            if c.startswith(key + "("):
                return df[c]
        return None

    labels = list(csv_files.keys())
    dfs: dict[str, pd.DataFrame] = {}
    for k in labels:
        df = pd.read_csv(csv_files[k], sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        for c in ["baseline_bytes", "masked_bytes", "object_present_model", "object_successfully_masked", "masked_alpha_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs[k] = df

    # 1) overall BSP (ratio-of-sums), all frames
    overall_all: list[float] = []
    # 2) overall BSP (ratio-of-sums), success-only
    overall_succ: list[float] = []
    # 3) per-30-frame BSP bar with std across groups (all frames)
    per30_mean: list[float] = []
    per30_std: list[float] = []

    # 8) success% vs masked alpha% (all frames)
    succ_pct: list[float] = []
    succ_pct_overall: list[float] = []
    masked_alpha_mean: list[float] = []
    changed_pixels_pct_mean: list[float] = []

    for k in labels:
        df = dfs[k]
        base = _get_col(df, "baseline_bytes")
        mask = _get_col(df, "masked_bytes")
        if base is None or mask is None:
            raise SystemExit(f"{csv_files[k]} missing baseline_bytes/masked_bytes columns")
        valid = base.notna() & mask.notna() & (base > 0)
        sb = float(base[valid].sum())
        sm = float(mask[valid].sum())
        overall_all.append(0.0 if sb <= 0 else (sb - sm) / sb * 100.0)

        succ = _get_col(df, "object_successfully_masked")
        if succ is not None:
            succ_m = succ.fillna(0) >= 1
            valid_s = valid & succ_m
            sb_s = float(base[valid_s].sum())
            sm_s = float(mask[valid_s].sum())
            overall_succ.append(0.0 if sb_s <= 0 else (sb_s - sm_s) / sb_s * 100.0)
        else:
            overall_succ.append(0.0)

        # Per-30-frame groups (just chunk by 30 frames, regardless of true FPS).
        agg = _group_sum_n(df.loc[valid, :], 30, [str(base.name), str(mask.name)])
        b = agg[str(base.name)].replace(0, np.nan)
        m = agg[str(mask.name)]
        bsp_g = ((b - m) / b) * 100.0
        bsp_g = bsp_g.replace([np.inf, -np.inf], np.nan).dropna()
        per30_mean.append(float(bsp_g.mean()) if len(bsp_g) else 0.0)
        per30_std.append(float(bsp_g.std(ddof=1)) if len(bsp_g) > 1 else 0.0)

        present = _get_col(df, "object_present_model")
        success = _get_col(df, "object_successfully_masked")
        if present is not None and success is not None:
            present_m = present.fillna(0) >= 1
            denom = int(present_m.sum())
            num = int((success.fillna(0) >= 1)[present_m].sum())
            succ_pct.append(0.0 if denom == 0 else 100.0 * float(num) / float(denom))
        else:
            succ_pct.append(0.0)

        if success is not None:
            succ_all = float((success.fillna(0) >= 1).mean() * 100.0)
            succ_pct_overall.append(succ_all)
        else:
            succ_pct_overall.append(0.0)

        ma = _get_col(df, "masked_alpha_pct")
        if ma is not None:
            ma_num = pd.to_numeric(ma, errors="coerce").dropna()
            masked_alpha_mean.append(float(ma_num.mean()) if len(ma_num) else 0.0)
        else:
            masked_alpha_mean.append(0.0)

        cp = _get_col(df, "changed_pixels_pct")
        if cp is not None:
            cp_num = pd.to_numeric(cp, errors="coerce").dropna()
            changed_pixels_pct_mean.append(float(cp_num.mean()) if len(cp_num) else 0.0)
        else:
            changed_pixels_pct_mean.append(0.0)

    _bar_with_optional_error(
        labels=labels,
        means=overall_all,
        stds=None,
        title="Overall bandwidth saving percentage (BSP) – all frames",
        ylabel="BSP (%)",
        out_pdf=out_dir / "new1_overall_bsp_all_frames_bar.pdf",
    )
    _bar_with_optional_error(
        labels=labels,
        means=overall_succ,
        stds=None,
        title="Overall bandwidth saving percentage (BSP) – successful masks only",
        ylabel="BSP (%)",
        out_pdf=out_dir / "new2_overall_bsp_success_only_bar.pdf",
    )
    _bar_with_optional_error(
        labels=labels,
        means=per30_mean,
        stds=per30_std,
        title="Per-30-frame BSP (mean ± SD across 30-frame groups) – all frames",
        ylabel="BSP (%)",
        out_pdf=out_dir / "new3_per30_bsp_bar_mean_std_all_frames.pdf",
    )

    # 4-7) distributions
    base_pf = [_get_col(dfs[k], "baseline_bytes").dropna().to_numpy() for k in labels]  # type: ignore[union-attr]
    mask_pf = [_get_col(dfs[k], "masked_bytes").dropna().to_numpy() for k in labels]  # type: ignore[union-attr]
    _boxplot_baseline_vs_masked(
        labels=labels,
        baseline_data=base_pf,
        masked_data=mask_pf,
        title="Distribution of per-frame encoded sizes (all frames)",
        ylabel="bytes/frame",
        out_pdf=out_dir / "new4_dist_per_frame_bytes_all.pdf",
    )

    base_pf_s = []
    mask_pf_s = []
    for k in labels:
        df = dfs[k]
        succ = _get_col(df, "object_successfully_masked")
        if succ is None:
            base_pf_s.append(np.array([], dtype=float))
            mask_pf_s.append(np.array([], dtype=float))
            continue
        m = succ.fillna(0) >= 1
        base_pf_s.append(_get_col(df.loc[m, :], "baseline_bytes").dropna().to_numpy())  # type: ignore[union-attr]
        mask_pf_s.append(_get_col(df.loc[m, :], "masked_bytes").dropna().to_numpy())  # type: ignore[union-attr]
    _boxplot_baseline_vs_masked(
        labels=labels,
        baseline_data=base_pf_s,
        masked_data=mask_pf_s,
        title="Distribution of per-frame encoded sizes (successful masks only)",
        ylabel="bytes/frame",
        out_pdf=out_dir / "new5_dist_per_frame_bytes_success_only.pdf",
    )

    base_g30 = []
    mask_g30 = []
    for k in labels:
        df = dfs[k]
        base = _get_col(df, "baseline_bytes")
        mask = _get_col(df, "masked_bytes")
        valid = base.notna() & mask.notna() & (base > 0)  # type: ignore[union-attr]
        agg = _group_sum_n(df.loc[valid, :], 30, [str(base.name), str(mask.name)])  # type: ignore[union-attr]
        base_g30.append(agg["baseline_bytes"].dropna().to_numpy())
        mask_g30.append(agg["masked_bytes"].dropna().to_numpy())
    _boxplot_baseline_vs_masked(
        labels=labels,
        baseline_data=base_g30,
        masked_data=mask_g30,
        title="Distribution of 30-frame encoded sizes (all frames; sum of 30 frames)",
        ylabel="bytes / 30 frames",
        out_pdf=out_dir / "new6_dist_30frame_bytes_all.pdf",
    )

    base_g30_s = []
    mask_g30_s = []
    for k in labels:
        df = dfs[k]
        succ = _get_col(df, "object_successfully_masked")
        if succ is None:
            base_g30_s.append(np.array([], dtype=float))
            mask_g30_s.append(np.array([], dtype=float))
            continue
        base = _get_col(df, "baseline_bytes")
        mask = _get_col(df, "masked_bytes")
        m = (succ.fillna(0) >= 1) & base.notna() & mask.notna() & (base > 0)  # type: ignore[union-attr]
        agg = _group_sum_n(df.loc[m, :], 30, [str(base.name), str(mask.name)])  # type: ignore[union-attr]
        base_g30_s.append(agg["baseline_bytes"].dropna().to_numpy())
        mask_g30_s.append(agg["masked_bytes"].dropna().to_numpy())
    _boxplot_baseline_vs_masked(
        labels=labels,
        baseline_data=base_g30_s,
        masked_data=mask_g30_s,
        title="Distribution of 30-frame encoded sizes (successful masks only; sum of 30 frames)",
        ylabel="bytes / 30 frames",
        out_pdf=out_dir / "new7_dist_30frame_bytes_success_only.pdf",
    )

    # 8) success % with avg pixel_changed marked on x-axis
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.bar(x, succ_pct, width=0.55, color="#2ca02c", alpha=0.9, label="Masking success % (given object_present_model>=1)")
    ax.set_title("Masking success % (conditional on object_present_model>=1; x-axis shows avg pixel_changed %)")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nchg={v:.2f}%" for lab, v in zip(labels, changed_pixels_pct_mean)])
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "new8_success_pct_vs_masked_alpha_pct_bar.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    # 8b) overall success % over all frames, same x-tick annotation
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.bar(x, succ_pct_overall, width=0.55, color="#2ca02c", alpha=0.9, label="Masking success % (over all frames)")
    ax.set_title("Masking success % (over all frames; x-axis shows avg pixel_changed %)")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nchg={v:.2f}%" for lab, v in zip(labels, changed_pixels_pct_mean)])
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "new8b_overall_success_pct_bar.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _plot_recon_quality_fig9(
    *,
    out_dir: Path,
    run_dirs: dict[str, Path],
    vmaf_frame_cnt: int,
    vmaf_subsample: int,
    vmaf_scale_height: int,
    vmaf_threads: int,
) -> None:
    """
    Figure 9: reconstruction quality (SSIM + VMAF).

    - SSIM is taken from report.json: avg_recovered_ssim
    - VMAF is computed between original_output.mp4 (reference) and recovered_output.mp4 (distorted),
      on a *sampled* decode to keep runtime reasonable.
    """
    import json
    import subprocess

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402

    labels = list(run_dirs.keys())
    ssim: list[float] = []
    vmaf_mean: list[float] = []
    vmaf_p10: list[float] = []

    cache_path = out_dir / "new9_vmaf_cache.json"
    cache: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    for k in labels:
        rd = run_dirs[k]
        rep = json.loads((rd / "report.json").read_text())
        ssim.append(float(rep.get("avg_recovered_ssim") or 0.0))

        # VMAF: reference is baseline-encoded output, distorted is recovered output.
        ref = rd / "original_output.mp4"
        dist = rd / "recovered_output.mp4"
        if not ref.exists() or not dist.exists():
            vmaf_mean.append(float("nan"))
            vmaf_p10.append(float("nan"))
            continue

        # Cache key includes parameters so "full video" doesn't reuse earlier short-run values.
        ck = f"{rd.resolve()}|fc={int(vmaf_frame_cnt)}|sub={int(vmaf_subsample)}|h={int(vmaf_scale_height)}|t={int(vmaf_threads)}"
        if ck in cache and "vmaf_mean" in cache[ck]:
            vmaf_mean.append(float(cache[ck].get("vmaf_mean")))
            vmaf_p10.append(float(cache[ck].get("vmaf_p10")))
            continue

        cmd = [
            "conda",
            "run",
            "-n",
            "yolov12",
            "python",
            str((_repo_root() / "tools" / "metrics" / "vmaf_score.py").resolve()),
            "--ref",
            str(ref),
            "--dist",
            str(dist),
            "--threads",
            str(int(vmaf_threads)),
            "--out-fmt",
            "csv",
            # Full duration by default (frame-cnt=0), but subsampled + scaled for tractability.
            "--frame-cnt",
            str(int(vmaf_frame_cnt)),
            "--subsample",
            str(int(vmaf_subsample)),
            "--scale-height",
            str(int(vmaf_scale_height)),
        ]
        out = subprocess.check_output(cmd, text=True)
        j = json.loads(out)
        vm = j.get("vmaf_mean")
        vp10 = j.get("vmaf_p10")
        cache[ck] = {"vmaf_mean": float(vm) if vm is not None else float("nan"), "vmaf_p10": float(vp10) if vp10 is not None else float("nan")}
        vmaf_mean.append(cache[ck]["vmaf_mean"])
        vmaf_p10.append(cache[ck]["vmaf_p10"])

    cache_path.write_text(json.dumps(cache, indent=2))

    # Plot: two panels (SSIM, VMAF mean with p10 annotation).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.6))
    x = np.arange(len(labels))

    ax1.bar(x, ssim, color="#1f77b4", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Reconstruction SSIM (avg)")
    ax1.set_ylabel("SSIM")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(ssim):
        ax1.text(i, min(1.0, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax2.bar(x, vmaf_mean, color="#9467bd", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.0, 100.0)
    ax2.set_title("Reconstruction VMAF (sampled)")
    ax2.set_ylabel("VMAF")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, (m, p10) in enumerate(zip(vmaf_mean, vmaf_p10)):
        if m == m:
            ax2.text(i, m + 1.2, f"{m:.1f}\n(p10 {p10:.1f})", ha="center", va="bottom", fontsize=8)

    if vmaf_frame_cnt and vmaf_frame_cnt > 0:
        span = f"first {int(vmaf_frame_cnt)} frames"
    else:
        span = "full duration"
    fig.suptitle(
        "Figure 9: Reconstruction quality (baseline vs recovered)\n"
        f"VMAF computed on {span}, every {int(max(1, vmaf_subsample))}th frame, scaled to {int(vmaf_scale_height)}p",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "new9_recon_quality_ssim_vmaf.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _plot_delay10_split_and_ordered(*, out_dir: Path, csv_files: dict[str, Path]) -> None:
    """
    Delay figures:
    - delay10_stacked_delay_means_all.pdf: end-to-end mean per-frame delay, stacked in execution order.
    - delay10_1_server.pdf: server-only components (ordered).
    - delay10_2_client.pdf: client-only components (ordered).

    Ordering follows deployment/offline_processor.cpp:
      detect -> paint(masking) -> recover(stitching) -> encode_masked(+SEI)

    Notes:
    - For YOLO games, detect is preprocess/inference/postprocess.
    - For pixel games, inference_ms already includes ROI/template/flow, so we avoid double counting by:
        use roi_filter_ms/template_matching_ms/motion_filter_ms and set preprocess/inference/postprocess=0.
    """
    import pandas as pd
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    def _get_col(df: pd.DataFrame, key: str):
        df.columns = [c.strip() for c in df.columns]
        if key in df.columns:
            return df[key]
        for c in df.columns:
            if c.startswith(key + " ") or c.startswith(key + "("):
                return df[c]
        return None

    # Stage definitions and colors (consistent palette).
    COLORS = {
        "preprocess_ms": "#4e79a7",
        "inference_ms": "#f28e2b",
        "postprocess_ms": "#e15759",
        "roi_filter_ms": "#59a14f",
        "template_matching_ms": "#edc948",
        "motion_filter_ms": "#b07aa1",
        "masking_ms": "#ff9da7",
        "stitching_ms": "#76b7b2",
        "encode_baseline_ms": "#1f77b4",
        "encode_masked_ms": "#ff7f0e",
    }
    LABELS = {
        "preprocess_ms": "preprocess",
        "inference_ms": "inference",
        "postprocess_ms": "postprocess",
        "roi_filter_ms": "roi_filter",
        "template_matching_ms": "template_match",
        "motion_filter_ms": "motion_filter",
        "masking_ms": "masking(paint)",
        "stitching_ms": "client_recover",
        "encode_masked_ms": "encode_masked(+SEI)",
    }

    # Execution order in offline_processor.cpp (top-first list for readability).
    # NOTE: Stacking is rendered bottom-up; we'll reverse these lists when drawing so
    # encode is at the bottom and preprocess is at the top (as requested).
    ORDER_E2E_TOP_FIRST = [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "roi_filter_ms",
        "template_matching_ms",
        "motion_filter_ms",
        "masking_ms",
        "stitching_ms",
        "encode_masked_ms",
    ]
    # Corrected split: ROI/template/motion are also server-side; client-only is recover/stitching.
    ORDER_SERVER_TOP_FIRST = [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "roi_filter_ms",
        "template_matching_ms",
        "motion_filter_ms",
        "masking_ms",
        "encode_masked_ms",
    ]
    ORDER_CLIENT_TOP_FIRST = ["stitching_ms"]

    traces = list(csv_files.keys())
    means: dict[str, list[float]] = {k: [] for k in ORDER_E2E_TOP_FIRST}

    for t in traces:
        df = pd.read_csv(csv_files[t], sep=None, engine="python")
        # pull cols (missing => zeros)
        vals = {}
        for k in ORDER_E2E_TOP_FIRST:
            col = _get_col(df, k)
            if col is None:
                vals[k] = 0.0
            else:
                v = pd.to_numeric(col, errors="coerce").dropna()
                vals[k] = float(v.mean()) if len(v) else 0.0

        # Detect pixel-mode traces (avoid double counting inference_ms).
        pixel_detect = (vals["roi_filter_ms"] + vals["template_matching_ms"] + vals["motion_filter_ms"]) > 0.5
        if pixel_detect:
            vals["preprocess_ms"] = 0.0
            vals["inference_ms"] = 0.0
            vals["postprocess_ms"] = 0.0
        else:
            vals["roi_filter_ms"] = 0.0
            vals["template_matching_ms"] = 0.0
            vals["motion_filter_ms"] = 0.0

        for k in ORDER_E2E_TOP_FIRST:
            means[k].append(float(vals[k]))

    def stacked_bar(top_first_keys: list[str], out_pdf: Path, title: str):
        # Draw bottom-up so "encoding at bottom, preprocess at top".
        keys = list(reversed(top_first_keys))
        fig, ax = plt.subplots(figsize=(9.6, 3.8))
        x = np.arange(len(traces))
        bottom = np.zeros(len(traces), dtype=float)
        for k in keys:
            v = np.array(means[k], dtype=float)
            if float(np.max(v)) <= 0.0:
                continue
            ax.bar(x, v, bottom=bottom, color=COLORS.get(k), label=LABELS.get(k, k))
            # Annotate each segment with its value.
            for i in range(len(traces)):
                if v[i] <= 0:
                    continue
                y = bottom[i] + v[i] / 2.0
                ax.text(
                    x[i],
                    y,
                    f"{v[i]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
            bottom += v
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Delay (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(traces)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    # (A) Server-only and client-only (kept for readability)
    stacked_bar(
        ORDER_SERVER_TOP_FIRST,
        out_dir / "delay10_1_server.pdf",
        "Delay breakdown (server-side only; encode_masked+SEI included)\n(top=preprocess, bottom=encode_masked)",
    )
    stacked_bar(
        ORDER_CLIENT_TOP_FIRST,
        out_dir / "delay10_2_client.pdf",
        "Delay breakdown (client-side only)\n(top=client_recover, bottom=client_recover)",
    )

    # (B) End-to-end as two parallel stacked bars per trace: server vs client
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    x = np.arange(len(traces))
    w = 0.34

    server_bottom = np.zeros(len(traces), dtype=float)
    client_bottom = np.zeros(len(traces), dtype=float)

    # Server stack (ordered)
    for k in reversed(ORDER_SERVER_TOP_FIRST):
        v = np.array(means[k], dtype=float)
        if float(np.max(v)) <= 0.0:
            continue
        ax.bar(x - w / 2, v, width=w, bottom=server_bottom, color=COLORS.get(k), label=f"server:{LABELS.get(k, k)}")
        for i in range(len(traces)):
            if v[i] <= 0:
                continue
            y = server_bottom[i] + v[i] / 2.0
            ax.text(x[i] - w / 2, y, f"{v[i]:.1f}", ha="center", va="center", fontsize=7, color="black")
        server_bottom += v

    # Client stack (ordered)
    for k in reversed(ORDER_CLIENT_TOP_FIRST):
        v = np.array(means[k], dtype=float)
        if float(np.max(v)) <= 0.0:
            continue
        ax.bar(x + w / 2, v, width=w, bottom=client_bottom, color=COLORS.get(k), label=f"client:{LABELS.get(k, k)}")
        for i in range(len(traces)):
            if v[i] <= 0:
                continue
            y = client_bottom[i] + v[i] / 2.0
            ax.text(x[i] + w / 2, y, f"{v[i]:.1f}", ha="center", va="center", fontsize=7, color="black")
        client_bottom += v

    totals = server_bottom + client_bottom
    xt = [f"{t}\n({totals[i]:.1f} ms)" for i, t in enumerate(traces)]
    ax.set_title(
        "End-to-end delay (server vs client; per-frame mean)\n"
        "Top=preprocess, bottom=encode_masked; encode_baseline excluded",
    )
    ax.set_xlabel("Trace")
    ax.set_ylabel("Delay (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(xt)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Reduce legend duplicates by using dict ordering.
    handles, labels_l = ax.get_legend_handles_labels()
    dedup = {}
    for h, lab in zip(handles, labels_l):
        dedup.setdefault(lab, h)
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    fig.tight_layout()
    (out_dir / "delay10_stacked_delay_means_all.pdf").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "delay10_stacked_delay_means_all.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _parse_sheet_map(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Invalid --sheet-map entry (expected ABBR=SHEET): {it}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise SystemExit(f"Invalid --sheet-map entry: {it}")
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bit = sub.add_parser("bitstream", help="Generate bitstream timeline CSV + plots from MP4s.")
    ap_bit.add_argument("--run-dir", type=Path, default=None, help="Run directory containing original_output.mp4 and segmented_output.mp4")
    ap_bit.add_argument("--orig", type=Path, default=None, help="Baseline/original MP4 (overrides --run-dir)")
    ap_bit.add_argument("--masked", type=Path, default=None, help="Masked/segmented MP4 (overrides --run-dir)")
    ap_bit.add_argument("--out-dir", type=Path, default=None, help="Output directory (required if using --orig/--masked)")
    ap_bit.add_argument("--size-source", choices=["frames", "packets"], default="frames")

    ap_ods = sub.add_parser("ods-suite", help="Export selected ODS sheets to CSV and plot BW/BSP figures.")
    ap_ods.add_argument("--ods", type=Path, required=True, help="Input .ods file containing per-frame tables in sheets.")
    ap_ods.add_argument("--out-dir", type=Path, required=True, help="Where to write figures and exported CSVs.")
    ap_ods.add_argument(
        "--sheet-map",
        action="append",
        default=[],
        help="Mapping ABBR=SHEET_NAME. Repeatable. If omitted, uses default mapping for your latest.ods.",
    )
    ap_ods.add_argument("--vmaf-frame-cnt", type=int, default=0, help="VMAF: if >0, only score first N frames; 0=full duration")
    ap_ods.add_argument("--vmaf-subsample", type=int, default=10, help="VMAF: score every Nth frame (>=1)")
    ap_ods.add_argument("--vmaf-scale-height", type=int, default=540, help="VMAF: decode scaled to this height (0 disables)")
    ap_ods.add_argument("--vmaf-threads", type=int, default=1, help="VMAF: threads for vmaf CLI")
    args = ap.parse_args()

    if args.cmd == "bitstream":
        if args.orig is not None or args.masked is not None:
            if args.orig is None or args.masked is None:
                raise SystemExit("If using --orig/--masked, provide both.")
            if args.out_dir is None:
                raise SystemExit("--out-dir is required when using --orig/--masked directly.")
            orig = args.orig
            masked = args.masked
            out_dir = args.out_dir
        else:
            if args.run_dir is None:
                raise SystemExit("Provide either --run-dir or both --orig and --masked.")
            run_dir = args.run_dir
            orig = run_dir / "original_output.mp4"
            masked = run_dir / "segmented_output.mp4"
            if not orig.exists():
                raise SystemExit(f"Missing {orig}")
            if not masked.exists():
                raise SystemExit(f"Missing {masked}")
            out_dir = args.out_dir or run_dir / ("bitstream_viz_packets" if args.size_source == "packets" else "bitstream_viz")

        out_dir.mkdir(parents=True, exist_ok=True)
        _run_bitstream_timeline(orig=orig, masked=masked, out_dir=out_dir, size_source=args.size_source)
        print(str(out_dir))
        return

    if args.cmd == "ods-suite":
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        sheet_map = _parse_sheet_map(args.sheet_map)
        if not sheet_map:
            # Default mapping (matches the sheet names you showed in latest.ods)
            sheet_map = {
                "FC5c": "fc5_crop_codec_exact_full_x264_crf18_k60_repomodel_per_frame",
                "FC5uc": "fc5_uncrop_codec_exact_full_x264_crf18_k60_repomodel_per_frame",
                "FC5cAV1": "fc5_crop_av1_crf42",
                "FM6": "fm6_x264_crf18",
                "Mario": "Mario_x264_crf18",
            }

        sheets = _ods_sheets(args.ods)
        data_dir = out_dir / "data"
        csv_files: dict[str, Path] = {}
        for abbr, sheet_name in sheet_map.items():
            if sheet_name not in sheets:
                raise SystemExit(f"ODS missing sheet {sheet_name!r} (needed for {abbr})")
            csv_path = data_dir / f"{abbr}.csv"
            _write_csv_from_rows(sheets[sheet_name], csv_path)
            csv_files[abbr] = csv_path

        # Generate the BW/BSP/delay figures from the original plotting implementation.
        import matplotlib

        matplotlib.use("Agg")
        plotting_py = _repo_root() / "video" / "figures" / "plotting.py"
        if not plotting_py.exists():
            raise SystemExit(f"Missing plotting script: {plotting_py}")
        spec = importlib.util.spec_from_file_location("vp_plotting", plotting_py)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Failed to load module spec from {plotting_py}")
        vp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vp)  # type: ignore[misc]

        vp.OUTDIR = out_dir
        vp.OUTDIR.mkdir(parents=True, exist_ok=True)
        vp.CSV_FILES = {k: str(v) for k, v in csv_files.items()}

        # Figures that are fps-agnostic (per-frame; safe across mixed-FPS traces).
        vp.plot5_box_baseline_vs_masked_all()
        vp.plot6_box_baseline_vs_masked_success_only()
        vp.plot7_box_bsp_all()
        vp.plot8_box_bsp_success_only()
        vp.plot9_masking_success_percentage()
        vp.plot_bsp_bar_mean_std_all_csvs()
        vp.plot_bsp_bar_mean_std_success_only_all_csvs()

        # Extra overall BSP bars (ratio-of-sums, no FPS assumptions).
        _overall_bsp_bar(csv_files, out_dir / "bw_overall_bsp_all.pdf", success_only=False)
        _overall_bsp_bar(csv_files, out_dir / "bw_overall_bsp_success_only.pdf", success_only=True)

        # New, more stable suite requested (30-frame units).
        _plot_new_suite(csv_files, out_dir)
        # Delay figures: server/client split + ordered end-to-end.
        _plot_delay10_split_and_ordered(out_dir=out_dir, csv_files=csv_files)

        # Figure 9: reconstruction quality (SSIM + VMAF).
        # Map ABBR -> run directory (used to read report.json and MP4 outputs).
        run_dirs = {
            "FC5c": _repo_root() / "record" / "final3" / "fc5_crop_codec_exact_full_x264_crf18_k60_repomodel",
            "FC5uc": _repo_root() / "record" / "final3" / "fc5_uncrop_codec_exact_full_x264_crf18_k60_repomodel",
            "FC5cAV1": _repo_root() / "record" / "final3" / "fc5_crop_updated_best_libsvtav1_crf42",
            "FM6": _repo_root() / "record" / "final3" / "fm6_best_x264_crf18_k60_dominant_full",
            "Mario": _repo_root() / "record" / "final3" / "mario_best_x264_crf18_k60_pixel_singleTpl_full",
        }
        # Only include keys present in the ODS sheet_map (and existing run dirs).
        run_dirs = {k: v for k, v in run_dirs.items() if k in csv_files and v.exists() and (v / "report.json").exists()}
        if run_dirs:
            _plot_recon_quality_fig9(
                out_dir=out_dir,
                run_dirs=run_dirs,
                vmaf_frame_cnt=int(args.vmaf_frame_cnt),
                vmaf_subsample=int(max(1, args.vmaf_subsample)),
                vmaf_scale_height=int(max(0, args.vmaf_scale_height)),
                vmaf_threads=int(max(1, args.vmaf_threads)),
            )

        print(str(out_dir))
        return

    raise SystemExit("Unhandled command")


if __name__ == "__main__":
    main()

