import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRACE_LABEL = {
    "fc5_crop": "FC5 (cropped)",
    "fm6": "FM6 (racing)",
    "mario": "Mario (pixel)",
}


def _set_style():
    plt.rcParams.update(
        {
            "figure.figsize": (6.8, 4.0),
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _plot_trace_best(df: pd.DataFrame, trace: str, best_codec: str, out: Path):
    d = df[(df["trace"] == trace) & (df["codec"] == best_codec)].copy()
    d = d.sort_values("crf")
    if d.empty:
        return

    # Two curves:
    # - Baseline: (baseline_bps, vmaf_baseline)
    # - Masked+meta+recovered: (masked_plus_meta_bps, vmaf_recovered)
    fig, ax = plt.subplots()
    ax.plot(d["baseline_bps"] / 1e6, d["vmaf_baseline"], marker="o", label="Baseline (orig_enc vs source)")
    ax.plot(
        d["masked_plus_meta_bps"] / 1e6,
        d["vmaf_recovered"],
        marker="o",
        label="Masked+meta (recovered vs source)",
    )

    # annotate points with CRF
    for _, r in d.iterrows():
        ax.annotate(f"CRF{int(r['crf'])}", (r["baseline_bps"] / 1e6, r["vmaf_baseline"]), textcoords="offset points", xytext=(6, 4))
        ax.annotate(
            f"CRF{int(r['crf'])}",
            (r["masked_plus_meta_bps"] / 1e6, r["vmaf_recovered"]),
            textcoords="offset points",
            xytext=(6, -10),
        )

    ax.set_title(f"{TRACE_LABEL.get(trace, trace)} — best codec: {best_codec}")
    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel("VMAF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-trace RD curves for best codec from codec sweep analysis.")
    ap.add_argument("--rd-csv", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/codec/codec_rd_points.csv"))
    ap.add_argument(
        "--best-csv", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/codec/best_codec_per_trace.csv")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/codec"))
    args = ap.parse_args()

    _set_style()
    df = pd.read_csv(args.rd_csv)
    best = pd.read_csv(args.best_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_map = {r["trace"]: r["codec"] for _, r in best.iterrows()}
    for trace, codec in best_map.items():
        out = args.out_dir / f"rd_curve_best_{trace}.png"
        _plot_trace_best(df, trace, codec, out)
        print("WROTE", out)


if __name__ == "__main__":
    main()

