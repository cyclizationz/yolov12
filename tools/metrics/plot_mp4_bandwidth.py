import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    # Sort by PTS time (presentation order). File order can be non-monotonic with B-frames.
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


def window_avg(x: list[float], w: int) -> list[float]:
    out: list[float] = []
    for i in range(0, len(x), w):
        chunk = x[i : i + w]
        if len(chunk) < w:
            break
        out.append(float(sum(chunk)) / float(w))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot baseline vs masked bytes/frame from MP4 packet sizes (PTS aligned).")
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--masked", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output PNG path")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--window", type=int, default=10, help="Averaging window in frames")
    ap.add_argument("--tol-ms", type=float, default=6.0, help="PTS alignment tolerance (ms)")
    ap.add_argument("--topk", type=int, default=5, help="Highlight top-k savings windows")
    args = ap.parse_args()

    p0 = ffprobe_packets(args.orig)
    p1 = ffprobe_packets(args.masked)
    ts, b0, b1 = align_by_time(p0, p1, tol_ms=float(args.tol_ms))
    if not b0 or not b1:
        raise SystemExit("failed to align packets by PTS time")

    base_arr = np.array(b0, dtype=np.float64)
    mask_arr = np.array(b1, dtype=np.float64)
    mean_saving_pct = float((base_arr.mean() - mask_arr.mean()) / max(base_arr.mean(), 1e-9) * 100.0)

    w = int(args.window)
    base_avg = window_avg(list(base_arr), w)
    mask_avg = window_avg(list(mask_arr), w)
    m = min(len(base_avg), len(mask_avg))
    base_avg = base_avg[:m]
    mask_avg = mask_avg[:m]

    base_w = np.array(base_avg, dtype=np.float64)
    mask_w = np.array(mask_avg, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        savings = (base_w - mask_w) / base_w
        savings[np.isnan(savings)] = 0.0

    order = np.argsort(-savings)
    top_k = order[: max(0, int(args.topk))]

    fig, ax = plt.subplots(figsize=(12, 6))
    groups = np.arange(m)
    ax.plot(groups, base_avg, label=f"Baseline bytes/frame (avg{w})", color="tab:blue")
    ax.plot(groups, mask_avg, label=f"Masked bytes/frame (avg{w}); mean saving={mean_saving_pct:.2f}%", color="tab:orange")
    if len(top_k) > 0:
        ax.scatter(top_k, mask_w[top_k], color="black", marker="o", zorder=5, label=f"Top-{len(top_k)} savings windows")
    ax.set_xlabel(f"Group index (each = {w} frames)")
    ax.set_ylabel(f"bytes per frame ({w}-frame average)")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    title = args.title.strip() or f"Bandwidth ({w}-frame Averages) – mean saving={mean_saving_pct:.2f}%"
    ax.set_title(title)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()


