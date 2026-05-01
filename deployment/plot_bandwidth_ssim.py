import json
import subprocess
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ffprobe_packets(video_path: Path):
    """
    Use ffprobe to get packet PTS times and packet sizes (bytes) for the video stream.
    For MP4/H.264 this is typically one packet per coded picture (access unit).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_packets",
        "-show_entries", "packet=pts_time,size",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    packets = []
    for p in data.get("packets", []) or []:
        pts_t = p.get("pts_time")
        sz = p.get("size")
        if pts_t is None or sz is None:
            continue
        packets.append((float(pts_t), int(sz)))
    return packets


def align_packets_by_time(a, b, tol_ms: float = 6.0):
    """
    Align two (pts_time, size) lists by pts_time with a tolerance (ms).
    This is robust when the two MP4s differ slightly in fps/timebase metadata.
    Returns (aligned_pts_time, a_sizes, b_sizes).
    """
    tol = tol_ms / 1000.0
    # Sort by pts_time so we align in presentation order (B-frames can make pts_time non-monotonic in file order).
    a = sorted([(t, s) for (t, s) in a if t is not None], key=lambda x: x[0])
    b = sorted([(t, s) for (t, s) in b if t is not None], key=lambda x: x[0])
    i = 0
    j = 0
    ts = []
    asz = []
    bsz = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "outputs"),
        help="Directory containing original_output.mp4, segmented_output.mp4, recovered_output.mp4 (optional), and report.json",
    )
    ap.add_argument("--window", type=int, default=10, help="Frame averaging window size")
    ap.add_argument("--tag", type=str, default="", help="Suffix tag for output filenames (plots)")
    args = ap.parse_args()

    out_dir = args.out_dir

    baseline_video = out_dir / "original_output.mp4"     # no masking
    masked_video = out_dir / "segmented_output.mp4"      # with masking
    report_json = out_dir / "report.json"                # metrics from C++

    if not baseline_video.exists() or not masked_video.exists():
        raise FileNotFoundError(
            f"Expected videos not found:\n"
            f"  baseline: {baseline_video}\n"
            f"  masked:   {masked_video}\n"
            f"Run Yolov12Deployment first."
        )

    # 1) Per-frame bandwidth (bytes) as packet sizes, aligned by PTS time
    print("Running ffprobe on baseline (no masking) video packets...")
    baseline_pkts = ffprobe_packets(baseline_video)
    print("Running ffprobe on masked (segmented) video packets...")
    masked_pkts = ffprobe_packets(masked_video)
    pts_time, baseline_bytes, masked_bytes = align_packets_by_time(baseline_pkts, masked_pkts, tol_ms=6.0)
    if not baseline_bytes or not masked_bytes:
        raise RuntimeError("Failed to align packet streams by PTS time; check ffprobe output and timestamps.")

    # 2) Load per-frame SSIM from C++ report
    if not report_json.exists():
        raise FileNotFoundError(f"Metrics report not found: {report_json}")

    with report_json.open("r") as f:
        report = json.load(f)

    per_frame = report.get("per_frame", [])
    if not per_frame:
        raise RuntimeError("No 'per_frame' metrics in report.json; "
                           "ensure offline_processor.cpp was rebuilt and rerun.")

    # Masked vs baseline metrics
    ssim_values = [entry["ssim"] for entry in per_frame]
    psnr_values = [entry["psnr"] for entry in per_frame]

    # Recovered vs baseline metrics (may be missing if some frames not reconstructed)
    rec_ssim_values = [entry.get("rec_ssim", entry["ssim"]) for entry in per_frame]
    rec_psnr_values = [entry.get("rec_psnr", entry["psnr"]) for entry in per_frame]

    # Align series lengths (use the minimum common length)
    n = min(len(baseline_bytes), len(masked_bytes), len(ssim_values), len(psnr_values), len(rec_ssim_values), len(rec_psnr_values))
    baseline_bytes = baseline_bytes[:n]
    masked_bytes = masked_bytes[:n]
    pts_time = pts_time[:n]
    ssim_values = ssim_values[:n]
    psnr_values = psnr_values[:n]
    rec_ssim_values = rec_ssim_values[:n]
    rec_psnr_values = rec_psnr_values[:n]

    # 3) Aggregate over windows of 10 frames to reduce noise
    window = int(args.window)

    # Mean saving ratio over all frames (not windowed)
    base_arr_full = np.array(baseline_bytes, dtype=np.float64)
    mask_arr_full = np.array(masked_bytes, dtype=np.float64)
    mean_saving = float((base_arr_full.mean() - mask_arr_full.mean()) / max(base_arr_full.mean(), 1e-9))

    def window_avg(seq, w):
        out = []
        for i in range(0, len(seq), w):
            chunk = seq[i:i + w]
            if len(chunk) < w:
                break  # drop incomplete tail window
            out.append(sum(chunk) / float(w))
        return out

    baseline_avg = window_avg(baseline_bytes, window)
    masked_avg = window_avg(masked_bytes, window)
    ssim_avg = window_avg(ssim_values, window)
    psnr_avg = window_avg(psnr_values, window)
    rec_ssim_avg = window_avg(rec_ssim_values, window)
    rec_psnr_avg = window_avg(rec_psnr_values, window)

    m = min(len(baseline_avg), len(masked_avg), len(ssim_avg), len(psnr_avg),
            len(rec_ssim_avg), len(rec_psnr_avg))
    baseline_avg = baseline_avg[:m]
    masked_avg = masked_avg[:m]
    ssim_avg = ssim_avg[:m]
    psnr_avg = psnr_avg[:m]
    rec_ssim_avg = rec_ssim_avg[:m]
    rec_psnr_avg = rec_psnr_avg[:m]

    # Per-group bandwidth savings ratio: (baseline - masked) / baseline
    baseline_arr = np.array(baseline_avg, dtype=np.float64)
    masked_arr = np.array(masked_avg, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        savings = (baseline_arr - masked_arr) / baseline_arr
        savings[np.isnan(savings)] = 0.0

    # Best and worst savings
    best_idx = int(np.argmax(savings))
    best_ratio = float(savings[best_idx]) * 100.0
    negative = savings[savings < 0.0]
    avg_worst = float(negative.mean() * 100.0) if negative.size > 0 else 0.0

    # Top-5 saving groups (indices and representative frame ids)
    order = np.argsort(-savings)  # descending
    top_k = order[:5]
    example_frames = {int(i): int(i * window + window // 2) for i in top_k}

    print("=== Bandwidth savings over {}-frame groups ===".format(window))
    print(f"Best saving: {best_ratio:.2f}% at group {best_idx} (example frame {example_frames.get(best_idx, 0)})")
    print(f"Average worst (where masked > baseline): {avg_worst:.2f}%")
    print("Top-5 saving groups:")
    for gi in top_k:
        print(f"  group {int(gi)}: saving {savings[gi]*100.0:.2f}%  example_frame={example_frames[int(gi)]}")

    # Use group index (each group = `window` frames) on X-axis
    groups = np.arange(m)

    # 4a) Bandwidth figure (lines only) + highlight top-5 groups
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(groups, baseline_avg, label=f"Baseline bytes/frame (avg{window})", color="tab:blue")
    ax1.plot(groups, masked_avg, label=f"Masked bytes/frame (avg{window}); mean saving={mean_saving*100:.2f}%", color="tab:orange")

    # Mark top-5 saving groups on masked curve
    ax1.scatter(top_k, masked_arr[top_k], color="black", marker="o", zorder=5, label="Top-5 savings")

    ax1.set_xlabel(f"Group index (each = {window} frames)")
    ax1.set_ylabel(f"bytes per frame ({window}-frame average)")
    ax1.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right")
    plt.title(f"Bandwidth ({window}-frame Averages) – mean saving={mean_saving*100:.2f}% (PTS-aligned packets)")
    plt.tight_layout()
    suffix = f"_{args.tag}" if args.tag else ""
    out_band = out_dir / f"bandwidth{suffix}.png"
    fig1.savefig(out_band, dpi=150)
    print(f"Saved bandwidth plot to {out_band}")

    # 4b) Quality figure (bars for SSIM & PSNR)
    fig2, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    bar_width = 0.35

    # SSIM bars
    ax_s.bar(groups - bar_width/2, ssim_avg, width=bar_width,
             label=f"SSIM masked (avg{window})", color="tab:green")
    ax_s.bar(groups + bar_width/2, rec_ssim_avg, width=bar_width,
             label=f"SSIM recovered (avg{window})", color="tab:olive")
    ax_s.set_ylabel(f"SSIM ({window}-frame avg)")
    ax_s.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_s.legend(loc="lower right")

    # PSNR bars
    ax_p.bar(groups - bar_width/2, psnr_avg, width=bar_width,
             label=f"PSNR masked (avg{window})", color="tab:red")
    ax_p.bar(groups + bar_width/2, rec_psnr_avg, width=bar_width,
             label=f"PSNR recovered (avg{window})", color="tab:pink")
    ax_p.set_ylabel(f"PSNR (dB, {window}-frame avg)")
    ax_p.set_xlabel(f"Group index (each = {window} frames)")
    ax_p.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_p.legend(loc="lower right")

    plt.suptitle(f"Quality Metrics (SSIM & PSNR, {window}-frame Averages)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_qual = out_dir / f"quality{suffix}.png"
    fig2.savefig(out_qual, dpi=150)
    print(f"Saved quality plot to {out_qual}")


if __name__ == "__main__":
    main()


