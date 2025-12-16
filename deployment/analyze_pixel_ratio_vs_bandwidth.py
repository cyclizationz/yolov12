import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ffprobe_frame_sizes(video_path: Path):
    """
    Use ffprobe to get per-frame packet sizes (bytes) before decoding.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_size",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    sizes = []
    for fr in data.get("frames", []):
        v = fr.get("pkt_size")
        if v is None:
            continue
        sizes.append(int(v))
    return np.array(sizes, dtype=np.float64)


def window_avg(seq: np.ndarray, w: int):
    out = []
    n = len(seq)
    for i in range(0, n, w):
        chunk = seq[i:i + w]
        if len(chunk) < w:
            break
        out.append(chunk.mean())
    return np.array(out, dtype=np.float64)


def get_frame(video_path: Path, idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
    return frame


def mask_from_diff(orig: np.ndarray, masked: np.ndarray, thresh: int = 10):
    """
    Approximate painted region by thresholding pixel differences between
    original and masked frames.
    """
    diff = cv2.absdiff(orig, masked)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def ensure_scaled_video(src: Path, scale: float, dst: Path):
    if dst.exists():
        return
    print(f"[ffmpeg] scaling {src.name} by {scale} -> {dst.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale=iw*{scale}:ih*{scale}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "outputs"),
        help="Directory containing original_output.mp4, segmented_output.mp4, and report.json",
    )
    ap.add_argument("--window", type=int, default=10, help="Frame averaging window size")
    ap.add_argument("--tag", type=str, default="", help="Suffix tag for output filenames")
    args = ap.parse_args()

    out_dir = args.out_dir

    baseline = out_dir / "original_output.mp4"
    masked = out_dir / "segmented_output.mp4"
    report_json = out_dir / "report.json"

    if not baseline.exists() or not masked.exists():
        raise FileNotFoundError("Run Yolov12Deployment first to generate original_output.mp4 and segmented_output.mp4.")

    # --- 1. Load per-frame bytes and compute 10-frame savings groups ---
    base_bytes = ffprobe_frame_sizes(baseline)
    mask_bytes = ffprobe_frame_sizes(masked)

    n = min(len(base_bytes), len(mask_bytes))
    base_bytes = base_bytes[:n]
    mask_bytes = mask_bytes[:n]

    window = int(args.window)
    base_avg = window_avg(base_bytes, window)
    mask_avg = window_avg(mask_bytes, window)

    m = min(len(base_avg), len(mask_avg))
    base_avg = base_avg[:m]
    mask_avg = mask_avg[:m]

    with np.errstate(divide="ignore", invalid="ignore"):
        savings = (base_avg - mask_avg) / base_avg
        savings[np.isnan(savings)] = 0.0

    # Sort groups by savings (descending)
    order = np.argsort(-savings)
    top_k = order[:5]

    # --- 2. For top groups, compute pixel ratio from original vs masked ---
    top_group_records = []
    print("=== Pixel ratio vs bandwidth savings (top 5 groups, window=10) ===")
    print("group\tframe_id\tpixel_ratio(%)\tsaving(%)")

    h, w = None, None
    for gi in top_k:
        # Representative frame id in group
        frame_id = int(gi * window + window // 2)
        orig_f = get_frame(baseline, frame_id)
        mask_f = get_frame(masked, frame_id)
        if h is None:
            h, w = orig_f.shape[:2]
        # Compute changed-pixel mask
        mask = mask_from_diff(orig_f, mask_f, thresh=8)
        pix_ratio = float(np.count_nonzero(mask)) / float(h * w) * 100.0
        save_ratio = float(savings[gi]) * 100.0
        record = {
            "group_index": int(gi),
            "frame_id": frame_id,
            "pixel_ratio_percent": pix_ratio,
            "saving_percent": save_ratio,
        }
        top_group_records.append(record)
        print(f"{int(gi)}\t{frame_id}\t{pix_ratio:7.3f}\t\t{save_ratio:7.3f}")

    # --- 3. Resolution scaling experiment ---
    scale_records = []
    scales = [0.5, 1.0, 1.5]
    print("\n=== Resolution scaling experiment (mean savings over all frames) ===")
    print("scale\tmean_baseline_bytes\tmean_masked_bytes\tmean_saving(%)")

    for s in scales:
        if abs(s - 1.0) < 1e-3:
            b_vid = baseline
            m_vid = masked
        else:
            b_vid = out_dir / f"original_output_scale{s:.1f}.mp4"
            m_vid = out_dir / f"segmented_output_scale{s:.1f}.mp4"
            ensure_scaled_video(baseline, s, b_vid)
            ensure_scaled_video(masked, s, m_vid)

        b_bytes = ffprobe_frame_sizes(b_vid)
        m_bytes = ffprobe_frame_sizes(m_vid)
        n2 = min(len(b_bytes), len(m_bytes))
        b_bytes = b_bytes[:n2]
        m_bytes = m_bytes[:n2]

        mean_b = float(b_bytes.mean())
        mean_m = float(m_bytes.mean())
        saving = (mean_b - mean_m) / mean_b * 100.0 if mean_b > 0 else 0.0
        rec = {
            "scale": s,
            "mean_baseline_bytes": mean_b,
            "mean_masked_bytes": mean_m,
            "saving_percent": saving,
        }
        scale_records.append(rec)
        print(f"{s:.1f}\t{mean_b:10.1f}\t\t{mean_m:10.1f}\t\t{saving:7.3f}")

    # --- 4. Persist analysis to JSON ---
    analysis = {
        "window": window,
        "top_groups": top_group_records,
        "resolution_scaling": scale_records,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = out_dir / f"pixel_bandwidth_analysis{suffix}.json"
    with out_json.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis data to {out_json}")

    # --- 5. Plot bandwidth saving vs resolution ---
    scales_list = [rec["scale"] for rec in scale_records]
    savings_list = [rec["saving_percent"] for rec in scale_records]

    plt.figure(figsize=(6, 4))
    plt.plot(scales_list, savings_list, "-o", color="tab:blue")
    plt.xlabel("Scale factor (relative to 1280x720)")
    plt.ylabel("Mean bandwidth saving (%)")
    plt.title("Bandwidth Saving vs Resolution (window = 10 frames)")
    plt.grid(True, linestyle="--", alpha=0.3)
    for s, sv in zip(scales_list, savings_list):
        plt.text(s, sv, f"{sv:.1f}%", ha="center", va="bottom", fontsize=8)

    out_png = out_dir / f"bandwidth_saving_by_resolution{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved resolution plot to {out_png}")


if __name__ == "__main__":
    main()


