import json
import subprocess
from pathlib import Path
import argparse


def ffprobe_frame_sizes(video_path: Path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_size",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return [int(fr["pkt_size"]) for fr in data.get("frames", []) if "pkt_size" in fr]


def summarize_dir(d: Path):
    base = d / "original_output.mp4"
    mask = d / "segmented_output.mp4"
    if not base.exists() or not mask.exists():
        return None
    b = ffprobe_frame_sizes(base)
    m = ffprobe_frame_sizes(mask)
    n = min(len(b), len(m))
    if n <= 0:
        return None
    b = b[:n]
    m = m[:n]
    bmean = sum(b) / n
    mmean = sum(m) / n
    saving = (bmean - mmean) / bmean * 100.0 if bmean > 1e-9 else 0.0
    worse = sum(1 for i in range(n) if m[i] > b[i]) * 100.0 / n
    return {
        "dir": d.name,
        "frames": n,
        "saving_pct": saving,
        "baseline_kbpf": bmean / 1024.0,
        "masked_kbpf": mmean / 1024.0,
        "worse_pct": worse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "outputs"),
        help="Directory containing run subfolders (e.g., deployment/outputs)",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="pixel",
        help="Only include subfolders starting with this prefix (default: pixel)",
    )
    args = ap.parse_args()

    root = args.outputs_dir
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if args.prefix and not d.name.startswith(args.prefix):
            continue
        s = summarize_dir(d)
        if s:
            rows.append(s)

    rows.sort(key=lambda x: x["saving_pct"], reverse=True)
    print(f"Found {len(rows)} runs under {root} (prefix={args.prefix!r})")
    print("dir\tframes\tsaving%\tbaseline_kBpf\tmasked_kBpf\tworse%")
    for r in rows:
        print(
            f"{r['dir']}\t{r['frames']}\t{r['saving_pct']:.2f}\t"
            f"{r['baseline_kbpf']:.2f}\t{r['masked_kbpf']:.2f}\t{r['worse_pct']:.1f}"
        )


if __name__ == "__main__":
    main()


