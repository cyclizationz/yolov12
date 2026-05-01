import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _default_threads() -> int:
    return max(1, min(32, (os.cpu_count() or 8)))


def ffmpeg_to_y4m(
    src: Path,
    dst_y4m: Path,
    *,
    frame_cnt: int | None,
    subsample: int | None,
    scale_height: int | None,
    decode_threads: int,
) -> None:
    """
    Decode to y4m for VMAF CLI.

    Notes:
    - VMAF CLI consumes Y4M; writing full-resolution full-length y4m is huge.
    - `frame_cnt`, `subsample`, and `scale_height` are applied during decode to keep I/O manageable.
    """
    vf = []
    if subsample is not None and subsample > 1:
        # Keep every Nth frame (frame index based). Reset timestamps to keep constant frame progression.
        vf.append(f"select='not(mod(n\\,{int(subsample)}))',setpts=N/FRAME_RATE/TB")
    if scale_height is not None and int(scale_height) > 0:
        # Preserve aspect ratio. -2 picks an even width automatically.
        vf.append(f"scale=-2:{int(scale_height)}")
    vf_arg = ",".join(vf) if vf else None

    dt = max(1, int(decode_threads))
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-threads",
        str(dt),
        "-i",
        str(src),
    ]
    if vf_arg:
        cmd += ["-vf", vf_arg]
    if frame_cnt is not None and frame_cnt > 0:
        cmd += ["-frames:v", str(int(frame_cnt))]
    # Use yuv420p for compatibility with vmaf defaults.
    cmd += [
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "yuv4mpegpipe",
        str(dst_y4m),
    ]
    subprocess.check_call(cmd)


def run_vmaf(
    vmaf_bin: Path,
    ref_y4m: Path,
    dist_y4m: Path,
    out_path: Path,
    threads: int,
    out_fmt: str,
    frame_cnt: int | None,
    subsample: int | None,
) -> None:
    cmd = [
        str(vmaf_bin),
        "--reference",
        str(ref_y4m),
        "--distorted",
        str(dist_y4m),
        "--model",
        "version=vmaf_v0.6.1",
        "--output",
        str(out_path),
        "--threads",
        str(int(threads)),
        "--quiet",
    ]
    if out_fmt == "json":
        cmd.append("--json")
    elif out_fmt == "csv":
        cmd.append("--csv")
    else:
        raise ValueError(f"Unsupported out_fmt: {out_fmt}")

    if frame_cnt is not None and frame_cnt > 0:
        cmd.extend(["--frame_cnt", str(int(frame_cnt))])
    if subsample is not None and subsample > 1:
        cmd.extend(["--subsample", str(int(subsample))])

    env = os.environ.copy()
    # Ensure libvmaf shared library is discoverable (install uses lib/x86_64-linux-gnu).
    libdir = vmaf_bin.parent.parent / "lib" / "x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    subprocess.check_call(cmd, env=env)


def _quantile_from_hist(counts: np.ndarray, q: float, scale: float) -> float | None:
    total = int(counts.sum())
    if total <= 0:
        return None
    target = int(np.ceil(q * total))
    if target <= 0:
        target = 1
    c = 0
    for i, v in enumerate(counts):
        c += int(v)
        if c >= target:
            return float(i) / float(scale)
    return float(len(counts) - 1) / float(scale)


def parse_vmaf_csv_streaming(path: Path) -> tuple[float | None, float | None]:
    """
    Streaming parser for vmaf CLI --csv output.

    Keeps O(1) memory by computing:
    - mean: running sum / count
    - p10: fixed histogram over [0, 100] at 0.001 resolution
    """
    vmaf_key: str | None = None
    n = 0
    s = 0.0
    # 0..100 with 0.001 step => 100000 bins + 1
    scale = 1000.0
    bins = np.zeros(100000 + 1, dtype=np.int64)

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            if vmaf_key is None:
                # Find a column that looks like the per-frame vmaf value.
                # Typical is "vmaf", but tolerate variants.
                for k in row.keys():
                    if k is None:
                        continue
                    lk = str(k).strip().lower()
                    if lk == "vmaf" or lk.endswith(".vmaf") or ("vmaf" in lk and "mean" not in lk):
                        vmaf_key = str(k)
                        break
                if vmaf_key is None:
                    # No recognizable vmaf column; bail out early.
                    break

            raw = row.get(vmaf_key)
            if raw is None:
                continue
            raw = str(raw).strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except Exception:
                continue
            if not (v == v):  # NaN
                continue
            n += 1
            s += v
            # clamp and bin
            if v < 0.0:
                v = 0.0
            elif v > 100.0:
                v = 100.0
            idx = int(round(v * scale))
            if idx < 0:
                idx = 0
            elif idx >= bins.shape[0]:
                idx = bins.shape[0] - 1
            bins[idx] += 1

    if n <= 0:
        return None, None
    mean = float(s / float(n))
    p10 = _quantile_from_hist(bins, 0.10, scale)
    return mean, p10


def parse_vmaf_json(path: Path) -> tuple[float | None, float | None]:
    # Fallback parser (loads JSON in RAM).
    j = json.loads(path.read_text())
    mean = None
    pm = j.get("pooled_metrics", {}) or {}
    if isinstance(pm, dict) and "vmaf" in pm and isinstance(pm["vmaf"], dict):
        try:
            mean = float(pm["vmaf"].get("mean"))
        except Exception:
            mean = None

    # p10 from per-frame vmaf values
    p10 = None
    frames = j.get("frames", []) or []
    vals = []
    for fr in frames:
        v = fr.get("metrics", {}).get("vmaf")
        if v is None:
            v = fr.get("vmaf")
        if v is None:
            continue
        vals.append(float(v))
    if vals:
        p10 = float(np.quantile(np.array(vals, dtype=np.float64), 0.10))
    return mean, p10


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute VMAF (mean and p10) for dist vs reference videos.")
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--dist", required=True, type=Path, help="Distorted video (e.g., reconstructed output)")
    ap.add_argument(
        "--vmaf-bin",
        type=Path,
        default=Path("/home/tiehangz/proj/yolov12/third_party/vmaf/libvmaf/install/bin/vmaf"),
        help="Path to vmaf CLI binary",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=_default_threads(),
        help="libvmaf worker threads (default: min(32, CPU count)).",
    )
    ap.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="ffmpeg decode threads per input (default: same as --threads; use 1 to force single-thread decode).",
    )
    ap.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/home/tiehangz/proj/yolov12/record/_tmp_vmaf"),
        help="Temp dir for intermediate y4m files. Put this on disk (NOT /tmp if /tmp is tmpfs).",
    )
    ap.add_argument(
        "--out-fmt",
        choices=["csv", "json"],
        default="csv",
        help="VMAF CLI output format. CSV is smaller and can be parsed streaming (lower RAM).",
    )
    ap.add_argument(
        "--frame-cnt",
        type=int,
        default=0,
        help="If >0, only process first N frames (useful for quick checks).",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="If >1, compute VMAF every N frames (reduces runtime/output size).",
    )
    ap.add_argument(
        "--scale-height",
        type=int,
        default=0,
        help="If >0, decode videos scaled to this height (keeps y4m smaller). Uses scale=-2:H to preserve aspect.",
    )
    args = ap.parse_args()

    if not args.vmaf_bin.exists():
        raise SystemExit(f"vmaf binary not found: {args.vmaf_bin}")

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vmaf_", dir=str(args.temp_dir)) as td:
        td = Path(td)
        ref_y4m = td / "ref.y4m"
        dist_y4m = td / "dist.y4m"
        out_path = td / ("out.csv" if args.out_fmt == "csv" else "out.json")

        frame_cnt = (args.frame_cnt if args.frame_cnt > 0 else None)
        subsample = (args.subsample if args.subsample > 1 else None)
        scale_h = (args.scale_height if args.scale_height > 0 else None)
        ff_thr = int(args.ffmpeg_threads) if int(args.ffmpeg_threads) > 0 else int(args.threads)

        ffmpeg_to_y4m(
            args.ref, ref_y4m, frame_cnt=frame_cnt, subsample=subsample, scale_height=scale_h, decode_threads=ff_thr
        )
        ffmpeg_to_y4m(
            args.dist, dist_y4m, frame_cnt=frame_cnt, subsample=subsample, scale_height=scale_h, decode_threads=ff_thr
        )
        run_vmaf(
            args.vmaf_bin,
            ref_y4m,
            dist_y4m,
            out_path,
            args.threads,
            args.out_fmt,
            frame_cnt,
            subsample,
        )

        if args.out_fmt == "csv":
            mean, p10 = parse_vmaf_csv_streaming(out_path)
        else:
            mean, p10 = parse_vmaf_json(out_path)
        print(json.dumps({"vmaf_mean": mean, "vmaf_p10": p10}, indent=2))


if __name__ == "__main__":
    main()


