import argparse
import csv
import json
import re
import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RdPoint:
    rate_bps: float
    metric: float


def _ffprobe_duration_s(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)]
    ).decode("utf-8", errors="replace")
    try:
        return float(out.strip())
    except Exception:
        return 0.0


def _ffprobe_bitrate_bps(path: Path) -> float:
    # bitrate = (file_size_bits) / duration
    dur = _ffprobe_duration_s(path)
    if dur <= 1e-9:
        return 0.0
    return (path.stat().st_size * 8.0) / dur


def _msk1_overhead_bps(msk1_path: Path, duration_s: float) -> float:
    if not msk1_path.exists() or duration_s <= 1e-9:
        return 0.0
    return (msk1_path.stat().st_size * 8.0) / duration_s


def _run_vmaf_score_json(ref: Path, dist: Path, threads: int) -> dict[str, Any]:
    # Use the repo's vmaf_score.py (uses vmaf CLI from third_party).
    cmd = [
        "python",
        "/home/tiehangz/proj/yolov12/tools/metrics/vmaf_score.py",
        "--ref",
        str(ref),
        "--dist",
        str(dist),
        "--threads",
        str(int(threads)),
        # Keep intermediate y4m off tmpfs (/tmp) to avoid OOM on low-RAM systems.
        "--temp-dir",
        "/home/tiehangz/proj/yolov12/record/_tmp_vmaf",
        # CSV is smaller and parsed streaming (lower RAM).
        "--out-fmt",
        "csv",
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="replace").strip()
    return json.loads(out)


def _bd_rate(points_ref: list[RdPoint], points_test: list[RdPoint]) -> float | None:
    # Approx BD-Rate:
    # - Interpolate log(rate) over metric using linear interpolation (numpy-only).
    # - Integrate with trapezoid rule.
    #
    # This avoids SciPy (and its RAM/dep footprint), and is stable enough for sweeps.
    # Requires 4 points per curve.
    if len(points_ref) < 4 or len(points_test) < 4:
        return None

    p1 = sorted(points_ref, key=lambda p: p.metric)
    p2 = sorted(points_test, key=lambda p: p.metric)

    def _prep(points: list[RdPoint]) -> tuple[np.ndarray, np.ndarray]:
        m = np.array([p.metric for p in points], dtype=np.float64)
        r = np.log(np.array([max(1e-9, p.rate_bps) for p in points], dtype=np.float64))
        # Ensure strictly increasing metric for np.interp by collapsing duplicates.
        order = np.argsort(m)
        m = m[order]
        r = r[order]
        if m.size == 0:
            return m, r
        uniq_m = [float(m[0])]
        agg_r = [float(r[0])]
        agg_n = [1]
        for i in range(1, m.size):
            if float(m[i]) == uniq_m[-1]:
                agg_r[-1] += float(r[i])
                agg_n[-1] += 1
            else:
                uniq_m.append(float(m[i]))
                agg_r.append(float(r[i]))
                agg_n.append(1)
        m2 = np.array(uniq_m, dtype=np.float64)
        r2 = np.array([agg_r[i] / agg_n[i] for i in range(len(uniq_m))], dtype=np.float64)
        return m2, r2

    m1, r1 = _prep(p1)
    m2, r2 = _prep(p2)
    if m1.size < 2 or m2.size < 2:
        return None

    min_int = max(float(m1.min()), float(m2.min()))
    max_int = min(float(m1.max()), float(m2.max()))
    if max_int <= min_int:
        return None

    samples, interval = np.linspace(min_int, max_int, num=100, retstep=True)
    v1 = np.interp(samples, m1, r1)
    v2 = np.interp(samples, m2, r2)

    int_v1 = float(np.trapz(v1, dx=float(interval)))
    int_v2 = float(np.trapz(v2, dx=float(interval)))
    avg = (int_v2 - int_v1) / (max_int - min_int)
    return float(np.exp(avg) - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze codec sweep runs under record/codec (bitrate, VMAF, BD-Rate).")
    ap.add_argument("--codec-dir", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/codec"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/codec"))
    ap.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Threads passed to VMAF CLI (higher uses more RAM).",
    )
    ap.add_argument("--force-vmaf", action="store_true", help="Recompute VMAF even if vmaf_cache.json exists.")
    args = ap.parse_args()

    src = {
        "fc5_crop": Path("/home/tiehangz/proj/yolov12/video/fc5_crop.mkv"),
        "fm6": Path("/home/tiehangz/proj/yolov12/video/fm6.mkv"),
        "mario": Path("/home/tiehangz/proj/yolov12/video/pixel.mkv"),
    }

    # Folder name schema used in this project:
    #  - fc5_crop_codec-libx264_crf18_...
    #  - mario_codec-libx264_crf18_...
    rx = re.compile(r"^(?P<trace>fc5_crop|fm6|mario)_codec-(?P<codec>[^_]+)_crf(?P<crf>\d+)_")

    runs: list[dict[str, Any]] = []
    for d in sorted(args.codec_dir.iterdir()):
        if not d.is_dir():
            continue
        m = rx.match(d.name)
        if not m:
            continue
        trace = m.group("trace")
        codec = m.group("codec")
        crf = int(m.group("crf"))

        ref = src.get(trace)
        if not ref or not ref.exists():
            continue

        orig = d / "original_output.mp4"
        masked = d / "segmented_output.mp4"
        recovered = d / "recovered_output.mp4"
        report = d / "report.json"
        msk1 = d / "msk1_payloads.bin"
        if not (orig.exists() and masked.exists() and recovered.exists() and report.exists()):
            continue

        dur = _ffprobe_duration_s(orig)
        baseline_bps = _ffprobe_bitrate_bps(orig)
        masked_bps = _ffprobe_bitrate_bps(masked)
        meta_bps = _msk1_overhead_bps(msk1, dur)
        masked_plus_meta_bps = masked_bps + meta_bps
        masked_plus_meta_over_baseline = (masked_plus_meta_bps / baseline_bps) if baseline_bps > 1e-9 else None
        saving_pct = (1.0 - masked_plus_meta_over_baseline) * 100.0 if masked_plus_meta_over_baseline is not None else None

        cache_path = d / "vmaf_cache.json"
        vmaf_baseline = None
        vmaf_recovered = None
        if cache_path.exists() and not args.force_vmaf:
            try:
                c = json.loads(cache_path.read_text())
                vmaf_baseline = float(c.get("vmaf_baseline"))
                vmaf_recovered = float(c.get("vmaf_recovered"))
            except Exception:
                vmaf_baseline = None
                vmaf_recovered = None

        if vmaf_baseline is None or vmaf_recovered is None:
            vb = _run_vmaf_score_json(ref, orig, args.threads)
            vr = _run_vmaf_score_json(ref, recovered, args.threads)
            vmaf_baseline = float(vb.get("vmaf_mean"))
            vmaf_recovered = float(vr.get("vmaf_mean"))
            cache_path.write_text(
                json.dumps({"vmaf_baseline": vmaf_baseline, "vmaf_recovered": vmaf_recovered}, indent=2)
            )

        runs.append(
            {
                "run_dir": str(d),
                "trace": trace,
                "codec": codec,
                "crf": crf,
                "duration_s": dur,
                "baseline_bps": baseline_bps,
                "masked_bps": masked_bps,
                "meta_bps": meta_bps,
                "masked_plus_meta_bps": masked_plus_meta_bps,
                # Requested: saving ratio = (masked+meta)/baseline. Lower is better.
                "masked_plus_meta_over_baseline": masked_plus_meta_over_baseline,
                # Convenience: percent saving (positive means fewer bits than baseline).
                "saving_pct": saving_pct,
                "vmaf_baseline": vmaf_baseline,
                "vmaf_recovered": vmaf_recovered,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not runs:
        raise SystemExit(f"No valid runs found under {args.codec_dir}")

    rd_csv = args.out_dir / "codec_rd_points.csv"
    with rd_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        for r in sorted(runs, key=lambda x: (x["trace"], x["codec"], x["crf"])):
            w.writerow(r)

    # BD-Rate: compare (masked+meta vs recovered VMAF) against (baseline vs baseline VMAF) within each trace+codec.
    bd_rows: list[dict[str, Any]] = []
    for trace in sorted({r["trace"] for r in runs}):
        for codec in sorted({r["codec"] for r in runs if r["trace"] == trace}):
            rr = [r for r in runs if r["trace"] == trace and r["codec"] == codec]
            base_pts = [RdPoint(rate_bps=float(r["baseline_bps"]), metric=float(r["vmaf_baseline"])) for r in rr]
            test_pts = [RdPoint(rate_bps=float(r["masked_plus_meta_bps"]), metric=float(r["vmaf_recovered"])) for r in rr]
            bd = _bd_rate(base_pts, test_pts)
            bd_rows.append(
                {
                    "trace": trace,
                    "codec": codec,
                    # Negative means masked pipeline needs fewer bits for same VMAF (good).
                    "bd_rate_masked_vs_baseline_at_equal_vmaf": bd,
                }
            )

    bd_csv = args.out_dir / "codec_bd_rate_summary.csv"
    with bd_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "codec", "bd_rate_masked_vs_baseline_at_equal_vmaf"])
        w.writeheader()
        for r in sorted(bd_rows, key=lambda x: (x["trace"], x["codec"])):
            w.writerow(r)

    # Best codec per trace (min BD-rate)
    best = {}
    for r in bd_rows:
        bd = r["bd_rate_masked_vs_baseline_at_equal_vmaf"]
        if bd is None:
            continue
        t = r["trace"]
        if t not in best or float(bd) < float(best[t]["bd_rate_masked_vs_baseline_at_equal_vmaf"]):
            best[t] = r

    best_csv = args.out_dir / "best_codec_per_trace.csv"
    with best_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "codec", "bd_rate_masked_vs_baseline_at_equal_vmaf"])
        w.writeheader()
        for t in sorted(best.keys()):
            w.writerow(best[t])

    print("WROTE", rd_csv)
    print("WROTE", bd_csv)
    print("WROTE", best_csv)
    for t in sorted(best.keys()):
        bd = float(best[t]["bd_rate_masked_vs_baseline_at_equal_vmaf"])
        print(f"BEST trace={t} codec={best[t]['codec']} bd_rate={bd*100:.2f}%")

    # Also report best (min ratio) point per trace across all runs.
    best_ratio = {}
    for r in runs:
        ratio = r.get("masked_plus_meta_over_baseline")
        if ratio is None:
            continue
        t = r["trace"]
        if t not in best_ratio or float(ratio) < float(best_ratio[t]["masked_plus_meta_over_baseline"]):
            best_ratio[t] = r
    for t in sorted(best_ratio.keys()):
        r = best_ratio[t]
        ratio = float(r["masked_plus_meta_over_baseline"])
        sp = r.get("saving_pct")
        sp_s = f"{float(sp):.2f}%" if sp is not None else "NA"
        print(
            f"BEST_SAVING trace={t} codec={r['codec']} crf={r['crf']} "
            f"(masked+meta)/baseline={ratio:.4f} saving_pct={sp_s}"
        )


if __name__ == "__main__":
    main()

