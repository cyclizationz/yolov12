import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class Case:
    key: str
    label: str
    report: Path
    vmaf_ref: Path
    vmaf_dist: Path


def load_report_series(report_json: Path) -> tuple[list[float], list[float], list[float]]:
    rep = json.loads(report_json.read_text())
    per = rep.get("per_frame", []) or []
    ssim = []
    psnr = []
    total_ms = []
    for e in per:
        if "rec_ssim" in e:
            ssim.append(float(e["rec_ssim"]))
        if "rec_psnr" in e:
            psnr.append(float(e["rec_psnr"]))
        if "total_ms" in e:
            total_ms.append(float(e["total_ms"]))
    return ssim, psnr, total_ms


def compute_vmaf_per_frame(vmaf_bin: Path, ref: Path, dist: Path, threads: int) -> list[float]:
    libdir = vmaf_bin.parent.parent / "lib" / "x86_64-linux-gnu"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"

    with tempfile.TemporaryDirectory(prefix="vmaf_cdf_") as td:
        td = Path(td)
        ref_y4m = td / "ref.y4m"
        dist_y4m = td / "dist.y4m"
        out_json = td / "out.json"

        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error", "-i", str(ref), "-pix_fmt", "yuv420p", "-an", "-f", "yuv4mpegpipe", str(ref_y4m)]
        )
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error", "-i", str(dist), "-pix_fmt", "yuv420p", "-an", "-f", "yuv4mpegpipe", str(dist_y4m)]
        )
        subprocess.check_call(
            [
                str(vmaf_bin),
                "--reference",
                str(ref_y4m),
                "--distorted",
                str(dist_y4m),
                "--model",
                "version=vmaf_v0.6.1",
                "--json",
                "--output",
                str(out_json),
                "--threads",
                str(int(threads)),
                "--quiet",
            ],
            env=env,
        )

        j = json.loads(out_json.read_text())
        vals: list[float] = []
        for fr in j.get("frames", []) or []:
            v = fr.get("metrics", {}).get("vmaf")
            if v is None:
                v = fr.get("vmaf")
            if v is None:
                continue
            vals.append(float(v))
        return vals


def ecdf(vals: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(sorted(vals), dtype=np.float64)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / float(len(x))
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot CDFs for reconstruction quality + delay across cases.")
    ap.add_argument("--out", required=True, type=Path, help="Output PNG")
    ap.add_argument(
        "--vmaf-bin",
        type=Path,
        default=Path("/home/tiehangz/proj/yolov12/third_party/vmaf/libvmaf/install/bin/vmaf"),
    )
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cases = [
        Case(
            key="pixel",
            label="Pixel (Mario)",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/report.json"),
            vmaf_ref=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/original_output.mp4"),
            vmaf_dist=Path("/home/tiehangz/proj/yolov12/deployment/outputs/pixel_kalman_band2_flow_v1/recovered_output.mp4"),
        ),
        Case(
            key="fm6",
            label="Racing (FM6)",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/report.json"),
            vmaf_ref=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/original_output.mp4"),
            vmaf_dist=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fm6_latent_hybrid_v2_reuse_stats/recovered_output.mp4"),
        ),
        Case(
            key="fc5gun",
            label="FC5 (gun-heavy crop)",
            report=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/report.json"),
            vmaf_ref=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/original_output.mp4"),
            vmaf_dist=Path("/home/tiehangz/proj/yolov12/experiments/fc5_inband_flow_gunheavy/server/recovered_output.mp4"),
        ),
        Case(
            key="fc5full",
            label="FC5 (full, same ROI; defaults)",
            report=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/report.json"),
            vmaf_ref=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/original_output.mp4"),
            vmaf_dist=Path("/home/tiehangz/proj/yolov12/deployment/outputs/yolo_fc5_fullcrop_1366_rerun_v5_defaults_c02_s06_a10/recovered_output.mp4"),
        ),
    ]

    # Load series
    data = {}
    for c in cases:
        ssim, psnr, total_ms = load_report_series(c.report)
        vmaf = compute_vmaf_per_frame(args.vmaf_bin, c.vmaf_ref, c.vmaf_dist, args.threads)
        data[c.key] = {"label": c.label, "vmaf": vmaf, "ssim": ssim, "psnr": psnr, "ms": total_ms}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)

    metrics = [
        ("vmaf", "VMAF (reconstructed vs baseline)", "Score"),
        ("ssim", "SSIM (reconstructed vs baseline)", "SSIM"),
        ("psnr", "PSNR (reconstructed vs baseline)", "PSNR (dB)"),
        ("ms", "End-to-end delay proxy", "total ms/frame"),
    ]

    for ax, (k, title, xlabel) in zip(axes, metrics):
        for ck, d in data.items():
            vals = d[k]
            if not vals:
                continue
            x, y = ecdf(vals)
            ax.plot(x, y, label=d["label"])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote CDF figure: {args.out}")


if __name__ == "__main__":
    main()


