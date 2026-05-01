import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO = Path("/home/tiehangz/proj/yolov12")
DEP = REPO / "deployment"
BIN = DEP / "build" / "Yolov12Deployment"
MODEL = DEP / "yolov12n_fc5_seg_v1.onnx"
VIDEO = Path("/home/tiehangz/proj/datasets/fps/roi_720p.mp4")


@dataclass
class RunCfg:
    conf: float
    mask_thr: float
    paint_alpha: float
    max_frames: int = 40

    def tag(self) -> str:
        def f(x: float) -> str:
            s = f"{x:.3f}".rstrip("0").rstrip(".")
            return s.replace(".", "p")
        return f"c{f(self.conf)}_s{f(self.mask_thr)}_a{f(self.paint_alpha)}_{self.max_frames}f"


def sh(cmd: List[str], log_path: Path | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed rc={r.returncode}: {' '.join(cmd)}")


def ffprobe_pkt_sizes(video_path: Path) -> List[int]:
    data = json.loads(
        subprocess.check_output(
            [
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
            ],
            text=True,
        )
    )
    return [int(fr["pkt_size"]) for fr in data.get("frames", []) if "pkt_size" in fr]


def saving_pct(orig: Path, masked: Path) -> Tuple[int, float, float, float, float]:
    b = ffprobe_pkt_sizes(orig)
    m = ffprobe_pkt_sizes(masked)
    n = min(len(b), len(m))
    b = b[:n]
    m = m[:n]
    bm = sum(b) / n
    mm = sum(m) / n
    saving = (bm - mm) / bm * 100.0 if bm > 0 else 0.0
    worse = sum(1 for i in range(n) if m[i] > b[i]) * 100.0 / n
    return n, bm, mm, saving, worse


def run_maskall(cfg: RunCfg, out_dir: Path) -> Dict:
    out = out_dir / "runs" / cfg.tag()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    cmd = [
        str(BIN),
        "-i",
        str(VIDEO),
        "-o",
        str(out),
        "-m",
        str(MODEL),
        "-c",
        str(cfg.conf),
        "-s",
        str(cfg.mask_thr),
        "--paint-alpha",
        str(cfg.paint_alpha),
        "--yolo-force-mask-all",
        "--yolo-class-color",
        "--max-frames",
        str(cfg.max_frames),
        "--timing",
    ]
    sh(cmd, log_path=log)

    rep = json.load(open(out / "report.json"))
    n, bm, mm, saving, worse = saving_pct(out / "original_output.mp4", out / "segmented_output.mp4")

    return {
        "tag": cfg.tag(),
        "conf": cfg.conf,
        "mask_thr": cfg.mask_thr,
        "paint_alpha": cfg.paint_alpha,
        "frames": n,
        "orig_avg_pkt": bm,
        "masked_avg_pkt": mm,
        "saving_pct": saving,
        "worse_pct": worse,
        "total_det": rep.get("total_detections"),
        "forced_masked": rep.get("yolo_forced_masked_regions"),
        "avg_ssim": rep.get("avg_ssim"),
    }


def ffmpeg_reencode(inp: Path, out: Path, crf: int, preset: str, keyint: int, bframes: int, tune: str | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    x264_params = f"keyint={keyint}:min-keyint={keyint}:scenecut=0:bframes={bframes}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(inp),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-x264-params",
        x264_params,
        "-an",
        str(out),
    ]
    if tune:
        cmd[cmd.index("-crf") + 2 : cmd.index("-x264-params")] = ["-tune", tune]
    sh(cmd)


def plot_figures(rows: List[Dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag_help = (
        "Tag format: c0p05_s0p3_a1_40f  =>  conf=0.05, mask_thr=0.3, paint_alpha=1.0, frames=40\n"
        "conf (-c): detection confidence threshold (filters boxes)\n"
        "mask_thr (-s): mask binarization threshold on sigmoid(mask) (changes mask area)"
    )

    # A) heatmap: conf x mask_thr -> saving
    confs = sorted({r["conf"] for r in rows})
    masks = sorted({r["mask_thr"] for r in rows})
    grid = [[None for _ in masks] for _ in confs]
    for r in rows:
        i = confs.index(r["conf"])
        j = masks.index(r["mask_thr"])
        grid[i][j] = r["saving_pct"]

    plt.figure(figsize=(7.6, 5.2))
    im = plt.imshow(grid, aspect="auto", cmap="viridis", vmin=min(min(r for r in row if r is not None) for row in grid), vmax=max(max(r for r in row if r is not None) for row in grid))
    plt.colorbar(im, label="saving % (orig vs masked)")
    plt.xticks(range(len(masks)), [str(m) for m in masks])
    plt.yticks(range(len(confs)), [str(c) for c in confs])
    plt.xlabel("mask threshold (-s)")
    plt.ylabel("confidence (-c)")
    plt.title("A) Upper-bound saving% vs YOLO thresholds (mask-all mode)")
    # annotate each cell
    for i in range(len(confs)):
        for j in range(len(masks)):
            v = grid[i][j]
            if v is None:
                continue
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.gcf().text(0.01, 0.01, tag_help, fontsize=8, va="bottom", ha="left", alpha=0.85)
    plt.tight_layout()
    plt.savefig(fig_dir / "A_threshold_heatmap.png", dpi=160)
    plt.close()

    # B) paint alpha curves per (conf, mask_thr) + overall mean
    alphas = sorted({r["paint_alpha"] for r in rows})
    pairs = sorted({(r["conf"], r["mask_thr"]) for r in rows})
    plt.figure(figsize=(7.2, 4.4))
    for (c, s) in pairs:
        rs = [r for r in rows if r["conf"] == c and r["mask_thr"] == s]
        rs.sort(key=lambda x: x["paint_alpha"])
        xs = [r["paint_alpha"] for r in rs]
        ys = [r["saving_pct"] for r in rs]
        plt.plot(xs, ys, marker="o", linewidth=1.0, alpha=0.35, label=f"c={c}, s={s}")
    alpha_avg = []
    for a in alphas:
        rs = [r for r in rows if r["paint_alpha"] == a]
        alpha_avg.append(sum(r["saving_pct"] for r in rs) / max(1, len(rs)))
    plt.plot(alphas, alpha_avg, marker="o", linewidth=2.5, color="black", label="mean")
    plt.xlabel("paintAlpha (--paint-alpha)")
    plt.ylabel("saving %")
    plt.title("B) Saving% vs paint alpha (mask-all mode)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=3, frameon=False)
    plt.gcf().text(0.01, 0.01, tag_help, fontsize=8, va="bottom", ha="left", alpha=0.85)
    plt.tight_layout()
    plt.savefig(fig_dir / "B_paint_alpha.png", dpi=160)
    plt.close()

    # C) top-K configs bar chart
    # C) Encoder sweep: take best (c, s) from Fig A at alpha=1.0, and sweep x264 (preset x CRF).
    rows_a1 = [r for r in rows if float(r["paint_alpha"]) >= 0.999]
    best = max(rows_a1 or rows, key=lambda r: r["saving_pct"])

    # Locate the run outputs for the selected best tag.
    run_dir = out_dir / "runs" / best["tag"]
    orig_in = run_dir / "original_output.mp4"
    masked_in = run_dir / "segmented_output.mp4"
    if not orig_in.exists() or not masked_in.exists():
        raise FileNotFoundError(f"missing run outputs for encoder sweep: {run_dir}")

    presets = ["ultrafast", "veryfast", "medium"]
    crfs = [18, 23, 28]
    # Default evaluation encoder settings (not strict latency):
    # tune=none, bframes=0
    keyint = 60
    bframes = 0
    tune = None

    enc_rows = []
    enc_dir = out_dir / "encoder_sweep"
    for preset in presets:
        for crf in crfs:
            o = enc_dir / f"orig_{preset}_crf{crf}.mp4"
            m = enc_dir / f"masked_{preset}_crf{crf}.mp4"
            ffmpeg_reencode(orig_in, o, crf=crf, preset=preset, keyint=keyint, bframes=bframes, tune=tune)
            ffmpeg_reencode(masked_in, m, crf=crf, preset=preset, keyint=keyint, bframes=bframes, tune=tune)
            n, bm, mm, saving, worse = saving_pct(o, m)
            enc_rows.append({
                "preset": preset,
                "crf": crf,
                "frames": n,
                "saving_pct": saving,
                "worse_pct": worse,
            })

    # Save encoder sweep CSV
    enc_csv = out_dir / "results_encoder_sweep.csv"
    with open(enc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(enc_rows[0].keys()))
        w.writeheader()
        w.writerows(enc_rows)

    # Plot heatmap: preset x CRF -> saving%
    grid = [[None for _ in crfs] for _ in presets]
    for r in enc_rows:
        i = presets.index(r["preset"])
        j = crfs.index(r["crf"])
        grid[i][j] = r["saving_pct"]
    plt.figure(figsize=(7.8, 4.6))
    im = plt.imshow(grid, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="saving % (re-encoded orig vs masked)")
    plt.xticks(range(len(crfs)), [str(c) for c in crfs])
    plt.yticks(range(len(presets)), presets)
    plt.xlabel("CRF (x264 quality)")
    plt.ylabel("x264 preset")
    plt.title("C) Encoder sweep (best thresholds from Fig A; alpha=1.0)")
    for i in range(len(presets)):
        for j in range(len(crfs)):
            v = grid[i][j]
            if v is None:
                continue
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.gcf().text(
        0.01, 0.01,
        f"Encoder params: tune={tune}, keyint={keyint}, bframes={bframes}\n"
        f"Best thresholds from Fig A: conf={best['conf']}, mask_thr={best['mask_thr']}, paint_alpha=1.0",
        fontsize=8, va="bottom", ha="left", alpha=0.85
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "C_best_in_sweep.png", dpi=160)
    plt.close()


def main() -> None:
    out_dir = REPO / "experiments" / "fc5_upperbound"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal-but-real sweep (kept small so it runs fast):
    # Use a wider range so thresholds actually change mask area / detection filtering.
    confs = [0.05, 0.20, 0.40]
    masks = [0.10, 0.30, 0.60]
    alphas = [0.3, 0.6, 1.0]

    rows: List[Dict] = []
    for a in alphas:
        for c in confs:
            for s in masks:
                cfg = RunCfg(conf=c, mask_thr=s, paint_alpha=a, max_frames=40)
                rows.append(run_maskall(cfg, out_dir))

    # Save results
    csv_path = out_dir / "results_maskall.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Figures A/B/C (from actual sweep)
    plot_figures(rows, out_dir)

    # D) synthetic 50% rectangle replacement (encode-only), using ffmpeg drawbox on the original 40-frame clip
    # Create a 40-frame clip to keep everything aligned.
    clip = out_dir / "synthetic" / "orig_40f.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(VIDEO), "-frames:v", "40", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", str(clip)])

    half = out_dir / "synthetic" / "half_replaced.mp4"
    # replace left half with black
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf", "drawbox=x=0:y=0:w=iw/2:h=ih:color=black@1:t=fill", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", str(half)])

    n, bm, mm, saving, worse = saving_pct(clip, half)
    synth = {
        "frames": n,
        "orig_avg_pkt": bm,
        "half_avg_pkt": mm,
        "saving_pct": saving,
        "worse_pct": worse,
        "note": "synthetic: replace 50% rectangle (left half) with constant black",
    }
    (out_dir / "synthetic" / "synthetic_half_result.json").write_text(json.dumps(synth, indent=2))

    # Simple D figure (bar compare): synthetic half vs best mask-all, with labels + area ratio
    import matplotlib.pyplot as plt
    best = max(rows, key=lambda r: r["saving_pct"])
    plt.figure(figsize=(6.6, 3.8))
    xs = ["mask-all best", "synthetic (area=50%)"]
    ys = [best["saving_pct"], synth["saving_pct"]]
    bars = plt.bar(xs, ys)
    plt.ylabel("saving %")
    plt.title("D) Saving%: best mask-all vs synthetic 50% area replacement")
    for b, v in zip(bars, ys):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.text(0.98, 0.06, "synthetic replaced area = 50% of pixels", transform=plt.gca().transAxes,
             ha="right", va="bottom", fontsize=9, alpha=0.8)
    plt.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "D_synthetic_vs_maskall.png", dpi=160)
    plt.close()

    # Extra: encoder knobs sweep (bframes, keyint, tune) on the best (c,s) at alpha=1.0.
    # This answers: do GOP/B-frames/tune change the ratio even without recovery?
    best_a1 = max([r for r in rows if float(r["paint_alpha"]) >= 0.999] or rows, key=lambda r: r["saving_pct"])
    run_dir = out_dir / "runs" / best_a1["tag"]
    orig_in = run_dir / "original_output.mp4"
    masked_in = run_dir / "segmented_output.mp4"
    if orig_in.exists() and masked_in.exists():
        preset = "veryfast"  # hold preset constant for this knobs study
        crf = 23            # hold CRF constant for this knobs study
        bframes_list = [0, 2]
        keyints = [30, 60, 120]
        tunes = [None, "zerolatency"]

        enc2 = []
        enc2_dir = out_dir / "encoder_sweep_knobs"
        for keyint in keyints:
            for bframes in bframes_list:
                for tune in tunes:
                    tname = "none" if tune is None else tune
                    o = enc2_dir / f"orig_{preset}_crf{crf}_k{keyint}_b{bframes}_t{tname}.mp4"
                    m = enc2_dir / f"masked_{preset}_crf{crf}_k{keyint}_b{bframes}_t{tname}.mp4"
                    ffmpeg_reencode(orig_in, o, crf=crf, preset=preset, keyint=keyint, bframes=bframes, tune=tune)
                    ffmpeg_reencode(masked_in, m, crf=crf, preset=preset, keyint=keyint, bframes=bframes, tune=tune)
                    n2, bm2, mm2, saving2, worse2 = saving_pct(o, m)
                    enc2.append({
                        "preset": preset,
                        "crf": crf,
                        "keyint": keyint,
                        "bframes": bframes,
                        "tune": tname,
                        "frames": n2,
                        "saving_pct": saving2,
                        "worse_pct": worse2,
                    })

        enc2_csv = out_dir / "results_encoder_knobs.csv"
        with open(enc2_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(enc2[0].keys()))
            w.writeheader()
            w.writerows(enc2)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.8, 4.4))
        for bframes in bframes_list:
            for tune in tunes:
                tname = "none" if tune is None else tune
                rs = [r for r in enc2 if r["bframes"] == bframes and r["tune"] == tname]
                rs.sort(key=lambda x: x["keyint"])
                xs = [r["keyint"] for r in rs]
                ys = [r["saving_pct"] for r in rs]
                plt.plot(xs, ys, marker="o", label=f"bframes={bframes}, tune={tname}")
        plt.xlabel("keyint (GOP length)")
        plt.ylabel("saving % (re-encoded orig vs masked)")
        plt.title("C-extra) Encoder knobs sweep (preset=veryfast, CRF=23)")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / "C_encoder_knobs.png", dpi=160)
        plt.close()

    print("Wrote:", csv_path)
    print("Figures in:", fig_dir)


if __name__ == "__main__":
    if not BIN.exists():
        raise SystemExit(f"Missing binary: {BIN}")
    if not MODEL.exists():
        raise SystemExit(f"Missing model: {MODEL}")
    main()


