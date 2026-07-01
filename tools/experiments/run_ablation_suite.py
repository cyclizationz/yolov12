#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from common import CURRENT_BASELINE_BIN, DEFAULT_MODEL, OFFLINE_BIN, RESPAWN2026_DIR, ensure_dir, load_manifest, safe_name
from pixel_mario_defaults import mario_pixel_args
from video_metrics import compute_video_metrics


PIXEL_TEMPLATES = Path("/home/tiehangz/proj/datasets/pixel/templates_rescale")
PIXEL_SINGLE_TEMPLATE = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/_pixel_single_template")
FM6_MODEL = Path("/home/tiehangz/proj/yolov12/deployment/yolov12n_racing_e300_split1.onnx")
FC5_MODEL = Path("/home/tiehangz/proj/yolov12/deployment/yolov12n_fc5_seg_v1.onnx")
FM6_LATENT_BANK = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/fm6_index_full_v1/dict/latent_bank.json")
FC5_LATENT_BANK = Path("/home/tiehangz/proj/yolov12/experiments/encoder_eval/fc5_crop_index_full_v1/dict/latent_bank.json")


def ffprobe_bitrate_bps(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
        text=True,
    ).strip()
    dur = float(out or 0.0)
    return (path.stat().st_size * 8.0) / dur if dur > 1e-9 else 0.0


def delivered_bps(run_dir: Path) -> float:
    video = run_dir / "segmented_output.mp4"
    meta = run_dir / "msk1_payloads.bin"
    bps = ffprobe_bitrate_bps(video) if video.exists() else 0.0
    if meta.exists() and video.exists():
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(video)],
            text=True,
        ).strip()
        dur = float(out or 0.0)
        if dur > 1e-9:
            bps += (meta.stat().st_size * 8.0) / dur
    return bps


def reference_bps(run_dir: Path) -> float:
    ref = run_dir / "original_output.mp4"
    return ffprobe_bitrate_bps(ref) if ref.exists() else 0.0


def materialize_source_window(clip: Any, out_root: Path) -> str:
    source = Path(clip.clip_path if clip.clip_path and Path(clip.clip_path).exists() else clip.source_path)
    try:
        source_duration = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(source)],
                text=True,
            ).strip()
            or 0.0
        )
    except Exception:
        source_duration = 0.0
    if float(clip.start_s) <= 1e-6 and source_duration > 0.0 and float(clip.duration_s) >= source_duration - 0.5:
        return str(source)
    out_path = out_root / "_inputs" / f"{safe_name(clip.clip_id)}.mp4"
    if out_path.exists():
        return str(out_path)
    ensure_dir(out_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{float(clip.start_s):.3f}",
        "-t",
        f"{float(clip.duration_s):.3f}",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "12",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)


def artifact_incidents_from_report(report: dict[str, Any], *, ssim_thr: float = 0.95) -> float | None:
    per = report.get("per_frame", []) or []
    if not per:
        return None
    bad = 0
    for item in per:
        rec_ssim = item.get("rec_ssim", None)
        if rec_ssim is not None and float(rec_ssim) < ssim_thr:
            bad += 1
    return (bad * 10000.0) / len(per)


def artifact_frame_count_from_report(report: dict[str, Any], *, ssim_thr: float = 0.95) -> int:
    per = report.get("per_frame", []) or []
    bad = 0
    for item in per:
        rec_ssim = item.get("rec_ssim", None)
        if rec_ssim is not None and float(rec_ssim) < ssim_thr:
            bad += 1
    return bad


def learned_base_args(clip: Any) -> list[str]:
    latent_thr = "0.85" if clip.game == "fc5" else "0.86"
    base = [
        "--latent-key",
        "--yolo-heal-only",
        "--latent-thr",
        latent_thr,
        "--latent-motion-iou",
        "0.6",
        "--latent-motion-center",
        "20",
        "--latent-motion-scale",
        "0.25",
        "--latent-motion-boost",
        "6",
        "--mask-color",
        "dominant",
        "--mask-color-period",
        "200",
        "--fill-mode",
        "solid",
        "--feather-px",
        "4",
    ]
    if clip.game == "fc5":
        base += ["--yolo-heal-iou", "0.999", "--yolo-heal-extra", "0.01"]
    if clip.game == "fm6":
        return base + ["--latent-bank", str(FM6_LATENT_BANK)]
    if clip.game == "fc5":
        return base + ["--latent-bank", str(FC5_LATENT_BANK)]
    return base


def learned_ablation_configs(clip: Any) -> list[dict[str, Any]]:
    base = learned_base_args(clip)
    configs: list[dict[str, Any]] = [
        {"name": "latent_key_cosine", "args": base + ["--yolo-matcher", "latent_key"], "kind": "matching"},
        {"name": "phash_roi", "args": base + ["--yolo-matcher", "phash"], "kind": "matching"},
        {"name": "iou_only_tracking", "args": base + ["--yolo-matcher", "iou_only"], "kind": "matching"},
        {"name": "rgb_histogram", "args": base + ["--yolo-matcher", "rgb_hist"], "kind": "matching"},
    ]
    for tau_cos in (0.85, 0.90, 0.95, 0.98):
        for tau_iou in (0.98, 0.99, 0.995, 0.999):
            for tau_spill in (0.0, 0.005, 0.01, 0.02):
                configs.append(
                    {
                        "name": f"gate_cos{tau_cos}_iou{tau_iou}_spill{tau_spill}",
                        "args": base + [
                            "--yolo-matcher",
                            "latent_key",
                            "--latent-thr",
                            str(tau_cos),
                            "--yolo-heal-iou",
                            str(tau_iou),
                            "--yolo-heal-extra",
                            str(tau_spill),
                        ],
                        "kind": "gating",
                    }
                )
    for fill in ("dominant", "black", "rgb:128,128,128"):
        for feather in (0, 3):
            configs.append(
                {
                    "name": f"fill_{safe_name(fill)}_feather_{feather}",
                    "args": base + [
                        "--yolo-matcher",
                        "latent_key",
                        "--mask-color",
                        fill,
                        "--feather-px",
                        str(feather),
                    ],
                    "kind": "fill",
                }
            )
    return configs


def mario_ablation_configs() -> list[dict[str, Any]]:
    base = mario_pixel_args()
    return [
        {"name": "full_pipeline", "args": base, "kind": "mario"},
        {"name": "no_optical_flow", "args": base + ["--pixel-flow", "0"], "kind": "mario"},
        {"name": "no_kalman", "args": base + ["--pixel-kalman", "0"], "kind": "mario"},
        {"name": "template_only", "args": base + ["--pixel-flow", "0", "--pixel-kalman", "0"], "kind": "mario"},
    ]


def plot_ablation(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if plt is None:
        return
    learned = [r for r in rows if r.get("status") == "ok" and r.get("game") in {"fc5", "fm6"}]
    if learned:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(
            [float(r.get("artifact_incidents_per_10k_frames", 0.0) or 0.0) for r in learned],
            [float(r.get("bsp_pct", 0.0) or 0.0) for r in learned],
        )
        ax.set_title("Ablation Pareto: artifacts vs BSP")
        ax.set_xlabel("Artifact incidents per 10k frames")
        ax.set_ylabel("BSP (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_pareto.png")
        plt.close(fig)

        fill_rows = [r for r in learned if r.get("kind") == "fill"]
        if fill_rows:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.scatter(
                [float(r.get("respawn_vmaf_mean", 0.0) or 0.0) for r in fill_rows],
                [float(r.get("bsp_pct", 0.0) or 0.0) for r in fill_rows],
            )
            for row in fill_rows:
                ax.annotate(str(row["config_name"]), (float(row.get("respawn_vmaf_mean", 0.0) or 0.0), float(row.get("bsp_pct", 0.0) or 0.0)), fontsize=7)
            ax.set_title("Fill strategy vs savings/quality")
            ax.set_xlabel("VMAF")
            ax.set_ylabel("BSP (%)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "fill_strategy_vs_vmaf_bsp.png")
            plt.close(fig)


def encoder_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "exp35_encoder", False):
        return [
            "--enc-codec",
            "libx264",
            "--enc-crf",
            "23",
            "--enc-preset",
            "medium",
            "--enc-tune",
            "none",
            "--enc-profile",
            "none",
            "--enc-level",
            "none",
            "--enc-open-gop-defaults",
            "--enc-aud",
            "1",
            "--enc-repeat-headers",
            "0",
        ]
    base = [
        "--enc-gop",
        "60",
        "--enc-preset",
        "superfast",
        "--enc-tune",
        "none",
        "--enc-scenecut",
        "0",
    ]
    if args.bitrate_mbps > 0:
        return [
            "--enc-bitrate-mbps",
            str(args.bitrate_mbps),
            "--enc-maxrate-mbps",
            str(args.bitrate_mbps),
            "--enc-bufsize-mbits",
            str(2.0 * args.bitrate_mbps),
        ] + base
    return ["--enc-crf", str(args.enc_crf)] + base


def model_for_clip(clip: Any) -> Path:
    if clip.game == "fm6":
        return FM6_MODEL
    if clip.game == "fc5":
        return FC5_MODEL
    return Path(DEFAULT_MODEL)


def fps_from_report(report: dict[str, Any], default_fps: float) -> float:
    s = (((report.get("input_ffprobe") or {}).get("streams") or [{}])[0].get("avg_frame_rate", "0/1"))
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) if float(b) else default_fps
        return float(s)
    except Exception:
        return default_fps


def main() -> None:
    ap = argparse.ArgumentParser(description="Run offline ablation matrix for Experiment 4.")
    ap.add_argument("--manifest", type=Path, default=RESPAWN2026_DIR / "manifest" / "offline_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "exp4")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--bitrate-mbps", type=float, default=0.0)
    ap.add_argument("--enc-crf", type=int, default=18)
    ap.add_argument("--limit-clips-per-game", type=int, default=2)
    ap.add_argument("--compute-video-metrics", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--config-kinds", nargs="*", default=[])
    ap.add_argument("--clip-ids", nargs="*", default=[])
    ap.add_argument("--exp35-encoder", action="store_true", help="Use the CRF23/open-GOP encoder stack from Exp3/Exp5.")
    ap.add_argument("--use-normalized-input", action="store_true", help="Use manifest-normalized clips directly instead of rematerializing source windows.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest = load_manifest(args.manifest)
    by_game: dict[str, list[Any]] = defaultdict(list)
    for clip in manifest:
        by_game[clip.game].append(clip)
    clips = []
    for game, items in by_game.items():
        clips.extend(items[: args.limit_clips_per_game])
    if args.clip_ids:
        clip_id_filter = set(args.clip_ids)
        clips = [clip for clip in clips if clip.clip_id in clip_id_filter]

    rows: list[dict[str, Any]] = []
    for clip in clips:
        clip_configs = mario_ablation_configs() if clip.game == "mario" else learned_ablation_configs(clip)
        if args.config_kinds:
            clip_configs = [cfg for cfg in clip_configs if cfg.get("kind") in set(args.config_kinds)]
        if not clip_configs:
            continue
        clip_input = str(Path(clip.normalized_path)) if args.use_normalized_input else materialize_source_window(clip, args.out_dir)
        clip_model = model_for_clip(clip)
        baseline_dir = args.out_dir / safe_name(clip.clip_id) / "_baseline_current"
        if not args.exp35_encoder and not (baseline_dir / "report.json").exists():
            ensure_dir(baseline_dir)
            baseline_game_args = []
            if clip.game == "mario":
                baseline_game_args = ["--pixel", "-T", str(PIXEL_SINGLE_TEMPLATE)]
            subprocess.run(
                [
                    str(CURRENT_BASELINE_BIN),
                    "-i",
                    clip_input,
                    "-o",
                    str(baseline_dir),
                    "-m",
                    str(clip_model),
                ]
                + encoder_args(args)
                + (["--max-frames", str(args.max_frames)] if args.max_frames > 0 else [])
                + baseline_game_args
                + ["--latent-key", "--yolo-heal-only"],
                check=True,
            )
        for cfg in clip_configs:
            if cfg.get("status") == "unsupported":
                rows.append(
                    {
                        "clip_id": clip.clip_id,
                        "game": clip.game,
                        "config_name": cfg["name"],
                        "kind": cfg["kind"],
                        "status": "unsupported",
                    }
                )
                continue
            run_dir = args.out_dir / safe_name(clip.clip_id) / safe_name(cfg["name"])
            ensure_dir(run_dir)
            if not (run_dir / "report.json").exists():
                cmd = [
                    str(OFFLINE_BIN),
                    "-i",
                    clip_input,
                    "-o",
                    str(run_dir),
                    "-m",
                    str(clip_model),
                ] + encoder_args(args) + (["--max-frames", str(args.max_frames)] if args.max_frames > 0 else []) + list(cfg["args"])
                subprocess.run(cmd, check=True)
            report = json.loads((run_dir / "report.json").read_text())
            per = report.get("per_frame", []) or []
            duration_min = (len(per) / max(1e-9, fps_from_report(report, clip.source_fps) * 60.0)) if per else 0.0
            ref_ratio = (
                sum(1.0 if int(item.get("frame_flags", 0) or 0) > 0 else 0.0 for item in per) / len(per)
                if per
                else 0.0
            )
            masked_present = (
                sum(float(item.get("object_successfully_masked", 0) or 0) for item in per)
                / max(1.0, sum(float(item.get("object_present_model", 0) or 0) for item in per))
                if per
                else 0.0
            )
            respawn_bps = delivered_bps(run_dir)
            ref_bps = reference_bps(run_dir)
            bsp = (1.0 - respawn_bps / ref_bps) * 100.0 if ref_bps > 1e-9 else None
            row = {
                "clip_id": clip.clip_id,
                "game": clip.game,
                "config_name": cfg["name"],
                "kind": cfg["kind"],
                "status": "ok",
                "match_rate": (float(report.get("matched_detections", 0.0) or 0.0) / float(max(1, report.get("total_detections", 0.0) or 0.0))),
                "latent_minted": float(report.get("latent_minted", 0.0) or 0.0),
                "latent_reused": float(report.get("latent_reused", 0.0) or 0.0),
                "ref_ratio": ref_ratio,
                "masked_present_ratio": masked_present,
                "avg_recovered_ssim": report.get("avg_recovered_ssim"),
                "avg_recovered_psnr": report.get("avg_recovered_psnr"),
                "artifact_frames": artifact_frame_count_from_report(report),
                "artifact_incidents_per_10k_frames": artifact_incidents_from_report(report),
                "matcher_new_templates": int(report.get("matcher_new_templates", 0) or 0),
                "matcher_id_switches": int(report.get("matcher_id_switches", 0) or 0),
                "new_templates_per_min": ((float(report.get("matcher_new_templates", 0) or 0) / duration_min) if duration_min > 1e-9 else None),
                "id_switches_per_min": ((float(report.get("matcher_id_switches", 0) or 0) / duration_min) if duration_min > 1e-9 else None),
                "yolo_matcher": report.get("yolo_matcher", "pixel"),
                "bsp_pct": bsp,
                "timing_total_ms": ((report.get("timing_avg_ms", {}) or {}).get("total")),
                "run_dir": str(run_dir),
            }
            if args.compute_video_metrics:
                metrics = compute_video_metrics(
                    ref_video=Path(clip.normalized_path),
                    dist_video=run_dir / "recovered_output.mp4",
                    msk1_bin=run_dir / "msk1_payloads.bin",
                    threads=1,
                    scale_height=1080,
                )
                row["respawn_vmaf_mean"] = metrics["full_frame"]["vmaf_mean"]
                row["roi_vmaf_mean"] = metrics["roi"]["vmaf_mean"]
                row["roi_ssim_mean"] = metrics["roi"]["ssim_mean"]
                row["roi_psnr_mean"] = metrics["roi"]["psnr_mean"]
            rows.append(row)

    out_csv = args.out_dir / "ablation_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    plot_ablation(rows, args.out_dir)
    print(json.dumps({"summary_csv": str(out_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
