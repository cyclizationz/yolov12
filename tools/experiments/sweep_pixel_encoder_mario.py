#!/usr/bin/env python3
"""
Mario pixel + H.264 sweep: coverage-first, single anchor bitrate, open-GOP axis.

Default VMAF: segmented vs same-run original (`--vmaf-mode segmented_vs_original`).
Pass `--vmaf-mode recovered_vs_source` for normalized source vs recovered.

See record/RESPAWN2026/pixel_enc_sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from common import (
    DEFAULT_MODEL,
    OFFLINE_BIN,
    REPO_ROOT,
    RESPAWN2026_DIR,
    ensure_dir,
    load_manifest,
    normalize_clip,
    python_bin,
)
from pixel_mario_defaults import PIXEL_TEMPLATES, mario_pixel_args
from video_metrics import compute_video_metrics

# Plan defaults
DEFAULT_ANCHOR_MBPS = 16.0
DEFAULT_VMAF_MIN = 55.0
DEFAULT_TOPK_VMAF = 12
MARIO_CLIP_IDS = ("mario_00", "mario_01", "mario_02", "mario_03", "mario_04")

VMAF_SEGMENTED_VS_ORIGINAL = "segmented_vs_original"
VMAF_RECOVERED_VS_SOURCE = "recovered_vs_source"
VMAF_MODES = (VMAF_SEGMENTED_VS_ORIGINAL, VMAF_RECOVERED_VS_SOURCE)


def vmaf_paths(
    respawn_dir: Path,
    *,
    source_clip: Path,
    mode: str,
) -> tuple[Path, Path, Path | None]:
    """Return (ref_video, dist_video, msk1_optional) for compute_video_metrics."""
    if mode == VMAF_SEGMENTED_VS_ORIGINAL:
        return (
            respawn_dir / "original_output.mp4",
            respawn_dir / "segmented_output.mp4",
            None,
        )
    if mode == VMAF_RECOVERED_VS_SOURCE:
        msk = respawn_dir / "msk1_payloads.bin"
        return source_clip, respawn_dir / "recovered_output.mp4", msk if msk.exists() else None
    raise ValueError(f"unknown vmaf mode: {mode!r}")


def ffprobe_duration_s(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def bitrate_bps(video: Path) -> float:
    if not video.exists():
        return 0.0
    dur = ffprobe_duration_s(video)
    if dur <= 1e-9:
        return 0.0
    return float(video.stat().st_size) * 8.0 / dur


def delivered_bps(segmented_mp4: Path, msk1: Path) -> tuple[float, float, float]:
    vb = bitrate_bps(segmented_mp4)
    dur = ffprobe_duration_s(segmented_mp4)
    mb = ((msk1.stat().st_size * 8.0) / dur) if msk1.exists() and dur > 1e-9 else 0.0
    return vb + mb, vb, mb


def last_cli_flag(cmdline: str, flag: str) -> str | None:
    """Return the value following the last occurrence of `--flag value` in a command line string."""
    if not cmdline:
        return None
    parts = cmdline.split()
    last: str | None = None
    i = 0
    while i < len(parts) - 1:
        if parts[i] == flag:
            last = parts[i + 1]
            i += 2
            continue
        i += 1
    return last


def stats_changed_pixels(report: dict[str, Any]) -> tuple[float, float]:
    frames = report.get("per_frame") or []
    vals: list[float] = []
    for f in frames:
        v = f.get("changed_pixels_pct")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return float("nan"), float("nan")
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mu, sd


def mario_pixel_base(peaks: int, bootstrap: int, flow: int, thr_k: float) -> list[str]:
    """Mario preset aligned with run_rd_suite.common_variant_args, with swept pixel knobs."""
    return mario_pixel_args(peaks=peaks, bootstrap=bootstrap, flow=flow, thr_k=thr_k)


def enc_args(
    anchor_mbps: float,
    *,
    open_gop_defaults: bool,
) -> list[str]:
    maxr = anchor_mbps
    buf = 2.0 * anchor_mbps
    out = [
        "--enc-codec",
        "libx264",
        "--enc-bitrate-mbps",
        str(anchor_mbps),
        "--enc-maxrate-mbps",
        str(maxr),
        "--enc-bufsize-mbits",
        str(buf),
        "--enc-gop",
        "60",
        "--enc-preset",
        "medium",
        "--enc-tune",
        "none",
        "--enc-scenecut",
        "0",
        "--enc-aud",
        "1",
        "--enc-repeat-headers",
        "1",
    ]
    if open_gop_defaults:
        out.append("--enc-open-gop-defaults")
    return out


@dataclass(frozen=True)
class SweepConfig:
    peaks: int
    bootstrap: int
    flow: int
    thr_k: float
    open_gop: bool

    def slug(self) -> str:
        og = "og1" if self.open_gop else "og0"
        return f"p{self.peaks}_b{self.bootstrap}_f{self.flow}_k{self.thr_k:.2f}_{og}"


def config_grid() -> list[SweepConfig]:
    peaks_l = [60, 90, 120]
    boot_l = [15, 30]
    flow_l = [0, 1]
    thr_l = [0.80, 0.85, 0.90]
    out: list[SweepConfig] = []
    for p, b, f, k, og in product(peaks_l, boot_l, flow_l, thr_l, (False, True)):
        out.append(SweepConfig(p, b, f, k, og))
    return out


def ensure_pure_baseline(
    *,
    clip_path: Path,
    cache_dir: Path,
    anchor_mbps: float,
    open_gop: bool,
) -> Path:
    """Return path to pure_streaming original_output.mp4 (cached per clip and open-gop mode)."""
    tag = "og1" if open_gop else "og0"
    d = cache_dir / f"{clip_path.stem}_{tag}"
    target = d / "original_output.mp4"
    if target.exists() and (d / "report.json").exists():
        return target
    ensure_dir(d)
    cmd = [
        python_bin(),
        str(ENCODE_PURE),
        "--input",
        str(clip_path),
        "--out-dir",
        str(d),
        "--bitrate-mbps",
        str(anchor_mbps),
        "--enc-maxrate-mbps",
        str(anchor_mbps),
        "--enc-bufsize-mbits",
        str(2.0 * anchor_mbps),
        "--enc-gop",
        "60",
        "--enc-preset",
        "medium",
        "--enc-tune",
        "none",
        "--enc-scenecut",
        "0",
        "--enc-aud",
        "1",
        "--enc-repeat-headers",
        "1",
    ]
    if open_gop:
        cmd.append("--enc-open-gop-defaults")
    subprocess.run(cmd, check=True)
    return target


def run_respawn(
    *,
    clip_path: Path,
    out_dir: Path,
    cfg: SweepConfig,
    anchor_mbps: float,
    use_cuda: bool,
    max_frames: int,
) -> None:
    ensure_dir(out_dir)
    cmd = [str(OFFLINE_BIN)]
    if use_cuda:
        cmd.append("-d")
    cmd += [
        "-i",
        str(clip_path),
        "-o",
        str(out_dir),
        "-m",
        str(DEFAULT_MODEL),
    ]
    cmd += mario_pixel_base(cfg.peaks, cfg.bootstrap, cfg.flow, cfg.thr_k)
    cmd += enc_args(anchor_mbps, open_gop_defaults=cfg.open_gop)
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    subprocess.run(cmd, check=True)


def clip_path_for(clip_id: str) -> Path:
    clips = [c for c in load_manifest(MANIFEST) if c.eligible_rd]
    clip = next(c for c in clips if c.clip_id == clip_id)
    return normalize_clip(clip)


def verify_duplicate_flag_overrides() -> None:
    """Plan todo: duplicate --pixel-max-peaks appears; last wins in argv (stored in report commandline)."""
    tmp_root = RESPAWN2026_DIR / "pixel_enc_sweep" / "_verify_flag_override"
    ensure_dir(tmp_root)
    clip_path = clip_path_for("mario_00")
    out_dir = tmp_root / "runA"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    cmd = [str(OFFLINE_BIN), "-i", str(clip_path), "-o", str(out_dir), "-m", str(DEFAULT_MODEL)]
    cmd += mario_pixel_base(60, 30, 1, 0.85)
    cmd += ["--pixel-max-peaks", "111", "--pixel-max-peaks", "222"]
    cmd += enc_args(16.0, open_gop_defaults=False)
    cmd += ["--max-frames", "8"]
    subprocess.run(cmd, check=True)
    report = json.loads((out_dir / "report.json").read_text())
    cli = report.get("commandline", "")
    peaks = last_cli_flag(cli, "--pixel-max-peaks")
    if peaks != "222":
        raise SystemExit(f"verify-flag-override FAILED: expected last --pixel-max-peaks 222, got {peaks!r} in:\n{cli}")
    print("[verify-flag-override] OK: last --pixel-max-peaks =", peaks)


def sweep_stage0_csv_path(out_root: Path) -> Path:
    return out_root / "leaderboard_stage0.csv"


def run_stage0(
    *,
    out_root: Path,
    anchor_mbps: float,
    use_cuda: bool,
    limit: int,
    max_frames: int,
    pure_cache: Path,
) -> list[dict[str, Any]]:
    ensure_dir(out_root)
    clip_path = clip_path_for("mario_00")
    rows: list[dict[str, Any]] = []
    cfgs = config_grid()
    if limit > 0:
        cfgs = cfgs[:limit]

    # Pure baselines for savings (two enc modes)
    pure_deliv: dict[bool, float] = {}
    for og in (False, True):
        pv = ensure_pure_baseline(
            clip_path=clip_path, cache_dir=pure_cache, anchor_mbps=anchor_mbps, open_gop=og
        )
        d = pv.parent
        pure_deliv[og], _, _ = delivered_bps(d / "segmented_output.mp4", d / "msk1_payloads.bin")

    for cfg in cfgs:
        slug = cfg.slug()
        rd = out_root / slug / "mario_00" / "respawn"
        if not ((rd / "report.json").exists() and (rd / "segmented_output.mp4").exists()):
            run_respawn(
                clip_path=clip_path,
                out_dir=rd,
                cfg=cfg,
                anchor_mbps=anchor_mbps,
                use_cuda=use_cuda,
                max_frames=max_frames,
            )
        report = json.loads((rd / "report.json").read_text())
        mu, sd = stats_changed_pixels(report)
        tot, vb, mb = delivered_bps(rd / "segmented_output.mp4", rd / "msk1_payloads.bin")
        pure = pure_deliv[cfg.open_gop]
        savings = (1.0 - tot / pure) * 100.0 if pure > 1e-6 else float("nan")
        rows.append(
            {
                "cfg_id": slug,
                "clip_id": "mario_00",
                "peaks": cfg.peaks,
                "bootstrap": cfg.bootstrap,
                "flow": cfg.flow,
                "thr_k": cfg.thr_k,
                "open_gop": int(cfg.open_gop),
                "enc_preset": "medium",
                "enc_tune": "none",
                "mean_changed_pixels_pct": mu,
                "std_changed_pixels_pct": sd,
                "delivered_bps": tot,
                "video_bps": vb,
                "meta_bps": mb,
                "pure_delivered_bps": pure,
                "saving_pct_vs_pure": savings,
                "respawn_dir": str(rd),
            }
        )
        print(f"[stage0] {slug} cov={mu:.3f} save={savings:.2f}%", flush=True)

    with sweep_stage0_csv_path(out_root).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    return rows


def run_vmaf_on_rows(
    rows: list[dict[str, Any]],
    *,
    source_clip: Path,
    vmaf_threads: int,
    topk: int,
    vmaf_min: float,
    vmaf_mode: str,
) -> list[dict[str, Any]]:
    # rank by coverage (desc), take topk
    ranked = sorted(
        [r for r in rows if not math.isnan(float(r.get("mean_changed_pixels_pct", float("nan"))))],
        key=lambda r: float(r["mean_changed_pixels_pct"]),
        reverse=True,
    )[:topk]
    enriched: list[dict[str, Any]] = []
    for r in ranked:
        rd = Path(r["respawn_dir"])
        ref_v, dist_v, msk = vmaf_paths(rd, source_clip=source_clip, mode=vmaf_mode)
        if not ref_v.exists() or not dist_v.exists():
            raise SystemExit(f"VMAF inputs missing under {rd}: need {ref_v.name}, {dist_v.name}")
        m = compute_video_metrics(
            ref_video=ref_v,
            dist_video=dist_v,
            msk1_bin=msk,
            threads=vmaf_threads,
            scale_height=1080,
        )
        vmaf = m["full_frame"].get("vmaf_mean")
        r2 = dict(r)
        r2["vmaf_mode"] = vmaf_mode
        r2["vmaf_mean"] = vmaf
        r2["ssim_mean"] = m["full_frame"].get("ssim_mean")
        r2["psnr_mean"] = m["full_frame"].get("psnr_mean")
        r2["passes_vmaf_min"] = int(vmaf is not None and float(vmaf) >= vmaf_min)
        enriched.append(r2)
    return enriched


def run_stage2(
    winners: list[dict[str, Any]],
    *,
    out_root: Path,
    anchor_mbps: float,
    use_cuda: bool,
    max_frames: int,
    pure_cache: Path,
    vmaf_threads: int,
    vmaf_min: float,
    vmaf_mode: str,
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    for clip_id in MARIO_CLIP_IDS:
        clip_path = clip_path_for(clip_id)
        for w in winners:
            cfg = SweepConfig(
                peaks=int(w["peaks"]),
                bootstrap=int(w["bootstrap"]),
                flow=int(w["flow"]),
                thr_k=float(w["thr_k"]),
                open_gop=bool(int(w["open_gop"])),
            )
            slug = cfg.slug() + f"_{clip_id}"
            rd = out_root / "stage2" / slug / "respawn"
            if not ((rd / "report.json").exists() and (rd / "recovered_output.mp4").exists()):
                run_respawn(
                    clip_path=clip_path,
                    out_dir=rd,
                    cfg=cfg,
                    anchor_mbps=anchor_mbps,
                    use_cuda=use_cuda,
                    max_frames=max_frames,
                )
            pv = ensure_pure_baseline(
                clip_path=clip_path, cache_dir=pure_cache, anchor_mbps=anchor_mbps, open_gop=cfg.open_gop
            )
            pure, _, _ = delivered_bps(
                pv.parent / "segmented_output.mp4", pv.parent / "msk1_payloads.bin"
            )
            report = json.loads((rd / "report.json").read_text())
            mu, sd = stats_changed_pixels(report)
            tot, vb, mb = delivered_bps(rd / "segmented_output.mp4", rd / "msk1_payloads.bin")
            savings = (1.0 - tot / pure) * 100.0 if pure > 1e-6 else float("nan")
            ref_v, dist_v, msk = vmaf_paths(rd, source_clip=clip_path, mode=vmaf_mode)
            if not ref_v.exists() or not dist_v.exists():
                raise SystemExit(f"VMAF inputs missing under {rd}: need {ref_v.name}, {dist_v.name}")
            m = compute_video_metrics(
                ref_video=ref_v,
                dist_video=dist_v,
                msk1_bin=msk,
                threads=vmaf_threads,
                scale_height=1080,
            )
            vmaf = m["full_frame"].get("vmaf_mean")
            final_rows.append(
                {
                    "stage": "2",
                    "cfg_id": cfg.slug(),
                    "clip_id": clip_id,
                    "vmaf_mode": vmaf_mode,
                    "peaks": cfg.peaks,
                    "bootstrap": cfg.bootstrap,
                    "flow": cfg.flow,
                    "thr_k": cfg.thr_k,
                    "open_gop": int(cfg.open_gop),
                    "mean_changed_pixels_pct": mu,
                    "std_changed_pixels_pct": sd,
                    "delivered_bps": tot,
                    "saving_pct_vs_pure": savings,
                    "vmaf_mean": vmaf,
                    "ssim_mean": m["full_frame"].get("ssim_mean"),
                    "psnr_mean": m["full_frame"].get("psnr_mean"),
                    "passes_vmaf_min": int(vmaf is not None and float(vmaf) >= vmaf_min),
                    "respawn_dir": str(rd),
                }
            )
    outp = out_root / "leaderboard_stage2.csv"
    if final_rows:
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            w.writeheader()
            w.writerows(final_rows)
        shutil.copy2(outp, out_root / "leaderboard.csv")
    return final_rows


def write_summary(
    out_root: Path,
    *,
    vmaf_min: float,
    anchor_mbps: float,
    vmaf_mode: str,
    stage0: list[dict[str, Any]],
    stage1: list[dict[str, Any]],
    stage2: list[dict[str, Any]],
) -> None:
    vmaf_blurb = {
        VMAF_SEGMENTED_VS_ORIGINAL: "`segmented_output.mp4` vs same-run `original_output.mp4` (masking effect at matched encode)",
        VMAF_RECOVERED_VS_SOURCE: "normalized source vs `recovered_output.mp4`",
    }.get(vmaf_mode, vmaf_mode)
    lines = [
        "# Pixel + encoder sweep summary (Mario)",
        "",
        f"- Anchor bitrate: {anchor_mbps} Mbps (maxrate=buf matched to Exp1 style)",
        f"- Encoder: `--enc-preset medium --enc-tune none`",
        f"- VMAF floor used for gating: {vmaf_min}",
        f"- VMAF (`--vmaf-mode {vmaf_mode}`): {vmaf_blurb}",
        "",
        "- **Delivered bitrate (bandwidth):** `ffprobe` bitrate of `segmented_output.mp4` plus `msk1_payloads.bin` bits/frame. "
        "**Pure baseline** uses `encode_pure_streaming_baseline.py`: copies the full encode to `segmented_output.mp4` (same bytes as `original_output.mp4`) with an empty `msk1_payloads.bin`. "
        "**Savings vs pure:** `(1 - delivered_respawn / delivered_pure) × 100%`.",
        "",
        "## Stage 0 (mario_00, all configs)",
        f"- CSV: `{sweep_stage0_csv_path(out_root)}`",
        "",
        "## Stage 1 (VMAF on top-K by coverage)",
        "- See `leaderboard_stage1.csv`",
        "",
        "## Stage 2 (winner configs × 5 clips)",
        "- See `leaderboard_stage2.csv` (also copied to `leaderboard.csv`).",
        "",
        "## Recommended winners",
    ]
    passing = [r for r in stage1 if int(r.get("passes_vmaf_min", 0))]
    if passing:
        best = max(
            passing,
            key=lambda r: (
                float(r.get("mean_changed_pixels_pct", 0.0)),
                float(r.get("saving_pct_vs_pure", 0.0)),
            ),
        )
        lines += [
            f"- **cfg_id:** `{best['cfg_id']}`",
            f"- mean changed_pixels_pct: {float(best['mean_changed_pixels_pct']):.3f}",
            f"- saving % vs pure: {float(best['saving_pct_vs_pure']):.2f}",
            f"- vmaf_mean: {best.get('vmaf_mean')}",
            "",
            "Respawn extra-style flags (pixel deltas vs baseline 60/30/1/0.85):",
            f"  --pixel-max-peaks {best['peaks']} --pixel-bootstrap {best['bootstrap']} "
            f"--pixel-flow {best['flow']} --pixel-thr-k {best['thr_k']}",
            "Encoder:",
            "  --enc-preset medium --enc-tune none "
            + ("--enc-open-gop-defaults" if int(best["open_gop"]) else "(locked GOP 60)"),
        ]
    else:
        lines.append("- No config passed VMAF min in stage1 top-K; inspect CSVs or relax floor.")
        if stage1:
            probe = max(
                stage1,
                key=lambda r: (
                    float(r.get("mean_changed_pixels_pct", 0.0)),
                    float(r.get("saving_pct_vs_pure", 0.0)),
                ),
            )
            lines += [
                "",
                "## Best-effort leader (Stage1 top-K, **does not** meet VMAF floor)",
                f"- **cfg_id:** `{probe['cfg_id']}`",
                f"- mean changed_pixels_pct: {float(probe['mean_changed_pixels_pct']):.3f}",
                f"- saving % vs pure: {float(probe['saving_pct_vs_pure']):.2f}",
                f"- vmaf_mean: {probe.get('vmaf_mean')} (VMAF_mean ≥ {vmaf_min} required)",
            ]
    if stage2:
        lines += ["", "## Stage 2 (cross-clip; worst clip per cfg)"]
        by_cfg: dict[str, list[dict[str, Any]]] = {}
        for r in stage2:
            cid = str(r.get("cfg_id", ""))
            by_cfg.setdefault(cid, []).append(r)
        for cfg_id in sorted(by_cfg.keys()):
            rows_c = by_cfg[cfg_id]
            vmafs: list[float] = []
            saves: list[float] = []
            for row in rows_c:
                vm = row.get("vmaf_mean")
                try:
                    if vm is not None and str(vm).strip():
                        vmafs.append(float(vm))
                except (TypeError, ValueError):
                    pass
                try:
                    saves.append(float(row["saving_pct_vs_pure"]))
                except (KeyError, TypeError, ValueError):
                    pass
            parts = []
            if vmafs:
                parts.append(f"worst vmaf_mean={min(vmafs):.3f}")
            if saves:
                parts.append(f"worst saving %={min(saves):.2f}")
            lines.append(
                f"- `{cfg_id}`: {', '.join(parts)} ({len(rows_c)} clips)"
                if parts
                else f"- `{cfg_id}` ({len(rows_c)} clips)"
            )
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=RESPAWN2026_DIR / "pixel_enc_sweep",
        help="Sweep output directory",
    )
    ap.add_argument("--anchor-mbps", type=float, default=DEFAULT_ANCHOR_MBPS)
    ap.add_argument("--vmaf-min", type=float, default=DEFAULT_VMAF_MIN)
    ap.add_argument("--vmaf-topk", type=int, default=DEFAULT_TOPK_VMAF)
    ap.add_argument("--threads", type=int, default=max(1, min(32, (os.cpu_count() or 8))))
    ap.add_argument("--limit", type=int, default=0, help="Limit stage0 configs (0=all 72)")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, pass --max-frames to offline binary (quick test)",
    )
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument(
        "--vmaf-mode",
        choices=VMAF_MODES,
        default=VMAF_SEGMENTED_VS_ORIGINAL,
        help="VMAF reference/distorted pair: segmented_vs_original (default) or recovered_vs_source.",
    )
    ap.add_argument(
        "--phase",
        choices=("verify", "stage0", "stage1", "stage2", "all"),
        default="all",
    )
    ap.add_argument(
        "--stage2-winners",
        type=int,
        default=3,
        help="Number of stage1 rows (after sort) to expand in stage2",
    )
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    pure_cache = out_root / "_pure_cache"
    ensure_dir(out_root)

    if args.phase == "verify":
        verify_duplicate_flag_overrides()
        return

    use_cuda = not args.no_cuda
    clip_path_ref = clip_path_for("mario_00")

    if args.phase in ("stage0", "all"):
        stage0 = run_stage0(
            out_root=out_root,
            anchor_mbps=args.anchor_mbps,
            use_cuda=use_cuda,
            limit=args.limit,
            max_frames=args.max_frames,
            pure_cache=pure_cache,
        )
    else:
        p = sweep_stage0_csv_path(out_root)
        stage0 = list(csv.DictReader(p.open())) if p.exists() else []

    if args.phase in ("stage1", "all"):
        stage0_objs = []
        if args.phase == "all" or stage0:
            p = sweep_stage0_csv_path(out_root)
            if not p.exists():
                raise SystemExit("Missing leaderboard_stage0.csv; run phase stage0")
            stage0_objs = list(csv.DictReader(p.open()))
        # numeric cast
        for r in stage0_objs:
            for k in ("peaks", "bootstrap", "flow", "open_gop"):
                r[k] = int(float(r[k]))
            r["thr_k"] = float(r["thr_k"])
            r["mean_changed_pixels_pct"] = float(r["mean_changed_pixels_pct"])
            r["saving_pct_vs_pure"] = float(r["saving_pct_vs_pure"])
        stage1 = run_vmaf_on_rows(
            stage0_objs,
            source_clip=clip_path_ref,
            vmaf_threads=args.threads,
            topk=args.vmaf_topk,
            vmaf_min=args.vmaf_min,
            vmaf_mode=args.vmaf_mode,
        )
        outp = out_root / "leaderboard_stage1.csv"
        if stage1:
            with outp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(stage1[0].keys()))
                w.writeheader()
                w.writerows(stage1)
    else:
        stage1 = []

    if args.phase in ("stage2", "all"):
        p1 = out_root / "leaderboard_stage1.csv"
        if not p1.exists():
            raise SystemExit("Missing leaderboard_stage1.csv")
        stage1_rows = list(csv.DictReader(p1.open()))
        # sort coverage desc then savings
        for r in stage1_rows:
            r["mean_changed_pixels_pct"] = float(r["mean_changed_pixels_pct"])
            r["saving_pct_vs_pure"] = float(r["saving_pct_vs_pure"])
        stage1_rows.sort(key=lambda r: (r["mean_changed_pixels_pct"], r["saving_pct_vs_pure"]), reverse=True)
        winners = []
        seen: set[str] = set()
        for r in stage1_rows:
            if int(r.get("passes_vmaf_min", 0)) != 1:
                continue
            cid = r["cfg_id"]
            if cid in seen:
                continue
            seen.add(cid)
            winners.append(r)
            if len(winners) >= args.stage2_winners:
                break
        if not winners:
            winners = stage1_rows[: args.stage2_winners]
        stage2 = run_stage2(
            winners,
            out_root=out_root,
            anchor_mbps=args.anchor_mbps,
            use_cuda=use_cuda,
            max_frames=args.max_frames,
            pure_cache=pure_cache,
            vmaf_threads=args.threads,
            vmaf_min=args.vmaf_min,
            vmaf_mode=args.vmaf_mode,
        )
    else:
        stage2 = []

    if args.phase in ("all", "stage2"):
        p0 = sweep_stage0_csv_path(out_root)
        stage0_final = list(csv.DictReader(p0.open())) if p0.exists() else []
        p1 = out_root / "leaderboard_stage1.csv"
        stage1_final = list(csv.DictReader(p1.open())) if p1.exists() else []
        write_summary(
            out_root,
            vmaf_min=args.vmaf_min,
            anchor_mbps=args.anchor_mbps,
            vmaf_mode=args.vmaf_mode,
            stage0=stage0_final,
            stage1=stage1_final,
            stage2=stage2,
        )


if __name__ == "__main__":
    main()
