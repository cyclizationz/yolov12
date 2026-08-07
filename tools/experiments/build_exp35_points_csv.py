#!/usr/bin/env python3
"""Assemble Exp3/Exp5 intake CSV from CRF23/open-GOP exploratory runs (not Exp1 VBV)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import REPO_ROOT, RESPAWN2026_DIR, ensure_dir, load_manifest, python_bin

CRF_RATE_LABEL = 23.0


@dataclass(frozen=True)
class IntakeSpec:
    clip_id: str
    game: str
    rate_point_mbps: float
    respawn_dir: Path
    pure_dir: Path
    encoder_note: str


def ffprobe_duration_s(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out or 0.0)


def bitrate_bps(video: Path) -> float:
    dur = ffprobe_duration_s(video)
    return (video.stat().st_size * 8.0) / dur if dur > 1e-9 else 0.0


def delivered_bps(run_dir: Path) -> tuple[float, float, float]:
    video = run_dir / "segmented_output.mp4"
    meta = run_dir / "msk1_payloads.bin"
    video_bps = bitrate_bps(video) if video.exists() else 0.0
    meta_bps = 0.0
    if meta.exists() and video.exists():
        dur = ffprobe_duration_s(video)
        if dur > 1e-9:
            meta_bps = (meta.stat().st_size * 8.0) / dur
    return video_bps, meta_bps, video_bps + meta_bps


def run_build_per_frame(out_dir: Path) -> None:
    if not (out_dir / "report.json").exists():
        raise FileNotFoundError(f"missing report.json in {out_dir}")
    subprocess.run(
        [
            python_bin(),
            str(REPO_ROOT / "tools/metrics/build_per_frame_csv.py"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def ensure_pure_from_original(respawn_dir: Path, pure_dir: Path) -> None:
    """Pure baseline = original encode only (no masking, empty MSK1)."""
    ensure_dir(pure_dir)
    orig = respawn_dir / "original_output.mp4"
    if not orig.exists():
        raise FileNotFoundError(f"missing {orig}")
    for name in ("original_output.mp4", "segmented_output.mp4", "recovered_output.mp4"):
        dst = pure_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(orig.resolve(), dst)
    msk1 = pure_dir / "msk1_payloads.bin"
    msk1.write_bytes(b"")
    rep_src = respawn_dir / "report.json"
    rep = json.loads(rep_src.read_text())
    rep["pure_streaming_baseline"] = True
    (pure_dir / "report.json").write_text(json.dumps(rep), encoding="utf-8")


def ensure_per_frame(respawn_dir: Path, pure_dir: Path) -> None:
    if not (respawn_dir / "per_frame_metrics.csv").exists():
        run_build_per_frame(respawn_dir)
    ensure_pure_from_original(respawn_dir, pure_dir)
    if not (pure_dir / "per_frame_metrics.csv").exists():
        run_build_per_frame(pure_dir)


def fc5_specs() -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"fc5_{i:02d}"
        respawn = RESPAWN2026_DIR / "gop_analysis" / f"{clip_id}_offline_open_x264_crf23"
        pure = RESPAWN2026_DIR / "exp35_crf" / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="fc5",
                rate_point_mbps=CRF_RATE_LABEL,
                respawn_dir=respawn,
                pure_dir=pure,
                encoder_note="CRF23 open x264 (gop_analysis)",
            )
        )
    return out


def mario_specs() -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"mario_{i:02d}"
        respawn = RESPAWN2026_DIR / "gop_analysis" / f"{clip_id}_offline_open_x264_crf23"
        pure = RESPAWN2026_DIR / "exp35_crf" / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="mario",
                rate_point_mbps=CRF_RATE_LABEL,
                respawn_dir=respawn,
                pure_dir=pure,
                encoder_note="CRF23 open x264 pixel pipeline (gop_analysis)",
            )
        )
    return out


def fm6_specs(
    *,
    respawn_root: Path = RESPAWN2026_DIR / "gop_analysis",
    pure_root: Path = RESPAWN2026_DIR / "exp35_crf",
) -> list[IntakeSpec]:
    out: list[IntakeSpec] = []
    for i in range(5):
        clip_id = f"fm6_{i:02d}"
        respawn = respawn_root / f"{clip_id}_offline_open_x264_crf23"
        pure = pure_root / clip_id / "crf23/pure_streaming"
        out.append(
            IntakeSpec(
                clip_id=clip_id,
                game="fm6",
                rate_point_mbps=CRF_RATE_LABEL,
                respawn_dir=respawn,
                pure_dir=pure,
                encoder_note="CRF23 open x264 (gop_analysis)",
            )
        )
    return out


def row_for(
    spec: IntakeSpec,
    *,
    variant: str,
    pure_bps: tuple[float, float, float],
    respawn_bps: tuple[float, float, float],
    run_dir: Path,
    pure_dir: Path,
    respawn_dir: Path,
) -> dict[str, Any]:
    pb_v, pb_m, pb_t = pure_bps
    rs_v, rs_m, rs_t = respawn_bps
    if variant == "pure_streaming":
        vo, mo, tot = pb_v, pb_m, pb_t
    else:
        vo, mo, tot = rs_v, rs_m, rs_t
    return {
        "clip_id": spec.clip_id,
        "game": spec.game,
        "rate_point_mbps": spec.rate_point_mbps,
        "variant": variant,
        "achieved_bps": tot,
        "achieved_mbps": tot / 1_000_000.0,
        "video_only_bps": vo,
        "meta_bps": mo,
        "vmaf_mean": "",
        "vmaf_p10": "",
        "ssim_mean": "",
        "psnr_mean": "",
        "roi_vmaf_mean": "",
        "roi_ssim_mean": "",
        "roi_psnr_mean": "",
        "roi_frame_count": "",
        "baseline_achieved_bps": pb_t,
        "respawn_achieved_bps": rs_t,
        "baseline_vmaf_mean": "",
        "respawn_vmaf_mean": "",
        "baseline_ssim_mean": "",
        "respawn_ssim_mean": "",
        "baseline_psnr_mean": "",
        "respawn_psnr_mean": "",
        "baseline_run_dir": str(pure_dir),
        "respawn_run_dir": str(respawn_dir),
        "stitch_full_vmaf_mean": "",
        "stitch_full_ssim_mean": "",
        "stitch_full_psnr_mean": "",
        "stitch_roi_vmaf_mean": "",
        "stitch_roi_ssim_mean": "",
        "stitch_roi_psnr_mean": "",
        "frame_mode_hysteresis_pre_mean_run_length": "",
        "frame_mode_hysteresis_post_mean_run_length": "",
        "frame_mode_hysteresis_pre_switch_fraction": "",
        "frame_mode_hysteresis_post_switch_fraction": "",
        "run_dir": str(run_dir),
        "encoder_note": spec.encoder_note,
    }


def build_specs(
    *,
    include_fm6: bool,
    skip_fc5: bool,
    skip_mario: bool,
    fm6_respawn_root: Path,
    fm6_pure_root: Path,
) -> list[IntakeSpec]:
    specs: list[IntakeSpec] = []
    if not skip_fc5:
        specs.extend(fc5_specs())
    if not skip_mario:
        specs.extend(mario_specs())
    if include_fm6:
        specs.extend(
            fm6_specs(
                respawn_root=fm6_respawn_root,
                pure_root=fm6_pure_root,
            )
        )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Exp3/Exp5 CRF exploratory intake CSV.")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "exp35_crf")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--include-fm6", action="store_true", help="Include FM6 gop_analysis CRF23 dirs.")
    ap.add_argument(
        "--fm6-respawn-root",
        type=Path,
        default=RESPAWN2026_DIR / "gop_analysis",
    )
    ap.add_argument("--fm6-pure-root", type=Path, default=None)
    ap.add_argument("--skip-fc5", action="store_true")
    ap.add_argument("--skip-mario", action="store_true")
    ap.add_argument("--require-complete", action="store_true", help="Fail if any respawn dir is missing.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_csv = args.out_csv or (args.out_dir / "exp35_points.csv")
    manifest = {c.clip_id: c for c in load_manifest(RESPAWN2026_DIR / "manifest/offline_manifest.json")}

    rows: list[dict[str, Any]] = []
    for spec in build_specs(
        include_fm6=args.include_fm6,
        skip_fc5=args.skip_fc5,
        skip_mario=args.skip_mario,
        fm6_respawn_root=args.fm6_respawn_root,
        fm6_pure_root=args.fm6_pure_root or args.out_dir,
    ):
        if spec.clip_id not in manifest:
            print(f"[skip] unknown clip {spec.clip_id}", file=sys.stderr)
            continue
        if not spec.respawn_dir.is_dir():
            msg = f"missing respawn dir {spec.respawn_dir}"
            if args.require_complete:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}", file=sys.stderr)
            continue
        if spec.game == "mario" and not spec.pure_dir.is_dir() and not spec.respawn_dir.is_dir():
            msg = f"missing mario respawn dir {spec.respawn_dir}"
            if args.require_complete:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}", file=sys.stderr)
            continue

        if spec.game in ("fc5", "fm6", "mario"):
            ensure_per_frame(spec.respawn_dir, spec.pure_dir)
        else:
            if not (spec.respawn_dir / "per_frame_metrics.csv").exists():
                run_build_per_frame(spec.respawn_dir)
            if not (spec.pure_dir / "per_frame_metrics.csv").exists():
                run_build_per_frame(spec.pure_dir)

        pure_bps = delivered_bps(spec.pure_dir)
        respawn_bps = delivered_bps(spec.respawn_dir)
        rows.append(
            row_for(
                spec,
                variant="pure_streaming",
                pure_bps=pure_bps,
                respawn_bps=respawn_bps,
                run_dir=spec.pure_dir,
                pure_dir=spec.pure_dir,
                respawn_dir=spec.respawn_dir,
            )
        )
        rows.append(
            row_for(
                spec,
                variant="respawn",
                pure_bps=pure_bps,
                respawn_bps=respawn_bps,
                run_dir=spec.respawn_dir,
                pure_dir=spec.pure_dir,
                respawn_dir=spec.respawn_dir,
            )
        )
        print(f"[ok] {spec.clip_id} pure={pure_bps[2]/1e6:.2f} Mbps respawn={respawn_bps[2]/1e6:.2f} Mbps")

    if not rows:
        raise SystemExit("no intake rows written")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    protocol = args.out_dir / "exp35_protocol.md"
    protocol.write_text(
        "\n".join(
            [
                "# Exp3/Exp5 intake protocol (CRF / exploratory)",
                "",
                "- Exp1 RD suite remains fixed VBV under `record/RESPAWN2026/exp1/`.",
                "- This CSV feeds Exp3 overhead and Exp5 generality only.",
                "",
                "## Sources",
                f"- FC5: `gop_analysis/fc5_XX_offline_open_x264_crf23` (CRF 23, open GOP)",
                "- Mario: `gop_analysis/mario_XX_offline_open_x264_crf23` (CRF 23, open GOP, pixel pipeline)",
                "- FM6: `gop_analysis/fm6_XX_offline_open_x264_crf23` (CRF 23, open GOP, heal-only)",
                "",
                f"- Points CSV: `{out_csv}`",
                f"- Rows: {len(rows)} ({len(rows)//2} clip pairs)",
                "",
            ]
        )
        + "\n"
    )
    print(json.dumps({"points_csv": str(out_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
