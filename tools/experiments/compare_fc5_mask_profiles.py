#!/usr/bin/env python3
"""Compare FC5 mask profiles and write decision.md for Exp1 refresh."""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, RESPAWN2026_DIR, python_bin
from pixel_mario_defaults import FC5_MASK_GLOBAL, FC5_MASK_LOCAL_PAD64, FC5_MASK_PROFILES


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def savings_pct(rows: list[dict[str, str]], *, clip_id: str, rate: float) -> float | None:
    base = next(
        (
            r
            for r in rows
            if r["clip_id"] == clip_id
            and r["variant"] == "pure_streaming"
            and abs(float(r["rate_point_mbps"]) - rate) < 1e-6
        ),
        None,
    )
    resp = next(
        (
            r
            for r in rows
            if r["clip_id"] == clip_id
            and r["variant"] == "respawn"
            and abs(float(r["rate_point_mbps"]) - rate) < 1e-6
        ),
        None,
    )
    if not base or not resp:
        return None
    b = float(base["achieved_bps"] or 0)
    r = float(resp["achieved_bps"] or 0)
    if b <= 1e-6:
        return None
    return (1.0 - r / b) * 100.0


def profile_stats(rows: list[dict[str, str]], *, rates: list[float]) -> dict[str, float | None]:
    clip_ids = sorted({r["clip_id"] for r in rows if r["game"] == "fc5"})
    vals: list[float] = []
    vmafs: list[float] = []
    for clip_id in clip_ids:
        for rate in rates:
            s = savings_pct(rows, clip_id=clip_id, rate=rate)
            if s is not None:
                vals.append(s)
            resp = next(
                (
                    r
                    for r in rows
                    if r["clip_id"] == clip_id
                    and r["variant"] == "respawn"
                    and abs(float(r["rate_point_mbps"]) - rate) < 1e-6
                ),
                None,
            )
            if resp and resp.get("vmaf_mean") not in (None, ""):
                vmafs.append(float(resp["vmaf_mean"]))
    return {
        "mean_saving_pct": statistics.mean(vals) if vals else None,
        "mean_vmaf": statistics.mean(vmafs) if vmafs else None,
        "n_points": float(len(vals)),
    }


def write_decision(out_root: Path, *, winner: str, stats: dict[str, dict[str, float | None]]) -> None:
    a, b = FC5_MASK_GLOBAL, FC5_MASK_LOCAL_PAD64
    lines = [
        "# FC5 mask profile A/B decision",
        "",
        "Compared on FC5 shortclips (`fc5_00`–`fc5_04`) at 16 and 20 Mbps.",
        "",
        "## Results",
        "",
        f"| Profile | Mean saving % (16+20 Mbps) | Mean respawn VMAF | Points |",
        f"| --- | ---: | ---: | ---: |",
    ]
    for profile in FC5_MASK_PROFILES:
        st = stats[profile]
        ms = st["mean_saving_pct"]
        mv = st["mean_vmaf"]
        ms_txt = f"{ms:.2f}%" if ms is not None else ""
        mv_txt = f"{mv:.2f}" if mv is not None else ""
        lines.append(f"| `{profile}` | {ms_txt} | {mv_txt} | {int(st['n_points'])} |")
    lines += [
        "",
        f"## Selected profile: `{winner}`",
        "",
        "Selection rule: higher mean delivered savings at 16–20 Mbps; tie-break higher respawn VMAF.",
        "",
    ]
    (out_root / "decision.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=RESPAWN2026_DIR / "exp1_fc5_mask_ab",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-video-metrics", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rd = REPO_ROOT / "tools" / "experiments" / "run_rd_suite.py"
    clip_ids = ["fc5_00", "fc5_01", "fc5_02", "fc5_03", "fc5_04"]
    stats: dict[str, dict[str, float | None]] = {}

    for profile in FC5_MASK_PROFILES:
        prof_dir = out_root / profile
        cmd = [
            python_bin(),
            str(rd),
            "--out-dir",
            str(prof_dir),
            "--summary-stem",
            f"fc5_ab_{profile}",
            "--clip-ids",
            *clip_ids,
            "--fc5-mask-profile",
            profile,
            "--force-respawn",
            "--force-pure",
            "--threads",
            str(args.threads),
        ]
        if args.skip_video_metrics:
            cmd.append("--skip-video-metrics")
        print("[fc5-ab]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        csv_path = prof_dir / f"fc5_ab_{profile}_points.csv"
        stats[profile] = profile_stats(load_rows(csv_path), rates=[16.0, 20.0])

    a_stats = stats[FC5_MASK_GLOBAL]
    b_stats = stats[FC5_MASK_LOCAL_PAD64]
    winner = FC5_MASK_GLOBAL
    if (b_stats["mean_saving_pct"] or float("-inf")) > (a_stats["mean_saving_pct"] or float("-inf")):
        winner = FC5_MASK_LOCAL_PAD64
    elif (b_stats["mean_saving_pct"] or 0) == (a_stats["mean_saving_pct"] or 0):
        if (b_stats["mean_vmaf"] or 0) > (a_stats["mean_vmaf"] or 0):
            winner = FC5_MASK_LOCAL_PAD64

    write_decision(out_root, winner=winner, stats=stats)
    print(f"[fc5-ab] winner={winner}", flush=True)


if __name__ == "__main__":
    main()
