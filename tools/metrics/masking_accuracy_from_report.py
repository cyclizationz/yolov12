#!/usr/bin/env python3
"""
Compute "masking accuracy" proxies from an offline run folder.

This script is intentionally lightweight and relies only on artifacts we already
produce in deployment runs:
  - report.json (quality + per-frame counters)
  - run.log (encoder + masking knobs echo)

It does NOT require ground truth masks/labels (since we don't have them for the
video traces). Metrics are therefore *proxies* for deployment reliability.

Outputs JSON to stdout (optionally writes to --out).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _parse_config_line(run_log: Path) -> dict[str, Any]:
    """
    Parse the single "[Config] ..." line we print in deployment/main.cpp.
    Returns a dict of key->value (strings).
    """

    if not run_log.exists():
        return {}
    txt = run_log.read_text(errors="replace")
    m = re.search(r"^\\[Config\\]\\s+(.*)$", txt, flags=re.MULTILINE)
    if not m:
        return {}
    rest = m.group(1).strip()
    # Tokens are key=value separated by spaces; values don't contain spaces.
    out: dict[str, Any] = {}
    for tok in rest.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize deployment masking proxies from report.json.")
    ap.add_argument("--run-dir", required=True, type=Path, help="Folder containing report.json and optionally run.log")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    run_dir = args.run_dir
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing {report_path}")

    rep = _read_json(report_path)
    per = rep.get("per_frame", []) or []

    total_frames = int(rep.get("total_frames", len(per)) or len(per))
    total_dets = int(rep.get("total_detections", rep.get("matched_detections", 0)) or 0)
    matched_dets = int(rep.get("matched_detections", 0) or 0)

    # Ref/raw proxy: frame_flags (v4) if present; otherwise nreg>0 in MSK1.
    # report.json contains frame_flags per frame (we store it).
    ref_frames = 0
    raw_frames = 0
    obj_present_sum = 0
    obj_masked_sum = 0
    masked_alpha_pcts: list[float] = []
    changed_pcts: list[float] = []

    for e in per:
        ff = e.get("frame_flags", None)
        if ff is None:
            # fall back: if any region was successfully masked, treat as ref
            if int(e.get("object_successfully_masked", 0) or 0) > 0:
                ref_frames += 1
            else:
                raw_frames += 1
        else:
            if int(ff) == 0:
                raw_frames += 1
            else:
                ref_frames += 1
        obj_present_sum += int(e.get("object_present_model", 0) or 0)
        obj_masked_sum += int(e.get("object_successfully_masked", 0) or 0)
        masked_alpha_pcts.append(float(e.get("masked_alpha_pct", 0.0) or 0.0))
        changed_pcts.append(float(e.get("changed_pixels_pct", 0.0) or 0.0))

    config = _parse_config_line(run_dir / "run.log")

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source": config.get("source", ""),
        "model": config.get("model", ""),  # not currently echoed; kept for forward-compat
        # Reliability / masking proxies
        "total_frames": total_frames,
        "ref_frames": int(ref_frames),
        "raw_frames": int(raw_frames),
        "ref_frame_ratio": (ref_frames / total_frames) if total_frames else 0.0,
        "matched_detections": matched_dets,
        "total_detections": total_dets,
        "match_rate": (matched_dets / total_dets) if total_dets else 0.0,
        "objects_present_total": int(obj_present_sum),
        "objects_masked_total": int(obj_masked_sum),
        "object_masked_over_present": (obj_masked_sum / obj_present_sum) if obj_present_sum else 0.0,
        # Coverage proxies (deployment-time)
        "masked_alpha_pct_mean": _mean(masked_alpha_pcts),
        "changed_pixels_pct_mean": _mean(changed_pcts),
        # Quality (masked vs recovered)
        "avg_ssim_masked": rep.get("avg_ssim"),
        "avg_psnr_masked": rep.get("avg_psnr"),
        "avg_ssim_recovered": rep.get("avg_recovered_ssim"),
        "avg_psnr_recovered": rep.get("avg_recovered_psnr"),
        # Useful context for the paper table
        "latent_bank_size": rep.get("latent_bank_size", None),
        "latent_reuse_ratio": rep.get("latent_reuse_ratio", None),
        "config": config,
    }

    s = json.dumps(out, indent=2)
    print(s)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s)


if __name__ == "__main__":
    main()

