#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from common import DUO_BIN, RESPAWN2026_DIR, classify_duo_viability, ensure_dir, load_manifest, moving_average, safe_name

PIXEL_TEMPLATES = Path("/home/tiehangz/proj/yolov12/deployment/pixel_kinds_templates")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch duo-stream viability screen across the offline manifest.")
    ap.add_argument("--manifest", type=Path, default=RESPAWN2026_DIR / "manifest" / "offline_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "duo_screen")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--extra-args", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clips = [c for c in load_manifest(args.manifest) if c.eligible_duo]
    if args.limit > 0:
        clips = clips[: args.limit]
    ensure_dir(args.out_dir)

    rows: list[dict[str, Any]] = []
    for clip in clips:
        clip_out = args.out_dir / safe_name(clip.clip_id)
        ensure_dir(clip_out)
        cmd = [str(DUO_BIN), "-i", clip.normalized_path, "-o", str(clip_out)]
        if clip.game == "mario":
            cmd += ["--pixel", "-T", str(PIXEL_TEMPLATES)]
        if args.model:
            cmd += ["-m", str(args.model)]
        cmd += args.extra_args
        subprocess.run(cmd, check=True)
        report_path = clip_out / "report.json"
        if not report_path.exists():
            raise SystemExit(f"Missing duo report: {report_path}")
        report = json.loads(report_path.read_text())
        base = report.get("baseline_a", {}) or {}
        mode_stats = base.get("mode_stats", {}) or {}
        variant = report.get("variant_c", {}) or {}
        avg_gap = variant.get("average_temporal_gap_frames", {}) or {}
        mean_run = float(mode_stats.get("mean_run_length_ref_raw_states", 0.0) or 0.0)
        row = {
            "clip_id": clip.clip_id,
            "game": clip.game,
            "source_tag": clip.source_tag,
            "mean_run_length_ref_raw_states": mean_run,
            "fraction_mode_switches": float(mode_stats.get("fraction_mode_switches", 0.0) or 0.0),
            "masked_avg_temporal_gap_frames": float(avg_gap.get("masked_only", 0.0) or 0.0),
            "unmasked_avg_temporal_gap_frames": float(avg_gap.get("unmasked_only", 0.0) or 0.0),
            "baseline_a_total_bytes": int((report.get("comparison", {}) or {}).get("total_bytes", {}).get("baseline_a", 0) or 0),
            "baseline_b_total_bytes": int((report.get("comparison", {}) or {}).get("total_bytes", {}).get("baseline_b", 0) or 0),
            "variant_c_total_bytes": int((report.get("comparison", {}) or {}).get("total_bytes", {}).get("variant_c", 0) or 0),
            "classification": classify_duo_viability(mean_run),
        }
        rows.append(row)

    out_csv = args.out_dir / "duo_screen_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clip_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    game_summary: dict[str, dict[str, Any]] = {}
    for game in sorted({r["game"] for r in rows}):
        rr = [r for r in rows if r["game"] == game]
        means = [float(r["mean_run_length_ref_raw_states"]) for r in rr]
        labels = [str(r["classification"]) for r in rr]
        counts = {label: labels.count(label) for label in sorted(set(labels))}
        game_summary[game] = {
            "clip_count": len(rr),
            "mean_run_length_ref_raw_states": moving_average(means),
            "classification_counts": counts,
        }

    summary = {"summary_csv": str(out_csv), "game_summary": game_summary}
    (args.out_dir / "duo_screen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
