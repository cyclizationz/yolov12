#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _pick(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return default


def _load_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_per_frame(run_dir: Path) -> list[dict[str, Any]]:
    csv_path = run_dir / "per_frame_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            return list(csv.DictReader(f))
    report = _load_report(run_dir)
    per = report.get("per_frame", []) or []
    return per if isinstance(per, list) else []


def _transition_count(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return sum(1 for a, b in zip(values, values[1:]) if a != b)


def _run_lengths(values: list[int]) -> list[int]:
    if not values:
        return []
    runs: list[int] = []
    last = values[0]
    n = 1
    for v in values[1:]:
        if v == last:
            n += 1
        else:
            runs.append(n)
            last = v
            n = 1
    runs.append(n)
    return runs


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    m = _mean(values)
    assert m is not None
    return math.sqrt(sum((v - m) * (v - m) for v in values) / len(values))


def summarize(run_dir: Path, *, label: str, max_frames: int) -> dict[str, Any]:
    report = _load_report(run_dir)
    rows = _load_per_frame(run_dir)
    if max_frames > 0:
        rows = rows[:max_frames]

    masked = [
        1 if _to_float(_pick(r, "object_successfully_masked"), 0.0) > 0 else 0
        for r in rows
    ]
    flags = [
        1 if _to_int(_pick(r, "frame_flags"), 0) > 0 else 0
        for r in rows
        if "frame_flags" in r
    ]
    frame_state = flags if flags else masked

    changed = [
        _to_float(
            _pick(
                r,
                "changed_pixels_pct",
                "changed_pixels_pct (changed_pixels / frame area)",
            )
        )
        for r in rows
    ]
    object_counts = [_to_float(_pick(r, "object_present_model")) for r in rows]
    minted = [_to_float(_pick(r, "latent_minted_regions")) for r in rows]
    reused = [_to_float(_pick(r, "latent_reused_regions")) for r in rows]
    rec_ssim = [_to_float(_pick(r, "rec_ssim"), float("nan")) for r in rows if _pick(r, "rec_ssim", default="") != ""]
    rec_psnr = [_to_float(_pick(r, "rec_psnr"), float("nan")) for r in rows if _pick(r, "rec_psnr", default="") != ""]
    template_ms = [
        _to_float(_pick(r, "template_matching_ms", "pixel_roi_ms"))
        for r in rows
    ]

    runs = _run_lengths(frame_state)
    masked_runs = [n for n, v in zip(runs, _run_values(frame_state)) if v == 1]
    raw_runs = [n for n, v in zip(runs, _run_values(frame_state)) if v == 0]
    reused_sum = sum(reused)
    minted_sum = sum(minted)

    hyst = report.get("frame_mode_hysteresis", {}) if isinstance(report, dict) else {}
    post_hyst = (hyst.get("post_smoothing", {}) or {}) if isinstance(hyst, dict) else {}
    pre_hyst = (hyst.get("pre_smoothing", {}) or {}) if isinstance(hyst, dict) else {}

    return {
        "label": label,
        "run_dir": str(run_dir),
        "frames": len(rows),
        "avg_recovered_ssim_report": report.get("avg_recovered_ssim", ""),
        "avg_recovered_psnr_report": report.get("avg_recovered_psnr", ""),
        "masked_fraction": _mean([float(v) for v in masked]),
        "state_switch_count": _transition_count(frame_state),
        "state_switch_fraction": (_transition_count(frame_state) / max(1, len(frame_state) - 1)) if frame_state else None,
        "mean_state_run": _mean([float(v) for v in runs]),
        "max_masked_run": max(masked_runs) if masked_runs else 0,
        "max_raw_run": max(raw_runs) if raw_runs else 0,
        "changed_pixels_pct_mean": _mean(changed),
        "changed_pixels_pct_std": _std(changed),
        "changed_pixels_pct_jumps_gt_1pp": sum(1 for a, b in zip(changed, changed[1:]) if abs(b - a) > 1.0),
        "object_count_mean": _mean(object_counts),
        "object_count_std": _std(object_counts),
        "latent_minted_sum": minted_sum,
        "latent_reused_sum": reused_sum,
        "latent_reuse_ratio": reused_sum / (reused_sum + minted_sum) if (reused_sum + minted_sum) > 0 else 0.0,
        "rec_ssim_mean_frame": _mean(rec_ssim),
        "rec_ssim_min_frame": min(rec_ssim) if rec_ssim else None,
        "rec_psnr_mean_frame": _mean(rec_psnr),
        "rec_psnr_min_frame": min(rec_psnr) if rec_psnr else None,
        "template_matching_ms_p95": _percentile(template_ms, 0.95),
        "pre_hyst_switch_fraction": pre_hyst.get("fraction_mode_switches", ""),
        "post_hyst_switch_fraction": post_hyst.get("fraction_mode_switches", ""),
        "post_hyst_mean_run": post_hyst.get("mean_run_length_ref_raw_states", ""),
        "commandline": report.get("commandline", ""),
    }


def _run_values(values: list[int]) -> list[int]:
    if not values:
        return []
    out = [values[0]]
    for a, b in zip(values, values[1:]):
        if a != b:
            out.append(b)
    return out


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return None
    idx = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize 5-second RESPAWN stability metrics from run directories.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    summaries = [
        summarize(path, label=(args.labels[i] if i < len(args.labels) else path.name), max_frames=args.max_frames)
        for i, path in enumerate(args.run_dirs)
    ]

    print(json.dumps(summaries, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(summaries[0].keys()) if summaries else ["label"]
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
