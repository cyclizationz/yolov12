#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from common import RESPAWN2026_DIR, ensure_dir, infer_template_bytes, load_manifest, seconds_to_frames
from msk1 import load_payloads


@dataclass
class CacheEntry:
    size_bytes: int
    last_used_frame: int


@dataclass(frozen=True)
class SyntheticTemplateConfig:
    enabled: bool
    template_size_bytes: int
    reuse_prob: float
    cache_hit_prob: float
    loss_prob: float
    bbox_quant: int
    seed: int


def load_template_sizes(run_dir: Path, report: dict[str, Any]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    outputs = report.get("outputs", {}) or {}
    dict_dir = Path(outputs.get("dict_dir", run_dir / "dict"))
    if dict_dir.exists():
        for path in dict_dir.iterdir():
            if path.is_file():
                sizes[path.name] = path.stat().st_size
                sizes[str(path)] = path.stat().st_size
    latent_path = report.get("latent_bank_path", None)
    if latent_path:
        latent_bank = json.loads(Path(latent_path).read_text())
        for item in latent_bank:
            path = item.get("path", "")
            if path and Path(path).exists():
                sizes[path] = Path(path).stat().st_size
                sizes[Path(path).name] = Path(path).stat().st_size
    return sizes


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def synthetic_template_id(clip_id: str, frame_idx: int, region_idx: int, region: Any, config: SyntheticTemplateConfig, rng: random.Random) -> str:
    quant = max(1, int(config.bbox_quant))
    qx = int(region.x) // quant
    qy = int(region.y) // quant
    qw = max(1, int(region.w) // quant)
    qh = max(1, int(region.h) // quant)
    base = f"syn:{clip_id}:c{int(region.class_id)}:{qx}:{qy}:{qw}:{qh}"
    if rng.random() <= config.reuse_prob:
        return base
    return f"{base}:mint:{frame_idx}:{region_idx}"


def active_template_ids(
    *,
    clip_id: str,
    frame_idx: int,
    payload: Any,
    config: SyntheticTemplateConfig | None,
    rng: random.Random,
) -> list[str]:
    if config is not None and config.enabled:
        ids = [
            synthetic_template_id(clip_id, frame_idx, region_idx, region, config, rng)
            for region_idx, region in enumerate(payload.regions)
        ]
    else:
        ids = [region.path for region in payload.regions if region.path]

    # Preserve order while avoiding duplicate charges for repeated regions in a frame.
    return list(dict.fromkeys(tpl for tpl in ids if tpl))


def simulate_cache(
    *,
    clip_id: str,
    payloads: list[Any],
    template_sizes: dict[str, int],
    baseline_bytes: list[int],
    respawn_bytes: list[int],
    msk1_bytes_per_frame: list[int],
    fps: float,
    delay_rtts: int,
    rtt_ms: float,
    cache_budget_bytes: int | None,
    warm_start: bool,
    synthetic_config: SyntheticTemplateConfig | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    rng = rng or random.Random(0)
    cache: OrderedDict[str, CacheEntry] = OrderedDict()
    cache_footprint = 0
    if warm_start:
        for path, size in sorted(template_sizes.items()):
            cache[path] = CacheEntry(size_bytes=size, last_used_frame=-1)
            cache_footprint += int(size)
    delivery_due: dict[str, int] = {}
    template_bytes_sent = 0
    forced_raw = 0
    first_sighting_frame: int | None = None
    first_ref_eligible_frame: int | None = None
    net_saved_cumulative: list[float] = []
    cumulative_baseline = 0.0
    cumulative_respawn = 0.0

    delay_frames = seconds_to_frames((delay_rtts * rtt_ms) / 1000.0, fps)
    max_cache_footprint = 0
    synthetic_template_ids_seen: set[str] = set()
    forced_raw_penalty_bytes = 0.0

    for frame_idx, payload in enumerate(payloads):
        baseline_frame_bytes = float(baseline_bytes[frame_idx] if frame_idx < len(baseline_bytes) else 0)
        respawn_frame_bytes = float(respawn_bytes[frame_idx] if frame_idx < len(respawn_bytes) else 0)
        cumulative_baseline += baseline_frame_bytes
        cumulative_respawn += respawn_frame_bytes
        cumulative_respawn += float(msk1_bytes_per_frame[frame_idx] if frame_idx < len(msk1_bytes_per_frame) else 0)

        active_templates = active_template_ids(clip_id=clip_id, frame_idx=frame_idx, payload=payload, config=synthetic_config, rng=rng)
        if active_templates and first_sighting_frame is None:
            first_sighting_frame = frame_idx

        frame_has_ref = False
        frame_forced_raw = False
        for tpl in active_templates:
            synthetic_template_ids_seen.add(tpl)
            size = template_sizes.get(tpl) or template_sizes.get(Path(tpl).name, 0)
            if synthetic_config is not None and synthetic_config.enabled:
                size = int(synthetic_config.template_size_bytes)
            if tpl in cache:
                if synthetic_config is not None and synthetic_config.enabled and rng.random() > synthetic_config.cache_hit_prob:
                    dropped = cache.pop(tpl, None)
                    if dropped is not None:
                        cache_footprint -= dropped.size_bytes
                else:
                    entry = cache.pop(tpl)
                    entry.last_used_frame = frame_idx
                    cache[tpl] = entry
                    frame_has_ref = True
                    continue

            if synthetic_config is not None and synthetic_config.enabled and warm_start and rng.random() <= synthetic_config.cache_hit_prob:
                cache[tpl] = CacheEntry(size_bytes=int(size), last_used_frame=frame_idx)
                cache_footprint += int(size)
                frame_has_ref = True
                continue

            if tpl in cache:
                entry = cache.pop(tpl)
                entry.last_used_frame = frame_idx
                cache[tpl] = entry
                frame_has_ref = True
                continue

            if tpl not in delivery_due:
                delivery_delay = delay_frames
                if synthetic_config is not None and synthetic_config.enabled and rng.random() < synthetic_config.loss_prob:
                    delivery_delay += max(1, delay_frames)
                delivery_due[tpl] = frame_idx + delivery_delay
                template_bytes_sent += int(size)
                cumulative_respawn += float(size)

            if frame_idx >= delivery_due[tpl]:
                cache[tpl] = CacheEntry(size_bytes=int(size), last_used_frame=frame_idx)
                cache_footprint += int(size)
                frame_has_ref = True
                delivery_due.pop(tpl, None)
            else:
                frame_forced_raw = True

        if frame_forced_raw:
            forced_raw += 1
            if synthetic_config is not None and synthetic_config.enabled:
                penalty = max(0.0, baseline_frame_bytes - respawn_frame_bytes)
                forced_raw_penalty_bytes += penalty
                cumulative_respawn += penalty

        if frame_has_ref and not frame_forced_raw and first_ref_eligible_frame is None:
            first_ref_eligible_frame = frame_idx

        if cache_budget_bytes is not None and cache_budget_bytes > 0:
            while cache_footprint > cache_budget_bytes and cache:
                _, evicted = cache.popitem(last=False)
                cache_footprint -= evicted.size_bytes
        max_cache_footprint = max(max_cache_footprint, cache_footprint)
        net_saved_cumulative.append(cumulative_baseline - cumulative_respawn)

    return {
        "frames": len(payloads),
        "template_bytes_sent": template_bytes_sent,
        "max_cache_footprint_bytes": max_cache_footprint,
        "forced_raw_frames": forced_raw,
        "forced_raw_fraction": (forced_raw / len(payloads)) if payloads else 0.0,
        "time_to_first_ref_eligible_s": None
        if first_sighting_frame is None or first_ref_eligible_frame is None
        else max(0.0, (first_ref_eligible_frame - first_sighting_frame) / max(1e-9, fps)),
        "net_saved_cumulative_bytes": net_saved_cumulative,
        "break_even_frame": next((i for i, v in enumerate(net_saved_cumulative) if v > 0), None),
        "synthetic_unique_templates": len(synthetic_template_ids_seen) if synthetic_config is not None and synthetic_config.enabled else "",
        "forced_raw_penalty_bytes": forced_raw_penalty_bytes,
    }


def plot_cumulative(run_name: str, series: dict[str, list[float]], out_dir: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, values in series.items():
        mode = "warm_start" if label.startswith("warm_start") else "cold_start" if label.startswith("cold_start") else "partial_cache"
        linestyle = {"cold_start": "-", "warm_start": "--", "partial_cache": ":"}.get(mode, "-")
        marker = "o" if "_rtt20_" in label else "s" if "_rtt60_" in label else "^"
        color = "tab:blue" if "_delay0" in label else "tab:orange" if "_delay1" in label else "tab:green"
        markevery = max(1, len(values) // 12)
        ax.plot(
            range(len(values)),
            values,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            markersize=3.5,
            linewidth=1.6,
            color=color,
            alpha=0.9,
        )
    ax.set_title(f"{run_name} cumulative net bytes saved")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Net bytes saved")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_cumulative_net_bytes.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze template/control overhead for offline RESPAWN runs.")
    ap.add_argument("--rd-dir", type=Path, default=RESPAWN2026_DIR / "exp1")
    ap.add_argument("--rd-points", type=Path, default=RESPAWN2026_DIR / "exp1" / "rd_suite_points.csv")
    ap.add_argument("--manifest", type=Path, default=RESPAWN2026_DIR / "manifest" / "offline_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "exp3")
    ap.add_argument("--rtt-ms", nargs="+", type=float, default=[20.0, 60.0])
    ap.add_argument("--delay-rtts", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cache-mb", nargs="+", type=float, default=[16.0, 32.0, 64.0, 128.0])
    ap.add_argument("--synthetic-template-model", action="store_true", help="Use synthetic template IDs from region boxes when MSK1 paths are missing.")
    ap.add_argument("--synthetic-template-size-bytes", type=int, default=4096, help="Bytes charged when a synthetic template is delivered.")
    ap.add_argument("--synthetic-reuse-prob", type=float, default=0.90, help="Probability that a region reuses its quantized synthetic template ID.")
    ap.add_argument("--synthetic-cache-hit-prob", type=float, default=0.85, help="Probability that a synthetic cached/warm-start template is available.")
    ap.add_argument("--synthetic-loss-prob", type=float, default=0.05, help="Probability that a synthetic template delivery incurs one extra delay window.")
    ap.add_argument("--synthetic-bbox-quant", type=int, default=32, help="Pixel quantization for bbox-derived synthetic template IDs.")
    ap.add_argument("--synthetic-seed", type=int, default=7, help="Seed for deterministic synthetic sensitivity runs.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest = {clip.clip_id: clip for clip in load_manifest(args.manifest)}
    rows: list[dict[str, Any]] = []
    synthetic_config = SyntheticTemplateConfig(
        enabled=bool(args.synthetic_template_model),
        template_size_bytes=max(0, int(args.synthetic_template_size_bytes)),
        reuse_prob=min(1.0, max(0.0, float(args.synthetic_reuse_prob))),
        cache_hit_prob=min(1.0, max(0.0, float(args.synthetic_cache_hit_prob))),
        loss_prob=min(1.0, max(0.0, float(args.synthetic_loss_prob))),
        bbox_quant=max(1, int(args.synthetic_bbox_quant)),
        seed=int(args.synthetic_seed),
    )

    pair_rows: dict[tuple[str, float], dict[str, str]] = {}
    with args.rd_points.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["clip_id"], float(row.get("rate_point_mbps", 0.0) or 0.0))
            pair_rows.setdefault(key, {})
            pair_rows[key][row["variant"]] = row["run_dir"]
    for (clip_id, rate_point), variants in sorted(pair_rows.items()):
        ref_key = "pure_streaming" if "pure_streaming" in variants else "baseline"
        if ref_key not in variants or "respawn" not in variants:
            continue
        run_dir = Path(variants["respawn"])
        baseline_dir = Path(variants[ref_key])
        run_name = f"{clip_id}_{str(rate_point).replace('.', 'p')}"
        report = json.loads((run_dir / "report.json").read_text())
        clip = manifest.get(clip_id)
        if clip is None:
            continue
        payloads = load_payloads(run_dir / "msk1_payloads.bin")
        per_frame = report.get("per_frame", []) or []
        fps = clip.target_fps
        template_sizes = load_template_sizes(run_dir, report)

        baseline_bytes = []
        respawn_bytes = []
        baseline_per_frame_csv = baseline_dir / "per_frame_metrics.csv"
        respawn_per_frame_csv = run_dir / "per_frame_metrics.csv"
        if baseline_per_frame_csv.exists() and respawn_per_frame_csv.exists():
            with baseline_per_frame_csv.open("r", newline="") as fb, respawn_per_frame_csv.open("r", newline="") as fr:
                rb = list(csv.DictReader(fb))
                rr = list(csv.DictReader(fr))
                n = min(len(rb), len(rr))
                for i in range(n):
                    baseline_bytes.append(int(float(rb[i].get("masked_bytes", 0) or 0)))
                    respawn_bytes.append(int(float(rr[i].get("masked_bytes", 0) or 0)))
        msk1_bytes = []
        bin_path = run_dir / "msk1_payloads.bin"
        if bin_path.exists():
            data = bin_path.read_bytes()
            off = 0
            while off + 4 <= len(data):
                n = int.from_bytes(data[off : off + 4], "little")
                off += 4
                msk1_bytes.append(4 + n)
                off += n

        cumulative_series: dict[str, list[float]] = {}
        for warm_start in (False, True):
            mode = "warm_start" if warm_start else "cold_start"
            no_budget = None if warm_start else int(args.cache_mb[-1] * 1024 * 1024)
            for rtt in args.rtt_ms:
                for delay_rtts in args.delay_rtts:
                    result = simulate_cache(
                        clip_id=clip.clip_id,
                        payloads=payloads,
                        template_sizes=template_sizes,
                        baseline_bytes=baseline_bytes,
                        respawn_bytes=respawn_bytes,
                        msk1_bytes_per_frame=msk1_bytes,
                        fps=fps,
                        delay_rtts=delay_rtts,
                        rtt_ms=rtt,
                        cache_budget_bytes=no_budget,
                        warm_start=warm_start,
                        synthetic_config=synthetic_config if synthetic_config.enabled else None,
                        rng=random.Random(stable_seed(synthetic_config.seed, run_name, mode, rtt, delay_rtts)),
                    )
                    label = f"{mode}_rtt{int(rtt)}_delay{delay_rtts}"
                    cumulative_series[label] = result["net_saved_cumulative_bytes"]
                    rows.append(
                        {
                            "run_name": run_name,
                            "clip_id": clip.clip_id,
                            "game": clip.game,
                            "cache_mode": mode,
                            "rtt_ms": rtt,
                            "delay_rtts": delay_rtts,
                            "cache_budget_mb": "" if no_budget is None else (no_budget / (1024.0 * 1024.0)),
                            "template_model": "synthetic" if synthetic_config.enabled else "payload_path",
                            "synthetic_template_size_bytes": synthetic_config.template_size_bytes if synthetic_config.enabled else "",
                            "synthetic_reuse_prob": synthetic_config.reuse_prob if synthetic_config.enabled else "",
                            "synthetic_cache_hit_prob": synthetic_config.cache_hit_prob if synthetic_config.enabled else "",
                            "synthetic_loss_prob": synthetic_config.loss_prob if synthetic_config.enabled else "",
                            "synthetic_bbox_quant": synthetic_config.bbox_quant if synthetic_config.enabled else "",
                            "synthetic_unique_templates": result["synthetic_unique_templates"],
                            "template_bytes_sent": result["template_bytes_sent"],
                            "max_cache_footprint_bytes": result["max_cache_footprint_bytes"],
                            "forced_raw_frames": result["forced_raw_frames"],
                            "forced_raw_fraction": result["forced_raw_fraction"],
                            "forced_raw_penalty_bytes": result["forced_raw_penalty_bytes"],
                            "time_to_first_ref_eligible_s": result["time_to_first_ref_eligible_s"],
                            "break_even_frame": result["break_even_frame"],
                            "final_net_saved_bytes": result["net_saved_cumulative_bytes"][-1] if result["net_saved_cumulative_bytes"] else 0.0,
                            "template_footprint_bytes_total": infer_template_bytes(Path((report.get("outputs", {}) or {}).get("dict_dir", run_dir / "dict"))),
                        }
                    )
            for cache_mb in args.cache_mb:
                result = simulate_cache(
                    clip_id=clip.clip_id,
                    payloads=payloads,
                    template_sizes=template_sizes,
                    baseline_bytes=baseline_bytes,
                    respawn_bytes=respawn_bytes,
                    msk1_bytes_per_frame=msk1_bytes,
                    fps=fps,
                    delay_rtts=1,
                    rtt_ms=20.0,
                    cache_budget_bytes=int(cache_mb * 1024 * 1024),
                    warm_start=False,
                    synthetic_config=synthetic_config if synthetic_config.enabled else None,
                    rng=random.Random(stable_seed(synthetic_config.seed, run_name, "partial_cache", cache_mb)),
                )
                rows.append(
                    {
                        "run_name": run_name,
                        "clip_id": clip.clip_id,
                        "game": clip.game,
                        "cache_mode": "partial_cache",
                        "rtt_ms": 20.0,
                        "delay_rtts": 1,
                        "cache_budget_mb": cache_mb,
                        "template_model": "synthetic" if synthetic_config.enabled else "payload_path",
                        "synthetic_template_size_bytes": synthetic_config.template_size_bytes if synthetic_config.enabled else "",
                        "synthetic_reuse_prob": synthetic_config.reuse_prob if synthetic_config.enabled else "",
                        "synthetic_cache_hit_prob": synthetic_config.cache_hit_prob if synthetic_config.enabled else "",
                        "synthetic_loss_prob": synthetic_config.loss_prob if synthetic_config.enabled else "",
                        "synthetic_bbox_quant": synthetic_config.bbox_quant if synthetic_config.enabled else "",
                        "synthetic_unique_templates": result["synthetic_unique_templates"],
                        "template_bytes_sent": result["template_bytes_sent"],
                        "max_cache_footprint_bytes": result["max_cache_footprint_bytes"],
                        "forced_raw_frames": result["forced_raw_frames"],
                        "forced_raw_fraction": result["forced_raw_fraction"],
                        "forced_raw_penalty_bytes": result["forced_raw_penalty_bytes"],
                        "time_to_first_ref_eligible_s": result["time_to_first_ref_eligible_s"],
                        "break_even_frame": result["break_even_frame"],
                        "final_net_saved_bytes": result["net_saved_cumulative_bytes"][-1] if result["net_saved_cumulative_bytes"] else 0.0,
                        "template_footprint_bytes_total": infer_template_bytes(Path((report.get("outputs", {}) or {}).get("dict_dir", run_dir / "dict"))),
                    }
                )
        plot_cumulative(run_name, cumulative_series, args.out_dir)

    out_csv = args.out_dir / "overhead_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps({"summary_csv": str(out_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
