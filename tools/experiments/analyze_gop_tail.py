#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from common import RESPAWN2026_DIR, ensure_dir


@dataclass(frozen=True)
class GopRecord:
    game: str
    clip_id: str
    rate_point_mbps: float
    run_dir: Path
    baseline_run_dir: Path
    gop_id: int
    local_gop_index: int
    start_idx: int
    stop_idx: int
    baseline_bytes: float
    respawn_video_bytes: float
    msk1_bytes: float
    gop_bsp_pct: float
    frame_savings_pct: list[float]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def len_prefixed_sizes(path: Path) -> list[int]:
    if not path.exists():
        return []
    data = path.read_bytes()
    sizes: list[int] = []
    off = 0
    while off + 4 <= len(data):
        n = int.from_bytes(data[off : off + 4], "little")
        off += 4
        if off + n > len(data):
            break
        sizes.append(4 + n)
        off += n
    return sizes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_respawn_rows(primary_rd_points: Path, extra_rd_points: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    primary_games: set[str] = set()
    for row in read_csv(primary_rd_points):
        rows.append(row)
        if row.get("variant") == "respawn":
            primary_games.add(str(row.get("game", "")))
    for extra in extra_rd_points:
        for row in read_csv(extra):
            if str(row.get("game", "")) not in primary_games:
                rows.append(row)
    return [row for row in rows if row.get("variant") == "respawn"]


def collect_gops(rows: list[dict[str, str]], gop_size: int) -> list[GopRecord]:
    records: list[GopRecord] = []
    next_gop_id: dict[str, int] = {}
    for row in rows:
        game = str(row.get("game", "unknown"))
        clip_id = str(row["clip_id"])
        run_dir = Path(row["run_dir"])
        baseline_run_dir = Path(row.get("baseline_run_dir", ""))
        per_csv = run_dir / "per_frame_metrics.csv"
        baseline_csv = baseline_run_dir / "per_frame_metrics.csv"
        if not per_csv.exists() or not baseline_csv.exists() or not (run_dir / "segmented_output.mp4").exists():
            continue

        respawn_rows = read_csv(per_csv)
        baseline_rows = read_csv(baseline_csv)
        msk1_sizes = len_prefixed_sizes(run_dir / "msk1_payloads.bin")
        n = min(len(respawn_rows), len(baseline_rows))

        for start in range(0, n, gop_size):
            stop = min(n, start + gop_size)
            baseline_bytes = 0.0
            respawn_video_bytes = 0.0
            msk1_bytes = 0.0
            frame_savings: list[float] = []
            for idx in range(start, stop):
                base = safe_float(baseline_rows[idx].get("masked_bytes", 0.0))
                resp = safe_float(respawn_rows[idx].get("masked_bytes", 0.0))
                meta = float(msk1_sizes[idx]) if idx < len(msk1_sizes) else 0.0
                baseline_bytes += base
                respawn_video_bytes += resp
                msk1_bytes += meta
                frame_savings.append((1.0 - ((resp + meta) / base)) * 100.0 if base > 1e-9 else 0.0)
            if baseline_bytes <= 1e-9:
                continue
            gop_bsp = (1.0 - ((respawn_video_bytes + msk1_bytes) / baseline_bytes)) * 100.0
            gop_id = next_gop_id.get(game, 0)
            next_gop_id[game] = gop_id + 1
            records.append(
                GopRecord(
                    game=game,
                    clip_id=clip_id,
                    rate_point_mbps=safe_float(row.get("rate_point_mbps", 0.0)),
                    run_dir=run_dir,
                    baseline_run_dir=baseline_run_dir,
                    gop_id=gop_id,
                    local_gop_index=start // gop_size,
                    start_idx=start,
                    stop_idx=stop,
                    baseline_bytes=baseline_bytes,
                    respawn_video_bytes=respawn_video_bytes,
                    msk1_bytes=msk1_bytes,
                    gop_bsp_pct=gop_bsp,
                    frame_savings_pct=frame_savings,
                )
            )
    return records


def select_gops(records: list[GopRecord]) -> list[tuple[str, GopRecord]]:
    selected: list[tuple[str, GopRecord]] = []
    by_game: dict[str, list[GopRecord]] = {}
    for record in records:
        by_game.setdefault(record.game, []).append(record)

    for game in sorted(by_game):
        game_records = sorted(by_game[game], key=lambda item: item.gop_bsp_pct)
        if not game_records:
            continue
        most_negative = game_records[0]
        selected.append(("most_negative", most_negative))

        negative_candidates = [
            item
            for item in game_records
            if item.gop_bsp_pct < 0.0
            and (item.clip_id, item.rate_point_mbps, item.local_gop_index) != (most_negative.clip_id, most_negative.rate_point_mbps, most_negative.local_gop_index)
        ]
        different_clip = [item for item in negative_candidates if item.clip_id != most_negative.clip_id]
        if different_clip:
            selected.append(("negative", different_clip[0]))
        elif negative_candidates:
            selected.append(("negative", negative_candidates[0]))

        positive_candidates = [item for item in by_game[game] if item.gop_bsp_pct > 0.0]
        best_positive = max(positive_candidates, key=lambda item: item.gop_bsp_pct) if positive_candidates else max(by_game[game], key=lambda item: item.gop_bsp_pct)
        selected.append(("best_positive", best_positive))
    return selected


def dump_gop_frames(selection_label: str, record: GopRecord, out_dir: Path) -> int:
    original_path = record.run_dir / "original_output.mp4"
    video_path = record.run_dir / "segmented_output.mp4"
    if not original_path.exists() or not video_path.exists():
        return 0
    group_dir = ensure_dir(out_dir / record.game / selection_label)
    original_dir = ensure_dir(group_dir / "original")
    masked_dir = ensure_dir(group_dir / "masked")
    orig_cap = cv2.VideoCapture(str(original_path))
    cap = cv2.VideoCapture(str(video_path))
    if not orig_cap.isOpened() or not cap.isOpened():
        orig_cap.release()
        cap.release()
        return 0
    orig_cap.set(cv2.CAP_PROP_POS_FRAMES, record.start_idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, record.start_idx)
    written = 0
    frame_table: list[dict[str, Any]] = []
    for offset, frame_idx in enumerate(range(record.start_idx, record.stop_idx)):
        ok_orig, orig_frame = orig_cap.read()
        ok_seg, seg_frame = cap.read()
        if not ok_orig or not ok_seg:
            break
        if orig_frame.shape[:2] != seg_frame.shape[:2]:
            orig_frame = cv2.resize(orig_frame, (seg_frame.shape[1], seg_frame.shape[0]), interpolation=cv2.INTER_AREA)
        frame_id = frame_idx + 1
        frame_saving = record.frame_savings_pct[offset] if offset < len(record.frame_savings_pct) else 0.0
        filename = (
            f"gop{record.gop_id}_frame{frame_id}_"
            f"framesaving_{frame_saving:.2f}_gopsaving_{record.gop_bsp_pct:.2f}.png"
        )
        safe_filename = sanitize(filename)
        cv2.imwrite(str(original_dir / safe_filename), orig_frame)
        cv2.imwrite(str(masked_dir / safe_filename), seg_frame)
        frame_table.append(
            {
                "frame_id": frame_id,
                "frame_saving_pct": f"{frame_saving:.2f}",
                "gop_saving_pct": f"{record.gop_bsp_pct:.2f}",
                "original_png": str(Path("original") / safe_filename),
                "masked_png": str(Path("masked") / safe_filename),
            }
        )
        written += 1
    orig_cap.release()
    cap.release()
    write_frame_table(group_dir, record, frame_table)
    return written


def write_frame_table(group_dir: Path, record: GopRecord, frame_table: list[dict[str, Any]]) -> None:
    fields = ["frame_id", "frame_saving_pct", "gop_saving_pct", "original_png", "masked_png"]
    with (group_dir / "per_frame_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_table)

    lines = [
        "# Per-Frame GOP Savings",
        "",
        f"- Game: `{record.game}`",
        f"- Clip: `{record.clip_id}`",
        f"- Rate: `{record.rate_point_mbps:.1f} Mbps`",
        f"- GOP: `{record.gop_id}` (local `{record.local_gop_index}`)",
        f"- GOP saving: `{record.gop_bsp_pct:.2f}%`",
        "",
        "| Frame | Frame saving (%) | GOP saving (%) | Original | Masked |",
        "|---:|---:|---:|---|---|",
    ]
    for row in frame_table:
        lines.append(
            f"| {row['frame_id']} | {row['frame_saving_pct']} | {row['gop_saving_pct']} | "
            f"`{row['original_png']}` | `{row['masked_png']}` |"
        )
    (group_dir / "per_frame_table.md").write_text("\n".join(lines) + "\n")


def write_summary(selected: list[tuple[str, GopRecord]], frame_counts: dict[tuple[str, int], int], out_dir: Path) -> None:
    csv_path = out_dir / "gop_summary.csv"
    fields = [
        "selection",
        "game",
        "clip_id",
        "rate_point_mbps",
        "gop_id",
        "local_gop_index",
        "frame_start",
        "frame_stop",
        "baseline_bytes",
        "respawn_video_bytes",
        "msk1_bytes",
        "respawn_total_bytes",
        "gop_bsp_pct",
        "frames_dumped",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for selection, record in selected:
            writer.writerow(
                {
                    "selection": selection,
                    "game": record.game,
                    "clip_id": record.clip_id,
                    "rate_point_mbps": f"{record.rate_point_mbps:.1f}",
                    "gop_id": record.gop_id,
                    "local_gop_index": record.local_gop_index,
                    "frame_start": record.start_idx + 1,
                    "frame_stop": record.stop_idx,
                    "baseline_bytes": f"{record.baseline_bytes:.0f}",
                    "respawn_video_bytes": f"{record.respawn_video_bytes:.0f}",
                    "msk1_bytes": f"{record.msk1_bytes:.0f}",
                    "respawn_total_bytes": f"{record.respawn_video_bytes + record.msk1_bytes:.0f}",
                    "gop_bsp_pct": f"{record.gop_bsp_pct:.2f}",
                    "frames_dumped": frame_counts.get((record.game, record.gop_id), 0),
                    "run_dir": str(record.run_dir),
                }
            )

    md_lines = [
        "# GOP Tail Analysis",
        "",
        "GOP BSP is computed as `1 - (RESPAWN video bytes + MSK1 bytes) / baseline video bytes` over 60-frame bins.",
        "",
        "| Selection | Game | Clip | Rate (Mbps) | GOP ID | Local GOP | Frames | Baseline bytes | RESPAWN+MSK1 bytes | GOP BSP (%) | Dumped frames |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for selection, record in selected:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    selection,
                    record.game,
                    record.clip_id,
                    f"{record.rate_point_mbps:.1f}",
                    str(record.gop_id),
                    str(record.local_gop_index),
                    f"{record.start_idx + 1}-{record.stop_idx}",
                    f"{record.baseline_bytes:.0f}",
                    f"{record.respawn_video_bytes + record.msk1_bytes:.0f}",
                    f"{record.gop_bsp_pct:.2f}",
                    str(frame_counts.get((record.game, record.gop_id), 0)),
                ]
            )
            + " |"
        )
    (out_dir / "gop_summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump segmented frames for GOPs that explain the per-GOP savings CDF tails.")
    ap.add_argument("--rd-points", type=Path, default=RESPAWN2026_DIR / "exp1_tuned_upper_bound_shortclips" / "rd_suite_tuned_upper_bound_shortclips_points.csv")
    ap.add_argument("--extra-rd-points", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=RESPAWN2026_DIR / "gop_analysis")
    ap.add_argument("--gop-size", type=int, default=60)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    for stale in args.out_dir.glob("*/*/original/*.png"):
        stale.unlink()
    for stale in args.out_dir.glob("*/*/masked/*.png"):
        stale.unlink()
    for stale in args.out_dir.glob("*/*/*.png"):
        stale.unlink()
    for stale in args.out_dir.glob("*/*.png"):
        stale.unlink()
    rows = load_respawn_rows(args.rd_points, args.extra_rd_points)
    records = collect_gops(rows, max(1, int(args.gop_size)))
    selected = select_gops(records)
    frame_counts: dict[tuple[str, int], int] = {}
    for selection, record in selected:
        frame_counts[(record.game, record.gop_id)] = dump_gop_frames(selection, record, args.out_dir)
    write_summary(selected, frame_counts, args.out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "gops_considered": len(records),
                "gops_selected": len(selected),
                "summary_csv": str(args.out_dir / "gop_summary.csv"),
                "summary_md": str(args.out_dir / "gop_summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
