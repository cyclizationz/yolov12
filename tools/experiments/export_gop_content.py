#!/usr/bin/env python3
"""Export complete GOP videos, every-frame contact sheets, and byte traces."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from msk1 import read_len_prefixed_payloads


REPO = Path(__file__).resolve().parents[2]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as src:
        return list(csv.DictReader(src))


def value(row: dict[str, str], prefix: str, default: float = 0.0) -> float:
    key = prefix if prefix in row else next((name for name in row if name.startswith(prefix)), "")
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def payload_sizes(path: Path) -> list[int]:
    return [4 + len(payload) for payload in read_len_prefixed_payloads(path)]


def load_gops(path: Path, clip_id: str, ids: set[int]) -> list[dict[str, str]]:
    return [
        row
        for row in csv_rows(path)
        if row["clip_id"] == clip_id and int(row["gop_index"]) in ids
    ]


def export_gop_frames(
    video: Path,
    *,
    clip_id: str,
    gop_id: int,
    start: int,
    stop: int,
    out_dir: Path,
    page_size: int = 50,
) -> tuple[Path, list[Path]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read frame {start}")
    height, width = first.shape[:2]
    video_out = out_dir / f"{clip_id}_gop{gop_id}_full_content.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frames = [first]
    writer.write(first)
    for _ in range(start + 1, stop):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        writer.write(frame)
    writer.release()
    cap.release()

    pages: list[Path] = []
    cols, rows_per_page = 10, 5
    thumb_w, thumb_h = 256, 144
    for page_idx, page_start in enumerate(range(0, len(frames), page_size)):
        subset = frames[page_start : page_start + page_size]
        canvas = np.full((rows_per_page * thumb_h, cols * thumb_w, 3), 238, dtype=np.uint8)
        for local_idx, frame in enumerate(subset):
            row_idx, col_idx = divmod(local_idx, cols)
            thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            absolute_idx = start + page_start + local_idx
            cv2.rectangle(thumb, (0, 0), (92, 20), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                f"f{absolute_idx}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0, x0 = row_idx * thumb_h, col_idx * thumb_w
            canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
        page_path = out_dir / f"{clip_id}_gop{gop_id}_all_frames_page{page_idx + 1}.png"
        cv2.imwrite(str(page_path), canvas)
        pages.append(page_path)
    return video_out, pages


def plot_diagnostics(
    *,
    clip_id: str,
    gops: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    respawn_rows: list[dict[str, str]],
    rmd_sizes: list[int],
    out_dir: Path,
) -> Path:
    ordered_gop_ids = [
        int(row["gop_index"])
        for row in sorted(gops, key=lambda item: int(item["gop_index"]))
    ]
    gop_label = "_".join(str(gop_id) for gop_id in ordered_gop_ids)
    readable_gops = ", ".join(str(gop_id) for gop_id in ordered_gop_ids)
    start = min(int(row["start_frame"]) for row in gops)
    stop = max(int(row["end_frame_exclusive"]) for row in gops)
    frames = np.arange(start, stop)
    base = np.asarray([value(baseline_rows[idx], "masked_bytes") for idx in frames])
    resp_video = np.asarray([value(respawn_rows[idx], "masked_bytes") for idx in frames])
    rmd = np.asarray([rmd_sizes[idx] if idx < len(rmd_sizes) else 0 for idx in frames])
    resp_total = resp_video + rmd
    delta = base - resp_total
    mask = np.asarray([value(respawn_rows[idx], "masked_alpha_pct") for idx in frames])
    rec = np.asarray([
        1.0 if value(respawn_rows[idx], "object_successfully_masked") > 0 else 0.0
        for idx in frames
    ])

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True, height_ratios=[2.1, 1.3, 1.0])
    axes[0].plot(frames, base / 1000.0, label="Baseline video", color="black", linewidth=1.1)
    axes[0].plot(frames, resp_total / 1000.0, label="RESPAWN video + RMD", color="tab:blue", linewidth=1.1)
    axes[0].set_ylabel("Bytes/frame (KB)")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    axes[1].bar(frames, delta / 1000.0, color=np.where(delta >= 0, "tab:blue", "tab:red"), width=1.0)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_ylabel("Saved bytes/frame (KB)")
    axes[1].grid(axis="y", alpha=.2)
    axes[2].plot(frames, mask, color="tab:purple", label="Mask coverage (%)")
    axes[2].fill_between(frames, 0, rec * max(1.0, float(mask.max())), color="tab:green", alpha=.18, label="Rec frame")
    axes[2].set_ylabel("Mask (%) / Rec")
    axes[2].set_xlabel("Absolute frame index")
    axes[2].legend(loc="upper left", ncol=2)
    axes[2].grid(alpha=.2)
    for row in sorted(gops, key=lambda item: int(item["gop_index"])):
        boundary = int(row["start_frame"])
        for ax in axes:
            ax.axvline(boundary, color="goldenrod", linestyle="--", linewidth=1.2)
        axes[0].text(
            boundary + 2,
            axes[0].get_ylim()[1] * .92,
            f"GOP {row['gop_index']}",
            fontsize=9,
            color="darkgoldenrod",
        )
    fig.suptitle(f"{clip_id}: per-frame byte behavior across complete GOP(s) {readable_gops}")
    fig.tight_layout()
    out = out_dir / f"{clip_id}_gop{gop_label}_per_frame_diagnostics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="fm6_03")
    ap.add_argument("--gop-ids", nargs="+", type=int, default=[27, 28])
    ap.add_argument("--baseline-video", required=True, type=Path)
    ap.add_argument("--baseline-csv", required=True, type=Path)
    ap.add_argument("--respawn-csv", required=True, type=Path)
    ap.add_argument("--msk1", required=True, type=Path)
    ap.add_argument(
        "--gop-csv",
        type=Path,
        default=REPO / "record" / "RESPAWN2026" / "evaluation_revision"
        / "generated_eval_tables" / "per_gop_net_bsp.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "record" / "RESPAWN2026" / "evaluation_revision"
        / "generated_eval_figures" / "gop_content",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gops = load_gops(args.gop_csv, args.clip_id, set(args.gop_ids))
    if len(gops) != len(set(args.gop_ids)):
        raise SystemExit("One or more requested GOP IDs were not found")

    outputs: list[str] = []
    for row in sorted(gops, key=lambda item: int(item["gop_index"])):
        video, pages = export_gop_frames(
            args.baseline_video,
            clip_id=args.clip_id,
            gop_id=int(row["gop_index"]),
            start=int(row["start_frame"]),
            stop=int(row["end_frame_exclusive"]),
            out_dir=args.out_dir,
        )
        outputs.extend([str(video), *(str(page) for page in pages)])
    diagnostic = plot_diagnostics(
        clip_id=args.clip_id,
        gops=gops,
        baseline_rows=csv_rows(args.baseline_csv),
        respawn_rows=csv_rows(args.respawn_csv),
        rmd_sizes=payload_sizes(args.msk1),
        out_dir=args.out_dir,
    )
    outputs.append(str(diagnostic))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
