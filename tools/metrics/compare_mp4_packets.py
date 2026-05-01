import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Packet:
    idx: int
    pts: int | None
    dts: int | None
    pts_time: float | None
    dts_time: float | None
    size: int


def _run_ffprobe_packets(video: Path) -> list[Packet]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts,dts,pts_time,dts_time,size",
        "-of",
        "json",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True)
    j = json.loads(out)
    packets = j.get("packets", []) or []
    res: list[Packet] = []
    for i, p in enumerate(packets):
        pts_i = p.get("pts", None)
        dts_i = p.get("dts", None)
        pts = p.get("pts_time", None)
        dts = p.get("dts_time", None)
        size = int(p.get("size", 0) or 0)
        res.append(
            Packet(
                idx=i,
                pts=int(pts_i) if pts_i is not None else None,
                dts=int(dts_i) if dts_i is not None else None,
                pts_time=float(pts) if pts is not None else None,
                dts_time=float(dts) if dts is not None else None,
                size=size,
            )
        )
    return res


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MP4 video payload sizes using ffprobe packet sizes.")
    ap.add_argument("--orig", required=True, type=Path, help="Baseline/original MP4")
    ap.add_argument("--masked", required=True, type=Path, help="Masked/segmented MP4")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument(
        "--align",
        choices=["index", "time"],
        default="index",
        help="How to align packets across videos. 'index' assumes same packet count/order. 'time' matches by pts_time with tolerance.",
    )
    ap.add_argument(
        "--time-tol-ms",
        type=float,
        default=5.0,
        help="Alignment tolerance (milliseconds) when --align time.",
    )
    args = ap.parse_args()

    p0 = _run_ffprobe_packets(args.orig)
    p1 = _run_ffprobe_packets(args.masked)

    if args.align == "index":
        n = min(len(p0), len(p1))
        pairs = [(p0[i], p1[i]) for i in range(n)]
    else:
        # Align by time (PTS). This is robust to different time bases (e.g., 29.97 vs 30 fps) and B-frame reordering.
        tol = args.time_tol_ms / 1000.0
        a = sorted([p for p in p0 if p.pts_time is not None], key=lambda p: p.pts_time)  # type: ignore[arg-type]
        b = sorted([p for p in p1 if p.pts_time is not None], key=lambda p: p.pts_time)  # type: ignore[arg-type]
        i = 0
        j = 0
        pairs = []
        while i < len(a) and j < len(b):
            t0 = a[i].pts_time  # type: ignore[assignment]
            t1 = b[j].pts_time  # type: ignore[assignment]
            dt = t0 - t1
            if abs(dt) <= tol:
                pairs.append((a[i], b[j]))
                i += 1
                j += 1
            elif dt < 0:
                i += 1
            else:
                j += 1

    orig_bytes = sum(a.size for a, _ in pairs)
    masked_bytes = sum(b.size for _, b in pairs)
    saving_pct = 100.0 * (1.0 - _safe_div(masked_bytes, orig_bytes))

    worse = sum(1 for a, b in pairs if b.size > a.size)
    worse_pct = 100.0 * _safe_div(worse, len(pairs)) if pairs else float("nan")

    report: dict[str, Any] = {
        "align": args.align,
        "pairs": len(pairs),
        "orig_packets": len(p0),
        "masked_packets": len(p1),
        "orig_bytes": orig_bytes,
        "masked_bytes": masked_bytes,
        "saving_pct": saving_pct,
        "worse_pct": worse_pct,
        "orig": str(args.orig),
        "masked": str(args.masked),
    }

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


