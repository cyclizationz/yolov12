#!/usr/bin/env python3
"""
Raise masked-area coverage by sweeping Mario pixel knobs (peaks, optional thr_k / mask-pad / min-score).

Encoder stack matches sweep_pixel_encoder_mario (16 Mbps medium/none og1 unless --no-open-gop).

Note: large `--mask-pads` grows emitted mask boxes (stitching footprint). Prefer `--thr-k-list` /
`--min-scores` or scan geometry if boxes must stay tight to tiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from common import DEFAULT_MODEL, OFFLINE_BIN, REPO_ROOT, RESPAWN2026_DIR, ensure_dir
from sweep_pixel_encoder_mario import clip_path_for, enc_args, mario_pixel_base, stats_changed_pixels


def _replace_argv_value(argv: list[str], flag: str, value: str) -> None:
    for i in range(len(argv) - 1):
        if argv[i] == flag:
            argv[i + 1] = value
            return
    raise KeyError(f"missing flag {flag} in argv")


def mario_argv(
    peaks: int,
    bootstrap: int,
    flow: int,
    thr_k: float,
    *,
    mask_pad: int,
    min_score: float,
) -> list[str]:
    args = mario_pixel_base(peaks, bootstrap, flow, thr_k)
    _replace_argv_value(args, "--pixel-mask-pad", str(mask_pad))
    _replace_argv_value(args, "--pixel-min-score", str(min_score))
    return args


def run_one(
    *,
    peaks: int,
    bootstrap: int,
    flow: int,
    thr_k: float,
    open_gop: bool,
    band_topk: int | None,
    mask_pad: int,
    min_score: float,
    out_respawn: Path,
    use_cuda: bool,
    max_frames: int,
) -> None:
    ensure_dir(out_respawn)
    cmd = [str(OFFLINE_BIN)]
    if use_cuda:
        cmd.append("-d")
    clip = clip_path_for("mario_00")
    cmd += ["-i", str(clip), "-o", str(out_respawn), "-m", str(DEFAULT_MODEL)]
    cmd += mario_argv(peaks, bootstrap, flow, thr_k, mask_pad=mask_pad, min_score=min_score)
    if band_topk is not None:
        cmd += ["--pixel-band-topk", str(band_topk)]
    cmd += enc_args(16.0, open_gop_defaults=open_gop)
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=RESPAWN2026_DIR / "pixel_peak_tune_p90_flow0",
        help="Subdirs created per combo under this root.",
    )
    ap.add_argument("--bootstrap", type=int, default=15)
    ap.add_argument("--flow", type=int, default=0)
    ap.add_argument("--thr-k", type=float, default=0.80)
    ap.add_argument(
        "--thr-k-list",
        type=str,
        default="",
        help="If non-empty (comma-separated), sweep these instead of --thr-k.",
    )
    ap.add_argument("--no-open-gop", action="store_true", help="Omit --enc-open-gop-defaults (default: encoder uses open GOP).")
    ap.add_argument("--peaks-list", type=str, default="90", help="Comma-separated --pixel-max-peaks values.")
    ap.add_argument(
        "--mask-pads",
        type=str,
        default="2",
        help="Comma-separated --pixel-mask-pad (expands emitted mask boxes toward edges).",
    )
    ap.add_argument(
        "--min-scores",
        type=str,
        default="0.65",
        help="Comma-separated --pixel-min-score values.",
    )
    ap.add_argument("--band-topk-values", type=str, default="", help='Optional comma list; omit "" = default CLI (2). Use 0 for all templates.')
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    peaks_vals = [int(x.strip()) for x in args.peaks_list.split(",") if x.strip()]
    mask_pads = [int(x.strip()) for x in args.mask_pads.split(",") if x.strip()]
    min_scores = [float(x.strip()) for x in args.min_scores.split(",") if x.strip()]
    if args.thr_k_list.strip():
        thr_vals = [float(x.strip()) for x in args.thr_k_list.split(",") if x.strip()]
    else:
        thr_vals = [float(args.thr_k)]
    tops: list[int | None]
    if not args.band_topk_values.strip():
        tops = [None]
    else:
        tops = [int(x.strip()) for x in args.band_topk_values.split(",") if x.strip()]

    out_root = args.out_root.resolve()
    ensure_dir(out_root)
    rows: list[dict[str, object]] = []
    use_cuda = not args.no_cuda
    open_gop = not args.no_open_gop

    for peaks in peaks_vals:
        for bt in tops:
            for thr_k in thr_vals:
                for mp in mask_pads:
                    for mq in min_scores:
                        slug_bt = "def" if bt is None else str(bt)
                        slug = (
                            f"p{peaks}_bt{slug_bt}_mp{mp}_mq{mq:.2f}_b{args.bootstrap}_"
                            f"f{args.flow}_k{thr_k:.2f}_{'og1' if open_gop else 'og0'}"
                        )
                        rd = out_root / slug / "respawn"
                        if not ((rd / "report.json").exists() and (rd / "recovered_output.mp4").exists()):
                            run_one(
                                peaks=peaks,
                                bootstrap=args.bootstrap,
                                flow=args.flow,
                                thr_k=thr_k,
                                open_gop=open_gop,
                                band_topk=bt,
                                mask_pad=mp,
                                min_score=mq,
                                out_respawn=rd,
                                use_cuda=use_cuda,
                                max_frames=args.max_frames,
                            )
                        report = json.loads((rd / "report.json").read_text())
                        mu, sd = stats_changed_pixels(report)
                        rows.append(
                            {
                                "slug": slug,
                                "pixel_max_peaks": peaks,
                                "pixel_thr_k": thr_k,
                                "pixel_mask_pad": mp,
                                "pixel_min_score": mq,
                                "pixel_band_topk": "" if bt is None else bt,
                                "mean_changed_pixels_pct": mu,
                                "std_changed_pixels_pct": sd,
                                "respawn_dir": str(rd),
                            }
                        )

    csv_path = out_root / "peak_tune_results.csv"
    if not rows:
        raise SystemExit("nothing to write (empty sweep)")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: float(r["mean_changed_pixels_pct"]))
    print(json.dumps(best, indent=2))
    print("wrote", csv_path, "repo", str(REPO_ROOT))


if __name__ == "__main__":
    main()
