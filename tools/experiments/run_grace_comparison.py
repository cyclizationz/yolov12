#!/usr/bin/env python3
"""Run and reduce a quick RESPAWN × GRACE comparison.

GRACE remains in an external checkout because its academic license restricts
redistribution. This script stages two inputs for the unmodified upstream
``grace-gpu.py`` evaluator:

* ``source.mp4``: the normalized full-frame clip.
* ``respawn_masked.mp4``: RESPAWN's segmented/masked stream.

The upstream evaluator reports modeled entropy size and source-relative
PSNR/SSIM for each input. The reducer adds prorated RESPAWN metadata bytes and
computes a size proxy. It is intentionally not an end-to-end reconstructed-
quality or wire-bandwidth claim because upstream GRACE emits neither an
interoperable bitstream nor decoded videos for RESPAWN stitching.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TypedDict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "record/RESPAWN2026/manifest/normalized/fc5/fc5_00.mp4"
)
DEFAULT_RESPAWN_RUN = (
    REPO_ROOT
    / "record/RESPAWN2026/gop_analysis/fc5_00_offline_open_x264_crf23"
)
DEFAULT_OUTPUT = REPO_ROOT / "record/RESPAWN2026/grace_comparison/fc5_00_quick"
SOURCE_LABEL = "source.mp4"
MASKED_LABEL = "respawn_masked.mp4"
REQUIRED_UPSTREAM_COLUMNS = {
    "video",
    "model_id",
    "loss",
    "nframes",
    "size",
    "psnr",
    "ssim",
}
PUBLISHED_OUTPUTS = ("comparison.csv", "summary.json")


class ComparisonRow(TypedDict):
    model_id: str
    loss: float
    burst_frames: int
    evaluated_frames: int
    source_grace_estimated_bytes: float
    respawn_masked_grace_estimated_bytes: float
    respawn_rmd_prorated_bytes: float
    respawn_delivered_bytes: float
    proxy_bsp_percent: float
    source_codec_psnr_mean: float
    respawn_masked_codec_psnr_mean: float
    source_codec_ssim_mean: float
    respawn_masked_codec_ssim_mean: float


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def ffprobe_frame_count(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        timeout=60,
    )
    stream = json.loads(output)["streams"][0]
    raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    return int(raw)


def stage_upstream_work(
    *,
    grace_root: Path,
    source: Path,
    masked: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, Path]:
    """Create a self-contained working directory using symlinks to GRACE."""
    grace_root = grace_root.expanduser().resolve()
    upstream_script = require_file(grace_root / "grace-gpu.py", "GRACE evaluator")
    for relative in ("grace", "libs", "models"):
        if not (grace_root / relative).exists():
            raise FileNotFoundError(f"GRACE checkout is missing {relative}/: {grace_root}")

    work_dir = output_dir / "upstream_work"
    if work_dir.exists():
        if not force:
            raise FileExistsError(
                f"{work_dir} already exists; pass --force to replace it"
            )
        shutil.rmtree(work_dir)
    inputs = work_dir / "inputs"
    inputs.mkdir(parents=True)

    staged_source = inputs / SOURCE_LABEL
    staged_masked = inputs / MASKED_LABEL
    staged_source.symlink_to(source)
    staged_masked.symlink_to(masked)
    for relative in ("grace", "libs", "models"):
        (work_dir / relative).symlink_to(grace_root / relative, target_is_directory=True)
    (work_dir / "INDEX.txt").write_text(
        f"{staged_source}\n{staged_masked}\n",
        encoding="utf-8",
    )
    return work_dir, upstream_script


def build_upstream_command(
    *, python: str, grace_root: Path, work_dir: Path, upstream_script: Path
) -> list[str]:
    """Build a command that runs upstream without editing its checkout."""
    bootstrap = (
        "import os,runpy,sys;"
        f"sys.path.insert(0,{str(grace_root)!r});"
        f"os.chdir({str(work_dir)!r});"
        f"runpy.run_path({str(upstream_script)!r},run_name='__main__')"
    )
    return [python, "-c", bootstrap]


def run_upstream(
    command: list[str],
    *,
    grace_root: Path,
    work_dir: Path,
    timeout_s: int,
) -> Path:
    env = os.environ.copy()
    lib_dir = str(grace_root / "libs")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{env['LD_LIBRARY_PATH']}"
        if env.get("LD_LIBRARY_PATH")
        else lib_dir
    )
    subprocess.run(
        command,
        cwd=work_dir,
        env=env,
        check=True,
        timeout=timeout_s,
    )
    return require_file(
        work_dir / "results/grace/all.csv", "GRACE aggregate results"
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"GRACE CSV contains non-finite {field}: {row[field]!r}")
    return value


def collect_comparison(
    upstream_csv: Path,
    *,
    rmd_total_bytes: int,
    respawn_total_frames: int,
) -> list[ComparisonRow]:
    """Aggregate upstream frame rows into paired source/masked conditions."""
    if respawn_total_frames <= 0:
        raise ValueError("respawn_total_frames must be positive")
    with upstream_csv.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        missing = REQUIRED_UPSTREAM_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"GRACE CSV missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    grouped: dict[tuple[str, float, int, str], list[dict[str, str]]] = defaultdict(list)
    label_to_arm = {SOURCE_LABEL: "source", MASKED_LABEL: "respawn_masked"}
    for row in rows:
        video = Path(row["video"]).name
        arm = label_to_arm.get(video)
        if arm is None:
            continue
        key = (
            str(row["model_id"]),
            _finite_float(row, "loss"),
            int(_finite_float(row, "nframes")),
            arm,
        )
        grouped[key].append(row)

    conditions = sorted({key[:3] for key in grouped})
    output: list[ComparisonRow] = []
    rmd_per_frame = rmd_total_bytes / respawn_total_frames
    for model_id, loss, burst_frames in conditions:
        source_rows = grouped.get((model_id, loss, burst_frames, "source"))
        masked_rows = grouped.get(
            (model_id, loss, burst_frames, "respawn_masked")
        )
        if not source_rows or not masked_rows:
            missing_arm = "source" if not source_rows else "respawn_masked"
            raise ValueError(
                "Incomplete GRACE condition "
                f"model={model_id}, loss={loss}, burst={burst_frames}: "
                f"missing {missing_arm} rows"
            )
        if len(source_rows) != len(masked_rows):
            raise ValueError(
                "Mismatched GRACE row counts for "
                f"model={model_id}, loss={loss}, burst={burst_frames}: "
                f"source={len(source_rows)}, respawn_masked={len(masked_rows)}"
            )
        source_bytes = sum(_finite_float(row, "size") for row in source_rows)
        masked_video_bytes = sum(
            _finite_float(row, "size") for row in masked_rows
        )
        if source_bytes <= 0:
            raise ValueError(
                "Non-positive source byte total for "
                f"model={model_id}, loss={loss}, burst={burst_frames}"
            )
        evaluated_frames = len(masked_rows)
        rmd_bytes = rmd_per_frame * evaluated_frames
        delivered_masked_bytes = masked_video_bytes + rmd_bytes
        bsp = 100.0 * (1.0 - delivered_masked_bytes / source_bytes)
        output.append(
            {
                "model_id": model_id,
                "loss": loss,
                "burst_frames": burst_frames,
                "evaluated_frames": evaluated_frames,
                "source_grace_estimated_bytes": source_bytes,
                "respawn_masked_grace_estimated_bytes": masked_video_bytes,
                "respawn_rmd_prorated_bytes": rmd_bytes,
                "respawn_delivered_bytes": delivered_masked_bytes,
                "proxy_bsp_percent": bsp,
                "source_codec_psnr_mean": _mean(
                    _finite_float(row, "psnr") for row in source_rows
                ),
                "respawn_masked_codec_psnr_mean": _mean(
                    _finite_float(row, "psnr") for row in masked_rows
                ),
                "source_codec_ssim_mean": _mean(
                    _finite_float(row, "ssim") for row in source_rows
                ),
                "respawn_masked_codec_ssim_mean": _mean(
                    _finite_float(row, "ssim") for row in masked_rows
                ),
            }
        )
    if not output:
        raise ValueError(
            "GRACE CSV contained no paired source.mp4/respawn_masked.mp4 rows"
        )
    return output


def write_outputs(
    rows: list[ComparisonRow],
    *,
    output_dir: Path,
    provenance: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(csv_path)

    summary = {
        "status": "preliminary_proxy",
        "interpretation": (
            "PSNR/SSIM are measured against each arm's own GRACE input. "
            "GRACE size is a modeled entropy estimate, not an emitted bitstream. "
            "Its loss parameter zeros latent coefficients, not network packets. "
            "proxy_bsp_percent includes prorated RMD but excludes template delivery. "
            "Do not present this as wire bandwidth or end-to-end quality."
        ),
        "provenance": provenance,
        "conditions": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary_tmp.replace(summary_path)


def grace_revision(grace_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(grace_root), "rev-parse", "HEAD"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def invalidate_published_outputs(output_dir: Path) -> None:
    for filename in PUBLISHED_OUTPUTS:
        (output_dir / filename).unlink(missing_ok=True)


def validate_collect_manifest(
    manifest_path: Path,
    *,
    upstream_csv: Path,
    source: Path,
    masked: Path,
    rmd: Path,
    respawn_frames: int,
) -> dict[str, Any]:
    manifest = json.loads(require_file(manifest_path, "collection manifest").read_text())
    expected = {
        "source": str(source),
        "respawn_masked": str(masked),
        "respawn_rmd": str(rmd),
        "respawn_total_frames": respawn_frames,
        "upstream_csv": str(upstream_csv),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={actual!r}, current={expected_value!r}"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Collection manifest does not match inputs: {details}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grace-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--respawn-run", type=Path, default=DEFAULT_RESPAWN_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--collect-only",
        type=Path,
        help="Skip GRACE execution and reduce an existing upstream all.csv",
    )
    parser.add_argument(
        "--collect-manifest",
        type=Path,
        help="Required with --collect-only; binds the CSV to its staged inputs",
    )
    parser.add_argument(
        "--grace-timeout-s",
        type=int,
        default=3600,
        help="Maximum upstream GRACE runtime (default: 3600 seconds)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grace_root = args.grace_root.expanduser().resolve()
    source = require_file(args.source, "normalized source video")
    respawn_run = args.respawn_run.expanduser().resolve()
    masked = require_file(respawn_run / "segmented_output.mp4", "RESPAWN masked video")
    rmd = require_file(respawn_run / "msk1_payloads.bin", "RESPAWN RMD payload")
    output_dir = args.output_dir.expanduser().resolve()
    respawn_frames = ffprobe_frame_count(masked)

    command: list[str] | None = None
    if args.collect_only:
        upstream_csv = require_file(args.collect_only, "existing GRACE results")
        if args.collect_manifest is None:
            raise ValueError("--collect-manifest is required with --collect-only")
        run_manifest = validate_collect_manifest(
            args.collect_manifest,
            upstream_csv=upstream_csv,
            source=source,
            masked=masked,
            rmd=rmd,
            respawn_frames=respawn_frames,
        )
    else:
        work_dir, upstream_script = stage_upstream_work(
            grace_root=grace_root,
            source=source,
            masked=masked,
            output_dir=output_dir,
            force=args.force,
        )
        command = build_upstream_command(
            python=args.python,
            grace_root=grace_root,
            work_dir=work_dir,
            upstream_script=upstream_script,
        )
        upstream_csv = (
            work_dir / "results/grace/all.csv"
        ).resolve()
        run_manifest: dict[str, Any] = {
            "grace_root": str(grace_root),
            "grace_revision": grace_revision(grace_root),
            "upstream_csv": str(upstream_csv),
            "source": str(source),
            "respawn_masked": str(masked),
            "respawn_rmd": str(rmd),
            "respawn_total_frames": respawn_frames,
            "command": command,
            "upstream_semantics": {
                "frames_per_clip": 16,
                "frame_resize": "next multiple of 128",
                "size": "modeled entropy bytes plus BPG refresh estimate",
                "loss": "latent-coefficient zeroing, not packet loss",
                "input_path": "post-encode MP4; stacks codecs",
            },
        }
        if not args.dry_run and args.grace_timeout_s <= 0:
            raise ValueError("--grace-timeout-s must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
        if args.dry_run:
            print(json.dumps(run_manifest, indent=2))
            return
        invalidate_published_outputs(output_dir)
        upstream_csv = run_upstream(
            command,
            grace_root=grace_root,
            work_dir=work_dir,
            timeout_s=args.grace_timeout_s,
        )

    rows = collect_comparison(
        upstream_csv,
        rmd_total_bytes=rmd.stat().st_size,
        respawn_total_frames=respawn_frames,
    )
    write_outputs(
        rows,
        output_dir=output_dir,
        provenance=run_manifest,
    )
    print(json.dumps({"output_dir": str(output_dir), "conditions": len(rows)}))


if __name__ == "__main__":
    main()
