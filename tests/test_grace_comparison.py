from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "experiments"
    / "run_grace_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("run_grace_comparison", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_upstream_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["video", "model_id", "loss", "nframes", "size", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_collect_comparison_pairs_arms_and_prorates_rmd(tmp_path: Path) -> None:
    upstream = tmp_path / "all.csv"
    write_upstream_csv(
        upstream,
        [
            {
                "video": "source.mp4",
                "model_id": 4096,
                "loss": 0.1,
                "nframes": 1,
                "size": 100,
                "psnr": 30,
                "ssim": 0.90,
            },
            {
                "video": "source.mp4",
                "model_id": 4096,
                "loss": 0.1,
                "nframes": 1,
                "size": 100,
                "psnr": 28,
                "ssim": 0.86,
            },
            {
                "video": "respawn_masked.mp4",
                "model_id": 4096,
                "loss": 0.1,
                "nframes": 1,
                "size": 70,
                "psnr": 32,
                "ssim": 0.94,
            },
            {
                "video": "respawn_masked.mp4",
                "model_id": 4096,
                "loss": 0.1,
                "nframes": 1,
                "size": 70,
                "psnr": 30,
                "ssim": 0.90,
            },
        ],
    )

    rows = MODULE.collect_comparison(
        upstream, rmd_total_bytes=100, respawn_total_frames=20
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source_grace_estimated_bytes"] == 200
    assert row["respawn_masked_grace_estimated_bytes"] == 140
    assert row["respawn_rmd_prorated_bytes"] == 10
    assert row["respawn_delivered_bytes"] == 150
    assert row["proxy_bsp_percent"] == pytest.approx(25.0)
    assert row["source_codec_psnr_mean"] == pytest.approx(29.0)
    assert row["respawn_masked_codec_ssim_mean"] == pytest.approx(0.92)


def test_collect_comparison_rejects_unpaired_inputs(tmp_path: Path) -> None:
    upstream = tmp_path / "all.csv"
    write_upstream_csv(
        upstream,
        [
            {
                "video": "source.mp4",
                "model_id": 4096,
                "loss": 0,
                "nframes": 0,
                "size": 100,
                "psnr": 30,
                "ssim": 0.9,
            }
        ],
    )

    with pytest.raises(ValueError, match="Incomplete GRACE condition"):
        MODULE.collect_comparison(
            upstream, rmd_total_bytes=0, respawn_total_frames=1
        )


def test_collect_comparison_rejects_mismatched_arm_counts(tmp_path: Path) -> None:
    upstream = tmp_path / "all.csv"
    row = {
        "model_id": 4096,
        "loss": 0.1,
        "nframes": 1,
        "size": 100,
        "psnr": 30,
        "ssim": 0.9,
    }
    write_upstream_csv(
        upstream,
        [
            {"video": "source.mp4", **row},
            {"video": "source.mp4", **row},
            {"video": "respawn_masked.mp4", **row},
        ],
    )

    with pytest.raises(ValueError, match="Mismatched GRACE row counts"):
        MODULE.collect_comparison(
            upstream, rmd_total_bytes=0, respawn_total_frames=20
        )


def test_collect_comparison_validates_upstream_schema(tmp_path: Path) -> None:
    upstream = tmp_path / "all.csv"
    upstream.write_text("video,size\nsource.mp4,100\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        MODULE.collect_comparison(
            upstream, rmd_total_bytes=0, respawn_total_frames=1
        )


def test_stage_upstream_work_builds_isolated_layout(tmp_path: Path) -> None:
    grace_root = tmp_path / "Grace"
    for relative in ("grace", "libs", "models"):
        (grace_root / relative).mkdir(parents=True)
    (grace_root / "grace-gpu.py").write_text("# evaluator\n", encoding="utf-8")
    source = tmp_path / "source-input.mp4"
    masked = tmp_path / "masked-input.mp4"
    source.touch()
    masked.touch()
    output = tmp_path / "output"

    work_dir, script = MODULE.stage_upstream_work(
        grace_root=grace_root,
        source=source,
        masked=masked,
        output_dir=output,
        force=False,
    )

    assert script == (grace_root / "grace-gpu.py").resolve()
    assert (work_dir / "inputs/source.mp4").resolve() == source.resolve()
    assert (work_dir / "inputs/respawn_masked.mp4").resolve() == masked.resolve()
    assert (work_dir / "grace").resolve() == (grace_root / "grace").resolve()
    assert (work_dir / "INDEX.txt").read_text(encoding="utf-8").splitlines() == [
        str(work_dir / "inputs/source.mp4"),
        str(work_dir / "inputs/respawn_masked.mp4"),
    ]

    with pytest.raises(FileExistsError):
        MODULE.stage_upstream_work(
            grace_root=grace_root,
            source=source,
            masked=masked,
            output_dir=output,
            force=False,
        )


def test_collect_manifest_rejects_unrelated_csv(tmp_path: Path) -> None:
    source = (tmp_path / "source.mp4").resolve()
    masked = (tmp_path / "masked.mp4").resolve()
    rmd = (tmp_path / "msk1_payloads.bin").resolve()
    upstream = (tmp_path / "all.csv").resolve()
    for path in (source, masked, rmd, upstream):
        path.touch()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        (
            '{"source": "%s", "respawn_masked": "%s", "respawn_rmd": "%s", '
            '"respawn_total_frames": 20, "upstream_csv": "/other/all.csv"}'
        )
        % (source, masked, rmd),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="upstream_csv"):
        MODULE.validate_collect_manifest(
            manifest,
            upstream_csv=upstream,
            source=source,
            masked=masked,
            rmd=rmd,
            respawn_frames=20,
        )
