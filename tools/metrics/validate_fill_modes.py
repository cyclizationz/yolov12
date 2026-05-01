import argparse
import csv
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FillCfg:
    mode: str
    blur_sigma: float | None = None
    bg_ema_alpha: float | None = None
    inpaint_radius: int | None = None
    inpaint_method: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(cmd)} (see {log_path})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate that different fill modes produce different masked outputs (first N frames)."
    )
    ap.add_argument("--bin", type=Path, default=Path("/home/tiehangz/proj/yolov12/deployment/build/Yolov12Deployment"))
    ap.add_argument("--model", type=Path, default=Path("/home/tiehangz/proj/yolov12/modification/yolov12n-seg-ppc5x5.onnx"))
    ap.add_argument("--video", type=Path, default=Path("/home/tiehangz/proj/yolov12/video/fc5_crop.mkv"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/tiehangz/proj/yolov12/record/fill/validate_200"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--enc-crf", type=int, default=18)
    ap.add_argument("--enc-preset", type=str, default="veryfast")
    ap.add_argument("--enc-gop", type=int, default=60)
    ap.add_argument("--mask-color", type=str, default="black")
    args = ap.parse_args()

    if not args.bin.exists():
        raise SystemExit(f"missing binary: {args.bin}")
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")
    if not args.video.exists():
        raise SystemExit(f"missing video: {args.video}")

    # Use force-mask-all so validation does not depend on latent banks / heal-only.
    fill_cfgs = [
        FillCfg("solid"),
        FillCfg("blur", blur_sigma=10.0),
        FillCfg("bg_ema", bg_ema_alpha=0.98),
        FillCfg("inpaint", inpaint_radius=5, inpaint_method="telea"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for cfg in fill_cfgs:
        run_dir = args.out_dir / f"{cfg.mode}_{args.frames}f"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"

        cmd = [
            str(args.bin),
            "-i",
            str(args.video),
            "-m",
            str(args.model),
            "-o",
            str(run_dir),
            "--yolo-force-mask-all",
            "--mask-color",
            str(args.mask_color),
            "--fill-mode",
            cfg.mode,
            "--feather-px",
            "0",
            "--max-frames",
            str(int(args.frames)),
            "--enc-codec",
            "libx264",
            "--enc-crf",
            str(int(args.enc_crf)),
            "--enc-preset",
            str(args.enc_preset),
            "--enc-gop",
            str(int(args.enc_gop)),
            "--enc-scenecut",
            "0",
            "--enc-tune",
            "none",
        ]
        if cfg.blur_sigma is not None:
            cmd += ["--fill-blur-sigma", str(float(cfg.blur_sigma))]
        if cfg.bg_ema_alpha is not None:
            cmd += ["--fill-bg-ema-alpha", str(float(cfg.bg_ema_alpha))]
        if cfg.inpaint_radius is not None:
            cmd += ["--fill-inpaint-radius", str(int(cfg.inpaint_radius))]
        if cfg.inpaint_method is not None:
            cmd += ["--fill-inpaint-method", str(cfg.inpaint_method)]

        _run(cmd, log_path=log_path)

        seg = run_dir / "segmented_output.mp4"
        orig = run_dir / "original_output.mp4"
        if not seg.exists() or not orig.exists():
            raise SystemExit(f"missing outputs under {run_dir}")

        rows.append(
            {
                "run_dir": str(run_dir),
                "fill_mode": cfg.mode,
                "frames": args.frames,
                "seg_bytes": seg.stat().st_size,
                "orig_bytes": orig.stat().st_size,
                "seg_sha256": _sha256(seg),
                "orig_sha256": _sha256(orig),
            }
        )

    # Write summary CSV
    out_csv = args.out_dir / "validate_fill_modes_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", out_csv)

    # Simple check: ensure not all segmented hashes are identical
    seg_hashes = {r["seg_sha256"] for r in rows}
    if len(seg_hashes) <= 1:
        raise SystemExit("ERROR: all segmented outputs are identical across fill modes (fill not applied)")
    print("OK: segmented outputs differ across fill modes.")


if __name__ == "__main__":
    main()

