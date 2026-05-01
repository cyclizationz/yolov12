import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.quantile(np.array(xs, dtype=np.float64), q))


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.array(xs, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize deployment report.json quality/compute metrics.")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rep = json.loads(args.report.read_text())
    per = rep.get("per_frame", []) or []

    def col(name: str) -> list[float]:
        out: list[float] = []
        for e in per:
            v = e.get(name)
            if v is None:
                continue
            out.append(float(v))
        return out

    total_ms = col("total_ms")
    ssim = col("ssim")
    psnr = col("psnr")
    rec_ssim = col("rec_ssim")
    rec_psnr = col("rec_psnr")

    out: dict[str, Any] = {
        "frames": len(per),
        "avg_ssim": rep.get("avg_ssim"),
        "avg_psnr": rep.get("avg_psnr"),
        "avg_recovered_ssim": rep.get("avg_recovered_ssim"),
        "avg_recovered_psnr": rep.get("avg_recovered_psnr"),
        "latent_bank_size": rep.get("latent_bank_size"),
        "latent_reuse_ratio": rep.get("latent_reuse_ratio"),
        "compute_ms": {
            "mean": _mean(total_ms),
            "p50": _q(total_ms, 0.50),
            "p90": _q(total_ms, 0.90),
            "p95": _q(total_ms, 0.95),
            "p99": _q(total_ms, 0.99),
        },
        "quality_masked": {
            "ssim_mean": _mean(ssim),
            "ssim_p50": _q(ssim, 0.50),
            "ssim_p10": _q(ssim, 0.10),
            "psnr_mean": _mean(psnr),
            "psnr_p50": _q(psnr, 0.50),
            "psnr_p10": _q(psnr, 0.10),
        },
        "quality_recovered": {
            "ssim_mean": _mean(rec_ssim) if rec_ssim else float("nan"),
            "ssim_p50": _q(rec_ssim, 0.50) if rec_ssim else float("nan"),
            "ssim_p10": _q(rec_ssim, 0.10) if rec_ssim else float("nan"),
            "psnr_mean": _mean(rec_psnr) if rec_psnr else float("nan"),
            "psnr_p50": _q(rec_psnr, 0.50) if rec_psnr else float("nan"),
            "psnr_p10": _q(rec_psnr, 0.10) if rec_psnr else float("nan"),
        },
    }

    s = json.dumps(out, indent=2)
    print(s)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s)


if __name__ == "__main__":
    main()


