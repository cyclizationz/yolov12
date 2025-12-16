import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Det:
    frame: int
    cls: int
    conf: float
    w: int
    h: int
    emb: np.ndarray  # shape (32,), L2 normalized


def load_latents_jsonl(path: Path, conf_thr: float) -> list[Det]:
    dets: list[Det] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("conf", 0.0) < conf_thr:
                continue
            emb = np.array(j["emb"], dtype=np.float32)
            # normalize defensively
            n = np.linalg.norm(emb)
            if n > 1e-8:
                emb = emb / n
            dets.append(
                Det(
                    frame=int(j["frame"]),
                    cls=int(j["cls"]),
                    conf=float(j.get("conf", 0.0)),
                    w=int(j["w"]),
                    h=int(j["h"]),
                    emb=emb,
                )
            )
    dets.sort(key=lambda d: d.frame)
    return dets


def simulate_assignment(
    dets: list[Det],
    cosine_thr: float,
    size_tol: int = 2,
) -> dict:
    """
    Simulate server-side template_id assignment:
    - Maintain a template bank per class with embeddings.
    - For each detection, find best cosine sim among same class & similar size.
    - Reuse if bestSim >= cosine_thr else mint new template_id.
    Returns summary stats + bestSim distribution.
    """
    bank_by_cls: dict[int, list[tuple[np.ndarray, tuple[int, int]]]] = defaultdict(list)
    minted = 0
    reused = 0
    best_sims: list[float] = []
    frames_seen = set()

    for d in dets:
        frames_seen.add(d.frame)
        bank = bank_by_cls[d.cls]

        best = -1.0
        if bank:
            for emb, (bw, bh) in bank:
                if abs(bw - d.w) > size_tol or abs(bh - d.h) > size_tol:
                    continue
                sim = float(np.dot(d.emb, emb))
                if sim > best:
                    best = sim
        best_sims.append(best if best >= 0 else np.nan)

        if best >= cosine_thr:
            reused += 1
        else:
            minted += 1
            bank.append((d.emb, (d.w, d.h)))

    n = len(dets)
    return {
        "cosine_thr": cosine_thr,
        "detections": n,
        "frames": len(frames_seen),
        "minted": minted,
        "reused": reused,
        "mint_rate": minted / n if n else 0.0,
        "bank_size": sum(len(v) for v in bank_by_cls.values()),
        "best_sims": np.array(best_sims, dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True, help="Path to latents.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for plots/results")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold to filter detections")
    ap.add_argument("--size-tol", type=int, default=2, help="BBox size tolerance for matching")
    ap.add_argument("--target-mint-rate", type=float, default=0.5, help="Desired minted template ratio (e.g. 0.5)")
    ap.add_argument("--thr-min", type=float, default=0.85)
    ap.add_argument("--thr-max", type=float, default=0.995)
    ap.add_argument("--thr-step", type=float, default=0.005)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dets = load_latents_jsonl(args.latents, conf_thr=args.conf)
    if not dets:
        raise SystemExit("No detections found in latents file after filtering.")

    thrs = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step, dtype=np.float32)
    results = []
    for t in thrs:
        results.append(simulate_assignment(dets, float(t), size_tol=args.size_tol))

    mint_rates = np.array([r["mint_rate"] for r in results], dtype=np.float32)
    bank_sizes = np.array([r["bank_size"] for r in results], dtype=np.float32)

    # Recommended threshold: closest mint rate to target
    target = float(args.target_mint_rate)
    idx = int(np.argmin(np.abs(mint_rates - target)))
    recommended = results[idx]

    # Plot: mint rate vs threshold + bank size
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(thrs, mint_rates, label="mint_rate (new templates / dets)", color="tab:blue")
    ax1.axhline(target, linestyle="--", color="tab:blue", alpha=0.4)
    ax1.axvline(recommended["cosine_thr"], linestyle="--", color="black", alpha=0.6, label=f"recommended={recommended['cosine_thr']:.3f}")
    ax1.set_xlabel("cosine threshold")
    ax1.set_ylabel("mint rate")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thrs, bank_sizes, label="bank_size", color="tab:orange")
    ax2.set_ylabel("template bank size")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    out_png = args.out_dir / "latent_threshold_sweep.png"
    fig.savefig(out_png, dpi=150)

    # Save summary JSON
    out_json = args.out_dir / "latent_threshold_sweep.json"
    payload = {
        "latents": str(args.latents),
        "conf": args.conf,
        "size_tol": args.size_tol,
        "target_mint_rate": target,
        "recommended": {k: v for k, v in recommended.items() if k != "best_sims"},
        "grid": [
            {k: v for k, v in r.items() if k != "best_sims"}
            for r in results
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    # Also dump bestSim distribution at the recommended threshold
    sims = recommended["best_sims"]
    sims = sims[np.isfinite(sims)]
    if sims.size:
        q = np.quantile(sims, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).tolist()
    else:
        q = []
    (args.out_dir / "latent_bestSim_quantiles.json").write_text(
        json.dumps({"recommended_thr": recommended["cosine_thr"], "quantiles": q}, indent=2)
    )

    print("detections:", len(dets))
    print("recommended_thr:", recommended["cosine_thr"])
    print("mint_rate:", recommended["mint_rate"])
    print("bank_size:", recommended["bank_size"])
    print("saved:", out_png)
    print("saved:", out_json)


if __name__ == "__main__":
    main()


