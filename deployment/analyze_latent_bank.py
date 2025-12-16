import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_bank(path: Path):
    bank = json.loads(path.read_text())
    ids = np.array([b["id"] for b in bank], dtype=np.int64)
    counts = np.array([b.get("use_count", 0) for b in bank], dtype=np.int64)
    embs = np.array([b["emb"] for b in bank], dtype=np.float32)  # already L2-normalized
    return bank, ids, counts, embs


def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    # embs are L2 normalized
    return np.clip(embs @ embs.T, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="Run output dir containing latent_bank.json + report.json")
    ap.add_argument("--top", type=int, default=40, help="Top-N templates by use_count for heatmap/clustering")
    args = ap.parse_args()

    out_dir = args.out_dir
    bank_path = out_dir / "latent_bank.json"
    report_path = out_dir / "report.json"
    if not bank_path.exists():
        raise FileNotFoundError(f"missing {bank_path}")

    bank, ids, counts, embs = load_bank(bank_path)

    # Sort by usage
    order = np.argsort(-counts)
    top = int(min(args.top, len(order)))
    top_idx = order[:top]
    top_ids = ids[top_idx]
    top_counts = counts[top_idx]
    top_embs = embs[top_idx]

    # Heatmap of cosine similarities between hottest templates
    C = cosine_matrix(top_embs)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, vmin=0.95, vmax=1.0, cmap="viridis")
    ax.set_title(f"Cosine similarity heatmap (top {top} templates)")
    ax.set_xlabel("template rank (by use_count)")
    ax.set_ylabel("template rank (by use_count)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_png = out_dir / "latent_top_cosine_heatmap.png"
    fig.savefig(heatmap_png, dpi=160)

    # Simple clustering: greedy grouping by similarity threshold (fast, no extra deps)
    sim_thr = 0.995
    clusters = []
    used = np.zeros(top, dtype=bool)
    for i in range(top):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, top):
            if used[j]:
                continue
            if C[i, j] >= sim_thr:
                used[j] = True
                members.append(j)
        clusters.append(members)

    clusters.sort(key=lambda mem: int(top_counts[mem].sum()), reverse=True)

    # Write cluster summary
    cluster_summary = []
    for ci, mem in enumerate(clusters[:20]):
        cluster_summary.append(
            {
                "cluster": ci,
                "size": len(mem),
                "total_use_count": int(top_counts[mem].sum()),
                "template_ids": [int(top_ids[k]) for k in mem],
                "use_counts": [int(top_counts[k]) for k in mem],
            }
        )

    out_json = out_dir / "latent_clusters_top.json"
    out_json.write_text(json.dumps({"sim_thr": sim_thr, "top": top, "clusters": cluster_summary}, indent=2))

    # Also plot usage bar chart for top templates
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(np.arange(top), top_counts)
    ax2.set_title(f"Top {top} template usage counts")
    ax2.set_xlabel("template rank")
    ax2.set_ylabel("use_count")
    plt.tight_layout()
    usage_png = out_dir / "latent_top_usage.png"
    fig2.savefig(usage_png, dpi=160)

    # Print quick summary
    total_sel = int(counts.sum())
    top_sel = int(top_counts.sum())
    print("bank_size:", len(bank))
    print("total_selections:", total_sel)
    print(f"top{top}_selections_share:", (top_sel / total_sel) if total_sel else 0.0)
    print("saved:", heatmap_png)
    print("saved:", out_json)
    print("saved:", usage_png)


if __name__ == "__main__":
    main()


