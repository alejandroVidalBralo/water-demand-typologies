"""Step 3 - Exclude degenerate partitions and compute the Pareto front of Silhouette (max) and Davies-Bouldin (min).

Usage:  python step3_pareto_front.py                                   uses output/grid_summary.csv (step 2)
        python step3_pareto_front.py --summary results/grid_summary.csv   reproduces the front of the paper
                                                                         from the published grid metrics
Output: output/pareto_front.csv, output/pareto_front.png
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

import config
from typologies.model_selection import select_candidates

plt.switch_backend("Agg")

COLUMNS = ["config_id", "scaling", "reduction", "reduction_params", "clustering", "clustering_params",
           "silhouette_mean", "silhouette_sd", "davies_bouldin_mean", "davies_bouldin_sd", "n_clusters_mean",
           "n_noise_mean", "cluster_sizes_first_run"]


def plot(candidates, front, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for reduction, group in candidates.groupby("reduction"):
        ax.scatter(group["silhouette_mean"], group["davies_bouldin_mean"], s=4, alpha=0.35, label=reduction)
    f = front.sort_values("silhouette_mean")
    ax.plot(f["silhouette_mean"], f["davies_bouldin_mean"], "k-o", ms=5, lw=1, label="Pareto front")
    ax.set_xlabel("Mean Silhouette (higher is better)")
    ax.set_ylabel("Mean Davies-Bouldin index (lower is better)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=config.OUTPUT_DIR / "grid_summary.csv")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    front, candidates, degenerate = select_candidates(summary, config.DEGENERATE_SHARE)
    print(f"{len(summary):,} configurations; with valid indices: {len(candidates) + len(degenerate):,}; "
          f"degenerate (one group > {config.DEGENERATE_SHARE:.0%} of the days): {len(degenerate)}; "
          f"candidates: {len(candidates):,}")
    print(f"\nPareto front ({len(front)} configurations):")
    for _, r in front.iterrows():
        print(f"  {r['config_id']}  Silhouette {r['silhouette_mean']:.3f} +- {r['silhouette_sd']:.3f}  "
              f"Davies-Bouldin {r['davies_bouldin_mean']:.3f} +- {r['davies_bouldin_sd']:.3f}  | {r['scaling']} | "
              f"{r['reduction']} {r['reduction_params']} | {r['clustering']} {r['clustering_params']}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    front[COLUMNS].to_csv(config.OUTPUT_DIR / "pareto_front.csv", index=False)
    plot(candidates, front, config.OUTPUT_DIR / "pareto_front.png")
    print("\n-> output/pareto_front.csv, output/pareto_front.png")


if __name__ == "__main__":
    main()
