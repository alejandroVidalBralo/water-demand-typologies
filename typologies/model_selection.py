"""Model selection: exclusion of degenerate partitions and Pareto front of Silhouette (max) and Davies-Bouldin (min)."""
import ast

import numpy as np


def largest_group_share(cluster_sizes):
    """Share of the days in the largest group, counting the HDBSCAN noise label (-1) as a group."""
    sizes = ast.literal_eval(cluster_sizes) if isinstance(cluster_sizes, str) else cluster_sizes
    return max(sizes.values()) / sum(sizes.values()) if sizes else np.nan


def split_degenerate(summary, threshold, sizes_column="cluster_sizes_first_run"):
    """Split the configurations with valid indices into (candidates, degenerate).

    A configuration is degenerate when one group holds more than `threshold` of the days (e.g. 362 + 4 days, or all
    days labelled as noise). This rule is applied before, and independently of, the Pareto analysis.
    """
    valid = summary.dropna(subset=["silhouette_mean", "davies_bouldin_mean"]).copy()
    valid["largest_group_share"] = valid[sizes_column].apply(largest_group_share)
    degenerate = valid["largest_group_share"] > threshold
    return valid[~degenerate].copy(), valid[degenerate].copy()


def pareto_front(silhouette, davies_bouldin):
    """Non-dominated configurations: no other one has Silhouette >= and Davies-Bouldin <=, with one strictly better.

    Returns (mask of the front, number of configurations that dominate each one).
    """
    sil, dbi = np.asarray(silhouette, dtype=float), np.asarray(davies_bouldin, dtype=float)
    dominated_by = np.array([((sil >= s) & (dbi <= d) & ((sil > s) | (dbi < d))).sum() for s, d in zip(sil, dbi)])
    return dominated_by == 0, dominated_by


def select_candidates(summary, threshold):
    """Exclude degenerate configurations and return (front, all candidates with `dominated_by`, degenerate)."""
    candidates, degenerate = split_degenerate(summary, threshold)
    on_front, dominated_by = pareto_front(candidates["silhouette_mean"], candidates["davies_bouldin_mean"])
    candidates["dominated_by"] = dominated_by
    front = candidates[on_front].sort_values("silhouette_mean", ascending=False)
    return front, candidates, degenerate
