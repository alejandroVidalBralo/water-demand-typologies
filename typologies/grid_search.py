"""Exhaustive grid search over scaling x reduction x clustering, repeated with several seeds (Section 3.7)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .clustering import cluster, validity_indices
from .representation import apply_scaling, fit_embedding, fit_scaling

CONFIG_KEYS = ["scaling", "reduction", "reduction_params", "clustering", "clustering_params"]
METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters", "n_noise"]


def evaluate_embedding(scaling, X_scaled, reduction, reduction_params, seed, clusterings):
    """Fit one embedding and evaluate every clustering configuration on it. Returns one row per run."""
    try:
        embedding, _, _ = fit_embedding(X_scaled, reduction, reduction_params, seed)
    except Exception:
        return []
    rows = []
    for method, param_list in clusterings.items():
        for params in param_list:
            try:
                labels = cluster(embedding, method, params, seed)
                silhouette, davies_bouldin, calinski_harabasz = validity_indices(embedding, labels)
            except Exception:
                continue
            groups, sizes = np.unique(labels, return_counts=True)
            rows.append({
                "scaling": scaling, "reduction": reduction, "reduction_params": str(reduction_params),
                "clustering": method, "clustering_params": str(params), "seed": seed,
                "silhouette": silhouette, "davies_bouldin": davies_bouldin, "calinski_harabasz": calinski_harabasz,
                "n_clusters": len(set(labels) - {-1}), "n_noise": int(np.sum(labels == -1)),
                "cluster_sizes": str({int(g): int(s) for g, s in zip(groups, sizes)}),
            })
    return rows


def run_grid(daily_profiles, scalings, reductions, clusterings, seeds, n_jobs=-1, verbose=5):
    """Run every configuration with every seed. Returns one row per run (configuration x seed).

    The work is split into one task per (scaling, reduction setting, seed): the embedding is fitted once and all the
    clustering configurations are evaluated on it.
    """
    X = np.asarray(daily_profiles, dtype=float)
    scaled = {s: apply_scaling(X, fit_scaling(X, s)) for s in scalings}
    tasks = [(s, scaled[s], red, params, seed)
             for s in scalings for red, param_list in reductions.items() for params in param_list for seed in seeds]
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_embedding)(s, Xs, red, params, seed, clusterings) for s, Xs, red, params, seed in tasks)
    return pd.DataFrame([row for rows in results for row in rows])


def summarise_over_seeds(runs):
    """Mean and standard deviation of the metrics over seeds, for each configuration, with a stable identifier."""
    grouped = runs.groupby(CONFIG_KEYS, dropna=False)
    summary = grouped[METRICS].mean().add_suffix("_mean").join(grouped[METRICS].std().add_suffix("_sd"))
    summary["cluster_sizes_first_run"] = grouped["cluster_sizes"].first()
    summary = summary.reset_index()
    summary.insert(0, "config_id", config_ids(summary))
    return summary.sort_values("config_id").reset_index(drop=True)


def config_ids(summary):
    """Identifier M00001, M00002, ... in the alphabetical order of the configuration (independent of the results)."""
    order = summary[CONFIG_KEYS].astype(str).sort_values(CONFIG_KEYS).index
    ids = pd.Series([f"M{i:05d}" for i in range(1, len(order) + 1)], index=order)
    return ids.reindex(summary.index).values
