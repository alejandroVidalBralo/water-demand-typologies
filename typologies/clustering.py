"""Clustering algorithms and internal validity indices (Sections 3.4 and 3.5 of the paper)."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def make_clusterer(method, params, seed, prediction_data=False):
    """Unfitted clustering model. `prediction_data=True` lets HDBSCAN assign new points later."""
    if method == "KMeans":
        return KMeans(n_clusters=params.get("n_clusters", 5), random_state=seed, n_init="auto")
    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=params.get("n_clusters", 5), linkage="ward")
    if method == "HDBSCAN":
        from hdbscan import HDBSCAN
        return HDBSCAN(min_cluster_size=params.get("min_cluster_size", 20), min_samples=params.get("min_samples", 10),
                       cluster_selection_method=params.get("cluster_selection_method", "eom"),
                       prediction_data=prediction_data)
    raise ValueError(f"Unknown clustering: {method}")


def cluster(X, method, params, seed):
    """Labels of each observation. HDBSCAN labels noise points as -1."""
    return make_clusterer(method, params, seed).fit_predict(X)


def validity_indices(X, labels):
    """Silhouette, Davies-Bouldin and Calinski-Harabasz indices, computed in the space where X was clustered.

    The HDBSCAN noise label (-1) is treated as one more group, so every partition is evaluated on all observations.
    Returns NaN for partitions with fewer than two clusters.
    """
    if len(set(labels) - {-1}) < 2:
        return np.nan, np.nan, np.nan
    try:
        return silhouette_score(X, labels), davies_bouldin_score(X, labels), calinski_harabasz_score(X, labels)
    except Exception:
        return np.nan, np.nan, np.nan
