"""Global scaling and dimensionality reduction."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def fit_scaling(X, method):
    """Statistics of a global scaling: a single shift and scale for the whole matrix."""
    X = np.asarray(X, dtype=float)
    if method == "zscore":
        return {"shift": X.mean(), "scale": X.std()}
    if method == "minmax":
        return {"shift": X.min(), "scale": X.max() - X.min()}
    if method == "none":
        return {"shift": 0.0, "scale": 1.0}
    raise ValueError(f"Unknown scaling: {method}")


def apply_scaling(X, params):
    return (np.asarray(X, dtype=float) - params["shift"]) / params["scale"]


def make_reducer(method, params, seed):
    if method == "NONE":
        return None
    if method == "PCA":
        return PCA(n_components=params.get("n_components", 2))
    if method == "TSNE":
        n = params.get("n_components", 2)
        return TSNE(n_components=n, method="barnes_hut" if n <= 3 else "exact",
                    perplexity=params.get("perplexity", 30), early_exaggeration=params.get("early_exaggeration", 12),
                    random_state=seed)
    if method == "UMAP":
        from umap import UMAP
        return UMAP(n_components=params.get("n_components", 2), n_neighbors=params.get("n_neighbors", 15),
                    min_dist=params.get("min_dist", 0.1), random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown reduction: {method}")


def _finite(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def fit_embedding(X, method, params, seed):
    """Return (embedding, reducer, rescaler).

    Each embedding dimension is rescaled to [0, 1]. Without reduction, the 24 hourly columns are not rescaled and the
    reducer and rescaler are None.
    """
    reducer = make_reducer(method, params, seed)
    if reducer is None:
        return np.asarray(_finite(X), dtype=float), None, None
    raw = _finite(reducer.fit_transform(X))
    rescaler = MinMaxScaler(feature_range=(0, 1)).fit(raw)
    return rescaler.transform(raw), reducer, rescaler


def transform_embedding(X_new, reducer, rescaler):
    """Place new observations in a fitted embedding (not available for t-SNE)."""
    if reducer is None:
        return np.asarray(_finite(X_new), dtype=float)
    return rescaler.transform(_finite(reducer.transform(X_new)))
