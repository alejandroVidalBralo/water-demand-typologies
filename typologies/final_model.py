"""Fitting of the selected configuration and assignment of new days without retraining.

New days are processed only with quantities estimated on the training days: the global scaling statistics, the
fitted reduction model, the minimum and maximum of the training embedding, and the typology centroids.
"""
from dataclasses import dataclass, field

import numpy as np

from .clustering import make_clusterer
from .representation import apply_scaling, fit_embedding, fit_scaling, transform_embedding


@dataclass
class TypologyModel:
    scaling_params: dict
    reducer: object
    rescaler: object
    clusterer: object
    clustering: str
    embedding: np.ndarray
    labels: np.ndarray
    centroids: dict = field(default_factory=dict)


def fit_typology_model(daily_profiles, scaling, reduction, reduction_params, clustering, clustering_params, seed):
    if reduction == "TSNE":
        raise ValueError("t-SNE cannot place new observations in a fitted embedding.")
    scaling_params = fit_scaling(daily_profiles, scaling)
    X = apply_scaling(daily_profiles, scaling_params)
    embedding, reducer, rescaler = fit_embedding(X, reduction, reduction_params, seed)
    clusterer = make_clusterer(clustering, clustering_params, seed, prediction_data=True)
    labels = clusterer.fit_predict(embedding)
    centroids = {c: embedding[labels == c].mean(axis=0) for c in sorted(set(labels) - {-1})}
    return TypologyModel(scaling_params, reducer, rescaler, clusterer, clustering, embedding, labels, centroids)


def assign_new_days(model, daily_profiles_new):
    """Return (embedding, typology, assignment margin) of the new days.

    Ward has no prediction rule, so each day is assigned to the nearest centroid. The margin,
    1 - d(nearest) / d(second nearest), is close to 0 for days between two typologies. For HDBSCAN, the third value
    is the membership strength of its approximate prediction.
    """
    X_new = apply_scaling(daily_profiles_new, model.scaling_params)
    embedding = transform_embedding(X_new, model.reducer, model.rescaler)
    if model.clustering == "HDBSCAN":
        import hdbscan
        labels, strength = hdbscan.approximate_predict(model.clusterer, embedding)
        return embedding, labels, strength
    ids = np.array(list(model.centroids))
    centroids = np.array(list(model.centroids.values()))
    distances = np.linalg.norm(embedding[:, None, :] - centroids[None, :, :], axis=2)
    if model.clustering == "KMeans":
        labels = model.clusterer.predict(embedding)
    else:
        labels = ids[distances.argmin(axis=1)]
    nearest_two = np.sort(distances, axis=1)[:, :2]
    margin = 1 - nearest_two[:, 0] / nearest_two[:, 1]
    return embedding, labels, margin
