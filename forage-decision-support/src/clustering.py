"""Gower dissimilarity and hierarchical clustering."""

import gower
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .preprocessing import CLIMATE_COLUMNS, SOIL_FEATURES


def build_gower_matrix(df):
    """Compute the Gower matrix using the 27 descriptors used in the study."""
    features_cluster = SOIL_FEATURES + ["prod_norm"] + CLIMATE_COLUMNS
    df_cluster = df[features_cluster].copy()
    matrix = gower.gower_matrix(df_cluster)
    return matrix, features_cluster


def run_hac(matrix, k_min=2, k_max=6):
    """Evaluate complete-linkage HAC for k=2,...,6 using the Gower matrix."""
    best_k = k_min
    best_silhouette = -1.0
    best_labels = None
    history = []

    for k in range(k_min, k_max + 1):
        hac = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="complete",
        )
        labels = hac.fit_predict(matrix)
        sil = silhouette_score(matrix, labels, metric="precomputed")
        history.append((k, sil))

        if sil > best_silhouette:
            best_k = k
            best_silhouette = sil
            best_labels = labels

    return {
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "best_labels": best_labels,
        "history": history,
    }
