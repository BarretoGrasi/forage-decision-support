"""Gaussian Mixture Model evaluation using MDS coordinates derived from Gower distance.

The Gower dissimilarity matrix is projected into Euclidean spaces of different
dimensionalities using metric MDS. Gaussian Mixture Models are then fitted
across multiple covariance structures and numbers of components. Final cluster
partitions are evaluated using the silhouette coefficient computed in the
original Gower dissimilarity space.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    print("=" * 78)
    print("GAUSSIAN MIXTURE MODEL (GMM) + GOWER-MDS")
    print("=" * 78)

    D = np.asarray(gower_matrix, dtype=float)

    # Numerical safeguards.
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    print(f"\nNumber of observations: {D.shape[0]}")

    dimensions = [2, 3, 5, 10]
    covariance_types = [
        "full",
        "tied",
        "diag",
        "spherical",
    ]

    rows = []

    best_result = None
    best_labels = None

    for n_dim in dimensions:
        print("\n" + "=" * 78)
        print(f"MDS — {n_dim} DIMENSIONS")
        print("=" * 78)

        mds = MDS(
            n_components=n_dim,
            dissimilarity="precomputed",
            random_state=42,
            n_init=4,
            max_iter=500,
        )

        X_mds = mds.fit_transform(D)

        print(f"MDS stress: {mds.stress_:.4f}")
        print("-" * 78)

        for covariance_type in covariance_types:
            for k in range(2, 7):
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=covariance_type,
                        random_state=42,
                        n_init=20,
                        max_iter=1000,
                        reg_covar=1e-6,
                    )

                    labels = gmm.fit_predict(X_mds)
                    n_clusters = len(np.unique(labels))

                    if n_clusters < 2:
                        continue

                    # Evaluate the partition in the original Gower space
                    # for comparability with the other clustering strategies.
                    silhouette = silhouette_score(
                        D,
                        labels,
                        metric="precomputed",
                    )

                    bic = gmm.bic(X_mds)
                    aic = gmm.aic(X_mds)

                    result = {
                        "MDS_dim": n_dim,
                        "Covariance": covariance_type,
                        "k": k,
                        "Clusters": n_clusters,
                        "Silhouette_Gower": silhouette,
                        "BIC": bic,
                        "AIC": aic,
                        "MDS_stress": mds.stress_,
                    }

                    rows.append(result)

                    print(
                        f"cov={covariance_type:10s} | "
                        f"k={k} | "
                        f"Silhouette={silhouette:.4f} | "
                        f"BIC={bic:.2f}"
                    )

                    if (
                        best_result is None
                        or silhouette > best_result["Silhouette_Gower"]
                    ):
                        best_result = result.copy()
                        best_labels = labels.copy()

                except Exception as exc:
                    print(
                        f"ERROR | dim={n_dim} | "
                        f"cov={covariance_type} | "
                        f"k={k}: {exc}"
                    )

    results = pd.DataFrame(rows)

    if results.empty:
        print("\nNo valid GMM configuration was produced.")
        return

    print("\n" + "=" * 78)
    print("TOP 20 GMM CONFIGURATIONS BY GOWER SILHOUETTE")
    print("=" * 78)

    top20 = (
        results
        .sort_values(
            "Silhouette_Gower",
            ascending=False,
        )
        .head(20)
    )

    print(
        top20.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 78)
    print("BEST GMM CONFIGURATION — GOWER SILHOUETTE")
    print("=" * 78)

    print(
        f"MDS dimensions    : "
        f"{best_result['MDS_dim']}"
    )
    print(
        f"Covariance        : "
        f"{best_result['Covariance']}"
    )
    print(
        f"k                 : "
        f"{best_result['k']}"
    )
    print(
        f"Silhouette Gower  : "
        f"{best_result['Silhouette_Gower']:.4f}"
    )
    print(
        f"BIC               : "
        f"{best_result['BIC']:.4f}"
    )
    print(
        f"AIC               : "
        f"{best_result['AIC']:.4f}"
    )
    print(
        f"MDS stress        : "
        f"{best_result['MDS_stress']:.4f}"
    )

    print("\n" + "=" * 78)
    print("CLUSTER SIZES — BEST GMM")
    print("=" * 78)

    unique, counts = np.unique(
        best_labels,
        return_counts=True,
    )

    for cluster, n in zip(unique, counts):
        print(f"Cluster {cluster}: {n}")

    print("\n" + "=" * 78)
    print("BEST RESULT FOR EACH MDS DIMENSION")
    print("=" * 78)

    idx = (
        results
        .groupby("MDS_dim")["Silhouette_Gower"]
        .idxmax()
    )

    best_by_dimension = (
        results
        .loc[idx]
        .sort_values("MDS_dim")
    )

    print(
        best_by_dimension.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
