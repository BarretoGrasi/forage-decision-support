"""Spectral clustering evaluation using an affinity derived from Gower distance."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    D = np.asarray(gower_matrix, dtype=float)

    # Numerical safeguards.
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    gammas = [0.5, 1, 2, 5, 10]
    rows = []

    print("=" * 78)
    print("SPECTRAL CLUSTERING + GOWER")
    print("=" * 78)

    for gamma in gammas:
        # Convert Gower dissimilarity to a Gaussian affinity matrix.
        affinity = np.exp(-gamma * (D ** 2))
        np.fill_diagonal(affinity, 1.0)

        print(f"\nGAMMA = {gamma}")
        print("-" * 78)

        for k in range(2, 7):
            try:
                model = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    assign_labels="kmeans",
                    random_state=42,
                    n_init=50,
                )

                labels = model.fit_predict(affinity)
                n_clusters_found = len(np.unique(labels))

                if n_clusters_found < 2:
                    silhouette = np.nan
                else:
                    # Silhouette is evaluated in the original Gower space
                    # for direct comparability with the other clustering methods.
                    silhouette = silhouette_score(
                        D,
                        labels,
                        metric="precomputed",
                    )

                rows.append(
                    {
                        "Method": "Spectral + Gower",
                        "Gamma": gamma,
                        "k": k,
                        "Clusters found": n_clusters_found,
                        "Silhouette": silhouette,
                    }
                )

                if np.isnan(silhouette):
                    print(f"k = {k} | Silhouette = nan")
                else:
                    print(
                        f"k = {k} | "
                        f"Silhouette = {silhouette:.4f}"
                    )

            except Exception as exc:
                rows.append(
                    {
                        "Method": "Spectral + Gower",
                        "Gamma": gamma,
                        "k": k,
                        "Clusters found": np.nan,
                        "Silhouette": np.nan,
                    }
                )
                print(f"k = {k} | ERROR: {exc}")

    results = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("COMPLETE RESULTS")
    print("=" * 78)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    valid = results.dropna(
        subset=["Silhouette"]
    )

    if valid.empty:
        print("\nNo valid Spectral Clustering solution was produced.")
        return

    best = valid.loc[
        valid["Silhouette"].idxmax()
    ]

    print("\n" + "=" * 78)
    print("BEST SPECTRAL CONFIGURATION")
    print("=" * 78)

    print(f"Gamma          : {best['Gamma']}")
    print(f"Optimal k      : {int(best['k'])}")
    print(f"Silhouette     : {best['Silhouette']:.4f}")

    best_by_gamma = (
        valid.loc[
            valid
            .groupby("Gamma")["Silhouette"]
            .idxmax()
        ]
        .sort_values("Gamma")
    )

    print("\n" + "=" * 78)
    print("BEST RESULT FOR EACH GAMMA")
    print("=" * 78)

    print(
        best_by_gamma[
            ["Gamma", "k", "Silhouette"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
