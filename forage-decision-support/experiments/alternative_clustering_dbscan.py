"""DBSCAN evaluation using the precomputed Gower distance matrix."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
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

    eps_values = np.arange(0.05, 0.51, 0.025)
    min_samples_values = [3, 4, 5, 6, 8, 10]

    rows = []

    print("=" * 78)
    print("DBSCAN + GOWER")
    print("=" * 78)

    for min_samples in min_samples_values:
        for eps in eps_values:
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="precomputed",
            )

            labels = model.fit_predict(D)

            mask = labels != -1
            valid_labels = labels[mask]

            n_valid = int(np.sum(mask))
            n_noise = int(np.sum(~mask))

            noise_percent = (
                n_noise / len(labels) * 100
            )

            clusters = np.unique(valid_labels)
            n_clusters = len(clusters)

            silhouette = np.nan

            if (
                n_clusters >= 2
                and n_valid > n_clusters
            ):
                D_valid = D[np.ix_(mask, mask)]

                try:
                    silhouette = silhouette_score(
                        D_valid,
                        valid_labels,
                        metric="precomputed",
                    )
                except ValueError:
                    silhouette = np.nan

            rows.append(
                {
                    "eps": round(float(eps), 3),
                    "min_samples": min_samples,
                    "Clusters": n_clusters,
                    "Noise_n": n_noise,
                    "Noise_%": noise_percent,
                    "Silhouette": silhouette,
                }
            )

    results = pd.DataFrame(rows)
    valid = results.dropna(
        subset=["Silhouette"]
    ).copy()

    print("\n" + "=" * 78)
    print("NUMBER OF CONFIGURATIONS TESTED")
    print("=" * 78)
    print(len(results))

    print("\n" + "=" * 78)
    print("VALID CONFIGURATIONS")
    print("=" * 78)
    print(len(valid))

    top20 = (
        valid
        .sort_values(
            ["Silhouette", "Noise_%"],
            ascending=[False, True],
        )
        .head(20)
    )

    print("\n" + "=" * 78)
    print("TOP 20 CONFIGURATIONS BY SILHOUETTE")
    print("=" * 78)

    print(
        top20.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    for threshold in [20, 10]:
        subset = valid[
            valid["Noise_%"] <= threshold
        ].copy()

        print("\n" + "=" * 78)
        print(
            f"BEST CONFIGURATIONS WITH <= {threshold}% NOISE"
        )
        print("=" * 78)

        if len(subset) > 0:
            best_subset = (
                subset
                .sort_values(
                    ["Silhouette", "Noise_%"],
                    ascending=[False, True],
                )
                .head(15)
            )

            print(
                best_subset.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        else:
            print(
                f"No valid configuration with <= {threshold}% noise."
            )

    if len(valid) > 0:
        best_global = valid.loc[
            valid["Silhouette"].idxmax()
        ]

        print("\n" + "=" * 78)
        print("BEST GLOBAL DBSCAN CONFIGURATION")
        print("=" * 78)

        print(
            f"eps             : {best_global['eps']:.3f}"
        )
        print(
            f"min_samples     : "
            f"{int(best_global['min_samples'])}"
        )
        print(
            f"Clusters        : "
            f"{int(best_global['Clusters'])}"
        )
        print(
            f"Noise           : "
            f"{int(best_global['Noise_n'])} "
            f"({best_global['Noise_%']:.2f}%)"
        )
        print(
            f"Silhouette      : "
            f"{best_global['Silhouette']:.4f}"
        )
    else:
        print(
            "\nNo valid DBSCAN configuration was found."
        )


if __name__ == "__main__":
    main()
