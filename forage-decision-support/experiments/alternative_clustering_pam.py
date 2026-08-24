"""PAM / K-Medoids evaluation using the precomputed Gower distance matrix."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def pam_kmedoids(distance_matrix, k, max_iter=300, random_state=42):
    """Simple PAM/K-Medoids implementation for a precomputed distance matrix."""
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, size=k, replace=False)

    for _ in range(max_iter):
        distances_to_medoids = D[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)

        new_medoids = medoids.copy()

        for cluster in range(k):
            members = np.where(labels == cluster)[0]

            if len(members) == 0:
                continue

            cluster_distances = D[np.ix_(members, members)]
            total_distance = cluster_distances.sum(axis=1)

            best_member = members[np.argmin(total_distance)]
            new_medoids[cluster] = best_member

        if np.array_equal(
            np.sort(new_medoids),
            np.sort(medoids),
        ):
            medoids = new_medoids
            break

        medoids = new_medoids

    labels = np.argmin(
        D[:, medoids],
        axis=1,
    )

    return labels, medoids


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    rows = []

    print("=" * 70)
    print("PAM / K-MEDOIDS + GOWER")
    print("=" * 70)

    for k in range(2, 7):
        labels, medoids = pam_kmedoids(
            gower_matrix,
            k=k,
            random_state=42,
        )

        silhouette = silhouette_score(
            gower_matrix,
            labels,
            metric="precomputed",
        )

        rows.append(
            {
                "Method": "PAM + Gower",
                "k": k,
                "Silhouette": silhouette,
                "Medoids": ",".join(map(str, medoids)),
            }
        )

        print(
            f"k = {k} | "
            f"Silhouette = {silhouette:.4f}"
        )

    results = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results[
            ["Method", "k", "Silhouette"]
        ].to_string(
            index=False,
            formatters={
                "Silhouette": lambda x: f"{x:.4f}"
            },
        )
    )

    best = results.loc[
        results["Silhouette"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST PAM CONFIGURATION")
    print("=" * 70)

    print(f"Optimal k     : {int(best['k'])}")
    print(f"Silhouette    : {best['Silhouette']:.4f}")
    print(f"Medoid indices: {best['Medoids']}")


if __name__ == "__main__":
    main()
