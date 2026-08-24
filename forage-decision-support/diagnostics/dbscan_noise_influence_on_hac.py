"""Assess the influence of DBSCAN noise observations on HAC structure.

This diagnostic compares HAC on all observations, HAC restricted to the
DBSCAN non-noise subset without refitting, HAC refitted on that subset, and
DBSCAN on the same subset. It also reports ARI agreement and identifies the
noise observations within the original HAC partition.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    D = np.asarray(gower_matrix, dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    print("=" * 80)
    print("INFLUENCE OF DBSCAN-NOISE OBSERVATIONS ON HAC")
    print("=" * 80)
    print(f"\nNumber of observations in Gower matrix: {D.shape[0]}")

    # --------------------------------------------------------
    # 1. Fixed DBSCAN configuration
    # --------------------------------------------------------
    dbscan = DBSCAN(
        eps=0.100,
        min_samples=3,
        metric="precomputed",
    )

    labels_dbscan = dbscan.fit_predict(D)

    noise_mask = labels_dbscan == -1
    non_noise_mask = labels_dbscan != -1

    print("\n" + "=" * 80)
    print("DBSCAN")
    print("=" * 80)
    print("Non-noise observations:", int(non_noise_mask.sum()))
    print("Noise observations:", int(noise_mask.sum()))

    # --------------------------------------------------------
    # 2. HAC on all observations
    # --------------------------------------------------------
    hac_all = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average",
    )

    labels_hac_all = hac_all.fit_predict(D)

    sil_hac_all = silhouette_score(
        D,
        labels_hac_all,
        metric="precomputed",
    )

    print("\n" + "=" * 80)
    print("HAC — ALL OBSERVATIONS")
    print("=" * 80)
    print(f"Silhouette = {sil_hac_all:.4f}")

    print("\nCluster sizes:")
    unique, counts = np.unique(labels_hac_all, return_counts=True)
    for cluster, n in zip(unique, counts):
        print(f"Cluster {cluster}: {n}")

    # --------------------------------------------------------
    # 3. Restrict to DBSCAN non-noise observations
    # --------------------------------------------------------
    D_sub = D[np.ix_(non_noise_mask, non_noise_mask)]

    labels_hac_original_sub = labels_hac_all[
        non_noise_mask
    ]

    # --------------------------------------------------------
    # 4. Original HAC labels restricted to the subset
    # --------------------------------------------------------
    sil_hac_original_sub = silhouette_score(
        D_sub,
        labels_hac_original_sub,
        metric="precomputed",
    )

    print("\n" + "=" * 80)
    print("ORIGINAL HAC RESTRICTED TO DBSCAN NON-NOISE OBSERVATIONS")
    print("=" * 80)
    print(f"Silhouette = {sil_hac_original_sub:.4f}")

    # --------------------------------------------------------
    # 5. Refit HAC on the same subset
    # --------------------------------------------------------
    hac_sub = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average",
    )

    labels_hac_sub = hac_sub.fit_predict(D_sub)

    sil_hac_sub = silhouette_score(
        D_sub,
        labels_hac_sub,
        metric="precomputed",
    )

    print("\n" + "=" * 80)
    print("HAC REFITTED ON DBSCAN NON-NOISE OBSERVATIONS")
    print("=" * 80)
    print(f"Silhouette = {sil_hac_sub:.4f}")

    # --------------------------------------------------------
    # 6. DBSCAN silhouette on the same subset
    # --------------------------------------------------------
    labels_dbscan_sub = labels_dbscan[
        non_noise_mask
    ]

    sil_dbscan = silhouette_score(
        D_sub,
        labels_dbscan_sub,
        metric="precomputed",
    )

    print("\n" + "=" * 80)
    print("DBSCAN — SAME NON-NOISE OBSERVATIONS")
    print("=" * 80)
    print(f"Silhouette = {sil_dbscan:.4f}")

    # --------------------------------------------------------
    # 7. ARI agreement
    # --------------------------------------------------------
    ari = adjusted_rand_score(
        labels_dbscan_sub,
        labels_hac_sub,
    )

    print("\n" + "=" * 80)
    print("HAC vs DBSCAN AGREEMENT")
    print("=" * 80)
    print(f"Adjusted Rand Index (ARI) = {ari:.4f}")

    # --------------------------------------------------------
    # 8. Noise observations in original HAC
    # --------------------------------------------------------
    columns = [
        "especie",
        "grupo",
        "ph_min",
        "ph_max",
        "text_num",
        "fert_num",
        "profundidade_raiz_max",
        "altura_media",
        "produtividade_media",
    ]

    columns = [c for c in columns if c in df.columns]

    noise_df = df.loc[
        noise_mask,
        columns,
    ].copy()

    noise_df["cluster_HAC_all"] = labels_hac_all[
        noise_mask
    ]

    print("\n" + "=" * 80)
    print("DBSCAN-NOISE OBSERVATIONS IN ORIGINAL HAC")
    print("=" * 80)

    print(
        noise_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)

    print(
        f"HAC — all observations                  : "
        f"{sil_hac_all:.4f}"
    )
    print(
        f"Original HAC restricted to non-noise    : "
        f"{sil_hac_original_sub:.4f}"
    )
    print(
        f"HAC refitted on non-noise observations  : "
        f"{sil_hac_sub:.4f}"
    )
    print(
        f"DBSCAN on the same observations         : "
        f"{sil_dbscan:.4f}"
    )
    print(
        f"ARI HAC vs DBSCAN                       : "
        f"{ari:.4f}"
    )


if __name__ == "__main__":
    main()
