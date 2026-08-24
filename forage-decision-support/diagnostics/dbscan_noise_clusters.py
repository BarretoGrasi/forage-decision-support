"""Diagnostic analysis of DBSCAN noise points and small clusters.

This script identifies observations classified as DBSCAN noise, reports the
smallest non-noise DBSCAN cluster, and compares it with the smallest HAC
cluster obtained from the same non-noise subset.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


DISPLAY_COLUMNS = [
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


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    D = np.asarray(gower_matrix, dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    # --------------------------------------------------------
    # 1. Fixed DBSCAN configuration
    # --------------------------------------------------------
    dbscan = DBSCAN(
        eps=0.100,
        min_samples=3,
        metric="precomputed",
    )

    labels_db = dbscan.fit_predict(D)

    df_diag = df.copy()
    df_diag["cluster_dbscan"] = labels_db

    # --------------------------------------------------------
    # 2. Noise observations
    # --------------------------------------------------------
    noise = df_diag[
        df_diag["cluster_dbscan"] == -1
    ].copy()

    print("=" * 80)
    print("OBSERVATIONS CLASSIFIED AS DBSCAN NOISE")
    print("=" * 80)

    print("Number of observations:", len(noise))

    print(
        noise[
            [c for c in DISPLAY_COLUMNS if c in noise.columns]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 3. DBSCAN cluster distribution
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("DBSCAN CLUSTER SIZES")
    print("=" * 80)

    print(
        df_diag["cluster_dbscan"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 4. Smallest non-noise DBSCAN cluster
    # --------------------------------------------------------
    valid_clusters = df_diag[
        df_diag["cluster_dbscan"] != -1
    ]

    cluster_counts = (
        valid_clusters["cluster_dbscan"]
        .value_counts()
    )

    smallest_cluster_id = cluster_counts.idxmin()

    smallest_cluster = df_diag[
        df_diag["cluster_dbscan"] == smallest_cluster_id
    ].copy()

    print("\n" + "=" * 80)
    print("SMALLEST DBSCAN CLUSTER")
    print("=" * 80)

    print(f"Cluster: {smallest_cluster_id}")
    print(f"Number of observations: {len(smallest_cluster)}")

    print(
        smallest_cluster[
            [c for c in DISPLAY_COLUMNS if c in smallest_cluster.columns]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. HAC on the same non-noise observations
    # --------------------------------------------------------
    mask = labels_db != -1
    D_sub = D[np.ix_(mask, mask)]

    hac = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average",
    )

    labels_hac = hac.fit_predict(D_sub)

    df_hac_sub = df_diag.loc[mask].copy()
    df_hac_sub["cluster_hac"] = labels_hac

    # --------------------------------------------------------
    # 6. HAC cluster sizes
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("HAC CLUSTER SIZES — SAME NON-NOISE OBSERVATIONS")
    print("=" * 80)

    print(
        df_hac_sub["cluster_hac"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 7. Smallest HAC cluster
    # --------------------------------------------------------
    smallest_hac_id = (
        df_hac_sub["cluster_hac"]
        .value_counts()
        .idxmin()
    )

    smallest_hac = df_hac_sub[
        df_hac_sub["cluster_hac"] == smallest_hac_id
    ].copy()

    print("\n" + "=" * 80)
    print("SMALLEST HAC CLUSTER")
    print("=" * 80)

    print(f"Cluster: {smallest_hac_id}")
    print(f"Number of observations: {len(smallest_hac)}")

    print(
        smallest_hac[
            [c for c in DISPLAY_COLUMNS if c in smallest_hac.columns]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
