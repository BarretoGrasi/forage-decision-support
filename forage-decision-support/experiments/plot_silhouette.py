"""Generate the Gower-HAC silhouette figure reported in the manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "figures"
    / "gower_based_silhouette_coefficient.png"
)


def main():
    k_values = [2, 3, 4, 5, 6]

    silhouette_values = [
        0.5139,
        0.3015,
        0.3179,
        0.3416,
        0.2938,
    ]

    best_index = np.argmax(silhouette_values)
    best_k = k_values[best_index]
    best_silhouette = silhouette_values[best_index]

    plt.figure(figsize=(7, 5))

    plt.plot(
        k_values,
        silhouette_values,
        marker="o",
        linewidth=2,
        markersize=7,
    )

    plt.scatter(
        best_k,
        best_silhouette,
        s=90,
        zorder=3,
    )

    plt.annotate(
        f"Optimal k = {best_k}\nSilhouette = {best_silhouette:.4f}",
        xy=(best_k, best_silhouette),
        xytext=(3.0, 0.47),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
        fontsize=10,
    )

    plt.xlabel("Number of clusters (k)", fontsize=12)
    plt.ylabel("Silhouette coefficient", fontsize=12)
    plt.title("Gower-based Silhouette Coefficient", fontsize=13)

    plt.xticks(k_values)
    plt.ylim(0.25, 0.55)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
