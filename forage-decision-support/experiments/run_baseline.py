"""Run the reference Cerrado scenario reported in the manuscript."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import build_gower_matrix, run_hac
from src.preprocessing import load_and_preprocess
from src.recommendation import recommend


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    df = load_and_preprocess(DATA_PATH)
    matrix, _ = build_gower_matrix(df)

    clustering = run_hac(matrix)
    df["cluster_ia"] = clustering["best_labels"]

    result = recommend(
        df,
        matrix,
        ph=5.0,
        clima="cerrado",
        text="argiloso",
        fert="media",
    )

    print(
        f"HAC: k={clustering['best_k']} | "
        f"Silhouette={clustering['best_silhouette']:.4f}"
    )
    print("\nTOP INDIVIDUAL")
    print(result["top_individual"].to_string(index=False))
    print("\nTOP INTERCROPPING")
    print(result["intercropping"].to_string(index=False))


if __name__ == "__main__":
    main()
