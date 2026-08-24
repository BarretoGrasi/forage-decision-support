"""Run the three edaphoclimatic scenarios evaluated in the manuscript."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess
from src.recommendation import recommend


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"

SCENARIOS = {
    "baseline_cerrado": {
        "ph": 5.0,
        "clima": "cerrado",
        "text": "argiloso",
        "fert": "media",
    },
    "acidic_tropical": {
        "ph": 4.5,
        "clima": "tropical",
        "text": "arenoso",
        "fert": "baixa",
    },
    "subtropical": {
        "ph": 6.0,
        "clima": "subtropical",
        "text": "arenoso",
        "fert": "baixa",
    },
}


def main():
    df = load_and_preprocess(DATA_PATH)
    matrix, _ = build_gower_matrix(df)

    for name, params in SCENARIOS.items():
        result = recommend(df, matrix, **params)

        print("\n" + "=" * 75)
        print(name)
        print(params)
        print("=" * 75)
        print("\nTOP INDIVIDUAL")
        print(result["top_individual"].to_string(index=False))
        print("\nTOP INTERCROPPING")
        print(result["intercropping"].to_string(index=False))


if __name__ == "__main__":
    main()
