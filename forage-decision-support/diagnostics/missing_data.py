"""Missing-data diagnostic for numerical variables used by the framework."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocessing import load_and_preprocess


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    # This diagnostic uses the standardized dataset produced by preprocessing.
    df = load_and_preprocess(DATA_PATH)

    variables = [
        "profundidade_raiz_max",
        "altura_media",
        "ph_min",
        "ph_max",
        "text_num",
        "fert_num",
        "produtividade_media",
    ]

    rows = []
    for col in variables:
        total = len(df)
        missing = df[col].isna().sum() if col in df.columns else None
        present = df[col].notna().sum() if col in df.columns else None
        rows.append(
            {
                "Variable": col,
                "Total": total,
                "Present": present,
                "Missing": missing,
                "Missing (%)": (
                    round(100 * missing / total, 2)
                    if missing is not None
                    else None
                ),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
