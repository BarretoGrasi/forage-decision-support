"""Diagnostic summary of root-depth values before and after model imputation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocessing import (
    NUMERIC_COLUMNS,
    SOIL_FEATURES,
    limpar_num,
    normalizar_coluna,
)


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [normalizar_coluna(c) for c in df.columns]

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(limpar_num)

    root = df["profundidade_raiz_max"].copy()
    median_root = root.median()

    print("=" * 65)
    print("ROOT DEPTH — BEFORE IMPUTATION")
    print("=" * 65)
    print(f"Total species: {len(root)}")
    print(f"Available values: {root.notna().sum()}")
    print(f"Missing values: {root.isna().sum()}")
    print(f"Median: {median_root:.2f}")
    print(root.dropna().describe())
    print(f"Unique original values: {root.dropna().nunique()}")

    for col in SOIL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print("\n" + "=" * 65)
    print("ROOT DEPTH — AFTER IMPUTATION")
    print("=" * 65)
    print(df["profundidade_raiz_max"].describe())
    print(
        "Unique values after imputation:",
        df["profundidade_raiz_max"].nunique(),
    )


if __name__ == "__main__":
    main()
