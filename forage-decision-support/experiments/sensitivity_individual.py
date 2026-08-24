"""Sensitivity of individual rankings to ecological-productivity weights."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.preprocessing import load_and_preprocess
from src.scoring import calcular_score_normalizado, e_especie_perene_pastejavel


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"

WEIGHT_SCENARIOS = {
    "70/30": (0.70, 0.30),
    "60/40": (0.60, 0.40),
    "50/50": (0.50, 0.50),
}


def generate_ranking(df, w_eco, w_prod):
    work = df.copy()

    work["score_teste"] = work.apply(
        lambda r: calcular_score_normalizado(
            r,
            ph=5.0,
            clima="cerrado",
            text="argiloso",
            fert="media",
            w_eco=w_eco,
            w_prod=w_prod,
        )[0],
        axis=1,
    )

    work = work[
        work.apply(e_especie_perene_pastejavel, axis=1)
    ].copy()

    ranking = (
        work[["especie", "grupo", "score_teste"]]
        .sort_values("score_teste", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(len(ranking)) + 1
    return ranking


def main():
    df = load_and_preprocess(DATA_PATH)

    rankings = {
        name: generate_ranking(df, w_eco, w_prod)
        for name, (w_eco, w_prod) in WEIGHT_SCENARIOS.items()
    }

    baseline = rankings["60/40"]
    rows = []

    for name, ranking in rankings.items():
        comp = baseline[["especie", "rank"]].merge(
            ranking[["especie", "rank"]],
            on="especie",
            suffixes=("_base", "_teste"),
        )

        rho, p_value = spearmanr(
            comp["rank_base"],
            comp["rank_teste"],
        )

        overlap = len(
            set(baseline.head(10)["especie"])
            & set(ranking.head(10)["especie"])
        )

        rows.append(
            {
                "Scenario": name,
                "Ecological weight": WEIGHT_SCENARIOS[name][0],
                "Productivity weight": WEIGHT_SCENARIOS[name][1],
                "Spearman rho": rho,
                "p-value": p_value,
                "Top-10 overlap": overlap,
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))

    for name, ranking in rankings.items():
        print(f"\n--- TOP 10 — {name} ---")
        print(
            ranking.head(10)[
                ["rank", "especie", "grupo", "score_teste"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
