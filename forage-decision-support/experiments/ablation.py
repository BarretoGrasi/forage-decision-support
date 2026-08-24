"""Ablation analysis of the grass-legume intercropping recommendation model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess
from src.scoring import calcular_score_normalizado, e_especie_perene_pastejavel


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def prepare_reference_scenario(df):
    """Score species under the reference Cerrado scenario."""
    work = df.copy()

    results = work.apply(
        lambda r: calcular_score_normalizado(
            r,
            ph=5.0,
            clima="cerrado",
            text="argiloso",
            fert="media",
            w_eco=0.60,
            w_prod=0.40,
        ),
        axis=1,
    )

    work["score"] = [r[0] for r in results]

    work = work[
        work.apply(e_especie_perene_pastejavel, axis=1)
    ].copy()

    return work


def select_candidates(df):
    """Apply the 95% candidate-grass threshold and select eligible legumes."""
    grasses = df[
        df["grupo"].astype(str).str.contains("gramin", case=False, na=False)
    ].copy()

    grasses = grasses[grasses["score"] > 0].copy()

    best_score = grasses["score"].max()
    threshold_95 = 0.95 * best_score

    grasses_95 = grasses[
        grasses["score"] >= threshold_95
    ].copy()

    legumes = df[
        df["grupo"].astype(str).str.contains("legumin", case=False, na=False)
    ].copy()

    legumes = legumes[
        legumes["score"] > 0
    ].copy()

    return grasses_95, legumes


def generate_ablation_ranking(
    grasses,
    legumes,
    gower_matrix,
    use_gower=True,
    use_root=True,
    use_stratum=True,
):
    """Generate an intercropping ranking under one ablation configuration."""
    results = []

    for _, grass in grasses.iterrows():
        for _, legume in legumes.iterrows():

            base_score = (
                0.40 * grass["score"]
                + 0.40 * legume["score"]
            )

            root_difference = abs(
                grass["profundidade_raiz_max"]
                - legume["profundidade_raiz_max"]
            )
            root_component = min(
                root_difference / 10.0,
                10.0,
            )
            if not use_root:
                root_component = 0.0

            stratum_difference = abs(
                grass["estrato_num"]
                - legume["estrato_num"]
            )
            stratum_component = (
                10.0 if stratum_difference >= 1 else 3.0
            )
            if not use_stratum:
                stratum_component = 0.0

            d_gower_pair = gower_matrix[
                int(grass["pos_matriz"]),
                int(legume["pos_matriz"]),
            ]

            gower_component = round(
                10.0 * (1.0 - d_gower_pair),
                2,
            )
            if not use_gower:
                gower_component = 0.0

            score = round(
                base_score
                + root_component
                + stratum_component
                + gower_component,
                2,
            )

            results.append(
                {
                    "grass": grass["especie"],
                    "legume": legume["especie"],
                    "score": score,
                }
            )

    ranking = pd.DataFrame(results)

    ranking["pair"] = (
        ranking["grass"].astype(str)
        + " + "
        + ranking["legume"].astype(str)
    )

    ranking = ranking.sort_values(
        ["score", "pair"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1,
    )

    return ranking


def compare_ablation(base, test):
    """Compare one reduced ranking against the complete baseline ranking."""
    comp = base[
        ["pair", "rank"]
    ].merge(
        test[["pair", "rank"]],
        on="pair",
        suffixes=("_base", "_test"),
    )

    rho, p_value = spearmanr(
        comp["rank_base"],
        comp["rank_test"],
    )

    top10_base = set(
        base.head(10)["pair"]
    )

    top10_test = set(
        test.head(10)["pair"]
    )

    overlap = len(
        top10_base.intersection(top10_test)
    )

    return rho, p_value, overlap


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    df = prepare_reference_scenario(df)
    grasses_95, legumes = select_candidates(df)

    complete = generate_ablation_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        use_gower=True,
        use_root=True,
        use_stratum=True,
    )

    without_gower = generate_ablation_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        use_gower=False,
        use_root=True,
        use_stratum=True,
    )

    without_root = generate_ablation_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        use_gower=True,
        use_root=False,
        use_stratum=True,
    )

    without_stratum = generate_ablation_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        use_gower=True,
        use_root=True,
        use_stratum=False,
    )

    individual_only = generate_ablation_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        use_gower=False,
        use_root=False,
        use_stratum=False,
    )

    scenarios = {
        "Without Gower compatibility": without_gower,
        "Without root complementarity": without_root,
        "Without canopy stratum": without_stratum,
        "Individual scores only": individual_only,
    }

    rows = []

    for name, test in scenarios.items():
        rho, p_value, overlap = compare_ablation(
            complete,
            test,
        )

        rows.append(
            {
                "Configuration": name,
                "Spearman rho": rho,
                "p-value": p_value,
                "Top-10 overlap": overlap,
                "Top-10 overlap (%)": overlap * 10,
            }
        )

    df_ablation = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print("ABLATION ANALYSIS OF THE INTERCROPPING RECOMMENDATION FRAMEWORK")
    print("=" * 75)

    print(
        df_ablation[
            [
                "Configuration",
                "Spearman rho",
                "Top-10 overlap (%)",
            ]
        ].to_string(
            index=False,
            formatters={
                "Spearman rho": lambda x: f"{x:.4f}",
                "Top-10 overlap (%)": lambda x: f"{x:.0f}%",
            },
        )
    )

    print("\n--- TOP-10 OF THE COMPLETE MODEL ---")

    print(
        complete.head(10)[
            [
                "grass",
                "legume",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
