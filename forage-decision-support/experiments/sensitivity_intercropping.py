"""Sensitivity of intercropping rankings to component-weight perturbations."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess
from src.scoring import calcular_score_normalizado, e_especie_perene_pastejavel


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"
FIGURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "weight_sensitivity.png"
)


def prepare_reference_scenario(df):
    """Score and filter species under the reference Cerrado scenario."""
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
    """Select grasses at the 95% threshold and eligible legumes."""
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


def generate_intercropping_ranking(
    grasses,
    legumes,
    gower_matrix,
    w_g=0.40,
    w_l=0.40,
    w_root=1.0,
    w_stratum=1.0,
    w_gower=1.0,
):
    """Generate an intercropping ranking with parameterized component weights."""
    results = []

    for _, grass in grasses.iterrows():
        for _, legume in legumes.iterrows():

            base_score = (
                w_g * grass["score"]
                + w_l * legume["score"]
            )

            root_difference = abs(
                grass["profundidade_raiz_max"]
                - legume["profundidade_raiz_max"]
            )

            root_component = min(
                root_difference / 10.0,
                10.0,
            )

            stratum_difference = abs(
                grass["estrato_num"]
                - legume["estrato_num"]
            )

            stratum_component = (
                10.0
                if stratum_difference >= 1
                else 3.0
            )

            d_gower_pair = gower_matrix[
                int(grass["pos_matriz"]),
                int(legume["pos_matriz"]),
            ]

            gower_component = round(
                10.0 * (1.0 - d_gower_pair),
                2,
            )

            score = round(
                base_score
                + w_root * root_component
                + w_stratum * stratum_component
                + w_gower * gower_component,
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


def compare_rankings(base, test):
    """Compare one sensitivity scenario with the baseline ranking."""
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

    baseline = generate_intercropping_ranking(
        grasses_95,
        legumes,
        gower_matrix,
        w_g=0.40,
        w_l=0.40,
        w_root=1.0,
        w_stratum=1.0,
        w_gower=1.0,
    )

    scenarios = {
        "More individual suitability": {
            "w_g": 0.45,
            "w_l": 0.45,
            "w_root": 0.8,
            "w_stratum": 0.8,
            "w_gower": 0.8,
        },
        "More functional complementarity": {
            "w_g": 0.35,
            "w_l": 0.35,
            "w_root": 1.2,
            "w_stratum": 1.2,
            "w_gower": 1.0,
        },
        "More Gower compatibility": {
            "w_g": 0.40,
            "w_l": 0.40,
            "w_root": 1.0,
            "w_stratum": 1.0,
            "w_gower": 1.2,
        },
        "Less Gower compatibility": {
            "w_g": 0.40,
            "w_l": 0.40,
            "w_root": 1.0,
            "w_stratum": 1.0,
            "w_gower": 0.8,
        },
    }

    rows = []
    rankings = {}

    for name, params in scenarios.items():
        test_ranking = generate_intercropping_ranking(
            grasses_95,
            legumes,
            gower_matrix,
            **params,
        )

        rankings[name] = test_ranking

        rho, p_value, overlap = compare_rankings(
            baseline,
            test_ranking,
        )

        rows.append(
            {
                "Scenario": name,
                "Spearman": rho,
                "p-value": p_value,
                "Top10": overlap,
                "Top10 (%)": overlap * 10,
            }
        )

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print("INTERCROPPING WEIGHT SENSITIVITY")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Spearman": lambda x: f"{x:.4f}",
                "p-value": lambda x: f"{x:.4e}",
                "Top10 (%)": lambda x: f"{x:.0f}%",
            },
        )
    )

    plt.figure(figsize=(8, 5))

    y_pos = np.arange(
        len(results_df)
    )

    plt.barh(
        y_pos,
        results_df["Spearman"],
    )

    plt.yticks(
        y_pos,
        results_df["Scenario"],
    )

    plt.xlabel(
        "Spearman rank correlation",
        fontsize=12,
    )

    plt.title(
        "Sensitivity of Intercropping Rankings to Weight Perturbations",
        fontsize=13,
    )

    min_rho = results_df[
        "Spearman"
    ].min()

    plt.xlim(
        max(0, min_rho - 0.03),
        1.01,
    )

    for i, value in enumerate(
        results_df["Spearman"]
    ):
        plt.text(
            value + 0.001,
            i,
            f"{value:.4f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
