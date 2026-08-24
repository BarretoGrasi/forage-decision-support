"""Candidate-grass threshold analysis for intercropping recommendations.

Evaluates 90%, 95% (baseline), and 100% thresholds relative to the highest
individual grass suitability score under the reference environmental scenario.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.clustering import build_gower_matrix
from src.preprocessing import load_and_preprocess
from src.scoring import calcular_score_normalizado, e_especie_perene_pastejavel


DATA_PATH = PROJECT_ROOT / "data" / "forage_dataset.csv"


def prepare_reference_scenario(df):
    """Score and retain eligible perennial forage species for the baseline scenario."""
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


def select_grasses_and_legumes(df):
    """Return all eligible grasses and legumes with positive individual scores."""
    grasses = df[
        df["grupo"].astype(str).str.contains("gramin", case=False, na=False)
    ].copy()

    grasses = grasses[
        grasses["score"] > 0
    ].copy()

    legumes = df[
        df["grupo"].astype(str).str.contains("legumin", case=False, na=False)
    ].copy()

    legumes = legumes[
        legumes["score"] > 0
    ].copy()

    return grasses, legumes


def generate_threshold_ranking(
    threshold,
    grasses,
    legumes,
    gower_matrix,
):
    """Generate the intercropping ranking for one candidate-grass threshold."""
    best_grass_score = grasses["score"].max()
    limit = threshold * best_grass_score

    selected_grasses = grasses[
        grasses["score"] >= limit
    ].copy()

    results = []

    for _, grass in selected_grasses.iterrows():
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

            final_score = round(
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
                    "score": final_score,
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

    return ranking, selected_grasses


def calculate_overlap(baseline_ranking, test_ranking):
    """Calculate Top-10 overlap relative to the 95% baseline ranking."""
    top10_baseline = set(
        baseline_ranking.head(10)["pair"]
    )

    top10_test = set(
        test_ranking.head(10)["pair"]
    )

    overlap = len(
        top10_baseline.intersection(top10_test)
    )

    percentage = overlap / 10 * 100

    return overlap, percentage


def main():
    df = load_and_preprocess(DATA_PATH)
    gower_matrix, _ = build_gower_matrix(df)

    df = prepare_reference_scenario(df)
    grasses, legumes = select_grasses_and_legumes(df)

    best_grass_score = grasses["score"].max()

    print("=" * 75)
    print("CANDIDATE-GRASS THRESHOLD ANALYSIS")
    print("=" * 75)

    print(
        f"\nHighest individual grass score: "
        f"{best_grass_score:.4f}"
    )

    best_grasses = grasses[
        np.isclose(
            grasses["score"],
            best_grass_score,
        )
    ][["especie", "score"]]

    print("\nGrass(es) with maximum individual score:")
    print(
        best_grasses.to_string(
            index=False,
        )
    )

    print(
        f"\nNumber of eligible legumes: "
        f"{len(legumes)}"
    )

    ranking_90, grasses_90 = generate_threshold_ranking(
        0.90,
        grasses,
        legumes,
        gower_matrix,
    )

    ranking_95, grasses_95 = generate_threshold_ranking(
        0.95,
        grasses,
        legumes,
        gower_matrix,
    )

    ranking_100, grasses_100 = generate_threshold_ranking(
        1.00,
        grasses,
        legumes,
        gower_matrix,
    )

    overlap_90, percentage_90 = calculate_overlap(
        ranking_95,
        ranking_90,
    )

    overlap_95, percentage_95 = calculate_overlap(
        ranking_95,
        ranking_95,
    )

    overlap_100, percentage_100 = calculate_overlap(
        ranking_95,
        ranking_100,
    )

    summary = pd.DataFrame(
        {
            "Threshold": [
                "90%",
                "95% (baseline)",
                "100%",
            ],
            "Eligible grasses": [
                len(grasses_90),
                len(grasses_95),
                len(grasses_100),
            ],
            "Candidate pairs": [
                len(ranking_90),
                len(ranking_95),
                len(ranking_100),
            ],
            "Top-10 overlap": [
                overlap_90,
                overlap_95,
                overlap_100,
            ],
            "Top-10 overlap (%)": [
                percentage_90,
                percentage_95,
                percentage_100,
            ],
        }
    )

    print("\n" + "=" * 75)
    print("THRESHOLD ANALYSIS SUMMARY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            formatters={
                "Top-10 overlap (%)":
                    lambda x: f"{x:.0f}%"
            },
        )
    )

    for name, selected in [
        ("90%", grasses_90),
        ("95% (baseline)", grasses_95),
        ("100%", grasses_100),
    ]:
        print("\n" + "-" * 75)
        print(f"ELIGIBLE GRASSES — {name}")
        print("-" * 75)

        print(
            selected[
                ["especie", "score"]
            ]
            .sort_values(
                "score",
                ascending=False,
            )
            .to_string(
                index=False,
            )
        )

    for name, ranking in [
        ("90%", ranking_90),
        ("95% BASELINE", ranking_95),
        ("100%", ranking_100),
    ]:
        print("\n" + "=" * 75)
        print(f"TOP-10 INTERCROPPING — THRESHOLD {name}")
        print("=" * 75)

        print(
            ranking.head(10)[
                [
                    "grass",
                    "legume",
                    "score",
                    "rank",
                ]
            ].to_string(
                index=False,
            )
        )


if __name__ == "__main__":
    main()
