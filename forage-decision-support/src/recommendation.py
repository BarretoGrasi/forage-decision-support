"""End-to-end individual and grass-legume recommendation workflow."""

import pandas as pd

from .intercropping import score_consorcio_avancado
from .scoring import calcular_score_normalizado, e_especie_perene_pastejavel


CONTRIBUTION_COLUMNS = {
    "ph": "contrib_pH",
    "textura": "contrib_Textura",
    "fertilidade": "contrib_Fertilidade",
    "uso": "contrib_Uso",
    "raiz": "contrib_Raiz",
    "produtividade": "contrib_Prod",
}


def recommend(
    df,
    gower_matrix,
    ph=5.0,
    clima="cerrado",
    text="argiloso",
    fert="media",
    candidate_threshold=0.95,
    w_eco=0.60,
    w_prod=0.40,
    top_n=10,
):
    """Generate individual and intercropping rankings for one scenario."""
    work = df.copy()

    results = work.apply(
        lambda r: calcular_score_normalizado(
            r,
            ph,
            clima,
            text,
            fert,
            w_eco=w_eco,
            w_prod=w_prod,
        ),
        axis=1,
    )

    work["score"] = [r[0] for r in results]

    for key, column in CONTRIBUTION_COLUMNS.items():
        work[column] = [r[1].get(key, 0.0) for r in results]

    eligible = work[
        work.apply(e_especie_perene_pastejavel, axis=1)
    ].copy()

    individual_columns = [
        "especie",
        "grupo",
        "score",
        "contrib_pH",
        "contrib_Textura",
        "contrib_Fertilidade",
        "contrib_Uso",
        "contrib_Raiz",
        "contrib_Prod",
    ]

    top_individual = (
        eligible.sort_values("score", ascending=False)
        .head(top_n)[individual_columns]
        .copy()
    )

    grasses = eligible[
        eligible["grupo"].str.contains("gramin", na=False)
    ].sort_values("score", ascending=False)

    if grasses.empty:
        return {
            "all_scored": work,
            "eligible": eligible,
            "top_individual": top_individual,
            "candidate_grasses": grasses,
            "intercropping": pd.DataFrame(),
        }

    max_grass_score = grasses.iloc[0]["score"]
    candidate_grasses = grasses[
        grasses["score"] >= candidate_threshold * max_grass_score
    ].copy()

    legumes = eligible[
        eligible["grupo"].str.contains("legumin", na=False)
    ]

    pairs = []
    for _, grass in candidate_grasses.iterrows():
        for _, legume in legumes.iterrows():
            if legume["score"] == 0:
                continue

            pos_g = int(grass["pos_matriz"])
            pos_l = int(legume["pos_matriz"])
            d_gower_pair = gower_matrix[pos_g, pos_l]

            pairs.append(
                {
                    "Gramínea (Principal)": grass["especie"],
                    "Leguminosa (Acompanhante)": legume["especie"],
                    "Score Consórcio": score_consorcio_avancado(
                        grass,
                        legume,
                        d_gower_pair,
                    ),
                }
            )

    intercropping = (
        pd.DataFrame(pairs)
        .sort_values("Score Consórcio", ascending=False)
        .head(top_n)
        if pairs
        else pd.DataFrame(
            columns=[
                "Gramínea (Principal)",
                "Leguminosa (Acompanhante)",
                "Score Consórcio",
            ]
        )
    )

    return {
        "all_scored": work,
        "eligible": eligible,
        "top_individual": top_individual,
        "candidate_grasses": candidate_grasses,
        "intercropping": intercropping,
    }
