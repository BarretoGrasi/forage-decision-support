"""Grass-legume pairwise intercropping scoring."""

def score_consorcio_avancado(g, l, d_gower_pair):
    """Compute the composite intercropping ranking score."""
    base_score = 0.40 * g["score"] + 0.40 * l["score"]

    diff_raiz = abs(
        g["profundidade_raiz_max"] - l["profundidade_raiz_max"]
    )
    comp_raiz = min(diff_raiz / 10.0, 10.0)

    diff_estrato = abs(g["estrato_num"] - l["estrato_num"])
    comp_estrato = 10.0 if diff_estrato >= 1 else 3.0

    compat_gower_multivariada = round(
        10.0 * (1.0 - d_gower_pair),
        2,
    )

    return round(
        base_score
        + comp_raiz
        + comp_estrato
        + compat_gower_multivariada,
        2,
    )
