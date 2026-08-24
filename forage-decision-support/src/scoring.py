"""Eligibility rules and individual forage suitability scoring."""

import numpy as np

from .preprocessing import limpar_texto, mapear_fertilidade, mapear_textura


def categorizar_ciclo(val):
    txt = limpar_texto(val)
    if any(
        p in txt
        for p in ["peren", "pluri", "long", "semi-peren", "semiperen"]
    ):
        return "perene"
    if any(p in txt for p in ["anual", "bienal", "curt", "temp", "estacion"]):
        return "anual_bienal"
    return "desconhecido"


def e_especie_perene_pastejavel(row):
    ciclo_cat = categorizar_ciclo(row.get("ciclo", ""))
    if ciclo_cat != "perene":
        return False

    return (
        row.get("pastejo", 0) == 1
        or row.get("pastagem", 0) == 1
        or row.get("consorcio", 0) == 1
        or row.get("forragem", 0) == 1
        or row.get("forragem_ruminantes", 0) == 1
        or row.get("pastagem_consorciada", 0) == 1
        or row.get("pastejo_direto", 0) == 1
    )


def normalize_min_max(val, val_min, val_max):
    """Normalize a raw score to [0, 1]."""
    if val_max == val_min:
        return 0.0
    norm = (val - val_min) / (val_max - val_min)
    return max(0.0, min(1.0, norm))


def calcular_score_normalizado(
    row,
    ph,
    clima,
    text,
    fert,
    w_eco=0.60,
    w_prod=0.40,
):
    """Compute individual suitability and realized criterion contributions."""
    if clima == "cerrado":
        aptidao_clima = (
            row.get("cerrado", 0) == 1
            or row.get("tropical", 0) == 1
            or row.get("quente", 0) == 1
            or row.get("tropical_umido", 0) == 1
        )
    elif clima in ["tropical", "quente"]:
        aptidao_clima = (
            row.get("tropical", 0) == 1
            or row.get("cerrado", 0) == 1
            or row.get("quente", 0) == 1
        )
    else:
        aptidao_clima = row.get(clima, 0) == 1

    if not aptidao_clima:
        return 0.0, {}

    ph_min = row["ph_min"]
    ph_max = row["ph_max"]
    delta_ph = 0.2

    if ph < (ph_min - delta_ph) or ph > (ph_max + delta_ph):
        return 0.0, {}

    if ph_min <= ph <= ph_max:
        ph_norm = 1.0
    elif (ph_min - delta_ph) <= ph < ph_min:
        ph_norm = (ph - (ph_min - delta_ph)) / delta_ph
    elif ph_max < ph <= (ph_max + delta_ph):
        ph_norm = ((ph_max + delta_ph) - ph) / delta_ph
    else:
        ph_norm = 0.0

    ph_norm = float(np.clip(ph_norm, 0.0, 1.0))

    textura_alvo = mapear_textura(text)
    raw_text = (
        15.0
        if (
            np.isfinite(textura_alvo)
            and abs(row["text_num"] - textura_alvo) <= 1
        )
        else -10.0
    )

    fertilidade_alvo = mapear_fertilidade(fert)
    raw_fert = (
        15.0
        if (
            np.isfinite(fertilidade_alvo)
            and abs(row["fert_num"] - fertilidade_alvo) <= 1
        )
        else -10.0
    )

    raw_uso = 20.0 if "pastagem" in limpar_texto(row["uso_str"]) else 5.0
    raw_raiz = min(row["profundidade_raiz_max"] / 20.0, 10.0)

    norm_scores = {
        "ph": ph_norm,
        "textura": normalize_min_max(raw_text, -10, 15),
        "fertilidade": normalize_min_max(raw_fert, -10, 15),
        "uso": normalize_min_max(raw_uso, 5, 20),
        "raiz": normalize_min_max(raw_raiz, 0, 10),
    }

    eco_weights = {
        "ph": 1.1,
        "textura": 1.0,
        "fertilidade": 1.0,
        "uso": 0.8,
        "raiz": 1.0,
    }
    sum_eco_weights = sum(eco_weights.values())

    s_eco_norm = (
        sum(eco_weights[k] * norm_scores[k] for k in eco_weights)
        / sum_eco_weights
    )

    s_prod_norm = float(np.clip(row["prod_norm"], 0.0, 1.0))

    score_norm = (w_eco * s_eco_norm) + (w_prod * s_prod_norm)
    score_final = score_norm * 100.0

    if score_norm > 0:
        contrib = {}
        for k in eco_weights:
            part = w_eco * (
                eco_weights[k] * norm_scores[k] / sum_eco_weights
            )
            contrib[k] = round((part / score_norm) * 100.0, 1)

        contrib["produtividade"] = round(
            ((w_prod * s_prod_norm) / score_norm) * 100.0,
            1,
        )
    else:
        contrib = {
            k: 0.0
            for k in list(eco_weights.keys()) + ["produtividade"]
        }

    return round(score_final, 2), contrib
