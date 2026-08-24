"""Dataset loading, cleaning, encoding, and preprocessing utilities."""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


CLIMATE_COLUMNS = [
    "cerrado",
    "tropical",
    "subtropical",
    "temperado",
    "subtropical_secundario",
    "subtropical_quente",
    "subtropical_de_altitude",
    "subtropical_umido",
    "subtropical_frio",
    "quente",
    "chuvoso",
    "semiarido",
    "subtropical_semiarido",
    "tropical_altitude",
    "estacao_fria",
    "quente_umido",
    "tropical_umido",
    "temperado_frio",
    "temperado_medio",
    "regioes_frias",
]

USE_COLUMNS = [
    "pastagem",
    "pastejo",
    "feno",
    "silagem",
    "consorcio",
    "forragem",
    "forragem_ruminantes",
    "pastagem_consorciada",
    "pastejo_direto",
    "adubacao_verde",
]

SOIL_FEATURES = [
    "profundidade_raiz_max",
    "altura_media",
    "ph_min",
    "ph_max",
    "text_num",
    "fert_num",
]

NUMERIC_COLUMNS = [
    "altura_min",
    "altura_max",
    "profundidade_raiz_max",
    "ph_min",
    "ph_max",
    "produtividade_minima_ms",
    "produtividade_maxima_ms",
]


def normalizar_coluna(col):
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(c for c in col if not unicodedata.combining(c))
    return col.replace(" ", "_")


def limpar_num(val):
    if pd.isna(val):
        return np.nan
    permitido = "".join(c for c in str(val) if c.isdigit() or c in ",.-")
    try:
        return float(permitido.replace(",", "."))
    except (TypeError, ValueError):
        return np.nan


def limpar_texto(val):
    return str(val).strip().lower() if pd.notna(val) else ""


def valor_binario(val):
    return (
        1
        if str(val).strip().lower() in ["1", "1.0", "sim", "yes", "true", "x"]
        else 0
    )


def mapear_fertilidade(txt):
    txt = limpar_texto(txt)
    if txt.startswith("baix"):
        return 1
    if txt.startswith("med") or txt.startswith("méd"):
        return 2
    if txt.startswith("alt"):
        return 3
    return np.nan


def mapear_textura(txt):
    txt = limpar_texto(txt)
    if txt in ["", "na", "nan"]:
        return np.nan

    areia = ["aren", "leve"]
    media = ["media", "média", "franco", "medio"]
    argila = ["argil", "pesad"]

    valores = []
    if any(p in txt for p in areia):
        valores.append(1)
    if any(p in txt for p in media):
        valores.append(2)
    if any(p in txt for p in argila):
        valores.append(3)

    return np.mean(valores) if valores else np.nan


def classificar_estrato(altura):
    if pd.isna(altura):
        return np.nan
    if altura < 0.4:
        return 1
    if altura < 1.2:
        return 2
    return 3


def load_and_preprocess(csv_path):
    """Load the forage database and reproduce the preprocessing used by the model."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df.columns = [normalizar_coluna(col) for col in df.columns]
    df = df[df["especie"].notna()].copy()
    df = df.reset_index(drop=True)
    df["especie"] = df["especie"].astype(str).str.strip()

    df["grupo"] = df["grupo"].apply(normalizar_coluna)

    for col in CLIMATE_COLUMNS + USE_COLUMNS:
        df[col] = df[col].apply(valor_binario) if col in df.columns else 0

    df["uso_str"] = df.apply(
        lambda r: ", ".join([u for u in USE_COLUMNS if r.get(u, 0) == 1]),
        axis=1,
    )

    if "exigencia_fertilidade" in df.columns:
        df["fert_num"] = df["exigencia_fertilidade"].apply(mapear_fertilidade)
    else:
        df["fert_num"] = np.nan

    if "textura_solo" in df.columns:
        df["text_num"] = df["textura_solo"].apply(mapear_textura)
    else:
        df["text_num"] = np.nan

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(limpar_num)

    df["altura_media"] = (
        df.get("altura_min", 0) + df.get("altura_max", 0)
    ) / 2

    df["produtividade_media"] = (
        df.get("produtividade_minima_ms", 0)
        + df.get("produtividade_maxima_ms", 0)
    ) / 2

    df["produtividade_media"] = df["produtividade_media"].fillna(
        df["produtividade_media"].median()
    )

    max_prod = df["produtividade_media"].max()
    df["prod_norm"] = (
        df["produtividade_media"] / max_prod if max_prod > 0 else 0
    )

    # Median imputation of numerical features required by the model.
    for col in SOIL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["estrato_num"] = df["altura_media"].apply(classificar_estrato)
    df["pos_matriz"] = df.index

    return df
